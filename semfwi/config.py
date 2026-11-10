"""Parameters of an inversion case."""
import dataclasses
import os
from dataclasses import dataclass, field
import numpy as np

# Root of the examples' run directories (SEMFWI_RUNS)
RUNS_DIR = os.path.expanduser(os.environ.get("SEMFWI_RUNS", "~/SEM/exp/runs"))


@dataclass
class Grid:
    """Regular parameter grid; xmin and xmax are the positions of the first and last samples. h: one
    spacing for the three axes, or one per axis (hx, hy, hz)."""
    xmin: tuple = (-50., -50., -450.)
    h: float = 25.
    n: tuple = (21, 21, 19)

    @property
    def hs(self):
        """Spacing per axis."""
        return tuple(float(x) for x in self.h) if np.ndim(self.h) else (float(self.h),) * 3

    @property
    def isotropic(self):
        return not np.ndim(self.h)

    @property
    def shape(self):
        return tuple(self.n)

    @property
    def xmax(self):
        return tuple(self.xmin[c] + self.hs[c] * (self.n[c] - 1) for c in range(3))

    def axis(self, c):
        return self.xmin[c] + self.hs[c] * np.arange(self.n[c])

    def mesh(self):
        return np.meshgrid(*[self.axis(c) for c in range(3)], indexing="ij")


@dataclass
class Config:
    workdir: str
    # Mesh (on-the-fly mesher): cube [x0, x0 + L]^2 x [-depth, 0], npml layers of PML elements
    # all around except at the surface. layers: list of (thickness, number of elements) from top to
    # bottom; a single layer by default. With several layers, the lateral PMLs take the properties
    # of each layer (the profile is extended into the PMLs).
    L: float = 400.
    depth: float = 400.
    h_elem: float = 50.
    ngll: int = 5
    x0: float = 0.
    layers: list = None
    # Properties of each layer of layers: None, read from the model files on the parameter grid (the
    # default for all), or (lambda, mu, rho), constant and not inverted (a bedrock below the grid)
    layer_props: list = None
    npml: int = 1
    # Same grid written by us (hexmesh, HDF5 Hex8 input of the mesher) instead of the on-the-fly
    # mesher; forced by a topography: npz file (x, y axes, z elevations of shape (len(y), len(x)))
    # that the free surface follows, the mesh becoming flat at z = -topo_depth
    mesh_hex8: bool = False
    topography: str = None
    topo_depth: float = 0.
    # Strain output: one step every strain_period (the number of steps must be a multiple of it)
    strain_period: int = 1
    # Kernels computed by SEM3D during the adjoint simulation (strain_mode 1 and 2): the forward
    # simulation writes displacements, 6 times smaller than strains, and no adjoint strain is
    # written. SEMFWI_KERNELS_IN_SEM=0 falls back to the strain files and the kernels in Python.
    kernels_in_sem: bool = os.environ.get("SEMFWI_KERNELS_IN_SEM", "1") != "0"
    # P velocity of the damping profile of the PMLs (pml_damping_vp of SEM3D); 0: the local
    # velocity, so that the damping changes with the model when the PMLs extend it
    pml_damping_vp: float = 0.
    # Adjoint sources: "points" (one func = file source per station and component) or
    # "extended" (3 extended sources on the station grid receiver_grid = (xs, ys, z))
    adjoint: str = "points"
    receiver_grid: tuple = None
    # Misfit weights, one per station and component (nsta x 3), or one set per source (nsrc x nsta x 3);
    # None: all 1 (misfit.cost)
    weights: list = None
    # Radius (m) of the taper of the gradient around the point sources (model.source_taper): 0 at the
    # source, back to 1 at this distance; 0: none
    source_taper: float = 0.
    # Time: dt is imposed (M1), courant only serves as a stability bound
    sim_time: float = 0.7
    dt: float = 8e-4
    courant: float = 0.3
    grid: Grid = field(default_factory=Grid)
    lam0: float = 4e9
    mu0: float = 2e9
    rho0: float = 2000.
    stations: list = field(default_factory=list)
    # Point sources of the forward problem, as dictionaries of the input.spec source section
    sources: list = field(default_factory=list)
    nprocs: int = 1        # MPI ranks per simulation
    workers: int = 4       # concurrent simulations
    # Directory of sem3d.exe and mesher (SEM3D_BIN): no default, so as not to pick up a SEM3D
    # from the PATH that would lack the strain output
    sem3d_bin: str = os.path.expanduser(os.environ.get("SEM3D_BIN", ""))

    @property
    def nt(self):
        return int(round(self.sim_time / self.dt))

    @property
    def layer_list(self):
        return self.layers or [(self.depth, int(round(self.depth / self.h_elem)))]

    @property
    def file_layers(self):
        """Indices of the layers whose properties come from the model files."""
        props = self.layer_props or [None] * len(self.layer_list)
        assert len(props) == len(self.layer_list), (len(props), len(self.layer_list))
        return [i for i, p in enumerate(props) if p is None]

    def layer_index(self, z):
        """layer_of_depth for an array of depths (z <= 0; deeper points: the last layer)."""
        bottoms = -np.cumsum([th for th, _ in self.layer_list])
        return np.minimum(np.sum(np.asarray(z)[..., None] < bottoms[None, :] - 1e-9, axis=-1), len(bottoms) - 1)

    def layer_of_depth(self, z):
        """Index of the layer containing depth z (z <= 0)."""
        top = 0.
        for i, (th, _) in enumerate(self.layer_list):
            if z >= top - th - 1e-9:
                return i
            top -= th
        return len(self.layer_list) - 1


# Fields specific to the client machine, left out of the serializable form
LOCAL_FIELDS = ("workdir", "sources", "sem3d_bin")


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def to_dict(cfg):
    """Config without the local fields (workdir, sources, sem3d_bin), as JSON types."""
    out = {f.name: _plain(getattr(cfg, f.name)) for f in dataclasses.fields(cfg) if f.name not in LOCAL_FIELDS}
    out["grid"] = {f.name: _plain(getattr(cfg.grid, f.name)) for f in dataclasses.fields(cfg.grid)}
    return out


def from_dict(d, workdir, sem3d_bin=None):
    """Inverse of to_dict; sources stay empty (they are already written in the inputs)."""
    d = dict(d)
    g = d.pop("grid")
    grid = Grid(xmin=tuple(g["xmin"]), h=tuple(g["h"]) if np.ndim(g["h"]) else g["h"], n=tuple(g["n"]))
    kw = dict(sem3d_bin=sem3d_bin) if sem3d_bin is not None else {}
    return Config(workdir=workdir, grid=grid, **d, **kw)

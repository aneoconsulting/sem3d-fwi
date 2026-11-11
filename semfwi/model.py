"""Model on the parameter grid, SEM3D material.spec files and trilinear interpolation."""
import os

import h5py
import numpy as np

from . import hexmesh

MATERIAL_SPEC = """material {num} {{
domain = solid;
deftype = Lambda_Mu;
spacedef = file;
filename0 = "Lambda.h5";
filename1 = "Mu.h5";
filename2 = "Rho.h5";
}};
"""


# PML material: SEM3D copies the definition of material 0 (the model files) and samples them on the
# inner edge of the PML, which extends the model into the PML point by point
PML_COPY_SPEC = """material {num} {{
copy = 0;
}};
"""


class UnphysicalModel(ValueError):
    pass


def check_bounds(lam, mu, rho):
    if np.any(mu <= 0) or np.any(lam + 2 * mu / 3 <= 0) or np.any(rho <= 0):
        raise UnphysicalModel("non-physical model: mu > 0, lambda + 2 mu / 3 > 0 and rho > 0 required")


def write_model(d, grid, lam, mu, rho, nmat=1, file_layers=None):
    """lam, mu, rho: arrays (nx, ny, nz), written as (nz, ny, nx) for SEM3D's Fortran order, on
    grid.file_grid (the grid itself when regular).
    The nmat solid materials (one mesher layer each), or those of file_layers, read the same files;
    the others keep the constant properties of material.input."""
    check_bounds(lam, mu, rho)
    fg, k = grid.file_grid()
    for name, arr in (("Lambda", lam), ("Mu", mu), ("Rho", rho)):
        assert arr.shape == grid.shape, (name, arr.shape, grid.shape)
        with h5py.File(f"{d}/{name}.h5", "w") as f:
            g = f.create_group(name)
            g["samples"] = np.ascontiguousarray(np.transpose(arr[:, :, k], (2, 1, 0)), dtype="f8")
            g.attrs["xMinGlob"] = np.array(fg.xmin, dtype="f8")
            g.attrs["xMaxGlob"] = np.array(fg.xmax, dtype="f8")
    with open(f"{d}/material.spec", "w") as f:
        f.write("".join(MATERIAL_SPEC.format(num=k) for k in (range(nmat) if file_layers is None else file_layers)))


def write_pml_copy(d):
    """Appends to material.spec, written by write_model, a copy of material 0 for each PML material
    of material.input: the PMLs then extend the current model, as in Fathi et al. Without
    material.input (no mesh in d, as in the unit tests), there is nothing to extend."""
    if not os.path.exists(f"{d}/material.input"):
        return
    lines = open(f"{d}/material.input").read().splitlines()
    nmat = int(lines[0])
    pml = [i - 1 for i in range(1, nmat + 1) if lines[i].startswith("P")]
    with open(f"{d}/material.spec", "a") as f:
        f.write("".join(PML_COPY_SPEC.format(num=k) for k in pml))


def node_layers(cfg):
    """Mesh layer (cfg.layer_list) of each node of the parameter grid, from its depth in the flat
    mesh (hexmesh.flat_z); -1 above the free surface."""
    X, Y, Z = cfg.grid.mesh()
    zf = hexmesh.flat_z(cfg, X, Y, Z)
    return np.where(zf > 1e-6, -1, cfg.layer_index(np.minimum(zf, 0.)))


def write_pml_materials(d, cfg, lam, mu, rho):
    """Extends the model into the PMLs: each PML material gets the mean properties
    (over the grid nodes of the layer, below the free surface) of the solid layer it depends on,
    or its constant properties (cfg.layer_props).
    Rewrites the P lines of material.input (the mesh only stores material indices)."""
    lines = open(f"{d}/material.input").read().splitlines()
    nmat = int(lines[0])
    ipml = lines.index("# PML properties")
    parents = [int(l.split()[-1]) for l in lines[ipml + 2:] if l.strip()]
    layer_of_node = node_layers(cfg)
    nl = len(cfg.layer_list)
    const = cfg.layer_props or [None] * nl
    props = []
    for l in range(nl):
        sel = layer_of_node == l
        lm, mm, rm = const[l] or (a[sel].mean() for a in (lam, mu, rho))
        props.append((np.sqrt((lm + 2 * mm) / rm), np.sqrt(mm / rm), rm))
    k = 0
    for i in range(1, nmat + 1):
        if lines[i].startswith("P"):
            vp, vs, r = props[parents[k]]
            rest = lines[i].split()[4:]
            lines[i] = "P %f %f %f " % (vp, vs, r) + " ".join(rest)
            k += 1
    assert k == len(parents), (k, len(parents))
    with open(f"{d}/material.input", "w") as f:
        f.write("\n".join(lines) + "\n")


def pml_layer_gradient(cfg, g_layer):
    """Gradient on the grid of the PML properties written by write_pml_materials: those of layer l
    are the means over its nodes, so dJ/dlayer l (g_layer[l]) is shared equally among them."""
    layer_of_node = node_layers(cfg)
    count = np.bincount(layer_of_node[layer_of_node >= 0], minlength=len(g_layer))
    per_layer = np.where(count > 0, g_layer / np.maximum(count, 1), 0.)
    return np.where(layer_of_node >= 0, per_layer[np.maximum(layer_of_node, 0)], 0.)


def source_position(src):
    """(x, y, z) of a point source: a FileSource or an input.spec source section; None otherwise."""
    if hasattr(src, "coords"):
        return tuple(float(c) for c in src.coords)
    if isinstance(src, dict) and "coords" in src:
        return tuple(float(c) for c in src["coords"].split())
    return None


def source_taper(grid, positions, radius):
    """Factor on the grid that removes the gradient near the sources, where the kernels are singular and
    the inversion would explain the data by changing the medium at the source: sin^2(pi d / (2 radius)) at
    a distance d < radius of the nearest source, 1 beyond."""
    X, Y, Z = grid.mesh()
    w = np.ones(grid.shape)
    for x, y, z in positions:
        d = np.sqrt((X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2)
        w *= np.where(d < radius, np.sin(0.5 * np.pi * np.minimum(d / radius, 1.)) ** 2, 1.)
    return w


def gll_axis(edges, ngll):
    """Ascending coordinates of the GLL points of the elements between consecutive edges (ascending),
    each shared edge once."""
    from numpy.polynomial import legendre
    n = ngll - 1
    xi = np.concatenate([[-1.], np.sort(legendre.Legendre.basis(n).deriv().roots()), [1.]])
    edges = np.asarray(edges, dtype=float)
    pts = [a + 0.5 * (b - a) * (xi + 1) for a, b in zip(edges[:-1], edges[1:])]
    return np.concatenate([p[:-1] for p in pts] + [edges[-1:]])


def read_model(d):
    out = []
    for name in ("Lambda", "Mu", "Rho"):
        with h5py.File(f"{d}/{name}.h5", "r") as f:
            out.append(np.transpose(f[name]["samples"][...], (2, 1, 0)))
    return out


def trilinear(grid, coords):
    """Grid node indices (npts, 8) and weights (npts, 8) of the trilinear interpolation."""
    n = np.array(grid.n)
    coords = np.asarray(coords)
    t = (coords - np.array(grid.xmin)) / np.array(grid.hs)
    i0 = np.clip(np.floor(t).astype(int), 0, n - 2)
    f = np.clip(t - i0, 0, 1)
    if grid.z is not None:
        z = np.array(grid.z)
        i0[:, 2] = np.clip(np.searchsorted(z, coords[:, 2], side="right") - 1, 0, n[2] - 2)
        f[:, 2] = np.clip((coords[:, 2] - z[i0[:, 2]]) / (z[i0[:, 2] + 1] - z[i0[:, 2]]), 0, 1)
    idx, w = [], []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                idx.append(np.ravel_multi_index((i0[:, 0] + a, i0[:, 1] + b, i0[:, 2] + c), grid.shape))
                w.append((f[:, 0] if a else 1 - f[:, 0]) * (f[:, 1] if b else 1 - f[:, 1])
                         * (f[:, 2] if c else 1 - f[:, 2]))
    return np.stack(idx, 1), np.stack(w, 1)


def interpolate(grid, field, idx, w):
    """Values of field (nx, ny, nz) at the points."""
    return np.sum(field.ravel()[idx] * w, axis=1)


def project(grid, values, idx, w):
    """Transpose of interpolate: accumulates values (npts) onto the grid."""
    out = np.zeros(np.prod(grid.shape))
    np.add.at(out, idx.ravel(), (w * np.asarray(values)[:, None]).ravel())
    return out.reshape(grid.shape)


def interior_mask(grid, L, depth, x0=0., surface=None):
    """Nodes whose trilinear support (clipped to the grid) lies entirely in the physical domain
    [x0, x0 + L]^2 x [-depth, 0], outside the PMLs: only these are inverted, since the gradient
    is not computed in the PMLs. surface(x, y): elevation of a free surface that replaces z = 0;
    the nodes above it whose support reaches below it, at its centre or one of its 8 horizontal
    neighbours, are kept (the medium near the surface depends on them), the others are in the air."""
    X, Y, Z = grid.mesh()
    eps = 1e-6
    lo = [max(grid.xmin[c], -1e30) for c in range(3)]
    hi = list(grid.xmax)
    ok = np.ones(grid.shape, bool)
    for A, c, a, b in ((X, 0, x0, x0 + L), (Y, 1, x0, x0 + L), (Z, 2, -depth, 0.)):
        if c == 2 and grid.z is not None:  # previous and next nodes of an explicit z axis
            z = np.array(grid.z)
            prev = np.broadcast_to(np.concatenate([z[:1], z[:-1]]), A.shape)
            nxt = np.broadcast_to(np.concatenate([z[1:], z[-1:]]), A.shape)
        else:
            prev, nxt = A - grid.hs[c], A + grid.hs[c]
        below = (np.maximum(prev, lo[c]) >= a - eps)
        if c == 2 and surface is not None:
            top = np.max([surface(X + i * grid.hs[0], Y + j * grid.hs[1]) for i in (-1, 0, 1) for j in (-1, 0, 1)], axis=0)
            ok &= below & (np.maximum(prev, lo[c]) < top - eps)
        else:
            ok &= below & (np.minimum(nxt, hi[c]) <= b + eps)
    return ok


def node_volume(grid, h_elem, lo):
    """Volume carried by each node of the parameter grid when it coincides with the GLL nodes of
    quadratic elements (ngll = 3, grid.h = h_elem / 2, element edges at lo + k h_elem on each axis):
    the assembled GLL weights, h_elem/6 at the element ends and 4 h_elem/6 in the middle, per axis.
    The gradient of a node is its kernel times this volume, so that the nodes in the middle of the
    elements get up to 8 times the gradient of their neighbours: dividing by it gives the gradient
    of the continuous field, smooth where the kernel is (None if the grid does not match)."""
    if not grid.isotropic or abs(2 * grid.h - h_elem) > 1e-9 * h_elem:
        return None
    w = []
    for c in range(3):
        k = np.rint((grid.axis(c) - lo[c]) / grid.h)
        if np.max(np.abs(grid.axis(c) - lo[c] - k * grid.h)) > 1e-6 * grid.h:
            return None
        wc = np.where(k % 2 == 1, 4., 1.) * h_elem / 6
        wc[(k % 2 == 0) & (k > k.min()) & (k < k.max())] *= 2  # shared by two elements
        w.append(wc)
    return w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]


def node_scaling(grid, h_elem, lo, mask):
    """Mean node volume / node volume on the nodes of mask (node_volume): multiplying the gradient by
    it, the descent directions follow the gradient of the continuous field, instead of moving the
    nodes in the middle of the elements up to 8 times more (checkerboard). The nodes on the faces of
    the grid (free surface, sides and bottom against the PMLs) are scaled as interior nodes, as if
    their elements continued beyond the grid: with their own, smaller volume they moved up to 8 times
    more again (25.8 instead of 3.2 at a corner), and the model collapsed there. None if the grid
    does not match the GLL nodes."""
    if node_volume(grid, h_elem, lo) is None:
        return None
    # volume of an interior node, per axis: 4 h_elem / 6 in the middle of an element, 2 h_elem / 6 on an edge
    w = [np.where(np.rint((grid.axis(c) - lo[c]) / grid.h) % 2 == 1, 2., 1.) for c in range(3)]
    w = (w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :])[mask]
    return w.mean() / w

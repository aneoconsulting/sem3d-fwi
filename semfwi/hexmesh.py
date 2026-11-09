"""Mesh written by us for the HDF5 Hex8 input of the SEM3D mesher (choice 4): the grid of the on-the-fly
mesher (same nodes, elements, materials and PML numbering), whose nodes can follow a topography.

The mesher reads /Nodes (npoints x 3), /Sem3D/Hexa8 (nelem x 8) and /Sem3D/Mat (nelem) from mesh.h5,
and the materials, PMLs included, from material.input, which we write as the on-the-fly mesher does."""
import functools

import h5py
import numpy as np

# PML damping profile written by the on-the-fly mesher (npow, Apow)
PML_PROFILE = "2 10."


def grid_mesh(cfg):
    """Nodes (npoints x 3), elements (nelem x 8), material per element and material table of the
    on-the-fly mesher for cfg (mat.dat of sem_io.write_mesher_inputs: PMLs on the sides and at the
    bottom, none on top).

    The material table is a list of (type, layer, pml): type "S" or "P", layer the solid material
    it derives from, pml (xpos, xwidth, ypos, ywidth, zpos, zwidth) for "P". The PML materials are
    numbered in the order the elements meet them (RectMesh::get_mat), layer by layer, then z, y, x."""
    npml, h = cfg.npml, cfg.h_elem
    layers = [list(l) for l in cfg.layer_list]
    xmin0, xmax0 = cfg.x0, cfg.x0 + cfg.L
    zmin0 = -sum(th for th, _ in layers)
    # PMLs outside [x0, x0 + L]^2 x [zmin0, 0] (RectMesh::apply_pml_borders)
    xmin = xmin0 - npml * h
    nxy = int((xmax0 - xmin0 + 2 * npml * h) / h)
    zstep_bottom = layers[-1][0] / layers[-1][1]
    layers[-1] = [layers[-1][0] + npml * zstep_bottom, layers[-1][1] + npml]
    # Nodes: layer by layer from the top, x fastest (RectMesh::create_linear_grid_nodes)
    zs, top = [0.], 0.
    for th, n in layers:
        zs += list(top - (th / n) * np.arange(1, n + 1))
        top -= th
    zs = np.array(zs)
    xs = xmin + h * np.arange(nxy + 1)
    Z, Y, X = np.meshgrid(zs, xs, xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def idx(i, j, k):
        return (k * (nxy + 1) + j) * (nxy + 1) + i

    nz = len(zs) - 1
    k, j, i = (a.ravel() for a in np.meshgrid(np.arange(nz), np.arange(nxy), np.arange(nxy), indexing="ij"))
    # Lower face then upper face, counterclockwise seen from above (RectMesh::create_linear_element)
    elems = np.stack([idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1),
                      idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k)], axis=1)
    layer = np.repeat(np.arange(len(layers)), [n for _, n in layers])[k]
    W, E = i < npml, i > nxy - npml - 1
    S, N = j < npml, j > nxy - npml - 1
    D = k > nz - npml - 1
    table = [("S", l, None) for l in range(len(layers))]
    known = {}
    mats = layer.copy()
    for e in np.nonzero(W | E | S | N | D)[0]:
        key = (int(layer[e]), bool(W[e]), bool(E[e]), bool(S[e]), bool(N[e]), bool(D[e]))
        if key not in known:
            l, w, ea, s, n, d = key
            xp, xw = (xmin0, -npml * h) if w else (xmax0, npml * h) if ea else (0., 0.)
            yp, yw = (xmin0, -npml * h) if s else (xmax0, npml * h) if n else (0., 0.)
            zp, zw = (zmin0, -npml * zstep_bottom) if d else (0., 0.)
            known[key] = len(table)
            table.append(("P", l, (xp, xw, yp, yw, zp, zw)))
        mats[e] = known[key]
    return nodes, elems, mats, table


@functools.lru_cache(maxsize=4)
def _load(path):
    t = np.load(path)
    return t["x"], t["y"], t["z"]


def surface_offset(path, x, y):
    """Elevation of the free surface at (x, y), bilinear in the npz file path (x, y: axes, z: values
    with shape (len(y), len(x)), in m above z = 0)."""
    tx, ty, tz = _load(path)
    fx = np.clip((x - tx[0]) / (tx[1] - tx[0]), 0, len(tx) - 1 - 1e-9)
    fy = np.clip((y - ty[0]) / (ty[1] - ty[0]), 0, len(ty) - 1 - 1e-9)
    i, j = fx.astype(int), fy.astype(int)
    u, v = fx - i, fy - j
    return ((1 - u) * (1 - v) * tz[j, i] + u * (1 - v) * tz[j, i + 1]
            + (1 - u) * v * tz[j + 1, i] + u * v * tz[j + 1, i + 1])


def follow_surface(nodes, dz, depth):
    """Moves each node vertically by dz times a weight going linearly from 1 at z = 0 to 0 at
    z = -depth: the free surface follows dz, the mesh is flat below depth. The elements keep a
    positive height if dz > -depth everywhere."""
    assert np.all(dz > -depth), "topography deeper than the depth where the mesh becomes flat"
    w = np.clip(1 + nodes[:, 2] / depth, 0., 1.)
    out = nodes.copy()
    out[:, 2] += dz * w
    return out


def surface(cfg, x, y):
    """Elevation of the free surface of cfg at (x, y): 0 without topography."""
    if not cfg.topography:
        return np.zeros(np.broadcast(x, y).shape)
    return surface_offset(cfg.topography, x, y)


def flat_z(cfg, x, y, z):
    """Inverse of follow_surface: z of the point of the flat mesh that the topography moved to
    (x, y, z), so that the layers of cfg.layer_list (thicknesses from z = 0) apply below a topography.
    Positive above the free surface."""
    if not cfg.topography:
        return np.asarray(z, float)
    dz = surface(cfg, x, y)
    d = cfg.topo_depth
    return np.where(z > -d, (z - dz) / (1 + dz / d), z)


def on_surface(cfg, points):
    """Points given by (x, y, depth below the free surface, <= 0) as (x, y, z) of cfg."""
    p = np.array(points, float)
    p[:, 2] += surface(cfg, p[:, 0], p[:, 1])
    return [tuple(q) for q in p]


def build(cfg):
    """grid_mesh of cfg, its free surface moved to cfg.topography (cfg.topo_depth) if set. The PMLs
    must stay flat boxes (their damping depends on the distance to their planes): the topography
    must be zero over the PML elements."""
    nodes, elems, mats, table = grid_mesh(cfg)
    if cfg.topography:
        dz = surface_offset(cfg.topography, nodes[:, 0], nodes[:, 1])
        pml_nodes = np.unique(elems[np.array([table[m][0] == "P" for m in mats])])
        assert np.allclose(dz[pml_nodes][nodes[pml_nodes, 2] > -cfg.topo_depth], 0.), \
            "the topography is not zero over the PMLs"
        nodes = follow_surface(nodes, dz, cfg.topo_depth)
    return nodes, elems, mats, table


def write(d, cfg, nodes, elems, mats, table):
    """mesh.h5 and material.input in d (reference properties of sem_io.write_mesher_inputs: those
    of cfg.layer_props, or lam0, mu0, rho0 for the layers read from the model files)."""
    with h5py.File(f"{d}/mesh.h5", "w") as f:
        f["Nodes"] = nodes
        f["Sem3D/Hexa8"] = elems.astype(np.int32)
        f["Sem3D/Mat"] = mats.astype(np.int32)
    props = cfg.layer_props or [None] * len(cfg.layer_list)
    lines = [f"{len(table)}"]
    pml = []
    for typ, l, p in table:
        lam, mu, rho = props[l] or (cfg.lam0, cfg.mu0, cfg.rho0)
        lines.append("%c %f %f %f %f %f" % (typ, np.sqrt((lam + 2 * mu) / rho), np.sqrt(mu / rho), rho, 0., 0.))
        if typ == "P":
            pml.append(f"{PML_PROFILE} " + " ".join("%8.1f" % v for v in p) + " %2d" % l)
    lines += ["# PML properties", "# npow,Apow,posX,widthX,posY,widthY,posZ,widthZ,mat"] + pml
    with open(f"{d}/material.input", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(f"{d}/mesh.input", "w") as f:
        f.write(f"{cfg.nprocs}\n4\n1\nmesh.h5\n")

"""Model on the parameter grid, SEM3D material.spec files and trilinear interpolation."""
import h5py
import numpy as np

MATERIAL_SPEC = """material {num} {{
domain = solid;
deftype = Lambda_Mu;
spacedef = file;
filename0 = "Lambda.h5";
filename1 = "Mu.h5";
filename2 = "Rho.h5";
}};
"""


def check_bounds(lam, mu, rho):
    if np.any(mu <= 0) or np.any(lam + 2 * mu / 3 <= 0) or np.any(rho <= 0):
        raise ValueError("non-physical model: mu > 0, lambda + 2 mu / 3 > 0 and rho > 0 required")


def write_model(d, grid, lam, mu, rho, nmat=1):
    """lam, mu, rho: arrays (nx, ny, nz), written as (nz, ny, nx) for SEM3D's Fortran order.
    The nmat solid materials (one mesher layer each) read the same files."""
    check_bounds(lam, mu, rho)
    for name, arr in (("Lambda", lam), ("Mu", mu), ("Rho", rho)):
        assert arr.shape == grid.shape, (name, arr.shape, grid.shape)
        with h5py.File(f"{d}/{name}.h5", "w") as f:
            g = f.create_group(name)
            g["samples"] = np.ascontiguousarray(np.transpose(arr, (2, 1, 0)), dtype="f8")
            g.attrs["xMinGlob"] = np.array(grid.xmin, dtype="f8")
            g.attrs["xMaxGlob"] = np.array(grid.xmax, dtype="f8")
    with open(f"{d}/material.spec", "w") as f:
        f.write("".join(MATERIAL_SPEC.format(num=k) for k in range(nmat)))


def write_pml_materials(d, cfg, lam, mu, rho):
    """Extends the model into the PMLs: each PML material gets the mean properties
    (over x, y) of the solid layer it depends on, read on the parameter grid.
    Rewrites the P lines of material.input (the mesh only stores material indices)."""
    lines = open(f"{d}/material.input").read().splitlines()
    nmat = int(lines[0])
    ipml = lines.index("# PML properties")
    parents = [int(l.split()[-1]) for l in lines[ipml + 2:] if l.strip()]
    z = cfg.grid.axis(2)
    layer_of_node = np.array([cfg.layer_of_depth(zz) for zz in z])
    nl = len(cfg.layer_list)
    props = []
    for l in range(nl):
        sel = layer_of_node == l
        lm, mm, rm = (a[:, :, sel].mean() for a in (lam, mu, rho))
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


def read_model(d):
    out = []
    for name in ("Lambda", "Mu", "Rho"):
        with h5py.File(f"{d}/{name}.h5", "r") as f:
            out.append(np.transpose(f[name]["samples"][...], (2, 1, 0)))
    return out


def trilinear(grid, coords):
    """Grid node indices (npts, 8) and weights (npts, 8) of the trilinear interpolation."""
    n = np.array(grid.n)
    t = (np.asarray(coords) - np.array(grid.xmin)) / grid.h
    i0 = np.clip(np.floor(t).astype(int), 0, n - 2)
    f = np.clip(t - i0, 0, 1)
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


def interior_mask(grid, L, depth, x0=0.):
    """Nodes whose trilinear support (clipped to the grid) lies entirely in the physical domain
    [x0, x0 + L]^2 x [-depth, 0], outside the PMLs: only these are inverted, since the gradient
    is not computed in the PMLs."""
    X, Y, Z = grid.mesh()
    h, eps = grid.h, 1e-6
    lo = [max(grid.xmin[c], -1e30) for c in range(3)]
    hi = list(grid.xmax)
    ok = np.ones(grid.shape, bool)
    for A, c, a, b in ((X, 0, x0, x0 + L), (Y, 1, x0, x0 + L), (Z, 2, -depth, 0.)):
        ok &= (np.maximum(A - h, lo[c]) >= a - eps) & (np.minimum(A + h, hi[c]) <= b + eps)
    return ok

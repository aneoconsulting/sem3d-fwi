"""Tests of the E2VP case (examples/euroseistest/e2vp.py) that need no simulation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "euroseistest"))
import e2vp  # noqa: E402
import e2vp2  # noqa: E402


def test_moment_tensor():
    """Vertical strike-slip: Mxx = -Myy = M0 sin 2 phi, Mxy = M0 cos 2 phi (x east, y north), no vertical
    term; scalar moment sqrt(sum Mij^2 / 2) = M0 for any double couple."""
    m = e2vp.moment_tensor(22.5, 90., 0., 2.)
    assert np.allclose(m, [2 * np.sin(np.pi / 4), -2 * np.sin(np.pi / 4), 0., 2 * np.cos(np.pi / 4), 0., 0.])
    for strike, dip, rake in ((0., 45., 90.), (30., 60., -45.), (200., 20., 170.)):
        mxx, myy, mzz, mxy, mxz, myz = e2vp.moment_tensor(strike, dip, rake, 3.)
        M = np.array([[mxx, mxy, mxz], [mxy, myy, myz], [mxz, myz, mzz]])
        assert np.isclose(np.trace(M), 0., atol=1e-12) and np.isclose(np.sqrt(np.sum(M ** 2) / 2), 3.)
        assert np.isclose(np.linalg.det(M), 0., atol=1e-9)  # double couple: one zero eigenvalue
    # thrust on a fault dipping 45 degrees to the east (strike north): compression east-west, Mzz > 0
    mxx, _, mzz, _, _, _ = e2vp.moment_tensor(0., 45., 90., 1.)
    assert np.isclose(mxx, -1.) and np.isclose(mzz, 1.)


def test_sh_transfer():
    """One layer on a half-space: peaks at Vs / 4H with the impedance ratio as amplitude; no layer: 1."""
    layers = [(50., 1000., 200., 2000.)]
    bed = (4000., 2000., 2500.)
    f = np.linspace(0.1, 2., 1901)
    T = e2vp.sh_transfer(f, layers, bed)
    assert np.isclose(f[np.argmax(T)], 200. / (4 * 50.), atol=1e-3)
    assert np.isclose(T.max(), 2500. * 2000. / (2000. * 200.), rtol=1e-3)
    assert np.allclose(e2vp.sh_transfer(f, [(50., 4000., 2000., 2500.)], bed), 1.)


def test_true_model_and_grid():
    """Sharp model at the nodes of config_1d's grid, the 13 GLL depths of the three sedimentary elements:
    L1, L2, L3 (an interface node in the layer below it), bedrock below the grid; the exact configuration
    has constant properties everywhere."""
    cfg = e2vp.config_1d("/nonexistent", sources=())
    z = cfg.grid.axis(2)
    edges = -np.cumsum([0.] + [l[0] for l in e2vp.SHARP])
    assert len(z) == 13 and np.allclose(z[[0, 4, 8, 12]], edges[::-1]) and cfg.file_layers == [0, 1, 2]
    lam, mu, rho = e2vp.true_model_1d(cfg.grid)
    assert np.allclose(mu[3, 4, :], mu[0, 0, :])
    for zz, (vp, vs, r) in ((0., (1500., 200., 2100.)), (-17.3, (1800., 350., 2100.)), (-50., (1800., 350., 2100.)),
                            (-150., (2500., 650., 2200.)), (-205.4, (2500., 650., 2200.))):
        k = np.argmin(np.abs(z - zz))
        assert np.allclose((lam[3, 4, k], mu[3, 4, k], rho[3, 4, k]), e2vp.lame(vp, vs, r))
    assert e2vp.config_1d("/nonexistent", exact=True, sources=()).file_layers == []
    assert sum(th for th, _ in cfg.layer_list) == cfg.depth


def test_time_functions():
    """I1: unit integral, peak at 3 / fmax, no spectrum above fmax beyond the truncation leakage, flat
    (within 1 %) up to 0.5 fmax; I2: its integral, from 0 to 1."""
    for fmax in (1., 2.):
        t, i1 = e2vp.time_function("I1", fmax)
        _, i2 = e2vp.time_function("I2", fmax)
        dt = t[1] - t[0]
        assert np.isclose(np.sum(0.5 * (i1[1:] + i1[:-1])) * dt, 1.) and i2[0] == 0. and np.isclose(i2[-1], 1.)
        assert np.isclose(t[np.argmax(i1)], 3. / fmax)
        S = np.abs(np.fft.rfft(i1, 4 * len(i1))) * dt
        fr = np.fft.rfftfreq(4 * len(i1), dt)
        assert S[fr > 1.02 * fmax].max() < 1e-2 and np.allclose(S[fr <= 0.5 * fmax], 1., atol=1e-2)


needs_e2vp2 = pytest.mark.skipif(not os.path.exists(f"{e2vp2.DATA}/velocity_profile.txt"),
                                 reason="E2VP2-MODEL archive not found (E2VP2_DIR)")


@needs_e2vp2
def test_e2vp2_basin(tmp_path):
    """3-D model: the profile at the epicentre (two gradient layers then the bedrock and the crust),
    the meshed surface brought back to 0 at the edge of the box, stations on it, the source at its depth;
    the expected values are read from the archive (its values are not copied here)."""
    npz = tmp_path / "basin.npz"
    e2vp2.prepare(npz, half=3000., margin=200.)
    cfg, b = e2vp2.config_3d(str(tmp_path / "c"), npz, fmax=0.5)
    s0 = b.mesh_surface(0., 0.)
    s = b._bilinear(b.surface, 0., 0.)
    t1, t2 = s - b._bilinear(b.mp, 0., 0.), s - b._bilinear(b.bedrock, 0., 0.)
    assert 0. < t1 < t2
    d = np.array([0., t1 / 2, t2 - 1e-3, t2 + 1., 3000.])
    vp, vs, rho = b.properties(np.zeros(5), np.zeros(5), s0 - d)
    L = e2vp2.read_profile()
    assert np.allclose(vs, [L[0]["vs"][0], np.mean(L[0]["vs"]), L[1]["vs"][1], L[2]["vs"][0], L[3]["vs"][0]], atol=1.)
    assert np.allclose(vp[[0, 3, 4]], [L[0]["vp"][0], L[2]["vp"][0], L[3]["vp"][0]]) and np.isclose(rho[0], L[0]["rho"][0])
    edge = np.linspace(-3000., 3000., 13)
    assert np.allclose(b.mesh_surface(edge, 3000.), 0.) and np.allclose(b.mesh_surface(-3000., edge), 0.)
    st = np.array(cfg.stations)
    assert np.allclose(st[:-3, 2], [b.mesh_surface(x, y) for x, y in st[:-3, :2]])
    assert np.isclose(cfg.sources[0].coords[2], e2vp2.read_source()["z"] - b.z0)


def test_profile_basis():
    """Sediment Vs of interface-pw: the hat functions of the knots sum to 1 and interpolate linearly between them; the
    knots of v0 + g depth give back that line down to the last knot, and their basis that of interface-v0g."""
    import run_e2vp3d as r
    depth = np.linspace(0., 500., 101)
    phi = r.profile_basis(depth, "interface-pw")
    assert phi.shape == (len(r.ZK), depth.size) and np.allclose(phi.sum(axis=0), 1.)
    p = np.array([150., 260., 330., 470., 600., 900.])
    assert np.allclose(p @ phi, np.interp(depth, r.ZK, p))
    k = r.knots(dict(v0=np.array(300.), g=np.array(1.25)))
    assert np.allclose(k @ phi, 300. + 1.25 * depth)
    assert np.allclose(np.array([300., 1.25]) @ r.profile_basis(depth, "interface-v0g"), 300. + 1.25 * depth)
    assert np.array_equal(r.knots(dict(zk=np.array(r.ZK), vsk=p)), p)

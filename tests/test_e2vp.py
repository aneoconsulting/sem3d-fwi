"""Tests of the E2VP case (examples/euroseistest/e2vp.py) that need no simulation."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "euroseistest"))
import e2vp  # noqa: E402


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
    """Sharp model at the nodes of config_1d's grid: L1, L2, L3 then the bedrock, the grid reaching below
    the sediments; the exact configuration has constant properties everywhere."""
    cfg = e2vp.config_1d("/nonexistent", sources=())
    assert cfg.grid.xmin[2] <= -e2vp.sediment_bottom() and cfg.grid.xmax[2] == 0. and cfg.file_layers == [0, 1, 2]
    lam, mu, rho = e2vp.true_model_1d(cfg.grid)
    z = cfg.grid.axis(2)
    for zz, (vp, vs, r) in ((0., (1500., 200., 2100.)), (-50., (1800., 350., 2100.)),
                            (-150., (2500., 650., 2200.)), (-207., e2vp.BEDROCK)):
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

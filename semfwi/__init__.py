"""Adjoint-state full-waveform inversion on top of SEM3D.

Modules:
  config   : case parameters (mesh, time, parameter grid, stations, sources);
  model    : parameter grid, Lambda/Mu/Rho files, trilinear interpolation;
  sem_io   : writing SEM3D inputs, reading traces, the log and strains;
  runner   : single mesh, launching simulations, cache, parallel execution;
  misfit   : residuals, cost function, adjoint source histories;
  gradient : K_lambda, K_mu kernels and projection onto the parameter grid;
  task     : one source end to end (forward, residual, adjoint, kernels);
  backend  : running the sources on the local machine (LocalBackend);
  problem  : J(x) and grad J(x) over all sources;
  optim    : L-BFGS with Armijo line search, saving and resuming iterations;
  armonik_backend, armonik_worker, armonik_io : running the sources on ArmoniK.
"""

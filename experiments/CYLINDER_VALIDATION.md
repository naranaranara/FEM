# Three-stage crossed-cylinder study

Run from the repository root:

```powershell
& "C:\Users\Owner\lrgb_venv\Scripts\python.exe" -B -u run.py cylinder-validation --repeats 3 --steps 8
```

New learning entry point: `src/simulation/test_cylinder_validation.py`.
Meaningful changes use `##r##` comments/docstrings.

1. Co-refine `(n_angle,n_axis,R)` through `(32,7,.65)`, `(64,13,.325)`, `(128,25,.1625)`. On each cylinder, bisect RIMLS roots along exact surface normals at a fixed 9x9 grid. Select the first level whose maximum displacement is at most 10% of the imposed indentation. This is a pilot geometric criterion, not a force-accuracy guarantee. Only the existing quadrature and normal interpolation are used.
2. Hold selected geometry and R fixed; solve complete load paths with `d_hat=.001,.0005,.00025`. Keep barrier coefficient 1. Stop the sweep as soon as two completed final forces change by at most 2%; a third distance is then unnecessary. Store all load statuses, forces and gaps. Compare adjacent forces (2% pilot threshold), and separately compare with Hertz (10% pilot threshold).
3. Use the smallest activation distance with a completed load path. Start all methods from the SAME GN state at the penultimate load and apply the same final closure. Warm each method once, then rotate the method order over three measured repeats. Every run resets curvature history. Compare GN, local limited-memory Gap SR1 and aggregate Gap Structured BFGS. Times include nonlinear iterations and CCD, exclude assembly, condensation, diagnostics and construction of the shared starting state. These are final-load timings, not entire-simulation speedups.

The exact Schur complement in `src/mechanics/condensation.py` eliminates interior FEM coordinates for constant linear elasticity. All geometric dependencies remain in the retained coordinates; full coordinates are reconstructed for volume checks, CCD and force diagnostics. All methods use the same Cholesky direction solver, with eigenvalue shifting only when factorization fails. This is different from the default sparse solver and is applied equally to every method in this study.

Structured BFGS uses the current weighted geometry secant `z=sum(w*b1*delta(grad gap))` and the SAME current GN Hessian. It updates only the missing term. Its aggregate history resets when any active gap changes; local SR1 resets individual gap histories. This is a documented baseline design difference, not a proof of superiority over all structured quasi-Newton methods. It adapts the update formula, not Petra's complete algorithm.

Results are checkpointed under `experiments/cylinder_validation_results/<timestamp>/`: metadata/source hashes, surface errors, normal-ray roots, load paths, activation summary, final-load raw traces/medians and equilibrium coordinates. If the surface criterion fails or every load path fails, the runner stops without reporting successful timing comparisons.

Finite cylinders have clamped end disks and a fixed contact query patch. They are not Hertz half-spaces. Radial element refinement is NOT part of this study (`n_radial=1`), so a small activation sensitivity alone does not establish FEM mesh convergence. The printed Hertz validation flag only applies to the stated pilot checks; boundary/domain and volume-mesh convergence remain separate requirements.

## Resume only the solver comparison

```powershell
& "C:\Users\Owner\lrgb_venv\Scripts\python.exe" -B -u run.py cylinder-validation --benchmark-from experiments/cylinder_validation_results/2026-10-31_130924 --repeats 3 --max-iter 60
```

This reuses the completed surface and activation studies, rebuilds the selected condensed FEM problem, and reconstructs the penultimate equilibrium ONCE by unloading from the saved final state. That common initial state is used by all three methods at the original final closure. The reconstruction is outside measured timings and is recorded in `common_start.json`. It does not repeat the activation sweep. An interrupted third activation path is stored as `aborted_loads_...json` and excluded from convergence and timing summaries.

## Accepted-step stiffness ratio

```powershell
& "C:\Users\Owner\lrgb_venv\Scripts\python.exe" -B -u run.py cylinder-validation --rk-from experiments/cylinder_validation_results/2026-10-31_130924 --max-iter 60
```

The optional `track_rk=True` argument in `two_body.solve` records `r_k = d.T @ (g_new-g_old) / (d.T @ H_GN_old @ d)` for accepted GN steps. It reuses the existing total gradients and unregularized GN Hessians. No extra gap Hessian or gradient evaluation is requested within that solve. Ratios are evaluated in the exact Schur-complement coordinates, equivalent to the full free-DOF energy along the eliminated-interior path. This is an average along an accepted finite step, not the instantaneous curvature correction or a generalized eigenvalue. In particular, barrier coefficient variation across a large step can change this ratio even if the local GN Hessian is accurate. The separate diagnostic replay is excluded from the benchmark times; results are saved in `rk_diagnostic.json`.

# Completed cylinder validation study

Activation sweep stopped after two completed paths at user request. The force changes by 0.0093578 percent when d_hat is halved (0.001 to 0.0005). The third, partial path is archived separately and is excluded.

Surface zero-set maximum error: 0.00551012491 -> 0.00138879005 -> 0.00040349604. Selected n_angle=128, n_axis=25, n_radial=1, R=0.1625; nodes/body=3225. Full free DOFs=17802, exact condensed contact DOFs=1938.

Final force=0.09184139385, Hertz reference=0.01947354697, force/reference=4.71621. Hertz validation did NOT pass. Geometry improvement and activation insensitivity do not establish volume mesh or domain convergence.

| Method | Converged | Median updates | Median final-load wall seconds |
| --- | --- | --- | --- |
| gn | 3/3 | 9 | 68.3063 |
| sr1_gap | 3/3 | 9 | 71.5243 |
| structured_bfgs_gap | 3/3 | 9 | 72.1290 |

All methods used the same start, energy, gradient, final closure, CCD and condensed direction solver. Times exclude assembly, condensation, shared start reconstruction and final diagnostics. Each timed run starts with fresh curvature history. All final retained coordinates differ by less than 4.3e-13 from GN. In this case the curvature approximations do not reduce iterations and add about 5 percent overhead; this does not establish their performance on other contact conditions.

The first BFGS run applied 6 updates and 3 active-set resets. Local Gap SR1 applied 1842 per-gap updates. None of the three methods required a solver eigenvalue shift, and each had two half-step backtracks.

## GN accepted-step r_k

| Step k | r_k | alpha_k | step norm |
| --- | --- | --- | --- |
| 0 | 1.00002833 | 0.5 | 0.00564924 |
| 1 | 1.00008359 | 0.5 | 0.00282445 |
| 2 | 0.999997992 | 1 | 0.00282444 |
| 3 | 0.586674311 | 1 | 2.14878e-07 |
| 4 | 0.653737905 | 1 | 1.73312e-08 |
| 5 | 0.781968419 | 1 | 1.4137e-08 |
| 6 | 0.928219488 | 1 | 5.17059e-09 |
| 7 | 0.99405731 | 1 | 5.9945e-10 |
| 8 | 0.999747779 | 1 | 3.53859e-11 |

These ratios compare total-gradient secants with the previous total GN Hessian. They are finite-step average stiffness ratios, not eigenvalues or direct measurements of the missing geometric term. Ratios near one at the end indicate a good directional secant prediction there. Intermediate ratios below one may include barrier-coefficient variation across the step; they cannot by themselves identify missing gap curvature as the cause. The diagnostic replay reaches the same final GN state (difference 0), and is excluded from timing.

54 regression checks passed, including exact condensation equivalence, coupled equilibrium, negative missing-term BFGS secants, activation early stopping, and r_k=1 for an exact quadratic.

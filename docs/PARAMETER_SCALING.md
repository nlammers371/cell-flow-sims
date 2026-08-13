# Parameter dimensions and scaling audit

## Status and scope

This document derives the dimensions and useful nondimensional combinations of
the model implemented in `cell_sphere_sim/forces.py` and
`cell_sphere_sim/planar/engine.py`. It is intended to make parameter audits
reproducible. It does **not** establish that the force law or default values are
biologically correct; that requires experimental calibration and sensitivity
analysis.

The example configuration currently assigns no physical unit system. Its
numbers should therefore be treated as internally consistent simulation units,
not automatically as micrometres, minutes, or nanonewtons.

## Dimensional table

Let `[L]`, `[T]`, and `[F]` denote length, time, and force.

| Parameter | Code name | Dimension | Role |
|---|---|---:|---|
| Cell radius | `R` | `[L]` | Contact length scale |
| Box lengths | `Lx`, `Ly` | `[L]` | Periodic domain |
| Motility force | `Fm` | `[F]` | Active force along polarity |
| Drag | `gamma_s` | `[F T/L]` | Converts force to speed |
| Repulsion strength | `k_rep` | `[F]` | Scale of the repulsive force |
| Adhesion parameter | `w` | `[F]^(1/2)` | Pair adhesion uses `w_i w_j` |
| CIL rate | `fcil` | `[T]^-1` | Polarity relaxation rate |
| Rotational diffusion | `Dr` | `[T]^-1` | Angular variance rate; radians are dimensionless |
| Timestep | `dt` | `[T]` | Outer integration interval |
| Division rate | `lambda_div` | `[T]^-1` | Retained state parameter; unused in planar MVP |
| Division pause | `tau_div` | `[T]` | Retained state parameter; unused in planar MVP |
| Hard-core ratio | `alpha_dmin` | dimensionless | Sets `d_min / sigma` |
| Initial clearance | `initial_min_separation_factor` | dimensionless | Initialization only |

The perhaps surprising dimension of `w` follows directly from

```text
F_adh = (w_i w_j / r_bar) * (sigma - d).
```

The length ratio is dimensionless, so `w_i w_j` must have force units. `w` is
therefore not itself an adhesion force in the implemented formula.

`eps` is added to a distance and consequently has length dimension, although
the default numerical value is specified in simulation length units.

## Reference scales

Choose a reference contact diameter `sigma_0`, force `k_rep`, and drag
`gamma_s`. They define

```text
velocity scale V_0 = k_rep / gamma_s
time scale     T_0 = gamma_s * sigma_0 / k_rep
```

The main nondimensional controls are then

```text
motility ratio       M_i   = Fm_i / k_rep
adhesion ratio       A_ij  = w_i * w_j / k_rep
CIL rate             C_i   = fcil_i * T_0
diffusion rate       D_i   = Dr_i * T_0
dimensionless step   h     = dt / T_0 = dt * k_rep / (gamma_s * sigma_0)
hard-core ratio      a     = alpha_dmin
```

These, plus geometry (`R_i/sigma_0`, box size, density, and state fractions),
control dynamically similar runs. The raw parameters are **not** individually
dimensionless in the current code.

For equal or unequal radii, `r_bar = sigma/2`. Writing normalized distance
`q=d/sigma`, the pair force becomes

```text
F_pair / k_rep = ((1-q)/(q-a))^(3/2) - 2 A_ij (1-q),   a < q < 1.
```

Thus `A_ij` and `a` determine the passive pair equilibrium separation when it
exists. Increasing adhesion can move that equilibrium into a much stiffer part
of the repulsive curve even though the adhesion term itself is linear.

## Useful relative times and lengths

Several ratios are useful when interpreting collective behavior:

```text
motility speed                 v_m = Fm / gamma_s
persistence length / diameter     = Fm / (gamma_s * Dr * sigma_0) = M/D
CIL / rotational diffusion        = fcil / Dr = C/D
CIL during one diameter crossing  = fcil * sigma_0 / v_m = C/M
adhesion / motility force          = w_i*w_j / Fm = A_ij/M
```

The last ratio compares force scales only. Actual adhesive force also depends
on overlap through `2(1-q)`.

## Numerical scaling and current limitation

The fixed `dt` is not by itself a guarantee of stability because the
contact-force slope grows rapidly near `d_min`. Strong adhesion can pull a
many-cell cluster into that stiff region; one explicit Euler step can then
cross the nominal hard core and produce an enormous periodic jump. The current
force regularization limits division by zero but does not impose a geometric
non-crossing constraint. This is a known implementation weakness, not a valid
physical prediction.

The proposed correction is an explicit core-separation constraint plus
displacement-triggered substeps only when a tentative move could tunnel through
a core. Unlike routine stiffness-based substepping, normal steps retain one
force evaluation. A smooth finite force regularization may still be useful, but
a force cap alone cannot guarantee non-crossing at finite `dt`.

## Calibration implications

A physical calibration should state at minimum:

1. the length and time represented by one simulation unit;
2. measured or inferred cell radius, free speed, and polarity persistence;
3. which measurement constrains drag versus force (speed alone constrains only
   `Fm/gamma_s`);
4. contact-separation or force data used to identify `A_ij`, `k_rep`, and
   `alpha_dmin`;
5. a CIL reorientation timescale used to constrain `fcil`;
6. sensitivity to `dt` and any hard-core numerical controls.

Without such information, many raw parameter sets are related by scaling and
cannot be uniquely identified from trajectories alone.

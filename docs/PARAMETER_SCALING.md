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

Let $[L]$, $[T]$, and $[F]$ denote length, time, and force.

| Parameter | Code name | Dimension | Role |
|---|---|---:|---|
| Cell radius | `R` | $[L]$ | Contact length scale |
| Box lengths | `Lx`, `Ly` | $[L]$ | Periodic domain |
| Motility force | `Fm` | $[F]$ | Active force along polarity |
| Drag | `gamma_s` | $[F\,T/L]$ | Converts force to speed |
| Repulsion strength | `k_rep` | $[F]$ | Scale of the repulsive force |
| Adhesion parameter | `w` | $[F]^{1/2}$ | Pair adhesion uses $w_iw_j$ |
| CIL rate | `fcil` | $[T]^{-1}$ | Polarity relaxation rate |
| Rotational diffusion | `Dr` | $[T]^{-1}$ | Angular variance rate; radians are dimensionless |
| Timestep | `dt` | $[T]$ | Outer integration interval |
| Division rate | `lambda_div` | $[T]^{-1}$ | State-specific unrestricted division hazard |
| Division pause | `tau_div` | $[T]$ | State-specific post-division motility pause |
| Division separation | `division_separation_factor` | dimensionless | Daughter separation divided by $R_a+R_b$ |
| Projection tolerance | `division_projection_tolerance` | $[L]$ | Allowed residual birth-component overlap |
| Hard-core ratio | `alpha_dmin` | dimensionless | Sets $d_{\min}/\sigma$ |
| Initial clearance | `initial_min_separation_factor` | dimensionless | Initialization only |

The perhaps surprising dimension of `w` follows directly from

```math
F_{\mathrm{adh}}=\frac{w_iw_j}{\bar r}\left(\sigma-d\right).
```

The length ratio is dimensionless, so $w_iw_j$ must have force units. The code
parameter `w` is therefore not itself an adhesion force in the implemented
formula.

The code parameter `eps` ($\varepsilon$) is added to a distance and consequently
has length dimension, although its default numerical value is specified in
simulation length units.

## Equations of motion

For cell $i$, let $\mathbf{p}_i$ be its unit polarity and let
$\mathcal{N}_i$ contain its contacting neighbors. The shared overdamped
translational dynamics are

```math
\begin{aligned}
\gamma_s\frac{d\mathbf{x}_i}{dt}
  &= g_i F_i^{\mathrm m}\mathbf{p}_i + \mathbf{F}_i^{\mathrm{contact}}, \\
\mathbf{F}_i^{\mathrm{contact}}
  &= \sum_{j\in\mathcal{N}_i}\mathbf{F}_{ij}, \\
\mathbf{F}_{ij}
  &= \left[
       k_{\mathrm{rep}}
       \left(\frac{\sigma_{ij}-d_{ij}^{\mathrm{eff}}}
       {d_{ij}^{\mathrm{eff}}-d_{ij}^{\min}}\right)^{3/2}
       - \frac{w_iw_j}{\bar r_{ij}}
       \left(\sigma_{ij}-d_{ij}^{\mathrm{eff}}\right)
     \right]
     \frac{\mathbf{d}_{ij}}{d_{ij}^{\mathrm{eff}}}, \\
\sigma_{ij} &= R_i+R_j, \\
d_{ij}^{\min} &= \alpha_{\mathrm{dmin}}\sigma_{ij}, \\
d_{ij}^{\mathrm{eff}} &= \max\!\left(\lVert\mathbf{d}_{ij}\rVert,
                                      d_{ij}^{\min}+\varepsilon\right), \\
\bar r_{ij} &= \frac{R_i+R_j}{2}.
\end{aligned}
```

Here $\mathbf{d}_{ij}$ points from cell $j$ to cell $i$: it is
$\mathbf{x}_i-\mathbf{x}_j$ on the sphere and the corresponding minimum-image
displacement in the periodic plane. The pair force is zero when
$\lVert\mathbf{d}_{ij}\rVert\geq\sigma_{ij}$. In both engines, the motility
gate $g_i$ is zero during a post-division pause and one otherwise. This gate
does not disable contact forces or further division events.

With $\mathbf{v}_i=d\mathbf{x}_i/dt$, the code advances positions by explicit
Euler, followed by the geometry constraint:

```math
\begin{aligned}
\mathbf{x}_i(t+\Delta t)
  &= \bigl(\mathbf{x}_i(t)+\Delta t\,\mathbf{v}_i\bigr)
     \bmod (L_x,L_y)
  && \text{(periodic plane)}, \\
\mathbf{x}_i(t+\Delta t)
  &= R_E\frac{\mathbf{x}_i(t)+\Delta t\,\mathbf{v}_i}
                 {\left\lVert\mathbf{x}_i(t)+\Delta t\,\mathbf{v}_i\right\rVert}
  && \text{(sphere)}.
\end{aligned}
```

Polarity first relaxes toward the normalized direction away from contacts and
then undergoes rotational diffusion. Here $\widehat{\mathbf{t}}_{ij}$ is the
unit direction from cell $i$ toward neighbor $j$, projected into the local
tangent plane on the sphere:

```math
\begin{aligned}
\mathbf{c}_i
  &= \sum_{j\in\mathcal{N}_i}\widehat{\mathbf{t}}_{ij}, \\
\mathbf{p}_i^{\mathrm{flee}}
  &= -\frac{\mathbf{c}_i}{\lVert\mathbf{c}_i\rVert}, \\
\mathbf{p}_i^{\mathrm{det}}
  &= \mathbf{p}_i^{\mathrm{flee}}
     + e^{-f_i^{\mathrm{CIL}}\Delta t}
       \left(\mathbf{p}_i^{\mathrm{tr}}-\mathbf{p}_i^{\mathrm{flee}}\right), \\
\Delta\theta_i
  &\sim \mathcal{N}\!\left(0,\,2D_{r,i}\Delta t\right), \\
\mathbf{p}_i(t+\Delta t)
  &= \operatorname{normalize}\!\left[
       \mathcal{R}(\Delta\theta_i)\mathbf{p}_i^{\mathrm{det}}
     \right].
\end{aligned}
```

The relaxation line is applied only when
$\lVert\mathbf{c}_i\rVert>\varepsilon$; otherwise
$\mathbf{p}_i^{\mathrm{det}}=\mathbf{p}_i^{\mathrm{tr}}$. In the plane
$\mathbf{p}_i^{\mathrm{tr}}=\mathbf{p}_i$, while on the sphere it is the old
polarity parallel-transported to the new tangent plane. Rotation is within the
plane or about the new local sphere normal, respectively.

## Reference scales

Choose a reference contact diameter $\sigma_0$, repulsive force scale
$k_{\mathrm{rep}}$, and drag $\gamma_s$. They define

```math
\begin{aligned}
V_0 &= \frac{k_{\mathrm{rep}}}{\gamma_s}
&& \text{(velocity scale)}, \\
T_0 &= \frac{\gamma_s\sigma_0}{k_{\mathrm{rep}}}
&& \text{(time scale)}.
\end{aligned}
```

The main nondimensional controls are then

```math
\begin{aligned}
M_i &= \frac{F_i^{\mathrm m}}{k_{\mathrm{rep}}}
&& \text{(motility ratio)}, \\
A_{ij} &= \frac{w_iw_j}{k_{\mathrm{rep}}}
&& \text{(adhesion ratio)}, \\
C_i &= f_i^{\mathrm{CIL}}T_0
&& \text{(CIL rate)}, \\
D_i &= D_{r,i}T_0
&& \text{(diffusion rate)}, \\
\Lambda_i &= \lambda_i^{\mathrm{div}}T_0
&& \text{(division rate)}, \\
\Theta_i &= \frac{\tau_i^{\mathrm{div}}}{T_0}
&& \text{(division pause)}, \\
h &= \frac{\Delta t}{T_0}
   = \frac{\Delta t\,k_{\mathrm{rep}}}{\gamma_s\sigma_0}
&& \text{(dimensionless step)}, \\
a &= \alpha_{\mathrm{dmin}}
&& \text{(hard-core ratio)}.
\end{aligned}
```

These, plus division placement, geometry ($R_i/\sigma_0$, box size, density,
and state fractions), control dynamically similar runs. The raw parameters are
**not** individually dimensionless in the current code.

For equal or unequal radii, $\bar r=\sigma/2$. Writing normalized distance
$q=d/\sigma$, the pair force becomes

```math
\frac{F_{\mathrm{pair}}}{k_{\mathrm{rep}}}
= \left(\frac{1-q}{q-a}\right)^{3/2}
  - 2A_{ij}(1-q),
\qquad a<q<1.
```

Thus $A_{ij}$ and $a$ determine the passive pair equilibrium separation when
it exists. Increasing adhesion can move that equilibrium into a much stiffer
part of the repulsive curve even though the adhesion term itself is linear.

## Useful relative times and lengths

Several ratios are useful when interpreting collective behavior:

```math
\begin{aligned}
v_m &= \frac{F^{\mathrm m}}{\gamma_s}
&& \text{(motility speed)}, \\
\frac{\ell_{\mathrm p}}{\sigma_0}
  &= \frac{F^{\mathrm m}}{\gamma_sD_r\sigma_0}
   = \frac{M}{D}
&& \text{(persistence length per diameter)}, \\
\frac{f^{\mathrm{CIL}}}{D_r}
  &= \frac{C}{D}
&& \text{(CIL relative to rotational diffusion)}, \\
\frac{f^{\mathrm{CIL}}\sigma_0}{v_m}
  &= \frac{C}{M}
&& \text{(CIL during one diameter crossing)}, \\
\frac{w_iw_j}{F_i^{\mathrm m}}
  &= \frac{A_{ij}}{M_i}
&& \text{(adhesion relative to motility force)}, \\
\lambda_i^{\mathrm{div}}\tau_i^{\mathrm{div}}
  &= \Lambda_i\Theta_i
&& \text{(expected division hazard accumulated during a pause)}.
\end{aligned}
```

The last ratio compares force scales only. Actual adhesive force also depends
on overlap through $2(1-q)$.

## Proliferation scaling

The implementation treats `lambda_div` as a continuous-time Poisson hazard and
uses `1 - exp(-lambda_div * dt)` per timestep rather than the small-step
approximation `lambda_div * dt`. With no state changes, deaths, or regulation,

```math
E[N_i(t)] = N_i(0)e^{\lambda_i^{\mathrm{div}}t},
\qquad
t_{2,i} = \frac{\ln 2}{\lambda_i^{\mathrm{div}}}.
```

Thus changing the mechanics reference scale while preserving similarity also
requires holding $\Lambda_i=\lambda_i^{\mathrm{div}}T_0$ and
$\Theta_i=\tau_i^{\mathrm{div}}/T_0$ fixed. The placement factor is already
dimensionless; at its configured value of one, equal-radius daughter centers
are separated by one contact diameter.

The optional post-division projection is not a physical timescale or force. It
is a zero-time configuration correction, and its displacements therefore do
not enter the reported velocity. The maximum and RMS projection displacement
should be audited relative to $\sigma_0$; values that are not small indicate
that division events are substantially rearranging the local configuration.

## Numerical scaling and current limitation

The fixed timestep $\Delta t$ is not by itself a guarantee of stability because
the contact-force slope grows rapidly near $d_{ij}^{\min}$. Strong adhesion can
pull a many-cell cluster into that stiff region; one explicit Euler step can
then cross the nominal hard core and produce an enormous periodic jump. The
current force regularization limits division by zero but does not impose a
geometric non-crossing constraint. This is a known implementation weakness,
not a valid physical prediction.

Division insertion now has an event-local geometric projection, but ordinary
mechanics steps still have no core-separation constraint. A future global
correction would require an explicit core constraint plus
displacement-triggered substeps when a tentative move could tunnel through a
core. A smooth finite force regularization may still be useful, but a force cap
alone cannot guarantee non-crossing at finite $\Delta t$.

## Calibration implications

A physical calibration should state at minimum:

1. the length and time represented by one simulation unit;
2. measured or inferred cell radius, free speed, and polarity persistence;
3. which measurement constrains drag versus force (speed alone constrains only
   $F_i^{\mathrm m}/\gamma_s$);
4. contact-separation or force data used to identify $A_{ij}$,
   $k_{\mathrm{rep}}$, and $\alpha_{\mathrm{dmin}}$;
5. a CIL reorientation timescale used to constrain $f_i^{\mathrm{CIL}}$;
6. measured division hazards or doubling times and post-division pause times;
7. sensitivity to $\Delta t$ and any hard-core numerical controls.

Without such information, many raw parameter sets are related by scaling and
cannot be uniquely identified from trajectories alone.

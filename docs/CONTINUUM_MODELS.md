# Continuum phase-separation and aggregation models

This document is the audit reference for the continuum workbench in
`cell_sphere_sim.continuum`. It states the equations that the code actually
solves, the assumptions behind them, the parameter combinations that matter,
and the numerical compromises. The continuum workbench is separate from the
off-lattice cell-agent simulator: a pixel is a coarse-grained field sample,
not a cell.

## Scope and interpretation

All four models evolve continuous fields on a square, periodic, two-dimensional
domain. They are intentionally minimal teaching and hypothesis-building models.
They do not include an embryo boundary, measured flow, discrete proliferation,
cell shapes, or parameter fitting. A dense region is not automatically a
literal aggregate of resolved cells.

The numbers supplied by the presets are in **consistent arbitrary units**. They
are not inherently micrometres, minutes, forces, temperatures, or single-cell
speeds. A biological interpretation requires the user to choose reference
length, time, density/order-parameter, and (for Model B) energy scales. The
scaling sections below identify dimensionless combinations that can be compared
without committing to a unit convention.

## 1. Passive Model B (Cahn–Hilliard)

The conserved scalar order parameter is `phi`:

\[
\partial_t\phi=M\nabla^2\mu+\nabla\cdot\boldsymbol\Lambda,
\qquad
\mu=a\phi+b\phi^3-\kappa\nabla^2\phi .
\]

Without dynamic noise, this is gradient flow of

\[
F[\phi]=\int\left[\frac{a}{2}\phi^2+\frac{b}{4}\phi^4+
\frac{\kappa}{2}|\nabla\phi|^2\right]dA,
\qquad
\frac{dF}{dt}=-M\int|\nabla\mu|^2dA\leq0.
\]

The implementation copies the zero Fourier mode exactly, so the spatial
integral of `phi` is conserved. For deterministic Passive Model B, the engine
also evaluates `F` on every proposed step and rejects/halves a step that would
increase it beyond floating-point tolerance.

For `a < 0` and `b > 0`, the symmetric bulk model has coexistence (binodal)
values

\[
\phi_{\rm bin}=\pm\sqrt{-a/b}
\]

and spinodal limits

\[
\phi_{\rm sp}=\pm\sqrt{-a/(3b)}.
\]

A uniform mean inside the spinodal is linearly unstable. Between spinodal and
binodal it is metastable: a sufficiently small droplet dissolves while a
sufficiently large one grows. The workbench's droplet initializer subtracts a
uniform offset after constructing the droplet, so changing droplet radius does
not silently change total mass.

Linearizing about mean `phi_bar` gives

\[
\sigma(k)=-Mk^2\left(a+3b\bar\phi^2+\kappa k^2\right).
\]

When the term in parentheses at `k=0` is negative, the fastest continuum mode is

\[
k_*^2=-\frac{a+3b\bar\phi^2}{2\kappa}.
\]

This relation is a useful grid check: `k*` should be well below the de-aliasing
cutoff and the associated wavelength should fit multiple times in the box.

Parameter roles:

| GUI name | Symbol | Role |
| --- | --- | --- |
| Mobility | `M` | Changes relaxation time, not equilibrium coexistence. |
| Bulk a | `a` | Controls local convexity and whether demixing is possible. |
| Bulk b | `b` | Saturates the order parameter; must be positive here. |
| Interface cost | `kappa` | Penalizes gradients; sets interface width and tension. |

A natural amplitude is `phi0=sqrt(abs(a)/b)`, interface length is
`xi=sqrt(kappa/abs(a))`, and diffusive time at length `L0` is
`T0=L0^2/(M abs(a))`. After choosing these scales, only ratios such as `xi/L0`
and `phi_bar/phi0` remain. Thus increasing `M` only accelerates a deterministic
trajectory in continuum time; changing `a`, `b`, or `kappa` changes its shape
and thermodynamics.

Presets: **Stable uniform**, **Spinodal**, **Nucleation**, **Many droplets**, and
**Flat interface**.

## 2. Active Model B

Active Model B uses the same conserved update but adds a gradient-dependent,
non-integrable chemical-potential term:

\[
\mu=a\phi+b\phi^3-\kappa\nabla^2\phi+\lambda|\nabla\phi|^2.
\]

`lambda` is an activity coefficient. It is **not** propulsion speed and it is
**not** temperature. For `lambda=0`, the implementation calls the same stepping
function as Passive Model B and is bit-for-bit identical for the same state,
seed, and parameters. For nonzero `lambda`, the passive `F` above is no longer a
Lyapunov function and the GUI deliberately does not plot it.

Using the passive interface scale `xi`, the ratio `lambda phi0/kappa` measures
the active term relative to the square-gradient term. Its sign matters; its
absolute value is not meaningful until `phi`, length, and chemical-potential
scales have been chosen.

This is the original minimal Active Model B, not Active Model B+. It can alter
interfacial dynamics but does not contain the additional non-gradient current
responsible for the full AMB+ phenomenology.

Presets: **Passive limit**, **Active demixing**, and **Active droplet**.

## 3. Density–polarization model (mechanistic MIPS)

This model retains density `rho` and polarization density `p=(px,py)`:

\[
\partial_t\rho=-\nabla\cdot[v(\rho)\mathbf p]+D_\rho\nabla^2\rho,
\]

\[
\partial_t\mathbf p=-D_r\mathbf p-
\frac12\nabla[v(\rho)\rho]+D_p\nabla^2\mathbf p+\boldsymbol\eta,
\]

\[
v(\rho)=v_{\min}+(v_0-v_{\min})e^{-\alpha\rho}.
\]

The factor `1/2` is the two-dimensional angular-moment closure. `p` is a
coarse-grained polarization density, not a prescribed velocity field. The
closure neglects higher orientational moments and is most defensible when
polarization relaxes faster than density patterns.

At the current mean density, the GUI reports

\[
1+\frac{d\ln v}{d\ln\rho}
=1-\frac{\alpha\rho(v_0-v_{\min})e^{-\alpha\rho}}{v(\rho)}.
\]

A negative value is the familiar local slowdown criterion for MIPS in the
diffusive closure. It is a diagnostic, not a finite-domain phase boundary:
`D_rho`, `D_p`, box size, and permitted wave numbers can stabilize a run even
when it is negative.

Useful dimensionless groups are `alpha rho_bar`, `vmin/v0`, the persistence
length relative to the box `v0/(Dr L)`, and diffusivities
`D_rho Dr/v0^2` and `D_p Dr/v0^2`. These show why `v0`, `Dr`, and diffusion
cannot be interpreted independently after rescaling.

The zero density mode is copied exactly. Noise is added to polarization, not
directly to density. Negative trial density is never silently clipped: the
engine retries at half `dt`.

Presets: **Constant speed**, **Weak slowing**, **MIPS**, **Seeded cluster**,
**High persistence**, **Low persistence**, and **Matched field**. The last uses
a positive scalar initializer intended for comparison mode; it does not claim
that parameters in the two model families are matched.

## 4. Keller–Segel autochemotaxis

Cell density `rho` secretes and follows a signal `c`:

\[
\partial_t\rho=D_\rho\nabla^2\rho-
\chi\nabla\cdot(\rho\nabla c)+
D_{\rm crowd}\nabla^2(\rho^m)+R(\rho),
\]

\[
\partial_t c=D_c\nabla^2c+\alpha_c\rho-k_c c.
\]

The optional growth term is logistic,
`R=r rho (1-rho/K)`, and `r=0` in every standard preset. With growth off,
density mass is conserved. The **dynamic** signal mode integrates the second
equation. The **quasistatic** mode instead solves

\[
(k_c-D_c\nabla^2)c=\alpha_c\rho
\]

in Fourier space each step. It assumes signal relaxation is fast compared with
density evolution and requires `kc>0` for a defined zero mode.

The screened signaling length is

\[
\ell_c=\sqrt{D_c/k_c}.
\]

For a homogeneous density, an approximate long-wave competition is

\[
\frac{\chi\alpha_c\bar\rho}{k_c+D_ck^2}
\quad\hbox{versus}\quad
D_\rho+mD_{\rm crowd}\bar\rho^{m-1}.
\]

The left side promotes aggregation and the right side disperses it. This is a
linear heuristic, not an exact threshold for finite-amplitude or finite-box
dynamics. Classical Keller–Segel attraction without enough diffusion/crowding
can concentrate without bound. The workbench warns when chemotaxis is enabled
with zero crowding, but that warning does not prove other parameter choices are
safe.

For numerical stability, the nonlinear crowding flux is split into an explicit
residual and an implicit linear stabilizer using the maximum local derivative
`m max(rho)^(m-1)`. This does not change the stated PDE in the small-`dt` limit.
It avoids making the inexpensive teaching solver unusably stiff, but convergence
still has to be checked by reducing `dt` and increasing resolution.

Presets: **Diffusion only**, **Weak response**, **Aggregation**, **Fast signal**,
**Merging peaks**, **Long range**, **Short range**, and **Classical collapse**.
The aggregation presets use crowding regularization. Classical collapse sets
crowding to zero on purpose and carries a visible hazard warning.

## Noise and initial conditions

`Initial amplitude` controls only the one-time random perturbation. `Dynamic
noise` injects new random values at each PDE step. The seed controls both and
runs are deterministic for a fixed NumPy/FFT implementation.

For conserved scalar noise, the code generates two real Gaussian flux fields
and adds their spectral divergence. The zero mode is explicitly zero, so noise
cannot change total mass. In the polarization model, dynamic noise is additive
in `p`. Noise amplitudes are discretization-level values in the present code;
they have not been calibrated to a continuum fluctuation–dissipation convention
and should not be compared across grid spacings without an explicit scaling
choice.

Available scalar initial conditions are uniform noise, one mass-compensated
droplet, multiple periodic droplets, a two-interface periodic stripe, and a
radial profile. Programmatic import is supported by passing a `(N,N)` float
array as `initial_scalar` to `ContinuumEngine`; see “Using measured data” below.

## Numerics and safeguards

- Square periodic grid, `float64`, default `128 x 128`.
- FFT pseudospectral gradients and Laplacians.
- Nonlinear products are formed in real space and filtered with the 2/3 rule.
- Fourth-order Model B stiffness is implicit; its local chemical potential is
  explicit. Diffusion, polarization relaxation, and signal decay are implicit.
- `dt` is the integration step. `substeps_per_frame` controls work per rendered
  frame. Matplotlib's interval controls display cadence. Changing rendering
  frequency cannot alter a fixed number of PDE steps.
- Cached coordinate, wave-number, Laplacian, biharmonic, and de-alias arrays are
  reused for the entire run.
- A conserved zero mode is copied exactly. Typical relative mass error is near
  `1e-15`, well below the `1e-8` acceptance threshold.
- A trial with a materially negative density or non-finite value is rejected and
  retried at half `dt`. If no valid step exists above `minimum_dt`, the run stops
  with `StepRejected`; it does not continue with clipped physics.
- A negative value no larger than `negative_tolerance` is treated as FFT
  roundoff: it is set to zero, the pre-correction sum is restored, and the event
  is written to export metadata.

The semi-implicit methods are first-order in time. They are efficient and
appropriate for exploration, but they are not a substitute for a convergence
study. For a scientific result, repeat at half `dt`, increase `N`, hold physical
box size fixed, and verify the reported observable is unchanged.

## Diagnostics

The live and exported diagnostics include mass and relative error, variance,
minimum, maximum, periodic four-connected cluster count, largest cluster area
fraction, and a structure-factor length

\[
\ell_S=2\pi\frac{\sum_{k>0}S(k)}{\sum_{k>0}kS(k)}.
\]

Cluster threshold is an explicit configuration value. If omitted, it defaults
to the instantaneous spatial mean. Cluster counts are therefore operational
measurements and should always be reported with the threshold. Periodic labels
touching opposite edges are merged.

Passive free energy is shown only for Passive Model B and the exact `lambda=0`
Active Model B limit. The MIPS slowdown criterion and Keller–Segel signaling
length are model-specific diagnostics.

## Using measured data

To initialize from an image or reconstructed density field:

1. Register the data onto a square, uniformly spaced grid.
2. Decide whether pixel values represent a signed order parameter (`phi`) or a
   nonnegative density (`rho`). Do not feed arbitrary image intensity into a
   density model without that calibration.
3. Decide how the observed field should tile periodically. Nonmatching opposite
   edges create an artificial seam and high-frequency transient.
4. Choose physical `L`, field normalization, and time unit; convert parameters
   consistently using the scaling relations above.
5. Pass the resulting `float64` array to
   `ContinuumEngine(config, initial_scalar=array)`. Its shape must equal
   `(grid_size, grid_size)`.
6. Export the preprocessing recipe with the run. The built-in metadata captures
   simulation configuration but cannot infer how an external image was
   normalized.

If polarization or signal measurements exist, assign those secondary fields
after engine construction before stepping. If they do not, the present
initializers use zero polarization, an initially uniform dynamic signal, or the
quasistatic signal implied by the imported density.

## Workbench, batch runs, and exports

Activate the repository environment and launch:

```bash
conda activate cell-flow-sims
python examples/run_continuum_workbench.py
```

Headless example:

```bash
python examples/run_continuum_workbench.py \
  --headless --model density_polarization --preset mips --steps 2000
```

The GUI provides run/pause, one-frame step, reset, model and preset selectors,
live primary/secondary fields, polarization or signal-gradient vectors, live
diagnostics, and registry-generated parameters. Numerical sliders update the
next PDE step immediately. A field-semantic control such as signal mode resets
the model. **Setup** exposes the reset-required grid, domain, seed, mean,
initial perturbation, droplet radius/count, and initial-condition type. Dynamic
noise and the visible cluster threshold also have main-window controls. Each
model retains its parameter configuration when switching away and back.

“Export run” writes a PNG, `float64` final fields, diagnostics CSV, and JSON
metadata containing the model/equation version, actual parameters, numerical
grid, `dt`, seed, resolved initial condition, corrections, and Git revision.
`save_animation` supports GIF/MP4 when the relevant Matplotlib writer is
installed.

Two-model programmatic comparison is available through `ContinuumComparison`.
It copies the first model's initial scalar into the second model. A signed
`phi` field cannot initialize a nonnegative-density model.

Two-parameter/seed sweeps use `examples/run_continuum_sweep.py`. The outputs are
CSV, JSON metadata, and an image labeled **operational map (finite-time,
finite-size)**. The runner exposes progress and cancellation; sweeps are not
called equilibrium phase diagrams.

## Validation status and limitations

`tests/test_continuum.py` covers deterministic mass conservation, passive
free-energy descent, exact `lambda=0` equivalence, stable decay, spinodal growth,
subcritical/supercritical droplets, MIPS stability diagnostics and pattern
growth, the exact chemotaxis-free diffusion update, seeded determinism, render
cadence independence, long finite runs, UI controller behavior, periodic
clusters, comparisons, and one saved small regression trajectory per model.

Known limitations:

- Periodic boundaries are not embryo geometry.
- No hydrodynamic momentum equation or externally measured advection is present.
- `phi`, `rho`, and `p` are coarse-grained fields, not individual cells.
- The density–polarization closure omits nematic and higher angular moments.
- Minimal Active Model B omits AMB+ currents.
- Keller–Segel runs can approach true or discretization-induced concentration;
  adaptive `dt` is a detector/mitigation, not proof of a regular solution.
- Cluster metrics depend on threshold and resolution.
- No parameter inference, uncertainty model, or mapping to a particular tissue
  is supplied.

## Extending the registry

Implement a subclass of `ContinuumModel` with `initialize` and `step`, declare
short `ParameterSpec` labels and `PresetSpec` presets, then add the class to
`MODEL_REGISTRY`. The GUI, headless route, diagnostics shell, configuration
memory, sweeps, and base export metadata will discover it automatically. A new
model must declare its nonnegative and conserved fields and add a deterministic
small-grid regression trajectory plus conservation/stability tests.

## Foundations

The passive free-energy construction follows Cahn and Hilliard,
[“Free Energy of a Nonuniform System. I”](https://doi.org/10.1063/1.1744102),
*Journal of Chemical Physics* 28, 258–267 (1958), and the Model B naming follows
Hohenberg and Halperin,
[“Theory of dynamic critical phenomena”](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.49.435),
*Reviews of Modern Physics* 49, 435 (1977).

Continuum coarse-graining of active Brownian-particle separation is developed
by Stenhammar et al.,
[“Continuum Theory of Phase Separation Kinetics for Active Brownian Particles”](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.145702),
*Physical Review Letters* 111, 145702 (2013). The active scalar extension used
here follows Wittkowski et al.,
[“Scalar phi4 field theory for active-particle phase separation”](https://www.nature.com/articles/ncomms5351),
*Nature Communications* 5, 4351 (2014).

The deferred Active Model B+ extension and reverse-Ostwald interpretation are
described by Tjhung, Nardini, and Cates,
[“Cluster Phases and Bubbly Phase Separation in Active Fluids”](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031080),
*Physical Review X* 8, 031080 (2018).

The density-dependent-speed mechanism is grounded in Tailleur and Cates,
[“Statistical Mechanics of Interacting Run-and-Tumble Bacteria”](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.218103),
*Physical Review Letters* 100, 218103 (2008), and reviewed by Cates and Tailleur,
[“Motility-Induced Phase Separation”](https://www.annualreviews.org/content/journals/10.1146/annurev-conmatphys-031214-014710),
*Annual Review of Condensed Matter Physics* 6, 219–244 (2015).

The chemotaxis model descends from Keller and Segel,
[“Initiation of slime mold aggregation viewed as an instability”](https://pubmed.ncbi.nlm.nih.gov/5462335/),
*Journal of Theoretical Biology* 26, 399–415 (1970), DOI
`10.1016/0022-5193(70)90092-5`.

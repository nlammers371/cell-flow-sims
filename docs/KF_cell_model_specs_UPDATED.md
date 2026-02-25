# KF cell model specs (UPDATED)

> Purpose: simulate cell motility + contact mechanics + CIL on the surface of a sphere, with intrinsic hybrid cell states, optional proliferation, and a future-compatible interface for reaction–diffusion ligand fields and a napari-first interactive GUI + HT parameter sweeps.

---

## 0) Core objects and notation

- Cells live on a sphere of radius **R_E** in **3D**:
  - position: **x_i ∈ R^3**, constrained: ||x_i|| = R_E
  - normal: **n_i = x_i / R_E**
- Each cell has a **unit 3D polarity vector** constrained to the local tangent plane:
  - polarity: **p_i ∈ R^3**, with **p_i · n_i = 0** and **||p_i|| = 1**
- Each cell has a **hybrid intrinsic state**:
  - discrete state id: **state_id_i ∈ {0..K-1}** (K usually 1–3)
  - continuous internal variables: **state_vars_i ∈ R^M** (M can be 0 in v1, but engine must support M≥0)
- Optional ligand/field channels (future add-on):
  - C channels (C = 2 default; support C = 3)

---

## 1) Contact geometry

Define chord distance between cells i and j:

- **d_ij = ||x_i - x_j||**
- per-state radii: **R_i = R[state_id_i]**
- **σ_ij = R_i + R_j**
- contact if **d_ij < σ_ij**

Hard-core exclusion distance:

- **d_min_ij = α * σ_ij**, where α ∈ (0,1) is global (or per-state/pair later)
- Numerics: clamp in force evaluation:
  - **d_eff = max(d_ij, d_min_ij + eps)**

Force direction (on i due to j):

- **n̂_ij = (x_i - x_j) / d_eff**

---

## 2) Contact forces: repulsion + adhesion

Only defined for **d_ij < σ_ij** (no long-range attraction).

### 2.1 Repulsion (Hertz-like blowup near d_min)

- **F_rep(d) = k_rep * ((σ - d_eff) / (d_eff - d_min))^(3/2)**

### 2.2 Adhesion (linear in overlap, product-form)

- overlap: **δ = σ - d_eff** (δ > 0 in contact)
- **R̄ = (R_i + R_j)/2**
- per-state adhesion strength: **w_i = w[state_id_i]**
- **F_adh(d) = (w_i * w_j / R̄) * δ**

### 2.3 Net pairwise force magnitude and vector

- **F_mag = F_rep - F_adh**  (positive = net repulsion)
- Force on i:
  - **F_ij = F_mag * n̂_ij**
- Total contact force:
  - **F_contact_i = Σ_{j in neighbors(i), d_ij < σ_ij} F_ij**

---

## 3) Motility and overdamped position update (explicit Euler)

Per-state motility magnitude:

- **F_m_i = F_m[state_id_i]** (may later be overridden by behavior law)

Overdamped velocity:

- **v_i = ( gate_i * F_m_i * p_i + F_contact_i ) / γ_s**

Where **gate_i ∈ {0,1}** is a motility gate (e.g. 0 during division pause).

Explicit Euler:

1. **x_i_tmp = x_i + dt * v_i**
2. project back to sphere:
   - **x_i' = R_E * x_i_tmp / ||x_i_tmp||**
3. update normal: **n_i' = x_i' / R_E**

---

## 4) Polarity update in 3D tangent space

Polarity update consists of:

A) parallel transport due to motion on sphere  
B) CIL relaxation toward a flee direction when contacts exist  
C) rotational diffusion as a random tangent rotation

### 4.1 Parallel transport (minimal rotation mapping n -> n')

Compute:

- **a = n × n'**
- if ||a|| ≤ eps: p_tr = p (then reproject to tangent)
- else:
  - **u = a / ||a||**
  - **θ = atan2(||a||, n·n')**
  - Rodrigues rotation:
    - **p_tr = p cosθ + (u×p) sinθ + u(u·p)(1-cosθ)**

Cleanup (enforce tangent + unit):

- **p_tr ← p_tr - (p_tr·n') n'**
- **p_tr ← p_tr / ||p_tr||**

### 4.2 CIL flee direction (when contacts exist)

Let contacts_i be the set of j with d_ij < σ_ij.

For each contacting neighbor j:

- dvec = x_j - x_i
- tangent component at i:
  - d_t = dvec - (dvec·n_i) n_i
- if ||d_t|| small: skip
- d̂_t = d_t / ||d_t||

Sum and invert:

- s = Σ d̂_t
- if ||s|| small: no CIL target (fallback)
- else:
  - **p_flee = - s / ||s||**
  - (ensure tangent at new normal n': p_flee ← p_flee - (p_flee·n')n'; normalize)

### 4.3 CIL relaxation + rotational diffusion (discrete update)

Per-state parameters:

- **f_cil_i = f_cil[state_id_i]**
- **D_r_i = D_r[state_id_i]**

If contacts_i is non-empty and p_flee defined:

Deterministic exponential relaxation toward p_flee:

- **p_det = p_flee + exp(-f_cil_i * dt) * (p_tr - p_flee)**

Then apply rotational diffusion as a random rotation about axis n':

- sample δ ~ Normal(0, σ_δ^2), with **σ_δ^2 = 2 D_r_i dt**
- rotate:
  - **p' = p_det cosδ + (n'×p_det) sinδ**

Normalize:

- **p' ← p' / ||p'||**

If contacts_i empty (or p_flee undefined):

- skip relaxation; use p_tr in place of p_det and apply only the random rotation:
  - **p' = p_tr cosδ + (n'×p_tr) sinδ**

---

## 5) Proliferation (optional, event-driven)

Per-state division rate:

- **λ_div_i = λ_div[state_id_i]** (later can depend on fields/state_vars via cell_update)

At each step, cell i divides with probability:

- **P(divide) = 1 - exp(-λ_div_i dt)**

Division operation for cell i at time t:

1. Create new cell j.
2. Daughters inherit mother state exactly:
   - state_id_i' = state_id_j = state_id_i
   - state_vars_i' = state_vars_j = state_vars_i
3. Choose random division axis u in tangent plane at n = x_i/R_E:
   - g ~ Normal(0,I3)
   - u_raw = g - (g·n) n
   - u = u_raw / ||u_raw||
4. Place daughters via small symmetric split and reproject:
   - x_i' = Π(x_i + s u)
   - x_j  = Π(x_i - s u)
   - Π(y) = R_E y / ||y||
5. Motility pause (“cells pause during division”):
   - set **gate=0** for both daughters for a duration **τ_div[state]**
   - contact forces still act; only self-propulsion is paused.
6. Stability: perform a short local relaxation (few tiny dt steps) with motility off to resolve overlaps.

---

## 6) Intrinsic state + behavior law (no hardcoded multiplicative modulation)

### 6.1 Hybrid state representation

- state_id: discrete label (K small)
- state_vars: continuous internal variables (M≥0)

### 6.2 Behavior law (Option A recommended)

Behavior parameters (Fm, Dr, f_cil, w, λ_div, τ_div, …) are computed each step by a user-supplied map:

- **behavior_i = B( state_id_i, state_vars_i, fields_i, contact_metrics_i, dt, rng )**

Option A (intended): baseline-by-discrete-state + analytic modulation by continuous vars / fields:

- baseline from tables indexed by state_id
- analytic functions (Hill, sigmoid, etc.) apply deltas/overrides
- engine does not assume multiplicative form; it accepts the output behavior params.

### 6.3 Source/sink coupling to fields (future)

Cells contribute sources/sinks to field channels via:

- **sources_i = S( state_id_i, state_vars_i, fields_i, dt )**  → shape (C,)

These per-cell sources are deposited into the field solver each step.

---

## 7) Future-compatible field/PDE interface (stub now)

Define FieldModel with:

- C channels (C=2 default; allow C=3)
- sample(x, t) -> (N,C)
- reset_sources()
- accumulate_sources(x, sources) where sources is (N,C)
- step(dt)

Engine uses operator splitting each dt:

1) fields_i = field.sample(x,t)  
2) cell_update (state + behavior) using fields_i  
3) mechanics step (positions, polarity, division)  
4) compute sources_i and field.accumulate_sources  
5) field.step(dt), field.reset_sources

No assumptions about separability or multiplicative effects are built in.

---

## 8) HT sweeps + GUI requirements (first-class)

### 8.1 Target scale

- N up to 5000 (typical 1000–3000)

### 8.2 Initialization modes

A) random initialization: configurable N, state composition/allocation, anisotropy
- anisotropy can affect initial position distribution and/or initial polarity distribution

B) data-driven initialization:
- ingest pandas DataFrame in napari Tracks format (e.g., columns [track_id, t, z, y, x])
- choose a time t0 to initialize x
- initialize polarity from finite-difference velocity if available, else random tangent

### 8.3 Visualization (napari-first)

- Points (cells) colored by:
  - state_id (categorical), or
  - scalar features (speed, contact count, field value, state_vars, etc.)
- Vectors for polarity
- Sphere mesh layer
- Density/field shading on sphere mesh (vertex colors):
  - from KDE / binned density now
  - from PDE fields later
- Interactive controls: parameter toggles/sliders + forward simulation

---

## 9) Numerical notes / invariants

Required invariants after each step:

- ||x_i|| == R_E (within tolerance)
- p_i · n_i == 0 (within tolerance)
- ||p_i|| == 1 (within tolerance)
- Contact force evaluation uses d_eff = max(d, d_min + eps)

---

## 10) Versioning

- v1: mechanics + polarity + CIL + division (optional), NullField, random init + tracks init, basic recording
- v2: napari plugin + sphere shading + interactive controls
- v3: sweep runner + metrics + storage
- v4: PDE field solver implementation


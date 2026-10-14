# DMFT Self-Energy Prediction with LSTM Autoencoders

A working example of applying sequence-to-sequence LSTM autoencoders to accelerate
Dynamical Mean-Field Theory (DMFT) calculations for correlated electron materials.

The model predicts the imaginary part of the Matsubara self-energy Im Σ(iω) from the
non-interacting Green's function Im G₀(iω), conditioned on a convergence label
(CT-HYB Monte Carlo warmup steps). This enables extrapolation to well-converged
self-energies from cheaper, under-converged simulation data.

---

## Repository layout

```
dmft-lstm-autoencoder/
├── 01_preprocess_svo.ipynb          ← Step 1: parse raw HDF5 → .npz dataset
├── 02_train_svo_lstm.ipynb          ← Step 2: train base LSTM on SVO
├── 03_transfer_learning_cvo.ipynb   ← Step 3: transfer to CVO with 2 fine-tune points
├── beta_variation/
│   ├── 04_beta_variation_svo.ipynb  ← Variation: train across 8 temperatures (β)
│   ├── model.py                     ← FiLM-conditioned LSTM builder (reusable)
│   └── methodology.md               ← Methods write-up for the beta experiment
├── data/
│   ├── dataSVO.npz                  ← SrVO₃ preprocessed dataset (~5 MB)
│   ├── dataCVO.npz                  ← CrVO₃ preprocessed dataset (~1 MB)
│   └── README.md                    ← Data schema and how to regenerate
├── models/
│   └── trained_model_svo.keras      ← Pretrained SVO model (used in notebook 03)
├── requirements.txt
└── .gitignore
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebooks in order

Open JupyterLab and run the four notebooks in sequence:

```bash
jupyter lab
```

---

## Step-by-step guide

### Step 1 — Preprocessing (`01_preprocess_svo.ipynb`)

**Skip this step** if you are using the provided `data/dataSVO.npz`.

If you have raw TRIQS/DFTTools HDF5 output files, this notebook shows how to:
- Open each `.h5` file and extract Im/Re Σ(iω) and G₀(iω)
- Iterate over 25 DMFT self-consistency iterations and 6 t₂g spin-orbital channels
- Mask the full Matsubara frequency grid (−161 to +161 eV) to the positive window
  (0, 10) eV → 64 frequency points
- Shuffle and save as a compressed `.npz`

Place raw files in `sims_SVO/` matching the name pattern `svo_beta40_nwarmup<N>.h5`.

---

### Step 2 — Train the Base SVO Model (`02_train_svo_lstm.ipynb`)

Trains a conditional LSTM sequence-to-sequence autoencoder on SrVO₃ data.

**Train / test split:**  
Every 4th unique nwarmup value is used for training (the "1/4 split"). The model
must generalise to the 3/4 of nwarmup values it has never seen — a strict test of
whether it learned the convergence trend rather than memorised individual profiles.

**Architecture (50 241 parameters):**

```
Im G₀(iω)  →  [Encoder LSTM 64]  →  [RepeatVector]  →  [Decoder LSTM 64]  →  Im Σ(iω)
nwarmup label (normalised scalar, broadcast to all 64 timesteps via RepeatVector)
```

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Latent dim | 64 |
| Sequence length | 64 |
| Epochs | 40 |
| Batch size | 128 |
| Loss | MSE |
| Optimiser | Adam (default LR) |

**Output:** `models/trained_model_svo.keras`

---

### Step 3 — Transfer Learning to CVO (`03_transfer_learning_cvo.ipynb`)

Fine-tunes the pretrained SVO model on CrVO₃ (a different correlated material)
using only **2 nwarmup conditions** out of 6 available.

**Why transfer learning works here:**  
SVO and CVO are both d¹ perovskite oxides with similar Matsubara structure.
The encoder already learned how G₀ and Σ relate in this family; the decoder only
needs to adapt to CVO's different bandwidth and quasiparticle weight.

**Two-phase fine-tuning:**

| Phase | Frozen? | LR | Epochs | Purpose |
|-------|---------|-----|--------|---------|
| 1 | Encoder (4 layers) | default | 30 | Adapt decoder without corrupting encoder |
| 2 | None | 1 × 10⁻⁵ | 40 | Global refinement without catastrophic forgetting |

**Evaluation:** relative RMS residual `rms((ŷ − y) / y)` per nwarmup group.

---

### Variation — Beta Study (`beta_variation/04_beta_variation_svo.ipynb`)

Explores generalisation across inverse temperature β (8 values from 20 to 90 eV⁻¹)
rather than convergence steps. Uses an upgraded **FiLM-conditioned** architecture.

**FiLM conditioning** (Feature-wise Linear Modulation):  
Instead of naively concatenating the β label to the input sequence, two small Dense
layers produce per-unit scale γ and shift β applied directly to the encoder hidden
state. This gives the conditioning signal a dedicated modulation pathway:

```
state_h_mod = γ(β) ⊙ state_h  +  β_shift(β)
```

**Architecture (3 377 parameters — 15× smaller than base model):**

```
Im G₀(iω)  →  [Encoder LSTM 16]  →  FiLM(β)  →  [Decoder LSTM 16]  →  Im Σ(iω)
β label  →  Dense(16) → γ
          →  Dense(16) → β_shift
```

**Train / val / test split by β:**

| Split | β values |
|-------|----------|
| Train | 20, 40, 60, 80, 90 |
| Val   | 50 (held out during training) |
| Test  | 30, 70 (never seen) |

Training uses early stopping (patience 40) and ReduceLROnPlateau, so it runs up to
1 000 epochs but typically converges well before that.

A **linear interpolation baseline** is included for comparison: given two neighbouring
training β values, linearly interpolate the profiles to estimate the test β.

The reusable `model.py` module builds both the FiLM and concatenation variants
— useful if you want to adapt this architecture to other materials or conditioning
variables.

---

## Physics background

### DMFT and the self-energy

Dynamical Mean-Field Theory maps a lattice of correlated electrons onto a single
impurity problem solved self-consistently. The key output is the self-energy
Σ(iω), which encodes correlation effects (Mott insulation, quasiparticle
renormalisation, scattering rate) on the imaginary frequency axis.

### CT-HYB Monte Carlo warmup

The impurity is solved by continuous-time hybridisation expansion (CT-HYB) Monte
Carlo. Before accumulating statistics, a "warmup" phase thermalises the Markov chain.
Fewer warmup steps → faster but noisier self-energies. The ML model learns to predict
the well-converged limit from under-converged inputs.

### Matsubara frequency grid

Fermionic Matsubara frequencies are ωₙ = (2n+1)π/β. At β = 40 eV⁻¹ the spacing
is ≈ 0.16 eV. We work in the positive window (0, 10) eV where the self-energy has
most of its physically relevant structure.

---

## Citation

If you use this code or methodology, please cite the associated paper (link TBD).

---

## License

MIT

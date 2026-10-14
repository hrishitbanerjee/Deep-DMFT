# Data

## Included files

| File | Size | Description |
|------|------|-------------|
| `dataSVO.npz` | ~5 MB | Preprocessed SrVO₃ (SVO) DMFT dataset |
| `dataCVO.npz` | ~1 MB | Preprocessed CrVO₃ (CVO) DMFT dataset |

## Array keys (all `.npz` files share the same schema)

| Key | Shape | Description |
|-----|-------|-------------|
| `nwarupsteps` | `(N,)` | CT-HYB warmup steps — convergence proxy label |
| `imgpart_g0` | `(N, 64)` | Im G₀(iω) — non-interacting Green's function |
| `imgpart_sigma` | `(N, 64)` | Im Σ(iω) — self-energy (prediction target) |
| `realpart_g0` | `(N, 64)` | Re G₀(iω) |
| `realpart_sigma` | `(N, 64)` | Re Σ(iω) |

Each row is one sample: a single DMFT iteration × spin-orbital channel pair from one simulation file.

The 64 frequency points correspond to the positive Matsubara window (0, 10) eV sampled from the full grid `np.linspace(-161, 161, 2050)`.

## Raw simulation files (not included)

The raw HDF5 files produced by TRIQS/DFTTools are ~22 MB each (720 MB total for SVO).
They are not tracked in this repository.

To regenerate `dataSVO.npz` from your own raw files, place your `.h5` files under
`sims_SVO/` and run `01_preprocess_svo.ipynb`. The expected filename format is:

```
sims_SVO/svo_beta40_nwarmup<N>.h5
```

Each file must have the HDF5 structure:
```
dmft_output/
  Sigma_iw/        ← converged self-energy
  Sigma-{1..25}/   ← per-iteration self-energy
  G0-{1..25}/      ← per-iteration non-interacting Green's function
```

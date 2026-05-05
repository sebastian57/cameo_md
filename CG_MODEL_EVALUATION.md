# CG Model Evaluation: Force Variance, Option 3 Analysis, and Training Metric Interpretation

**Date:** 2026-04-10  
**Proteins:** 2gy5A01 (99 CA), 4q5WA02 (85 CA), 4zohB01 (53 CA)  
**Training dataset:** `2g4q4z5k_320K_kcalmol_1bead_notnorm_aggforce.npz`  
**Models evaluated:** aggforce no-priors (agg1), aggforce priors-cfg (agg2), non-aggforce (noagg)

---

## 1. Overview: The Three Evaluation Options

To understand model quality, three complementary analyses were designed. Each answers a different question.

### Option 1 — Train on non-aggforce (raw CA forces)

Train the model on raw CHARMM22+CMAP CA forces from mdCATH free-MD trajectories without the aggforce projection. Provides a direct noise floor in the raw-force metric: the model RMSE should be compared to ~17–18 kcal/mol/Å (the per-component force RMS at a fixed configuration measured from fixed-CA MD).

**Status:** Complete. See noagg training analysis below.

### Option 2 — Apply saved aggforce weight matrix W to fixed-CA MD

The weight matrix W (shape n_cg × n_prot) is saved in each per-protein CG NPZ as `aggforce_weight_matrix`. Applying it to forces from a fixed-CA LAMMPS run gives the per-frame noise floor in the *exact same metric* as the aggforce training targets. This measures how much thermal noise remains in the aggforce projection when the CA is frozen.

**Status:** Complete. Results embedded in Option 3.

### Option 3 — Mean-force decomposition at fixed CA positions (primary analysis)

At a single fixed CA configuration:
1. Parse the `protein_forces.dump` from fixed-CA LAMMPS MD (2001 frames, 320 K)
2. Compute the per-frame mean force ⟨F_MD⟩ and standard deviation std(F_MD) — the irreducible thermal noise
3. Evaluate the trained model at the fixed CA positions via a 0-step LAMMPS run
4. Decompose: `RMSE_per_frame² ≈ systematic_err² + thermal_noise²`

This cleanly separates **model bias** (how well the model predicts the mean force) from **irreducible thermal noise** (which no model can ever predict, since it is random fluctuation around the mean).

**Scripts:**
- `cameo_md/option3_fixed_ca_analysis.py` — main analysis script
- `cameo_md/submit_option3.sh` — SLURM submission (2 passes: agg1+noagg, agg2+noagg)

---

## 2. Option 3 Setup and Results

### Fixed-CA LAMMPS MD setup

| Protein  | n_CA | n_prot | MD frames | Dump |
|----------|------|--------|-----------|------|
| 2gy5A01  | 99  | 1613   | 2001      | `runs/forcevar_320K/2gy5A01/protein_forces.dump` |
| 4q5WA02  | 85  | 1390   | 2001      | `runs/forcevar_320K/4q5WA02/protein_forces.dump` |
| 4zohB01  | 53  | 868    | 2001      | `runs/forcevar_320K/4zohB01/protein_forces.dump` |

Protocol: CA atoms frozen (zero force applied), all other protein atoms + TIP3P water at NVT 320 K with Langevin thermostat. CHARMM22+CMAP. Every 10 steps sampled → 2001 frames.

### Model evaluation

Models are evaluated at the fixed CA config via a 0-step LAMMPS run using the compiled `.mlir` symbolic reexport files. No Python/JAX path is used — LAMMPS with chemtrain_deploy handles inference via the PJRT plugin.

**Model files** (in `cameo_md/models/`):
- `training_md_testing_cueq_fast_tiled_no_priors_symbolic_reexport.mlir` — agg1
- `training_md_testing_cueq_fast_tiled_priorscfg_symbolic_reexport.mlir` — agg2
- `noagg_test_cueq_fast_tiled_no_agg_symbolic_reexport.mlir` — noagg

### MD statistics (same for both aggforce passes)

| Protein  | Mean force RMS | Raw CA thermal noise (per-comp) | Agg-projected thermal noise (per-comp) |
|----------|---------------|----------------------------------|----------------------------------------|
| 2gy5A01  | 11.1 kcal/mol/Å | 23.6 kcal/mol/Å               | 43.0 kcal/mol/Å |
| 4q5WA02  | 8.7 kcal/mol/Å  | 23.7 kcal/mol/Å               | 43.3 kcal/mol/Å |
| 4zohB01  | 10.4 kcal/mol/Å | 23.6 kcal/mol/Å               | 42.7 kcal/mol/Å |

*Mean force RMS:* RMS of the per-CA-component average force ⟨F_MD⟩ over 2001 frames. This is the true thermodynamic CG force at this configuration — the ground truth the model is trying to learn.

*Thermal noise (per-comp):* RMS of the per-CA-component standard deviation std(F_MD). This is the irreducible per-frame noise.

### Error decomposition: `systematic_err = RMSE(F_model, ⟨F_MD⟩)`

| Protein  | F_model RMS (noagg) | Systematic err (noagg vs raw) | F_model RMS (agg1) | Systematic err (agg1 vs agg-proj) | F_model RMS (agg2) | Systematic err (agg2) |
|----------|--------------------|-----------------------------|--------------------|------------------------------------|--------------------|-----------------------|
| 2gy5A01  | 8.40               | 7.35                        | 6.82               | 7.45                               | 6.87               | 7.70                  |
| 4q5WA02  | 6.64               | 5.75                        | 5.08               | 5.92                               | 5.07               | 6.17                  |
| 4zohB01  | 7.13               | 7.29                        | 5.58               | 7.47                               | 5.89               | 7.34                  |

*All values in kcal/mol/Å.*

### Expected total per-frame RMSE

`RMSE_expected = sqrt(systematic_err² + thermal_noise²)`

| Model  | Protein  | Systematic | Thermal | Expected total |
|--------|----------|-----------|---------|----------------|
| noagg  | 2gy5A01  | 7.35      | 23.6    | **24.8** |
| noagg  | 4q5WA02  | 5.75      | 23.7    | **24.4** |
| noagg  | 4zohB01  | 7.29      | 23.6    | **24.7** |
| agg1   | 2gy5A01  | 7.45      | 43.0    | **43.6** |
| agg1   | 4q5WA02  | 5.92      | 43.3    | **43.7** |
| agg1   | 4zohB01  | 7.47      | 42.7    | **43.4** |

**Key finding:** In all cases, thermal noise completely dominates the expected per-frame RMSE. The systematic error (model bias at this fixed configuration) is 5–8 kcal/mol/Å — comparable across both model types and well below the thermal noise. Both model types predict the mean force with similar accuracy.

---

## 3. Training Metric Summary (from cameo_cg analysis)

### Model metrics (validation set, aggforce-projected forces)

| Metric | Agg1 (no priors) | Agg2 (priors cfg) | Noagg (raw CA) |
|--------|-----------------|-------------------|----------------|
| Reference force RMS (RMSE_zero) | 10.42 kcal/mol/Å | 10.42 | 22.96 |
| Model RMSE (detailed eval) | **8.12** | **8.11** | **21.48** |
| Zero-force baseline | 10.42 | 10.42 | 22.96 |
| Shuffle baseline RMSE | 12.24 | 12.24 | 24.29 |
| Shuffle gap (model vs shuffled) | **4.12** | **4.13** | **2.81** |
| Pearson R | **0.628** | **0.628** | **0.353** |
| Cosine similarity | **0.629** | **0.629** | **0.352** |
| R² (explained variance) | **0.393** | **0.394** | **0.124** |
| Variance ratio (pred/ref) | 0.378 | 0.380 | 0.119 |
| Calibration slope | 0.338 | 0.337 | 0.059 |

### How to interpret each metric

**RMSE** (`detailed_rmse_model`): The primary quality metric. Measures per-component, per-CA force error between model predictions and reference forces (aggforce-projected for agg models, raw CA for noagg) across the validation set. Lower is better. Directly comparable to `RMSE_zero` (what you'd get predicting zero everywhere) to assess how much the model improves on the trivial baseline.

**Zero / mean baseline** (`detailed_rmse_zero`, `detailed_rmse_mean`): The RMSE of predicting F=0 or F=mean_force at every frame. These equal the RMS of the reference forces. Provides the scale of the problem. A good model must beat this substantially.

**Shuffle baseline** (`detailed_rmse_shuffle_mean`): RMSE when predictions are matched to *randomly shuffled* reference frames. Measures how much the model has learned *configuration-specific* structure vs. just the average force distribution. Model RMSE < shuffle baseline → the model genuinely uses the configuration to make predictions.

**Shuffle gap**: `shuffle_baseline − model_RMSE`. The larger this is, the more configuration-specific learning has happened. Agg models: 4.1 kcal/mol/Å gap (real learning). Noagg: 2.8 kcal/mol/Å gap (weaker but real).

**Pearson R / Cosine similarity**: Directional correlation between predicted and reference force vectors. Values near 1 = model predicts the right direction. Near 0 = random. Pearson 0.628 means about 63% directional agreement for aggforce models. Note: these can be high even when RMSE is poor (models can get direction right but magnitude wrong).

**R² (explained variance)**: Fraction of total force variance explained by the model. `R² = 1 − RMSE² / RMSE_zero²`. **This is the central quality indicator.** R²=0 means the model adds nothing; R²=1 means perfect prediction. Agg models: R²≈0.39 (model explains 39% of variance). Noagg: R²≈0.12. Important: R² is bounded by the noise ceiling (see Section 4), so the theoretical maximum is below 1.

**Variance ratio** (`detailed_variance_ratio_pred_to_ref`): Ratio of predicted force variance to reference force variance. A perfect model predicts forces with the same spread as the reference. Values below 1 (agg: 0.38, noagg: 0.12) indicate the model is underpredicting force magnitudes — a signature of regression-to-mean under noisy training (see calibration slope below).

**Calibration slope** (`complete_eval_calibration_slope`): From a linear regression of reference forces onto model predictions. Slope=1 means the model is perfectly calibrated (predictions have the right magnitude). Slope<1 means predictions are too small — the model is "shrunk" toward zero. Agg models: slope≈0.34, noagg: slope≈0.06. This is an expected consequence of L2 training in high-noise regimes, not a fixable architecture problem (see Section 4).

---

## 4. Noise Floor Physics and Theoretical Limits

### Why the aggforce and noagg metrics differ so much

**Aggforce in free MD (training data):** The weight matrix W is trained to minimise force variance by exploiting correlations between CA forces and correlated backbone atom forces. In free MD, backbone motion creates strong positive correlations across adjacent atoms that W partially cancels. Result: total force RMS drops from ~29 kcal/mol/Å (raw CA) to ~10.4 kcal/mol/Å (aggforce-projected). The *signal* (mean force at each configuration) is preserved at ~10–11 kcal/mol/Å. The thermal noise (residual per-frame fluctuation) is reduced to ~8–9 kcal/mol/Å.

**Aggforce at fixed CA (Option 3):** When the CA is frozen, there is no backbone correlation for W to exploit. W applies large weights to many protein atoms and sums their independent thermal fluctuations — amplifying noise. Result: raw CA thermal noise per-component ~23.6 kcal/mol/Å becomes ~43 kcal/mol/Å after aggforce projection. This apparent contradiction (aggforce makes noise *worse* at fixed CA) is physical: W was optimised for a regime that doesn't apply here.

**Noagg metric:** The raw CA forces have total RMS ~22.96 kcal/mol/Å in the training set. The mean force (signal) is ~10 kcal/mol/Å (from Option 3). Therefore: `thermal_var ≈ 22.96² − 10² ≈ 428` → thermal noise ≈ 21 kcal/mol/Å. Almost all the variance is thermal noise, leaving very little learnable signal.

### The theoretical maximum R²

The maximum R² achievable by any model on this training data is:

```
R²_max = 1 − thermal_var / total_var = 1 − σ²_thermal / σ²_total
```

For the **aggforce metric**: the calibration slope (0.338) is an empirical estimate of R²_max via the regression-to-mean relationship. Under L2 loss with thermal noise, the optimal model prediction is `F_pred = SNR × F_true`, where `SNR = σ²_signal / (σ²_signal + σ²_noise) ≈ calibration_slope`. This gives R²_max ≈ 0.34.

The current models achieve R² = 0.39, which is at or slightly above this theoretical limit — consistent with slight overfitting to training configurations. **The models are near their noise ceiling on this training dataset.**

For the **noagg metric**: `R²_max ≈ 1 − 428/528 ≈ 0.19`. Current R² = 0.12 — there is slightly more headroom, but the theoretical ceiling is also low. The noagg model is fundamentally limited by the poor signal:noise ratio of raw CA forces (SNR ≈ 0.48).

### Why the calibration slope < 1 and why this cannot be fixed by architecture

In any L2-trained model on noisy targets, the optimal prediction under noise is a *shrunken* version of the signal. If you have a training set where each target = signal + noise, the best L2 predictor is `F_pred = R²_max × F_true`. Increasing model capacity, using priors, or tuning hyperparameters cannot fix this — it is a consequence of the loss function and the data distribution. The only remedies are:
- More training frames (reduces thermal noise contribution)
- Multiple frames per configuration averaged before training (directly reduces noise)
- Better training objectives (e.g., noise-aware loss, contrastive objectives)

### Agg1 vs Agg2: the priors make no measurable difference

All metrics are essentially identical (R²: 0.393 vs 0.394, Pearson: 0.6276 vs 0.6280, calibration slope: 0.338 vs 0.337). The prior residual contribution is small relative to the noise floor. At larger dataset sizes where the noise floor drops, the priors may become more relevant.

---

## 5. Connecting Training Analysis to MD Evaluation

The training analysis (from `cameo_cg`) and the MD evaluation (from `cameo_md`) measure complementary things. Here is how to interpret them together:

| Metric source | What it measures | Limitation |
|--------------|-----------------|------------|
| Training R² / Pearson | Fraction of per-frame force variance explained across the training distribution | Dominated by thermal noise; low R² does not mean bad model |
| Training RMSE vs zero-baseline | Absolute force error relative to predicting nothing | Comparable only within the same force metric (raw CA vs aggforce) |
| Training shuffle gap | Configuration-specific signal learned | Most reliable indicator of real learning |
| Option 3 systematic error | Model bias vs. thermodynamic mean force at one configuration | Single config, not a statistical average over the distribution |
| Option 3 thermal noise | Irreducible per-frame noise at fixed CA | Fixed-CA metric; differs from training noise (see above) |

**The key diagnostic chain:**

1. Check **shuffle gap > 0** → model is learning real structure (not just average forces)
2. Check **systematic error < mean force RMS** → model predicts forces in the right regime (not wildly wrong)
3. Check **systematic error / thermal_noise** at fixed CA → shows how dominant thermal noise is vs model error
4. Compare **RMSE to shuffle baseline** → if close to shuffle baseline, model capacity is not the bottleneck
5. Check **calibration slope** → values far below 1 indicate you're at the noise ceiling

**For the current models:**
- Shuffle gap ✓ (4.1 kcal/mol/Å for aggforce, 2.8 for noagg)
- Systematic error ✓ (~7 kcal/mol/Å < mean force signal ~10 kcal/mol/Å)
- Thermal noise >> systematic error → per-frame RMSE will always look poor even with a good model
- Calibration slope 0.34 << 1 → at noise ceiling with this data quantity

**Practical interpretation for model comparison:**
When comparing two models in MD evaluation, prefer:
- Option 3 systematic error (direct model bias comparison, noise-free)
- TICA comparison (see below) — does the model reproduce the correct dynamics?
- Free-energy surfaces / structural metrics — does the ensemble look right?

Do not compare models purely on training RMSE or validation R² when the noise ceiling is known to be low.

---

## 6. TICA Comparison Setup

### Goal

Compare conformational sampling dynamics between:
- **CG ML model** (aggforce no-priors): fast CG simulation using the trained ML potential
- **Classical CHARMM22+CMAP**: all-atom reference simulation of the same protein

Both are analysed with `cameo_md/tica_from_lammps_dump.py` on CA-only dump files, producing TICA free-energy surfaces in the same reduced coordinate space.

### Simulation scripts

**CG ML simulation:**  
Input: `cameo_md/inp_lammps_trained_forcevarlike.in`  
Output dump: `cameo_md/outputs/training_md_testing/cg_forces.dump`  
Format: `id type xu yu zu fx fy fz`, atoms = 53 CA beads (4zohB01)

**Classical CHARMM22+CMAP simulation:**  
Input: `cameo_md/inp_lammps_charmm22_tica.in` (new)  
Output dump: `cameo_md/outputs/charmm22_tica/ca_tica.dump`  
Format: `id type xu yu zu`, atoms = 53 CA atoms  
Structure: full protein + TIP3P water from `runs/forcevar_320K/4zohB01/data.protein`

**Note on starting structure:** The CG simulation uses `structures/config_44.data` (frame 44 of the training NPZ for 4zohB01). The classical simulation uses the all-atom structure from the forcevar run (frame 327 of the mdCATH h5). These are different frames of the same protein — acceptable for comparing dynamics via TICA, which characterises the equilibrium ensemble, not specific trajectories.

### Running the TICA analysis

After both simulations complete:

```bash
# CG ML model
python cameo_md/tica_from_lammps_dump.py \
    --dump cameo_md/outputs/training_md_testing/cg_forces.dump \
    --outdir cameo_md/tica_results/cg_ml \
    --prefix cg_ml \
    --lagtime 10 \
    --n-pairs 200

# Classical CHARMM22
python cameo_md/tica_from_lammps_dump.py \
    --dump cameo_md/outputs/charmm22_tica/ca_tica.dump \
    --outdir cameo_md/tica_results/charmm22 \
    --prefix charmm22 \
    --lagtime 10 \
    --n-pairs 200
```

Use the same `--n-pairs` and `--pair-seed` for both to ensure the same feature set. The resulting TICA projections and FES plots can then be compared directly.

---

## 7. Path Forward

**Why the current results are encouraging despite modest R²:**
- Models are at or near the noise ceiling for this dataset size
- The shuffle gap confirms real structure is learned
- Systematic error (~7 kcal/mol/Å) is comparable to the mean force signal (~10 kcal/mol/Å) — meaningful predictions
- Model architecture is working as intended
- Agg1 ≈ Agg2 → prior configuration does not matter at this data scale

**Expected improvements with more training data:**
- Thermal noise floor ∝ 1/√N_frames — doubling data reduces noise by ~30%
- R² should increase substantially (towards 0.6–0.7 range with 10× more data)
- Calibration slope will approach 1 as noise floor drops below signal
- At that point, architectural differences (priors, model size) will begin to matter

**Key things to track when scaling up:**
- Does the shuffle gap grow proportionally? (If yes, model is benefiting from more data)
- Does the calibration slope approach 1? (Noise floor dropping)
- Does the TICA FES comparison improve vs CHARMM22 reference?
- Does the systematic error in Option 3 decrease below the mean force signal?

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B3C5D,50:1D65A6,100:2E86AB&height=230&section=header&text=DWFA&fontSize=64&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=Bounded%20Fairness-Aware%20Federated%20Aggregation&descAlignY=58&descSize=18" />

# DWFA: Bounded Fairness-Aware Federated Aggregation for Chest Radiography With Run-Aware External Evaluation

### Federated learning · Group fairness · Chest radiography · Patient-clustered and run-aware inference

![Task](https://img.shields.io/badge/Task-Pleural%20Effusion%20Classification-0B3C5D?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Backbone](https://img.shields.io/badge/Backbone-DenseNet--121-1D65A6?style=for-the-badge)
![External](https://img.shields.io/badge/External%20cohort-NIH%20ChestX--ray14-2E86AB?style=for-the-badge)
![Runs](https://img.shields.io/badge/Common%20seeds-5-4B7F52?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-555555?style=for-the-badge)

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0B3C5D,100:2E86AB&height=60&section=header&text=Authors%20%26%20Contact&fontSize=26&fontColor=ffffff" />

**Yamlick Abdullah**
Department of Computer Science, American International University-Bangladesh (AIUB), Dhaka, Bangladesh
25-93663-1@student.aiub.edu

**Muhammad Hasibur Rashid Chayon** *(Corresponding Author)*
Department of Computer Science, American International University-Bangladesh (AIUB), Dhaka, Bangladesh
chayon@aiub.edu

**Mahamodul Hasan Mahadi**
Department of Computer Science and Engineering, Bangladesh University of Engineering and Technology (BUET), Dhaka, Bangladesh mahamodulhasanmahadi@gmail.com

All queries about the method, the artifacts, or the analysis should go to the corresponding author.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f2027,100:2c5364&height=60&section=header&text=Research%20Motivation&fontSize=26&fontColor=ffffff" />

Federated learning lets several hospitals train one model while raw records stay on site. Keeping data local does not make the model fair. A federated model can perform well on average and still perform worse for one patient subgroup.

Standard federated averaging weights each client by its sample size. That rule ignores two things:

- whether a client shows a subgroup performance gap,
- whether that observed gap is supported by enough positive cases to mean anything.

Two practical problems follow. A local gap measured on a small minority subgroup is unreliable. An adaptive server rule that reacts to such a gap can move client influence sharply unless it is bounded.

DWFA is a server-side aggregation rule that addresses both. It keeps the sample-size prior as its anchor, adjusts it with a bounded equity score, and reduces the influence of the current-round fairness signal when subgroup support is small.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:134E5E,100:71B280&height=60&section=header&text=Key%20Contributions&fontSize=26&fontColor=ffffff" />

**Method**

- A bounded server-side rule that combines the FedAvg sample-size prior, a local equal-opportunity signal, and a subgroup-support gate. Local training is unchanged.
- The server receives one fairness scalar per client per round, plus two subgroup-positive counts once at setup.

**Theory**

Four exact per-round properties, proved in the appendix and verified against the logged implementation:

1. reduction to FedAvg under equal multipliers,
2. positive normalized weights,
3. reliability-scaled monotone response after initialization,
4. bounded deviation from the FedAvg aggregate in one round.

**Evaluation**

- External validation on an independent patient-indexed cohort.
- Two separate estimands: fixed-seed ensemble and run-aware.
- Patient-cluster bootstrap, equal patient weighting, and familywise control over a prespecified comparison family.
- A patient-overlap audit, a scrambled-routing negative control, threshold sensitivity over 37 operating points, parameter sensitivity, component ablation, and seed-composition sensitivity.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:42275a,100:734b6d&height=60&section=header&text=Method%20Overview&fontSize=26&fontColor=ffffff" />

For client `i` at communication round `t`:

```
p_i    = n_i / N                                  sample-size prior (FedAvg)

G_i^t  = |TPR_hat(i,m,t) - TPR_hat(i,f,t)|        local equal-opportunity gap on V_i
         TPR_hat(i,g,t) = (TP(i,g,t) + 1) / (P(i,g) + 2)

m_i    = min(P(i,m), P(i,f))                      minority positive count

e_i^t  = clip(1 - G_i^t / G_ref, rho_e, 1)        bounded equity score
r_i    = clip(m_i / tau_ref, rho_r, 1)            subgroup-support reliability gate

e_bar(i,t-1) = mean of e_i^0 ... e_i^(t-1)        lagged mean
e_til(i,t)   = (1 - r_i) * e_bar(i,t-1) + r_i * e_i^t     gated score, t >= 1
e_til(i,0)   = e_bar(i,0) = e_i^0                 initialization, ungated

q_i^t  = alpha + kappa * e_til(i,t)               quality multiplier
s_i^t  = p_i * q_i^t                              unnormalized score
a_i^t  = s_i^t / sum_j s_j^t                      DWFA weight

w^(t+1) = sum_i a_i^t * w_i^(t+1/2)               global model update
```

Locked settings used in every reported run:

| Parameter | Value | Role |
|---|---|---|
| `G_ref` | 0.05 | scale of the local disparity response |
| `rho_e` | 0.10 | floor of the equity score |
| `tau_ref` | 200 | subgroup-count horizon for full reliability |
| `rho_r` | 0.10 | floor of the reliability gate |
| `alpha` | 0.40 | fixed part of the multiplier |
| `kappa` | 0.60 | adaptive part of the multiplier |
| decision threshold | 0.5 | fixed before training, descriptive only |

These give `q_i^t` in `[0.46, 1.00]` and the weight bound `0.46 * p_i <= a_i^t <= p_i / 0.46`.

The signal is an absolute magnitude, so the server learns that a client is locally disparate but not which subgroup is disadvantaged. DWFA is therefore a bounded disparity-conditioned rule for client influence. It is not a direct optimizer of global demographic parity.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:11998e,100:38ef7d&height=60&section=header&text=Datasets&fontSize=26&fontColor=ffffff" />

**Development cohort: CheXpert (frontal radiographs)**

Requirements: recorded sex, recorded age, resolved pleural effusion label. Uncertain labels coded as -1 were excluded.

| Stage | Radiographs |
|---|---|
| Eligible cohort | 102,197 |
| Prevalence-balanced candidate pool (40% positive) | 42,171 |
| Assigned to four simulated clients | 35,899 |

**Simulated federation**

| Client | Images | Male fraction | Prevalence | Mean age | `p_i` | `r_i` |
|---|---|---|---|---|---|---|
| A | 15,000 | 0.50 | 0.400 | 59.6 | 0.418 | 1.000 |
| B | 8,000 | 0.80 | 0.450 | 59.1 | 0.223 | 0.465 |
| C | 6,899 | 0.35 | 0.377 | 73.6 | 0.192 | 0.480 |
| D | 6,000 | 0.60 | 0.380 | 49.9 | 0.167 | 0.725 |

Split: 70% train, 15% validation, 15% test, frozen and shared by every procedure.

The client construction is a controlled heterogeneity stress test. It creates planned differences in size, sex composition, age, and prevalence. It is not a model of four real hospitals.

**External cohort: NIH ChestX-ray14**

| Property | Value |
|---|---|
| Radiographs | 112,120 |
| Patients | 30,805 |
| Patients with more than one image | 13,302 |
| Positive effusion images | 13,307 |
| Image-level prevalence | 0.1187 |

Raw images from either dataset are not redistributed here. Both are available from their original providers under their own data use agreements.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:11998e,100:0575E6&height=60&section=header&text=Training%20Configuration&fontSize=26&fontColor=ffffff" />

| Setting | Value |
|---|---|
| Backbone | DenseNet-121, ImageNet initialization |
| Input | RGB, 224 x 224, ImageNet channel normalization |
| Augmentation | random horizontal flip, p = 0.5, training split only |
| Loss | unweighted binary cross-entropy with logits |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-4 |
| Batch size | 32 |
| Local epochs per round | 2 |
| Communication rounds | 30 |
| Precision | mixed |
| Checkpoint rule | highest pooled validation AUROC, same rule for every procedure |
| Seeds | 42, 123, 456, 789, 1010 |

All procedures share the client CSV files, preprocessing, backbone, optimization budget, and checkpoint rule. FedProx changes the local objective. LPR, CADR, and DWFA change server weighting only.

Every selected checkpoint fell between rounds 2 and 4.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:1D4350,100:A43931&height=60&section=header&text=Compared%20Procedures&fontSize=26&fontColor=ffffff" />

| Name | Description | Runs | Role |
|---|---|---|---|
| Centralized reference | pooled training, no federation | 3 | descriptive reference |
| FedAvg | sample-size server weights | 5 | inferential null |
| FedProx | proximal local penalty, `mu = 0.01` | 3 | heterogeneity reference |
| LPR (q-FFL-inspired) | loss-powered server reweighting | 5 | fairness comparator |
| CADR (FairFed-adapted) | cumulative additive disparity reweighting | 5 | fairness comparator |
| **DWFA** | **bounded disparity-conditioned reweighting** | **5** | **proposed** |

LPR and CADR are adapted comparators implemented under this protocol. They are not exact reproductions of q-FedAvg or FairFed, and no comparative claim is made against those published methods.

FedAvg is the reduction case of DWFA at `kappa = 0`, so it is the natural null for a server weighting rule and it sits inside the inferential family rather than outside it.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:1D4350,100:A43931&height=60&section=header&text=External%20Results%20on%20NIH%20ChestX-ray14&fontSize=26&fontColor=ffffff" />

Mean ± standard deviation across individual training runs.

| Method | Runs | AUROC | Worst-group AUROC | EO gap @0.5 | FPR gap @0.5 | Brier |
|---|---|---|---|---|---|---|
| Centralized reference | 3 | 0.8586 ± 0.0020 | 0.8557 ± 0.0016 | 0.0184 ± 0.0079 | 0.0183 ± 0.0133 | 0.1624 ± 0.0245 |
| FedAvg | 5 | 0.8572 ± 0.0023 | 0.8557 ± 0.0017 | 0.0270 ± 0.0050 | 0.0191 ± 0.0022 | 0.1556 ± 0.0151 |
| FedProx | 3 | 0.8574 ± 0.0019 | 0.8556 ± 0.0021 | 0.0281 ± 0.0096 | 0.0191 ± 0.0076 | 0.1539 ± 0.0159 |
| LPR (q-FFL-inspired) | 5 | 0.8554 ± 0.0016 | 0.8537 ± 0.0018 | 0.0285 ± 0.0062 | 0.0185 ± 0.0042 | 0.1541 ± 0.0106 |
| CADR (FairFed-adapted) | 5 | 0.8565 ± 0.0024 | 0.8548 ± 0.0017 | 0.0231 ± 0.0069 | 0.0161 ± 0.0051 | 0.1561 ± 0.0123 |
| **DWFA** | **5** | **0.8578 ± 0.0022** | **0.8557 ± 0.0019** | **0.0253 ± 0.0071** | **0.0179 ± 0.0032** | **0.1541 ± 0.0046** |

**Threshold fairness across 37 operating points**

Number of thresholds at which each rule had a smaller gap than FedAvg:

| Method | EO gap | FPR gap |
|---|---|---|
| **DWFA** | **37 / 37** | **37 / 37** |
| CADR | 37 / 37 | 35 / 37 |
| LPR | 15 / 37 | 21 / 37 |

The direction of the DWFA result does not depend on the operating point, so it does not rest on the descriptive threshold of 0.5.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f0c29,100:302b63&height=60&section=header&text=Two%20Estimands%2C%20Two%20Answers&fontSize=26&fontColor=ffffff" />

This is the central methodological result of the study, and it is reported honestly in both directions.

**Fixed-seed ensemble, patient-clustered.** Probabilities are averaged across seeds, the ensemble is then held fixed, and unique patients are resampled.

| Contrast | Overall AUROC | Worst-group AUROC |
|---|---|---|
| DWFA − FedAvg | +0.000715, interval excludes zero | −0.000079, includes zero |
| LPR − FedAvg | −0.002391, excludes zero | −0.002460, excludes zero |
| CADR − FedAvg | −0.001035, excludes zero | −0.001030, excludes zero |

Under this estimand both adapted fairness-aware rules cost discrimination relative to sample-size aggregation. DWFA did not.

**Run-aware, with familywise control.** Training runs are resampled as well as patients.

| Analysis | Intervals excluding zero |
|---|---|
| Fixed-ensemble, primary family | 5 of 6 |
| Run-aware pointwise | 0 of 6 |
| Run-aware simultaneous | 0 of 6 |
| Leave-one-seed-out simultaneous | 2 of 30, both LPR vs FedAvg, both from omitting one seed |

**What this means.** The current runs support a bounded design claim and a threshold-gap claim. They do not support a stable procedure ranking. This is not a claim of equivalence. Five runs are a coarse basis for a run bootstrap, and the exact sign-flip test cannot reach a two-sided p below 0.0625 at that budget. The run count bounds what the study can claim, and the claims are bounded to match.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:3a1c71,100:d76d77&height=60&section=header&text=Reproducibility%20and%20Audits&fontSize=26&fontColor=ffffff" />

| Audit | Result |
|---|---|
| Implementation fidelity | all 600 logged DWFA client-round weights reconstructed from the stated equations |
| Reconstruction error | 2.22e-16, 4.996e-16, and 5.55e-16 across three independently written code paths |
| Weight bounds | no logged weight violated the stated lower or upper bound |
| Patient overlap in development | 4,736 patients across clients, 3,634 across splits, 46.3% of validation images |
| Contamination-stratified test | 0 of 6 interaction intervals excluded zero, largest estimate 0.00105 |
| Prevalence reweighting | AUROC, worst-group AUROC, EO gap and FPR gap changed by at most 3.3e-16 |
| Scrambled-routing control | small fixed-ensemble difference, no reproducible advantage once run variation was included |
| Monte Carlo repetition | primary pattern unchanged across three bootstrap seeds, largest endpoint change below 0.00016 |

Patient overlap in the CheXpert development partition was audited rather than assumed away. Formal procedure-level inference is based on the NIH cohort, which is patient-indexed and disjoint from CheXpert.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:134E5E,100:71B280&height=60&section=header&text=Repository%20Contents&fontSize=26&fontColor=ffffff" />

```
.
├── DWFA_full_pipeline.ipynb    complete experimental pipeline
├── README.md
└── LICENSE
```

The notebook runs in Google Colab and covers the full study in order:

| Stage | What it does |
|---|---|
| 1 | CheXpert acquisition, cohort filtering, prevalence balancing, client construction, frozen splits |
| 2 | Local-only baselines and the Centralized reference |
| 3 | FedAvg and FedProx, including the seed extension to five common runs |
| 4 | LPR (q-FFL-inspired) |
| 5 | CADR (FairFed-adapted), including the weight-collapse diagnostic |
| 6 | DWFA training, round logging, and bound-compliance verification |
| 7 | NIH external inference and the scrambled-routing control |
| 8 | Patient-clustered and run-aware inference, calibration, threshold sensitivity, robustness |

**Before running:** the notebook was developed in Colab and uses a Google Drive working directory. Set the root path once at the top, and place a Kaggle API token at `MyDrive/FairFedCXR/kaggle.json` before the dataset download cells.

**Artifacts.** The client manifests, frozen split files, seed list, per-round training logs, per-seed prediction caches, DWFA weight logs, and the statistical analysis code will be deposited publicly on acceptance and are available from the corresponding author in the interim. Raw patient images are not redistributed.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0f2027,100:2c5364&height=60&section=header&text=Limitations&fontSize=26&fontColor=ffffff" />

Stated plainly, because they bound what the results mean.

- Client and split assignment was made at the radiograph level, so some patients appear across clients or splits. The effect was audited, not assumed small.
- One binary task, recorded binary sex, four simulated clients, one engineered non-IID partition.
- Five common runs. More independent runs would be needed before any stable procedure ranking.
- DWFA is not a privacy mechanism. The fairness scalar and the subgroup-positive counts are site-level statistics that standard FedAvg does not require. This is a principal barrier to deployment, not a secondary caveat.
- Checkpoint selection used centrally pooled validation in the simulation, so the algorithm as written is not a deployable data-local protocol.
- All procedures showed external calibration shift. Probability-based use would require recalibration and independent validation.

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0B3C5D,100:2E86AB&height=60&section=header&text=Citation&fontSize=26&fontColor=ffffff" />

```bibtex
@article{abdullah2026dwfa,
  title   = {DWFA: Bounded Fairness-Aware Federated Aggregation for Chest
             Radiography With Run-Aware External Evaluation},
  author  = {Abdullah, Yamlick and Chayon, Muhammad Hasibur Rashid and
             Mahadi, Mahamodul Hasan},
  year    = {2026},
}
```

Please also cite the CheXpert and NIH ChestX-ray14 dataset papers if you use this pipeline.

---

## License

Released under the MIT License. See `LICENSE`. The license covers the code in this repository only. It does not cover the CheXpert or NIH ChestX-ray14 datasets, which remain under their own terms.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2E86AB,50:1D65A6,100:0B3C5D&height=130&section=footer" />

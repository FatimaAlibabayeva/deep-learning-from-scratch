# Strict rubric score

| Part | Points | Score | Notes |
|---|---|---|---|
| A1 Forward pass by hand | 4 | 4 | Correct, fully shown numerically. |
| A2 Backprop over the graph | 6 | 5.5 | Full op-level graph + rigorous softmax-Jacobian-to-`p-y` derivation; -0.5 for omitting the `W`/`b` leaf-tensor edges from the diagram itself (they're named in the caption instead, purely a space/legibility trade-off). |
| A3 Activation derivatives & dying ReLU | 4 | 4 | All three derivatives shown from first principles; dying-ReLU explained via pre-activation + zero gradient, tied to an actual test. |
| A4 NLL / log-sum-exp | 3 | 3 | Correct likelihood argument and overflow derivation. |
| A5 Momentum & Adam | 5 | 5 | Correct, matches the actual optimizer code. |
| A6 Weight decay & dropout | 3 | 3 | Correct, matches code. |
| **Part A total** | **25** | **24.5** | |
| B.1 Implementation summary | (of 60) | 9/10 | Accurate, references real code; -1 for length (slightly over the "half a page" guidance). |
| B.3 Gradient check | 10 | 10 | All checks ≤ 1e-4 (several ~1e-11); loss gradient additionally verified in closed form. |
| B.4 Training curves & accuracy | 10 | 10 | Real figure, final test accuracy 0.9778 ≥ 0.95 bar, caption matches the plot. |
| B.4/B.6 Optimizer comparison | 10 | 10 | Real figure, learning rates in legend, caption states only what the plot shows. |
| B.5/B.6 Training behaviour & regularization | 10 | 10 | Real log lines, correct hyperparameters, honest account of the train/val gap. |
| B.8 Analysis | 10 | 9/10 | Thoughtful, references the actual stability bug found during review; -1 for leaning on that bug rather than a training-time gradient bug (none was found, which is itself a fair outcome, but the prompt's framing expects a debugging story from training). |
| **Part B total** | **60** | **58** | |
| C1 Trace a gradient | 4 | 4 | Correct, names the exact closures. |
| C2 Contrast optimizers | 4 | 4 | Correct, tied to the actual figure. |
| C3 Dropout train/eval | 4 | 4 | Correct mechanism, not just symptom. |
| C4 Adam bias-correction line | 3 | 3 | Quotes the exact two lines from the submitted file. |
| **Part C total** | **15** | **15** | |
| Bonus 1 (RMSProp + schedule) | +3 | +3 | Implemented, tested, plotted. |
| Bonus 2 (BatchNorm) | +4 | +4 | Implemented, gradient-checked, two-panel before/after figure. |
| Bonus 3 (unbiasedness proof) | +3 | +3 | Full derivation, matches Adam's actual bias correction. |
| **Bonus total** | +10 | +10 | |
| **Raw total** | | 24.5 + 58 + 15 + 10 = **107.5** | |
| **Final (capped)** | | **100 / 100** | Per the stated bonus policy: "total capped at 100." |

**Caveats on this score:** this is my own strict read against the rubric headings visible in the
provided template, not an official grade — the course's actual point allocation within Part B
(B.1–B.8) isn't fully specified anywhere I have access to, so the B.1/B.8 sub-splits above are a
reasonable estimate, not a certainty. Everything scored here was independently verified by
re-running the code, not taken on faith from the report text.

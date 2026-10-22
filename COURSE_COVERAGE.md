# Course coverage — every lesson, where it lives, what it shows

Course: *Fundamentals of Machine Learning: From Data to Deployed Models* — 6 modules, 25 lessons.

Each row names the lesson's main claims, the code that implements them, and the
result measured on this code. Reproduce any module's results with
`python -m modelforge module N`; the HTML report (`python -m modelforge lab`)
shows the same results in sections 8–13, after the core lab in sections 1–7.

## Module 1 — What Machine Learning Really Is

| Lesson | Claims from the lesson | Where | Result on this code |
|---|---|---|---|
| ML vs. AI: Drawing the Real Boundaries | ML infers patterns from data and gives probabilistic outputs; rule-based systems are transparent but brittle | `experiments.rule_vs_learned`, `predict_proba` in every model | hand-written rule 81.6% on moons vs logistic regression 84.2%, tree 90.8%, network 97.4% |
| The Three Learning Paradigms | supervised (labels), unsupervised (structure), reinforcement (rewards, delayed) | `paradigms.py` (k-means, Q-learning); every other model is supervised | k-means elbow at k=3 with 100% purity; Q-learning goes from 100 steps to the shortest 8-step path from a goal-only reward |
| The ML Workflow End to End | framing → data → train → evaluate on unseen data → deploy → monitor; quality beats quantity | `pipeline.py`, `registry.py`, `experiments.label_noise`, `experiments.learning_curve` | 15% random label noise: 97.2% → 87.0%, 4× data recovers to 93.5%; 15% biased noise: 82.5%, 4× data only 83.5% |
| Where ML Actually Gets Used | vision, language, recommendation, forecasting; when *not* to use ML; No Free Lunch | conv / attention / RNN in `architectures.py`, k-means segments | logistic regression stuck at ~86% from 25 to 800 rows while k-NN climbs to ~98% — more data only helps a model that can use it |

## Module 2 — The Math Behind the Magic

| Lesson | Claims from the lesson | Where | Result on this code |
|---|---|---|---|
| Vectors, Matrices, and What Models Compute | data and weights are vectors; the prediction is a dot product | `linalg.py`, `regression.py` | the lesson's house: 230,500 and 180,000 exactly; regression recovers $200.3 per sq ft (true 200) |
| Loss Functions: Measuring Wrongness | MSE for numbers, cross-entropy for probabilities; squaring punishes big errors | `regression.mse`, `metrics.cross_entropy` | lesson example 380,250,000; confident wrong answer: cross-entropy 4.61 vs squared error 0.98 |
| Gradient Descent: How Models Learn | the update rule; learning rate too small / too large; batch vs SGD vs mini-batch | `regression.LinearRegression(batch_size=…)`, `experiments.learning_rate_sweep`, `gd_flavours` | lr 0.001 still at 0.67 after 100 epochs, lr 0.1 at 0.0005, lr 1.2 diverges; SGD: 48.2% of steps go uphill, batch and mini-batch 0% |
| Bias, Variance, and Overfitting | train-vs-test gap; validation set; L2 = λ × sum of squared weights | `pipeline.depth_sweep`, `experiments.ridge_sweep`, `early_stopping`, `depth_by_validation`, `MLP.fit(patience=…)` | ridge: test MSE 0.291 (λ=0) → 0.025 (λ=0.01) → 0.432 (λ=10); validation loss bottoms at epoch 137 then doubles |
| The Limits: No Free Lunch and VC Dimension | no universal winner; a 2-D line shatters 3 points, not 4; inductive bias | `pipeline.compare` (3 datasets), `experiments.vc_shattering` | 8/8 labelings of a triangle, 14/16 of a square (the two XOR labelings fail) |

## Module 3 — Classification: Teaching Machines to Decide

| Lesson | Claims from the lesson | Where | Result on this code |
|---|---|---|---|
| The Decision Boundary Mindset | a classifier carves feature space; the boundary *is* the learned weights | `viz.boundary_svg`, `experiments.decision_boundary` | logistic regression boundary printed as an equation in its own weights; boundary plots for all 5 models × 3 datasets |
| Logistic Regression and Beyond | sigmoid → probability, threshold; softmax for many classes; softmax(2 classes) = sigmoid | `linear.py`, `linalg.softmax/sigmoid` | sigmoid(2.2) = 0.900; softmax(2.0, 1.0, 0.1) = (0.659, 0.242, 0.099) — the lesson rounds this to (0.63, 0.23, 0.14) |
| Trees, Forests, and Neighbors | Gini or entropy; bagging + random features; k-NN, small vs large k, high dimensions | `tree.py` (`criterion=`), `neighbors.py`, `experiments.curse_of_dimensionality` | entropy 96.1% vs Gini 90.8% on moons; nearest/farthest distance 0.004 in 2-D → 0.913 in 1000-D |
| Evaluating Classifiers Honestly | confusion matrix; accuracy lies on imbalance; precision/recall/F1; threshold trade-off; match the metric to the cost | `metrics.py`, `datasets.make_imbalanced`, `experiments.threshold_tradeoff` | "always no" scores 95.0% with recall 0; thresholds 0.9 → 0.1 move recall 0.47 → 0.93 and precision 1.00 → 0.50; lesson's F1 = 0.320 |

## Module 4 — Neural Networks from the Ground Up

| Lesson | Claims from the lesson | Where | Result on this code |
|---|---|---|---|
| From Neuron to Network | neuron = weighted sum + activation; one sigmoid neuron *is* logistic regression; layers | `neural.MLP`, `deep_experiments.neuron_is_logistic_regression` | no-hidden-layer network and logistic regression: identical predictions on 100% of test points |
| Activation Functions and Nonlinearity | stacked linear layers collapse; sigmoid/tanh saturate; ReLU; dying ReLU; Leaky ReLU, GELU, Swish | `neural.py` (6 activations), `lab.nonlinearity_ablation`, `deep_experiments.dying_relu` | no hidden layer 59.2% vs one hidden layer 100% on circles; dead ReLU net stuck at 50%, Leaky ReLU recovers to 80.3% |
| Backpropagation: Learning Across Layers | chain rule; forward then backward pass; backprop computes gradients, gradient descent applies them | `MLP._backward`, `MLP.gradient_check` | backprop matches numerical gradients to below 1e-7 for all six activations |
| Why Deep Networks Work (and When They Don't) | feature hierarchy; transfer learning and fine-tuning; domain mismatch; you don't always need massive data | `MLP.fine_tune`, `deep_experiments.transfer_learning` | with 8 labels: from scratch 64.5%, fine-tuned 80.8%, mismatched source 72.8%; by 16 labels the advantage is gone |

## Module 5 — Architectures for Vision and Sequences

| Lesson | Claims from the lesson | Where | Result on this code |
|---|---|---|---|
| CNNs: Seeing with Convolutions | kernels and feature maps; pooling; weight sharing; translation invariance; no free rotation invariance | `architectures.conv2d`, `max_pool`, `conv_param_counts`, `rotate90` | 1,728 conv weights vs 9,633,792 dense; response follows a shifted edge, global max unchanged; rotated edge → 0 |
| RNNs and the Vanishing Gradient Problem | hidden state; BPTT; vanishing (<1) and exploding (>1) gradients | `architectures.rnn_forward`, `bptt_gradient`, `tanh_rnn_gradient`, `vanishing_gradient` | factor 0.5 for 20 steps → 9.5e-7, factor 2 → 1,048,576; tanh RNN 1.2e-4 after 50 steps |
| LSTMs and GRUs: Gating Memory | forget/input/output gates; cell state as a gradient conveyor belt; GRU update/reset gates, fewer parameters; gradient clipping | `lstm_step`, `lstm_cell_gradient`, `gru_step`, `count_gate_params`, `clip_by_norm` | forget gate 0.95 keeps 8.5% of the gradient over 50 steps; GRU has 25% fewer parameters; clipping keeps direction |
| Transformers and Self-Attention | query/key/value; parallel; positional encoding; quadratic cost; sparse attention | `self_attention`, `positional_encoding`, `attention_cost` | "dog bit man" vs "man bit dog": identical attention output without positions (difference 0.000), 0.467 with; 10× longer = 100× the scores |

## Module 6 — From Notebook to Production — and the Open Frontier

| Lesson | Claims from the lesson | Where | Result on this code |
|---|---|---|---|
| MLOps Foundations: Versioning and CI/CD | version code, data and weights; data validation; model quality gates; registry with Staging/Production/Archived | `registry.py` (data fingerprint, code version, stages, floor, rollback), `mlops.validate_data`, `.github/workflows/ci.yml` | a column of nulls, out-of-range values and unknown labels are rejected; a 57.9% model is refused by the 0.80 floor; rollback restores the previous champion |
| Deploying Models Safely | containers; canary; shadow; training-serving skew; feature stores | `Dockerfile`, `mlops.canary_rollout`, `shadow_deploy`, `FeatureStore`, `ops_experiments.feature_store_skew` | a hidden serving bug is caught at the 5% canary stage after 31 of 2,000 requests; the re-implemented 30-day feature disagrees on 161 of 200 customers and flips 10 decisions, the feature store on none |
| Monitoring Drift: Why Models Decay | data drift vs concept drift; PSI and KS; alerts, retraining triggers, human review | `registry.psi`, `ks_test`, `drift_report`, `mlops.monitor_decision`, CLI `drift` | data drift: PSI 1.71, KS p 4e-25 → RETRAIN (66.3% → 98.3% after retraining); concept drift: PSI 0.089, accuracy 70% → HUMAN REVIEW |
| The Open Frontier: What ML Still Can't Do | interpretability; adversarial examples; catastrophic forgetting; epistemic uncertainty; scaling limits | `frontier.py`, `mlops.OODGuard`, `deep_experiments.catastrophic_forgetting`, `experiments.learning_curve` | FGSM at ε=0.3: 80.3% vs 89.5% for random noise; >99.99% confidence far outside the data (guard flags it); training on task B drops task A from 88.7% to 50.0% |

## What this project does *not* do

- It does not train full CNNs, LSTMs or Transformers — Module 5 implements and measures
  their defining computations on small inputs. Training the full models needs a GPU and a framework.
- Hallucination and scaling laws in large language models are covered in the study notes,
  not in code; the closest measurable analogue here is the learning curve (more data does
  not help a model that cannot represent the problem).
- The datasets are small and two-dimensional so every lesson is visible; they are not a
  benchmark of real-world performance.

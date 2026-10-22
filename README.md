# ModelForge — machine learning from scratch, from data to deployed model

Five machine-learning classifiers (one of them a neural network with backpropagation),
linear regression, k-means and a Q-learning agent, the core operations of CNNs,
RNNs, LSTMs, GRUs and Transformers, and a small production stack with a model
registry, quality gates, canary and shadow rollout, a feature store and drift
monitoring — all written by hand in **pure Python**. No numpy, no scikit-learn,
no GPU, no data downloads.

Built for the course *Fundamentals of Machine Learning: From Data to Deployed
Models*. The point of writing it from scratch is the course's point: a model is
arithmetic on vectors, and you understand it once you have written that
arithmetic yourself.

```
python demo/run_demo.py          # guided tour, module by module (~20 s)
python -m modelforge module 2    # every experiment for one course module (1-6)
python -m modelforge lab         # everything + an HTML lab report (~2-3 min)
python tests/test_all.py         # 67 tests (~30 s)
```

Every one of the course's 25 lessons maps to code and a measured result —
see **[COURSE_COVERAGE.md](COURSE_COVERAGE.md)**.

---

## What the lab shows

| Question from the course | Experiment | Result on this code |
|---|---|---|
| Does the best model depend on the data? | 5 models × 3 datasets | Yes — logistic regression scores 100% on blobs, 84.2% on moons and 57.9% (last) on circles |
| Why do neural networks need nonlinearity? | the same network with and without a hidden layer, on circles | no hidden layer: 59.2%; one hidden layer of 16 tanh units: 100% |
| What does overfitting look like? | tree depth sweep, k-NN sweep | a deep tree reaches 100% on training data but 94.7% on test; k = 1 memorizes the training set |
| Is my backprop correct? | numerical gradient check | max relative error below 1e-8 for tanh, sigmoid and ReLU (pass threshold 1e-4) |
| Why do deep sigmoid nets / plain RNNs forget? | gradient through 20 layers | sigmoid: 2.6e-13 — ReLU: 1.0 |
| How do you ship a model safely? | champion/challenger gate, quality floor, canary | a worse model is refused; a model with a hidden serving bug is rolled back after 31 of 2,000 canary requests |
| Why do models decay? | drift monitor (PSI + KS) | champion: 98% on fresh data, 66% after the world shifts; PSI and KS raise the alert, and retraining restores 98% |
| Can drift monitoring see everything? | concept drift | no — inputs unchanged (PSI 0.09) while accuracy falls to 70%, so the monitor routes it to a human |
| Does more data always help? | label noise, learning curves | 4× more data with biased labels: 82.5% → 83.5%; a straight line stays at ~86% from 25 to 800 rows |

`python -m modelforge lab` writes `report.html` — leaderboards, a decision-boundary
plot for every model on every dataset, loss curves, overfitting curves, the
gradient check, the architecture demos and the production section (sections 1–7),
then one section per course module with every remaining experiment (sections 8–13).
Every chart is SVG drawn by the project's own plotting code.

---

## How each course module shows up

| Module | Code | What it implements |
|---|---|---|
| **1 · What ML Really Is** — ML vs AI, the three paradigms, the workflow, where ML is used | `datasets.py`, `pipeline.py`, `paradigms.py`, `experiments.py` | generators; stratified split; scaler fit on train only; a rule vs learned models; k-means; Q-learning; label-noise and learning-curve experiments |
| **2 · The Math Behind the Magic** — vectors, loss, gradient descent, bias-variance, No Free Lunch & VC | `linalg.py`, `regression.py`, `metrics.py`, `linear.py`, `experiments.py` | dot products; linear regression on MSE vs exact least squares; learning-rate and batch/mini-batch/SGD experiments; L2 (ridge) sweep; validation set and early stopping; VC shattering |
| **3 · Classification** — decision boundaries, logistic regression, trees/forests/neighbours, honest evaluation | `linear.py`, `neighbors.py`, `tree.py`, `metrics.py`, `viz.py` | softmax regression; k-NN; CART tree on Gini or entropy; random forest; confusion matrix, precision/recall/F1; rare-event data and thresholds; boundary plots |
| **4 · Neural Networks** — neurons, activations, backpropagation, depth & transfer | `neural.py`, `deep_experiments.py` | multilayer perceptron; tanh / sigmoid / ReLU / Leaky ReLU / GELU / Swish; backprop + gradient check; dying ReLU; fine-tuning (transfer learning) |
| **5 · Architectures** — CNNs, RNNs, LSTMs & GRUs, Transformers | `architectures.py`, `deep_experiments.py` | convolution, pooling, weight sharing, rotation; RNN + backprop through time; gradient clipping; LSTM and GRU steps; self-attention, positional encoding, quadratic cost |
| **6 · Notebook to Production & the frontier** — versioning & CI/CD, safe deployment, drift, open problems | `registry.py`, `mlops.py`, `frontier.py`, `ops_experiments.py`, `Dockerfile`, `.github/workflows/ci.yml` | code + data + weight versioning; data validation; quality floor, stages, rollback; canary and shadow rollout; feature store; PSI + KS; concept drift; adversarial examples; out-of-distribution guard; permutation importance; catastrophic forgetting |

---

## Command line

The lab rebuilds its own registry (`lab_registry/`) from scratch on every run, so
its results are identical each time. The `train`, `registry`, `predict` and
`drift` commands share a separate registry (`model_registry/`, or `--registry DIR`
before the command).

```
python -m modelforge lab                         # everything, writes report.html
python -m modelforge compare --dataset circles   # leaderboard + confusion matrix
python -m modelforge sweep                       # overfitting curves
python -m modelforge gradcheck                   # verify backprop
python -m modelforge train --dataset moons --model neural_network   # save + try to promote
python -m modelforge registry                    # versions, * marks the champion
python -m modelforge predict 0.5 -0.3            # serve a prediction from the champion
python -m modelforge drift --dx 0.9 --dy -0.6 --scale 1.3   # PSI + KS; exits 2 if drift is detected
python -m modelforge rollback                    # put the previous champion back into production
python -m modelforge module 6                    # every experiment for one course module
```

The model is also containerized: `docker build -t modelforge .` runs the tests
inside the image, and `docker run --rm modelforge` runs the lab. CI builds the
image on every push.

---

## Design decisions worth naming

- **From scratch on purpose.** Every learning rule is visible in a few lines —
  the softmax-regression gradient, the Gini split, the backprop chain rule — so
  the code doubles as a study guide for the course.
- **Generated datasets.** No downloads, fully reproducible, and the true
  structure is known, so a result can be judged as learned rather than lucky.
- **Proof, not hope.** Backprop is checked against a numerical derivative; the
  linear model's failure on circles is pinned by a test; every model is checked
  to predict identically after a save/reload.
- **Same numbers on every Python version.** Float sums use `math.fsum`, so results
  do not drift between Python 3.9 and 3.12 (whose built-in `sum()` rounds
  differently). Seeds fix every random choice.
- **Fit the scaler on training data only.** Fitting on everything would leak the
  test set into training and inflate the scores.
- **Safe deployment is a gate, not a habit.** A new model must clear an absolute
  quality floor and beat the champion on held-out data by a margin, then a
  canary exposes it to a small slice of traffic before everyone sees it.
- **Monitor the world, not just the model.** The drift check compares live
  inputs to training inputs, because a model that was right at launch can be
  wrong later without a single line of code changing.

## Honest limits

- Pure Python is slow: fine for hundreds of points, not for real datasets. The
  same algorithms in numpy or PyTorch would be orders of magnitude faster.
- Module 5 implements the *defining computation* of CNNs, RNNs, LSTMs, GRUs and
  Transformers on small examples; it does not train full versions of them (that
  needs a GPU and a real framework).
- The Docker image and the canary/shadow/feature-store code run locally and in CI;
  nothing here is deployed to a live server.
- The datasets are two-dimensional toys chosen to make the lessons visible, not
  a benchmark of real-world performance.

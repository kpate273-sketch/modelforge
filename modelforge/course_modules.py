"""The course, module by module, as runnable experiments.

    python -m modelforge module 1      ...      python -m modelforge module 6

run_module(n) runs every experiment for module n and returns the raw results
(the HTML report reuses them); format_module(n, results) prints them under the
course's own lesson titles, so each section of output can be checked against
the lesson it comes from.
"""

from __future__ import annotations

from . import deep_experiments as D
from . import experiments as E
from . import ops_experiments as O

MODULE_TITLES = {
    1: "What Machine Learning Really Is",
    2: "The Math Behind the Magic",
    3: "Classification: Teaching Machines to Decide",
    4: "Neural Networks from the Ground Up",
    5: "Architectures for Vision and Sequences",
    6: "From Notebook to Production — and the Open Frontier",
}


def run_module(n: int) -> dict:
    if n == 1:
        return {"rule": E.rule_vs_learned(), "paradigms": E.paradigms_demo(),
                "label_noise": E.label_noise(), "learning_curve": E.learning_curve()}
    if n == 2:
        return {"examples": [r for r in E.course_examples() if r["module"] == 2],
                "houses": E.house_regression(), "lr": E.learning_rate_sweep(),
                "flavours": E.gd_flavours(), "ridge": E.ridge_sweep(),
                "early": E.early_stopping(), "depth_val": E.depth_by_validation(),
                "vc": E.vc_shattering()}
    if n == 3:
        return {"examples": [r for r in E.course_examples() if r["module"] == 3],
                "boundary": E.decision_boundary(), "gini_entropy": E.gini_vs_entropy(),
                "curse": E.curse_of_dimensionality(), "threshold": E.threshold_tradeoff()}
    if n == 4:
        from .lab import gradient_checks, nonlinearity_ablation
        return {"neuron": D.neuron_is_logistic_regression(), "activations": D.activation_comparison(),
                "dying": D.dying_relu(), "ablation": nonlinearity_ablation(),
                "gradcheck": gradient_checks(), "transfer": D.transfer_learning()}
    if n == 5:
        return {"examples": [r for r in E.course_examples() if r["module"] == 5],
                "cnn": D.cnn_properties(), "rnn": D.rnn_gradients(), "gating": D.gating(),
                "attention": D.attention_order_and_cost()}
    if n == 6:
        return {"versioning": O.versioning(), "validation": O.data_validation(),
                "lifecycle": O.lifecycle(), "canary": O.canary_and_shadow(),
                "feature_store": O.feature_store_skew(), "drift": O.drift_scenarios(),
                "frontier": O.frontier(), "forgetting": D.catastrophic_forgetting()}
    raise ValueError("module must be 1-6")


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _conf(v: float) -> str:
    """Confidence with enough digits to be honest: never print a rounded 100%."""
    return ">99.99%" if v > 0.9999 else f"{v * 100:.2f}%"


def _lesson(lines: list, title: str) -> None:
    lines.append("")
    lines.append(f"--- {title}")


def format_module(n: int, r: dict) -> str:
    L = ["=" * 72, f"MODULE {n} — {MODULE_TITLES[n]}", "=" * 72]
    getattr(_Formatter, f"m{n}")(L, r)
    return "\n".join(L)


class _Formatter:
    @staticmethod
    def m1(L, r):
        _lesson(L, "ML vs. AI: Drawing the Real Boundaries")
        rule = r["rule"]
        L.append(f"  hand-written rule  {rule['rule']}:  {_pct(rule['rule_accuracy'])} on moons (test)")
        for k, v in rule["learned"].items():
            L.append(f"  learned            {k:<22} {_pct(v)}")
        L.append("  -> the rule is readable; the learned models are more accurate but opaque")

        _lesson(L, "The Three Learning Paradigms")
        p = r["paradigms"]
        L.append("  unsupervised — k-means on blobs, labels hidden (used only to score purity):")
        for row in p["kmeans_elbow"]:
            L.append(f"    k={row['k']}  inertia {row['inertia']:>8.1f}  purity {_pct(row['purity'])}")
        L.append("    -> the elbow is at k=3, where the clusters match the 3 hidden groups exactly")
        L.append("  reinforcement — Q-learning in a 5x5 grid, reward only at the goal:")
        L.append(f"    first episode {p['rl_first_episode_steps']} steps (hit the 100-step cap); "
                 f"first 20 episodes avg {p['rl_first20_mean']:.1f}; last 20 avg {p['rl_last20_mean']:.1f} "
                 "(includes 20% random exploration)")
        L.append(f"    learned greedy path: {p['rl_greedy_steps']} steps "
                 f"(shortest possible {p['rl_optimal_steps']}) — reached goal: {p['rl_reached_goal']}")
        for row in p["rl_policy"]:
            L.append("      " + row)
        L.append("  supervised — every other experiment in this project")

        _lesson(L, "The ML Workflow End to End")
        L.append("  this project's pipeline: generate -> stratified split -> scale (fit on train only) -> "
                 "train -> evaluate on held-out data -> registry -> monitor")
        L.append("  quality beats quantity:")
        n = r["label_noise"]
        L.append("  decision tree on moons, clean test set, 15% of training labels wrong:")
        L.append(f"    clean labels, 300 rows             {_pct(n['clean_300'])}")
        L.append(f"    random 15% wrong, 300 rows         {_pct(n['random15_300'])}")
        L.append(f"    random 15% wrong, 1200 rows (4x)   {_pct(n['random15_1200'])}   <- more data partly helps")
        L.append(f"    biased 15% wrong, 300 rows         {_pct(n['biased_300'])}")
        L.append(f"    biased 15% wrong, 1200 rows (4x)   {_pct(n['biased_1200'])}   <- more data barely helps")
        L.append("  -> systematic label errors are learned faithfully; fixing the labels beats adding rows")

        L.append("  more data is not automatically better (test accuracy vs training rows):")
        L.append(f"    {'train rows':>10} {'logistic reg':>13} {'k-NN':>8}")
        for row in r["learning_curve"]:
            L.append(f"    {row['n']:>10} {_pct(row['logistic_regression']):>13} {_pct(row['knn']):>8}")
        L.append("  -> 32x more data does nothing for a straight line on curved data; k-NN gains, then plateaus")

        _lesson(L, "Where ML Actually Gets Used")
        L.append("  the core computation of each application area, as implemented here:")
        L.append("    computer vision      convolution + pooling           python -m modelforge module 5")
        L.append("    language             self-attention + positions      python -m modelforge module 5")
        L.append("    recommendation       k-means customer-style segments (above)")
        L.append("    forecasting          RNN / LSTM / GRU steps          python -m modelforge module 5")
        L.append("  when not to use ML: the rule above is the honest choice whenever a clear rule "
                 "already solves the problem (e.g. a fixed sales-tax rate)")

    @staticmethod
    def m2(L, r):
        _lesson(L, "Vectors, Matrices, and What Models Compute  /  Loss Functions: Measuring Wrongness")
        for e in r["examples"]:
            L.append(f"  lesson: {e['claim']}")
            L.append(f"    code: {e['computed']}")
        h = r["houses"]
        L.append(f"  house prices: linear regression on MSE, gradient descent, {h['n_train']} train / {h['n_test']} test")
        L.append(f"    {'feature':<12}{'true':>10}{'learned':>12}{'exact LSQ':>12}")
        for f, t, a, b in zip(h["features"], h["true"], h["learned"], h["exact_least_squares"]):
            L.append(f"    {f:<12}{t:>10,.0f}{a:>12,.1f}{b:>12,.1f}")
        L.append(f"    gradient descent vs exact least squares: max relative gap {h['max_gap_to_exact']:.1e}")
        L.append(f"    test RMSE ${h['test_rmse']:,.0f};  the lesson's house [1500, 3, 1990] -> "
                 f"${h['course_house_prediction']:,.0f}")

        _lesson(L, "Gradient Descent: How Models Learn")
        L.append("  learning rate (MSE on standardized prices, 100 epochs):")
        for row in r["lr"]:
            status = "DIVERGED" if row["diverged"] else ""
            L.append(f"    lr={row['lr']:<6} epoch 1 {row['loss_epoch1']:.4g}  epoch 10 {row['loss_epoch10']:.4g}  "
                     f"final {row['final_loss']:.4g} after {row['epochs_run']} epochs  {status}")
        L.append("  batch vs mini-batch vs stochastic (same 5 passes over the data):")
        for row in r["flavours"]:
            L.append(f"    {row['method']:<16} {row['updates']:>5} updates  final loss {row['final_loss']:.4f}  "
                     f"uphill steps {row['uphill_steps']:>4} ({_pct(row['uphill_fraction'])})")

        _lesson(L, "Bias, Variance, and Overfitting")
        rd = r["ridge"]
        L.append(f"  L2 (ridge): degree-{rd['degree']} polynomial, {rd['n_points']} noisy points of sin(pi x)")
        L.append(f"    {'lambda':>8} {'train MSE':>10} {'test MSE':>10} {'max |w|':>9}")
        for row in rd["rows"]:
            L.append(f"    {row['lambda']:>8g} {row['train_mse']:>10.4f} {row['test_mse']:>10.4f} {row['max_abs_weight']:>9.2f}")
        L.append(f"    -> lambda=0 overfits, lambda={rd['best_lambda']:g} generalizes best, lambda=10 underfits")
        es = r["early"]
        L.append(f"  validation set + early stopping ({es['sizes'][0]} train / {es['sizes'][1]} validation / "
                 f"{es['sizes'][2]} test, noisy moons):")
        L.append(f"    validation loss lowest at epoch {es['best_epoch']} ({es['val_loss_at_best']:.3f}), "
                 f"{es['val_loss_final']:.3f} by the last epoch while training loss kept falling "
                 f"({es['train_loss_at_best']:.3f} -> {es['train_loss_final']:.3f})")
        L.append(f"    patience {es['patience']}: would stop at epoch {es['stop_epoch']} and restore epoch {es['best_epoch']}")
        L.append(f"    test: full run {_pct(es['full_run']['test_accuracy'])} / loss {es['full_run']['test_loss']:.3f};  "
                 f"early-stopped {_pct(es['early_stopped']['test_accuracy'])} / loss {es['early_stopped']['test_loss']:.3f}")
        dv = r["depth_val"]
        L.append("  choosing tree depth on the validation set (test set touched once):")
        L.append("    " + "  ".join(f"d{row['depth']}:{_pct(row['validation'])}" for row in dv["rows"]))
        L.append(f"    chosen depth {dv['chosen_depth']} -> test {_pct(dv['test_accuracy'])}")

        _lesson(L, "The Limits: No Free Lunch and VC Dimension")
        for name, v in r["vc"].items():
            fail = f"  impossible: {', '.join(v['failed'])}" if v["failed"] else ""
            L.append(f"  a straight line on {name}: {v['separable']}/{v['labelings']} labelings separable{fail}")
        L.append("  -> a line shatters the triangle but not the square (the lesson's result: no 4 points can be")
        L.append("     shattered by a line, so a 2-D line has VC dimension 3)")
        L.append("  No Free Lunch: see the leaderboards in  python -m modelforge lab")

    @staticmethod
    def m3(L, r):
        _lesson(L, "The Decision Boundary Mindset")
        L.append(f"  logistic regression on moons — the boundary is the learned weights:")
        L.append(f"    {r['boundary']['equation']}")

        _lesson(L, "Logistic Regression and Beyond")
        for e in r["examples"]:
            if e["lesson"].startswith("Logistic"):
                L.append(f"  lesson: {e['claim']}")
                L.append(f"    code: {e['computed']}")

        _lesson(L, "Trees, Forests, and Neighbors")
        for row in r["gini_entropy"]:
            L.append(f"  depth-5 tree on {row['dataset']:<8} Gini {_pct(row['gini'])}   entropy {_pct(row['entropy'])}")
        L.append("  k-NN in high dimensions — nearest / farthest distance for random points:")
        for row in r["curse"]:
            L.append(f"    {row['dims']:>5} dims: {row['nearest_over_farthest']:.3f}")
        L.append("    -> near 1 means 'nearest' is barely nearer than anything else")

        _lesson(L, "Evaluating Classifiers Honestly")
        for e in r["examples"]:
            if e["lesson"].startswith("Evaluating"):
                L.append(f"  lesson: {e['claim']}")
                L.append(f"    code: {e['computed']}")
        t = r["threshold"]
        L.append(f"  rare-event data: {t['positives_in_test']} positives in {t['n_test']} test rows "
                 f"({_pct(t['positive_rate'])}), logistic regression:")
        L.append(f"    {'threshold':<13}{'accuracy':>9}{'precision':>10}{'recall':>8}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
        for row in t["rows"]:
            th = row["threshold"] if isinstance(row["threshold"], str) else f"p >= {row['threshold']}"
            L.append(f"    {th:<13}{_pct(row['accuracy']):>9}{row['precision']:>10.3f}{row['recall']:>8.3f}"
                     f"{row['f1']:>7.3f}{row['tp']:>5}{row['fp']:>5}{row['fn']:>5}")
        L.append("  -> 'always no' scores 95% accuracy and catches nothing; lowering the threshold trades "
                 "precision for recall")

    @staticmethod
    def m4(L, r):
        _lesson(L, "From Neuron to Network")
        n = r["neuron"]
        L.append(f"  logistic regression {_pct(n['logistic_regression'])} vs network with no hidden layer "
                 f"{_pct(n['no_hidden_layer_net'])}; identical predictions on {_pct(n['same_predictions'])} of test points")

        _lesson(L, "Activation Functions and Nonlinearity")
        ab = r["ablation"]
        L.append(f"  circles: no hidden layer {_pct(ab['no_hidden_layer'])} -> one hidden layer {_pct(ab['one_hidden_layer'])}")
        a = r["activations"]
        pts = a["points"]
        L.append(f"  slope f'(z) at z = {', '.join(f'{p:g}' for p in pts)}, and moons accuracy after {a['epochs']} epochs:")
        for row in a["rows"]:
            slopes = " ".join(f"{row['slope'][p]:>7.3f}" for p in pts)
            L.append(f"    {row['activation']:<11}{slopes}   {_pct(row['test_accuracy']):>6}  "
                     f"loss {row['final_loss']:.3f}")
        d = r["dying"]
        L.append("  dying ReLU — one huge update pushes all 16 units below zero, then training continues:")
        for k, v in d.items():
            L.append(f"    {k:<11} units stuck with zero gradient: {v['dead_after']}/16   test {_pct(v['test_accuracy'])}")

        _lesson(L, "Backpropagation: Learning Across Layers")
        for g in r["gradcheck"]:
            L.append(f"  {g['activation']:<10} max relative error {g['max_relative_error']:.2e}  "
                     f"{'PASS' if g['passed'] else 'FAIL'}")

        _lesson(L, "Why Deep Networks Work (and When They Don't)")
        t = r["transfer"]
        L.append(f"  transfer learning: {t['dims']}-dimensional sensor data, related target task, {t['n_test']} test points")
        L.append(f"    {'labels':>6} {'from scratch':>13} {'fine-tuned':>11} {'mismatched source':>18}")
        for row in t["rows"]:
            L.append(f"    {row['n_labels']:>6} {_pct(row['from_scratch']):>13} {_pct(row['fine_tuned']):>11} "
                     f"{_pct(row['mismatched_source']):>18}")
        L.append("  -> transfer helps most with the fewest labels; a mismatched source helps less")

    @staticmethod
    def m5(L, r):
        _lesson(L, "CNNs: Seeing with Convolutions")
        c = r["cnn"]
        p = c["params"]
        L.append(f"  a 224x224 colour image = {p['inputs']:,} numbers; 64 outputs need "
                 f"{p['dense_weights']:,} weights fully connected vs {p['conv_weights']:,} as 3x3 conv filters")
        L.append(f"  edge kernel (positive left, negative right): {c['edge_response']:.0f} at the edge, "
                 f"{c['flat_region_response']:.0f} on flat regions")
        for col, s in c["shifted"].items():
            L.append(f"    edge at column {col}: feature row {[int(v) for v in s['feature_row']]}  "
                     f"2x2 max-pooled {[int(v) for v in s['pooled_row']]}  global max {s['global_max']:.0f}")
        L.append("    -> the response moves with the edge (weight sharing); pooling keeps it; global max is unchanged")
        L.append(f"  image rotated 90 degrees: vertical-edge kernel {c['rotated_vertical_kernel']:.0f}, "
                 f"horizontal-edge kernel {c['rotated_horizontal_kernel']:.0f}  -> no free rotation invariance")

        _lesson(L, "RNNs and the Vanishing Gradient Problem")
        for e in r["examples"]:
            if e["lesson"].startswith("RNN"):
                L.append(f"  lesson: {e['claim']}")
                L.append(f"    code: {e['computed']}")
        rn = r["rnn"]
        L.append(f"  gradient after 20 steps back: factor 0.5 -> {rn['factor_0.5']:.2e}, 1.0 -> {rn['factor_1.0']:.1f}, "
                 f"2.0 -> {rn['factor_2.0']:,.0f}")
        L.append(f"  real tanh RNN (w=0.9): {rn['tanh_rnn_10']:.3f} after 10 steps, {rn['tanh_rnn_50']:.2e} after 50")

        _lesson(L, "LSTMs and GRUs: Gating Memory")
        g = r["gating"]
        L.append("  gradient along the LSTM cell state over 50 steps (product of forget gates):")
        for fb, v in g["lstm_cell_50"].items():
            L.append(f"    forget bias {fb:g} (gate {g['lstm_forget_gate'][fb]:.3f}): {v:.2e}")
        L.append(f"    plain tanh RNN over the same 50 steps: {g['tanh_rnn_50']:.2e}")
        L.append(f"  parameters for input 256, hidden 256: LSTM {g['params_256']['lstm']:,}  GRU {g['params_256']['gru']:,}"
                 f"  ({_pct(g['gru_saving'])} fewer)")
        L.append(f"  one GRU step: update gate {[round(v, 3) for v in g['gru_gates']['update']]}, "
                 f"reset gate {[round(v, 3) for v in g['gru_gates']['reset']]}")
        cl = rn["clip"]
        L.append(f"  gradient clipping: {cl['before']} (norm 500,000) -> {cl['after']} (norm {cl['max_norm']:g}); "
                 f"direction kept (cosine {cl['cosine']:.3f})")

        _lesson(L, "Transformers and Self-Attention")
        at = r["attention"]
        L.append(f"  '{at['sentences'][0]}' vs '{at['sentences'][1]}':")
        L.append(f"    attention output for 'dog', largest difference — no positional encoding "
                 f"{at['dog_difference_without_pe']:.3f}, with positional encoding {at['dog_difference_with_pe']:.3f}")
        L.append("    -> without positions the model cannot tell the two sentences apart")
        for row in at["cost"]:
            L.append(f"  length {row['n']:>5}: full attention {row['full']:>9,} scores   local (window 8) {row['local_w8']:>7,}")
        for e in r["examples"]:
            if e["lesson"].startswith("Transformers"):
                L.append(f"  lesson: {e['claim']}  ->  code: {e['computed']}")

    @staticmethod
    def m6(L, r):
        _lesson(L, "MLOps Foundations: Versioning and CI/CD")
        v = r["versioning"]
        L.append(f"  data version of the moons training set: {v['data_version']}; after nudging one value: "
                 f"{v['after_one_edit']}")
        val = r["validation"]
        for k in ("clean_batch", "column_with_nulls", "out_of_range", "unknown_label"):
            L.append(f"  validate {k.replace('_', ' '):<18} -> {val[k] or 'OK'}")
        lc = r["lifecycle"]
        for e in lc["events"]:
            L.append(f"  gate: {e['submitted']:<32} {_pct(e['test_accuracy']):>6}  "
                     f"{'PROMOTED' if e['promoted'] else 'REFUSED'} — {e['reason']}")
        L.append(f"  stages before rollback: {lc['stages_before_rollback']}")
        L.append(f"  stages after rollback:  {lc['stages_after_rollback']}")
        L.append(f"  production model records data {lc['champion_meta']['data_version']}, "
                 f"code {lc['champion_meta']['code_version']}")

        _lesson(L, "Deploying Models Safely")
        c = r["canary"]
        L.append(f"  offline: champion (tree) {_pct(c['offline']['champion'])}, challenger (network) "
                 f"{_pct(c['offline']['challenger'])}")
        for label, key in (("canary, correct serving path", "good"), ("canary, serving path with a hidden bug", "skewed")):
            run = c[key]
            stages = "  ".join(f"{int(s['share'] * 100)}%: {_pct(s['challenger_error'])} vs {_pct(s['champion_error'])}"
                               for s in run["stages"])
            L.append(f"  {label}: {run['result'].upper()}  ({stages})")
            L.append(f"    requests that ever reached the challenger: {run['requests_exposed']} of {run['total_requests']}")
        sh = c["shadow"]
        L.append(f"  shadow deployment: {sh['requests']} requests, agreement {_pct(sh['agreement'])}, "
                 f"champion {_pct(sh['champion_accuracy'])}, shadow {_pct(sh['shadow_accuracy'])}, "
                 f"users shown shadow answers: {sh['users_shown_shadow_answers']}")
        fs = r["feature_store"]
        L.append(f"  training-serving skew — 'average purchase over the last 30 days', {fs['n_test']} live customers:")
        L.append(f"    feature store:     {fs['store_rows_matching_training_code']}/{fs['n_test']} values match training, "
                 f"accuracy {_pct(fs['accuracy_feature_store'])}")
        L.append(f"    re-implementation: {fs['rewrite_rows_matching_training_code']}/{fs['n_test']} match "
                 f"(mean gap ${fs['mean_feature_gap']:.2f}), {fs['predictions_changed_by_rewrite']} decisions changed, "
                 f"accuracy {_pct(fs['accuracy_rewrite'])} — and no error message")

        _lesson(L, "Monitoring Drift: Why Models Decay")
        d = r["drift"]
        for name, s in d["scenarios"].items():
            L.append(f"  {name:<40} max PSI {s['max_psi']:.3f}  KS p {s['min_ks_p']:.1e}  "
                     f"live acc {_pct(s['live_accuracy'])}  -> {s['action']}")
        L.append(f"  retraining trigger fired for data drift: {_pct(d['old_model_on_moved_world'])} -> "
                 f"{_pct(d['retrained_on_moved_world'])} after retraining on the moved world")

        _lesson(L, "The Open Frontier: What ML Still Can't Do")
        f = r["frontier"]
        L.append("  adversarial robustness (network on moons, step size eps in scaled units):")
        for row in f["adversarial"]:
            L.append(f"    eps {row['eps']:.1f}: targeted (FGSM) {_pct(row['adversarial_accuracy'])}   "
                     f"random noise of the same size {_pct(row['random_noise_accuracy'])}")
        fl = f["flipped"]
        if fl:
            L.append(f"    a point the network was {_conf(fl['confidence_before'])} sure about drops to "
                     f"{_pct(fl['confidence_after_true_class'])} for its true class after a {fl['eps']} step")
        L.append(f"  epistemic uncertainty (out-of-distribution guard threshold {f['guard_threshold']:.2f}):")
        for row in f["overconfidence"]:
            L.append(f"    input {[round(v, 2) for v in row['x']]}: network confidence {_conf(row['confidence'])}  "
                     f"distance to training data {row['distance']:.2f}  "
                     f"{'-> FLAGGED: unlike anything in training' if row['flagged_unknown'] else ''}")
        it = f["interpretability"]
        L.append("  interpretability — permutation importance (accuracy drop when a column is shuffled):")
        for name, m in it["models"].items():
            imp = "  ".join(f"{fn} {v:+.3f}" for fn, v in zip(it["features"], m["importance"]))
            L.append(f"    {name:<20} {imp}")
        L.append("    -> every model ignores the noise column, but they split the credit differently")
        fg = r["forgetting"]
        L.append("  catastrophic forgetting (task A = moons on the left, task B = circles on the right):")
        L.append(f"    after training on A:       A {_pct(fg['after_A']['task_A'])}   B {_pct(fg['after_A']['task_B'])}")
        L.append(f"    then training only on B:   A {_pct(fg['after_B']['task_A'])}   B {_pct(fg['after_B']['task_B'])}")
        L.append(f"    training on both at once:  A {_pct(fg['joint']['task_A'])}   B {_pct(fg['joint']['task_B'])}")

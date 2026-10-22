"""Unit tests for ModelForge. No pytest required:  python tests/test_all.py
(also runs unchanged under pytest).

The tests check that the math is right, not just that the code runs: softmax
sums to one, Gini is zero for a pure node, backprop matches a numerical
derivative, a saved model predicts identically after reloading, and the
linear model's known limit on non-linear data is pinned so it can't silently
change. The second half pins one claim from each remaining lesson of the
course (see COURSE_COVERAGE.md).
"""

import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelforge import linalg
from modelforge.architectures import (SOBEL_X, conv2d, demo_image, demo_tokens, lstm_step,
                                      self_attention, tiny_lstm_params, vanishing_gradient)
from modelforge.datasets import (StandardScaler, make_blobs, make_circles, make_moons,
                                 train_test_split)
from modelforge.linear import SoftmaxRegression
from modelforge.metrics import (accuracy, confusion_matrix, cross_entropy,
                                precision_recall_f1)
from modelforge.neighbors import KNearestNeighbors
from modelforge.neural import MLP
from modelforge.pipeline import compare, depth_sweep, prepare
from modelforge.registry import ModelRegistry, drift_report, model_from_dict, psi, shift
from modelforge.tree import DecisionTree, RandomForest, gini


def _split(name="moons"):
    Xtr, Xte, ytr, yte, scaler = prepare(name)
    return Xtr, Xte, ytr, yte, scaler


# --- linear algebra (Module 2) ------------------------------------------------ #
def test_dot_and_matvec():
    assert linalg.dot([1, 2, 3], [4, 5, 6]) == 32
    assert linalg.matvec([[1, 0], [0, 2]], [3, 4]) == [3, 8]

def test_softmax_sums_to_one_and_is_stable():
    p = linalg.softmax([1000.0, 1001.0, 999.0])      # would overflow without the max shift
    assert abs(sum(p) - 1.0) < 1e-12 and linalg.argmax(p) == 1

def test_sigmoid_stable_at_extremes():
    assert linalg.sigmoid(-1000) == 0.0 and linalg.sigmoid(1000) == 1.0
    assert abs(linalg.sigmoid(0) - 0.5) < 1e-12

# --- datasets and preparation (Module 1) -------------------------------------- #
def test_generators_are_reproducible():
    assert make_moons(seed=3) == make_moons(seed=3)
    assert make_moons(seed=3) != make_moons(seed=4)

def test_stratified_split_keeps_every_class():
    X, y = make_blobs(n=90, centers=3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
    assert set(yte) == {0, 1, 2} and len(Xtr) + len(Xte) == 90

def test_scaler_centres_training_data():
    X, _ = make_circles()
    Z = StandardScaler().fit_transform(X)
    for j in range(2):
        col = [r[j] for r in Z]
        mean = sum(col) / len(col)
        std = math.sqrt(sum((v - mean) ** 2 for v in col) / len(col))
        assert abs(mean) < 1e-9 and abs(std - 1) < 1e-9

# --- metrics (Module 3) ------------------------------------------------------- #
def test_confusion_matrix_counts():
    cm = confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1])
    assert cm["matrix"] == [[1, 1], [0, 2]]

def test_precision_recall_f1_known_values():
    prf = precision_recall_f1([1, 1, 1, 0], [1, 1, 0, 0])
    c1 = prf["per_class"][1]
    assert c1["precision"] == 1.0 and abs(c1["recall"] - 2 / 3) < 1e-12

def test_cross_entropy_perfect_is_near_zero():
    assert cross_entropy([[1.0, 0.0], [0.0, 1.0]], [0, 1]) < 1e-9

# --- logistic regression + gradient descent (Modules 2-3) -------------------- #
def test_gradient_descent_reduces_loss():
    Xtr, _, ytr, _, _ = _split("blobs")
    m = SoftmaxRegression(epochs=100).fit(Xtr, ytr)
    assert m.loss_history[-1] < m.loss_history[0] * 0.2

def test_logistic_solves_linear_blobs():
    Xtr, Xte, ytr, yte, _ = _split("blobs")
    assert accuracy(yte, SoftmaxRegression().fit(Xtr, ytr).predict(Xte)) >= 0.95

def test_logistic_cannot_solve_circles():
    # Pinned honest limit: a straight line cannot separate one ring inside another.
    Xtr, Xte, ytr, yte, _ = _split("circles")
    assert accuracy(yte, SoftmaxRegression().fit(Xtr, ytr).predict(Xte)) < 0.7

# --- neighbours and trees (Module 3; bias-variance, Module 2) ----------------- #
def test_knn_k1_memorizes_training_set():
    Xtr, _, ytr, _, _ = _split("moons")
    assert accuracy(ytr, KNearestNeighbors(1).fit(Xtr, ytr).predict(Xtr)) == 1.0

def test_knn_rejects_bad_k():
    try:
        KNearestNeighbors(0); assert False, "expected ValueError"
    except ValueError:
        pass

def test_gini_extremes():
    assert gini([1, 1, 1]) == 0.0 and abs(gini([0, 1, 0, 1]) - 0.5) < 1e-12

def test_tree_respects_max_depth():
    Xtr, _, ytr, _, _ = _split("moons")
    assert DecisionTree(max_depth=3).fit(Xtr, ytr).depth() <= 3

def test_deeper_trees_fit_training_data_better():
    sweep = depth_sweep("moons", depths=(1, 12))
    assert sweep[1]["train_accuracy"] > sweep[0]["train_accuracy"]

def test_forest_beats_single_tree_on_moons():
    Xtr, Xte, ytr, yte, _ = _split("moons")
    tree = accuracy(yte, DecisionTree(max_depth=5).fit(Xtr, ytr).predict(Xte))
    forest = accuracy(yte, RandomForest(n_trees=15).fit(Xtr, ytr).predict(Xte))
    assert forest >= tree

# --- neural network + backprop (Module 4) ------------------------------------- #
def test_backprop_matches_numerical_gradient():
    X, y = make_moons(n=60)
    X = StandardScaler().fit_transform(X)
    for act in ("tanh", "sigmoid", "relu"):
        net = MLP(hidden=(6, 5), activation=act, epochs=2, seed=2).fit(X, y)
        cidx = {c: i for i, c in enumerate(net.classes)}
        assert net.gradient_check(X[1], cidx[y[1]]) < 1e-4, act

def test_neural_net_beats_linear_on_moons():
    Xtr, Xte, ytr, yte, _ = _split("moons")
    lin = accuracy(yte, SoftmaxRegression().fit(Xtr, ytr).predict(Xte))
    net = accuracy(yte, MLP(hidden=(16, 16), epochs=80).fit(Xtr, ytr).predict(Xte))
    assert net > lin + 0.05

def test_nonlinearity_is_what_matters():
    # With no hidden layer the "network" is just logistic regression: it fails circles.
    Xtr, Xte, ytr, yte, _ = _split("circles")
    flat = accuracy(yte, MLP(hidden=(), epochs=40).fit(Xtr, ytr).predict(Xte))
    deep = accuracy(yte, MLP(hidden=(16,), epochs=80).fit(Xtr, ytr).predict(Xte))
    assert flat < 0.75 and deep > 0.9

# --- architectures (Module 5) ------------------------------------------------- #
def test_sobel_fires_only_at_the_edge():
    fmap = conv2d(demo_image(8), SOBEL_X)
    for row in fmap:
        assert row == [0, 0, 4, 4, 0, 0]

def test_attention_rows_are_distributions():
    _, E = demo_tokens()
    _, w = self_attention(E, E, E)
    assert all(abs(sum(r) - 1) < 1e-12 for r in w)
    assert w[0][2] > w[0][1]        # "the" attends to "a" more than to "cat"

def test_sigmoid_gradients_vanish_relu_does_not():
    assert vanishing_gradient(20, "sigmoid")[-1] < 1e-10
    assert vanishing_gradient(20, "relu")[-1] == 1.0

def test_lstm_state_is_bounded():
    h, c, gates = lstm_step([1.0, 0.0], [0.0] * 3, [0.0] * 3, tiny_lstm_params())
    assert all(-1 < v < 1 for v in h) and all(0 < f < 1 for f in gates["forget"])

# --- production: registry, gate, drift (Module 6) ------------------------------ #
def test_every_model_survives_save_and_reload():
    Xtr, Xte, ytr, _, _ = _split("moons")
    for m in (SoftmaxRegression(epochs=50), KNearestNeighbors(3), DecisionTree(4),
              RandomForest(n_trees=4), MLP(hidden=(6,), epochs=5)):
        m.fit(Xtr, ytr)
        clone = model_from_dict(m.to_dict())
        assert clone.predict(Xte) == m.predict(Xte), m.name

def test_registry_gate_promotes_better_and_refuses_worse():
    tmp = tempfile.mkdtemp()
    try:
        Xtr, Xte, ytr, yte, scaler = _split("moons")
        reg = ModelRegistry(tmp)
        weak = SoftmaxRegression(epochs=50).fit(Xtr, ytr)
        strong = KNearestNeighbors(5).fit(Xtr, ytr)
        m1 = reg.save(weak, scaler, {"test_accuracy": 0.84}, "moons")
        assert reg.promote(m1)["promoted"]                       # first becomes champion
        m2 = reg.save(strong, scaler, {"test_accuracy": 0.97}, "moons")
        assert reg.promote(m2)["promoted"]                       # clearly better
        m3 = reg.save(DecisionTree(2).fit(Xtr, ytr), scaler, {"test_accuracy": 0.80}, "moons")
        assert not reg.promote(m3)["promoted"]                   # worse: refused
        model, _, meta = reg.load()
        assert meta["version"] == m2["version"]
        assert model.predict(Xte) == strong.predict(Xte)
    finally:
        shutil.rmtree(tmp)

def test_version_hash_is_content_addressed():
    tmp = tempfile.mkdtemp()
    try:
        Xtr, _, ytr, _, scaler = _split("blobs")
        reg = ModelRegistry(tmp)
        a = reg.save(KNearestNeighbors(3).fit(Xtr, ytr), scaler, {"test_accuracy": 1}, "blobs")
        b = reg.save(KNearestNeighbors(3).fit(Xtr, ytr), scaler, {"test_accuracy": 1}, "blobs")
        assert a["version"] == b["version"] and len(reg.versions()) == 1
    finally:
        shutil.rmtree(tmp)

def test_psi_low_for_same_world_high_for_shifted():
    train, _ = make_moons(seed=7)
    fresh, _ = make_moons(seed=99)
    assert psi([r[0] for r in train], [r[0] for r in fresh]) < 0.1
    moved = shift(fresh, dx=0.9, dy=-0.6, scale=1.3)
    rep = drift_report(train, moved)
    assert rep["alert"] and rep["max_psi"] > 0.25

# --- integration -------------------------------------------------------------- #
def test_lab_gives_identical_results_on_a_second_run():
    # The lab rebuilds its registry each run; otherwise run two would see run one's
    # champion and make different promotion decisions.
    from modelforge.lab import production_demo
    tmp = tempfile.mkdtemp()
    try:
        board = compare("moons", fast=True)
        a, b = production_demo(board, tmp), production_demo(board, tmp)
        assert [e["promoted"] for e in a["events"]] == [e["promoted"] for e in b["events"]]
        assert a["champion"]["version"] == b["champion"]["version"]
        assert a["drift_shifted"] == b["drift_shifted"]
    finally:
        shutil.rmtree(tmp)

def test_cli_train_promote_predict_drift():
    import contextlib
    import io
    from modelforge.cli import main
    tmp = tempfile.mkdtemp()
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for model in ("logistic_regression", "decision_tree", "neural_network", "knn"):
                assert main(["--registry", tmp, "train", "--model", model, "--fast"]) == 0
            assert main(["--registry", tmp, "predict", "0.5", "-0.3"]) == 0
            assert main(["--registry", tmp, "drift"]) == 0                       # same world
            assert main(["--registry", tmp, "drift", "--dx", "0.9", "--dy", "-0.6",
                         "--scale", "1.3"]) == 2                                   # alert
        text = out.getvalue()
        assert text.count("PROMOTED:") == 4 and text.count("NOT PROMOTED:") == 1
    finally:
        shutil.rmtree(tmp)

def test_compare_ranks_all_five_models():
    res = compare("blobs", fast=True)
    accs = [r["test_accuracy"] for r in res["rows"]]
    assert len(res["rows"]) == 5 and accs == sorted(accs, reverse=True)


# =========================================================================== #
# Course coverage: one test per lesson claim (Modules 1-6)
# =========================================================================== #
from modelforge import architectures as A
from modelforge import deep_experiments as DX
from modelforge import experiments as EX
from modelforge import mlops
from modelforge.frontier import adversarial_robustness, permutation_importance
from modelforge.paradigms import GridWorld, KMeans, purity, q_learning
from modelforge.regression import mse
from modelforge.registry import data_fingerprint, ks_test
from modelforge.tree import entropy


# --- Module 1 ----------------------------------------------------------------- #
def test_kmeans_recovers_the_three_hidden_blobs():
    X, y = make_blobs()
    km = KMeans(k=3).fit(X)
    assert purity(km.labels_, y) == 1.0
    assert KMeans(k=2).fit(X).inertia > km.inertia > KMeans(k=4).fit(X).inertia

def test_q_learning_finds_the_shortest_path_from_delayed_reward():
    env = GridWorld()
    res = q_learning(env)
    assert res["reached_goal"] and len(res["greedy_path"]) - 1 == 2 * (env.size - 1)
    assert res["steps_per_episode"][0] > sum(res["steps_per_episode"][-20:]) / 20

def test_biased_label_errors_are_not_fixed_by_more_data():
    n = EX.label_noise()
    assert abs(n["biased_rate"] - 0.15) < 0.02
    assert n["clean_300"] > n["random15_300"] and n["clean_300"] > n["biased_300"]
    assert (n["random15_1200"] - n["random15_300"]) > 3 * (n["biased_1200"] - n["biased_300"])

# --- Module 2 ----------------------------------------------------------------- #
def test_course_worked_examples():
    assert linalg.dot([1500, 3, 1990], [200, 10000, -50]) == 230500
    assert linalg.dot([2000, 4, 2000], [100, 5000, -20]) == 180000
    assert mse([250000], [230500]) == 380250000

def test_gradient_descent_reaches_the_exact_least_squares_answer():
    h = EX.house_regression()
    assert h["max_gap_to_exact"] < 1e-6
    assert abs(h["learned"][0] - 200) < 2            # $ per square foot recovered

def test_learning_rate_too_large_diverges_too_small_crawls():
    small, good, large = EX.learning_rate_sweep()
    assert large["diverged"] and not good["diverged"]
    assert good["final_loss"] < 0.01 < small["final_loss"]

def test_sgd_zigzags_and_batch_does_not():
    batch, mini, sgd = EX.gd_flavours()
    assert batch["uphill_steps"] == 0 and sgd["uphill_fraction"] > 0.2
    assert batch["updates"] < mini["updates"] < sgd["updates"]

def test_l2_is_the_bias_variance_dial():
    r = EX.ridge_sweep()
    rows = {x["lambda"]: x for x in r["rows"]}
    best = rows[r["best_lambda"]]["test_mse"]
    assert best < rows[0.0]["test_mse"] and best < rows[10.0]["test_mse"]
    assert rows[0.0]["train_mse"] < rows[10.0]["train_mse"]
    weights = [x["max_abs_weight"] for x in r["rows"]]
    assert weights == sorted(weights, reverse=True)

def test_early_stopping_restores_the_best_validation_epoch():
    Xtr, Xte, ytr, yte, _ = _split("moons")
    net = MLP(hidden=(8,), epochs=40).fit(Xtr, ytr, Xte, yte, patience=3)
    best = min(net.val_loss_history)
    cidx = {c: i for i, c in enumerate(net.classes)}
    assert abs(cross_entropy(net.predict_proba(Xte), [cidx[v] for v in yte]) - best) < 1e-12

def test_a_line_shatters_three_points_but_not_four():
    vc = EX.vc_shattering()
    assert vc["3 points (triangle)"]["separable"] == 8
    assert vc["4 points (square)"]["separable"] == 14
    assert sorted(vc["4 points (square)"]["failed"]) == ["0110", "1001"]

def test_linear_solver():
    x = linalg.solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0])
    assert abs(x[0] - 0.8) < 1e-12 and abs(x[1] - 1.4) < 1e-12

# --- Module 3 ----------------------------------------------------------------- #
def test_softmax_course_example_and_two_class_sigmoid():
    p = linalg.softmax([2.0, 1.0, 0.1])
    assert [round(v, 3) for v in p] == [0.659, 0.242, 0.099]
    assert abs(linalg.softmax([0.0, 2.2])[1] - linalg.sigmoid(2.2)) < 1e-12
    assert round(linalg.sigmoid(2.2), 2) == 0.90

def test_accuracy_lies_on_imbalanced_data_and_threshold_trades_off():
    t = EX.threshold_tradeoff()
    lazy, *rows = t["rows"]
    assert lazy["accuracy"] >= 0.94 and lazy["recall"] == 0.0
    recalls = [r["recall"] for r in rows]                 # thresholds 0.9 -> 0.1
    assert recalls == sorted(recalls) and rows[0]["precision"] > rows[-1]["precision"]

def test_entropy_criterion():
    assert entropy([1, 1, 1]) == 0.0 and abs(entropy([0, 1]) - 1.0) < 1e-12
    Xtr, _, ytr, _, _ = _split("moons")
    t = DecisionTree(max_depth=3, criterion="entropy").fit(Xtr, ytr)
    assert model_from_dict(t.to_dict()).predict(Xtr) == t.predict(Xtr)

def test_distances_lose_meaning_in_high_dimensions():
    ratios = [r["nearest_over_farthest"] for r in EX.curse_of_dimensionality()]
    assert ratios == sorted(ratios) and ratios[0] < 0.05 and ratios[-1] > 0.85

# --- Module 4 ----------------------------------------------------------------- #
def test_modern_activations_pass_the_gradient_check():
    X, y = make_moons()
    X = StandardScaler().fit_transform(X)
    for act in ("leaky_relu", "gelu", "swish"):
        net = MLP(hidden=(8, 6), activation=act, epochs=3, seed=1).fit(X, y)
        assert net.gradient_check(X[5], y[5], samples=30) < 1e-4, act

def test_network_without_hidden_layer_is_logistic_regression():
    r = DX.neuron_is_logistic_regression()
    assert r["same_predictions"] == 1.0

def test_dead_relu_units_never_recover_but_leaky_ones_learn():
    d = DX.dying_relu()
    assert d["relu"]["dead_after"] == 16 and d["relu"]["test_accuracy"] == 0.5
    assert d["leaky_relu"]["dead_after"] == 0 and d["leaky_relu"]["test_accuracy"] > 0.7

def test_fine_tuning_freezes_the_hidden_layers():
    Xtr, _, ytr, _, _ = _split("moons")
    net = MLP(hidden=(6,), epochs=5).fit(Xtr, ytr)
    hidden_before = [r[:] for r in net.W[0]]
    out_before = [r[:] for r in net.W[1]]
    net.fine_tune(Xtr[:20], ytr[:20], epochs=5)
    assert net.W[0] == hidden_before and net.W[1] != out_before

# --- Module 5 ----------------------------------------------------------------- #
def test_convolution_follows_a_shift_but_not_a_rotation():
    a = A.conv2d(A.edge_image(8, 4), A.COURSE_EDGE_KERNEL)
    b = A.conv2d(A.edge_image(8, 5), A.COURSE_EDGE_KERNEL)
    assert A.strongest_column(b) == A.strongest_column(a) + 1     # moves with the edge
    assert A.global_max(a) == A.global_max(b) == 4.0             # pooled answer unchanged
    rotated = A.rotate90(A.edge_image(8, 4))
    assert A.global_max(A.conv2d(rotated, A.COURSE_EDGE_KERNEL)) == 0.0

def test_gradient_clipping_caps_size_and_keeps_direction():
    g = A.clip_by_norm([3e5, 4e5], 5.0)
    assert abs(g[0] - 3.0) < 1e-9 and abs(g[1] - 4.0) < 1e-9
    assert A.clip_by_norm([0.3, 0.4], 5.0) == [0.3, 0.4]
    assert A.bptt_gradient(2.0, 20)[-1] == 2 ** 20 and A.bptt_gradient(0.5, 20)[-1] == 0.5 ** 20

def test_gru_is_smaller_than_lstm_and_bounded():
    p = A.count_gate_params(256, 256)
    assert p["gru"] / p["lstm"] == 0.75
    h, gates = A.gru_step([1.0, 0.0], [0.0] * 3, A.tiny_gru_params())
    assert all(-1 < v < 1 for v in h) and all(0 < z < 1 for z in gates["update"])

def test_open_forget_gate_keeps_the_gradient_alive():
    assert A.lstm_cell_gradient(3.0, 50)[-1] > 1e-2 > A.tanh_rnn_gradient(0.9, 50)[-1]
    assert A.lstm_cell_gradient(0.0, 50)[-1] < 1e-12

def test_attention_needs_positional_encoding_to_see_word_order():
    r = DX.attention_order_and_cost()
    assert r["dog_difference_without_pe"] < 1e-12 < r["dog_difference_with_pe"]
    assert A.attention_cost(1000) == 100 * A.attention_cost(100)
    assert A.attention_cost(1000, window=8) < 20000

# --- Module 6 ----------------------------------------------------------------- #
def test_data_validation_catches_nulls_ranges_and_labels():
    X, y = make_moons()
    schema = mlops.infer_schema(X, y)
    assert mlops.validate_data(X, y, schema) == []
    assert "missing" in mlops.validate_data([[0.1, None]] + X[1:], y, schema)[0]
    assert "outside" in mlops.validate_data([[999.0, 0.0]] + X[1:], y, schema)[0]
    assert "unknown labels" in mlops.validate_data(X, y[:-1] + [5], schema)[0]

def test_data_version_changes_when_one_value_changes():
    X, y = make_moons()
    X2 = [list(r) for r in X]
    X2[0][0] += 1e-9
    assert data_fingerprint(X, y) == data_fingerprint([list(r) for r in X], y) != data_fingerprint(X2, y)

def test_quality_floor_lifecycle_stages_and_rollback():
    tmp = tempfile.mkdtemp()
    try:
        Xtr, Xte, ytr, yte, scaler = _split("moons")
        reg = ModelRegistry(tmp)
        low = reg.save(DecisionTree(1).fit(Xtr, ytr), scaler, {"test_accuracy": 0.70}, "moons")
        assert not reg.promote(low)["promoted"] and reg.champion() is None    # below floor
        a = reg.save(SoftmaxRegression(epochs=50).fit(Xtr, ytr), scaler, {"test_accuracy": 0.84}, "moons")
        b = reg.save(KNearestNeighbors(5).fit(Xtr, ytr), scaler, {"test_accuracy": 0.97}, "moons")
        reg.promote(a)
        reg.promote(b)
        stages = {v["version"]: v["stage"] for v in reg.versions()}
        assert stages[a["version"]] == "archived" and stages[b["version"]] == "production"
        reg.rollback()
        assert reg.champion()["version"] == a["version"]
        stages = {v["version"]: v["stage"] for v in reg.versions()}
        assert stages[b["version"]] == "archived" and stages[low["version"]] == "staging"
    finally:
        shutil.rmtree(tmp)

def test_ks_test_same_world_vs_shifted_world():
    train, _ = make_moons(seed=7)
    fresh, _ = make_moons(seed=99)
    moved = shift(fresh, dx=0.9, dy=-0.6, scale=1.3)
    _, p_same = ks_test([r[0] for r in train], [r[0] for r in fresh])
    d, p_moved = ks_test([r[0] for r in train], [r[0] for r in moved])
    assert p_same > 0.05 and p_moved < 1e-10 and 0 < d <= 1

def test_canary_rolls_back_a_model_that_fails_on_live_traffic():
    Xtr, Xte, ytr, yte, scaler = _split("moons")
    champ = SoftmaxRegression(epochs=100).fit(Xtr, ytr)
    chall = DecisionTree(5).fit(Xtr, ytr)
    live_raw, live_y = make_moons(n=2000, seed=77)
    live = scaler.transform(live_raw)
    ok = mlops.canary_rollout(champ.predict, chall.predict, live, live_y)
    bad = mlops.canary_rollout(champ.predict, lambda X: chall.predict([[r[1], r[0]] for r in X]),
                               live, live_y)
    assert ok["result"] == "fully rolled out"
    assert bad["result"] == "rolled back" and bad["stopped_at"] == 0.05
    assert bad["requests_exposed"] < 0.1 * bad["total_requests"]

def test_feature_store_removes_training_serving_skew():
    from modelforge.ops_experiments import feature_store_skew
    r = feature_store_skew()
    assert r["store_rows_matching_training_code"] == r["n_test"]
    assert r["rewrite_rows_matching_training_code"] < r["n_test"]
    assert r["predictions_changed_by_rewrite"] > 0

def test_monitor_separates_data_drift_from_concept_drift():
    train, _ = make_moons(seed=7)
    fresh, _ = make_moons(seed=99)
    same = drift_report(train, fresh)
    moved = drift_report(train, shift(fresh, dx=0.9, dy=-0.6, scale=1.3))
    assert mlops.monitor_decision(same, 0.98)["action"] == "OK"
    assert mlops.monitor_decision(moved, 0.66)["action"] == "RETRAIN"
    assert mlops.monitor_decision(same, 0.70)["action"] == "HUMAN REVIEW"   # inputs fine, accuracy not

def test_targeted_perturbations_hurt_more_than_random_noise():
    Xtr, Xte, ytr, yte, _ = _split("moons")
    net = MLP(hidden=(8,), epochs=30).fit(Xtr, ytr)
    rows = adversarial_robustness(net, Xte, yte, eps_values=(0.0, 0.3))
    assert rows[0]["adversarial_accuracy"] == rows[0]["random_noise_accuracy"]
    assert rows[1]["adversarial_accuracy"] < rows[1]["random_noise_accuracy"]

def test_ood_guard_flags_inputs_far_from_training_data():
    Xtr, Xte, _, _, _ = _split("moons")
    guard = mlops.OODGuard().fit(Xtr)
    assert not guard.check(Xte[0])["out_of_distribution"]
    assert guard.check([8.0, 8.0])["out_of_distribution"]

def test_permutation_importance_ignores_a_pure_noise_feature():
    import random as _r
    X, y = make_moons()
    rng = _r.Random(4)
    X = StandardScaler().fit_transform([r + [rng.gauss(0, 1)] for r in X])
    tree = DecisionTree(max_depth=5).fit(X, y)
    imp = permutation_importance(tree, X, y)
    assert imp[2] < 0.02 < imp[0] and imp[1] > 0.02

def test_cli_module_rollback_and_stages():
    import contextlib
    import io
    from modelforge.cli import main
    tmp = tempfile.mkdtemp()
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert main(["module", "5"]) == 0
            for model in ("decision_tree", "knn"):
                assert main(["--registry", tmp, "train", "--model", model]) == 0
            assert main(["--registry", tmp, "rollback"]) == 0
            assert main(["--registry", tmp, "registry"]) == 0
            assert main(["--registry", tmp, "predict", "9", "9"]) == 0
        text = out.getvalue()
        assert "Transformers and Self-Attention" in text
        assert "champion is now decision_tree" in text and "archived" in text
        assert "WARNING: this input is" in text
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    import time
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    t0 = time.time()
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed in {time.time() - t0:.1f}s")
    raise SystemExit(1 if failed else 0)

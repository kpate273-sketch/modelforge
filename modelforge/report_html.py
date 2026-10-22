"""Render the full lab as one self-contained HTML report (no external assets).

Open the generated file in any browser. Every chart in it is SVG produced by
viz.py, every number comes from the lab run that generated it.
"""

from __future__ import annotations

import time
from html import escape

from .viz import boundary_svg, curve_fit_svg, line_svg

# Simplest to most flexible, so each row of boundary plots reads as a progression.
MODEL_ORDER = ("logistic_regression", "knn", "decision_tree", "random_forest", "neural_network")
PRETTY = {"logistic_regression": "Logistic regression", "knn": "k-NN",
          "decision_tree": "Decision tree", "random_forest": "Random forest",
          "neural_network": "Neural network"}


def pretty(name: str) -> str:
    return PRETTY.get(name, name)

CSS = """
:root{--bg:#f6f7f9;--card:#ffffff;--ink:#1d2330;--soft:#5b6372;--line:#dfe3ea;
--accent:#3b6fd4;--good:#1f8a52;--bad:#c2412f;--warn:#b5741a;}
@media (prefers-color-scheme: dark){:root{--bg:#10131b;--card:#181c27;--ink:#e8ebf2;
--soft:#9aa2b4;--line:#2b3142;--accent:#7ea2f0;--good:#4cc287;--bad:#ef7a68;--warn:#e0a650;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:19px;margin:34px 0 6px}
.sub{color:var(--soft);margin:0 0 6px}.tag{display:inline-block;font:12px ui-monospace,Menlo,monospace;
padding:2px 8px;border:1px solid var(--line);border-radius:999px;margin:6px 6px 0 0;color:var(--soft)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
.lead{color:var(--soft);margin:0 0 10px;max-width:780px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px 8px;
border-bottom:1px solid var(--line)}th{color:var(--soft);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.num{font-family:ui-monospace,Menlo,monospace;text-align:right}th.num{text-align:right}
.best{color:var(--good);font-weight:700}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
@media (max-width:900px){.grid{grid-template-columns:repeat(3,1fr)}}@media (max-width:560px){.grid{grid-template-columns:repeat(2,1fr)}}
td.mono{white-space:nowrap}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:12px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px;color:var(--ink);overflow:hidden}
.fig svg{max-width:100%;height:auto;display:block;margin:0 auto}
.pass{color:var(--good);font-weight:700}.fail{color:var(--bad);font-weight:700}.warn{color:var(--warn);font-weight:700}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:13px}
.heat td{text-align:center;font-family:ui-monospace,Menlo,monospace;font-size:12px;border:1px solid var(--line)}
.rowlabel{color:var(--soft);font-size:12px;margin:14px 0 4px;text-transform:uppercase;letter-spacing:.06em}
footer{color:var(--soft);font-size:13px;margin-top:40px}
pre.mono{margin:8px 0;font-size:15px;line-height:1.5}
.card table{margin-top:8px}.card{overflow-x:auto}td.num{white-space:nowrap}
.grid2{align-items:start}td span.mono{white-space:nowrap}
"""


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _leaderboard_table(board: dict) -> str:
    rows = board["rows"]
    best = max(r["test_accuracy"] for r in rows)
    h = ['<table><tr><th>Model</th><th class="num">Train</th><th class="num">Test</th>'
         '<th class="num">Macro F1</th><th class="num">Overfit gap</th><th class="num">Fit (s)</th></tr>']
    for r in rows:
        cls = ' class="num best"' if r["test_accuracy"] == best else ' class="num"'
        h.append(f'<tr><td>{escape(pretty(r["model"]))}</td><td class="num">{_pct(r["train_accuracy"])}</td>'
                 f'<td{cls}>{_pct(r["test_accuracy"])}</td><td class="num">{r["macro_f1"]:.3f}</td>'
                 f'<td class="num">{r["overfit_gap"]:+.3f}</td><td class="num">{r["fit_seconds"]:.2f}</td></tr>')
    h.append("</table>")
    return "".join(h)


def _heat(matrix: list, labels: list | None = None, fmt: str = "{:.2f}") -> str:
    flat = [v for row in matrix for v in row]
    lo, hi = min(flat), max(flat)
    rng = (hi - lo) or 1
    out = ['<table class="heat">']
    if labels:
        out.append("<tr><td></td>" + "".join(f"<td><b>{escape(l)}</b></td>" for l in labels) + "</tr>")
    for i, row in enumerate(matrix):
        out.append("<tr>")
        if labels:
            out.append(f"<td><b>{escape(labels[i])}</b></td>")
        for v in row:
            a = 0.08 + 0.6 * (v - lo) / rng
            out.append(f'<td style="background:rgba(59,111,212,{a:.2f})">{fmt.format(v)}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def render(lab: dict) -> str:
    boards = lab["boards"]
    arch = lab["architectures"]
    prod = lab["production"]
    s = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>ModelForge Lab Report</title><style>{CSS}</style></head><body><main>"]
    s.append("<h1>ModelForge — Lab Report</h1>")
    s.append(f"<p class='sub'>Machine learning from scratch, in pure Python. "
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}.</p>")
    s.append("<span class='tag'>zero dependencies</span><span class='tag'>no GPU</span>"
             "<span class='tag'>5 models built from scratch</span><span class='tag'>backprop gradient-checked</span>")

    # 1. leaderboards
    lr_acc = {d: next(r["test_accuracy"] for r in b["rows"] if r["model"] == "logistic_regression")
              for d, b in boards.items()}
    s.append("<h2>1 · The best model depends on the data</h2><p class='lead'>Five models trained by "
             "hand-written code on three generated datasets, scored on a held-out test set. The ranking "
             f"changes with the shape of the data: logistic regression scores {_pct(lr_acc['blobs'])} on "
             f"blobs but finishes last on circles at {_pct(lr_acc['circles'])}, because its one "
             "assumption — a straight-line boundary — fits one dataset and not the other.</p>")
    for d, b in boards.items():
        s.append(f"<div class='card'><b>{escape(d)}</b> <span class='sub'>· train {b['n_train']} / "
                 f"test {b['n_test']}</span>{_leaderboard_table(b)}</div>")

    # 2. decision boundaries
    s.append("<h2>2 · Decision boundaries</h2><p class='lead'>Each panel classifies a grid of points "
             "with the trained model. Logistic regression can only draw a straight line, so it falls short "
             "on moons and fails on circles; k-NN, the tree-based models and the neural network bend "
             "around the data.</p>")
    for d, b in boards.items():
        Xtr, Xte, ytr, yte = b["data"]
        allX, ally = Xtr + Xte, ytr + yte
        s.append(f"<div class='rowlabel'>{escape(d)}</div><div class='grid'>")
        order = {n: i for i, n in enumerate(MODEL_ORDER)}
        for r in sorted(b["rows"], key=lambda r: order.get(r["model"], 99)):
            title = f"{pretty(r['model'])} · {_pct(r['test_accuracy'])}"
            s.append(f"<div class='fig'>{boundary_svg(r['_model'], allX, ally, title)}</div>")
        s.append("</div>")
    ab = lab["ablation"]
    s.append(f"<div class='card'><b>Ablation: is nonlinearity the difference?</b><p class='sub'>The same "
             f"neural network on circles, with and without a hidden layer. Without one it is just logistic "
             f"regression.</p><table><tr><th>Network</th><th class='num'>Test accuracy</th></tr>"
             f"<tr><td>No hidden layer</td><td class='num'>{_pct(ab['no_hidden_layer'])}</td></tr>"
             f"<tr><td>One hidden layer (16 tanh units)</td><td class='num best'>{_pct(ab['one_hidden_layer'])}</td></tr>"
             f"</table></div>")

    # 3. gradient descent
    moons = {r["model"]: r for r in boards["moons"]["rows"]}
    s.append("<h2>3 · Gradient descent at work</h2><p class='lead'>Cross-entropy loss per epoch. "
             "The falling curve is the model learning; the plateau of logistic regression on moons is "
             "its linear limit.</p><div class='grid2'>")
    s.append("<div class='fig'>" + line_svg(
        {"logistic regression": moons["logistic_regression"]["loss_history"]},
        "Logistic regression loss — moons", "epoch", "loss") + "</div>")
    s.append("<div class='fig'>" + line_svg(
        {"neural network": moons["neural_network"]["loss_history"]},
        "Neural network loss — moons", "epoch", "loss") + "</div></div>")

    # 4. overfitting
    ds, ks = lab["depth_sweep"], lab["knn_sweep"]
    deep = ds[-1]
    s.append("<h2>4 · Bias, variance and overfitting</h2><p class='lead'>As a tree gets deeper, training "
             f"accuracy climbs to {_pct(deep['train_accuracy'])} — it can memorize every training point — "
             f"while test accuracy levels off at {_pct(deep['test_accuracy'])}; the gap between the two is "
             "what it memorized. k-NN shows the same trade-off from the other side: k = 1 memorizes the "
             "training set, and a very large k blurs the boundary until both scores drop.</p><div class='grid2'>")
    s.append("<div class='fig'>" + line_svg(
        {"train": [r["train_accuracy"] for r in ds], "test": [r["test_accuracy"] for r in ds]},
        "Decision tree: accuracy vs depth (moons)", "max depth", "accuracy",
        x_values=[r["depth"] for r in ds]) + "</div>")
    s.append("<div class='fig'>" + line_svg(
        {"train": [r["train_accuracy"] for r in ks], "test": [r["test_accuracy"] for r in ks]},
        "k-NN: accuracy vs k (moons)", "k neighbours", "accuracy",
        x_values=[r["k"] for r in ks]) + "</div></div>")

    # 5. backprop verified
    s.append("<h2>5 · Backpropagation, verified</h2><p class='lead'>The hand-derived gradients compared "
             "against a numerical derivative. A relative error below 1e-4 means the backward pass is correct.</p>"
             "<div class='card'><table><tr><th>Activation</th><th class='num'>Max relative error</th><th>Result</th></tr>")
    for g in lab["gradient_checks"]:
        res = "<span class='pass'>PASS</span>" if g["passed"] else "<span class='fail'>FAIL</span>"
        s.append(f"<tr><td>{g['activation']}</td><td class='num'>{g['max_relative_error']:.2e}</td><td>{res}</td></tr>")
    s.append("</table></div>")

    # 6. architectures
    vg = arch["vanishing"]
    s.append("<h2>6 · Inside the modern architectures</h2><p class='lead'>The defining computation of "
             "CNNs, LSTMs and Transformers, implemented by hand on small examples.</p><div class='grid2'>")
    s.append("<div class='card'><b>Convolution (CNN)</b><p class='sub'>An 8×8 image with one vertical "
             "edge, convolved with a Sobel kernel. The feature map lights up only at the edge.</p>"
             + _heat(arch["edges"], fmt="{:.0f}") + "</div>")
    s.append("<div class='card'><b>Self-attention (Transformer)</b><p class='sub'>Attention weights between "
             "four toy tokens (each row sums to 1). Similar tokens attend to each other: “the” and “a” put "
             "about 70% of their attention on that pair, and “cat” and “sat” do the same.</p>"
             + _heat(arch["attention"], labels=arch["tokens"]) + "</div>")
    s.append("<div class='fig'>" + line_svg(
        {"sigmoid": vg["sigmoid"], "tanh": vg["tanh"], "relu": vg["relu"]},
        "Vanishing gradient through 20 layers", "layers back", "gradient",
        x_values=list(range(1, 21)), y_min=0.0, y_max=1.0) + "</div>")
    lt = arch["lstm_trace"]
    s.append("<div class='card'><b>LSTM cell</b><p class='sub'>Four steps: two input pulses, then two "
             "empty steps. With the forget gate around 0.7, the cell keeps most of its memory each step "
             "instead of wiping it.</p><table><tr><th>t</th><th>input</th>"
             "<th>cell state c</th><th>forget gate</th></tr>")
    for r in lt:
        s.append(f"<tr><td>{r['t']}</td><td class='mono'>{r['x']}</td><td class='mono'>{r['c']}</td>"
                 f"<td class='mono'>{r['forget']}</td></tr>")
    s.append("</table></div></div>")

    # 7. production
    s.append("<h2>7 · From notebook to production</h2><p class='lead'>Every model is saved as a versioned, "
             "hashed artifact. A champion/challenger gate only promotes a model that beats the current "
             "champion on held-out data. A drift monitor compares live data against training data.</p>")
    s.append("<div class='card'><b>Champion/challenger gate</b><table><tr><th>Submitted</th><th>Version</th>"
             "<th class='num'>Test acc</th><th>Decision</th><th>Reason</th></tr>")
    for e in prod["events"]:
        dec = "<span class='pass'>PROMOTED</span>" if e["promoted"] else "<span class='warn'>KEPT CHAMPION</span>"
        s.append(f"<tr><td>{escape(pretty(e['model']))}</td><td class='mono'>{e['version']}</td>"
                 f"<td class='num'>{_pct(e['test_accuracy'])}</td><td>{dec}</td><td>{escape(e['reason'])}</td></tr>")
    s.append("</table></div>")
    s.append("<div class='card'><b>Registry after the run</b><p class='sub'>Every version records the code "
             "version, the data version (a fingerprint of the exact training rows) and its lifecycle stage.</p>"
             "<table><tr><th>Version</th><th>Model</th><th>Stage</th><th>Data version</th><th>Code</th></tr>")
    for v in prod["versions"]:
        s.append(f"<tr><td class='mono'>{v['version']}</td><td>{escape(pretty(v['model_type']))}</td>"
                 f"<td>{escape(v.get('stage', '-'))}</td><td class='mono'>{escape(v.get('data_version', '-'))}</td>"
                 f"<td class='mono'>{escape(v.get('code_version', '-'))}</td></tr>")
    s.append("</table></div>")
    for label, key in (("Live data from the same world", "drift_stable"),
                       ("Live data after the world moved", "drift_shifted")):
        d = prod[key]
        verdict_cls = "fail" if d["alert"] else ("warn" if d["max_psi"] >= 0.1 else "pass")
        s.append(f"<div class='card'><b>{label}</b> — <span class='{verdict_cls}'>{escape(d['verdict'])}</span>"
                 "<table><tr><th>Feature</th><th class='num'>PSI</th><th>Status</th>"
                 "<th class='num'>KS D</th><th class='num'>KS p-value</th></tr>")
        for f in d["features"]:
            s.append(f"<tr><td>x{f['feature']}</td><td class='num'>{f['psi']:.4f}</td><td>{f['status']}</td>"
                     f"<td class='num'>{f['ks']:.3f}</td><td class='num'>{f['ks_p']:.2e}</td></tr>")
        s.append("</table></div>")
    s.append(f"<div class='card'><b>Why models decay</b> — the champion "
             f"(<span class='mono'>{escape(prod['champion']['model_type'])} {prod['champion']['version']}</span>) "
             f"scores <span class='pass'>{_pct(prod['champion_acc_fresh'])}</span> on fresh data from the same "
             f"world and <span class='fail'>{_pct(prod['champion_acc_shifted'])}</span> after the shift. "
             f"Same model, same code — the world moved.</div>")

    if lab.get("modules"):
        s.append(render_modules(lab["modules"]))

    s.append("<footer>Every number and chart on this page was produced by the ModelForge lab run that "
             "generated it. Reproduce it with <span class='mono'>python -m modelforge lab</span>.</footer>")
    s.append("</main></body></html>")
    return "".join(s)


# --------------------------------------------------------------------------- #
# Sections 8-13: the course, module by module
# --------------------------------------------------------------------------- #
def _table(headers: list, rows: list, num_cols: tuple = ()) -> str:
    h = ["<table><tr>"] + [f"<th{' class=num' if i in num_cols else ''}>{escape(str(c))}</th>"
                           for i, c in enumerate(headers)] + ["</tr>"]
    for r in rows:
        h.append("<tr>" + "".join(f"<td{' class=num' if i in num_cols else ''}>{c}</td>"
                                  for i, c in enumerate(r)) + "</tr>")
    h.append("</table>")
    return "".join(h)


def _card(title: str, body: str, sub: str = "") -> str:
    sub_html = f"<p class='sub'>{sub}</p>" if sub else ""
    return f"<div class='card'><b>{escape(title)}</b>{sub_html}{body}</div>"


def _fig(svg: str) -> str:
    return f"<div class='fig'>{svg}</div>"


def _conf(v: float) -> str:
    return "&gt;99.99%" if v > 0.9999 else f"{v * 100:.2f}%"


def _examples(rows: list) -> str:
    return _table(["Lesson", "What the lesson says", "What the code computes"],
                  [[escape(r["lesson"]), escape(r["claim"]), f"<span class='mono'>{escape(r['computed'])}</span>"]
                   for r in rows])


def render_modules(m: dict) -> str:
    from .course_modules import MODULE_TITLES
    s = ["<h2 style='margin-top:48px'>Course coverage, module by module</h2><p class='lead'>Sections 1-7 "
         "are the core lab. Sections 8-13 run one experiment for each remaining idea in the course, in "
         "lesson order. The same output prints in the terminal with "
         "<span class='mono'>python -m modelforge module N</span>.</p>"]

    def head(n):
        s.append(f"<h2>{7 + n} · Module {n} — {escape(MODULE_TITLES[n])}</h2>")

    # ---- Module 1
    r = m[1]
    head(1)
    rule = r["rule"]
    s.append("<div class='grid2'>")
    s.append(_card("ML vs. AI — a rule vs. learned models", _table(
        ["Approach", "Test accuracy (moons)"],
        [[f"Rule: <span class='mono'>{escape(rule['rule'])}</span>", _pct(rule["rule_accuracy"])]] +
        [[escape(pretty(k)), _pct(v)] for k, v in rule["learned"].items()], (1,)),
        "The rule is fully readable; the learned models are more accurate but harder to explain."))
    n = r["label_noise"]
    s.append(_card("The ML Workflow — quality beats quantity", _table(
        ["Training labels", "Rows", "Test accuracy"],
        [["clean", 300, _pct(n["clean_300"])], ["15% wrong at random", 300, _pct(n["random15_300"])],
         ["15% wrong at random", 1200, _pct(n["random15_1200"])],
         ["15% wrong, all in one region", 300, _pct(n["biased_300"])],
         ["15% wrong, all in one region", 1200, _pct(n["biased_1200"])]], (1, 2)),
        "Decision tree on moons, clean test set. Four times more data partly averages out random label "
        "errors but barely touches systematic ones; cleaning the labels does."))
    p = r["paradigms"]
    s.append(_fig(line_svg({"inertia": [row["inertia"] for row in p["kmeans_elbow"]]},
                           "Unsupervised: k-means on blobs (labels hidden)", "k clusters", "inertia",
                           x_values=[row["k"] for row in p["kmeans_elbow"]], markers=True)))
    steps = p["rl_steps_per_episode"]
    smooth = [sum(steps[i:i + 10]) / len(steps[i:i + 10]) for i in range(0, len(steps), 10)]
    s.append(_fig(line_svg({"steps to goal (10-episode average)": smooth},
                           "Reinforcement: Q-learning in a 5x5 grid", "episode", "steps",
                           x_values=list(range(10, len(steps) + 1, 10)))))
    lc = r["learning_curve"]
    s.append(_fig(line_svg({"logistic regression": [x["logistic_regression"] for x in lc],
                            "k-NN": [x["knn"] for x in lc]},
                           "More data is not automatically better (moons)", "training rows", "test accuracy",
                           x_values=[x["n"] for x in lc], markers=True)))
    s.append(_card("Learned policy (Q-learning)", "<pre class='mono'>" + "\n".join(p["rl_policy"]) +
                   "</pre>", f"Greedy path: {p['rl_greedy_steps']} steps, the shortest possible "
                   f"({p['rl_optimal_steps']}). Nobody told the agent the right move; the only reward "
                   "was at the goal (G). # = wall."))
    s.append("</div>")

    # ---- Module 2
    r = m[2]
    head(2)
    h = r["houses"]
    s.append("<div class='grid2'>")
    s.append(_card("Worked examples from the lessons", _examples(r["examples"])))
    s.append(_card("House prices: regression on MSE", _table(
        ["Feature", "True weight", "Learned (gradient descent)", "Exact least squares"],
        [[f, f"{t:,.0f}", f"{a:,.1f}", f"{b:,.1f}"] for f, t, a, b in
         zip(h["features"], h["true"], h["learned"], h["exact_least_squares"])], (1, 2, 3)),
        f"Gradient descent reaches the exact least-squares answer (largest relative gap "
        f"{h['max_gap_to_exact']:.1e}); the remaining gap to the true weights is noise in the data. "
        f"Test RMSE ${h['test_rmse']:,.0f}. The lesson's house [1500, 3, 1990] is predicted at "
        f"${h['course_house_prediction']:,.0f} (lesson: $230,500)."))
    lrs = r["lr"]
    s.append(_fig(line_svg({f"lr {x['lr']}": x["loss_history"] for x in lrs},
                           "Learning rate: too small, about right, too large", "epoch", "MSE (log)",
                           log_y=True)))
    s.append(_card("Batch vs. mini-batch vs. stochastic", _table(
        ["Method", "Updates", "Final loss", "Uphill steps"],
        [[x["method"], x["updates"], f"{x['final_loss']:.4f}",
          f"{x['uphill_steps']} ({_pct(x['uphill_fraction'])})"] for x in r["flavours"]], (1, 2, 3)),
        "Same five passes over the data. Stochastic updates most often but half its steps go uphill — "
        "the zig-zag; mini-batch averages the noise away."))
    rd = r["ridge"]
    pick = {row["lambda"]: row for row in rd["rows"]}
    s.append(_fig(curve_fit_svg(rd["points"], rd["grid"],
                                {f"lambda {lam:g}": pick[lam]["curve"] for lam in (0.0, rd["best_lambda"], 10.0)},
                                rd["truth"], f"L2 as the bias-variance dial (degree-{rd['degree']} polynomial)")))
    s.append(_card("L2 regularization sweep", _table(
        ["lambda", "Train MSE", "Test MSE", "Largest |weight|"],
        [[f"{x['lambda']:g}", f"{x['train_mse']:.4f}", f"{x['test_mse']:.4f}", f"{x['max_abs_weight']:.2f}"]
         for x in rd["rows"]], (1, 2, 3)),
        "total loss = squared error + lambda x (sum of weights squared). The dashed line is the true curve."))
    es = r["early"]
    s.append(_fig(line_svg({"training loss": es["train_loss_history"], "validation loss": es["val_loss_history"]},
                           "Validation set: when to stop", "epoch", "cross-entropy")))
    s.append(_card("Early stopping", f"<p>Validation loss bottoms out at epoch {es['best_epoch']} "
                   f"({es['val_loss_at_best']:.3f}) and climbs to {es['val_loss_final']:.3f} while training loss "
                   f"keeps falling ({es['train_loss_at_best']:.3f} → {es['train_loss_final']:.3f}). With patience "
                   f"{es['patience']}, training stops at epoch {es['stop_epoch']} and restores epoch "
                   f"{es['best_epoch']}: same test accuracy ({_pct(es['early_stopped']['test_accuracy'])}), lower test "
                   f"loss ({es['full_run']['test_loss']:.3f} → {es['early_stopped']['test_loss']:.3f}).</p>"
                   + f"<p class='sub'>Tree depth chosen on validation data: {r['depth_val']['chosen_depth']} "
                   f"→ test {_pct(r['depth_val']['test_accuracy'])} (test set used once).</p>"))
    s.append(_card("VC dimension of a straight line", _table(
        ["Points", "Labelings", "Separable by a line", "Impossible"],
        [[k, v["labelings"], v["separable"], ", ".join(v["failed"]) or "none"] for k, v in r["vc"].items()],
        (1, 2)), "Some 3 points can take every labeling; 4 points in a square cannot (the XOR labelings)."))
    s.append("</div>")

    # ---- Module 3
    r = m[3]
    head(3)
    s.append("<div class='grid2'>")
    s.append(_card("The boundary is the weights",
                   f"<p class='mono'>{escape(r['boundary']['equation'])}</p>",
                   "Logistic regression on moons: the line where the two class scores tie."))
    s.append(_card("Worked examples from the lessons", _examples(r["examples"])))
    t = r["threshold"]
    s.append(_card("Accuracy lies on imbalanced data; thresholds trade precision for recall", _table(
        ["Threshold", "Accuracy", "Precision", "Recall", "F1", "TP", "FP", "FN"],
        [[escape(str(x["threshold"])) if isinstance(x["threshold"], str) else f"p ≥ {x['threshold']}",
          _pct(x["accuracy"]), f"{x['precision']:.3f}", f"{x['recall']:.3f}", f"{x['f1']:.3f}",
          x["tp"], x["fp"], x["fn"]] for x in t["rows"]], (1, 2, 3, 4, 5, 6, 7)),
        f"{t['positives_in_test']} positives in {t['n_test']} test rows. Saying 'no' to everything scores "
        "95% and catches nothing."))
    s.append(_card("Gini vs. entropy, and k-NN in high dimensions",
                   _table(["Dataset", "Gini", "Entropy"],
                          [[x["dataset"], _pct(x["gini"]), _pct(x["entropy"])] for x in r["gini_entropy"]], (1, 2))
                   + _table(["Dimensions", "Nearest / farthest distance"],
                            [[x["dims"], f"{x['nearest_over_farthest']:.3f}"] for x in r["curse"]], (0, 1)),
                   "Depth-5 trees. Near 1, the 'nearest' neighbour is barely nearer than any other point."))
    s.append("</div>")

    # ---- Module 4
    r = m[4]
    head(4)
    s.append("<div class='grid2'>")
    nr = r["neuron"]
    s.append(_card("A single layer is logistic regression",
                   f"<p>Logistic regression {_pct(nr['logistic_regression'])}; network with no hidden layer "
                   f"{_pct(nr['no_hidden_layer_net'])}; identical predictions on {_pct(nr['same_predictions'])} "
                   "of test points.</p>"))
    a = r["activations"]
    s.append(_card("Activation functions: slope decides gradient flow", _table(
        ["Activation"] + [f"f′({z:g})" for z in a["points"]] + [f"Moons, {a['epochs']} epochs"],
        [[x["activation"]] + [f"{x['slope'][z]:.3f}" for z in a["points"]] +
         [_pct(x["test_accuracy"])] for x in a["rows"]],
        tuple(range(1, len(a["points"]) + 2))),
        "A slope near zero chokes the gradient. Sigmoid learns slowest in the same budget; every activation "
        "passes the gradient check in section 5."))
    d = r["dying"]
    s.append(_card("Dying ReLU", _table(
        ["Activation", "Units with zero gradient", "Test accuracy"],
        [[k, f"{v['dead_after']}/16", _pct(v["test_accuracy"])] for k, v in d.items()], (1, 2)),
        "One huge update pushes every unit below zero. ReLU units then get zero gradient forever; "
        "Leaky ReLU's 0.01 slope lets them keep learning."))
    tr = r["transfer"]
    s.append(_card("Transfer learning and fine-tuning", _table(
        ["Labelled examples", "From scratch", "Fine-tuned", "Mismatched source"],
        [[x["n_labels"], _pct(x["from_scratch"]), _pct(x["fine_tuned"]), _pct(x["mismatched_source"])]
         for x in tr["rows"]], (0, 1, 2, 3)),
        f"{tr['dims']}-dimensional sensor data. Fine-tuning keeps the pretrained hidden layer and retrains the "
        "output layer. It wins with the fewest labels and stops mattering as labels grow; a source from a "
        "different 'sensor' helps less."))
    s.append("</div>")

    # ---- Module 5
    r = m[5]
    head(5)
    s.append("<div class='grid2'>")
    c = r["cnn"]
    cp = c["params"]
    s.append(_card("CNNs: weight sharing, translation, rotation", _table(
        ["Edge at column", "Feature map row", "2x2 max-pooled", "Global max"],
        [[col, f"<span class='mono'>{[int(v) for v in x['feature_row']]}</span>",
          f"<span class='mono'>{[int(v) for v in x['pooled_row']]}</span>", f"{x['global_max']:.0f}"]
         for col, x in c["shifted"].items()], (0, 3)),
        f"One 3x3 kernel reused everywhere: {cp['conv_weights']:,} weights for 64 filters vs "
        f"{cp['dense_weights']:,} for a dense layer on a 224×224 colour image. Rotate the image 90° and the "
        f"vertical-edge kernel responds {c['rotated_vertical_kernel']:.0f} — rotation invariance is not free."))
    g = r["gating"]
    rn = r["rnn"]
    s.append(_fig(line_svg({"plain tanh RNN": rn["tanh_rnn_curve"],
                            "LSTM, forget gate 0.73": g["lstm_cell_curves"][1.0],
                            "LSTM, gate 0.95": g["lstm_cell_curves"][3.0]},
                           "Gradient reaching earlier time steps", "steps back", "gradient (log)",
                           x_values=list(range(1, 51)), log_y=True)))
    cl = rn["clip"]
    s.append(_card("Vanishing, exploding, clipping, and the GRU",
                   f"<p>A recurrent factor of 0.5 for 20 steps leaves {rn['factor_0.5']:.2e} of the gradient; a "
                   f"factor of 2 multiplies it by {rn['factor_2.0']:,.0f}. Clipping scales "
                   f"<span class='mono'>{cl['before']}</span> down to <span class='mono'>{cl['after']}</span> "
                   f"(norm {cl['max_norm']:g}) without changing direction.</p><p>For 256 inputs and 256 hidden "
                   f"units a GRU has {g['params_256']['gru']:,} parameters against the LSTM's "
                   f"{g['params_256']['lstm']:,} ({_pct(g['gru_saving'])} fewer).</p>"))
    at = r["attention"]
    s.append(_card("Transformers: order needs positions; cost grows with the square",
                   f"<p>“{escape(at['sentences'][0])}” vs “{escape(at['sentences'][1])}”: the attention output for "
                   f"“dog” differs by {at['dog_difference_without_pe']:.3f} without positional encoding and "
                   f"{at['dog_difference_with_pe']:.3f} with it.</p>" + _table(
                       ["Sequence length", "Full attention scores", "Local attention (window 8)"],
                       [[x["n"], f"{x['full']:,}", f"{x['local_w8']:,}"] for x in at["cost"]], (0, 1, 2))))
    s.append("</div>")

    # ---- Module 6
    r = m[6]
    head(6)
    s.append("<div class='grid2'>")
    lc = r["lifecycle"]
    val = r["validation"]
    s.append(_card("Versioning, data validation, quality gates, rollback",
                   _table(["Batch", "Validation result"],
                          [[k.replace("_", " "), escape("; ".join(v) or "OK")] for k, v in val.items()
                           if k != "schema_ranges"])
                   + _table(["Submitted", "Test acc", "Decision"],
                            [[escape(e["submitted"]), _pct(e["test_accuracy"]),
                              ("PROMOTED — " if e["promoted"] else "REFUSED — ") + escape(e["reason"])]
                             for e in lc["events"]], (1,))
                   + f"<p class='sub'>After rollback the decision tree is back in production; the network is "
                     f"archived. Data version {escape(r['versioning']['data_version'])} becomes "
                     f"{escape(r['versioning']['after_one_edit'])} if one value changes.</p>"))
    cn = r["canary"]
    rows = []
    for label, key in (("correct", "good"), ("hidden bug", "skewed")):
        for st in cn[key]["stages"]:
            rows.append([label, f"{int(st['share'] * 100)}%", st["challenger_requests"],
                         _pct(st["challenger_error"]), _pct(st["champion_error"]),
                         "ok" if st["ok"] else "<span class='fail'>ROLL BACK</span>"])
    sh = cn["shadow"]
    s.append(_card("Canary and shadow deployment", _table(
        ["Serving path", "Traffic", "Requests", "New model error", "Old model error", "Decision"], rows, (2, 3, 4)),
        f"The buggy path was caught on {cn['skewed']['requests_exposed']} of "
        f"{cn['skewed']['total_requests']} requests. Shadow: {sh['requests']} requests, agreement "
        f"{_pct(sh['agreement'])}, shadow accuracy {_pct(sh['shadow_accuracy'])} vs champion "
        f"{_pct(sh['champion_accuracy'])}, zero answers shown to users."))
    fs = r["feature_store"]
    s.append(_card("Training-serving skew and the feature store", _table(
        ["Serving path", "Values matching training", "Decisions changed", "Accuracy"],
        [["feature store (one definition)", f"{fs['store_rows_matching_training_code']}/{fs['n_test']}", 0,
          _pct(fs["accuracy_feature_store"])],
         ["re-implemented window (includes today)", f"{fs['rewrite_rows_matching_training_code']}/{fs['n_test']}",
          fs["predictions_changed_by_rewrite"], _pct(fs["accuracy_rewrite"])]], (1, 2, 3)),
        "The lesson's example: 'average purchase over the last 30 days', computed twice. Nothing crashes."))
    dr = r["drift"]
    s.append(_card("Data drift vs. concept drift, and what the monitor does", _table(
        ["Scenario", "Max PSI", "KS p", "Live accuracy", "Action"],
        [[escape(k), f"{v['max_psi']:.3f}", f"{v['min_ks_p']:.1e}", _pct(v["live_accuracy"]),
          f"<b>{v['action']}</b>"] for k, v in dr["scenarios"].items()], (1, 2, 3)),
        f"PSI and KS cannot see concept drift — the inputs did not move — so a drop in labelled accuracy "
        f"routes it to a human. Retraining after data drift: {_pct(dr['old_model_on_moved_world'])} → "
        f"{_pct(dr['retrained_on_moved_world'])}."))
    fr = r["frontier"]
    s.append(_fig(line_svg({"targeted (FGSM)": [x["adversarial_accuracy"] for x in fr["adversarial"]],
                            "random noise": [x["random_noise_accuracy"] for x in fr["adversarial"]]},
                           "Adversarial robustness: same-size perturbations", "epsilon", "accuracy",
                           x_values=[x["eps"] for x in fr["adversarial"]], markers=True)))
    s.append(_card("Knowing when it doesn't know", _table(
        ["Input (scaled)", "Network confidence", "Distance to training data", "Guard"],
        [[f"<span class='mono'>{[round(v, 2) for v in x['x']]}</span>", _conf(x["confidence"]),
          f"{x['distance']:.2f}", "<span class='warn'>UNKNOWN</span>" if x["flagged_unknown"] else "ok"]
         for x in fr["overconfidence"]], (1, 2)),
        f"The network is just as sure far outside its data. The out-of-distribution guard (threshold "
        f"{fr['guard_threshold']:.2f}) is what lets the system say “I don't know”; "
        "<span class='mono'>predict</span> uses it."))
    it = fr["interpretability"]
    fg = r["forgetting"]
    s.append(_card("Interpretability and catastrophic forgetting",
                   _table(["Model"] + it["features"],
                          [[escape(pretty(k))] + [f"{v:+.3f}" for v in x["importance"]]
                           for k, x in it["models"].items()], (1, 2, 3))
                   + _table(["Network", "Task A (moons)", "Task B (circles)"],
                            [["after training on A", _pct(fg["after_A"]["task_A"]), _pct(fg["after_A"]["task_B"])],
                             ["then only on B", _pct(fg["after_B"]["task_A"]), _pct(fg["after_B"]["task_B"])],
                             ["on both together", _pct(fg["joint"]["task_A"]), _pct(fg["joint"]["task_B"])]],
                            (1, 2)),
                   "Permutation importance: every model ignores the pure-noise column but they split credit "
                   "between x0 and x1 differently. Training only on B erases A, even though one network can hold both."))
    s.append("</div>")
    return "".join(s)

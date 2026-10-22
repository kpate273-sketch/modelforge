"""Plots drawn from scratch as SVG — no matplotlib (Module 3: The Decision
Boundary Mindset; Module 2: Gradient Descent, Overfitting).

  boundary_svg   classifies a fine grid of points with a trained model and
                 shades each cell by the predicted class, then overlays the real
                 data. It makes the Decision Boundary lesson literal: you can
                 see logistic regression draw a straight line through the moons
                 and the neural network bend around them.
  line_svg       a small line chart, used for loss curves (gradient descent
                 converging) and for train-vs-test overfitting curves.

Output is plain SVG text, so the plots open in any browser and embed directly
in the HTML report.
"""

from __future__ import annotations

from html import escape

CLASS_COLORS = ["#3b6fd4", "#d9483b", "#2e9e5b", "#a05bd6"]


def _tint(hex_color: str, amount: float = 0.22) -> str:
    """Blend a color toward white. Opaque tints avoid the seams that overlapping
    translucent cells leave in the shaded regions."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: round(255 - (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def _bounds(X: list, pad: float = 0.4) -> tuple:
    xs, ys = [r[0] for r in X], [r[1] for r in X]
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def boundary_svg(model, X: list, y: list, title: str = "", size: int = 240,
                 grid: int = 48, bounds: tuple | None = None) -> str:
    x0, x1, y0, y1 = bounds or _bounds(X)
    cell = size / grid
    pts = []
    for gy in range(grid):
        for gx in range(grid):
            px = x0 + (gx + 0.5) / grid * (x1 - x0)
            py = y1 - (gy + 0.5) / grid * (y1 - y0)
            pts.append([px, py])
    preds = model.predict(pts)
    classes = sorted(set(y) | set(preds))
    color = {c: CLASS_COLORS[i % len(CLASS_COLORS)] for i, c in enumerate(classes)}

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size + 22}" '
             f'viewBox="0 0 {size} {size + 22}" font-family="Helvetica, Arial, sans-serif">']
    parts.append(f'<text x="{size / 2}" y="15" font-size="14" font-weight="bold" '
                 f'text-anchor="middle" fill="currentColor">{escape(title)}</text>')
    parts.append('<g transform="translate(0,22)"><g shape-rendering="crispEdges">')
    tint = {c: _tint(col) for c, col in color.items()}
    i = 0
    for gy in range(grid):
        for gx in range(grid):
            parts.append(f'<rect x="{gx * cell:.2f}" y="{gy * cell:.2f}" '
                         f'width="{cell + 0.5:.2f}" height="{cell + 0.5:.2f}" '
                         f'fill="{tint[preds[i]]}"/>')
            i += 1
    parts.append("</g>")
    for (px, py), label in zip(X, y):
        sx = (px - x0) / (x1 - x0) * size
        sy = (y1 - py) / (y1 - y0) * size
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.6" fill="{color[label]}" '
                     f'stroke="#ffffff" stroke-width="0.6"/>')
    parts.append(f'<rect x="0" y="0" width="{size}" height="{size}" fill="none" '
                 f'stroke="#8a8f9c" stroke-width="1"/>')
    parts.append("</g></svg>")
    return "".join(parts)


def _legend_width(name: str) -> float:
    return 22 + 5.6 * len(name)          # swatch + label, Helvetica 10px is ~5.6 px/char


def _tick_label(v: float, span: float) -> str:
    if span >= 100:
        return f"{v:,.0f}"
    if span >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def line_svg(series: dict, title: str = "", x_label: str = "", y_label: str = "",
             x_values: list | None = None, width: int = 420, height: int = 230,
             y_min: float | None = None, y_max: float | None = None,
             log_y: bool = False, markers: bool = False) -> str:
    """series = {name: [values...]} drawn as polylines sharing one x axis.

    log_y plots log10 of the values (for gradients and losses that span many
    orders of magnitude); ticks are then placed on whole powers of ten."""
    import math as _m
    palette = ["#3b6fd4", "#d9483b", "#2e9e5b", "#a05bd6"]
    if log_y:
        series = {k: [_m.log10(max(abs(v), 1e-300)) for v in vals] for k, vals in series.items()}
        y_min = None if y_min is None else _m.log10(y_min)
        y_max = None if y_max is None else _m.log10(y_max)
    allv = [v for vals in series.values() for v in vals]
    lo = min(allv) if y_min is None else y_min
    hi = max(allv) if y_max is None else y_max
    if log_y:
        lo, hi = _m.floor(lo), _m.ceil(hi)
    if hi == lo:
        hi = lo + 1
    if log_y:
        step = max(1, _m.ceil((hi - lo) / 5))
        ticks_y = list(range(int(hi), int(lo) - 1, -step))
        labels_y = [f"1e{t}" for t in ticks_y]
    else:
        ticks_y = [lo + (hi - lo) * k / 4 for k in range(5)]
        labels_y = [_tick_label(v, hi - lo) for v in ticks_y]
    left = max(46, 22 + 6 * max(len(t) for t in labels_y))
    right, top, bottom = 14, 44, 36
    pw, ph = width - left - right, height - top - bottom
    n = max(len(v) for v in series.values())
    xs = x_values or list(range(n))

    def X(i):
        return left + (i / max(n - 1, 1)) * pw

    def Y(v):
        return top + (1 - (v - lo) / (hi - lo)) * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" '
         f'font-size="10">']
    p.append(f'<text x="{width / 2}" y="16" font-size="12" font-weight="bold" '
             f'text-anchor="middle" fill="currentColor">{escape(title)}</text>')
    for v, label in zip(ticks_y, labels_y):
        yy = Y(v)
        p.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + pw}" y2="{yy:.1f}" '
                 f'stroke="#8a8f9c" stroke-opacity="0.25"/>')
        p.append(f'<text x="{left - 6}" y="{yy + 3:.1f}" text-anchor="end" '
                 f'fill="currentColor" fill-opacity="0.7">{label}</text>')
    if n <= 12:
        ticks = list(range(n))            # short sweeps: label every point
    else:
        ticks = sorted(set([0, n - 1] + [round((n - 1) * q / 4) for q in range(1, 4)]))
    for i in ticks:
        p.append(f'<text x="{X(i):.1f}" y="{top + ph + 14}" text-anchor="middle" '
                 f'fill="currentColor" fill-opacity="0.7">{xs[i]}</text>')
    p.append(f'<text x="{left + pw / 2}" y="{height - 4}" text-anchor="middle" '
             f'fill="currentColor" fill-opacity="0.8">{escape(x_label)}</text>')
    p.append(f'<text x="12" y="{top + ph / 2}" text-anchor="middle" fill="currentColor" '
             f'fill-opacity="0.8" transform="rotate(-90 12 {top + ph / 2})">{escape(y_label)}</text>')
    lx = left
    for idx, (name, vals) in enumerate(series.items()):
        col = palette[idx % len(palette)]
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        if markers:
            for i, v in enumerate(vals):
                p.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.5" fill="{col}"/>')
        # legend sits above the plot, never on the lines; entries laid out by text width
        p.append(f'<rect x="{lx:.1f}" y="{top - 13}" width="10" height="3" fill="{col}"/>')
        p.append(f'<text x="{lx + 14:.1f}" y="{top - 8}" fill="currentColor">{escape(name)}</text>')
        lx += _legend_width(name)
    p.append("</svg>")
    return "".join(p)


def curve_fit_svg(points: list, grid: list, curves: dict, truth: list, title: str = "",
                  width: int = 420, height: int = 230, y_lim: tuple = (-2.0, 2.0)) -> str:
    """Scatter of training points plus fitted curves (for the L2 sweep)."""
    palette = ["#d9483b", "#3b6fd4", "#2e9e5b", "#a05bd6"]
    left, right, top, bottom = 34, 12, 44, 26
    pw, ph = width - left - right, height - top - bottom
    x0, x1 = min(grid), max(grid)
    lo, hi = y_lim
    X = lambda x: left + (x - x0) / (x1 - x0) * pw
    Y = lambda v: top + (1 - (min(max(v, lo), hi) - lo) / (hi - lo)) * ph
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" font-size="10">']
    p.append(f'<text x="{width / 2}" y="16" font-size="12" font-weight="bold" '
             f'text-anchor="middle" fill="currentColor">{escape(title)}</text>')
    p.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" '
             f'stroke="#8a8f9c" stroke-opacity="0.5"/>')
    pts = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(grid, truth))
    p.append(f'<polyline points="{pts}" fill="none" stroke="#8a8f9c" stroke-width="1.5" '
             f'stroke-dasharray="4 3"/>')
    for idx, (name, vals) in enumerate(curves.items()):
        col = palette[idx % len(palette)]
        pts = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(grid, vals))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        lx = left + sum(_legend_width(k) for k in list(curves)[:idx])
        p.append(f'<rect x="{lx:.1f}" y="{top - 13}" width="10" height="3" fill="{col}"/>')
        p.append(f'<text x="{lx + 14:.1f}" y="{top - 8}" fill="currentColor">{escape(name)}</text>')
    for x, v in points:
        p.append(f'<circle cx="{X(x):.1f}" cy="{Y(v):.1f}" r="3" fill="currentColor"/>')
    p.append("</svg>")
    return "".join(p)

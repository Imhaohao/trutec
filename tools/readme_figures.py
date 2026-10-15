"""Build the README figures in docs/readme/ from measured results.

SVG charts carry an explicit white background so they read in GitHub dark mode.
Numbers come from artifacts/validation-*/official-metrics.json and the committed
cross-validation records under research_v5/synthw14/.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "readme"
CV = ROOT / "research_v5" / "synthw14"

INK = "#2b2b2a"
MUTED = "#85837b"
GRID = "#e4e2da"
ACCENT = "#0f6e56"
WARM = "#c2562d"
COOL = "#3b6fb6"
FONT = "font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
VERSIONS = ("v2", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14")


def svg(width: int, height: int, body: list[str]) -> str:
    head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="{FONT}">'
    return "\n".join([head, f'<rect width="{width}" height="{height}" fill="#ffffff"/>', *body, "</svg>"])


def text(x: float, y: float, value: str, size: int = 13, fill: str = INK, anchor: str = "start", weight: int = 400) -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{value}</text>'


def box(x, y, w, h, title, subtitle, stroke=INK, fill="#ffffff") -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>',
        text(x + w / 2, y + h / 2 - 3, title, 14, INK, "middle", 600),
        text(x + w / 2, y + h / 2 + 15, subtitle, 12, MUTED, "middle"),
    ]


def arrow(x1, y1, x2, y2, colour=INK) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="1.4" marker-end="url(#a)"/>'
    )


MARKER = (
    '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
    f'<path d="M1 1L9 5L1 9" fill="none" stroke="{INK}" stroke-width="1.5"/></marker></defs>'
)


def pipeline() -> str:
    steps = [
        ("Drawing + query", "label-free input"),
        ("Clean query", "drop text, stray lines"),
        ("Stage 1", "texture score per spot"),
        ("Stage 2", "judge cell from context"),
        ("Mask", "colour gate, threshold .25"),
    ]
    body = [MARKER]
    for i, (title, sub) in enumerate(steps):
        x = 20 + i * 184
        body += box(x, 30, 160, 64, title, sub, ACCENT if i in (1, 2, 3) else INK)
        if i:
            body.append(arrow(x - 22, 62, x - 2, 62))
    return svg(940, 124, body)


def crossfit() -> str:
    body = [MARKER, text(20, 26, "Stage 2 learns from stage-1 scores on drawings stage 1 never saw", 14, INK, weight=600)]
    for row in range(4):
        y = 50 + row * 34
        body.append(text(20, y + 17, f"Fold {row + 1} held out", 12, MUTED))
        for col in range(4):
            held = col == row
            colour = WARM if held else "#d9e9e2"
            body.append(f'<rect x="{150 + col * 92}" y="{y}" width="86" height="24" fill="{colour}"/>')
            label = "scored" if held else "train stage 1"
            body.append(text(193 + col * 92, y + 16, label, 11, "#ffffff" if held else ACCENT, "middle"))
    body.append(arrow(530, 116, 590, 116))
    body += box(596, 84, 200, 64, "Stage-2 training cells", "honest scores, 1.08M rows", ACCENT)
    return svg(820, 196, body)


def load_summary(version: str) -> dict:
    path = ROOT / "artifacts" / f"validation-{version}" / "official-metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def progress() -> str:
    series = {"IoU": ("document_macro_iou", ACCENT), "Precision": ("global_precision", WARM), "Recall": ("global_recall", COOL)}
    data = {v: load_summary(v) for v in VERSIONS}
    left, top, width, height = 60, 30, 720, 260
    body = []
    for tick in (0.3, 0.5, 0.7, 0.9):
        y = top + height * (1 - (tick - 0.2) / 0.85)
        body += [f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" stroke="{GRID}"/>', text(left - 8, y + 4, f"{tick:.1f}", 11, MUTED, "end")]
    step = width / (len(VERSIONS) - 1)
    for i, version in enumerate(VERSIONS):
        body.append(text(left + i * step, top + height + 20, version, 12, INK if version == "v14" else MUTED, "middle", 600 if version == "v14" else 400))
    for name, (key, colour) in series.items():
        points = [(left + i * step, top + height * (1 - (data[v][key] - 0.2) / 0.85)) for i, v in enumerate(VERSIONS)]
        body.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" fill="none" stroke="{colour}" stroke-width="2.4"/>')
        body += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colour}"/>' for x, y in points]
        body.append(text(points[-1][0] + 10, points[-1][1] + 4, f"{name} {data['v14'][key]:.3f}", 12, colour, weight=600))
    return svg(900, 330, body)


def bars(title: str, rows: list[tuple[str, float, bool]], low: float, high: float, fmt: str = "{:.3f}") -> str:
    body = [text(20, 24, title, 14, INK, weight=600)]
    for i, (label, value, chosen) in enumerate(rows):
        y = 44 + i * 32
        length = 380 * (value - low) / (high - low)
        colour = ACCENT if chosen else "#b9c9c2"
        body += [
            text(20, y + 16, label, 12, INK if chosen else MUTED, weight=600 if chosen else 400),
            f'<rect x="230" y="{y}" width="{max(length, 2):.1f}" height="22" fill="{colour}"/>',
            text(236 + length, y + 16, fmt.format(value), 12, INK),
        ]
    return svg(700, 60 + 32 * len(rows), body)


def capacity() -> str:
    rows = json.loads((CV / "cv_capacity12.json").read_text(encoding="utf-8"))["results"]
    labels = {"recipe": "200 rounds, 31 leaves (v12)", "iter600_leaf63": "600 rounds, 63 leaves (v14)", "iter800_leaf127_lr05": "800 rounds, 127 leaves"}
    data = [(labels[r["config"]["name"]], r["macro_iou"], r["config"]["name"] == "iter600_leaf63") for r in rows]
    return bars("Bigger stage 2: cross-validated IoU", data, 0.70, 0.80)


def synthetic() -> str:
    rows = json.loads((CV / "cv_results.json").read_text(encoding="utf-8"))["results"]
    data = [(f"synthetic weight {r['synthetic_weight']:g}", r["precision"], r["synthetic_weight"] == 0) for r in rows]
    return bars("Synthetic trap sheets: cross-validated precision", data, 0.60, 0.72)


def cad() -> str:
    rows = json.loads((CV / "cv_cadweight.json").read_text(encoding="utf-8"))["results"]
    data = [(f"CAD weight {r['cad_weight']:g}", r["macro_iou"], r["cad_weight"] == 1) for r in rows]
    return bars("Organizer CAD drawings: real-drawing IoU", data, 0.76, 0.83)


def per_document() -> str:
    data = [("11 regular documents", 0.9667, True), ("2 dense multi-pattern sheets", 0.6102, False)]
    return bars("v14 precision on public validation", data, 0.0, 1.0)


def crop_cleaning() -> None:
    source = ROOT / "research_v5" / "diagnose7" / "figures" / "q-e4b8a4efcb903bf5.png"
    with Image.open(source) as image:
        image.crop((2440, 140, 2830, 355)).save(OUT / "query-cleaning.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    charts = {
        "pipeline.svg": pipeline,
        "crossfit.svg": crossfit,
        "progress.svg": progress,
        "capacity.svg": capacity,
        "synthetic.svg": synthetic,
        "cad.svg": cad,
        "per-document.svg": per_document,
    }
    for name, build in charts.items():
        (OUT / name).write_text(build(), encoding="utf-8")
    crop_cleaning()
    print("\n".join(sorted(path.name for path in OUT.iterdir())))


if __name__ == "__main__":
    main()

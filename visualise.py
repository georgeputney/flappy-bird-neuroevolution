import json
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_LABELS = ["BIRD_Y", "VEL", "DIST_X", "GAP_ERR", "GAP_SIZE"]

_ROOT = Path(__file__).parent

# palette 
BG          = "#f7f6f3"
INK_30      = (14/255, 14/255, 13/255, 0.30)   # --ink-30, muted warm near-black
NODE_STROKE = (14/255, 14/255, 13/255, 0.18)
INPUT_FACE  = (0.0, 0.0, 0.0, 0.0)
HIDDEN_FACE = (0.78, 0.71, 0.59, 0.22)
OUTPUT_FACE = "#c8401a"
POS_EDGE    = (0.86, 0.84, 0.78, 0.70)
NEG_EDGE    = (0.78, 0.25, 0.10, 0.50)
DIS_EDGE    = (0.55, 0.55, 0.55, 0.10)


def load(path: Path | None = None) -> dict:
    
    # load genome JSON from disk, defaulting to best_genome.json in the project root
    return json.loads((path or _ROOT / "best_genome.json").read_text())


def layout(nodes: list[dict]) -> dict[int, tuple[float, float]]:
    """
    Compute (x, y) positions for each node, arranged in input, hidden, output columns.

    Nodes within each column are spaced evenly along the vertical axis. Positions
    are expressed in normalised coordinates between 0 and 1.

    Args:
    - nodes (list[dict]): All nodes in the genome, each with an "id" and "type".

    Returns:
    - dict[int, tuple[float, float]]: A mapping from node id to (x, y) position.
    """

    inputs  = [n for n in nodes if n["type"] == "input"]
    hidden  = [n for n in nodes if n["type"] == "hidden"]
    outputs = [n for n in nodes if n["type"] == "output"]

    pos: dict[int, tuple[float, float]] = {}

    def column(group, x):
        # space nodes evenly within their column
        n = len(group)
        for i, node in enumerate(group):
            pos[node["id"]] = (x, (i + 1) / (n + 1))

    column(inputs,  0.15)
    column(hidden,  0.50)
    column(outputs, 0.80)   # shifted left to reduce dead space right
    return pos


def _edge_point(x0, y0, x1, y1, r):
    """
    Return the point on the circumference of a circle at (x1, y1) facing toward (x0, y0).

    Used to terminate connection lines at node boundaries rather than centres,
    keeping arrowheads visually clean.

    Args:
    - x0, y0 (float): The source point the circumference point should face.
    - x1, y1 (float): The centre of the circle.
    - r (float): The radius of the circle.

    Returns:
    - tuple[float, float]: The (x, y) point on the circle's edge.
    """

    dx, dy = x1 - x0, y1 - y0
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist < 1e-9:
        return x1, y1
    return x1 - r * dx / dist, y1 - r * dy / dist


def draw(genome: dict, out_path: Path | None = None):
    """
    Render a genome as a network diagram and save it as a PNG.

    Nodes are drawn as circles in three columns: inputs on the left, hidden in the
    middle, and the output on the right. Connection thickness encodes weight magnitude;
    colour distinguishes excitatory (warm grey) from inhibitory (terracotta) connections.
    Disabled connections are shown faintly.

    Args:
    - genome (dict): A parsed genome dict with "nodes", "conns", and optional "fitness".
    - out_path (Path | None): Destination path for the PNG. Defaults to best_genome.png
      in the project root.
    """

    nodes   = genome["nodes"]
    conns   = genome["conns"]
    fitness = genome.get("fitness", "?")

    pos = layout(nodes)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0.0,   1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # connections
    r = 0.038
    enabled  = [c for c in conns if     c.get("on", True) and c["src"] in pos and c["dst"] in pos]
    disabled = [c for c in conns if not c.get("on", True) and c["src"] in pos and c["dst"] in pos]
    max_w    = max((abs(c["w"]) for c in enabled), default=1.0)

    for c in disabled:
        x0, y0 = pos[c["src"]]
        x1, y1 = _edge_point(x0, y0, *pos[c["dst"]], r)
        sx, sy  = _edge_point(x1, y1, x0, y0, r)
        ax.annotate("", xy=(x1, y1), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-", color=DIS_EDGE, lw=0.5,
                            connectionstyle="arc3,rad=0.06"), zorder=1)

    for c in enabled:
        x0, y0 = pos[c["src"]]
        x1, y1 = _edge_point(x0, y0, *pos[c["dst"]], r)
        sx, sy  = _edge_point(x1, y1, x0, y0, r)
        t       = abs(c["w"]) / max(max_w, 1e-6)
        colour  = POS_EDGE if c["w"] >= 0 else NEG_EDGE
        lw      = 0.5 + 3.5 * t
        ax.annotate("", xy=(x1, y1), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-", color=colour, lw=lw,
                            connectionstyle="arc3,rad=0.06"), zorder=1)

    # nodes
    label_gap = 0.055
    input_ids = [n["id"] for n in nodes if n["type"] == "input"]

    for node in nodes:
        nid = node["id"]
        if nid not in pos:
            continue
        x, y = pos[nid]
        t = node["type"]

        face = INPUT_FACE if t == "input" else (HIDDEN_FACE if t == "hidden" else OUTPUT_FACE)
        edge = OUTPUT_FACE if t == "output" else NODE_STROKE
        lw   = 0.0 if t == "output" else 0.8

        circle = plt.Circle((x, y), r, facecolor=face, edgecolor=edge,
                             linewidth=lw, zorder=3)
        ax.add_patch(circle)

        if t == "input":
            idx   = input_ids.index(nid)
            label = INPUT_LABELS[idx] if idx < len(INPUT_LABELS) else str(nid)
            ax.text(x - r - label_gap, y, label,
                    fontsize=7, ha="right", va="center",
                    color=INK_30, fontfamily="monospace")
        elif t == "output":
            ax.text(x + r + label_gap, y, "FLAP",
                    fontsize=7, ha="left", va="center",
                    color=OUTPUT_FACE, fontfamily="monospace")
        elif t == "hidden":
            ax.text(x, y, "H", fontsize=6, ha="center", va="center",
                    color=INK_30, fontfamily="monospace", zorder=4)

    # caption 
    n_enabled = len(enabled)
    n_hidden  = sum(1 for n in nodes if n["type"] == "hidden")
    caption   = (
        f"BEST GENOME  ·  FITNESS {fitness:.0f}  ·  "
        f"{n_hidden} HIDDEN NODE{'S' if n_hidden != 1 else ''}  ·  "
        f"{n_enabled} CONNECTIONS"
    )
    ax.text(0.0, -0.04, caption, transform=ax.transAxes,
            fontsize=6.5, color=INK_30, fontfamily="monospace",
            ha="left", va="top")

    out_path = out_path or _ROOT / "best_genome.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    draw(load())

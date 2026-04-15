import copy, math, random
from dataclasses import dataclass, field
from config import NUM_INPUTS

_innov = 0


def _new_innov() -> int:
    
    # increment and return the global innovation counter
    global _innov
    _innov += 1
    return _innov


@dataclass
class Genome:
    """
    A single NEAT genome encoding a neural network as a graph of nodes and connections.

    Nodes are typed as input, hidden, or output. Connections carry a weight and can be
    enabled or disabled. Fitness is assigned externally after evaluation.
    """

    nodes: list = field(default_factory=list)  # {"id", "type"}
    conns: list = field(default_factory=list)  # {"innov", "src", "dst", "w", "on"}
    fitness: float = 0.0

    def activate(self, inputs: list[float]) -> float:
        """
        Run a forward pass through the network and return the output activation.

        Iterates over connections until values stabilise, which handles any DAG
        topology produced by NEAT mutations. The output is squashed through a sigmoid.

        Args:
        - inputs (list[float]): Normalised sensor values, one per input node.

        Returns:
        - float: The sigmoid-activated output value in the range (0, 1).
        """

        vals: dict[int, float] = {}

        # seed input node values
        for i, v in enumerate(inputs):
            vals[i] = v

        # propagate values through enabled connections until stable
        for _ in range(len(self.nodes) + 1):
            for c in self.conns:
                if c["on"] and c["src"] in vals:
                    vals[c["dst"]] = vals.get(c["dst"], 0.0) + vals[c["src"]] * c["w"]

        out_id = next(n["id"] for n in self.nodes if n["type"] == "output")
        raw = vals.get(out_id, 0.0)
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, raw))))


def make_genome() -> Genome:
    """
    Create a minimal genome with input and output nodes fully connected.

    Each input node is connected directly to the single output node with a random
    Gaussian weight. No hidden nodes are created; NEAT grows structure from here.

    Returns:
    - Genome: A new minimal genome ready for evaluation.
    """

    g = Genome()

    # create one input node per sensor and a single output node
    for i in range(NUM_INPUTS):
        g.nodes.append({"id": i, "type": "input"})
    g.nodes.append({"id": NUM_INPUTS, "type": "output"})

    # connect every input directly to the output with a random weight
    for i in range(NUM_INPUTS):
        g.conns.append({"innov": _new_innov(), "src": i, "dst": NUM_INPUTS,
                        "w": random.gauss(0, 1), "on": True})
    return g


def mutate(g: Genome) -> Genome:
    """
    Apply structural and weight mutations to a copy of the genome.

    Three mutation types are applied independently:
    - Weight perturbation or reset on existing connections.
    - Add a new connection between two previously unconnected nodes.
    - Add a new hidden node by splitting an existing connection.

    Args:
    - g (Genome): The genome to mutate.

    Returns:
    - Genome: A new mutated genome; the original is not modified.
    """

    g = copy.deepcopy(g)

    # nudge or reset each connection weight independently
    for c in g.conns:
        if random.random() < 0.8:
            c["w"] += random.gauss(0, 0.1)
        elif random.random() < 0.1:
            c["w"] = random.gauss(0, 1)

    # add a new connection between two nodes that aren't already connected
    if random.random() < 0.3:
        src_ids = [n["id"] for n in g.nodes]
        dst_ids = [n["id"] for n in g.nodes if n["type"] != "input"]
        src, dst = random.choice(src_ids), random.choice(dst_ids)
        if src != dst and not any(c["src"] == src and c["dst"] == dst for c in g.conns):
            g.conns.append({"innov": _new_innov(), "src": src, "dst": dst,
                            "w": random.gauss(0, 1), "on": True})

    # add a hidden node by splitting a randomly chosen enabled connection
    live = [c for c in g.conns if c["on"]]
    if random.random() < 0.15 and live:
        c = random.choice(live)
        c["on"] = False
        new_id = max(n["id"] for n in g.nodes) + 1
        g.nodes.append({"id": new_id, "type": "hidden"})
        g.conns.append({"innov": _new_innov(), "src": c["src"], "dst": new_id, "w": 1.0, "on": True})
        g.conns.append({"innov": _new_innov(), "src": new_id, "dst": c["dst"], "w": c["w"], "on": True})

    return g


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    """
    Produce a child genome by crossing over two parent genomes.

    Matching genes (shared innovation numbers) are inherited randomly from either
    parent. Disjoint and excess genes are inherited from the fitter parent (parent_a).

    Args:
    - parent_a (Genome): The fitter parent; supplies all non-matching genes.
    - parent_b (Genome): The weaker parent; contributes only to matching genes.

    Returns:
    - Genome: A new child genome combining structure from both parents.
    """

    child = Genome(nodes=list(parent_a.nodes))

    # index parent_b connections by innovation number for fast lookup
    b_by_innov = {c["innov"]: c for c in parent_b.conns}

    # for each gene in parent_a, randomly pick from parent_b if a match exists
    for c in parent_a.conns:
        gene = b_by_innov.get(c["innov"])
        child.conns.append(dict(gene if gene and random.random() < 0.5 else c))
    return child


def distance(a: Genome, b: Genome) -> float:
    """
    Compute the compatibility distance between two genomes.

    Distance is a weighted sum of the number of disjoint genes and the average
    weight difference across shared genes. Used by speciation to group similar genomes.

    Args:
    - a (Genome): The first genome.
    - b (Genome): The second genome.

    Returns:
    - float: A non-negative compatibility distance; lower means more similar.
    """

    ai = {c["innov"]: c for c in a.conns}
    bi = {c["innov"]: c for c in b.conns}
    shared = set(ai) & set(bi)
    disjoint = len(set(ai) ^ set(bi))

    # average weight difference across genes present in both genomes
    w_diff = sum(abs(ai[i]["w"] - bi[i]["w"]) for i in shared) / max(len(shared), 1)
    return disjoint * 1.0 + w_diff * 0.4

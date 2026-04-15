import copy, json
from pathlib import Path

from config import (
    POPULATION, GENERATIONS, REPLAY_TOP_K, REPLAY_CONVERGE_K, REPLAY_STEPS,
    WORLD_WIDTH, WORLD_HEIGHT, PIPE_GAP, PIPE_SPEED, PIPE_SPACING,
)
from neat.genome import Genome, make_genome
from neat.evolution import speciate, evolve
from game.simulation import evaluate


def train():
    """
    Run the full NEAT training loop and write the best genome and replay to disk.

    Each generation evaluates all genomes on an identically seeded simulation,
    selects and breeds survivors, and records the top-k genomes for the web replay.
    Training stops early once the top performers all survive the full evaluation
    window, or after the configured maximum number of generations.

    Outputs:
    - best_genome.json: The highest-fitness genome found across all generations.
    - training_replay.json: Per-generation frame data for the browser replay viewer.
    """

    pop = [make_genome() for _ in range(POPULATION)]
    best_ever: Genome | None = None
    replay_generations = []
    replay_converged = False

    for gen in range(GENERATIONS):
        # evaluate every genome on the same seeded pipe sequence
        for g in pop:
            evaluate(g, seed=gen)

        ranked = sorted(pop, key=lambda g: g.fitness, reverse=True)
        best = ranked[0]

        # track the best genome seen across all generations
        if best_ever is None or best.fitness > best_ever.fitness:
            best_ever = copy.deepcopy(best)

        avg = sum(g.fitness for g in pop) / len(pop)
        n_species = len(speciate(pop))
        print(f"Gen {gen + 1:3d}  best={best.fitness:6.0f}  avg={avg:6.0f}  species={n_species}")

        # record top-k genomes for the web replay until the population has converged
        if not replay_converged:
            top_k = ranked[:REPLAY_TOP_K]
            gen_genomes = []

            for rank, g in enumerate(top_k, 1):
                _, frames = evaluate(g, seed=gen, record=True)
                gen_genomes.append({
                    "rank": rank, "fitness": g.fitness,
                    "pipes_passed": frames[-1]["pipes_passed"] if frames else 0,
                    "steps": len(frames),
                    "frames": frames,
                })

            replay_generations.append({
                "generation": gen,
                "best_pipes_passed": gen_genomes[0]["pipes_passed"] if gen_genomes else 0,
                "genomes": gen_genomes,
            })

            # stop recording once all top converge_k birds survive the full window
            if all(g["steps"] >= REPLAY_STEPS for g in gen_genomes[:REPLAY_CONVERGE_K]):
                replay_converged = True
                print(f"  [replay converged at gen {gen + 1} - top-{REPLAY_CONVERGE_K} birds survived the window]")
                break

        pop = evolve(pop)

    root = Path(__file__).parent

    # write the best genome as JSON for the visualiser
    output_path = root / "best_genome.json"
    output_path.write_text(json.dumps({"nodes": best_ever.nodes, "conns": best_ever.conns,
                                "fitness": best_ever.fitness}, indent=2))

    # write the full replay including world config for the browser viewer
    replay_path = root / "training_replay.json"
    replay_path.write_text(json.dumps({
        "config": {
            "world_width": WORLD_WIDTH, "world_height": WORLD_HEIGHT,
            "pipe_gap": PIPE_GAP, "pipe_speed": PIPE_SPEED,
            "pipe_spacing": PIPE_SPACING, "bird_x": 100.0,
        },
        "generations": replay_generations,
    }, separators=(",", ":")))

    print(f"\nBest fitness: {best_ever.fitness:.0f}")
    print(f"Saved: {output_path}")
    print(f"Saved: {replay_path}  (open index.html in a browser)")


if __name__ == "__main__":
    train()

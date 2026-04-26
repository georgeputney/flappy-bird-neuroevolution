import random
from config import WORLD_WIDTH, WORLD_HEIGHT, PIPE_SPACING, MAX_STEPS, REPLAY_STEPS
from game.bird import Bird
from game.pipe import Pipe


def _inputs(bird: Bird, pipes: list[Pipe]) -> list[float]:
    """
    Compute the normalised sensor inputs for a genome given the current game state.

    Returns five values describing the bird's position, velocity, and its relationship
    to the next pipe. All values are normalised to roughly the range [0, 1] or [-1, 1]
    to keep network weights in a consistent scale.

    Args:
    - bird (Bird): The bird being evaluated.
    - pipes (list[Pipe]): All active pipes currently in the simulation.

    Returns:
    - list[float]: Five normalised input values: bird height, velocity, horizontal
      distance to next pipe, gap error (bird relative to gap centre), and gap size.
    """

    ahead = [p for p in pipes if p.x + p.width >= bird.x]

    # if no pipe is ahead, return safe defaults pointing toward the centre
    if not ahead:
        return [bird.y / WORLD_HEIGHT, bird.velocity / 12.0, 1.0, 0.0, 0.0]

    p = min(ahead, key=lambda p: p.x)
    gap_centre = (p.top + p.bottom) / 2.0

    return [
        bird.y / WORLD_HEIGHT,
        bird.velocity / 12.0,
        (p.x - bird.x) / WORLD_WIDTH,
        (bird.y - gap_centre) / WORLD_HEIGHT,
        (p.bottom - p.top) / WORLD_HEIGHT,
    ]


def evaluate(genome, seed: int | None = None, record: bool = False):
    """
    Run a single genome through a full simulation and assign its fitness.

    The simulation is deterministic given the same seed, so all genomes in a
    generation face identical pipe sequences. Fitness rewards pipes cleared heavily,
    with survival frames as a tiebreaker.

    Args:
    - genome (Genome): The genome to evaluate; its fitness attribute is set in place.
    - seed (int | None): Random seed for pipe generation. None produces a random run.
    - record (bool): If True, captures per-frame state for replay export.

    Returns:
    - float: The genome's fitness score when record is False.
    - tuple[float, list[dict]]: Fitness and frame data when record is True.
    """

    rng = random.Random(seed)
    bird = Bird()
    pipes: list[Pipe] = [Pipe.spawn(WORLD_WIDTH + 60, rng)]
    passed = 0
    frames: list[dict] = []

    step_limit = REPLAY_STEPS if record else MAX_STEPS

    for step in range(step_limit):
        # spawn a new pipe when the last one has scrolled far enough left
        if pipes[-1].x < WORLD_WIDTH - PIPE_SPACING:
            pipes.append(Pipe.spawn(pipes[-1].x + PIPE_SPACING, rng))

        # remove pipes that have fully scrolled off the left edge
        pipes = [p for p in pipes if p.x + p.width > -5]

        # query the genome and apply the flap decision
        output = genome.activate(_inputs(bird, pipes))
        did_flap = output > 0.5
        if did_flap:
            bird.flap()
        bird.update()
        for p in pipes:
            p.update()

        # count each pipe the bird successfully passes through
        for p in pipes:
            if p.x + p.width < bird.x and not p.__dict__.get("_counted"):
                p.__dict__["_counted"] = True
                passed += 1

        alive = bird.alive() and not any(p.hits(bird) for p in pipes)

        # capture full frame state if recording for the web replay
        if record:
            frames.append({
                "x": bird.x, "y": round(bird.y, 1), "vy": round(bird.velocity, 2),
                "alive": int(alive), "flap": int(did_flap),
                "out": round(output, 3), "pipes_passed": passed,
                "pipes": [
                    {"x": round(p.x, 1), "gap_y": round((p.top + p.bottom) / 2, 1),
                     "gap_h": round(p.bottom - p.top, 1), "width": p.width}
                    for p in pipes
                ],
            })

        if not alive:
            break

    # Pipes cleared dominate the score, but survival frames are included as an
    # additive term. In early generations no bird reaches the first pipe, so
    # "pipes cleared" is uniformly zero and gives the algorithm no gradient to
    # work with. Counting frames keeps selection pressure alive by rewarding
    # birds that at least stay airborne longer.
    genome.fitness = float(passed * 100 + step)
    return (genome.fitness, frames) if record else genome.fitness

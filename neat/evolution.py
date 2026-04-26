import copy, random
from config import COMPAT_THRESHOLD
from neat.genome import Genome, distance, mutate, crossover


def speciate(pop: list[Genome]) -> list[list[Genome]]:
    """
    Partition a population into species based on genome compatibility distance.

    Each genome is assigned to the first species whose representative it falls
    within the compatibility threshold of. If no match is found, a new species
    is created with that genome as its representative.

    Args:
    - pop (list[Genome]): The full population to partition.

    Returns:
    - list[list[Genome]]: A list of species, each species being a list of genomes.
    """

    species: list[list[Genome]] = []

    for g in pop:
        # assign to the first compatible species, or start a new one
        for s in species:
            if distance(g, s[0]) < COMPAT_THRESHOLD:
                s.append(g)
                break
        else:
            species.append([g])
    return species


def evolve(pop: list[Genome]) -> list[Genome]:
    """
    Produce the next generation from the current population using NEAT evolution.

    Each species contributes offspring proportional to its share of total fitness.
    The best genome in each species is carried forward unchanged (elitism). The
    remainder are produced by crossover and mutation, or mutation alone.

    Args:
    - pop (list[Genome]): The current generation's population with fitness assigned.

    Returns:
    - list[Genome]: A new population of the same size ready for evaluation.
    """

    species = speciate(pop)
    total_fitness = sum(g.fitness for g in pop) or 1.0
    next_population: list[Genome] = []

    for s in species:
        # rank survivors within each species and keep the top half
        s.sort(key=lambda g: g.fitness, reverse=True)
        survivors = s[: max(1, len(s) // 2)]

        # allocate offspring proportional to this species' share of total fitness
        n_offspring = max(1, round(sum(g.fitness for g in s) / total_fitness * len(pop)))

        # Carry the best genome in each species forward unchanged (per-species
        # elitism). Without this, a lucky high-fitness genome can be mutated away
        # before the rest of the population catches up, causing fitness to collapse
        # between generations. Elitism is the main stabiliser against that regression.
        next_population.append(copy.deepcopy(survivors[0]))

        # fill the remaining allocation with crossover or cloned mutations
        for _ in range(n_offspring - 1):
            if len(survivors) > 1 and random.random() < 0.75:
                a, b = random.sample(survivors, 2)
                child = crossover(*((a, b) if a.fitness >= b.fitness else (b, a)))
            else:
                child = copy.deepcopy(random.choice(survivors))
            next_population.append(mutate(child))

    # trim or pad to exactly match the original population size
    while len(next_population) > len(pop):
        next_population.pop()
    while len(next_population) < len(pop):
        next_population.append(mutate(copy.deepcopy(random.choice(pop))))
    return next_population

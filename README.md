# Flappy Bird Neuroevolution
Writeup and live replay viewer: [georgeputney.com/projects/flappy-bird-neuroevolution](https://georgeputney.com/projects/flappy-bird-neuroevolution/)

NEAT (Neuroevolution of Augmenting Topologies) implemented from scratch in Python, applied to Flappy Bird. Networks evolve both structure and weights simultaneously, starting with no hidden layers and growing complexity only where it proves useful. Converges in ~15 generations; the evolved network clears pipes indefinitely.

## Structure
```
neat/         # NEAT algorithm: genome encoding, mutation, crossover, speciation
game/         # Flappy Bird simulation: bird physics, pipe generation, evaluation
web/          # Browser replay viewer (index.html + viewer.js)
main.py       # Training loop, runs evolution and writes output files
visualise.py  # Renders the best genome as a network diagram (matplotlib)
config.py     # All hyperparameters in one place
```

## Requirements
Python 3.10+
```bash
pip install -e .
```

## Usage
**Train:**
```bash
python main.py
```
Writes `best_genome.json` and `training_replay.json` on completion.

**Visualise the best genome:**
```bash
python visualise.py
```
Writes `best_genome.png`.

**Replay viewer:**
```bash
python -m http.server 8000
```
Open `http://localhost:8000/web/` to replay the full training run generation by generation.
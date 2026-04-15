import random
from dataclasses import dataclass
from config import PIPE_GAP, PIPE_SPEED, WORLD_HEIGHT


@dataclass
class Pipe:
    """
    A pair of vertically aligned obstacles with a gap the bird must pass through.

    The pipe scrolls leftward at a fixed speed each frame. Gap position is randomised
    on spawn, constrained so the gap never appears too close to the world edges.
    """

    x: float
    top: float
    bottom: float
    width: float = 60.0


    @classmethod
    def spawn(cls, x: float, rng: random.Random) -> "Pipe":
        """
        Create a new pipe at a given horizontal position with a randomised gap.

        Args:
        - x (float): The horizontal position at which to spawn the pipe.
        - rng (random.Random): A seeded random instance for deterministic generation.

        Returns:
        - Pipe: A new pipe with top and bottom edges defining the gap.
        """

        # place the gap centre within safe margins from the world edges
        gap_centre = rng.uniform(PIPE_GAP / 2 + 60, WORLD_HEIGHT - PIPE_GAP / 2 - 60)
        return cls(x=x, top=gap_centre - PIPE_GAP / 2, bottom=gap_centre + PIPE_GAP / 2)


    def update(self):

        # scroll the pipe leftward by one frame
        self.x -= PIPE_SPEED


    def hits(self, bird) -> bool:
        
        # check horizontal overlap first, then vertical collision against gap edges
        r = 14.0  # bird radius in pixels
        if bird.x + r < self.x or bird.x - r > self.x + self.width:
            return False
        return bird.y - r < self.top or bird.y + r > self.bottom

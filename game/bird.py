from dataclasses import dataclass
from config import WORLD_HEIGHT


@dataclass
class Bird:
    """
    A simulated bird governed by simple gravity and flap physics.

    The bird falls under constant gravity each frame and can jump upward by flapping.
    Velocity is clamped to prevent runaway acceleration in either direction.
    """

    y: float = 250.0
    velocity: float = 0.0
    x: float = 100.0


    def flap(self):
        
        # apply upward impulse
        self.velocity = -8.0


    def update(self):

        # apply gravity, clamp velocity, then update position
        self.velocity = max(-12.0, min(12.0, self.velocity + 0.5))
        self.y += self.velocity


    def alive(self) -> bool:

        # check the bird hasn't left the world vertically
        return 0 <= self.y < WORLD_HEIGHT

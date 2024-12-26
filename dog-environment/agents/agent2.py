import numpy as np
from dog.envs.dog_model import DogModel



class Agent2:

    def __init__(self):
        self.path = []
        self.model = DogModel()
        self.direction_to_action = {
            (1, 0): 0,    # RIGHT
            (0, 1): 1,    # UP
            (-1, 0): 2,   # LEFT
            (0, -1): 3,   # DOWN
        }

    def reset(self):
        """Reset the agent's state."""
        self.path = []
        self.model = DogModel()

    def agent_function(self, observation):
        """Compute the next action based on the observation."""
        self.model.update(observation)
        current_state = tuple(observation["agent"])

        # recalculate if path is invalid or empty
        if not self.path:
            self.path = self.model.a_star_search(current_state, self.model.target_location)

        if self.path:
            next_state = self.path.pop(0)
            direction = (next_state[0] - current_state[0], next_state[1] - current_state[1])
            return self.direction_to_action.get(direction, 0)
        return 0

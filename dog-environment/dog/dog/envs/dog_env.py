import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from dog.envs.dog_model import DogModel

class DogEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=15, num_obstacles=35):
        self.size = size  # size of the grid
        self.window_size = 512  # pygame window size
        self.num_obstacles = num_obstacles  # number of obstacles

        # the observation are dictionaries with the agent's and the target's location
        self.observation_space = spaces.Dict(
            {
                # represents the agents position, the agents position is constrained to between 0 and size-1 in coordinates
                # shape=(2,) means that the observation is 2d [x, y]
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # targets position, works similar to how the agent position works
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # obstacles are contrained to the grid like the agent and target
                # num_obstacles is the number of obstacles and the observation is 2d [x, y]
                "obstacles": spaces.Box(0, size - 1, shape=(num_obstacles, 2), dtype=int),
            }
        )

        """
        The action space is discrete
        0 : RIGHT
        1 : UP
        2 : LEFT
        3 : DOWN

        """
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),   # represents RIGHT
            1: np.array([0, 1]),   # represents UP
            2: np.array([-1, 0]),  # represents LEFT
            3: np.array([0, -1]),  # represents DOWN
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # create the environment state
        self._agent_location = None
        self._target_location = None
        self._obstacle_locations = set()

        # instance of dog model class
        self.dog_model = DogModel()

    # pygame stuff
        self.window = None
        self.clock = None

    def _generate_random_position(self, exclude=None):
        # this will generate a random position, if the position is in exclude then it will avoid generating it
        exclude = exclude or set()
        while True:
            position = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            if position not in exclude:
                return position

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # reset the agent, target, and es
        self._agent_location = self._generate_random_position()
        self._target_location = self._generate_random_position(exclude={self._agent_location})
        self._obstacle_locations = set()
        for _ in range(self.num_obstacles):
            obstacle_pos = self._generate_random_position(
                exclude={self._agent_location, self._target_location, *self._obstacle_locations}
            )
            self._obstacle_locations.add(obstacle_pos)

        # Update DogModel with the initial state, including obstacles
        self.dog_model.update(self._get_obs())

        # return the initial observation
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        proposed_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if tuple(proposed_location) not in self._obstacle_locations:
            self._agent_location = proposed_location 

        # see if agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0

        # Update DogModel with the current state, including obstacles
        self.dog_model.update(self._get_obs())

        # Return the new state, reward, and if the goal was met
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

# env current state
    def _get_info(self):
        return {
            "agent_location": self._agent_location,
            "target_location": self._target_location,
            "obstacle_locations": list(self._obstacle_locations),
        }

    def _get_obs(self):
        return {
            "agent": np.array(self._agent_location, dtype=int),
            "target": np.array(self._target_location, dtype=int),
            "obstacles": np.array(list(self._obstacle_locations), dtype=int),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((194, 218, 184)) 
        pix_square_size = self.window_size / self.size

       # target
        target_img = pygame.image.load("doghouse.png").convert_alpha()
        target_img = pygame.transform.smoothscale(
            target_img, (int(pix_square_size), int(pix_square_size))
        )
        target_pos = np.array(self._target_location) * pix_square_size
        canvas.blit(target_img, target_pos)

        #agent
        agent_img = pygame.image.load("dog.png").convert_alpha()
        agent_img = pygame.transform.smoothscale(
            agent_img, (int(pix_square_size), int(pix_square_size))
        )
        agent_pos = np.array(self._agent_location) * pix_square_size
        canvas.blit(agent_img, agent_pos)

        # obstacles
        obstacle_img = pygame.image.load("obstacles2.png").convert_alpha()
        obstacle_img = pygame.transform.smoothscale(
            obstacle_img, (int(pix_square_size), int(pix_square_size))
        )
        for obstacle in self._obstacle_locations:
            obstacle_pos = np.array(obstacle) * pix_square_size
            canvas.blit(obstacle_img, obstacle_pos)

        # grid stuff
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas, (194, 218, 184), (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=3
            )
            pygame.draw.line(
                canvas, (194, 218, 184), (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=3
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

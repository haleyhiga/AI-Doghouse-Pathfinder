from gymnasium.envs.registration import register
from dog.envs.dog_env import DogEnv
from dog.envs.dog_model import DogModel

register(
    id="Dog-v0",
    entry_point="dog.envs.dog_env:DogEnv",
)
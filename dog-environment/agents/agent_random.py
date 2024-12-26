#!/usr/bin/env python3

import random
from dog.envs.dog_model import DogModel

class AgentRandom:
    
    def __init__(self):
        self.model = DogModel()
        return

    def reset(self):
        self.model.reset() 
        return

    def agent_function(self, state):
        action = random.choice(range(4))  # choose a random action in action space
        return action

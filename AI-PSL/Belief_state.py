# Belief search algorithm

import numpy as np
import math
import random

class BeliefState:
    def __init__(self, belief, action, observation, reward, next_belief):
        self.belief = belief
        self.action = action
        self.observation = observation
        self.reward = reward
        self.next_belief = next_belief

    def __repr__(self):
        return "Belief: " + str(self.belief) + ", Action: " + str(self.action) + ", Observation: " + str(self.observation) + ", Reward: " + str(self.reward) + ", Next Belief: " + str(self.next_belief)




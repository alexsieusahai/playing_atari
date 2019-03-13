import numpy as np

from typing import Tuple, List


class ExperienceReplayBuffer:
    """
    Implementation of the Experience Replay buffer described in 
    Playing Atari with Deep Reinforcement Learning (Minh, 2015).
    """
    def __init__(self, maximum_capacity: int):
        """
        :param maximum_capacity: The maximum capacity of the memory, which 
        holds all of the  experiences.
        """
        self.maximum_capacity = maximum_capacity
        self.memory = []

    def append(self, experience: Tuple) -> None:
        """
        Handle the memory greater than maximum_capacity case.
        :param experience: A 4-tuple containing 
            (state_t, action_t, reward_t, state_{t+1}) for some t.
        """
        self.memory.append(experience)
        while len(self.memory) > maximum_capacity:
            self.memory.pop()

    def uniform_sample(self, minibatch_size: int) -> List[Tuple]:
        """
        Uniformly samples some experiences from its memory.
        :param minibatch_size: The amount of experiences to sample.
        :returns: A list of experiences uniformly sampled from memory.
        """
        return np.random.choice(self.memory, minibatch_size)

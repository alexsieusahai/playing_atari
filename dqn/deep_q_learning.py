import gym
import torch
import numpy as np

from experience_replay_buffer import ExperienceReplayBuffer
from function_approximator import ConvolutionApproximator
from preprocess import phi


def deep_q_learning(memory_max_capacity: int, num_episodes: int, 
                    epsilon: float, minibatch_size: int):
    
    """
    Implementation of the Deep Q-learning with Experience Replay algorithm from 
        Playing Atari with Deep Reinforcement Learning.
    :params memory_max_capacity: The maximum size of the memory in the ExperienceReplayBuffer.
    :params num_episodes: The number of episodes to train on.
    :params epsilon: The epsilon to use when constructing the epsilon greedy policy.
    :params minibatch_size: The size of the minibatch to sample / train on.
    :returns: A function approximator.
    """
    env = gym.make('Breakout-v0')
    device = torch.device('cuda')
    replay_memory = ExperienceReplayBuffer(memory_max_capacity)
    Q = ConvolutionApproximator(4).double().to(device)  # 4 is amount of moves in breakout

    for _ in range(num_episodes):
        phi_dict = {}
        obs = env.reset()
        obs_list = [obs] * 4
        state = phi(obs_list)

        done = False
        while not done:
            if np.random.random() < epsilon and False:
                action = env.action_space.sample()
            else:
                output = Q(state)
                print(output)
                action = output.max(1)[1].view(1, 1)
                print(action)
                raise NotImplementedError
            
            previous_state = state
            obs, reward, done, _ = env.step(action)
            obs_list.append(obs)
            obs_list.pop()
            state = phi(obs_list)

            replay_memory.append((previous_state, action, reward, state))

            # include training procedure here

if __name__ == "__main__":
    deep_q_learning(10e7, 1, 1)

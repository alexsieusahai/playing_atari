import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from experience_replay_buffer import ExperienceReplayBuffer
from function_approximator import ConvolutionApproximator
from preprocess import phi


def update_weights(Q, optimizer, experience_dataset, gamma) -> None:
    """
    I'm not sure if we're supposed to do it step by step like below, or
    as a whole batch... If we do it step by step, then we have Q-learning
    """
    for state, action, reward, next_state in experience_dataset:
        q_values = Q(state)
        next_q_values = Q(next_state)

        q_value_for_action = float(q_values[0][action])
        max_q_for_next = float(next_q_values.max(1)[0])
        
        expected_q_value = reward + gamma * max_q_for_next

        loss = F.mse_loss(q_value_for_action, expected_q_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss



def deep_q_learning(memory_max_capacity: int, num_episodes: int, minibatch_size: int,
                    gamma: float):
    """
    Implementation of the Deep Q-learning with Experience Replay algorithm from 
        Playing Atari with Deep Reinforcement Learning.
    No parameters were specified for RMSProp, so I'm using the default parameters.
    :params memory_max_capacity: The maximum size of the memory in the ExperienceReplayBuffer.
    :params num_episodes: The number of episodes to train on.
    :params minibatch_size: The size of the minibatch to sample / train on.
    :params gamma: The decay rate to use.
    :returns: A function approximator.
    """
    env = gym.make('Breakout-v0')
    device = torch.device('cpu')
    replay_memory = ExperienceReplayBuffer(memory_max_capacity)
    Q = ConvolutionApproximator(4).double().to(device)  # 4 is amount of moves in breakout
    optimizer = optim.RMSprop(Q.parameters())

    for num_episode in range(num_episodes):
        phi_dict = {}
        obs = env.reset()
        obs_list = [obs] * 4
        state = phi(obs_list)
        epsilon = max(0.1, 1 - num_episode / 1e7)

        done = False
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                output = Q(state)
                action = output.max(1)[1].view(1, 1)
            
            previous_state = state
            obs, reward, done, _ = env.step(action)
            obs_list.append(obs)
            obs_list.pop()
            state = phi(obs_list)

            replay_memory.append((previous_state, action, reward, state))
            experience_sample = replay_memory.uniform_sample(minibatch_size)

            update_weights(Q, optimizer, experience_sample, gamma)


if __name__ == "__main__":
    deep_q_learning(10e7, 1, 32, 0.99)

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

from experience_replay_buffer import ExperienceReplayBuffer
from function_approximator import ConvolutionApproximator
from preprocess import phi


Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)

def update_weights(Q, optimizer, experience_dataset, gamma) -> None:
    """
    I'm not sure if we're supposed to do it step by step like below, or
    as a whole batch... If we do it step by step, then we have Q-learning
    """
    device = torch.device('cuda')
    total_loss = 0
    for state, action, reward, next_state in experience_dataset:
        state = Variable(torch.DoubleTensor(np.float32(state)))
        action = Variable(torch.LongTensor(np.float32([action])))
        reward = Variable(torch.DoubleTensor(np.float32([reward])).to(device))
        next_state = Variable(torch.DoubleTensor(np.float32(next_state)))


        q_values = Q(state.to(device))
        next_q_values = Q(next_state.to(device))

        q_value_for_action = q_values.gather(1, action.unsqueeze(1).to(device)).squeeze(1)
        max_q_for_next = next_q_values.max(1)[0]
        
        expected_q_value = reward + gamma * max_q_for_next

        loss = (q_value_for_action - Variable(expected_q_value.data)).pow(2)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(experience_dataset)


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
    device = torch.device('cuda')
    replay_memory = ExperienceReplayBuffer(memory_max_capacity)
    Q = ConvolutionApproximator(4).double().to(device)  # 4 is amount of moves in breakout
    optimizer = optim.RMSprop(Q.parameters())

    for num_episode in range(num_episodes):
        print(f'Episode {num_episode}')
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
                output = Q(state.to(device))
                action = output.max(1)[1].view(1, 1)

            previous_state = state
            obs, reward, done, _ = env.step(action)
            obs_list.append(obs)
            obs_list.pop()
            state = phi(obs_list)

            replay_memory.append((previous_state, action, reward, state))
            experience_sample = replay_memory.uniform_sample(minibatch_size)

            update_weights(Q, optimizer, experience_sample, gamma)

    return Q, optimizer


if __name__ == "__main__":
    import pickle as pkl

    Q, optimizer = deep_q_learning(10e7, 1, 32, 0.99)
    pkl.dump(Q, open('Q.pkl', 'wb'))
    pkl.dump(optimizer, open('optimizer.pkl', 'wb'))

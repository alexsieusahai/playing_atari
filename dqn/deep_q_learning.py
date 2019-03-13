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


def deep_q_learning(memory_max_capacity: int, num_frames: int, minibatch_size: int,
                    gamma: float, Q=None, optimizer=None):
    """
    Implementation of the Deep Q-learning with Experience Replay algorithm from 
        Playing Atari with Deep Reinforcement Learning.
    No parameters were specified for RMSProp, so I'm using the default parameters.
    :param memory_max_capacity: The maximum size of the memory in the ExperienceReplayBuffer.
    :param num_frames: The number of time steps to train on before giving up.
    :param minibatch_size: The size of the minibatch to sample / train on.
    :param gamma: The decay rate to use.
    :param Q: A pretrained Q-value approximator for this task.
    :param optimizer: The associated optimizer for Q for this task.
    :returns: A function approximator.
    """
    env = gym.make('Breakout-v0')
    device = torch.device('cuda')
    replay_memory = ExperienceReplayBuffer(memory_max_capacity)
    Q = ConvolutionApproximator(4).double().to(device) if Q is None else Q  # 4 is amount of moves in breakout
    optimizer = optim.RMSprop(Q.parameters()) if optimizer is None else optimizer

    states_to_check = []
    progress = []
    step_count = 0
    num_episode = 0

    while step_count < num_frames:
        obs = env.reset()
        obs_list = [obs] * 4
        state = phi(obs_list)
        if num_episode == 0:
            states_to_check.append(state)

        epsilon = max(0.1, 1 - step_count / 1e7)

        done = False
        disc_return = 0
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                output = Q(state.to(device))
                action = output.max(1)[1].view(1, 1)

            previous_state = state
            obs, reward, done, _ = env.step(action)
            disc_return = reward + gamma * disc_return
            obs_list.append(obs)
            obs_list.pop()
            state = phi(obs_list)
            if num_episode == 0:
                states_to_check.append(state)

            replay_memory.append((previous_state, action, reward, state))
            experience_sample = replay_memory.uniform_sample(minibatch_size)

            update_weights(Q, optimizer, experience_sample, gamma)

            step_count += 1

        sum_max = 0
        for state in states_to_check:
            sum_max += float(Q(state.to(device)).max(1)[0])
        progress.append(sum_max)
        print(f'Episode {num_episode} completed with progress {progress[-1]}')
        print(f'Achieved a discounted return of {disc_return}')
        print(f'{step_count} steps taken so far')
        num_episode += 1

    return Q, optimizer, progress


if __name__ == "__main__":
    import pickle as pkl

    Q, optimizer, progress = deep_q_learning(10e7, 10e8, 32, 0.99)
    pkl.dump(progress, open('progress.pkl', 'wb'))
    pkl.dump(Q, open('Q.pkl', 'wb'))
    pkl.dump(optimizer, open('optimizer.pkl', 'wb'))

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionApproximator(nn.Module):
    """
    Model architecture from Playing Atari with Deep Reinforcement Learning.
    The input is an 84 x 84 x 4 tensor.
    The first hidden layer convolves 16 8x8 filters with stride 4 and applies ReLU.
    The second hidden layer convolves 32 4x4 filters with stride 2 and applies ReLU.
    The final hidden layer is fully connected, and consists of 256 units with ReLU activation
        mapping to |Action_space|.

    This implementation aims to be faithful to the paper.
    """
    def __init__(self, action_space_size: int):
        """
        :param action_space_size: The size of the set of possible actions for the environment.
        """
        super(ConvolutionApproximator, self).__init__()
        self.conv0 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.linear = nn.Linear(256, action_space_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param state: A 1 x 4 x 84 x 84 input tensor.
        :returns: A tensor which contains a value for each action in the space.
        """
        output = state
        print('input shape is', output.shape)
        output = F.relu(self.conv0(output))
        print('after conv0 my shape is', output.shape)
        output = F.relu(self.conv1(output))
        print('after conv1 my shape is', output.shape)
        print(output)
        output = output.view(-1, 256)
        print('output shape being passed into linear is', output.shape)
        output = F.relu(self.linear(output))
        return output

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

    This implementation aims to be faithful to the paper, but I can't seem to get it working
    using their setup. They have the above, connected to the hidden layer described.
    I added an extra linear hidden layer; I wonder what's the reprocussions of this, if anything.
    """
    def __init__(self, action_space_size: int):
        """
        :param action_space_size: The size of the set of possible actions for the environment.
        """
        super(ConvolutionApproximator, self).__init__()
        self.conv0 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.linear0 = nn.Linear(2592, 256)
        self.linear1 = nn.Linear(256, action_space_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param state: A 1 x 4 x 84 x 84 input tensor.
        :returns: A tensor which contains a value for each action in the space.
        """
        output = state
        output = F.relu(self.conv0(output))
        output = F.relu(self.conv1(output))
        output = output.view(-1, 2592)
        output = F.relu(self.linear0(output))
        output = F.relu(self.linear1(output))
        return output

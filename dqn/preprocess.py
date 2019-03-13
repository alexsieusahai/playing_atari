import torch
import numpy as np
from skimage.transform import rescale

from typing import List


def phi(images: List[np.array]) -> np.array:
    """
    Implements the preprocessing logic in Playing Atari using Deep Reinforcement Learning.
    :params image: A list(-like) of length 4 of arrays with (210, 160, 3), which represents 
        the rgb values observed from the Atari environment.
    :returns: An array of the processed image matrices passed in as images.

    The downsampling procedure wasn't described in the paper, so I'm just using a vanilla one.
    NOTE: Since I'm training only on Breakout, I can employ some domain knowledge, 
        hence the weird looking cropping. They didn't specify how they did this in the paper.
    """
    outputs = []
    for image in images:
        greyscale = np.mean(image, axis=2)
        downscaled = rescale(greyscale, 0.525)
        cropped = downscaled[18:102]
        outputs.append(cropped)
    return torch.from_numpy(np.stack(outputs)).double().unsqueeze(0)


if __name__ == "__main__":
    import gym

    env = gym.make('Breakout-v0')
    images = []

    images.append(env.reset())

    for _ in range(3):
        images.append(env.step(0)[0])

    assert phi(images).shape == (1, 4, 84, 84)

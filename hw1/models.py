"""The FP32, sequential CNN specified in section 3 of the assignment."""

import torch
from torch import nn


def build_model() -> nn.Sequential:
    """Return an untrained model in eval mode with 100 output logits.

    BatchNorm is not included because none is listed in the model table.
    Linear layers retain their usual biases; only convolution biases are disabled.
    """
    layers: list[nn.Module] = []
    for index, (cin, cout, kernel, stride) in enumerate(
        [(3, 32, 7, 2), (32, 64, 5, 1), (64, 128, 3, 2),
         (128, 256, 1, 1), (256, 256, 3, 2), (256, 512, 1, 1)]
    ):
        layers.extend([
            nn.Conv2d(cin, cout, kernel, stride=stride, padding=kernel // 2,
                      bias=False, dtype=torch.float32),
            nn.ReLU(inplace=True),
        ])
        if index == 0:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    layers.extend([
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(512, 256, dtype=torch.float32),
        nn.ReLU(inplace=True),
        nn.Linear(256, 100, dtype=torch.float32),
    ])
    return nn.Sequential(*layers).eval()

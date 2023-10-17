import torch
import torch.nn as nn
import numpy as np
import conv


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, custom_weights, custom_bias):
        super(CustomConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)
        self.conv.weight = nn.Parameter(custom_weights)
        self.conv.bias = nn.Parameter(custom_bias)

    def forward(self, x):
        return self.conv(x)


def test_Conv2D():

    torch.manual_seed(0)
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    m, n_H, n_W = 1, 10, 10

    test_input = torch.randn(m, in_channels, n_H, n_W)
    test_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    test_bias = torch.randn(8)
    test_stride = 2
    test_padding = 1

    custom_conv2d = conv.Conv2D(test_weight, test_bias, test_stride, test_padding)
    actual_output = custom_conv2d(test_input)

    custom_conv_layer = CustomConv2d(in_channels, out_channels, kernel_size,
                                     test_stride, test_padding, test_weight, test_bias)
    expected_output = custom_conv_layer(test_input)

    assert actual_output.size == expected_output.size
    assert np.isclose([actual_output[0, 0, 0, 0], actual_output[-1, -1, -1, -1]],
                      [expected_output[0, 0, 0, 0], expected_output[-1, -1, -1, -1]])
"""
2d Convolution Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################        
        x_padded = np.pad(x, pad_width=(
            (0, 0),
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding)
        ))
        
        N, C, H, W = x_padded.shape

        H_out = int(np.floor((H - self.kernel_size) / self.stride + 1))
        W_out = int(np.floor((W - self.kernel_size) / self.stride + 1))
        
        out = np.zeros((N, self.out_channels, H_out, W_out))
        
        for h in range(H_out):
            for w in range(W_out):
                h_lower = h * self.stride
                h_upper = h * self.stride + self.kernel_size
                w_lower = w * self.stride
                w_upper = w * self.stride + self.kernel_size
                
                x_window = x_padded[:, :, h_lower:h_upper, w_lower:w_upper]
                
                for out_c in range(self.out_channels):                    
                    out[:, out_c, h, w] = np.sum(
                        x_window * self.weight[out_c],
                        axis=(1, 2, 3)
                    ) + self.bias[out_c]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        x_padded = np.pad(x, pad_width=(
            (0, 0),
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding)
        ))
        
        dx_padded = np.zeros_like(x_padded)
        dw = np.zeros_like(self.weight)
        
        N, C, H, W = x_padded.shape
        
        H_out = int(np.floor((H - self.kernel_size) / self.stride + 1))
        W_out = int(np.floor((W - self.kernel_size) / self.stride + 1))
        
        for h in range(H_out):
            for w in range(W_out):
                h_lower = h * self.stride
                h_upper = h * self.stride + self.kernel_size
                w_lower = w * self.stride
                w_upper = w * self.stride + self.kernel_size
                
                x_window = x_padded[:, :, h_lower:h_upper, w_lower:w_upper]
                
                for out_c in range(self.out_channels):
                    for in_c in range(self.in_channels):
                        for i in range(self.kernel_size):
                            for j in range(self.kernel_size):        
                                dw[out_c, in_c, i, j] += np.sum(
                                    x_window[:, in_c, i, j] * dout[:, out_c, h, w]
                                )
                                
                                dx_padded[:, in_c, h_lower + i, w_lower + j] += \
                                    self.weight[out_c, in_c, i, j] * dout[:, out_c, h, w]

        self.dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        self.dw = dw
        self.db = np.sum(dout, axis=(0, 2, 3))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

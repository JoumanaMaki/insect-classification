import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import unfoldNd

from ..manifold import CustomLorentz
from ..layers import LorentzFullyConnected
from ...geoopt.manifolds import Stiefel
from .layer_utils import CustomWeightConv1d
from ..layers.linear_layers.LFC import (LorentzProjection,
                                           LorentzBoost,
                                           LorentzBoostScale)


def unfold1d(input, kernel_size: int, stride: int, padding: int):
    *shape, length = input.shape
    n_frames = (max(length, kernel_size) - kernel_size) // stride + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    input = input[..., :tgt_length].contiguous()
    strides = list(input.stride())
    strides = strides[:-1] + [stride, 1]
    out = input.as_strided(shape + [n_frames, kernel_size], strides)
    return out.transpose(-1, -2)


class LorentzConv1d(nn.Module):
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            rescale_before=False,
            rescale_after=False,
            LFC_normalize=False
    ):
        super(LorentzConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features,
            self.out_channels,
            bias=bias,
            normalize=LFC_normalize
        )

        self.rescale_before = rescale_before
        self.rescale_after = rescale_after

        self.unfold = unfoldNd.UnfoldNd(
            kernel_size, padding=0, stride=stride
        )

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = x.permute(0, 2, 1)
        #  patches = unfold1d(x, self.kernel_size, self.stride, self.padding)
        #  patches = patches.reshape(bsz, self.kernel_size * self.in_channels, -1)
        patches = self.unfold(x)
        patches = patches.permute(0, 2, 1)

        patches_space = patches.narrow(-1, self.kernel_size, patches.shape[-1] - self.kernel_size)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape)  # No need, but seems to improve runtime??
        patches_pre_kernel = self.manifold.add_time(patches_space)

        if self.rescale_before:
            patches_pre_kernel = self.manifold.rescale_to_max(patches_pre_kernel)

        out = self.linearized_kernel(patches_pre_kernel)

        if self.rescale_after:
            out = self.manifold.rescale_to_max(out)

        return out


class LorentzPureConv1d(nn.Module):
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            rescale_before=False,
            rescale_after=True,
    ):
        super(LorentzPureConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzProjection(self.manifold,
                                                   lin_features,
                                                   out_channels,)

        #self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size, padding=padding, stride=stride)

        self.rescale_before = rescale_before
        self.rescale_after = rescale_after

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = x.permute(0, 2, 1)
        patches = unfold1d(x, self.kernel_size, self.stride, padding=0)
        patches = patches.reshape(bsz, self.kernel_size*self.in_channels, -1).permute(0, 2, 1)

        patches_space = patches.narrow(-1, self.kernel_size, patches.shape[-1] - self.kernel_size)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape)  # No need, but seems to improve runtime??

        patches_pre_kernel = self.manifold.add_time(patches_space)

        if self.rescale_before:
            patches_pre_kernel = self.manifold.rescale_to_max(patches_pre_kernel)

        out = self.linearized_kernel(patches_pre_kernel)
        out = self.manifold.projx(out)
        if self.rescale_after:
            out = self.manifold.rescale_to_max(out)

        return out


class HyperbolicStiefelConv1D(nn.Module):
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            boost_type="lorentzboost"
    ):
        super(HyperbolicStiefelConv1D, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size = kernel_size

        self.rotation_manifold = Stiefel()

        self.rotate = nn.Conv1d(in_channels-1,
                                out_channels-1,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False)
        d_out, d_in, n = self.rotate.weight.shape

        self.debug_test = False

        if d_out >= d_in * n:

            self.rotate = CustomWeightConv1d(self.rotation_manifold,
                                             (d_out, d_in, n),
                                             in_channels - 1,
                                             out_channels - 1,
                                             kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             bias=False)

            self.debug_test = True

        self.boost = LorentzBoost(manifold)

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        with torch.no_grad():
            self.rotate.weight.copy_(self.rotation_manifold.projx(self.rotate.weight.data.uniform_(-stdv, stdv)))

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x N x C """

        # restore space_last representation
        out = self.rotate(x[..., 1:].permute(0, 2, 1)).permute(0, 2, 1)
        out = self.manifold.add_time(out)

        out = self.boost(out)
        out = self.manifold.rescale_to_max(out)

        return out


class HyperbolicCayleyConv1D(nn.Module):

    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            boost_type="lorentzboost"
    ):
        super(HyperbolicCayleyConv1D, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size = kernel_size

        self.rotation_manifold = Stiefel()

        self.rotate = nn.Conv1d(in_channels - 1,
                                out_channels - 1,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False)
        d_out, d_in, n = self.rotate.weight.shape

        if d_out >= d_in * n:
            self.rotate = CustomWeightConv1d(None,
                                             (d_out, d_in, n),
                                             in_channels - 1,
                                             out_channels - 1,
                                             kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             bias=False)

            torch.nn.utils.parametrizations.orthogonal(self.rotate,
                                                       name='weight',
                                                       orthogonal_map="cayley",
                                                       use_trivialization=False)

        #self.boost = LorentzBoost(manifold)
        self.boost = LorentzBoostScale(manifold)

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]))
        with torch.no_grad():
            self.rotate.weight.copy_(self.rotation_manifold.projx(self.rotate.weight.data.uniform_(-stdv, stdv)))

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x N x C """

        # restore space_last representation
        out = self.rotate(x[..., 1:].permute(0, 2, 1)).permute(0, 2, 1)
        out = self.manifold.add_time(out)

        out = self.boost(out)
        out = self.manifold.rescale_to_max(out)

        return out


if __name__ == '__main__':

    x = torch.rand((128, 4, 32))
    manifold = CustomLorentz(k=1, learnable=True)

    project_x = manifold.projx(x)

    layer = LorentzConv1d(manifold, 32,
                          64,
                          6)

    output = layer(project_x)

    print("break")

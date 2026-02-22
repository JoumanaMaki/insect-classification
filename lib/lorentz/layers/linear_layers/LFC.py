import torch
import torch.nn as nn

from ...manifold import CustomLorentz
from ..LMLR import LorentzMLR
from ..cayley_layers import CayleyLinear
from .rotations import LorentzRotationUp, LorentzRotationFixedNorm
from .boosts import (LorentzBoost,
                     LorentzBoostAlternate,
                     LorentzBoostScale,
                     LorentzBoostScaleAlternate,
                     LorentzPureBoost)


class LorentzFullyConnected(nn.Module):
    """
        Modified Lorentz fully connected layer of Chen et al. (2022).

        Code modified from https://github.com/chenweize1998/fully-hyperbolic-nn

        args:
            manifold: Instance of Lorentz manifold
            in_features, out_features, bias: Same as nn.Linear
            init_scale: Scale parameter for internal normalization
            learn_scale: If scale parameter should be learnable
            normalize: If internal normalization should be applied
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            in_features,
            out_features,
            bias=False,
            init_scale=None,
            learn_scale=False,
            normalize=False,
            activation=None,
            dropout=0.0,
            nheads=1
        ):
        super(LorentzFullyConnected, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.normalize = normalize

        self.weight = nn.Linear(self.in_features - 1, self.out_features - 1, bias=bias)
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.init_std = 0.02
        self.reset_parameters()

        # Scale for internal normalization
        if init_scale is not None:
            self.scale = nn.Parameter(torch.ones(()) * init_scale, requires_grad=learn_scale)
        else:
            self.scale = nn.Parameter(torch.ones(()) * 2.3, requires_grad=learn_scale)

    def forward(self, x):

        # changed the transformation to only include space values
        # need to check if the vastly different scale of the time element negatively affects results

        x_space = x.narrow(-1, 1, x.shape[-1] - 1)

        x_space = self.weight(x_space)

        if self.activation is not None:
            x_space = self.activation(x_space)
        x_space = self.dropout(x_space)


        if self.nheads>1:
            # Lorentz direct split
            x_space = x_space.view(x_space.size(0), x_space.size(1), self.nheads, self.out_features//self.nheads).transpose(1,2)

        if self.normalize:
            scale = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp()
            square_norm = (x_space * x_space).sum(dim=-1, keepdim=True)

            mask = square_norm <= 1e-10

            square_norm[mask] = 1
            unit_length = x_space/torch.sqrt(square_norm)
            x_space = scale*unit_length

            x_time = torch.sqrt(scale**2 + self.manifold.k + 1e-5)
            x_time = x_time.masked_fill(mask, self.manifold.k.sqrt())

            mask = mask==False
            x_space = x_space * mask

            x = torch.cat([x_time, x_space], dim=-1)
        else:
            x = self.manifold.add_time(x_space)

        return x

    def reset_parameters(self):
        nn.init.uniform_(self.weight.weight, -self.init_std, self.init_std)

        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzProjection(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, dropout=False):
        super(LorentzProjection, self).__init__()

        self.down = False

        if out_features >= in_features:
            # self.rotation = LorentzRotation_Up(manifold, in_features, out_features, if_regularize=False, if_projected=True)
            # self.rotation = orthogonal(self.rotation, "weight", orthogonal_map="cayley")

            # self.rotation = CayleyLinear(in_features-1, out_features-1, bias=False)
            self.rotation = LorentzRotationFixedNorm(manifold, in_features, out_features, if_regularize=False,
                                                      if_projected=True)
            self.down = True

        else:
            self.rotation = LorentzRotationFixedNorm(manifold, in_features, out_features, if_regularize=False,
                                               if_projected=True)
            self.down = True

        #self.boost = LorentzBoost(manifold, init_weight=1)
        self.boost = LorentzBoostScaleAlternate(manifold, in_features=out_features)
       #self.boost = LorentzBoostAlternate(manifold, dim=out_features)
        self.manifold = manifold

    def forward(self, input):

        if not self.down:
            xt = self.rotation(input[...,1:])
            xt = self.manifold.add_time(xt)
            #print("not down")
        else:
            xt = self.rotation(input)

        h = self.boost(xt)
        h = self.manifold.projx(h)
        # h = self.projection(h)

        return h


class LorentzMLRFF(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, dropout=False):
        super(LorentzMLRFF, self).__init__()

        self.manifold = manifold

        self.mlr = LorentzMLR(manifold, in_features, out_features)
        self.projection = nn.Linear(out_features, out_features-1)

    def forward(self, input):
        x = self.mlr(input)
        x = self.projection(x)

        x = self.manifold.add_time(x)

        return x


"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
#
# Copyright (c) 2018 Thalles Santos Silva
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
from audioop import add
import logging
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from network import Resnet
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights
from torchvision.utils import save_image
from scipy import ndimage as ndi
from kornia.morphology import dilation, erosion

import torchvision.models as models

from IPython import embed

from .pp import ContextDecoder

import labels

# NOTE(shjung13): These are for obtaining non-boundary masks
# We calculate the boundary mask by subtracting the eroded prediction map from the dilated one
# These are filters for erosion and dilation (L1)
selem = torch.ones((3, 3)).cuda()
selem_dilation = torch.FloatTensor(ndi.generate_binary_structure(2, 1)).cuda()

print(f'selem:\n\n{selem}')
print(f'selem_dilation:\n\n{selem_dilation}')

# NOTE(shjung13): Dilation filters to expand the boundary maps (L1)
d_k1 = torch.zeros((1, 1, 2 * 1 + 1, 2 * 1 + 1)).cuda()
d_k2 = torch.zeros((1, 1, 2 * 2 + 1, 2 * 2 + 1)).cuda()
d_k3 = torch.zeros((1, 1, 2 * 3 + 1, 2 * 3 + 1)).cuda()
d_k4 = torch.zeros((1, 1, 2 * 4 + 1, 2 * 4 + 1)).cuda()
d_k5 = torch.zeros((1, 1, 2 * 5 + 1, 2 * 5 + 1)).cuda()
d_k6 = torch.zeros((1, 1, 2 * 6 + 1, 2 * 6 + 1)).cuda()
d_k7 = torch.zeros((1, 1, 2 * 7 + 1, 2 * 7 + 1)).cuda()
d_k8 = torch.zeros((1, 1, 2 * 8 + 1, 2 * 8 + 1)).cuda()
d_k9 = torch.zeros((1, 1, 2 * 9 + 1, 2 * 9 + 1)).cuda()

d_ks = {1: d_k1, 2: d_k2, 3: d_k3, 4: d_k4, 5: d_k5, 6: d_k6, 7: d_k7, 8: d_k8, 9: d_k9}


for k, v in d_ks.items():
    v[:,:,k,k] = 1
    for i in range(k):
        v = dilation(v, selem_dilation)
    d_ks[k] = v.squeeze(0).squeeze(0)

    # print(f'dilation kernel at {k}:\n\n{d_ks[k]}')

from torch import Tensor

class GELU(nn.Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:
        :math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))

    Args:
        approximate (string, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['approximate']
    approximate: str

    def __init__(self, inplace=None) -> None:
        super(GELU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

def find_boundaries(label):
    """
    Calculate boundary mask by getting diff of dilated and eroded prediction maps
    """
    assert len(label.shape) == 4
    boundaries = (dilation(label.float(), selem_dilation) != erosion(label.float(), selem)).float()
    ### save_image(boundaries, f'boundaries_{boundaries.float().mean():.2f}.png', normalize=True)

    return boundaries

def expand_boundaries(boundaries, r=0):
    """
    Expand boundary maps with the rate of r
    """
    if r == 0:
        return boundaries
    expanded_boundaries = dilation(boundaries, d_ks[r])
    ### save_image(expanded_boundaries, f'expanded_boundaries_{r}_{boundaries.float().mean():.2f}.png', normalize=True)
    return expanded_boundaries


class BoundarySuppressionWithSmoothing(nn.Module):
    """
    Apply boundary suppression and dilated smoothing
    """
    def __init__(self, boundary_suppression=True, boundary_width=4, boundary_iteration=4,
                 dilated_smoothing=True, kernel_size=7, dilation=6):
        super(BoundarySuppressionWithSmoothing, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.boundary_suppression = boundary_suppression
        self.boundary_width = boundary_width
        self.boundary_iteration = boundary_iteration

        sigma = 1.0
        size = 7
        gaussian_kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
        gaussian_kernel /= np.sum(gaussian_kernel)
        gaussian_kernel = torch.Tensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)
        self.dilated_smoothing = dilated_smoothing

        self.first_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, bias=False)
        self.first_conv.weight = torch.nn.Parameter(torch.ones_like((self.first_conv.weight)))

        self.second_conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, dilation=self.dilation, bias=False)
        self.second_conv.weight = torch.nn.Parameter(gaussian_kernel)


    def forward(self, x, prediction=None):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_size = x.size()
        # B x 1 x H x W
        assert len(x.shape) == 4
        out = x
        if self.boundary_suppression:
            # obtain the boundary map of width 2 by default
            # this can be calculated by the difference of dilation and erosion
            boundaries = find_boundaries(prediction.unsqueeze(1))
            expanded_boundaries = None
            if self.boundary_iteration != 0:
                assert self.boundary_width % self.boundary_iteration == 0
                diff = self.boundary_width // self.boundary_iteration
            for iteration in range(self.boundary_iteration):
                if len(out.shape) != 4:
                    out = out.unsqueeze(1)
                prev_out = out
                # if it is the last iteration or boundary width is zero
                if self.boundary_width == 0 or iteration == self.boundary_iteration - 1:
                    expansion_width = 0
                # reduce the expansion width for each iteration
                else:
                    expansion_width = self.boundary_width - diff * iteration - 1
                # expand the boundary obtained from the prediction (width of 2) by expansion rate
                expanded_boundaries = expand_boundaries(boundaries, r=expansion_width)
                # invert it so that we can obtain non-boundary mask
                non_boundary_mask = 1. * (expanded_boundaries == 0)

                f_size = 1
                num_pad = f_size

                # making boundary regions to 0
                x_masked = out * non_boundary_mask
                x_padded = nn.ReplicationPad2d(num_pad)(x_masked)

                non_boundary_mask_padded = nn.ReplicationPad2d(num_pad)(non_boundary_mask)

                # sum up the values in the receptive field
                y = self.first_conv(x_padded)
                # count non-boundary elements in the receptive field
                num_calced_elements = self.first_conv(non_boundary_mask_padded)
                num_calced_elements = num_calced_elements.long()

                # take an average by dividing y by count
                # if there is no non-boundary element in the receptive field,
                # keep the original value
                avg_y = torch.where((num_calced_elements == 0), prev_out, y / num_calced_elements)
                out = avg_y

                # update boundaries only
                out = torch.where((non_boundary_mask == 0), out, prev_out)
                del expanded_boundaries, non_boundary_mask

            # second stage; apply dilated smoothing
            if self.dilated_smoothing == True:
                out = nn.ReplicationPad2d(self.dilation * 3)(out)
                out = self.second_conv(out)

            return out.squeeze(1)
        else:
            if self.dilated_smoothing == True:
                out = nn.ReplicationPad2d(self.dilation * 3)(out)
                out = self.second_conv(out)
            else:
                out = x

        return out.squeeze(1)


from network import coop, cocoop
import clip

class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-101', criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None, class_names=None):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.trunk = trunk
        self.num_classes = num_classes
        self.score_mode=args.score_mode
        self.enable_post_processing = (args.enable_boundary_suppression or args.enable_dilated_smoothing)
        if args.temp == 'fixed':
            self.T = args.T
            self.tau = args.tau
        elif args.temp == 'single':
            self.T = nn.Parameter(torch.tensor(self.args.T))
            self.tau = nn.Parameter(torch.tensor(self.args.tau))
        elif args.temp == 'classwise':
            self.T = nn.Parameter(torch.ones(len(class_names)) * self.args.T)
            self.tau = nn.Parameter(torch.ones(len(class_names)) * self.args.tau)
        

        assert class_names, "class names are not given"

        if trunk == 'shufflenetv2':
            channel_1st = 3
            channel_2nd = 24
            channel_3rd = 116
            channel_4th = 232
            prev_final_channel = 464
            final_channel = 1024
            resnet = models.shufflenet_v2_x1_0(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.maxpool)
            self.layer1 = resnet.stage2
            self.layer2 = resnet.stage3
            self.layer3 = resnet.stage4
            self.layer4 = resnet.conv5

            class Layer0(nn.Module):
                def __init__(self, iw):
                    super(Layer0, self).__init__()
                    self.layer = nn.Sequential(resnet.conv1, resnet.maxpool)
                    self.instance_norm_layer = resnet.instance_norm_layer1
                    self.iw = iw

                def forward(self, x):
                    x = self.layer(x)
                    return x

            class Layer4(nn.Module):
                def __init__(self, iw):
                    super(Layer4, self).__init__()
                    self.layer = resnet.conv5
                    self.instance_norm_layer = resnet.instance_norm_layer2
                    self.iw = iw

                def forward(self, x):

                    x = self.layer(x)
                    return x


            self.layer0 = Layer0(iw=0)
            self.layer1 = resnet.stage2
            self.layer2 = resnet.stage3
            self.layer3 = resnet.stage4
            self.layer4 = Layer4(iw=0)

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")
        elif trunk == 'mnasnet_05' or trunk == 'mnasnet_10':

            if trunk == 'mnasnet_05':
                resnet = models.mnasnet0_5(pretrained=True)
                channel_1st = 3
                channel_2nd = 16
                channel_3rd = 24
                channel_4th = 48
                prev_final_channel = 160
                final_channel = 1280

                print("# of layers", len(resnet.layers))
                self.layer0 = nn.Sequential(resnet.layers[0],resnet.layers[1],resnet.layers[2],
                                            resnet.layers[3],resnet.layers[4],resnet.layers[5],resnet.layers[6],resnet.layers[7])   # 16
                self.layer1 = nn.Sequential(resnet.layers[8], resnet.layers[9]) # 24, 40
                self.layer2 = nn.Sequential(resnet.layers[10], resnet.layers[11])   # 48, 96
                self.layer3 = nn.Sequential(resnet.layers[12], resnet.layers[13]) # 160, 320
                self.layer4 = nn.Sequential(resnet.layers[14], resnet.layers[15], resnet.layers[16])  # 1280
            else:
                resnet = models.mnasnet1_0(pretrained=True)
                channel_1st = 3
                channel_2nd = 16
                channel_3rd = 40
                channel_4th = 96
                prev_final_channel = 320
                final_channel = 1280

                print("# of layers", len(resnet.layers))
                self.layer0 = nn.Sequential(resnet.layers[0],resnet.layers[1],resnet.layers[2],
                                            resnet.layers[3],resnet.layers[4],resnet.layers[5],resnet.layers[6],resnet.layers[7])   # 16
                self.layer1 = nn.Sequential(resnet.layers[8], resnet.layers[9]) # 24, 40
                self.layer2 = nn.Sequential(resnet.layers[10], resnet.layers[11])   # 48, 96
                self.layer3 = nn.Sequential(resnet.layers[12], resnet.layers[13]) # 160, 320
                self.layer4 = nn.Sequential(resnet.layers[14], resnet.layers[15], resnet.layers[16])  # 1280

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")
        elif trunk == 'mobilenetv2':
            channel_1st = 3
            channel_2nd = 16
            channel_3rd = 32
            channel_4th = 64

            # prev_final_channel = 160
            prev_final_channel = 320

            final_channel = 1280
            resnet = models.mobilenet_v2(pretrained=True)
            self.layer0 = nn.Sequential(resnet.features[0],
                                        resnet.features[1])
            self.layer1 = nn.Sequential(resnet.features[2], resnet.features[3],
                                        resnet.features[4], resnet.features[5], resnet.features[6])
            self.layer2 = nn.Sequential(resnet.features[7], resnet.features[8], resnet.features[9], resnet.features[10])

            self.layer3 = nn.Sequential(resnet.features[11], resnet.features[12], resnet.features[13],
                                        resnet.features[14], resnet.features[15], resnet.features[16],
                                        resnet.features[17])
            self.layer4 = nn.Sequential(resnet.features[18])

            if self.variant == 'D':
                for n, m in self.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")
        else:
            channel_1st = 3
            channel_2nd = 64
            channel_3rd = 256
            channel_4th = 512
            prev_final_channel = 1024
            final_channel = 2048

            if trunk == 'resnet-18':
                channel_1st = 3
                channel_2nd = 64
                channel_3rd = 64
                channel_4th = 128
                prev_final_channel = 256
                final_channel = 512
                resnet = Resnet.resnet18()
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnet-50':
                resnet = Resnet.resnet50()
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnet-101': # three 3 X 3
                resnet = Resnet.resnet101(pretrained=False)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                              resnet.conv2, resnet.bn2, resnet.relu2,
                                              resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            elif trunk == 'resnet-152':
                resnet = Resnet.resnet152()
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnext-50':
                resnet = models.resnext50_32x4d(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'resnext-101':
                resnet = models.resnext101_32x8d(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'wide_resnet-50':
                resnet = models.wide_resnet50_2(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            elif trunk == 'wide_resnet-101':
                resnet = models.wide_resnet101_2(pretrained=True)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            else:
                raise ValueError("Not a valid network arch")

            self.layer0 = resnet.layer0
            self.layer1, self.layer2, self.layer3, self.layer4 = \
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            if self.variant == 'D':
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif self.variant == 'D4':
                for n, m in self.layer2.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (8, 8), (8, 8), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
            else:
                # raise 'unknown deepv3 variant: {}'.format(self.variant)
                print("Not using Dilation ")

        if self.variant == 'D':
            os = 8
        elif self.variant == 'D4':
            os = 4
        elif self.variant == 'D16':
            os = 16
        else:
            os = 32

        self.class_mean = None
        self.class_var = None

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                       output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        # self.to512 = nn.Sequential(
        #     nn.Conv2d(304, 512, kernel_size=3, padding=1, bias=False),
        #     Norm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
        #     Norm2d(512),
        #     nn.ReLU(inplace=True),
        # )
        if self.args.disable_le:
            self.final1 = nn.Sequential(
                nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
                Norm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                Norm2d(256),
                nn.ReLU(inplace=True))

            self.final2 = nn.Sequential(
                nn.Conv2d(256, num_classes, kernel_size=1, bias=True))
        elif self.args.stru:
            self.final1 = nn.Sequential(
                nn.Conv2d(304, 512, kernel_size=3, padding=1, bias=False),
                Norm2d(512),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.1),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                Norm2d(512),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.1),
                nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),  # Add another conv-layer to conpensate for the parameter loss of `self.final2`
            )
        else:
            self.final1 = nn.Sequential(
                nn.Conv2d(304, 512, kernel_size=3, padding=1, bias=False),
                Norm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                Norm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=True),  # Add another conv-layer to conpensate for the parameter loss of `self.final2`
                Norm2d(512),
                nn.ReLU(inplace=True)
            )

        if self.args.lang_aux:
            self.dsn = nn.Sequential(
                nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
                Norm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.dsn = nn.Sequential(
                nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
                Norm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        if self.enable_post_processing:
            self.multi_scale = BoundarySuppressionWithSmoothing(
                    boundary_suppression=args.enable_boundary_suppression,
                    boundary_width=args.boundary_width,
                    boundary_iteration=args.boundary_iteration,
                    dilated_smoothing=args.enable_dilated_smoothing,
                    kernel_size=args.smoothing_kernel_size,
                    dilation=args.smoothing_kernel_dilation)

        initialize_weights(self.dsn)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)

        if self.args.disable_le:
            initialize_weights(self.final2)

        # initialize_weights(self.to512)

        if not self.args.disable_le:
            # add l_model if necessary
            if args.context_optimize == "none":
                if args.orth_feat:
                    self.ref, self.others = labels.get_random_orth_matrix()
                else:
                    self.ref, self.others = labels.get_label_matrix(args.normalize)
            else:
                print("Building custom CLIP")
                clip_model, _ = clip.load("ViT-B/32", device='cpu')
                if self.args.context_optimize == 'coop':
                    self.l_model = coop.CustomCLIP(args, class_names, clip_model)
                elif self.args.context_optimize == 'cocoop':
                    self.l_model = cocoop.CustomCLIP(args, class_names, clip_model)
                else:
                    raise ValueError(args.context_optimize)


                print("Turning off gradients in both the image and the text encoder")
                for name, param in self.l_model.named_parameters():
                    if "prompt_learner" not in name:
                        param.requires_grad_(False)
                    else:
                        print("Prompt Learner's", name, "still have gradient.")


                # TODO: prompt learner's param init ?

                self.l_model.cuda()
            if self.args.post_prompt:
                self.gamma = nn.Parameter(torch.ones(512) * self.args.initial_size)
                self.feat_decoder = ContextDecoder()
                self.ref, self.others = self.ref.unsqueeze(0), self.others.unsqueeze(0)


    def force_init_decoder(self):
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)


    def set_statistics(self, mean, var):
        self.class_mean = mean
        self.class_var = var

    def forward(self, x, seg_gts=None, ood_gts=None, aux_gts=None, ignore_label=255, logits_guide=None):
        # ref, others = ref
        # DONE: get self.ref
        # embed()

        x_size = x.size()  # 800
        input_img = x
        x = self.layer0(x)  # 400
        x = self.layer1(x)  # 400
        low_level = x
        x = self.layer2(x)  # 100

        x = self.layer3(x)  # 100

        aux_out = x
        x = self.layer4(x)  # 100

        # embed()
        # represent = x

        x = self.aspp(x)

        dec0_up = self.bot_aspp(x)

        compressed_bottleneck = dec0_up

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)

        # torch.autograd.set_detect_anomaly(True)

        if self.args.disable_le:
            # original SML
            dec2 = self.final2(dec1)
        else:
            # with language embedding

            if self.args.normalize_feat:
                dec1 = F.normalize(dec1, dim=1)

            if self.args.post_prompt:
                downsample = nn.MaxPool2d(8, stride=8)
                abs_feat = downsample(dec1).flatten(start_dim=2).permute((0, 2, 1)) # B, S, C
                ref, others = self.ref.expand(dec1.shape[0], -1, -1), self.others.expand(dec1.shape[0], -1, -1) # B, K, C
                post_prompt_ref = self.feat_decoder(ref, abs_feat)
                post_prompt_others = self.feat_decoder(others, abs_feat)

            
            if self.args.context_optimize == 'none':
                # without prompt learning
                # assert self.args.logit_type == 'others_logsm'
                if self.args.logit_type == "simple_prod":
                    dec2 = torch.einsum("abcd,eb->aecd", (dec1, self.ref))
                else:
                    if not self.args.post_prompt:
                        # without post prompt
                        ref, others = self.ref, self.others
                        assert len(ref) == self.num_classes, f"Expecting {self.num_class}, got {len(ref)}"
                        dec2_class = torch.einsum("abcd,eb->aecd", (dec1, ref))
                        dec2_others = torch.einsum("abcd,eb->aecd", (dec1, others)).expand(dec2_class.shape)
                        dec2_cat = torch.stack((dec2_class, dec2_others), 1)
                        log_softmax = torch.nn.LogSoftmax(dim=1)
                    else:
                        # with post prompt
                        ref, others = ref + post_prompt_ref * self.gamma, others + post_prompt_others * self.gamma
                        dec2_class = torch.einsum("abcd,aeb->aecd", (dec1, ref))
                        dec2_others = torch.einsum("abcd,aeb->aecd", (dec1, others)).expand(dec2_class.shape)
                        dec2_cat = torch.stack((dec2_class, dec2_others), 1)
                        log_softmax = torch.nn.LogSoftmax(dim=1)

                    if self.args.temp == 'fixed' or self.args.temp == 'single':
                        dec2 = log_softmax(dec2_cat / self.T)[:,0]
                    else:
                        dec2 = log_softmax(dec2_cat / self.T.view(19, 1, 1))[:,0]

                   
                

            elif self.args.context_optimize == 'coop':
                # with fixed prompt
                if self.args.logit_type == "simple_prod":
                    dec2 = torch.einsum("abcd,eb->aecd", (dec1, self.l_model()))
                elif self.args.logit_type == "others_logsm":
                    ref = self.l_model()
                    ref, others = ref[:-1], ref[-1]
                    assert len(ref) == self.num_classes, f"Expecting {self.num_class}, got {len(ref)}"
                    dec2_class = torch.einsum("abcd,eb->aecd", (dec1, ref))
                    dec2_others = torch.einsum("abcd,eb->aecd", (dec1, others.view(1, 512))).expand(dec2_class.shape)
                    dec2_cat = torch.stack((dec2_class, dec2_others), 1)
                    log_softmax = torch.nn.LogSoftmax(dim=1)
                    dec2 = log_softmax(dec2_cat / self.T)[:,0]
                
                else:
                    raise ValueError

            elif self.args.context_optimize == 'cocoop':
                # with conditional prompt
                assert self.args.logit_type == 'others_logsm'
                if "pixelwise_prompt" in self.args and self.args.pixelwise_prompt:
                    ref = self.l_model(compressed_bottleneck)
                    raise NotImplementedError
                else:
                    ref = self.l_model(input_img)
                    ref, others = ref[:, :-1, :], ref[:, -1:, :]
                    assert len(ref[0]) == self.num_classes, f"Expecting {self.num_classes}, got {ref.shape}"

                    dec2_class = torch.einsum("abcd,aeb->aecd", (dec1, ref))
                    dec2_others = torch.einsum("abcd,aeb->aecd", (dec1, others)).expand(dec2_class.shape)
                    dec2_cat = torch.stack((dec2_class, dec2_others), 1)
                    log_softmax = torch.nn.LogSoftmax(dim=1)
                    dec2 = log_softmax(dec2_cat / self.T)[:,0]
            else:
                raise ValueError(f"self.args.context_optimize {self.args.context_optimize}")

    # calculate upsampled pixelwise logits
        main_out = Upsample(dec2, x_size[2:])
        def add_tau(_mo):
            if self.args.temp == 'fixed' or self.args.temp == 'single':
                return _mo / self.tau
            else:
                return _mo / self.tau.view(19, 1, 1)
    
        if self.args.enable_main_out_temp:
            main_out = add_tau(main_out)


        if self.score_mode == 'msp':
            anomaly_score, prediction = nn.Softmax(dim=1)(main_out.detach()).max(1)

        elif self.score_mode == 'max_logit':
            anomaly_score, prediction = main_out.detach().max(1)

        elif self.score_mode == 'standardized_max_logit':
            if self.class_mean is None or self.class_var is None:
                raise Exception("Class mean and var are not set!")
            anomaly_score, prediction = main_out.detach().max(1)
            if logits_guide is not None:
                print("Pred from guide")
                _, prediction = logits_guide.max(1)
                del _
            for c in range(self.num_classes):
                anomaly_score = torch.where(prediction == c,
                                            (anomaly_score - self.class_mean[c]) / np.sqrt(self.class_var[c]),
                                            anomaly_score)
        elif self.score_mode == 'max_standardized_logit':
            pass
        elif self.score_mode == 'all_mix':
            with torch.no_grad():
                # embed()
                if logits_guide is not None:
                    print("calculate with guided map")
                    prob_map = nn.Softmax(dim=1)(logits_guide / self.args.inf_temp)
                else:
                    prob_map = nn.Softmax(dim=1)(main_out / self.args.inf_temp)
                # print(self.args.inf_temp)
                max_val, prediction = prob_map.max(1)
                del max_val
                anomaly_score = main_out.clone()
                for c in range(self.num_classes):
                    anomaly_score[:, c, ...] = (anomaly_score[:, c, ...] - self.class_mean[c]) / np.sqrt(self.class_var[c])
                anomaly_score = torch.einsum('bchw,bchw->bhw', (anomaly_score, prob_map))
        elif self.score_mode == 'k_mix':
            with torch.no_grad():
                # embed()
                prob_map = nn.Softmax(dim=1)(main_out)
                max_val, prediction = prob_map.max(1)
                del max_val
                anomaly_score = main_out.clone()
                for c in range(self.num_classes):
                    anomaly_score[:, c, ...] = (anomaly_score[:, c, ...] - self.class_mean[c]) / np.sqrt(self.class_var[c])
                anomaly_score_accu = torch.zeros_like(prediction, dtype=torch.float32, device='cuda')
                prob_accu = torch.zeros_like(prediction, dtype=torch.float32, device='cuda')
                null = torch.zeros_like(prediction, dtype=torch.float32, device='cuda').unsqueeze(1)
                for i in range(self.args.top_k):
                    # print(i)
                    mv, mi = prob_map.max(1)
                    mi = mi.unsqueeze(1)
                    prob_accu += mv
                    anomaly_score_accu += torch.gather(anomaly_score, 1, mi).squeeze(1) * mv
                    prob_map.scatter_(1, mi, null)
                anomaly_score = anomaly_score_accu / prob_accu
        else:
            raise Exception(f"Not implemented score mode {self.score_mode}!")

        if self.enable_post_processing:
            with torch.no_grad():
                anomaly_score = self.multi_scale(anomaly_score, prediction)


        # embed()
        if self.training:
            if self.criterion and (seg_gts is not None):
                if not self.args.enable_main_out_temp:
                    main_out = add_tau(main_out)
                loss1 = self.criterion(main_out, seg_gts)
            else:
                loss1 = None
            if self.criterion_aux and (aux_gts is not None):
                # embed()
                aux_out = self.dsn(aux_out)
                if self.args.lang_aux:
                    aux_out = torch.einsum("abcd,eb->aecd", (aux_out, self.ref))

                if aux_gts.dim() == 1:
                    aux_gts = seg_gts
                aux_gts = aux_gts.unsqueeze(1).float()
                aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
                aux_gts = aux_gts.squeeze(1).long()
                loss2 = self.criterion_aux(aux_out, aux_gts)
            else:
                loss2 = None

            return loss1, loss2, main_out.max(1)
        else:
            return main_out, anomaly_score


def get_final_layer(model):
    unfreeze_weights(model.final)
    return model.final


def DeepR18V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 18 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-18")
    return DeepV3Plus(num_classes, trunk='resnet-18', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepR50V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepR50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD_OS8(args, num_classes, criterion, criterion_aux, **kw):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args, **kw)


def DeepR152V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 152 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-152")
    return DeepV3Plus(num_classes, trunk='resnet-152', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)



def DeepResNext50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnext 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-50 32x4d")
    return DeepV3Plus(num_classes, trunk='resnext-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepResNext101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-101 32x8d")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3Plus(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3Plus(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepWideResNet101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3Plus(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3Plus(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepResNext101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepResNext101V3PlusD_OS4(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D4', skip='m1', args=args)

def DeepShuffleNetV3PlusD_OS32(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepMNASNet05V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_0_5")
    return DeepV3Plus(num_classes, trunk='mnasnet_05', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMNASNet10V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_1_0")
    return DeepV3Plus(num_classes, trunk='mnasnet_10', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)


def DeepShuffleNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3Plus(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3Plus(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepShuffleNetV3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

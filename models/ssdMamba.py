"""
Copyright (c) 2017 Max deGroot, Ellis Brown
Released under the MIT license
https://github.com/amdegroot/ssd.pytorch
Updated by: Takuya Mouri
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import *
from data import voc, coco
import os
from models.vmamba import mamba_vision_B, ConvBlock
from models.extraLayers import ExtraLayers

class SSDMamba(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, size, base, head, num_classes, mamba_out):
        super(SSDMamba, self).__init__()
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)

        # self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.mamba = base
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)

        self.extras = ExtraLayers(dim=1024, depths=3, num_heads=[2,2],
                                  window_size=[8,8], mlp_ratio=4.,
                                  drop_path_rate=0.3) 

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        # PyTorch1.5.0 support new-style autograd function
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        self.mamba_out = mamba_out
        for i, i_layer in enumerate(self.mamba_out):
            layer = nn.BatchNorm2d(i_layer)
            layer_name = f'norm_mamba{i}'
            self.add_module(layer_name, layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, phase):
        """Applies network layers and ops on input image(s) x.

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        loc = list()
        conf = list()

        sources = self.mamba(x)

        for i in range(len(self.mamba_out)):
            norm_layer = getattr(self, f'norm_mamba{i}')
            x_out = norm_layer(sources[i])
            sources[i] = x_out

        extras = self.extras(sources[-1])
        sources = sources + extras

        x = sources[-1]

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if phase == "test":
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output, sources

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



def multibox(mamba_out, cfg, extras, num_classes):
    loc_layers = []
    conf_layers = []
    mamba_source = [i for i in range(len(mamba_out))]
    for k, v in enumerate(mamba_source):
        loc_layers += [nn.Conv2d(mamba_out[v],
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(mamba_out[v],
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    
    for k, v in enumerate(extras):
        loc_layers += [nn.Conv2d(v, cfg[-(k+1)]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v, cfg[-(k+1)]
                                  * num_classes, kernel_size=3, padding=1)]
        
    return (loc_layers, conf_layers)


extras = {
    '224': [512,256],
}
mbox = {
    '224': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}


def build_ssdMamba(size=224, num_classes=21):
    base_, mamba_out = mamba_vision_B(pretrained=True)

    head_ = multibox(mamba_out,
                    mbox[str(size)], 
                    extras[str(size)],
                    num_classes)

    return SSDMamba(size, base_, head_, num_classes, mamba_out)

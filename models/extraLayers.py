import torch
import torch.nn as nn
try:
    from models.vmamba import MambaVisionLayer, DropPath
except:
    from vmamba import MambaVisionLayer, DropPath



class ConvBlock(nn.Module):
    def __init__(self, dim,
                 drop_path=0.,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim//4, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim//4, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop_path(x)
        return x


class ExtraLayers(nn.Module):
    def __init__(self, dim, depths, num_heads, window_size,
                 mlp_ratio, drop_path_rate):
        super().__init__()

        self.dim = dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        
        self.dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,\
                                                 self.depths)]

        self.layer1 = nn.ModuleList()
        self.layer1.append(MambaVisionLayer(dim=self.dim,
                            depth=self.depths,
                            num_heads=self.num_heads[0],
                            window_size=self.window_size[0],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            conv=False,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=self.dpr,
                            downsample=True,
                            layer_scale=1e-5,
                            layer_scale_conv=None,
                            transformer_blocks=list(range(self.depths//2+1, self.depths)) if self.depths%2!=0 else list(range(self.depths//2, self.depths)),
                            ))
        self.layer1.append(ConvBlock(dim=self.dim*2))

        self.layer2 = nn.ModuleList()
        self.layer2.append(MambaVisionLayer(dim=self.dim//2,
                            depth=self.depths,
                            num_heads=self.num_heads[1],
                            window_size=self.window_size[1],
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            conv=False,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=self.dpr,
                            downsample=True,
                            layer_scale=1e-5,
                            layer_scale_conv=None,
                            transformer_blocks=list(range(self.depths//2+1, self.depths)) if self.depths%2!=0 else list(range(self.depths//2, self.depths)),
                            ))
        self.layer2.append(ConvBlock(dim=self.dim))

    def forward(self, x):
        sources = []

        x = self.layer1[1](self.layer1[0](x))
        sources.append(x)

        x = self.layer2[1](self.layer2[0](x))
        sources.append(x)

        return sources
    

import torch
from torch import Tensor
import torch.nn as nn
from einops import rearrange
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_fea: int,
        out_fea: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 1,
        relu_slop: float = 0.2,
    ) -> None:
        """
        Standard convolution operation

        Args:
            in_fea (int): Number of channels of input
            out_fea (int): Number of channels of output
            kernel_size (int or tuple): Size of the convolution kernel, default: 3
            stride (int or tuple): Step size of the convolution, default: 1
            padding (int or tuple): Zero-fill width, default: 1
            relu_slop (float): Parameters of relu, default: 0.2
        """
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_fea, out_fea, kernel_size, stride, padding))
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(
        self,
        in_fea: int,
        out_fea: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 1,
    ) -> None:
        """
        Convolution operation of the output part

        Args:
            in_fea (int): Number of channels of input
            out_fea (int): Number of channels of output
            kernel_size (int or tuple): Size of the convolution kernel, default: 3
            stride (int or tuple): Step size of the convolution, default: 1
            padding (int or tuple): Zero-fill width, default: 1
        """
        super(ConvBlock_Tanh, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_fea, out_fea, kernel_size, stride, padding))
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_fea: int,
        out_fea: int,
        kernel_size: int | Tuple[int, int] = 2,
        stride: int | Tuple[int, int] = 2,
        padding: int | Tuple[int, int] = 0,
        output_padding: int | Tuple[int, int] = 0,
        relu_slop: float = 0.2,
    ) -> None:
        """
        Deconvolution operation

        Args:
            in_fea (int): Number of channels of input
            out_fea (int): Number of channels of output
            kernel_size (int or tuple): Size of the convolution kernel, default: 2
            stride (int or tuple): Step size of the convolution, default: 2
            padding (int or tuple): Zero-fill width of input, default: 0
            output_padding (int or tuple): Zero-fill width of output, default: 0
            relu_slop (float): Parameters of relu, default: 0.2
        """
        super(DeconvBlock, self).__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_fea, out_fea, kernel_size, stride, padding, output_padding=output_padding))
        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ResBlock(nn.Module):
    """
    Bottleneck in ResNet

    Args:
        inplanes (int): Number of channels of input
        planes (int): Number of channels of bottleneck
        stride (int or tuple): Step size of the convolution, default: 1
        downsample (Optional[nn.Module]): Downsampling layer for input tensor, default: None
    """

    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int | Tuple[int, int] = 1, downsample: nn.Module = None
    ) -> None:
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1)
        self.bn1 = nn.BatchNorm2d(planes)

        if isinstance(stride, tuple):
            padding = ((stride[0] + 1) // 2, (stride[1] + 1) // 2)
        else:
            padding = (stride + 1) // 2
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet50 for FuteFWI

    Args:
        in_fea (int): Number of shots of seismic data
    """

    def __init__(self, in_fea: int) -> None:
        super(ResNet, self).__init__()
        self.expansion = ResBlock.expansion
        self.in_channels = 32
        self.layer1 = self._make_initial_layer(in_fea, self.in_channels)
        self.layer2 = self._make_res_layer(ResBlock, channels=32, stride=2, num_layers=3)
        self.layer3 = self._make_res_layer(ResBlock, channels=64, stride=2, num_layers=4)
        self.layer4 = self._make_res_layer(ResBlock, channels=128, stride=2, num_layers=6)
        self.layer5 = self._make_res_layer(ResBlock, channels=256, stride=(2, 1), num_layers=3)
        self.pad = nn.ConstantPad2d((0, 0, -2, -3), 0)

    def _make_initial_layer(self, in_fea: int, out_fea: int) -> nn.Module:
        """Create the initial layer consisting of Conv2d, BatchNorm, ReLU and MaxPool2d."""
        return nn.Sequential(
            nn.Conv2d(in_fea, out_fea, kernel_size=7, stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_fea),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
        )

    def _make_res_layer(
        self, block: nn.Module, channels: int, stride: int | Tuple[int, int], num_layers: int
    ) -> nn.Module:
        """Create a ResNet layer composed of blocks."""
        downsample = self._make_downsample_layer(block, channels, stride)

        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion

        for _ in range(1, num_layers):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def _make_downsample_layer(
        self, block: nn.Module, channels: int, stride: int | Tuple[int, int]
    ) -> Optional[nn.Module]:
        """Create a downsample layer if necessary."""
        if stride != 1 or self.in_channels != channels * block.expansion:
            return nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels * block.expansion),
            )
        return None

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)  # (None, 32, 500, 64)
        x = self.layer2(x)  # (None, 64, 250, 32)
        x = self.layer3(x)  # (None, 128, 125, 16)
        x = self.pad(x)  # (None, 256, 120, 16)
        x = self.layer4(x)  # (None, 512, 60, 16)
        x = self.layer5(x)  # (None, 1024, 30, 8)
        return x


class FeedForward(nn.Module):
    """
    Feed forward network in transformer

    Args:
        dim (int): Number of dimensions of input
        mlp_dim (int): Number of dimensions in MLP
        dropout (float): dropout in MLP, default: 0
    """

    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """
    Single layer in transformer

    Args:
        dim (int): Number of dimensions of input
        heads (int): Number of heads of multiheaded attention, default: 8
        dropout (float): Dropout of module, default: 0
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        dim_head = dim // heads

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer module in FuteFWI

    Args:
        dim (int): Number of dimensions of input sequence
        num_layers (int): Number of layers of transformer module
        num_heads (int): Number of heads in multiheaded attention
        mlp_dim (int): Number of dimensions in MLP
        dropout (float): Dropout in transformer module, default: 0
    """

    def __init__(self, dim: int, num_layers: int, num_heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([Attention(dim, num_heads, dropout), FeedForward(dim, mlp_dim, dropout)]))

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class FuteFWI(nn.Module):
    def __init__(self, hidden_size: int = 768, num_layers: int = 4, num_heads: int = 12) -> None:
        """
        Network architecture of FuteFWI

        Args:
            hidden_size (int): Number of dimensions of input sequence
            num_layers (int): Number of layers of transformer module
            num_heads (int): Number of heads of multiheaded attention
        """
        super(FuteFWI, self).__init__()
        self.resnet = ResNet(in_fea=5)

        self.patch_embedding = nn.Conv2d(1024, hidden_size, kernel_size=1, stride=1)
        self.pos_embedding = nn.Parameter(torch.randn(1, 30 * 8, hidden_size))
        self.transformer = Transformer(
            dim=hidden_size, num_layers=num_layers, num_heads=num_heads, mlp_dim=4 * hidden_size, dropout=0
        )

        self.deconv1_1 = DeconvBlock(hidden_size, 512, kernel_size=4, stride=2, padding=1)
        self.deconv1_2 = ConvBlock(512, 256)
        self.deconv2_1 = DeconvBlock(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(128, 64)
        self.deconv3_1 = DeconvBlock(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(32, 16)
        self.pad = nn.ConstantPad2d((-13, -13, -10, -10), 0)
        self.deconv4 = ConvBlock_Tanh(16, 1)

    def forward(self, x: Tensor) -> Tensor:
        """x: (None, 5, 2000, 70)"""
        # encoder part
        x = self.resnet(x)  # (None, 1024, 30, 8)

        # attention part
        x = self.patch_embedding(x)  # (None, 768, 30, 8)
        x = rearrange(x, "b c h w -> b (h w) c")  # (None, 240, 768)
        x += self.pos_embedding
        x = self.transformer(x)  # (None, 240, 1024)
        x = rearrange(x, "b (h w) c -> b c h w", h=20, w=12)  # (None, 768, 20, 12)

        # decoder part
        x = self.deconv1_1(x)  # (None, 512, 40, 24)
        x = self.deconv1_2(x)  # (None, 256, 40, 24)
        x = self.deconv2_1(x)  # (None, 128, 80, 48)
        x = self.deconv2_2(x)  # (None, 64, 80, 48)
        x = self.deconv3_1(x)  # (None, 32, 160, 96)
        x = self.deconv3_2(x)  # (None, 32, 160, 96)
        x = self.pad(x)  # (None, 16, 140, 70)
        x = self.deconv4(x)  # (None, 1, 140, 70)
        return x

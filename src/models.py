import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SelfAttentionCNN(nn.Module):
    """ A simple self-attention layer """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key   = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, c, width, height = x.size()
        q = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, width * height)
        v = self.value(x).view(batch_size, -1, width * height)

        attention_map = F.softmax(torch.bmm(q, k), dim=-1)

        out = torch.bmm(v, attention_map.permute(0, 2, 1))
        out = out.view(batch_size, c, width, height)

        return self.gamma * out + x # Add skip connection

class UpsampleBlock(nn.Module):
    """An upsampling block using Conv2d and PixelShuffle."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The Conv2d layer produces 4x the channels for a 2x upscale
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2) # Rearranges channels to upscale by 2x
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pixel_shuffle(self.conv(x)))

class GatedConv2d(nn.Module):
    """
    A Gated Convolutional Layer.
    It learns a dynamic feature mask for each channel at every location.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        # Convolution for the features
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )
        # Convolution for the gating mechanism
        self.conv_gate = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation
        )

    def forward(self, x):
        # Get the features and the gate
        features = self.conv_feature(x)
        gate = torch.sigmoid(self.conv_gate(x)) # Gate values are between 0 and 1

        # Element-wise multiplication to apply the learned gate
        return features * gate

class GatedResidualBlock(nn.Module):
    """A Residual Block that uses Gated Convolutions."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation

        # Replace nn.Conv2d with GatedConv2d
        self.conv1 = GatedConv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = GatedConv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual  # Residual connection
        out = self.relu(out)
        return out

class UNetSR(nn.Module):
    """
    A U-Net architecture with corrected channel dimensions for the decoder.
    """
    def __init__(self, in_channels=4, out_channels=3, num_channels=64):
        super().__init__()

        # --- Initial Convolution ---
        self.init_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)

        # --- Encoder Path ---
        self.enc1 = GatedResidualBlock(num_channels, dilation=1)
        self.enc2 = GatedResidualBlock(num_channels, dilation=1)
        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck with Dilation and Attention ---
        self.bottleneck = nn.Sequential(
            GatedResidualBlock(num_channels, dilation=2),
            # SelfAttention(num_channels), # Add attention layer
            GatedResidualBlock(num_channels, dilation=4)
        )

        # --- Decoder Path ---
        self.upconv2 = UpsampleBlock(num_channels, num_channels)
        # Input channels = upsampled (64) + skip connection from e2 (64) = 128
        self.dec2 = GatedResidualBlock(num_channels * 2, dilation=1)

        self.upconv1 = UpsampleBlock(num_channels * 2, num_channels)
        # Input channels = upsampled (64) + skip connection from e1 (64) = 128
        self.dec1 = GatedResidualBlock(num_channels * 2, dilation=1)

        # --- Final Output Layer ---
        # The input to this layer comes from dec1, which outputs 128 channels.
        self.out_conv = nn.Conv2d(num_channels * 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Initial feature extraction
        x0 = self.init_conv(x)

        # Encoder
        e1 = self.enc1(x0)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder with Skip Connections
        d2 = self.upconv2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final Output
        out = self.out_conv(d1)

        return torch.tanh(out)


# ============================================
# 1. GENERATOR - Residual Refinement Network
# ============================================

class ResidualBlock(nn.Module):
    """Residual block with spectral normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x = self.norm2(self.conv2(x))
        return x + residual

class SelfAttentionGAN(nn.Module):
    """Self-attention layer for capturing long-range dependencies"""
    def __init__(self, channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(channels, channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(channels, channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        v = self.value(x).view(b, -1, h * w)

        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        return self.gamma * out + x

class RefinementGenerator(nn.Module):
    """Generator that refines CNN output"""
    def __init__(self, in_channels=7, out_channels=3, base_channels=64):
        super().__init__()

        # Input: CNN output (3) + original masked (3) + mask (1) = 7 channels
        self.encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_channels, 7, 1, 3)),
            nn.LeakyReLU(0.2, True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.down2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, True)
        )

        # Residual blocks with attention
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            SelfAttentionGAN(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        # Upsampling
        self.up1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.up2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, True)
        )

        # Output residual
        self.output = nn.Sequential(
            spectral_norm(nn.Conv2d(base_channels, out_channels, 7, 1, 3)),
            nn.Tanh()
        )

        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, cnn_output, masked_image, mask):
        """
        Args:
            cnn_output: CNN's coarse prediction [B, 3, H, W]
            masked_image: Original image with mask applied [B, 3, H, W]
            mask: Binary mask [B, 1, H, W]
        """
        # Concatenate inputs
        x = torch.cat([cnn_output, masked_image, mask], dim=1)

        # Encode
        x = self.encoder(x)

        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(d1)

        # Process with residual blocks
        x = self.res_blocks(d2)

        # Upsample with skip connections
        x = self.up1(x)
        x = self.up2(x + d1)  # Skip connection

        # Generate residual
        residual = self.output(x)

        # Add weighted residual to CNN output
        refined = cnn_output + self.residual_weight * residual

        # Ensure output is in [-1, 1]
        return torch.tanh(refined)

# ============================================
# 2. DISCRIMINATOR - Multi-scale PatchGAN
# ============================================

class MultiscaleDiscriminator(nn.Module):
    """Multi-scale discriminator for better gradient flow"""
    def __init__(self, in_channels=3, base_channels=64, num_scales=3):
        super().__init__()

        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        for _ in range(num_scales):
            self.discriminators.append(self._make_discriminator(in_channels, base_channels))

        self.downsample = nn.AvgPool2d(2)

    def _make_discriminator(self, in_channels, base_channels):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1)),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(base_channels * 8, 1, 4, 1, 1))
        )

    def forward(self, x):
        outputs = []

        for i in range(self.num_scales):
            outputs.append(self.discriminators[i](x))
            if i < self.num_scales - 1:
                x = self.downsample(x)

        return outputs
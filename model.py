"""
ResUNet-a: PyTorch Implementation
Converted from TensorFlow/Keras version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block with BatchNorm before activation (ResNet v2 style)"""
    def __init__(self, channels, kernel_size=3, padding=1, dilation=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 
                              padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 
                              padding=padding, dilation=dilation, bias=True)
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class ResBlockAtrous(nn.Module):
    """Residual block with atrous convolution (dilated convolution)"""
    def __init__(self, channels, dilations=[1, 3, 15, 31]):
        super(ResBlockAtrous, self).__init__()
        self.blocks = nn.ModuleList()
        for dilation in dilations:
            padding = dilation  # For kernel_size=3
            self.blocks.append(ResBlock(channels, dilation=dilation, padding=padding))
        
    def forward(self, x):
        outputs = []
        for block in self.blocks:
            outputs.append(block(x))
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)


class ResUNetA(nn.Module):
    """
    ResUNet-a: Deep learning framework for semantic segmentation
    PyTorch implementation converted from TensorFlow/Keras version
    """
    def __init__(self, in_channels=3, out_channels=2, features=[32, 64, 128, 256, 512], 
                 use_atrous=False, layer_norm='batch'):
        super(ResUNetA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_atrous = use_atrous
        
        # Store original features for decoder
        self.features = features.copy()
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        # Initial convolution
        self.encoder_blocks.append(ConvBlock(in_channels, features[0]))
        
        # Encoder blocks with downsampling
        for i in range(1, len(features)):
            # Downsampling
            self.pool_layers.append(nn.MaxPool2d(2, 2))
            # Convolution block
            self.encoder_blocks.append(ConvBlock(features[i-1], features[i]))
            # Residual blocks (simplified - no atrous for now)
            self.res_blocks.append(ResBlock(features[i]))
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        # Reverse features for decoder
        decoder_features = features[::-1]
        
        for i in range(len(decoder_features) - 1):
            # Upsampling
            self.upconvs.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvBlock(decoder_features[i], decoder_features[i+1])
            ))
            # Decoder block (concatenates with skip connection)
            self.decoder_blocks.append(ConvBlock(decoder_features[i+1] * 2, decoder_features[i+1]))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder path with skip connections
        skip_connections = []
        
        # Initial block
        x = self.encoder_blocks[0](x)
        skip_connections.append(x)
        
        # Encoder blocks
        for i in range(1, len(self.encoder_blocks)):
            x = self.pool_layers[i-1](x)
            x = self.encoder_blocks[i](x)
            x = self.res_blocks[i-1](x)
            skip_connections.append(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        # Start from the deepest skip connection (index 0 after reverse)
        # The first upconv should connect with skip_connections[1] (second deepest)
        for i, (upconv, dec_block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            # Skip connection: use i+1 to skip the deepest feature (which is x itself)
            # Ensure we don't go out of bounds
            skip_idx = i + 1
            if skip_idx >= len(skip_connections):
                raise IndexError(f"Skip connection index {skip_idx} out of range. "
                               f"Available skip connections: {len(skip_connections)}, "
                               f"Decoder blocks: {len(self.decoder_blocks)}")
            skip = skip_connections[skip_idx]
            # Ensure same spatial dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            # Ensure channel dimensions match for concatenation
            if x.shape[1] != skip.shape[1]:
                # This shouldn't happen, but add check for safety
                raise ValueError(f"Channel mismatch: x has {x.shape[1]} channels, "
                              f"skip has {skip.shape[1]} channels")
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)
        
        # Final output
        x = self.final_conv(x)
        return x


def init_weights(m):
    """
    Initialize model weights to prevent gradient vanishing/explosion
    Uses Kaiming initialization for ReLU networks (He initialization)
    
    This function is applied to all modules in the model to ensure proper initialization.
    Proper initialization is crucial for:
    - Preventing gradient vanishing (weights too small)
    - Preventing gradient explosion (weights too large)
    - Ensuring stable training from the start
    """
    if isinstance(m, nn.Conv2d):
        # Kaiming (He) initialization for ReLU networks
        # This is the recommended initialization for networks using ReLU activation
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            # Initialize bias to zero
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        # BatchNorm: weight=1, bias=0 (standard initialization)
        # This ensures BatchNorm starts in a neutral state
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Linear layers: use Kaiming initialization
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_resunet_a(in_channels=3, out_channels=2, depth=7, layer_norm='batch'):
    """
    Factory function to create ResUNet-a model with proper weight initialization
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output classes (default: 2 for binary segmentation)
        depth: Network depth (default: 7)
        layer_norm: Normalization type ('batch', 'instance', 'layer')
    
    Returns:
        ResUNetA model instance with properly initialized weights
    """
    # Calculate feature sizes based on depth
    # Typical ResUNet-a uses: [32, 64, 128, 256, 512]
    base_features = 32
    features = []
    for i in range(min(depth // 2 + 1, 5)):
        features.append(base_features * (2 ** i))
    
    # Ensure we have at least 5 levels
    while len(features) < 5:
        features.append(features[-1] * 2 if features else base_features)
    
    # Create model
    model = ResUNetA(in_channels=in_channels, 
                     out_channels=out_channels, 
                     features=features[:5],
                     layer_norm=layer_norm)
    
    # Apply weight initialization - CRITICAL for preventing gradient vanishing!
    # This ensures all layers start with appropriate weights
    model.apply(init_weights)
    
    return model


if __name__ == "__main__":
    # Test model
    model = create_resunet_a(in_channels=3, out_channels=2)
    x = torch.randn(1, 3, 256, 256)
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

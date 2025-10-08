"""
TGVAnn (Texture-Guided Visual Attention Network)
Main architecture for crop leaf disease detection using RGB-Texture fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .tgva_attention import TGVABlock


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class TGVAnn(nn.Module):
    """
    Texture-Guided Visual Attention Network (TGVAnn)
    
    Architecture:
    - Input: RGB (256×256×3) and Texture (256×256×1)
    - Dual-stream ResNet with TGVA fusion at ResBlock-2
    - Output: C classes
    
    Key Parameters:
    - num_classes: Number of disease classes
    - input_channels_rgb: RGB channels (default: 3)
    - input_channels_texture: Texture channels (default: 1)
    """
    
    def __init__(self, num_classes=3, input_channels_rgb=3, input_channels_texture=1):
        super(TGVAnn, self).__init__()
        
        self.num_classes = num_classes
        
        # RGB Stream
        self.rgb_conv1 = nn.Conv2d(input_channels_rgb, 64, kernel_size=7, 
                                   stride=2, padding=3, bias=False)
        self.rgb_bn1 = nn.BatchNorm2d(64)
        self.rgb_relu = nn.ReLU(inplace=True)
        self.rgb_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Texture Stream
        self.texture_conv1 = nn.Conv2d(input_channels_texture, 64, kernel_size=7,
                                       stride=2, padding=3, bias=False)
        self.texture_bn1 = nn.BatchNorm2d(64)
        self.texture_relu = nn.ReLU(inplace=True)
        self.texture_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResBlock-1 (64×64×64)
        self.rgb_layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.texture_layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        
        # ResBlock-2 (32×32×128) - Before TGVA
        self.rgb_layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.texture_layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        
        # TGVA Attention at ResBlock-2
        self.tgva_block = TGVABlock(in_channels=128, d_model=128, d_ff=512,
                                    p_attn=0.10, p_ff=0.10)
        
        # ResBlock-3 (16×16×256) - After TGVA fusion
        self.layer3 = self._make_layer(256, 256, blocks=2, stride=2)
        
        # ResBlock-4 (8×8×512)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc = nn.Linear(512, 512)
        self.fc_relu = nn.ReLU(inplace=True)
        self.fc_dropout = nn.Dropout(0.5)
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a ResNet layer with specified number of blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_input, texture_input):
        """
        Forward pass
        
        Args:
            rgb_input: [B, 3, 256, 256] - RGB images
            texture_input: [B, 1, 256, 256] - Texture images
        
        Returns:
            output: [B, num_classes] - Class logits
            attention_weights: [B, 1024, 1024] - TGVA attention weights
        """
        # RGB Stream - Initial layers
        rgb_x = self.rgb_conv1(rgb_input)  # [B, 64, 128, 128]
        rgb_x = self.rgb_bn1(rgb_x)
        rgb_x = self.rgb_relu(rgb_x)
        rgb_x = self.rgb_maxpool(rgb_x)  # [B, 64, 64, 64]
        
        # Texture Stream - Initial layers
        texture_x = self.texture_conv1(texture_input)  # [B, 64, 128, 128]
        texture_x = self.texture_bn1(texture_x)
        texture_x = self.texture_relu(texture_x)
        texture_x = self.texture_maxpool(texture_x)  # [B, 64, 64, 64]
        
        # ResBlock-1
        rgb_x = self.rgb_layer1(rgb_x)  # [B, 64, 64, 64]
        texture_x = self.texture_layer1(texture_x)  # [B, 64, 64, 64]
        
        # ResBlock-2
        rgb_x = self.rgb_layer2(rgb_x)  # [B, 128, 32, 32]
        texture_x = self.texture_layer2(texture_x)  # [B, 128, 32, 32]
        
        # TGVA Attention Fusion
        fused_x, attention_weights = self.tgva_block(rgb_x, texture_x)  # [B, 256, 32, 32]
        
        # ResBlock-3
        fused_x = self.layer3(fused_x)  # [B, 256, 16, 16]
        
        # ResBlock-4
        fused_x = self.layer4(fused_x)  # [B, 512, 8, 8]
        
        # Global Average Pooling
        fused_x = self.avgpool(fused_x)  # [B, 512, 1, 1]
        fused_x = torch.flatten(fused_x, 1)  # [B, 512]
        
        # Fully Connected
        fused_x = self.fc(fused_x)  # [B, 512]
        fused_x = self.fc_relu(fused_x)
        fused_x = self.fc_dropout(fused_x)
        
        # Classifier
        output = self.classifier(fused_x)  # [B, num_classes]
        
        return output, attention_weights
    
    def get_attention_features(self, rgb_input, texture_input):
        """
        Get intermediate attention-enhanced features for visualization
        
        Returns:
            layer2_features: Features before TGVA
            fused_features: Features after TGVA fusion
        """
        # Forward pass up to ResBlock-2
        rgb_x = self.rgb_conv1(rgb_input)
        rgb_x = self.rgb_bn1(rgb_x)
        rgb_x = self.rgb_relu(rgb_x)
        rgb_x = self.rgb_maxpool(rgb_x)
        rgb_x = self.rgb_layer1(rgb_x)
        rgb_x = self.rgb_layer2(rgb_x)
        
        texture_x = self.texture_conv1(texture_input)
        texture_x = self.texture_bn1(texture_x)
        texture_x = self.texture_relu(texture_x)
        texture_x = self.texture_maxpool(texture_x)
        texture_x = self.texture_layer1(texture_x)
        texture_x = self.texture_layer2(texture_x)
        
        layer2_features = rgb_x.clone()
        
        # Apply TGVA fusion
        fused_features, _ = self.tgva_block(rgb_x, texture_x)
        
        return layer2_features, fused_features

import torch
import torch.nn as nn

class MultiChannelBackbone(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiChannelBackbone, self).__init__()
        self.conv_h = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU())
        self.conv_s = nn.Sequential(nn.Conv2d(1, 16, (5,1), padding=(2,0)), nn.BatchNorm2d(16), nn.GELU())
        self.conv_p = nn.Sequential(nn.Conv2d(1, 16, (7,3), padding=(3,1)), nn.BatchNorm2d(16), nn.GELU())
        self.multi_scale = nn.Conv2d(48, 64, 3, padding=1)
        self.dynamic_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(64, 64, 1), nn.Sigmoid())
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        h = self.conv_h(x[:, 0:1, :, :])
        s = self.conv_s(x[:, 1:2, :, :])
        p = self.conv_p(x[:, 2:3, :, :])
        fused = torch.cat([h, s, p], dim=1)
        feat = torch.relu(self.multi_scale(fused))
        dyn = self.dynamic_conv(feat)
        attn = self.attention(feat)
        feat = feat + dyn * attn
        pooled = torch.mean(feat, dim=[2, 3])
        logits = self.classifier(pooled)
        return logits, pooled

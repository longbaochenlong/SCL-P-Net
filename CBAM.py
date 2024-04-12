import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(nn.Linear(in_planes, in_planes // ratio),
                                 nn.ReLU(),
                                 nn.Linear(in_planes // ratio, in_planes))
        self.fc2 = nn.Sequential(nn.Linear(in_planes, in_planes // ratio),
                                 nn.ReLU(),
                                 nn.Linear(in_planes // ratio, in_planes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_pool = avg_pool.squeeze()
        max_pool = max_pool.squeeze()
        avg_out = self.fc1(avg_pool)
        max_out = self.fc2(max_pool)
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out.view(out.shape[0], out.shape[1], 1, 1)
        out = (out + 1) * x
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        out = (out + 1) * x
        return out


class ChannelFirst(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelFirst, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out

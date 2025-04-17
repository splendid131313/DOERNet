import torch
from kornia.filters import sobel, spatial_gradient
import torch.nn.functional as F


def compute_edge_direction(x):
    # compute the x/y gradients
    edges = spatial_gradient(torch.mean(x, dim=1, keepdim=True))

    # unpack the edges
    gx = edges[:, :, 0]
    gy = edges[:, :, 1]

    edge_directions = torch.atan2(gy, gx)

    return edge_directions


def adjust_offset_with_direction(offset, edge_direction, gradient_magnitude, weight=0.5):
    gradient_magnitude = F.interpolate(gradient_magnitude, size=edge_direction.shape[2:], mode='bilinear')

    adjusted_offset = offset + weight * gradient_magnitude * torch.sin(edge_direction)

    return adjusted_offset




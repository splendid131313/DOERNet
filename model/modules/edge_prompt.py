from typing import Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0

        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def generate_points(self, x):
        B, C, H, W = x.shape

        grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W))
        grid_x = grid_x.unsqueeze(0).repeat(B, 1, 1).to(device=x.device)
        grid_y = grid_y.unsqueeze(0).repeat(B, 1, 1).to(device=x.device)
        coords = torch.stack((grid_x, grid_y), dim=-1)  # 形状为 (B, C, H, W, 2)

        coords = coords.reshape(B, -1, 2)

        return coords

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords = self.generate_points(coords)
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class EdgePromptEncoder(nn.Module):
    def __init__(self, embed_dim, img_size):
        super(EdgePromptEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer((self.img_size, self.img_size)).unsqueeze(0)

    def get_dense_batch_pe(self, batch_size):
        dense_pes = [self.get_dense_pe() for _ in range(batch_size)]  # List of tensors
        dense_batch_pe = torch.cat(dense_pes, dim=0)  # Concatenate along batch dimension
        return dense_batch_pe

    def _embed_edge(self, edge_map):
        B, _, H, W = edge_map.shape
        edge_map = edge_map + 0.5
        edge_embedding = self.pe_layer.forward_with_coords(edge_map, image_size=(H, W))
        return edge_embedding.permute(0, 2, 1).reshape(B, -1, H, W)

    def forward(self, edge_map):
        B, _, H, W = edge_map.shape
        edge_embedding = self._embed_edge(edge_map)
        image_pe = self.get_dense_batch_pe(B)
        return edge_embedding, image_pe


class EdgePrompt(nn.Module):
    def __init__(self, img_size, out_channels, depth, embed_dim=64, channels=[16, 64, 128, 320, 512]):
        super(EdgePrompt, self).__init__()

        self.layers = nn.ModuleList()
        self.conv = BasicConv2d(1, channels[0], kernel_size=3, padding=1, stride=2)
        for i in range(depth):
            self.layers.append(
                BasicConv2d(channels[i], channels[i+1], kernel_size=3, padding=1, stride=2)
            )
        self.downscale = nn.Conv2d(out_channels, embed_dim, kernel_size=1)
        self.position = EdgePromptEncoder(embed_dim, img_size)

    def forward(self, edge_map):
        """

        :param edge_map:
        :return: edge_embedding, image_pe
        """
        x = self.conv(edge_map)
        for layer in self.layers:
            x = layer(x)
        x = self.downscale(x)
        return self.position(x)


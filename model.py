import torch
import math
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

import torch
import math
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from torch.nn import MultiheadAttention

torch.backends.cuda.enable_flash_sdp(True)
import os
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod
from safetensors.torch import load_file


# os.environ["TORCH_LOGS"] = "+dynamo,+inductor"


class Block(nn.Module):
    """
    A simple processing block consisting of:
    - Group Normalization
    - SiLU (Swish) activation
    - 2D Convolution
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8, groups: int = 1):
        '''
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_groups (int, optional): Number of groups for GroupNorm. Defaults to 8.
            groups (int, optional): Number of groups for the convolution. Defaults to 1 (standard convolution).
        '''
        super().__init__()
        self.norm = nn.GroupNorm(num_groups * groups, in_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1,
                              groups=groups,
                              bias=True)

    def forward(self, x) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    """A residual block module based on the original ResNet design (https://arxiv.org/abs/1512.03385).

    This block consists of:
    - A mandatory first processing block (`Block`), with GroupNorm, SiLU activation, and a 3×3 convolution.
    - An optional second processing block if `use_second_block=True`.
    - A residual connection that adds the original input to the processed output.
      If the number of input and output channels differ, a 1×1 convolution is applied to the input
      to match dimensions before addition.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_second_block (bool): Whether to include a second processing block after the first.
        num_groups (int, optional): Number of groups for GroupNorm. Defaults to 8.
    """

    def __init__(self, in_channels: int, out_channels: int, use_second_block: bool, num_groups=8, groups=1):
        '''
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_second_block (bool): Whether to include an additional block after the firts.
            num_groups (int, optional): Number of groups for the GroupNorm. Default to 8.
            groups (int, optional): Number of groups for the convolution. Default to 1 (standard convolution).
        '''
        super().__init__()

        self.block1 = Block(in_channels, out_channels, num_groups=num_groups, groups=groups)
        if use_second_block:
            self.block2 = Block(out_channels, out_channels, num_groups=num_groups, groups=groups)
        else:
            self.block2 = nn.Identity()

        if in_channels == out_channels:
            self.res_conv = nn.Identity()
        else:
            self.res_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      groups=groups,
                                      bias=True)

    def forward(self, x) -> torch.Tensor:
        '''
        Args:
            x (Tensor): Input tensor of shape (batch_size, dim, height, width)

        Returns:
            Tensor: Output tensor of shape (batch_size, dim_out, height, width)
        '''
        out = self.block1(x)
        out = self.block2(out)
        return out + self.res_conv(x)


class UpSample(nn.Module):
    """
    Up-sampling layer for feature maps in convolutional neural networks.

    This module first upsamples the input feature map by a factor of 2 using
    nearest-neighbor interpolation, and then applies a 2D convolution with
    a 3x3 kernel to refine the upsampled features.
    """

    def __init__(self, channels):
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels, groups: int = 1):
        """
        :param channels: is the number of channels
        :groups: is the number of groups for the convolution
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0, groups=groups, bias=True)

    def forward(self, x):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Add padding
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        # Apply convolution
        return self.conv(x)


class Encoder(nn.Module):
    class Encoder(nn.Module):
        """
        A hierarchical convolutional encoder module for multi-channel input images.

        This encoder consists of:
        - An initial 3×3 convolution to project input to the base feature dimension.
        - A sequence of ResNet blocks and downsampling operations.
        - Optional skip connections for multi-resolution decoding.

        Args:
            init_dim (int): Number of channels after the initial projection convolution (for each modality). Default is 64.
            num_inputs (int): Number of input channels (e.g., 4 for multi-modal MRI (T1, T1c, T2, FLAIR)).
            dim_mults (tuple): Multipliers for the feature dimension at each resolution level.
            blocks (bool, optional): If True, uses two blocks per ResNet stage; otherwise, one.
            grouped (bool, optional): If True, uses grouped convolutions to keep modalities separate, using a single encoder.
        """

    def __init__(self, init_dim=64, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), blocks=True, grouped=False):
        super(Encoder, self).__init__()

        self.groups = num_inputs if grouped else 1
        self.init_dim = init_dim * self.groups
        assert num_inputs % self.groups == 0, "num_inputs must be a multiple of groups"

        self.conv_initial = nn.Conv2d(num_inputs, self.init_dim, 3, padding=1, groups=self.groups, bias=True)

        dims = [self.init_dim, *map(lambda m: self.init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, blocks, groups=self.groups),  # ResNet block
                        DownSample(dim_out, groups=self.groups) if not is_last else nn.Identity(),
                        # Downsampling at each layer, except the last
                    ]
                )
            )

    def forward(self, x):
        """
            Args:
                x (Tensor): Input tensor of shape (B, num_inputs, H, W).

            Returns:
                x (Tensor): Output feature map at the coarsest resolution.
                h (List[Tensor]): List of intermediate feature maps for skip connections.
        """
        x = self.conv_initial(x)

        h = []
        # downsample
        for down in self.downs:
            block, downsample = down
            x = block(x)
            h.append(x)
            x = downsample(x)

        return x, h


class Decoder(nn.Module):
    """
    A hierarchical decoder module with optional skip connections for modality synthesis.

    This decoder:
    - Takes a low-resolution input,
    - Optionally concatenates skip connections from earlier encoder layers,
    - Applies ResNet blocks and upsampling at each resolution,
    - Outputs a high-resolution synthetized volume through a final 1×1 convolution.

    Args:
        init_dim (int): Base number of feature channels.
        num_outputs (int): Number of output channels.
        dim_mults (tuple): Multipliers for feature dimensions at each resolution (must match encoder).
        skip (bool): If True, enables U-Net-style skip connections via feature concatenation.
        blocks (bool): If True, uses two blocks per ResNet stage instead of one.
        skip_multiplier (int): Factor to adjust channel depth when concatenating skip features.
    """

    def __init__(self, init_dim=64, num_outputs=1, dim_mults=(1, 2, 4, 8, 10), skip=True, blocks=True,
                 skip_multiplier=2):
        super(Decoder, self).__init__()
        self.skip = skip
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            if skip:
                dim_skip = int(dim_out * skip_multiplier)
            else:
                dim_skip = dim_out
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_skip, dim_in, blocks),
                        UpSample(dim_in) if not is_last else nn.Identity(),  # Upsampling at each layer, except the last
                    ]
                )
            )

        self.conv_final = nn.Sequential(
            ResnetBlock(init_dim, init_dim, blocks), nn.GroupNorm(8, init_dim), nn.SiLU(),
            nn.Conv2d(init_dim, num_outputs, 1)
        )

    def forward(self, x, h):
        """
        Args:
            x (Tensor): Bottleneck input tensor from the encoder (B, C, H, W).
            h (List[Tensor]): List of intermediate features from the encoder for skip connections.

        Returns:
            Tensor: Output tensor of shape (B, num_outputs, H, W).
        """
        # upsample
        for n, up in enumerate(self.ups):
            block = up[0]
            upsample = up[1]
            if self.skip:
                x = torch.cat((x, h[::-1][n]), dim=1)
            x = block(x)
            x = upsample(x)

        return self.conv_final(x)


class CHattnblock(nn.Module):
    """
        Channel-wise attention block using global average pooling and a gated MLP.

        Learns a per-channel attention map (B, C, 1, 1) to reweight feature maps based on global context.

        Args:
            dim (int): Number of input channels.
    """

    def __init__(self, dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # compresses spatial dimensions (B, C, H, W) → (B, C, 1, 1)
            nn.Conv2d(dim, dim, 1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid())

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Channel-wise attention map (one scalar per channel) of shape (B, C, 1, 1).
        """
        w = self.attn(x)
        # print(w.shape)
        return w


class HFEncoder(nn.Module):
    """
    Multi-modal encoder with attention-based feature fusion.

    This encoder:
    - Processes all input modalities together to learn shared features.
    - Separately encodes each modality to capture unique characteristics.
    - Applies channel-wise attention (CHattnblock) to both shared and modality-specific representations.
    - Fuses features using attention-weighted addition across available modalities.

    Args:
        dim (int): Base number of channels for feature maps.
        num_inputs (int): Number of input modalities (e.g., 4 for T1, T2, FLAIR, DWI).
        dim_mults (tuple): Feature map scaling factors at each downsampling level.
        n_layers (int): Not used (reserved for future extension?).
        blocks (bool): Whether to use two-block ResNet units.
        n_tokens (int): Not used (placeholder for token-based fusion?).
    """

    def __init__(self, dim=64, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), n_layers=2, blocks=True, n_tokens=0):
        super().__init__()
        self.num_inputs = num_inputs
        self.encoder_early = Encoder(dim, num_inputs, dim_mults, blocks)
        self.encoder_middles = nn.ModuleList([Encoder(dim, 1, dim_mults, blocks) for i in range(num_inputs)])
        self.attn_blocks = nn.ModuleList([CHattnblock(dim * dim_mults[-1]) for i in range(num_inputs + 1)])
        self.conv1 = nn.Conv2d(dim * dim_mults[-1] * 2, dim * dim_mults[-1], 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, input_modals, train_mode=False):
        """
            Args:
            x (Tensor): Input tensor of shape (B, num_inputs, H, W), where B is the batch size.
            input_modals (List[List[int]]): For each sample in batch, a list of indices of available modalities.
            train_mode (bool): If True, combines half of the batch for augmented consistency.

            Returns:
                x (Tensor): Fused bottleneck feature map of shape (B, C, H, W).
                h (List[Tensor]): List of skip features at each resolution level, fused per sample.
        """
        x_early, h_early = self.encoder_early(x)
        x_fusion_s = torch.zeros_like(x_early, device=x_early.device)
        x_fusion_h = torch.zeros_like(x_early, device=x_early.device)
        h_fusion = [torch.zeros_like(h, device=h.device) for h in h_early]

        x_middles = []
        h_middles = []
        x_attns = []
        for i in range(self.num_inputs):  # specific modality encoding
            x_middle, h_middle = self.encoder_middles[i](x[:, i:i + 1, :])
            x_middles.append(x_middle)
            h_middles.append(h_middle)
            x_attns.append(self.attn_blocks[i](x_middle))  # attention block on the specific modality representation
        x_attns.append(self.attn_blocks[-1](x_early))  # attention block on the common-to-all-modalities representation

        for n, modals in enumerate(input_modals):
            x_att = []
            x_feat = []
            for i in modals:
                for h_fusion_feat, h_middle_feat in zip(h_fusion, h_middles[i]):
                    h_fusion_feat[n, :] += h_middle_feat[n, :] / len(modals)
                x_att.append(x_attns[i][n:n + 1, :])
                x_feat.append(x_middles[i][n, :])
            if len(modals) != 1:
                x_att = torch.concat(x_att, dim=0)
                for idx, i in enumerate(modals):
                    x_fusion_s[n, :] += x_middles[i][n, :] * x_att[idx, 0, :]
                x_fusion_s[n, :] += x_early[n, :] * x_attns[-1][n, :]

                x_att = self.softmax(x_att)
                for idx, i in enumerate(modals):
                    x_fusion_h[n, :] += x_middles[i][n, :] * x_att[idx, 0, :]
                x_fusion_h[n, :] += x_early[n, :]
            else:
                for idx, i in enumerate(modals):
                    x_fusion_s[n, :] = x_middles[i][n, :] * x_att[0][0, :]
                    x_fusion_h[n, :] = x_middles[i][n, :]

        x_fusion = self.conv1(torch.concat((x_fusion_s, x_fusion_h), dim=1))
        if train_mode:
            idx_1ch = x.shape[0] // 2
            x = x_fusion

            h_combination = [f_e[0:idx_1ch, :] + f_fusion[0:idx_1ch, :] for f_e, f_fusion in zip(h_early, h_fusion)]
            h_1ch = [f_fusion[idx_1ch:, :] for f_fusion in h_fusion]
            h = [torch.cat([f_comb, f_1ch], dim=0) for f_comb, f_1ch in zip(h_combination, h_1ch)]
        else:
            x = x_fusion

            h = []
            for f_early, f_fusion in zip(h_early, h_fusion):
                f = []
                for n, modals in enumerate(input_modals):
                    if len(modals) == 1:
                        f.append(f_fusion[n:n + 1, :])
                    else:
                        f_sum = f_early[n:n + 1, :] + f_fusion[n:n + 1, :]
                        f.append(f_sum)
                h.append(torch.cat(f, dim=0))

        return x, h


class HFEncoder_grouped(nn.Module):
    """
    Multi-modal encoder with attention-based feature fusion.

    This encoder:
    - Processes all input modalities together to learn shared features.
    - Separately encodes each modality to capture unique characteristics.
    - Applies channel-wise attention (CHattnblock) to both shared and modality-specific representations.
    - Fuses features using attention-weighted addition across available modalities.

    Args:
        dim (int): Base number of channels for feature maps.
        num_inputs (int): Number of input modalities (e.g., 4 for T1, T2, FLAIR, DWI).
        dim_mults (tuple): Feature map scaling factors at each downsampling level.
        n_layers (int): Not used (reserved for future extension?).
        blocks (bool): Whether to use two-block ResNet units.
        n_tokens (int): Not used (placeholder for token-based fusion?).
    """

    def __init__(self, dim=64, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), n_layers=2, blocks=True, n_tokens=0):
        super().__init__()
        self.num_inputs = num_inputs
        self.encoder_early = Encoder(dim, num_inputs, dim_mults, blocks)
        self.encoder_middles = Encoder(dim, num_inputs, dim_mults, blocks, grouped=True)
        self.attn_blocks = nn.ModuleList([CHattnblock(dim * dim_mults[-1]) for i in range(num_inputs + 1)])
        self.conv1 = nn.Conv2d(dim * dim_mults[-1] * 2, dim * dim_mults[-1], 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, input_modals, train_mode=False):
        """
            Args:
            x (Tensor): Input tensor of shape (B, num_inputs, H, W), where B is the batch size.
            input_modals (List[List[int]]): For each sample in batch, a list of indices of available modalities.
            train_mode (bool): If True, combines half of the batch for augmented consistency.

            Returns:
                x (Tensor): Fused bottleneck feature map of shape (B, C, H, W).
                h (List[Tensor]): List of skip features at each resolution level, fused per sample.
        """
        # shared “early encoder"
        x_early, h_early = self.encoder_early(x)

        # grouped encoder for modality-specific features
        x_middle_all, h_middle_all = self.encoder_middles(x)

        # Split bottleneck & every skip tensor back into per-modality chunks
        x_middles = list(torch.chunk(x_middle_all, self.num_inputs, dim=1))
        h_middles = [[] for _ in range(self.num_inputs)]
        for h_lvl in h_middle_all:
            chunks = torch.chunk(h_lvl, self.num_inputs, dim=1)
            for i in range(self.num_inputs):
                h_middles[i].append(chunks[i])

        # attention maps per modality
        x_attns = [self.attn_blocks[i](x_middles[i])
                   for i in range(self.num_inputs)]
        #  shared one attention maps
        x_attns.append(self.attn_blocks[-1](x_early))

        # fuse modality-specific + shared representations
        x_fusion_s = torch.zeros_like(x_early, device=x_early.device)
        x_fusion_h = torch.zeros_like(x_early, device=x_early.device)
        h_fusion = [torch.zeros_like(h, device=h.device) for h in h_early]

        for n, modals in enumerate(input_modals):
            x_att = []
            # x_feat = []
            for i in modals:
                for h_fusion_feat, h_middle_feat in zip(h_fusion, h_middles[i]):
                    h_fusion_feat[n, :] += h_middle_feat[n, :] / len(modals)
                x_att.append(x_attns[i][n:n + 1, :])
                # x_feat.append(x_middles[i][n,:])
            if len(modals) != 1:
                x_att = torch.concat(x_att, dim=0)
                for idx, i in enumerate(modals):
                    x_fusion_s[n, :] += x_middles[i][n, :] * x_att[idx, 0, :]
                x_fusion_s[n, :] += x_early[n, :] * x_attns[-1][n, :]

                x_att = self.softmax(x_att)
                for idx, i in enumerate(modals):
                    x_fusion_h[n, :] += x_middles[i][n, :] * x_att[idx, 0, :]
                x_fusion_h[n, :] += x_early[n, :]
            else:
                for idx, i in enumerate(modals):
                    x_fusion_s[n, :] = x_middles[i][n, :] * x_att[0][0, :]
                    x_fusion_h[n, :] = x_middles[i][n, :]

        x_fusion = self.conv1(torch.concat((x_fusion_s, x_fusion_h), dim=1))
        if train_mode:
            idx_1ch = x.shape[0] // 2
            x = x_fusion

            h_combination = [f_e[0:idx_1ch, :] + f_fusion[0:idx_1ch, :] for f_e, f_fusion in zip(h_early, h_fusion)]
            h_1ch = [f_fusion[idx_1ch:, :] for f_fusion in h_fusion]
            h = [torch.cat([f_comb, f_1ch], dim=0) for f_comb, f_1ch in zip(h_combination, h_1ch)]
        else:
            x = x_fusion

            h = []
            for f_early, f_fusion in zip(h_early, h_fusion):
                f = []
                for n, modals in enumerate(input_modals):
                    if len(modals) == 1:
                        f.append(f_fusion[n:n + 1, :])
                    else:
                        f_sum = f_early[n:n + 1, :] + f_fusion[n:n + 1, :]
                        f.append(f_sum)
                h.append(torch.cat(f, dim=0))

        return x, h


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, dim, patch_size, n_tokens):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=dim,
                                          out_channels=dim,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # B,C,W,H = x.shape()
        x = self.patch_embeddings(x)
        x = x.flatten(2, 3)
        h = x.permute(0, 2, 1)

        embeddings = h + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Attention module
class Attention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size)
        self.attn = Attention(hidden_size, n_heads)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if time.dtype == torch.int64:
            return embeddings
        else:
            return embeddings.type(time.dtype)


class ModalityInfuser(nn.Module):
    def __init__(self, hidden_size, patch_size, n_tokens, n_layers, n_heads, modality_embed):
        super().__init__()
        self.modality_embed = modality_embed
        # n_tokens = int((240/(2**n_downs)/patch_size)**2)
        self.modality_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.embedding = Embeddings(hidden_size, patch_size, n_tokens)
        self.layers = nn.ModuleList([])
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(n_layers):
            self.layers.append(
                TransformerBlock(hidden_size, n_heads)
            )

    def forward(self, x, m, v):
        B, C, W, H = x.shape
        h = self.embedding(x)
        if self.modality_embed:
            m = self.modality_embedding(m)
            h = rearrange(m, "b c -> b 1 c") + h

        for layer_block in self.layers:
            h = layer_block(h)
        h = self.encoder_norm(h)
        h = h.permute(0, 2, 1).contiguous().view(B, C, W, H)
        return h


# PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
class Discriminator(nn.Module):
    def __init__(self, channels=1, num_filters_last=32, n_layers=3, n_classes=4, ixi=False):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.GroupNorm(8, num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]
        self.model = nn.Sequential(*layers)
        self.final = nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1, bias=False)
        if ixi:
            self.classifier = nn.Conv2d(num_filters_last * num_filters_mult, n_classes, 31, bias=False)
        else:
            self.classifier = nn.Conv2d(num_filters_last * num_filters_mult, n_classes, 29, bias=False)

    def forward(self, x):
        features = self.model(x)
        logits = self.final(features)
        labels = self.classifier(features)
        return logits, labels.view(labels.size(0), labels.size(1))


class Discriminator_v2(nn.Module):
    def __init__(self, channels=1, num_filters_last=32, n_layers=3, n_classes=4, ixi=False):
        super().__init__()

        layers = [nn.Conv2d(channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.GroupNorm(8, num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]
        self.model = nn.Sequential(*layers)
        self.final = nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1, bias=False)
        if ixi:
            self.classifier = nn.Conv2d(num_filters_last * num_filters_mult, n_classes, 31, bias=False)
        else:
            self.classifier = nn.Conv2d(num_filters_last * num_filters_mult, n_classes, 29, bias=False)

    def forward(self, x):
        features = self.model(x)
        logits = self.final(features)
        labels = self.classifier(features)
        return logits, labels.view(labels.size(0), labels.size(1))


class FlashAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        self.mha = MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x):
        skip = x
        x = self.norm1(x)

        x, _ = self.mha(x, x, x, need_weights=False)
        x = x + skip

        skip = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + skip
        return x


class ModalityViewInfuser(nn.Module):
    def __init__(self, hidden_size, patch_size, n_tokens, n_layers, n_heads, modality_embed):
        super().__init__()

        self.modality_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.view_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.embedding = Embeddings(hidden_size, patch_size, n_tokens)
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.transformer = nn.Sequential(*[FlashAttentionBlock(hidden_size, n_heads) for _ in range(n_layers)])

    def forward(self, x, m, v):
        B, C, W, H = x.shape
        h = self.embedding(x)

        m = self.modality_embedding(m)
        v = self.view_embedding(v)
        h += rearrange(m, "b c -> b 1 c")
        h += rearrange(v, "b c -> b 1 c")

        h = self.transformer(h)
        h = self.encoder_norm(h)
        h = h.permute(0, 2, 1).contiguous().view(B, C, W, H)
        return h


class HFGAN(nn.Module):
    def __init__(self, dim, num_inputs, num_outputs, dim_mults, n_layers, skip, blocks, image_size=240,
                 grouped_encoder=False, infuse_view=False, num_heads=16):
        super().__init__()
        self.infuse_view = infuse_view
        patch_size = 1
        n_tokens = int((image_size / (2 ** (len(dim_mults) - 1)) / patch_size) ** 2)
        if grouped_encoder:
            self.encoder = HFEncoder_grouped(dim, num_inputs, dim_mults, n_layers, blocks, n_tokens=n_tokens)
        else:
            self.encoder = HFEncoder(dim, num_inputs, dim_mults, n_layers, blocks, n_tokens=n_tokens)
        self.decoder = Decoder(dim, num_outputs, dim_mults, skip, blocks)
        if self.infuse_view:
            self.middle = ModalityViewInfuser(hidden_size=dim * dim_mults[-1], patch_size=1, n_tokens=n_tokens,
                                              n_layers=n_layers, n_heads=num_heads, modality_embed=True)
        else:
            self.middle = ModalityInfuser(hidden_size=dim * dim_mults[-1], patch_size=1, n_tokens=n_tokens,
                                          n_layers=n_layers, n_heads=num_heads, modality_embed=True)

    def forward(self, x, avail_modals, missing_modals, views=None, train_mode=True):
        features, h = self.encoder(x, avail_modals, train_mode=train_mode)
        z = self.middle(features, missing_modals, views)
        targets_recon = self.decoder(z, h)
        return targets_recon, features


class ViewGenerator(ABC):
    def __init__(self, model):
        self.model = model.cuda()

    @abstractmethod
    def generate(self, brain, missing_modality) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __call__(self, *args):
        return self.generate(*args)


class AxiGenerator(ViewGenerator):
    def __init__(self, model):
        super().__init__(model)
        self.batch_size = 31

    def generate(self, brain, missing_modality) -> Tuple[torch.Tensor, torch.Tensor]:
        assert brain.shape[0] == 1, 'Do not give batched brains to this method bro'
        brain = brain.cuda().squeeze(0)  # (1, 4, 240, 240, 155) --> (4, 240, 240, 155)
        recon_volume_list = []
        for i in range(0, 155, self.batch_size):
            brain_slice = brain[:, :, :, i:i + self.batch_size]  # (4, 240, 240, 31)
            with torch.no_grad():
                recon, _ = self.model(brain_slice.permute(3, 0, 1, 2).contiguous(),
                                      [[x for x in range(4) if x != (missing_modality)] for _ in
                                       range(self.batch_size)],
                                      torch.full((self.batch_size,), missing_modality, device='cuda'),
                                      train_mode=False,
                                      views=torch.full((self.batch_size,), 0, device='cuda'))

            recon = recon.permute(1, 2, 3, 0)  # (31, 1, 240, 240) --> (1, 240, 240, 31)

            recon_volume_list.append(recon.cpu())
        recon_volume = torch.cat(recon_volume_list, dim=3).contiguous()  # (1, 240, 240, 155)
        recon_volume = recon_volume.unsqueeze(0)  # (1, 1, 240, 240, 155)
        brain_recon = brain.clone().unsqueeze(0).cpu()

        majority = torch.zeros((240, 240, 155), dtype=torch.int32).cuda()
        for i in range(4):
            if i == missing_modality.item():
                continue
            else:
                majority += (brain[i] == -1).int()
        brain_mask = (majority >= 2).cpu()
        recon_volume[brain_mask.unsqueeze(0).unsqueeze(0)] = -1
        brain_recon[:, missing_modality:missing_modality + 1] = recon_volume
        return recon_volume.clone(), brain_recon  # (1, 1, 240, 240, 155), (1, 4, 240, 240, 155)


class SagGenerator(ViewGenerator):
    def __init__(self, model):
        super().__init__(model)
        self.batch_size = 40

    def generate(self, brain, missing_modality) -> Tuple[torch.Tensor, torch.Tensor]:
        assert brain.shape[0] == 1, 'Do not give batched brains to this method bro'
        brain = brain.cuda().squeeze(0)  # (1, 4, 240, 240, 155) --> (4, 240, 240, 155)
        recon_volume_list = []
        for i in range(0, 240, self.batch_size):
            brain_slice = brain[:, i:i + self.batch_size, :, :]  # (4, 40, 240, 155)
            brain_slice = F.pad(brain_slice, (42, 43), value=-1)  # (4, 40, 240, 240)
            with torch.no_grad():
                recon, _ = self.model(brain_slice.permute(1, 0, 2, 3).contiguous(),  # (04, 4, 240, 240)
                                      [[x for x in range(4) if x != (missing_modality)] for _ in
                                       range(self.batch_size)],
                                      torch.full((self.batch_size,), missing_modality, device='cuda'),
                                      train_mode=False,
                                      views=torch.full((self.batch_size,), 1, device='cuda'))

            recon = recon.permute(1, 0, 2, 3)  # (40, 1, 240, 240) --> (1, 40, 240, 240)
            recon = recon[:, :, :, 42:155 + 42]  # (1, 40, 240, 155)

            recon_volume_list.append(recon.cpu())
        recon_volume = torch.cat(recon_volume_list, dim=1).contiguous()  # (1, 240, 240, 155)
        recon_volume = recon_volume.unsqueeze(0)  # (1, 1, 240, 240, 155)

        brain_recon = brain.clone().unsqueeze(0).cpu()

        majority = torch.zeros((240, 240, 155), dtype=torch.int32).cuda()
        for i in range(4):
            if i == missing_modality.item():
                continue
            else:
                majority += (brain[i] == -1).int()
        brain_mask = (majority >= 2).cpu()
        recon_volume[brain_mask.unsqueeze(0).unsqueeze(0)] = -1
        brain_recon[:, missing_modality:missing_modality + 1] = recon_volume
        return recon_volume.clone(), brain_recon  # (1, 1, 240, 240, 155), (1, 4, 240, 240, 155)


class CorGenerator(ViewGenerator):
    def __init__(self, model):
        super().__init__(model)
        self.batch_size = 40

    def generate(self, brain, missing_modality) -> Tuple[torch.Tensor, torch.Tensor]:
        assert brain.shape[0] == 1, 'Do not give batched brains to this method bro'
        brain = brain.cuda().squeeze(0)  # (1, 4, 240, 240, 155) --> (4, 240, 240, 155)
        recon_volume_list = []
        for i in range(0, 240, self.batch_size):
            brain_slice = brain[:, :, i:i + self.batch_size, :]  # (4, 240, 40, 155)
            brain_slice = F.pad(brain_slice, (42, 43), value=-1)  # (4, 240, 40, 240)
            with torch.no_grad():
                recon, _ = self.model(brain_slice.permute(2, 0, 1, 3).contiguous(),  # (04, 4, 240, 240)
                                      [[x for x in range(4) if x != (missing_modality)] for _ in
                                       range(self.batch_size)],
                                      torch.full((self.batch_size,), missing_modality, device='cuda'),
                                      train_mode=False,
                                      views=torch.full((self.batch_size,), 2, device='cuda'))

            recon = recon.permute(1, 2, 0, 3)  # (40, 1, 240, 240) --> (1, 240, 40, 240)
            recon = recon[:, :, :, 42:155 + 42]  # (1, 40, 240, 155)

            recon_volume_list.append(recon.cpu())
        recon_volume = torch.cat(recon_volume_list, dim=2).contiguous()  # (1, 240, 240, 155)
        recon_volume = recon_volume.unsqueeze(0)  # (1, 1, 240, 240, 155)
        brain_recon = brain.clone().unsqueeze(0).cpu()

        majority = torch.zeros((240, 240, 155), dtype=torch.int32).cuda()
        for i in range(4):
            if i == missing_modality.item():
                continue
            else:
                majority += (brain[i] == -1).int()
        brain_mask = (majority >= 2).cpu()
        recon_volume[brain_mask.unsqueeze(0).unsqueeze(0)] = -1
        brain_recon[:, missing_modality:missing_modality + 1] = recon_volume
        return recon_volume.clone(), brain_recon  # (1, 1, 240, 240, 155), (1, 4, 240, 240, 155)


class HFGAN_3D(nn.Module):
    '''
    This is a simple wrapper for the HFGAN class
    It is a virtual 3D-generating network. It simulated 3d behaviour, while iterating and generating in 2D
    It accepts batched 3D modules.

    Do not move it to gpu, it will manage all the memory movements itself.

    Usage, assuming the missing modality file is missing from the folder structure:
    #using ULTIMATE_brain_coll_3d
    for element in batch:
        brains = element['images']
        miss_mod = element['missing_modalities']
        recon_volumes, everything = hfgan(brains,miss_mod)

    NB: This model will not normalize/denormalize inputs and outputs.
    Pipeline to follow:
    1. Use ULTIMATE_brain_coll_3d to load normalized brains
    2. If your val directory has all the available modalities, mask the modality provided by the collate like:

        for i in range(element['missing_modalities'].shape[0]):
            brains[i,miss_mod[i]] = -1

       Remember: this nn.Module will assume the input to be already masked!
    3. Call the forward of this module. It will return both the generated volume and a copy of the input
       with the missing modality replaced with the generated modality. Remember: inputs and output are batched!
       Ex:
        recon_volumes, everything = hfgan(brains,miss_mod)

        print(recon_volumes.shape) --> [B, 1, 240, 240, 155]
        print(everything.shape) --> [B, 4, 240, 240, 155]

       NB: the inputs must to be on cpu
           the outputs will be on cpu
    4. Have fun! Remember, programming is just a funny activity we do to entertain ourselves! <3

    '''

    def __init__(self,
                 view: str,
                 weights_file: str | Path,
                 grouped_encoder: bool,
                 infuse_view: bool
                 ):
        super().__init__()
        assert view in ['axi', 'gli', 'sag'], "view must be one of: ['axi','gli','sag']"
        if isinstance(weights_file, str):
            weights_file = Path(weights_file)
        assert weights_file.exists() and weights_file.is_file and weights_file.is_absolute, f'Something wrong with the weights file: {str(weights_file)}'

        model = HFGAN(dim=64, num_inputs=4, num_outputs=1, dim_mults=(1, 2, 4, 8, 10), n_layers=4, skip=True,
                      blocks=False, grouped_encoder=grouped_encoder, infuse_view=infuse_view).cuda()
        state_dict = load_file(weights_file)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        self.batch_size = 31 if view == 'axi' else 40
        self._generators = {
            'axi': AxiGenerator(model),
            'sag': SagGenerator(model),
            'cor': CorGenerator(model)
        }
        self.generator = self._generators[view]

    @torch.no_grad()
    def forward(self, brains: torch.Tensor, missing_modalities: torch.Tensor):
        # brains                --> (B,4,240,240,155)
        # missing_modalities    --> (B)
        recon_list = []
        everything_list = []

        for i in range(brains.shape[0]):
            brain = brains[i:i + 1]
            modal = missing_modalities[i]
            recon, everything = self.generator(brain, modal)
            recon_list.append(recon)  # (1,1,240,240,155)
            everything_list.append(everything)  # (1,4,240,240,155)

        recon_all = torch.cat(recon_list, dim=0)  # (B,1,240,240,155)
        everything_all = torch.cat(everything_list, dim=0)  # (B,4,240,240,155)

        return recon_all, everything_all






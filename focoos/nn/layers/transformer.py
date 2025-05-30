import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from focoos.utils.logger import get_logger

from .base import _get_activation_fn
from .misc import DropPath
from .norm import LayerNorm

logger = get_logger(__name__)


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for transformer architectures."""

    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        """Initialize self-attention layer.

        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            dropout: Dropout probability
            normalize_before: Whether to apply normalization before attention
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional embeddings to the tensor.

        Args:
            tensor: Input tensor
            pos: Positional embedding tensor

        Returns:
            Tensor with positional embeddings added
        """
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Apply self-attention with post-normalization.

        Args:
            tgt: Target tensor
            tgt_mask: Attention mask
            tgt_key_padding_mask: Key padding mask
            query_pos: Query positional embedding

        Returns:
            Output tensor after self-attention
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Apply self-attention with pre-normalization.

        Args:
            tgt: Target tensor
            tgt_mask: Attention mask
            tgt_key_padding_mask: Key padding mask
            query_pos: Query positional embedding

        Returns:
            Output tensor after self-attention
        """
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Apply self-attention based on normalization preference.

        Args:
            tgt: Target tensor
            tgt_mask: Attention mask
            tgt_key_padding_mask: Key padding mask
            query_pos: Query positional embedding

        Returns:
            Output tensor after self-attention
        """
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for transformer architectures."""

    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        """Initialize cross-attention layer.

        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            dropout: Dropout probability
            normalize_before: Whether to apply normalization before attention
        """
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional embeddings to the tensor.

        Args:
            tensor: Input tensor
            pos: Positional embedding tensor

        Returns:
            Tensor with positional embeddings added
        """
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Apply cross-attention with post-normalization.

        Args:
            tgt: Target tensor
            memory: Memory tensor (key/value)
            memory_mask: Attention mask
            memory_key_padding_mask: Key padding mask
            pos: Memory positional embedding
            query_pos: Query positional embedding

        Returns:
            Output tensor after cross-attention
        """
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Apply cross-attention with pre-normalization.

        Args:
            tgt: Target tensor
            memory: Memory tensor (key/value)
            memory_mask: Attention mask
            memory_key_padding_mask: Key padding mask
            pos: Memory positional embedding
            query_pos: Query positional embedding

        Returns:
            Output tensor after cross-attention
        """
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Apply cross-attention based on normalization preference.

        Args:
            tgt: Target tensor
            memory: Memory tensor (key/value)
            memory_mask: Attention mask
            memory_key_padding_mask: Key padding mask
            pos: Memory positional embedding
            query_pos: Query positional embedding

        Returns:
            Output tensor after cross-attention
        """
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    """Feed-forward network layer for transformer architectures."""

    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation: Optional[str] = None,
        normalize_before=False,
        ffn_type="standard",
    ):
        """Initialize feed-forward network layer.

        Args:
            d_model: Dimension of the model
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            activation: Activation function
            normalize_before: Whether to apply normalization before FFN
            ffn_type: Type of FFN ('standard' or 'convnext')
        """
        super().__init__()

        assert ffn_type in [
            "standard",
            "convnext",
        ], "FFN can be of 'standard' or 'convnext' type"
        self.ffn_type = ffn_type
        self.normalize_before = normalize_before

        if self.ffn_type == "standard":
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            self.norm = nn.LayerNorm(d_model)

            self.activation = _get_activation_fn(activation) if activation is not None else nn.ReLU()

            self._reset_parameters()
        elif self.ffn_type == "convnext":
            if not normalize_before:
                logger.warning(
                    "Pre-normalizazion is applied by default with convnext FFN layer since "
                    "it supports only pre-normalization."
                )
                self.normalize_before = True

            layer_scale_init_value = 1e-6
            drop_path = 0.0

            self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)  # depthwise conv
            self.norm = LayerNorm(d_model, eps=1e-6)
            # pointwise/1x1 convs, implemented with linear layers
            self.pwconv1 = nn.Linear(d_model, dim_feedforward)
            self.act = _get_activation_fn(activation) if activation is not None else nn.GELU()
            self.pwconv2 = nn.Linear(dim_feedforward, d_model)
            self.gamma = (
                nn.Parameter(layer_scale_init_value * torch.ones(d_model), requires_grad=True)
                if layer_scale_init_value > 0
                else None
            )
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional embeddings to the tensor.

        Args:
            tensor: Input tensor
            pos: Positional embedding tensor

        Returns:
            Tensor with positional embeddings added
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        """Apply FFN with post-normalization.

        Args:
            tgt: Target tensor

        Returns:
            Output tensor after FFN
        """
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        """Apply FFN with pre-normalization.

        Args:
            tgt: Target tensor

        Returns:
            Output tensor after FFN
        """
        if self.ffn_type == "standard":
            tgt2 = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout(tgt2)
            return tgt
        elif self.ffn_type == "convnext":
            # tgt shape is -> QxNxC (Q: num. of query - N: batch size - C: num. of channels)
            Q, N, C = tgt.shape
            H = W = int(Q**0.5)
            x = tgt.permute(1, 2, 0).reshape(N, C, H, W)

            tgt2 = self.dwconv(x)
            tgt2 = tgt2.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            tgt2 = self.norm(tgt2)
            tgt2 = self.pwconv1(tgt2)
            tgt2 = self.act(tgt2)
            tgt2 = self.pwconv2(tgt2)
            if self.gamma is not None:
                tgt2 = self.gamma * tgt2
            tgt2 = tgt2.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

            tgt = tgt + self.drop_path(torch.flatten(tgt2, 2, 3).permute(2, 0, 1))
            return tgt

    def forward(self, tgt):
        """Apply FFN based on normalization preference.

        Args:
            tgt: Target tensor

        Returns:
            Output tensor after FFN
        """
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos_embed: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embed=pos_embed,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                if self.norm is not None:
                    output = self.norm(output)
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        batch_first=True,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

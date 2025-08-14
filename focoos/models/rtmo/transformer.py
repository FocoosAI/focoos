import warnings
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor


class SinePositionalEncoding(nn.Module):
    """Sine Positional Encoding Module. This module implements sine positional
    encoding, which is commonly used in transformer-based models to add
    positional information to the input sequences. It uses sine and cosine
    functions to create positional embeddings for each element in the input
    sequence.

    Args:
        out_channels (int): The number of features in the input sequence.
        temperature (int): A temperature parameter used to scale
            the positional encodings. Defaults to 10000.
        spatial_dim (int): The number of spatial dimension of input
            feature. 1 represents sequence data and 2 represents grid data.
            Defaults to 1.
        learnable (bool): Whether to optimize the frequency base. Defaults
            to False.
        eval_size (int, tuple[int], optional): The fixed spatial size of
            input features. Defaults to None.
    """

    def __init__(
        self,
        out_channels: int,
        spatial_dim: int = 1,
        temperature: int = 100000,
        learnable: bool = False,
        eval_size: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        super().__init__()

        assert out_channels % 2 == 0
        assert temperature > 0

        self.spatial_dim = spatial_dim
        self.out_channels = out_channels
        self.temperature = temperature
        self.eval_size = eval_size
        self.learnable = learnable

        pos_dim = out_channels // 2
        dim_t = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        dim_t = self.temperature ** (dim_t)

        if not learnable:
            self.register_buffer("dim_t", dim_t)
        else:
            self.dim_t = nn.Parameter(dim_t.detach())

        # set parameters
        if eval_size:
            if hasattr(self, f"pos_enc_{eval_size}"):
                delattr(self, f"pos_enc_{eval_size}")
            pos_enc = self.generate_pos_encoding(size=eval_size)
            self.register_buffer(f"pos_enc_{eval_size}", pos_enc)

    def forward(self, *args, **kwargs):
        return self.generate_pos_encoding(*args, **kwargs)

    def generate_pos_encoding(
        self, size: Optional[Union[int, Sequence[int]]] = None, position: Optional[Tensor] = None
    ):
        """Generate positional encoding for input features.

        Args:
            size (int or tuple[int]): Size of the input features. Required
                if position is None.
            position (Tensor, optional): Position tensor. Required if size
                is None.
        """

        assert (size is not None) ^ (position is not None)

        if (not (self.learnable and self.training)) and size is not None and hasattr(self, f"pos_enc_{size}"):
            return getattr(self, f"pos_enc_{size}")

        if self.spatial_dim == 1:
            if size is not None:
                if isinstance(size, (tuple, list)):
                    size = size[0]
                position = torch.arange(size, dtype=torch.float32, device=self.dim_t.device)  # type: ignore

            dim_t = self.dim_t.reshape(*((1,) * position.ndim), -1)  # type: ignore
            freq = position.unsqueeze(-1) / dim_t  # type: ignore
            pos_enc = torch.cat((freq.cos(), freq.sin()), dim=-1)

        elif self.spatial_dim == 2:
            if size is not None:
                if isinstance(size, (tuple, list)):
                    h, w = size[:2]
                elif isinstance(size, (int, float)):
                    h, w = int(size), int(size)
                else:
                    raise ValueError(f"got invalid type {type(size)} for size")
                grid_h, grid_w = torch.meshgrid(
                    torch.arange(int(h), dtype=torch.float32, device=self.dim_t.device),
                    torch.arange(int(w), dtype=torch.float32, device=self.dim_t.device),
                )
                grid_h, grid_w = grid_h.flatten(), grid_w.flatten()
            else:
                assert position is not None and position.size(-1) == 2
                grid_h, grid_w = torch.unbind(position, dim=-1)

            dim_t = self.dim_t.reshape(*((1,) * grid_h.ndim), -1)
            freq_h = grid_h.unsqueeze(-1) / dim_t
            freq_w = grid_w.unsqueeze(-1) / dim_t
            pos_enc_h = torch.cat((freq_h.cos(), freq_h.sin()), dim=-1)
            pos_enc_w = torch.cat((freq_w.cos(), freq_w.sin()), dim=-1)
            pos_enc = torch.stack((pos_enc_h, pos_enc_w), dim=-1)

        return pos_enc

    @staticmethod
    def apply_additional_pos_enc(feature: Tensor, pos_enc: Tensor, spatial_dim: int = 1):
        """Apply additional positional encoding to input features.

        Args:
            feature (Tensor): Input feature tensor.
            pos_enc (Tensor): Positional encoding tensor.
            spatial_dim (int): Spatial dimension of input features.
        """

        assert spatial_dim in (1, 2), f"the argument spatial_dim must be either 1 or 2, but got {spatial_dim}"
        if spatial_dim == 2:
            pos_enc = pos_enc.flatten(-2)
        for _ in range(feature.ndim - pos_enc.ndim):
            pos_enc = pos_enc.unsqueeze(0)
        return feature + pos_enc

    @staticmethod
    def apply_rotary_pos_enc(feature: Tensor, pos_enc: Tensor, spatial_dim: int = 1):
        """Apply rotary positional encoding to input features.

        Args:
            feature (Tensor): Input feature tensor.
            pos_enc (Tensor): Positional encoding tensor.
            spatial_dim (int): Spatial dimension of input features.
        """

        assert spatial_dim in (1, 2), f"the argument spatial_dim must be either 1 or 2, but got {spatial_dim}"

        for _ in range(feature.ndim - pos_enc.ndim + spatial_dim - 1):
            pos_enc = pos_enc.unsqueeze(0)

        x1, x2 = torch.chunk(feature, 2, dim=-1)
        if spatial_dim == 1:
            cos, sin = torch.chunk(pos_enc, 2, dim=-1)
            feature = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        elif spatial_dim == 2:
            pos_enc_h, pos_enc_w = torch.unbind(pos_enc, dim=-1)
            cos_h, sin_h = torch.chunk(pos_enc_h, 2, dim=-1)
            cos_w, sin_w = torch.chunk(pos_enc_w, 2, dim=-1)
            feature = torch.cat((x1 * cos_h - x2 * sin_h, x1 * cos_w + x2 * sin_w), dim=-1)

        return feature


class DetrTransformerEncoder(nn.Module):
    """Encoder of DETR.

    Args:
        num_layers (int): Number of encoder layers.
        layer_cfg (:obj:`ConfigDict` or dict): the config of each encoder
            layer. All the layers will share the same config.
        num_cp (int): Number of checkpointing blocks in encoder layer.
            Default to -1.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int = 1024,
        ffn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                DetrTransformerEncoderLayer(embed_dims, num_heads, feedforward_channels, ffn_drop)
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, query: Tensor, query_pos: Tensor, key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of encoder.

        Args:
            query (Tensor): Input queries of encoder, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional embeddings of the queries, has
                shape (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).

        Returns:
            Tensor: Has shape (bs, num_queries, dim) if `batch_first` is
            `True`, otherwise (num_queries, bs, dim).
        """
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask, **kwargs)
        return query


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
    """

    def __init__(self, embed_dims, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key ismissing in {self.__class__.__name__}.")
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.proj_drop(out)


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
    """

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        add_identity=True,
    ):
        super().__init__()
        assert num_fcs >= 2, f"num_fcs should be no less than 2. got {num_fcs}."
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(nn.Linear(in_channels, feedforward_channels), nn.GELU(), nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class DetrTransformerEncoderLayer(nn.Module):
    """Implements encoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int = 1024,
        ffn_drop: float = 0.0,
    ):
        super().__init__()

        self.self_attn = MultiheadAttention(embed_dims, num_heads)

        self.ffn = FFN(embed_dims, feedforward_channels, ffn_drop=ffn_drop)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(2)])

    def forward(self, query: Tensor, query_pos: Tensor, key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query

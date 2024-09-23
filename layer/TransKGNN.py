from layer.KANLayer_cus import KANLayer_cus
#from layer.efficient_kan import EfficientKANLinear as KANLayer_cus

from typing import Optional, Union, Callable, Tuple
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter
import math

class KANTransformerEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, grid=5, k=1, neuron_fun='sum') -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            grid=grid, k=k, neuron_fun=neuron_fun, **factory_kwargs)
        # Implementation of Feedforward model using KANLayer_cus
        self.linear1 = KANLayer_cus(in_dim=d_model, out_dim=dim_feedforward, num=grid, k=k)
        self.dropout = Dropout(dropout)
        self.linear2 = KANLayer_cus(in_dim=dim_feedforward, out_dim=d_model, num=grid, k=k)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = F._get_activation_fn(activation)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # Simplified for brevity; you can include the fast path checks if needed
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed-forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        bsz, sq_len, hiddim = x.size()
        x = x.view(-1, hiddim)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x.view(bsz, sq_len, hiddim)
        return self.dropout2(x)


class MultiheadAttention(Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, grid=None, k=None, neuron_fun=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Using KANLayer_cus for projections
        self.q_proj = KANLayer_cus(in_dim=embed_dim, out_dim=embed_dim, num=grid, k=k)
        self.k_proj = KANLayer_cus(in_dim=self.kdim, out_dim=embed_dim, num=grid, k=k)
        self.v_proj = KANLayer_cus(in_dim=self.vdim, out_dim=embed_dim, num=grid, k=k)

        # Using KANLayer_cus for output projection
        self.out_proj = KANLayer_cus(in_dim=embed_dim, out_dim=embed_dim, num=grid, k=k)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize KANLayer_cus layers if needed
        pass  # Assuming KANLayer_cus handles its own initialization


    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False) -> Tuple[Tensor, Optional[Tensor]]:

        # Simplified for brevity; include fast path checks if needed
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, "NestedTensor inputs are not supported in this implementation."

        if self.batch_first:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            out_proj=self.out_proj,
        )

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    q_proj=None,
    k_proj=None,
    v_proj=None,
    out_proj=None,
) -> Tuple[Tensor, Optional[Tensor]]:
    is_batched = query.dim() == 3

    if not is_batched:
        # Unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    # Compute in-projection using KANLayer_cus layers
    batch_size = bsz
    # Reshape inputs to 2D
    query_2d = query.contiguous().view(batch_size * tgt_len, embed_dim)
    key_2d = key.contiguous().view(batch_size * src_len, embed_dim)
    value_2d = value.contiguous().view(batch_size * src_len, embed_dim)

    # Apply KANLayer_cus and use only the first output (y)
    q_outs = q_proj(query_2d)
    if isinstance(q_outs, tuple):
        q = q_outs[0]
    else:
        q = q_outs
    q = q.view(batch_size, tgt_len, -1)

    k_outs = k_proj(key_2d)
    if isinstance(k_outs, tuple):
        k = k_outs[0]
    else:
        k = k_outs
    k = k.view(batch_size, src_len, -1)

    v_outs = v_proj(value_2d)
    if isinstance(v_outs, tuple):
        v = v_outs[0]
    else:
        v = v_outs
    v = v.view(batch_size, src_len, -1)

    # Prepare q, k, v for multi-head attention
    q = q.view(batch_size, tgt_len, num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, tgt_len, head_dim)
    k = k.view(batch_size, src_len, num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, src_len, head_dim)
    v = v.view(batch_size, src_len, num_heads, head_dim).transpose(1, 2)  # (batch_size, num_heads, src_len, head_dim)

    # Reshape q, k, v for scaled dot-product attention
    q = q.reshape(batch_size * num_heads, tgt_len, head_dim)
    k = k.reshape(batch_size * num_heads, src_len, head_dim)
    v = v.reshape(batch_size * num_heads, src_len, head_dim)

    # Handle biases if present
    if bias_k is not None and bias_v is not None:
        k = torch.cat([k, bias_k.repeat(batch_size * num_heads, 1, 1)], dim=1)
        v = torch.cat([v, bias_v.repeat(batch_size * num_heads, 1, 1)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    src_len = k.size(1)

    # Merge key padding and attention masks
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(batch_size * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    if not training:
        dropout_p = 0.0

    # Calculate attention
    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.view(batch_size, num_heads, tgt_len, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size * tgt_len, embed_dim)

        # Apply output projection using KANLayer_cus
        out_proj_outs = out_proj(attn_output)
        if isinstance(out_proj_outs, tuple):
            attn_output = out_proj_outs[0]
        else:
            attn_output = out_proj_outs

        attn_output = attn_output.view(batch_size, tgt_len, embed_dim).transpose(0, 1)

        attn_output_weights = attn_output_weights.view(batch_size, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # For the case when need_weights is False
        q = q.view(batch_size, num_heads, tgt_len, head_dim)
        k = k.view(batch_size, num_heads, src_len, head_dim)
        v = v.view(batch_size, num_heads, src_len, head_dim)
        attn_mask = attn_mask.view(batch_size, num_heads, -1, src_len)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size * tgt_len, embed_dim)

        # Apply output projection using KANLayer_cus
        out_proj_outs = out_proj(attn_output)
        if isinstance(out_proj_outs, tuple):
            attn_output = out_proj_outs[0]
        else:
            attn_output = out_proj_outs

        attn_output = attn_output.view(batch_size, tgt_len, embed_dim).transpose(0, 1)

        if not is_batched:
            attn_output = attn_output.squeeze(1)
        return attn_output, None
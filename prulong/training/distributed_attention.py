# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

import torch.distributed as dist

class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, scatter_idx: int, gather_idx: int, group: Any) -> Tensor:
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.group = group

        world_size = dist.get_world_size(group)

        input_list = [t.contiguous() for t in torch.tensor_split(input, world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        return (SeqAllToAll.apply(*grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.group), None, None, None)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention

    def forward(self, query: Tensor, key_values: Tensor, *args, group: Any = None, scatter_idx: int = -2, gather_idx: int = 1, **kwargs) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        #in shape : e.g.,  [s/p:h:]
        query_heads = SeqAllToAll.apply(query, scatter_idx, gather_idx, group)
        key_values_heads = SeqAllToAll.apply(key_values, scatter_idx, gather_idx, group)

        #out shape : e.g., [s:h/p:]
        output_heads = self.local_attn(query_heads, key_values_heads, *args, **kwargs)

        #out e.g., [s/p::h]
        return SeqAllToAll.apply(output_heads, gather_idx, scatter_idx, group)
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast
from src.data.typing import TokenType


def fixed_cross_entropy_with_token_types(
    source: torch.Tensor,
    target: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    loss_layout_scale: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    if num_items_in_batch is not None:
        reduction = "sum" if token_type_ids is None else "none"
    else:
        reduction = "mean"
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction=reduction
    )
    loss_layout = None
    loss_object = None
    if reduction in ("sum", "none"):
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        if token_type_ids is not None:
            layout_tokens_mask = token_type_ids == TokenType.LAYOUT
            object_tokens_mask = token_type_ids == TokenType.OBJECT
            valid_layout_tokens = layout_tokens_mask.sum().clip(min=1)
            valid_object_tokens = object_tokens_mask.sum().clip(min=1)
            loss_layout_masked = loss * layout_tokens_mask
            loss_layout_sum = loss_layout_masked.sum()
            loss_layout = loss_layout_sum / valid_layout_tokens
            loss_object = (loss * object_tokens_mask).sum() / valid_object_tokens
            if loss_layout_scale is not None:
                loss_others_masked = loss * ~layout_tokens_mask
                loss = loss_layout_scale * loss_layout_sum + loss_others_masked.sum()
            else:
                loss = loss.sum()
        loss = loss / num_items_in_batch
    return loss, loss_layout, loss_object


def causal_lm_loss_with_token_types(
    logits,
    labels,
    vocab_size: int,
    token_type_ids: Optional[torch.Tensor] = None,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    loss_layout_scale: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    if token_type_ids is not None:
        token_type_ids = nn.functional.pad(
            token_type_ids,
            (0, 1),
            value=TokenType.PADDING,
        )
        token_type_ids = token_type_ids[..., 1:].contiguous()
        token_type_ids = token_type_ids.view(-1)
        token_type_ids = token_type_ids.to(logits.device)

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss, loss_layout, loss_object = fixed_cross_entropy_with_token_types(
        logits,
        shift_labels,
        token_type_ids=token_type_ids,
        num_items_in_batch=num_items_in_batch,
        ignore_index=ignore_index,
        loss_layout_scale=loss_layout_scale,
        **kwargs,
    )
    return loss, loss_layout, loss_object


@dataclass
class CustomCausalLMOutputWithTokenTypes(CausalLMOutputWithPast):
    loss_layout: Optional[torch.Tensor] = None
    loss_object: Optional[torch.Tensor] = None

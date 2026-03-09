import re
import warnings
import torch
import torch.nn as nn
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import entropy_from_logits
from transformers.trainer import Trainer
from transformers.utils import is_peft_available, logging

if is_peft_available():
    from peft import PeftType

logger = logging.get_logger(__name__)


class MuonWrappedAdamW(torch.optim.Optimizer):
    def __init__(self, adamw_optimizer, muon_optimizer):
        self.adamw_optimizer = adamw_optimizer
        self.muon_optimizer = muon_optimizer
        self._update_param_groups()

    def _update_param_groups(self):
        self.param_groups = (
            self.adamw_optimizer.param_groups + self.muon_optimizer.param_groups
        )

    @torch._disable_dynamo
    def load_state_dict(self, state_dict):
        self.adamw_optimizer.load_state_dict(state_dict["adamw_state"])
        self.muon_optimizer.load_state_dict(state_dict["muon_state"])
        self._update_param_groups()

    @torch._disable_dynamo
    def state_dict(self):
        return {
            "adamw_state": self.adamw_optimizer.state_dict(),
            "muon_state": self.muon_optimizer.state_dict(),
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.adamw_optimizer.step()
        self.muon_optimizer.step()
        return loss

    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True):
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)


class CustomSFTConfig(SFTConfig):
    def __post_init__(self):
        orig_optim = None
        if self.optim == "adamw_muon":
            orig_optim = self.optim
            self.optim = "adamw_torch_fused"
        super().__post_init__()
        if orig_optim is not None:
            self.optim = orig_optim


class CustomSFTTrainer(SFTTrainer):
    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            use_muon = self.args.optim == "adamw_muon"
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if use_muon:
                self.args.optim = "adamw_torch_fused"
                muon_forbidden_layer_names = [
                    r".*embed.*",
                    r".*pos_emb.*",
                    r".*query.*",
                    r".*logits.*",
                    r".*head.*",
                ]
                muon_forbidden_layer_patterns = [
                    re.compile(pattern) for pattern in muon_forbidden_layer_names
                ]

                non_decay_param_group = optimizer_grouped_parameters[1]
                adam_params = []
                muon_params = []
                for n, p in opt_model.named_parameters():
                    if n not in decay_parameters or not p.requires_grad:
                        continue
                    if (
                        p.ndim != 2
                        or isinstance(p, nn.Embedding)
                        or any(
                            pattern.search(n.lower())
                            for pattern in muon_forbidden_layer_patterns
                        )
                    ):
                        adam_params.append(p)
                    else:
                        muon_params.append(p)
                optimizer_grouped_parameters = [
                    {
                        "params": adam_params,
                        "weight_decay": self.args.weight_decay,
                    },
                    non_decay_param_group,
                ]
                muon_optimizer_grouped_parameters = [
                    {
                        "params": muon_params,
                        "weight_decay": self.args.weight_decay,
                    }
                ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model
                )

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if use_muon:
                muon_optimizer = torch.optim.Muon(
                    muon_optimizer_grouped_parameters,
                    lr=optimizer_kwargs["lr"],
                    adjust_lr_fn="match_rms_adamw",
                )
                self.optimizer = MuonWrappedAdamW(self.optimizer, muon_optimizer)

            if (
                "bitsandbytes" in str(optimizer_cls)
                and optimizer_kwargs.get("optim_bits", None) == 8
            ):
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        mode = "train" if self.model.training else "eval"

        # Set aside labels as it will be dropped by super().compute_loss() if a custom `compute_loss_func` is used.
        # This can be removed when this issue is fixed.
        # When using CP or SP, labels are pre-shifted, we must use shift_labels instead.
        labels = inputs["labels"] if "shift_labels" not in inputs else None

        # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
        inputs["use_cache"] = False

        # Request token accuracy from Liger kernel and set token scaling if using DFT loss
        if self.args.use_liger_kernel:
            inputs["return_token_accuracy"] = True
            inputs["use_token_scaling"] = self.args.loss_type == "dft"

        (loss, outputs) = Trainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        len_diff = 0

        # Compute entropy
        if not self.args.use_liger_kernel:  # liger doesn't return logits
            with torch.no_grad():
                per_token_entropy = entropy_from_logits(outputs.logits)
                # When using Prompt Tuning, skip the virtual tokens in logits before entropy computation, since they
                # do not correspond to actual input tokens.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type
                    != PeftType.PREFIX_TUNING
                ):
                    per_token_entropy = per_token_entropy[:, self.num_virtual_tokens :]
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    # HACK: for BPT model, there is a hidden <sos> token at the beginning, so the per_token_entropy is longer
                    len_diff = per_token_entropy.shape[1] - attention_mask.shape[1]
                    if len_diff > 0:
                        per_token_entropy = per_token_entropy[:, len_diff:]
                    entropy = (
                        torch.sum(per_token_entropy * attention_mask)
                        / attention_mask.sum()
                    )
                elif "position_ids" in inputs:
                    entropy = torch.mean(per_token_entropy)
                else:
                    raise ValueError(
                        "Expected 'attention_mask' or 'position_ids' in inputs."
                    )
                entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(inputs["attention_mask"].sum())
                    .sum()
                    .item()
                )
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(
                    inputs["position_ids"].size(1), device=inputs["position_ids"].device
                )
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
                )
            else:
                raise ValueError(
                    "Expected 'attention_mask' or 'position_ids' in inputs."
                )
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        if self.args.use_liger_kernel:
            if (
                hasattr(outputs, "token_accuracy")
                and outputs.token_accuracy is not None
            ):
                token_accuracy = (
                    self.accelerator.gather_for_metrics(outputs.token_accuracy)
                    .mean()
                    .item()
                )
                self._metrics[mode]["mean_token_accuracy"].append(token_accuracy)
            else:
                # liger-kernel<=0.6.4 can omit token_accuracy even when requested; fixed for Gemma3 in
                # https://github.com/linkedin/Liger-Kernel/pull/1010
                warnings.warn(
                    "liger-kernel did not return token_accuracy when requested. The mean_token_accuracy metric will "
                    "not be logged. This may indicate an outdated liger-kernel version. Consider upgrading to the "
                    "latest version. If the issue persists after upgrading, please report it to the liger-kernel "
                    "repository.",
                    stacklevel=2,
                )
        else:
            # Compute accuracy from logits using argmax (traditional method)
            with torch.no_grad():
                if "shift_labels" in inputs:
                    # When using CP or SP, labels are pre-shifted. We must use these (and cannot manually shift) because:
                    # - The first discarded token from inputs["labels"] actually belongs to process n-1
                    # - The last logits require the label from process n+1
                    shift_logits = outputs.logits.contiguous()
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    if len_diff > 0:
                        shift_labels = labels.contiguous()
                    else:
                        shift_labels = labels[..., 1:].contiguous()

                # Prompt Tuning and P-Tuning output logits for virtual tokens but Prefix-Tuning does not.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type
                    != PeftType.PREFIX_TUNING
                ):
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                # Get predictions
                predictions = shift_logits.argmax(dim=-1)

                # Create mask for non-padding tokens (assuming ignore_index is -100)
                mask = shift_labels != -100

                # Calculate accuracy only on non-padding tokens
                correct_predictions = (predictions == shift_labels) & mask
                total_tokens = mask.sum()
                correct_tokens = correct_predictions.sum()

                # Gather the correct_tokens and total_tokens across all processes
                correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                total_tokens = self.accelerator.gather_for_metrics(total_tokens)

                # Compute the mean token accuracy and log it
                total_sum = total_tokens.sum()
                accuracy = (
                    (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                )
                self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # Log auxiliary loss if enabled (applies to both Liger and non-Liger)
        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss
            aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
            self._metrics[mode]["aux_loss"].append(aux_loss)

        if "loss_layout" in outputs:
            loss_layout = outputs.loss_layout
            loss_layout = self.accelerator.gather_for_metrics(loss_layout).mean().item()
            self._metrics[mode]["loss_layout"].append(loss_layout)

        if "loss_object" in outputs:
            loss_object = outputs.loss_object
            loss_object = self.accelerator.gather_for_metrics(loss_object).mean().item()
            self._metrics[mode]["loss_object"].append(loss_object)

        return (loss, outputs) if return_outputs else loss

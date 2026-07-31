import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def initialize_parameter_groups_from_ema(
    state_dict: Mapping[str, torch.Tensor],
    ema_params: Mapping[str, torch.Tensor],
    parameter_groups: Mapping[str, Sequence[str]],
    groups_to_initialize: Sequence[str],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Replace selected model parameter groups with checkpoint EMA values."""
    unknown_groups = set(groups_to_initialize) - set(parameter_groups)
    if unknown_groups:
        raise ValueError(
            f"Unknown EMA initialization groups: {sorted(unknown_groups)}. "
            f"Available groups: {sorted(parameter_groups)}"
        )

    initialized_state_dict = dict(state_dict)
    initialized_keys: list[str] = []

    for group_name in groups_to_initialize:
        prefixes = parameter_groups[group_name]
        group_keys: list[str] = []

        for ema_name, ema_param in ema_params.items():
            if not any(
                ema_name == prefix or ema_name.startswith(f"{prefix}.")
                for prefix in prefixes
            ):
                continue

            state_name = f"model.{ema_name}"
            if state_name not in initialized_state_dict:
                raise KeyError(
                    f"EMA parameter '{ema_name}' from group '{group_name}' "
                    f"is missing from the model state dict."
                )
            if initialized_state_dict[state_name].shape != ema_param.shape:
                raise ValueError(
                    f"Shape mismatch for '{state_name}': "
                    f"model {initialized_state_dict[state_name].shape} vs "
                    f"EMA {ema_param.shape}"
                )

            initialized_state_dict[state_name] = ema_param
            group_keys.append(state_name)

        if not group_keys:
            raise ValueError(
                f"No EMA parameters found for initialization group '{group_name}'."
            )
        initialized_keys.extend(group_keys)

    return initialized_state_dict, initialized_keys


class ExponentialMovingAverage:
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        submodules_to_ignore: Sequence[str] | None = None,
    ):
        """
        Args:
          model: The `torch.nn.Module` whose parameters will be tracked.
          decay: The exponential decay.
        """
        self.decay = decay

        self.submodules_to_ignore = (
            tuple(submodules_to_ignore) if submodules_to_ignore is not None else tuple()
        )

        # NOTE: We store all parameters regardless of `requires_grad` status
        # since only the confidence model is trained at the last phase,
        # while other parameters are frozen.
        self.shadow_params: dict[str, torch.Tensor] = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if not name.startswith(self.submodules_to_ignore)
        }

        # For temporary storage of parameters when applying EMA weights for evaluation.
        self.collected_params: dict[str, torch.Tensor] = {}
        self.device = next(iter(self.shadow_params.values())).device

    def to(self, device: torch.device):
        self.device = device
        self.shadow_params = {k: v.to(device) for k, v in self.shadow_params.items()}

    def update(self, model: torch.nn.Module):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          model: The `torch.nn.Module` containing the parameters to update.
        """
        decay = self.decay
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    s_param = self.shadow_params[name]
                    s_param.sub_(one_minus_decay * (s_param - param))

    def compatible(self, state_dict: dict[str, Any]) -> bool:
        """
        Check if the model parameters are compatible with the stored EMA parameters.
        Args:
            state_dict: The state dictionary of EMA to check compatibility with.
        """
        incoming_params = state_dict["shadow_params"]
        missing_keys = set(self.shadow_params.keys()) - set(incoming_params.keys())
        if missing_keys:
            warnings.warn(
                f"Missing keys in incoming model: {missing_keys}\n"
                f"EMA keys: {set(self.shadow_params.keys())}\n"
                f"Incoming keys: {set(incoming_params.keys())}"
            )
            return False
        unexpected_keys = set(incoming_params.keys()) - set(self.shadow_params.keys())
        if unexpected_keys:
            warnings.warn(
                f"Unexpected keys in incoming model: {unexpected_keys}\n"
                f"EMA keys: {set(self.shadow_params.keys())}\n"
                f"Incoming keys: {set(incoming_params.keys())}"
            )
            return False

        for name, s_param in self.shadow_params.items():
            param = incoming_params[name]
            if param.data.shape != s_param.data.shape:
                warnings.warn(
                    f"Parameter {name} shape mismatch: "
                    f"EMA {s_param.data.shape} vs Model {param.data.shape}"
                )
                return False
        return True

    def state_dict(self):
        return dict(
            decay=self.decay,
            shadow_params=self.shadow_params,
        )

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        device: torch.device,
    ):
        self.decay = state_dict["decay"]
        # Restore as dictionary
        for k, v in state_dict["shadow_params"].items():
            if k not in self.shadow_params:
                warnings.warn(f"Key {k} not found in current EMA parameters.")
            elif v.shape != self.shadow_params[k].shape:
                raise ValueError(
                    f"Shape mismatch for key {k}: "
                    f"EMA {self.shadow_params[k].shape} vs Loaded {v.shape}"
                )
            else:
                self.shadow_params[k] = v.clone().detach().to(device)

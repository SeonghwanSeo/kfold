import warnings
from typing import Any

import torch


class ExponentialMovingAverage:
    """from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py,
    Apache-2.0 license
    Maintains (exponential) moving average of a set of parameters."""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        use_num_updates: bool = True,
    ):
        """
        Args:
          model: The `torch.nn.Module` whose parameters will be tracked.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None

        # Save as {name: tensor}
        self.shadow_params: dict[str, torch.Tensor] = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.collected_params: dict[str, torch.Tensor] = {}

    def update(self, model: torch.nn.Module):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          model: The `torch.nn.Module` containing the parameters to update.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))

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
        if len(incoming_params) != len(self.shadow_params):
            warnings.warn(
                f"Parameter count mismatch: "
                f"EMA has {len(self.shadow_params)} vs Model has {len(incoming_params)}"
            )
            return False

        for name, s_param in self.shadow_params.items():
            if name not in incoming_params:
                warnings.warn(f"Key {name} not found in incoming model.")
                return False

            param = incoming_params[name]
            if param.data.shape != s_param.data.shape:
                warnings.warn(
                    f"Parameter {name} shape mismatch: "
                    f"EMA {s_param.data.shape} vs Model {param.data.shape}"
                )
                return False
        return True

    def copy_to(self, model: torch.nn.Module):
        """
        Copy current parameters into given collection of parameters.
        Args:
          model: The `torch.nn.Module` to update with the stored moving averages.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name].data)

    def store(self, model: torch.nn.Module):
        """
        Save the current parameters for restoring later.
        Args:
          model: The `torch.nn.Module` whose parameters are to be temporarily stored.
        """
        self.collected_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

    def restore(self, model: torch.nn.Module):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          model: The `torch.nn.Module` to update with the stored parameters.
        """
        for name, param in model.named_parameters():
            if name in self.collected_params:
                param.data.copy_(self.collected_params[name].data)

    def state_dict(self):
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        device: torch.device,
    ):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        # Restore as dictionary
        self.shadow_params = {
            k: v.to(device) for k, v in state_dict["shadow_params"].items()
        }

    def to(self, device: torch.device):
        self.shadow_params = {k: v.to(device) for k, v in self.shadow_params.items()}

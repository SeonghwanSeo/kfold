from abc import ABC, abstractmethod

import torch

import kfold.constants as C
from kfold.data.types.model_input import FoldingInput
from kfold.model.modules.score_model import BaseScoreModel
from kfold.utils.geometry.random_augment import do_centering
from kfold.utils.geometry.rigid_align import weighted_rigid_align
from kfold.utils.misc import repeat_dim
from kfold.utils.registry import STRUCTURE_MODULE, BaseConfig


@STRUCTURE_MODULE.register()
class BaseStructureModule(ABC):
    """High-level flow-based framework for structure generation."""

    def __init__(self, cfg: BaseConfig, score_model: BaseScoreModel):
        self.cfg = cfg
        self.score_model = score_model

    # === Model Call === #
    @abstractmethod
    def forward_model(
        self,
        x_noisy: torch.Tensor,
        t_hat: torch.Tensor | float,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        prior_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the score model.

        Parameters
        ----------
        x_noisy : torch.Tensor
            Noisy atom coordinates. Shape (B, N, La, 3),
            where N is number of diffusion samples and La is number of atoms.
        t_hat : torch.Tensor | float
            Time step or noise level. Shape (B, N) or float.
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        s_inputs : torch.Tensor
            Input sequence embeddings. Shape (B, Lt, c_s).
        s_trunk : torch.Tensor
            Trunk sequence embeddings. Shape (B, Lt, c_s).
        z_trunk : torch.Tensor
            Trunk pairwise embeddings. Shape (B, Lt, Lt, c_z).

        Returns
        -------
        denoised_coords : torch.Tensor
            Denoised atom coordinates. Shape (B, N, Lt, 3).
        """

    # === Sampling and Interpolation Methods === #
    @abstractmethod
    def sample_structure(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        num_steps: int | None = None,
        num_diffusion_samples: int = 1,
        max_parallel_samples: int | None = None,
        return_traj: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample structures via diffusion sampling."""

    @abstractmethod
    def sample_noise_level(
        self,
        batch_size: int,
        num_diffusion_samples: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample noise levels (t_hat) during model training. Shape: (B, N)."""

    @abstractmethod
    def get_sampling_schedule(
        self,
        num_steps: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Get the noise schedule for diffusion sampling. Shape: (num_steps,)."""

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Compute loss weights based on noise levels t_hat. Shape: (B, N)."""
        return torch.ones_like(t_hat)

    def apply_random_augmentation(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply random augmentation to coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            Coordinates. Shape (*, L, 3).
        mask : torch.Tensor
            Mask. Shape (*, L).

        Returns
        -------
        augmented_coords : torch.Tensor
            Augmented Coordinates. Shape (*, La, 3).
        """
        # Default: simple centering without augmentation
        coords = do_centering(coords, mask, mask_to_zero=True)
        return coords

    def sample_label(
        self, f_input: FoldingInput, num_diffusion_samples: int = 1
    ) -> torch.Tensor:
        """Sample label coordinates from input features.
        Return shape: [B, N, La, 3], where N is number of diffusion samples
        and La is number of atoms.

        Parameters
        -----------
        f_input: FoldingInput
            Input features
        num_diffusion_samples:
            Number of diffusion samples

        Returns
        -------
        label_coords: torch.Tensor
            Label coordinates. Shape: [B, N, La, 3]
        """
        # if the model is equivariance, skip augment
        random_augment = True

        holo_coords = self.sample_holo(f_input, num_diffusion_samples, random_augment)

        return holo_coords

    @abstractmethod
    def sample_prior(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
        label_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the prior distribution.
        Return shape: [B, N, La, 3], where N is number of diffusion samples
        and La is number of atoms.

        Parameters
        -----------
        f_input: FoldingInput
            Input features
        num_diffusion_samples:
            Number of diffusion samples
        label_coords: torch.Tensor
            Label coordinates. Shape: [B, N, La, 3]
            where N is the number of diffusion samples

        Returns
        -------
        prior_coords: torch.Tensor
            Prior coordinates. Shape: [B, N, La, 3]
        """

    @abstractmethod
    def interpolate(
        self,
        noise_coords: torch.Tensor,
        label_coords: torch.Tensor,
        t_hat: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate between noise and label coordinates.

        We may want to perform kabsch alignment here before interpolation.

        Parameters
        ----------
        noise_coords : torch.Tensor
            The noisy coordinates. Shape (B, N, La, 3).
        label_coords : torch.Tensor
            The label coordinates. Shape (B, N, La, 3).
        t_hat : torch.Tensor
            The dffusion noise levels (or sigmas of EDM). Shape (B, N).
        mask : torch.Tensor
            The atom mask. Shape (B, La).

        Returns
        -------
        interpolated_coords : torch.Tensor
            The interpolated coordinates. Shape (B, N, La, 3).
        """

    # === For model training === #
    def training_step(
        self,
        f_input: FoldingInput,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        diffusion_batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step for the structure module.
        See Section 5 of EDM paper.
        """
        batch_size = f_input.batch_size  # =B
        num_diffusion_samples = diffusion_batch_size  # =N
        mask = f_input.atom.pad_mask  # [B, La]

        with torch.no_grad():
            t_hat = self.sample_noise_level(
                batch_size, num_diffusion_samples, device=f_input.device
            )  # [B, N]

            # sample xT from label
            label_coords = self.sample_label(f_input, num_diffusion_samples)
            label_coords = label_coords * mask[..., None, :, None]

            # sample x0 from prior
            prior_coords = self.sample_prior(f_input, num_diffusion_samples, label_coords)
            prior_coords = prior_coords * mask[..., None, :, None]

            # sample xt via interpolation
            noised_atom_coords = self.interpolate(prior_coords, label_coords, t_hat, mask)
            noised_atom_coords = noised_atom_coords * mask[..., None, :, None]

        denoised_atom_coords = self.forward_model(
            x_noisy=noised_atom_coords,  # [B, N, La, 3]
            t_hat=t_hat,  # [B, N]
            f_input=f_input,
            s_inputs=s_inputs,  # [B, Lt, c_s]
            s_trunk=s_trunk,  # [B, Lt, c_s]
            z_trunk=z_trunk,  # [B, Lt, Lt, c_z]
            prior_coords=prior_coords,  # [B, N, La, 3]
        )  # [B, N, La, 3]

        loss_weights = self.loss_weights(t_hat)  # [B, N]

        return {
            "t_hat": t_hat,
            "loss_weights": loss_weights,
            "prior_atom_coords": prior_coords,
            "noised_atom_coords": noised_atom_coords,
            "denoised_atom_coords": denoised_atom_coords,
            "true_atom_coords": label_coords,
        }

    # === Sampling holo/apo structures === #
    def sample_holo(
        self,
        f_input: FoldingInput,
        num_diffusion_samples: int = 1,
        random_augment: bool = True,
    ) -> torch.Tensor:
        """Sample holo structures from input for model training.

        Parameters
        ----------
        f_input : FoldingInput
            FoldingInput object containing model inputs.
        num_diffusion_samples : int, optional
            Number of diffusion samples to generate, by default 1.
        random_augment: bool
            Whether to apply random augmentation to holo coordinates.

        Returns
        -------
        holo_coords : torch.Tensor
            Sampled holo coordinates. Shape (B, N, L, 3),
            where N is number of diffusion samples and L is the number of atoms.
        """
        holo_coords = f_input.atom.label_coords  # [B, L, 3]
        atom_mask = f_input.atom.resolved_mask  # [B, L]

        # repeat holo coords
        holo_coords = repeat_dim(
            holo_coords, num_diffusion_samples, dim=-3
        )  # [B, N, L, 3]
        atom_mask = atom_mask.unsqueeze(-2)  # [B, 1, L]

        # Apply coordinate augmentation or centering
        if random_augment:
            holo_coords = self.apply_random_augmentation(holo_coords, atom_mask)

        return holo_coords  # [B, N, L, 3]


@STRUCTURE_MODULE.register()
class BaseEDM(BaseStructureModule):
    """High-level EDM framework for structure generation.

    See Section 3.7, Algorithm18 (Sample Diffusion) of AlphaFold3 paper.
    """

    # === EDM (Elucidating Diffusion Models) preconditioning coefficients === #
    # Reference: Karras et al., "Elucidating the Design Space of Diffusion-Based "
    # Generative Models"
    # Check Boltz Implementation
    @abstractmethod
    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """Skip connection coefficient for EDM preconditioning."""

    @abstractmethod
    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """Output scaling coefficient for EDM preconditioning."""

    @abstractmethod
    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        """Input scaling coefficient for EDM preconditioning."""

    @abstractmethod
    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        """Noise level conditioning coefficient for EDM preconditioning."""

    def loss_weights(self, t_hat: torch.Tensor) -> torch.Tensor:
        """Compute loss weights based on noise levels t_hat. Shape: (B, N)."""
        return 1 / self.c_out(t_hat) ** 2


@STRUCTURE_MODULE.register()
class BaseECSI(BaseEDM):
    """High-level ECSI framework for structure generation."""

    def __init__(self, cfg: BaseConfig, score_model: BaseScoreModel):
        super().__init__(cfg, score_model)
        # alignment_entity_strategy: None (default) or "largest" or "random_non_ligand"
        self.alignment_entity_strategy = getattr(cfg, "alignment_entity_strategy", None)
        # alignment_level: "chain" (default) or "entity"
        self.alignment_level = getattr(cfg, "alignment_level", "chain")

    def _select_largest_entity(
        self,
        atom_entity_id: torch.Tensor,
        apo_mask_ref: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Select the largest entity (by atom count) for each batch element."""
        B, L = atom_entity_id.shape
        valid_mask = apo_mask_ref & (atom_entity_id >= 0)
        if not valid_mask.any():
            selected_entity_id = atom_entity_id.new_full((B,), -1)
            if num_samples > 1:
                selected_entity_id = selected_entity_id.unsqueeze(1).expand(
                    B, num_samples
                )
            return selected_entity_id

        entity_indices = atom_entity_id.clamp(min=0)
        max_entity_id = int(entity_indices.max().item()) + 1
        entity_counts = torch.zeros(
            B, max_entity_id, device=atom_entity_id.device, dtype=torch.long
        )
        entity_counts.scatter_add_(1, entity_indices, valid_mask.long())
        selected_entity_id = entity_counts.argmax(dim=-1)

        has_valid = valid_mask.any(dim=1)
        selected_entity_id = torch.where(
            has_valid,
            selected_entity_id,
            torch.full_like(selected_entity_id, -1),
        )

        if num_samples > 1:
            selected_entity_id = selected_entity_id.unsqueeze(1).expand(B, num_samples)

        return selected_entity_id

    def _select_random_non_ligand_entity(
        self,
        atom_entity_id: torch.Tensor,
        atom_chain_type: torch.Tensor,
        apo_mask_ref: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Select a random non-ligand entity, falling back to random if only ligands."""
        B, L = atom_entity_id.shape
        valid_mask = apo_mask_ref & (atom_entity_id >= 0)
        if not valid_mask.any():
            selected_entity_id = atom_entity_id.new_full((B,), -1)
            if num_samples > 1:
                selected_entity_id = selected_entity_id.unsqueeze(1).expand(
                    B, num_samples
                )
            return selected_entity_id

        max_entity_id = int(atom_entity_id.max().item()) + 1
        entity_indices = atom_entity_id.clamp(min=0)
        entity_counts = torch.zeros(
            B, max_entity_id, device=atom_entity_id.device, dtype=torch.long
        )
        entity_counts.scatter_add_(1, entity_indices, valid_mask.long())
        entity_exists = entity_counts > 0

        is_non_ligand_atom = (atom_chain_type != C.ChainType.LIGAND) & valid_mask
        non_ligand_counts = torch.zeros(
            B, max_entity_id, device=atom_entity_id.device, dtype=torch.long
        )
        non_ligand_counts.scatter_add_(1, entity_indices, is_non_ligand_atom.long())
        non_ligand_entities = non_ligand_counts > 0

        non_ligand_mask = entity_exists & non_ligand_entities

        has_non_ligand = non_ligand_mask.any(dim=1)
        candidate_mask = torch.where(
            has_non_ligand.unsqueeze(1),
            non_ligand_mask,
            entity_exists,
        )

        if num_samples > 1:
            # Independent random selection for each sample
            candidate_mask = candidate_mask.unsqueeze(1)  # (B, 1, max_entity_id)
            random_weights = torch.rand(
                B, num_samples, max_entity_id, device=atom_entity_id.device
            )
            random_weights = random_weights * candidate_mask.float()
            selected_entity_id = random_weights.argmax(dim=-1)  # (B, N)

            has_valid = valid_mask.any(dim=1).unsqueeze(1).expand(B, num_samples)
            selected_entity_id = torch.where(
                has_valid,
                selected_entity_id,
                torch.full_like(selected_entity_id, -1),
            )
        else:
            random_weights = torch.rand(B, max_entity_id, device=atom_entity_id.device)
            random_weights = random_weights * candidate_mask.float()
            selected_entity_id = random_weights.argmax(dim=-1)  # (B,)

            has_valid = valid_mask.any(dim=1)
            selected_entity_id = torch.where(
                has_valid,
                selected_entity_id,
                torch.full_like(selected_entity_id, -1),
            )

        return selected_entity_id

    def _select_entity_for_alignment(
        self,
        atom_entity_id: torch.Tensor,
        atom_chain_type: torch.Tensor,
        apo_mask_ref: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor | None:
        """Select entity IDs for alignment based on configured strategy."""
        if not self.alignment_entity_strategy:
            return None
        elif self.alignment_entity_strategy == "random_non_ligand":
            return self._select_random_non_ligand_entity(
                atom_entity_id, atom_chain_type, apo_mask_ref, num_samples=num_samples
            )
        elif self.alignment_entity_strategy == "largest":
            return self._select_largest_entity(
                atom_entity_id, apo_mask_ref, num_samples=num_samples
            )
        else:
            raise ValueError(
                f"Unknown alignment_entity_strategy: {self.alignment_entity_strategy}"
            )

    def align_apo_to_label(
        self,
        apo_coords: torch.Tensor,
        label_coords: torch.Tensor,
        f_input: FoldingInput,
    ) -> torch.Tensor:
        """Align apo coords to label coords using selected entity selection strategy.

        Strategies:
        - "largest": align using the largest apo entity (default)
        - "random_non_ligand": align using a random non-ligand entity
                               (falls back to random selection if only ligands exist)
        """
        apo_mask = ~(apo_coords == 0.0).all(-1)
        label_mask = ~(label_coords == 0.0).all(-1)

        added_batch = False
        if apo_coords.dim() == 3:
            apo_coords = apo_coords.unsqueeze(0)
            label_coords = label_coords.unsqueeze(0)
            apo_mask = apo_mask.unsqueeze(0)
            label_mask = label_mask.unsqueeze(0)
            added_batch = True

        B, N, L = apo_coords.shape[:3]
        apo_mask_ref = apo_mask.any(dim=1)

        if self.alignment_level == "entity":
            token_entity_id = f_input.token.entity_id
        elif self.alignment_level == "chain":
            token_entity_id = f_input.token.asym_id
        else:
            raise ValueError(f"Unknown alignment_level: {self.alignment_level}")

        token_chain_type = f_input.token.chain_type
        atom_token_index = f_input.atom.token_index
        if token_entity_id.dim() == 1:
            token_entity_id = token_entity_id.unsqueeze(0)
        if token_chain_type.dim() == 1:
            token_chain_type = token_chain_type.unsqueeze(0)
        if atom_token_index.dim() == 1:
            atom_token_index = atom_token_index.unsqueeze(0)
        if token_entity_id.shape[0] == 1 and B > 1:
            token_entity_id = token_entity_id.expand(B, -1)
        if token_chain_type.shape[0] == 1 and B > 1:
            token_chain_type = token_chain_type.expand(B, -1)
        if atom_token_index.shape[0] == 1 and B > 1:
            atom_token_index = atom_token_index.expand(B, -1)

        atom_entity_id = token_entity_id.gather(-1, atom_token_index.clamp(min=0))
        atom_chain_type = token_chain_type.gather(-1, atom_token_index.clamp(min=0))

        selected_entity_id = self._select_entity_for_alignment(
            atom_entity_id, atom_chain_type, apo_mask_ref, num_samples=N
        )

        if selected_entity_id is None:
            entity_mask_exp = apo_mask_ref.unsqueeze(1)
        else:
            if selected_entity_id.ndim == 1:
                selected_entity_id = selected_entity_id.unsqueeze(1).expand(B, N)

            entity_mask = (
                atom_entity_id.unsqueeze(1) == selected_entity_id.unsqueeze(2)
            ) & apo_mask_ref.unsqueeze(1)
            entity_mask_exp = entity_mask

        align_mask = entity_mask_exp & apo_mask & label_mask
        align_weights = align_mask.to(dtype=apo_coords.dtype)
        apo_coords = weighted_rigid_align(
            coords=apo_coords,
            target=label_coords,
            weights=align_weights,
            mask=align_mask,
        )
        apo_coords = apo_coords * apo_mask[..., None]

        if added_batch:
            apo_coords = apo_coords.squeeze(0)

        return apo_coords

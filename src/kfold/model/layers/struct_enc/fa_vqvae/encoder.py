import einops
import torch
import torch.nn.functional as F
from torch import nn

from .pair_bias_attn import MultiheadAttnAndTransition
from .pair_update import PairReprUpdate
from .utils import data_transforms
from .utils.coords_utils import bond_angles, canonicalize_residue_coords


class PairwiseResBlock(nn.Module):
    """Residual block for 2D pairwise features."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.bn2(self.conv2(F.gelu(self.bn1(self.conv1(x)))))
        return F.gelu(x + r)


class PairImageBackbone(nn.Module):
    """Lightweight ResNet stack that processes pairwise distance images."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.resblocks = nn.ModuleList(PairwiseResBlock(hidden_dim) for _ in range(depth))

    def init_features(self, pair_inputs: torch.Tensor, pair_mask: torch.Tensor):
        pair_rep = self.project(pair_inputs)
        return pair_rep.masked_fill(~pair_mask.unsqueeze(-1), 0.0)

    def step(self, pair_rep: torch.Tensor, pair_mask: torch.Tensor, idx: int):
        block = self.resblocks[idx]
        B, L, A, _, C = pair_rep.shape
        x = pair_rep.reshape(B * L, A, A, C).permute(0, 3, 1, 2).contiguous()
        x = block(x)
        x = x.permute(0, 2, 3, 1).reshape(B, L, A, A, C)
        return x.masked_fill(~pair_mask.unsqueeze(-1), 0.0)


class AtomisticImageEncoder(nn.Module):
    def __init__(
        self,
        c_s: int = 256,
        c_z: int = 128,
        c_out: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        update_pair_repr_every_n: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.c_s: int = c_s
        self.c_z: int = c_z
        self.c_out: int = c_out
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.distance_temperature: float = kwargs.get("distance_temperature", 10.0)

        feat_dims = [4, 1, 3 * 21, 4 * 21 + 4]
        self.feat_layers_atom1 = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, c_s),
            )
            for d in feat_dims
        )
        self.feat_layers_atom2 = nn.Sequential(
            nn.LayerNorm(c_s * len(feat_dims)),
            nn.Linear(c_s * len(feat_dims), c_s),
        )

        feat_dims = [4 * 37, 37, 3 * 21, 4 * 21 + 4]
        self.feat_layers_seq1 = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c_s)) for d in feat_dims
        )
        self.feat_layers_seq2 = nn.Sequential(
            nn.LayerNorm(c_s * len(feat_dims)),
            nn.Linear(c_s * len(feat_dims), c_s),
        )

        self.fuse_layer = nn.Sequential(
            nn.Linear(c_s * 2, c_s),
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.LayerNorm(c_s),
        )

        pair_input_dim = 5  # distance + 3-vector + mask
        self.pair_image_backbone = PairImageBackbone(
            input_dim=pair_input_dim,
            hidden_dim=c_z,
            depth=n_layers,
        )

        self.atom_transformer_layers = nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    c_s=c_s,
                    c_z=c_z + 1,
                    n_heads=n_heads,
                )
                for _ in range(n_layers)
            ]
        )

        self.atom_pair_update_layers = nn.ModuleList(
            [
                (PairReprUpdate(c_s, c_z) if i % update_pair_repr_every_n == 0 else None)
                for i in range(n_layers - 1)
            ]
        )

        self.pooling = nn.Sequential(
            nn.LayerNorm(c_s * 37),
            nn.Linear(c_s * 37, c_s),
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
        )
        self.pre_vq_proj = nn.Linear(c_s, c_out)

    def forward(
        self,
        aatypes: torch.Tensor,
        coords: torch.Tensor,
        res_idx: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        B, L = aatypes.shape
        A = 37
        if coords.shape != (B, L, A, 3):
            raise ValueError(f"Expected coords shape (B, L, 37, 3), got {coords.shape}")

        coords, atom_mask, res_mask = self._prepare_inputs(coords)

        if attention_mask is not None:
            res_mask = res_mask & attention_mask.bool()
            atom_mask = atom_mask & attention_mask.bool().unsqueeze(-1)

        with torch.autocast(coords.device.type, enabled=False):
            coords_rel = canonicalize_residue_coords(coords, atom_mask)

        bb_angles = self.get_backbone_angles(coords, atom_mask, res_idx)
        sc_angles = self.get_sidechain_angles(coords, aatypes, atom_mask)
        x_atom, bond_lengths, rel_coords_feat = self._atom_level_features(
            coords_rel, atom_mask, bb_angles, sc_angles
        )
        x_seq = self._sequence_level_features(
            rel_coords_feat, bond_lengths, bb_angles, sc_angles, res_mask
        )

        x = torch.cat([x_atom, x_seq.unsqueeze(-2).expand(-1, -1, A, -1)], dim=-1)
        x = self.fuse_layer(x)

        pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(3)

        with torch.autocast(coords.device.type, enabled=False):
            rel_delta = coords_rel.unsqueeze(3) - coords_rel.unsqueeze(2)
            pair_dist = torch.linalg.norm(rel_delta, dim=-1, keepdim=True)
            pair_bias = torch.exp(-pair_dist / self.distance_temperature)
            pair_bias = pair_bias.masked_fill(~pair_mask.unsqueeze(-1), 0.0)

        pair_inputs = torch.cat(
            [pair_dist, rel_delta, pair_mask.unsqueeze(-1).float()],
            dim=-1,
        )
        pair_rep = self.pair_image_backbone.init_features(pair_inputs, pair_mask)

        for i in range(self.n_layers):
            pair_rep = self.pair_image_backbone.step(pair_rep, pair_mask, i)

            pair_features = torch.cat([pair_rep, pair_bias], dim=-1)
            x = self.atom_transformer_layers[i](x, pair_features, atom_mask)

            if i < self.n_layers - 1 and self.atom_pair_update_layers[i] is not None:
                pair_rep = self.atom_pair_update_layers[i](x, pair_rep, atom_mask)
                pair_rep = pair_rep.masked_fill(~pair_mask.unsqueeze(-1), 0.0)

        return self._pool_and_project(x)

    def _prepare_inputs(self, coords: torch.Tensor):
        mask = coords.isfinite().all(dim=-1)
        residue_mask = mask.any()
        coords = coords.masked_fill(~mask.unsqueeze(-1), 0.0)
        return coords, mask, residue_mask

    def _atom_level_features(
        self,
        coords_rel: torch.Tensor,
        mask: torch.Tensor,
        bb_angles: torch.Tensor,
        sc_angles: torch.Tensor,
    ):
        bond_lengths = torch.linalg.norm(coords_rel, dim=-1, keepdim=True)
        bond_lengths = bond_lengths.masked_fill(~mask.unsqueeze(-1), 0.0)
        A = coords_rel.shape[2]
        bb_angles_atom = bb_angles.unsqueeze(-2).expand(-1, -1, A, -1)
        sc_angles_atom = sc_angles.unsqueeze(-2).expand(-1, -1, A, -1)
        rel_coords_feat = torch.cat([coords_rel, mask[..., None].float()], dim=-1)

        feat_list = [
            self.feat_layers_atom1[i](x)
            for i, x in enumerate(
                [rel_coords_feat, bond_lengths, bb_angles_atom, sc_angles_atom]
            )
        ]
        x_atom = torch.cat(feat_list, dim=-1)
        x_atom = self.feat_layers_atom2(x_atom)
        return x_atom, bond_lengths, rel_coords_feat

    def _sequence_level_features(
        self,
        rel_coords_feat: torch.Tensor,
        bond_lengths: torch.Tensor,
        bb_angles: torch.Tensor,
        sc_angles: torch.Tensor,
        residue_mask: torch.Tensor,
    ):
        rel_coords_flat = einops.rearrange(rel_coords_feat, "... a t -> ... (a t)")
        bond_lengths_flat = einops.rearrange(bond_lengths, "... a t -> ... (a t)")
        feat_list = [
            self.feat_layers_seq1[i](x)
            for i, x in enumerate(
                [rel_coords_flat, bond_lengths_flat, bb_angles, sc_angles]
            )
        ]
        x_seq = torch.cat(feat_list, dim=-1)
        x_seq = self.feat_layers_seq2(x_seq)
        x_seq = x_seq.masked_fill(~residue_mask.unsqueeze(-1), 0.0)
        return x_seq

    def _pool_and_project(self, x: torch.Tensor):
        return self.pre_vq_proj(self.pooling(x.flatten(-2)))

    def get_backbone_angles(self, coords, mask, residue_index=None):
        residue_mask = mask.sum(-1).bool()
        # All three backbone atoms (N, CA, C) must be present for angles to be valid.
        backbone_valid = mask[:, :, :3].all(dim=-1)  # [B, L]
        b, n = coords.shape[0], coords.shape[1]

        if residue_index is not None:
            idx = residue_index.to(coords.device).long()
        else:
            idx = torch.arange(n, device=coords.device).unsqueeze(0).expand(b, -1)

        N = coords[:, :, 0, :]
        CA = coords[:, :, 1, :]
        C = coords[:, :, 2, :]
        theta_1 = bond_angles(N, CA, C)
        theta_2 = bond_angles(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])
        theta_3 = bond_angles(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])

        # Cross-residue angles require consecutive residues AND valid backbone in both.
        pair_backbone_valid = backbone_valid[:, :-1] & backbone_valid[:, 1:]
        good_pair = (idx[:, 1:] - idx[:, :-1] == 1) & pair_backbone_valid
        theta_1 = theta_1.masked_fill(~backbone_valid, 0.0)
        theta_2 = theta_2.masked_fill(~good_pair, 0.0)
        theta_3 = theta_3.masked_fill(~good_pair, 0.0)

        zero_pad = torch.zeros((b, 1), device=coords.device)
        theta_2 = torch.cat([theta_2, zero_pad], dim=-1)
        theta_3 = torch.cat([theta_3, zero_pad], dim=-1)

        bb_angles = torch.stack([theta_1, theta_2, theta_3], dim=-1)
        bin_limits = torch.linspace(-torch.pi, torch.pi, 20, device=coords.device)
        bin_indices = torch.bucketize(bb_angles, bin_limits)

        angles_feat = F.one_hot(bin_indices, len(bin_limits) + 1).float()
        angles_feat = einops.rearrange(angles_feat, "b n t d -> b n (t d)")
        angles_feat = angles_feat * residue_mask.unsqueeze(-1)
        return angles_feat

    def get_sidechain_angles(self, coords, aatype, mask):
        p = {
            "aatype": aatype,
            "all_atom_positions": coords,
            "all_atom_mask": mask,
        }
        p = data_transforms.atom37_to_torsion_angles(p)
        torsion_angles_sin_cos = p["torsion_angles_sin_cos"]
        torsion_angles_sin_cos = torsion_angles_sin_cos / (
            torch.linalg.norm(torsion_angles_sin_cos, dim=-1, keepdim=True) + 1e-10
        )

        torsion_angles_mask = p["torsion_angles_mask"]
        mask_bool = torsion_angles_mask.bool()
        torsion_angles_sin_cos = torch.where(
            mask_bool[..., None],
            torsion_angles_sin_cos,
            torch.zeros_like(torsion_angles_sin_cos),
        )
        torsion_angles_sin_cos = torsion_angles_sin_cos[..., -4:, :]
        torsion_angles_mask = torsion_angles_mask[..., -4:]
        mask_bool = mask_bool[..., -4:]
        angles = torch.atan2(
            torsion_angles_sin_cos[..., 0], torsion_angles_sin_cos[..., 1]
        )
        angles = angles.masked_fill(~mask_bool, 0.0)

        bin_limits = torch.linspace(-torch.pi, torch.pi, 20, device=coords.device)
        bin_indices = torch.bucketize(angles, bin_limits)
        angles_feat = F.one_hot(bin_indices, len(bin_limits) + 1).float()
        angles_feat = angles_feat * torsion_angles_mask[..., None]
        angles_feat = einops.rearrange(angles_feat, "b n s d -> b n (s d)")
        angles_feat = torch.cat([angles_feat, torsion_angles_mask], dim=-1)
        return angles_feat

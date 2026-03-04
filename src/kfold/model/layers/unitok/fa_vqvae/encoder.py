import einops
import torch
import torch.nn.functional as F
from torch import nn

from .data_transforms import atom37_to_torsion_angles
from .modules.attn_n_transition import MultiheadAttnAndTransition
from .modules.pair_update import PairReprUpdate
from .utils.angle_utils import bond_angles


class PairwiseResBlock(nn.Module):
    """Residual block for 2D pairwise features."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B * L, C, A, A]
            mask: [B * L, 1, A, A] float mask.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.gelu(out + residual)
        return out


class PairImageBackbone(nn.Module):
    """Lightweight ResNet stack that processes pairwise distance images."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim)
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
        return x.masked_fill_(~pair_mask.unsqueeze(-1), 0.0)


class AtomisticImageEncoder(nn.Module):
    """Atom encoder that leverages image-style pairwise representations with
    Evoformer mixing."""

    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_out: int,
        n_heads: int,
        n_layers: int,
        update_pair_repr_every_n: int = 2,
        distance_temperature: float = 10.0,
    ):
        super().__init__()
        self.single_rep_dim: int = d_single
        self.pair_rep_dim: int = d_pair
        self.d_out: int = d_out
        self.n_heads: int = n_heads
        self.n_layers: int = n_layers
        self.update_pair_repr_every_n: int = update_pair_repr_every_n
        self.distance_temperature: float = distance_temperature

        # Pair features passed to attention include learned pair reps plus scalar bias.
        self.pair_bias_dim: int = self.pair_rep_dim + 1

        feat_dims = [4, 1, 3 * 21, 4 * 21 + 4]
        self.feat_layers_atom1 = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.single_rep_dim))
            for d in feat_dims
        )
        self.feat_layers_atom2 = nn.Sequential(
            nn.LayerNorm(self.single_rep_dim * len(feat_dims)),
            nn.Linear(self.single_rep_dim * len(feat_dims), self.single_rep_dim),
        )

        feat_dims = [4 * 37, 37, 3 * 21, 4 * 21 + 4]
        self.feat_layers_seq1 = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.single_rep_dim))
            for d in feat_dims
        )
        self.feat_layers_seq2 = nn.Sequential(
            nn.LayerNorm(self.single_rep_dim * len(feat_dims)),
            nn.Linear(self.single_rep_dim * len(feat_dims), self.single_rep_dim),
        )

        self.fuse_layer = nn.Sequential(
            nn.Linear(self.single_rep_dim * 2, self.single_rep_dim),
            nn.LayerNorm(self.single_rep_dim),
            nn.Linear(self.single_rep_dim, self.single_rep_dim),
            nn.LayerNorm(self.single_rep_dim),
        )

        pair_input_dim = 5  # distance + 3-vector + mask
        self.pair_image_backbone = PairImageBackbone(
            input_dim=pair_input_dim, hidden_dim=self.pair_rep_dim, depth=self.n_layers
        )

        self.atom_transformer_layers = nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.single_rep_dim,
                    dim_pair=self.pair_bias_dim,
                    nheads=self.n_heads,
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_qkln=True,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.atom_pair_update_layers = nn.ModuleList(
            [
                (
                    PairReprUpdate(
                        self.single_rep_dim, self.pair_rep_dim, use_tri_mult=True
                    )
                    if i % self.update_pair_repr_every_n == 0
                    else None
                )
                for i in range(self.n_layers - 1)
            ]
        )

        self.pooling = nn.Sequential(
            nn.LayerNorm(self.single_rep_dim * 37),
            nn.Linear(self.single_rep_dim * 37, self.single_rep_dim),
            nn.LayerNorm(self.single_rep_dim),
            nn.Linear(self.single_rep_dim, self.single_rep_dim),
        )
        self.pre_vq_proj = nn.Linear(self.single_rep_dim, self.d_out)

    def _prepare_inputs(self, coords: torch.Tensor):
        mask = torch.all(
            torch.isfinite(coords) & (coords < 1e6),
            dim=-1,
        )
        residue_mask = mask.sum(-1).bool()
        coords = coords.masked_fill(~mask.unsqueeze(-1), 0.0)
        return coords, mask, residue_mask

    def _atom_level_features(
        self,
        coords_rel: torch.Tensor,
        mask: torch.Tensor,
        bb_angles: torch.Tensor,
        sc_angles: torch.Tensor,
    ):
        dtype = coords_rel.dtype
        bond_lengths = torch.norm(coords_rel, dim=-1, keepdim=True)
        bond_lengths = bond_lengths.masked_fill_(~mask.unsqueeze(-1), 0.0)
        A = coords_rel.shape[2]
        bb_angles_atom = bb_angles.unsqueeze(-2).expand(-1, -1, A, -1)
        sc_angles_atom = sc_angles.unsqueeze(-2).expand(-1, -1, A, -1)
        rel_coords_feat = torch.cat([coords_rel, mask[..., None].to(dtype)], dim=-1)

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
        rel_coords_flat = einops.rearrange(rel_coords_feat, "b n a t -> b n (a t)")
        bond_lengths_flat = einops.rearrange(bond_lengths, "b n a t -> b n (a t)")
        feat_list = [
            self.feat_layers_seq1[i](x)
            for i, x in enumerate(
                [rel_coords_flat, bond_lengths_flat, bb_angles, sc_angles]
            )
        ]
        x_seq = torch.cat(feat_list, dim=-1)
        x_seq = self.feat_layers_seq2(x_seq)
        return x_seq

    def _pool_and_project(self, x: torch.Tensor, residue_mask: torch.Tensor):
        z = einops.rearrange(x, "b n a t -> b n (a t)")
        z = self.pre_vq_proj(self.pooling(z))
        return z.masked_fill_(~residue_mask.unsqueeze(-1), 0.0)

    def get_backbone_angles(self, coords):
        b, n = coords.shape[0], coords.shape[1]

        idx = torch.arange(n, device=coords.device).unsqueeze(0) + 1

        N = coords[:, :, 0, :]
        CA = coords[:, :, 1, :]
        C = coords[:, :, 2, :]
        theta_1 = bond_angles(N, CA, C)
        theta_2 = bond_angles(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])
        theta_3 = bond_angles(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])

        good_pair = (idx[:, 1:] - idx[:, :-1]) == 1
        theta_2 = theta_2.masked_fill_(~good_pair, 0.0)
        theta_3 = theta_3.masked_fill_(~good_pair, 0.0)

        zero_pad = torch.zeros((b, 1), device=coords.device)
        theta_2 = torch.cat([theta_2, zero_pad], dim=-1)
        theta_3 = torch.cat([theta_3, zero_pad], dim=-1)

        bb_angles = torch.stack([theta_1, theta_2, theta_3], dim=-1)
        bin_limits = torch.linspace(-torch.pi, torch.pi, 20, device=coords.device)
        bin_indices = torch.bucketize(bb_angles, bin_limits)

        angles_feat = F.one_hot(bin_indices, len(bin_limits) + 1).float()
        angles_feat = einops.rearrange(angles_feat, "b n t d -> b n (t d)")
        return angles_feat

    def get_sidechain_angles(self, coords, residue_type, mask):
        p = {
            "aatype": residue_type,
            "all_atom_positions": coords,
            "all_atom_mask": mask,
        }
        p = atom37_to_torsion_angles(p)
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

    def encode(self, coords: torch.Tensor, residue_type: torch.Tensor):
        # Disable autocast for the entire encoding process
        # to maintain numerical stability in geometric computations.
        B, L, A, _ = coords.shape
        with torch.autocast(device_type=coords.device.type, enabled=False):
            coords, mask, residue_mask = self._prepare_inputs(coords)
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(3)

            coords_rel = coords - coords[:, :, 1:2, :]
            coords_rel.masked_fill_(~mask.unsqueeze(-1), 0.0)

            bb_angles = self.get_backbone_angles(coords)
            sc_angles = self.get_sidechain_angles(coords, residue_type, mask)
            x_atom, bond_lengths, rel_coords_feat = self._atom_level_features(
                coords_rel, mask, bb_angles, sc_angles
            )
            x_seq = self._sequence_level_features(
                rel_coords_feat, bond_lengths, bb_angles, sc_angles, residue_mask
            )

            x = torch.cat([x_atom, x_seq.unsqueeze(-2).expand(B, L, A, -1)], dim=-1)
            x = self.fuse_layer(x)

            rel_delta = coords_rel.unsqueeze(3) - coords_rel.unsqueeze(2)
            pair_dist = torch.norm(rel_delta, dim=-1, keepdim=True)

            pair_inputs = torch.cat(
                [pair_dist, rel_delta, pair_mask.unsqueeze(-1).float()], dim=-1
            )
            pair_rep = self.pair_image_backbone.init_features(pair_inputs, pair_mask)

            pair_bias = torch.exp(-pair_dist / self.distance_temperature)

        for i in range(self.n_layers):
            pair_rep = self.pair_image_backbone.step(pair_rep, pair_mask, i)
            pair_features = torch.cat([pair_rep, pair_bias], dim=-1)
            x = self.atom_transformer_layers[i](x, pair_features, mask)
            if i < self.n_layers - 1:
                if self.atom_pair_update_layers[i] is not None:
                    pair_rep = self.atom_pair_update_layers[i](x, pair_rep, mask)

        z = self.pre_vq_proj(self.pooling(x.flatten(-2)))
        z.masked_fill_(~residue_mask.unsqueeze(-1), 0.0)
        return z

    def forward(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

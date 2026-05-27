import contextlib

import numpy as np
import torch

import kfold.constants as C
from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.struct_enc import (
    BackboneTokenizer,
    FullAtomTokenizer,
    ProteinNetEncoder,
)
from kfold.utils.registry import STRUCTURE_ENCODER, BaseConfig

# AF2 residue types
restypes = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]  # fmt: skip
restype_order = {restype: i for i, restype in enumerate(restypes)}


@STRUCTURE_ENCODER.register()
class StructureEncoder(torch.nn.Module):
    bb_tok: BackboneTokenizer
    fa_tok: FullAtomTokenizer
    encoder: ProteinNetEncoder

    class Config(BaseConfig):
        """Configuration for UniTok structure encoder.

        Attributes
        ----------
        path: str
            Path to pretrained weights.
        chain_type: str
            Type of sequence chain to encode. Must be one of "protein", "dna", or "rna".
        d_model: int
            Dimension of token embeddings and transformer hidden states.
        n_heads: int
            Number of attention heads in the transformer.
        n_layers: int
            Number of transformer layers.
        """

        path: str | None = None
        fa_tok_path: str | None = None
        bb_tok_path: str | None = None
        encoder_path: str | None = None
        chain_type: str = "protein"
        d_model: int = 2560
        n_layers: int = 33
        n_heads: int = 40

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg: StructureEncoder.Config = cfg
        self.chain_type = cfg.chain_type

        # Create model components
        # TODO: replace to hf hub link.
        if cfg.path is not None:
            self.bb_tok = BackboneTokenizer().to(torch.bfloat16)
            self.fa_tok = FullAtomTokenizer().to(torch.bfloat16)
            self.encoder = ProteinNetEncoder().to(torch.bfloat16)
            state_dict = torch.load(cfg.path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        else:
            if cfg.fa_tok_path is None:
                raise ValueError("fa_tok_path must be provided if path is not provided.")
            if cfg.bb_tok_path is None:
                raise ValueError("bb_tok_path must be provided if path is not provided.")
            if cfg.encoder_path is None:
                raise ValueError("encoder_path must be provided if path is not provided.")
            self.bb_tok = BackboneTokenizer.from_pretrained(cfg.bb_tok_path)
            self.fa_tok = FullAtomTokenizer.from_pretrained(cfg.fa_tok_path)
            self.encoder = ProteinNetEncoder.from_pretrained(cfg.encoder_path)
            # Set to bfloat16
            self.bb_tok = self.bb_tok.to(torch.bfloat16)
            self.fa_tok = self.fa_tok.to(torch.bfloat16)
            self.encoder = self.encoder.to(torch.bfloat16)

        # Set to eval mode
        self.eval()

        # Freeze parameters since we are only doing inference.
        for param in self.parameters():
            param.requires_grad = False

        # Backbone token offset
        self.offset = 4  # number of special tokens

        seq_to_restype = torch.full((64,), -1, dtype=torch.long)
        for aa_i, aa in enumerate(restypes):
            seq_i = C.sequence.encode_protein_amino_acid(aa)
            seq_to_restype[seq_i] = aa_i
        self.register_buffer("seq_to_restype", seq_to_restype, persistent=False)

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def tokenize(
        self,
        sequence: str,
        atom37_coords: np.ndarray | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Tokenize apo structure with the structure encoder's tokenizer.

        Parameters
        ----------
        seq: str
            Amino acid sequence of the protein.
        atom37_coords: torch.Tensor
            Full-atom coordinates of shape (L, 37, 3).

        Returns
        -------
        token_ids: dict[str, torch.Tensor]
            - "seq_token_id": Tensor of shape (B, L) containing sequence token IDs.
            - "bb_struct_token_id": Tensor of shape (B, L) containing backbone
                structure token IDs.
            - "fa_struct_token_id": Tensor of shape (B, L) containing full-atom
                structure token IDs.
        """
        device = self.device
        if isinstance(atom37_coords, np.ndarray):
            atom37_coords = torch.from_numpy(atom37_coords)
        atom37_coords = atom37_coords.to(device)

        length = len(sequence)
        if atom37_coords.shape != (length, 37, 3):
            raise ValueError(
                f"Expected atom37_coords to have shape ({length}, 37, 3), "
                f"but got {atom37_coords.shape}"
            )

        seq_tok_id = torch.tensor(
            C.sequence.encode_protein_sequence(sequence), dtype=torch.long, device=device
        )
        aatypes = self.seq_to_restype[seq_tok_id]
        bb_tok_id = self.bb_tok.tokenize(atom37_coords[..., :3, :])
        fa_tok_id = self.fa_tok.tokenize(aatypes, atom37_coords)
        return {
            "seq_token_id": seq_tok_id,
            "bb_token_id": bb_tok_id,
            "fa_token_id": fa_tok_id,
        }

    def tokenize_batch(
        self,
        batch: list[tuple[str, np.ndarray | torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Tokenize apo structure with the structure encoder's tokenizer.

        Parameters
        ----------
        batch: list[tuple[str, torch.Tensor]]
            A batch of (sequence, atom37_coords) pairs, where:
            - sequence: str, amino acid sequence of the protein.
            - atom37_coords: torch.Tensor, full-atom coordinates of shape (L, 37, 3).

        Returns
        -------
        token_ids: dict[str, torch.Tensor]
            - "seq_token_id": Tensor of shape (B, L) containing sequence token IDs.
            - "bb_struct_token_id": Tensor of shape (B, L) containing backbone
                structure token IDs.
            - "fa_struct_token_id": Tensor of shape (B, L) containing full-atom
                structure token IDs.
        """
        # Convert all coordinates to tensors and move to the correct device
        device = self.device
        to_tensor = lambda x: (  # noqa
            torch.as_tensor(x, device=device) if isinstance(x, np.ndarray) else x
        )
        batch: list[tuple[str, torch.Tensor]] = [
            (seq, to_tensor(coords)) for seq, coords in batch
        ]

        # Validate inputs
        if len(batch) == 0:
            raise ValueError("Batch cannot be empty.")

        for sequence, atom37_coords in batch:
            length = len(sequence)
            if atom37_coords.shape != (length, 37, 3):
                raise ValueError(
                    f"Expected atom37_coords to have shape ({length}, 37, 3), "
                    f"but got {atom37_coords.shape}"
                )

        # Stack inputs into tensors
        B = len(batch)
        L = max(len(sequence) for sequence, _ in batch)

        pad_idx = C.sequence.PAD_TOKEN_INDEX
        seq_tok_ids = torch.full((B, L), pad_idx, dtype=torch.long)
        coords = torch.full((B, L, 37, 3), float("nan"), dtype=torch.float)
        mask = torch.zeros((B, L), dtype=torch.bool)
        for i, (sequence, atom37_coords) in enumerate(batch):
            length = len(sequence)
            seq_tok_ids[i, :length] = torch.tensor(
                C.sequence.encode_protein_sequence(sequence), dtype=torch.long
            )
            coords[i, :length] = atom37_coords
            mask[i, :length] = True

        seq_tok_ids, coords, mask = (
            seq_tok_ids.to(device),
            coords.to(device),
            mask.to(device),
        )

        aatypes = self.seq_to_restype[seq_tok_ids]

        # Tokenize backbone and full-atom structures
        bb_tok_ids = self.bb_tok.tokenize_batch(coords[..., :3, :])
        fa_tok_ids = self.fa_tok.tokenize_batch(aatypes, coords)

        seq_tok_ids.masked_fill_(~mask, pad_idx)
        bb_tok_ids.masked_fill_(~mask, -1)  # set to -1 for invalid tokens
        fa_tok_ids.masked_fill_(~mask, -1)  # set to -1 for invalid tokens
        return {
            "seq_token_id": seq_tok_ids,
            "bb_token_id": bb_tok_ids,
            "fa_token_id": fa_tok_ids,
        }

    def forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations,
            where Ntoken is the number of tokens and D is the model dimension.
        """
        device_type = f_input.device.type
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device_type == "cuda"
            else contextlib.nullcontext(),
        ):
            return self._forward(f_input)

    @torch.compiler.disable
    def get_seq_mask(self, f_input: FoldingInput) -> torch.Tensor:
        """Prepare output mask"""
        if self.chain_type == "protein":
            return f_input.sequence.pad_mask & f_input.sequence.is_protein
        else:
            raise ValueError(f"Unsupported chain type: {self.chain_type}")

    @torch.compiler.disable
    def get_token_mask(self, f_input: FoldingInput) -> torch.Tensor:
        """Prepare output mask"""
        if self.chain_type == "protein":
            return f_input.token.pad_mask & f_input.token.is_protein
        else:
            raise ValueError(f"Unsupported chain type: {self.chain_type}")

    def _forward(self, f_input: FoldingInput) -> torch.Tensor:
        """Forward pass of sequence representation module.

        Parameters
        ----------
        f_input: FoldingInput
            The input features

        Returns
        -------
        x_token: torch.Tensor
            Tensor of shape (B, Ntoken, D) containing sequence representations.
        """
        # NOTE: padding tokens have bb_token_ids of -1, which will be masked out
        # in the attention computation.
        seq_token_ids = f_input.sequence.seq_token_id
        bb_token_ids = f_input.sequence.bb_struct_token_id
        fa_token_ids = f_input.sequence.fa_struct_token_id

        # HACK: (Seonghwan) Since we use the shared sequence vocab for both sequence
        # and structure encoder, structure encoder does not have vocab ids for
        # dna and rna tokens. We set those to 0 to prevent out-of-vocab errors.
        seq_mask = self.get_seq_mask(f_input)
        seq_token_ids = seq_token_ids.masked_fill(~seq_mask, 0)

        seq_id = f_input.sequence.asym_id
        pos_id = f_input.sequence.pos_id

        # Mask out unallowed tokens
        allow_mask = bb_token_ids != -1  # we set bb_token_id to -1 for invalid tokens.
        seq_id = seq_id.masked_fill(~allow_mask, -1)  # entity id >= 1 for valid tokens

        x = self.encoder(
            seq_token_ids + self.offset,
            bb_token_ids + self.offset,
            fa_token_ids + self.offset,
            seq_id=seq_id,
            pos_id=pos_id,
        )
        x = x * allow_mask[..., None]  # mask out invalid tokens

        # sequence -> token index mapping
        batch_index = torch.arange(x.shape[0], device=x.device)[:, None]
        seq_token_index = f_input.token.seq_token_index
        x = x[batch_index, seq_token_index]  # [B, Ntoken, D]

        # mask out non-protein tokens
        token_mask = self.get_token_mask(f_input)
        x.masked_fill_(~token_mask[..., None], 0.0)
        return x

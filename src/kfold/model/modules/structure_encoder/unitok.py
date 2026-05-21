import torch

from kfold.data.types.model_input import FoldingInput
from kfold.model.layers.unitok import BackboneTokenizer, FullAtomTokenizer, UniTokBackbone
from kfold.utils.registry import STRUCTURE_ENCODER

from .base import BaseStructureEncoder


@STRUCTURE_ENCODER.register()
class UniTok(BaseStructureEncoder):
    class Config(BaseStructureEncoder.Config):
        """Configuration for UniTok structure encoder.

        Attributes
        ----------
        path: str
            Path to pretrained weights.
        d_model: int
            Dimension of token embeddings and transformer hidden states.
        n_heads: int
            Number of attention heads in the transformer.
        n_layers: int
            Number of transformer layers.

        """

        path: str  # Path to pretrained weights.
        d_model: int = 1536
        n_heads: int = 24
        n_layers: int = 30

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg: UniTok.Config = cfg

        # Create model components
        # NOTE: this is hard-coded
        self.bb_tok: BackboneTokenizer = BackboneTokenizer()
        self.fa_tok: FullAtomTokenizer = FullAtomTokenizer()
        self.backbone: UniTokBackbone = UniTokBackbone(
            embed_dim=cfg.d_model, encoder_depth=cfg.n_layers, encoder_heads=cfg.n_heads
        )
        state_dict = torch.load(cfg.path, map_location="cpu")
        self.load_state_dict(state_dict)

        # Set to eval mode
        self.eval()

        # Convert backbone to bfloat16
        self.backbone = self.backbone.to(torch.bfloat16)

        # Freeze parameters since we are only doing inference.
        for param in self.parameters():
            param.requires_grad = False

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

    def tokenize(
        self, aatypes: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize apo structure with the structure encoder's tokenizer.

        Parameters
        ----------
        aatypes: torch.Tensor
            Amino acid type indices of shape (L,).
            These should be indices corresponding to the ESM sequence vocabulary.
        coords: torch.Tensor
            Full-atom coordinates of shape (L, 37, 3), where L is the number of residues
            and 37 is the number of atoms per residue.

        Returns
        -------
        bb_struct_id: torch.Tensor
            Backbone structure token IDs for each residue, of shape (L,).
        fa_struct_id: torch.Tensor
            Full-atom structure token IDs for each residue, of shape (L,).
        """
        bb_struct_id = self.bb_tok.tokenize(coords)
        fa_struct_id = self.fa_tok.tokenize(aatypes, coords)
        return bb_struct_id, fa_struct_id

    def tokenize_batch(
        self, aatypes: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize apo structure with the structure encoder's tokenizer.

        Parameters
        ----------
        aatypes: torch.Tensor
            Amino acid type indices of shape (B, L,).
            These should be indices corresponding to the ESM sequence vocabulary.
        coords: torch.Tensor
            Full-atom coordinates of shape (B, L, 37, 3), where B is the batch size,
            L is the number of residues and 37 is the number of atoms per residue.

        Returns
        -------
        bb_struct_id: torch.Tensor
            Backbone structure token IDs for each residue, of shape (B, L,).
        fa_struct_id: torch.Tensor
            Full-atom structure token IDs for each residue, of shape (B, L,).
        """
        bb_struct_id = self.bb_tok.tokenize_batch(coords)
        fa_struct_id = self.fa_tok.tokenize_batch(aatypes, coords)
        return bb_struct_id, fa_struct_id

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
        with (
            torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16),
            torch.no_grad(),
        ):
            return self.forward_no_attn(f_input)

    def forward_no_attn(self, f_input: FoldingInput) -> torch.Tensor:
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
        seq_token_ids = seq_token_ids.masked_fill(~f_input.sequence.is_protein, 0)

        seq_id = f_input.sequence.entity_id
        pos_id = f_input.sequence.pos_id

        # Mask out unallowed tokens
        allow_mask = bb_token_ids != -1  # we set bb_token_id to -1 for invalid tokens.
        seq_id = seq_id.masked_fill(~allow_mask, -1)  # entity id >= 1 for valid tokens

        # HACK: we skip masking out invalid tokens to 0 since UniTok backbone
        # add 4 special tokens before nn.Embedding.
        pass

        # Backbone forward pass
        x = self.backbone(
            seq_token_ids,
            bb_token_ids,
            fa_token_ids,
            seq_id=seq_id,
            pos_id=pos_id,
        )

        # Mask out invalid tokens in the output.
        x[~allow_mask] = 0.0  # mask out invalid tokens

        # sequence -> token index mapping
        batch_index = torch.arange(x.shape[0], device=x.device)[:, None]
        seq_token_index = f_input.token.seq_token_index
        x = x[batch_index, seq_token_index]  # [B, Ntoken, D]

        # mask out invalid tokens
        pad_mask = f_input.token.pad_mask

        # mask out non-protein tokens
        token_mask = pad_mask & f_input.token.is_protein
        x.masked_fill_(~token_mask[..., None], 0.0)
        return x

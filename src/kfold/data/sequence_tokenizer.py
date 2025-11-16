"""Tokenizer for sequence representation training.
This class was initially implemented by Seonghwan Seo (code manager).
If you have any questions about implementation, please contact to me.
"""

from functools import cached_property

import kfold.constants as C

# === Special token utilities === #
BOS_TOKEN = "<"
EOS_TOKEN = ">"
MASK_TOKEN = "_"

# NOTE(sshwan): If we decide to consider multi-chain sequence, you may want to use this
SEP_TOKEN = "-"  # chain-break


class SequenceTokenizer:
    """Universal biomolecular sequence tokenizer supporting multiple bio-modalities.

    Attributes:
        <chain>_tokens (list[str]) : one-letter codes of residues
        <chain>_vocab (dict[str, int]) : mapping from one-letter code to token ID
        <chain>_id_to_token (dict[int, str]) : mapping from token ID to one-letter code
    # <chain>: protein, rna, dna

    Methods:
        encode(sequence: str, chain_type: C.chain.ChainType, add_special_tokens: bool)
            -> list[int]:
            Encode a chain sequence into a list of token IDs.

        decode(
            token_ids: list[int],
            skip_special_tokens: bool,
            sanity_check: bool,
            chain_type: C.chain.ChainType | None,
        ) -> tuple[C.chain.ChainType, str]:
            Decode a list of token IDs into a single chain sequence.
    """

    def __init__(self):
        # === Define regular tokens === #

        # Protein tokens
        self.protein_tokens: list[str] = list(C.residue.PROTEIN_AMINO_ACIDS)
        self.protein_vocab: dict[str, int] = {
            aa: C.residue.protein_one_letter_to_residue_name[aa].value
            for aa in self.protein_tokens
        }  # [0, 21]
        self.protein_id_to_token: dict[int, str] = {
            v: k for k, v in self.protein_vocab.items()
        }
        self.protein_unk_token_id: int = self.protein_vocab["X"]
        self.num_protein_tokens: int = len(self.protein_vocab)

        # RNA tokens
        self.rna_tokens: list[str] = list(C.residue.RNA_BASES)
        self.rna_bases: dict[str, int] = {
            base: C.residue.rna_one_letter_to_residue_name[base].value
            for base in self.rna_tokens
        }  # [22, 26]
        self.rna_id_to_token: dict[int, str] = {v: k for k, v in self.rna_bases.items()}
        self.rna_unk_token_id: int = self.rna_bases["N"]
        self.num_rna_tokens: int = len(self.rna_bases)

        # DNA tokens
        self.dna_tokens: list[str] = list(C.residue.DNA_BASES)
        self.dna_bases: dict[str, int] = {
            base: C.residue.dna_one_letter_to_residue_name[base].value
            for base in self.dna_tokens
        }  # [27, 31]
        self.dna_id_to_token: dict[int, str] = {v: k for k, v in self.dna_bases.items()}
        self.dna_unk_token_id: int = self.dna_bases["N"]
        self.num_dna_tokens: int = len(self.dna_bases)

        self.num_regular_tokens: int = (
            self.num_protein_tokens + self.num_rna_tokens + self.num_dna_tokens
        )

        # === Sanity checks === #
        # check that there is no overlap in token IDs
        all_token_ids = (
            set(self.protein_vocab.values())
            | set(self.rna_bases.values())
            | set(self.dna_bases.values())
        )
        assert len(all_token_ids) == self.num_regular_tokens, "Token ID overlap detected"

        # check that the token IDs are contiguous
        assert all_token_ids == set(range(self.num_regular_tokens)), (
            "Token IDs are not contiguous"
        )

        # === Define special tokens === #
        # special tokens
        self.special_tokens: list[str] = [BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, SEP_TOKEN]
        self.special_vocab: dict[str, int] = {
            BOS_TOKEN: self.num_regular_tokens,
            EOS_TOKEN: self.num_regular_tokens + 1,
            MASK_TOKEN: self.num_regular_tokens + 2,
            SEP_TOKEN: self.num_regular_tokens + 3,
        }
        self.special_id_to_token: dict[int, str] = {
            v: k for k, v in self.special_vocab.items()
        }
        self.num_special_tokens: int = len(self.special_tokens)

        self.num_tokens: int = self.num_regular_tokens + self.num_special_tokens

        self.id_to_token: dict[int, str] = {
            **self.protein_id_to_token,
            **self.rna_id_to_token,
            **self.dna_id_to_token,
            **self.special_id_to_token,
        }

    # === Properties === #
    @cached_property
    def bos_token(self) -> str:
        """Get the beginning-of-sequence token ID."""
        return BOS_TOKEN

    @cached_property
    def eos_token(self) -> str:
        """Get the end-of-sequence token ID."""
        return EOS_TOKEN

    @cached_property
    def mask_token(self) -> str:
        """Get the mask token ID."""
        return MASK_TOKEN

    @cached_property
    def sep_token(self) -> str:
        """Get the separator token ID."""
        return SEP_TOKEN

    @cached_property
    def bos_token_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self.special_vocab[self.bos_token]

    @cached_property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self.special_vocab[self.eos_token]

    @cached_property
    def mask_token_id(self) -> int:
        """Get the mask token ID."""
        return self.special_vocab[self.mask_token]

    @cached_property
    def sep_token_id(self) -> int:
        """Get the separator token ID."""
        return self.special_vocab[self.sep_token]

    def get_id_to_token(self, chain_type: C.chain.ChainType) -> dict[int, str]:
        """Get the mapping from token ID to token for a given chain type."""
        match chain_type:
            case C.chain.ChainType.PROTEIN:
                return self.protein_id_to_token
            case C.chain.ChainType.RNA:
                return self.rna_id_to_token
            case C.chain.ChainType.DNA:
                return self.dna_id_to_token
            case _:
                raise ValueError(f"Unsupported chain type: {chain_type}")

    # === Tokenizer methods === #
    def encode(
        self,
        sequence: str,
        chain_type: C.chain.ChainType,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode a chain sequence into a list of token IDs."""
        token_ids: list[int] = []
        match chain_type:
            case C.chain.ChainType.PROTEIN:
                vocab = self.protein_vocab
                unk_token_id = self.protein_unk_token_id
            case C.chain.ChainType.RNA:
                vocab = self.rna_bases
                unk_token_id = self.rna_unk_token_id
            case C.chain.ChainType.DNA:
                vocab = self.dna_bases
                unk_token_id = self.dna_unk_token_id
            case _:
                raise ValueError(f"Unsupported chain type: {chain_type}")

        token_ids = [vocab.get(c, unk_token_id) for c in sequence]

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        sanity_check: bool = True,
        chain_type: C.chain.ChainType | None = None,
    ) -> tuple[C.chain.ChainType, str]:
        """Decode a list of token IDs into a single chain sequence."""
        # NOTE(sshwan): this decode method is implemented for debugging...

        def _decode(token_ids: list[int]) -> str:
            """Decode a list of token IDs into a chain sequence."""
            return " ".join([self.id_to_token[v] for v in token_ids])

        # === Sanity check === #

        # Ensure that the token IDs do not contain separator token ID
        forbidden_token_ids = {self.sep_token_id}
        if any(v in forbidden_token_ids for v in token_ids):
            raise ValueError(
                "Token IDs contain separator token ID. Use decode_chains for multi-chain"
                " decoding."
            )

        # If chain_type is provided, ensure that all token IDs belong to that chain type
        if chain_type is not None:
            id_to_token: dict[int, str] = self.get_id_to_token(chain_type)
            id_to_token = id_to_token | self.special_id_to_token
            if not set(token_ids).issubset(set(id_to_token.keys())):
                raise ValueError(
                    f"Token IDs do not match the specified chain type {chain_type}:"
                    f" {set(token_ids) - set(id_to_token.keys())}.\n"
                    f" Decoding sequence: {_decode(token_ids)}"
                )
        else:
            # Determine chain type from token IDs
            # First, determine the chain type from the token IDs
            for v in token_ids:
                if v in self.protein_id_to_token:
                    chain_type = C.chain.ChainType.PROTEIN
                    break
                elif v in self.rna_id_to_token:
                    chain_type = C.chain.ChainType.RNA
                    break
                elif v in self.dna_id_to_token:
                    chain_type = C.chain.ChainType.DNA
                    break
            else:
                raise ValueError(
                    "Input sequence contains no non-special tokens."
                    f" Decoding sequence: {_decode(token_ids)}"
                )
            if sanity_check:
                # Ensure all token IDs belong to a single chain type
                id_to_token = self.get_id_to_token(chain_type)
                id_to_token = id_to_token | self.special_id_to_token
                if not set(token_ids).issubset(set(id_to_token.keys())):
                    raise ValueError(
                        f"Token IDs do not match the specified chain type {chain_type}:"
                        f" {set(token_ids) - set(id_to_token.keys())}.\n"
                        f" Decoding sequence: {_decode(token_ids)}"
                    )

        # === Decode === #
        if skip_special_tokens:
            # Also skip beginning-of-sequence and end-of-sequence tokens
            # Keep mask tokens
            skip_token_ids = {self.bos_token_id, self.eos_token_id}
            token_ids = [v for v in token_ids if v not in skip_token_ids]

        sequence = _decode(token_ids)
        return chain_type, sequence

    def encode_complex(
        self,
        sequences: list[str],
        chain_types: list[C.chain.ChainType],
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode a chain sequence into a list of token IDs."""
        if len(sequences) != len(chain_types):
            raise ValueError(
                "Number of sequences and chain_types must be the same."
                f" ({len(sequences)} != {len(chain_types)})"
            )

        full_token_ids: list[int] = []

        for i, (sequence, chain_type) in enumerate(
            zip(sequences, chain_types, strict=True)
        ):
            full_token_ids.extend(
                self.encode(sequence, chain_type, add_special_tokens=False)
            )
            if add_special_tokens and i < len(sequences) - 1:
                full_token_ids.append(self.sep_token_id)

        if add_special_tokens:
            full_token_ids = [self.bos_token_id] + full_token_ids + [self.eos_token_id]

        return full_token_ids

    def decode_complex(
        self, token_ids: list[int]
    ) -> tuple[list[C.chain.ChainType], list[str]]:
        """Decode a list of token IDs into a chain sequence."""

        # NOTE(sshwan): This may not be used in project. Implement it if needed.
        raise NotImplementedError(
            "decode_complex is not implemented yet. Use decode for single-chain decoding."
        )

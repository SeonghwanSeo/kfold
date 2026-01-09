from pathlib import Path


def load_fasta(path: str | Path) -> dict[str, str]:
    """Load sequences from a fasta file (supports multi-line sequences).

    Parameters
    ----------
    path : str | Path
        Path to the fasta file.

    Returns
    -------
    sequences : dict[str, str]
        key: sequence ID
        value: sequence string

    """
    sequences: dict[str, str] = {}

    current_seq_id = None
    current_seq_parts: list[str] = []

    with open(path) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines if any exist
            if not line:
                continue

            if line.startswith(">"):
                if current_seq_id is not None:
                    sequences[current_seq_id] = "".join(current_seq_parts)

                # Start a new sequence entry
                current_seq_id = line[1:]  # Remove '>'
                current_seq_parts = []
            else:
                # Append sequence lines to the current list buffer
                if current_seq_id is not None:
                    current_seq_parts.append(line)

        # Add the last sequence after the loop ends
        if current_seq_id is not None:
            sequences[current_seq_id] = "".join(current_seq_parts)

    return sequences


def save_fasta(
    sequences: list[tuple[str, str]],
    path: str | Path,
    width: int | None = None,
) -> None:
    """Save sequences to a fasta file.

    Parameters
    ----------
    sequences : list[tuple[str, str]]
        List of tuples containing sequence ID and sequence string.
    path : str | Path
        Path to save the fasta file.
    width : int | None, optional
        Maximum line length for sequences.
        If None, sequences are written in a single line.
    """
    with open(path, "w") as f:
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n")
            if width is None:
                f.write(f"{seq}\n")
            else:
                for i in range(0, len(seq), width):
                    f.write(f"{seq[i : i + width]}\n")

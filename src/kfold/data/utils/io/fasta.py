from pathlib import Path


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Load sequences from a fasta file (supports multi-line sequences).

    Parameters
    ----------
    path : str | Path
        Path to the fasta file.

    Returns
    -------
    sequences : list[tuple[str, str]]
        List of tuples (ID, sequence).

    """
    res: list[tuple[str, str]] = []
    seq_id: str | None = None
    seq_parts: list[str] = []
    with open(path) as f:
        for line in f:
            line: str = line.strip()
            # Skip empty lines if any exist
            if not line:
                continue

            if line.startswith(">"):
                if seq_id is not None:
                    sequence = "".join(seq_parts)
                    res.append((seq_id, sequence))
                # Start a new sequence entry
                seq_id = line[1:]  # Remove '>'
                seq_parts = []
            else:
                # Append sequence lines to the current list buffer
                if seq_id is not None:
                    seq_parts.append(line)

        # Add the last sequence after the loop ends
        if seq_id is not None:
            sequence = "".join(seq_parts)
            res.append((seq_id, sequence))
    return res


def write_fasta(
    sequences: list[tuple[str, str]] | dict[str, str],
    path: str | Path,
    width: int | None = None,
) -> None:
    """Save sequences to a fasta file.

    Parameters
    ----------
    sequences : list[tuple[str, str]] | dict[str, str]
        List of tuples (ID, sequence) or a dictionary mapping IDs to sequences.
    path : str | Path
        Path to save the fasta file.
    width : int | None, optional
        Maximum line length for sequences.
        If None, sequences are written in a single line.
    """
    if isinstance(sequences, dict):
        sequences = list(sequences.items())
    with open(path, "w") as f:
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n")
            if width is None:
                f.write(f"{seq}\n")
            else:
                for i in range(0, len(seq), width):
                    f.write(f"{seq[i : i + width]}\n")

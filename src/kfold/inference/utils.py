"""Sequence alignment for supplied apo and prior structures."""


def align_sequences(
    target_sequence: str,
    source_sequence: str,
) -> tuple[slice, slice]:
    """Return query/source slices for the ungapped overlap with most matches."""
    if target_sequence == source_sequence:
        return slice(0, len(target_sequence)), slice(0, len(source_sequence))
    target_length = len(target_sequence)
    source_length = len(source_sequence)
    best_mapping = (0, 0, 0, 0)
    best_score = (-1, -1)

    for offset in range(1 - target_length, source_length):
        target_start = max(0, -offset)
        source_start = max(0, offset)
        overlap = min(
            target_length - target_start,
            source_length - source_start,
        )
        num_matches = sum(
            target_sequence[target_start + i] == source_sequence[source_start + i]
            for i in range(overlap)
        )
        score = (num_matches, overlap)
        if score > best_score:
            best_score = score
            best_mapping = (
                target_start,
                target_start + overlap,
                source_start,
                source_start + overlap,
            )

    if best_score[0] <= 0:
        raise ValueError("No matching residues between the query and source structure.")
    target_start, target_end, source_start, source_end = best_mapping
    return slice(target_start, target_end), slice(source_start, source_end)

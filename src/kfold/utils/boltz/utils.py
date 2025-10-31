from functools import lru_cache


@lru_cache(maxsize=256)
def convert_atom_name_to_tuple(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Args:
        name (str): The atom name.

    Returns:
        tuple[int, int, int, int]: The converted atom name.
    """
    name = name.strip()
    name_int = [ord(c) - 32 for c in name]
    name_int = name_int + [0] * (4 - len(name_int))  # pad to 4 characters
    return tuple(name_int)  # pyright: ignore


@lru_cache(maxsize=256)
def get_atom_name(id: tuple[int, int, int, int]) -> str:
    """Convert an atom name from a standard format.

    Args:
        id (tuple[int, int, int, int]): The atom name in standard format.

    Returns:
        str: The converted atom name.
    """
    name = "".join([chr(i + 32) for i in id if i != 0])
    return name

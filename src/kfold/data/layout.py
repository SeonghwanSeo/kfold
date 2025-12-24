import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Generic, Self, TypeVar

import numpy as np
import torch

# common type alias
torch_float = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
torch_int = (torch.int32, torch.int64)

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


# === Base class for dataclass with tensor === #
class ArrayObj(Generic[ArrayT]):
    """Dataclass base class with array fields."""

    def __getitem__(self, idx: int | slice | ArrayT) -> Self:
        fields = {name: arr[idx] for name, arr in self.to_dict().items()}
        return self.from_dict(fields)

    def keys(self) -> list[str]:
        """Get the field names of the dataclass."""
        # Ensure that self is a dataclass
        assert hasattr(self, "__dataclass_fields__"), (
            f"{self.__class__.__name__} must be a dataclass to use TensorObj.keys()"
        )
        return list(self.__dataclass_fields__.keys())  # type: ignore

    def to_dict(self) -> dict[str, ArrayT]:
        """Convert the dataclass fields to a dictionary.
        Note: dataclasses.asdict() is not used to avoid deep copying.
        """
        field_names = self.keys()
        field_dict = {name: getattr(self, name) for name in field_names}
        return field_dict

    @classmethod
    def from_dict(cls, data: dict[str, ArrayT]) -> Self:
        return cls(**data)

    def copy(self, deepcopy: bool = False) -> Self:
        """Create a copy of the object."""
        if deepcopy:
            return copy.deepcopy(self)
        else:
            return self.from_dict(self.to_dict())

    def copy_with(self, deepcopy: bool = False, **kwargs: ArrayT) -> Self:
        """Create a copy of the object with optional field updates."""
        data = self.to_dict()
        assert kwargs.keys() <= data.keys(), (
            f"Invalid field names: {kwargs.keys() - data.keys()}"
        )
        if deepcopy:
            for k in kwargs:
                data.pop(k)
            data = copy.deepcopy(data)
        data.update(kwargs)
        return self.from_dict(data)

    # === Save / Load methods === #
    def get_state(self) -> dict[str, ArrayT | Any]:
        """Get the state dictionary of the object.
        We may want to convert datatypes here to reduce the size on disk.
        """
        # convert datatypes if necessary
        return self.to_dict()

    @classmethod
    def load_state(cls, state: dict[str, ArrayT]) -> Self:
        """Set the state of the object from the state dictionary.
        Restore datatypes if necessary.
        """
        # convert datatypes if necessary
        return cls.from_dict(state)


class TensorObj(ArrayObj[torch.Tensor]):
    @cached_property
    def device(self) -> torch.device:
        for tensor in self.to_dict().values():
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        raise ValueError("No tensor found in the dataclass.")

    def to(self, device: str | torch.device) -> Self:
        fields = {
            name: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in self.to_dict().items()
        }
        return self.from_dict(fields)

    def clone(self, deepcopy: bool = True) -> Self:
        return self.copy(deepcopy=deepcopy)


class PlainLayout(ArrayObj[ArrayT], ABC):
    """Base class for layout information.
    Layout dimensions: [D1, ..., DN]
    Feature dimensions: [D1, ..., DN, F1, ..., FM]
    """

    @property
    @abstractmethod
    def layout_shape(self) -> tuple[int, ...]:
        """The shape of the layout"""

    @property
    def ndim(self) -> int:
        """The number of dimensions of the layout."""
        return len(self.layout_shape)

    @property
    def length(self) -> int:
        """Get the length of the layout."""
        return self.layout_shape[0]

    def __len__(self) -> int:
        """The length of the layout."""
        return self.length

    def __getitem__(self, idx: int | slice | ArrayT) -> Self:
        """Get a subset of the layout."""
        # Ensure that idx is not integer
        if isinstance(idx, int):
            raise ValueError(
                f"Integer indexing is not supported for {self.__class__.__name__}. "
                f"Use slicing instead: obj[{idx}:{idx + 1}]"
            )
        return super().__getitem__(idx)

    def __repr__(self) -> str:
        """Enhanced repr with layout-specific info."""
        class_name = self.__class__.__name__

        # Layout meta info
        length = len(self)

        shape_desc = f"length={length}"

        # Fields
        fields = self.to_dict()
        field_strs = []
        for name, value in fields.items():
            if isinstance(value, torch.Tensor):
                dtype_str = str(value.dtype).replace("torch.", "")
                shape_str = "x".join(map(str, value.shape))
                field_strs.append(f"  {name}: [{dtype_str}, {shape_str}]")

        fields_repr = "\n".join(field_strs)
        return f"{class_name}({shape_desc})\n{fields_repr}"

    def pad(self, *pad_shape: int) -> Self:
        """Pad the layout to the total length."""
        self._check_pad_input(pad_shape)
        raise NotImplementedError

    def _check_pad_input(self, pad_shape: tuple[int, ...]):
        assert len(pad_shape) == self.ndim, (
            f"Pad shape must have the same number of dimensions as layout: "
            f"{len(pad_shape)} != {self.ndim}"
        )
        for i in range(self.ndim):
            assert pad_shape[i] >= self.layout_shape[i], (
                f"Pad shape must be greater than or equal to layout shape: "
                f"{pad_shape} < {self.layout_shape}"
            )

    @classmethod
    def concatenate(cls, data_list: Sequence[Self], dim: int = 0) -> Self:
        """Concatenate multiple data_list along the specified dimension."""
        assert len(data_list) > 0, "data_list must not be empty."
        ref_layout = data_list[0]

        field_dict: dict[str, list[ArrayT]] = {k: [] for k in ref_layout.keys()}
        for layout in data_list:
            for name, arr in layout.to_dict().items():
                field_dict[name].append(arr)

        if isinstance(ref_layout, TensorObj):
            concatenated_fields = {
                name: torch.cat(tensors, dim=dim) for name, tensors in field_dict.items()
            }
        else:  # numpy array
            concatenated_fields = {
                name: np.concatenate(tensors, axis=dim)
                for name, tensors in field_dict.items()
            }
        return cls.from_dict(concatenated_fields)


class TensorLayout(TensorObj, PlainLayout):
    """Layout can be either batched or non-batched.
    [D1, D2, ..., DN] for non-batched layout
    [B, D1, D2, ..., DN] for batched layout
    where B is the batch size.

    # NOTE: padding and indexing is only supported for non-batched layout.
    """

    @property
    @abstractmethod
    def layout_shape(self) -> tuple[int, ...]:
        """The shape of the layout: batched or non-batched.
        [D1, D2, ..., DN] for non-batched layout.
        [B, D1, D2, ..., DN] for batched layout.
        """

    @property
    @abstractmethod
    def ndim_unbatched(self) -> int:
        """[ClassVar] The number of dimensions of the layout."""

    @property
    def is_batched(self) -> bool:
        """Whether the layout is batched."""
        return self.ndim == self.ndim_unbatched + 1

    @property
    def batch_size(self) -> int:
        """Batch size of layout."""
        assert self.is_batched, "Layout is not batched."
        return self.layout_shape[0]

    @property
    def length(self) -> int:
        """Get the length of the layout."""
        return self.layout_shape[1] if self.is_batched else self.layout_shape[0]

    def __repr__(self) -> str:
        """Enhanced repr with layout-specific info."""
        class_name = self.__class__.__name__

        # Layout meta info
        length = len(self)
        device = self.device

        if self.is_batched:
            shape_desc = f"batch_size={self.batch_size}, length={length}, device={device}"
        else:
            shape_desc = f"length={length}, device={device}"

        # Fields
        fields = self.to_dict()
        field_strs = []
        for name, value in fields.items():
            if isinstance(value, torch.Tensor):
                dtype_str = str(value.dtype).replace("torch.", "")
                shape_str = "x".join(map(str, value.shape))
                field_strs.append(f"  {name}: [{dtype_str}, {shape_str}]")

        fields_repr = "\n".join(field_strs)
        return f"{class_name}({shape_desc})\n{fields_repr}"

    # === Methods for unbatched layout === #
    def __getitem__(self, idx: int | slice | torch.Tensor) -> Self:
        """Get a subset of the layout."""
        assert not self.is_batched, "Batched layout is not supported."
        return super().__getitem__(idx)

    def pad(self, *pad_shape: int) -> Self:
        assert not self.is_batched, "Padding batched layout is not supported."
        return super().pad(*pad_shape)

    # === Methods for batched layout === #
    @classmethod
    def from_list(cls, data_list: list[Self]) -> Self:
        """Create a Batched Layout from a list of data."""
        assert len(data_list) > 0, "data_list must not be empty."
        ref_data = data_list[0]
        ref_length = len(ref_data)
        ref_device = ref_data.device

        field_dict: dict[str, list[torch.Tensor]] = {k: [] for k in ref_data.keys()}
        for data in data_list:
            assert not data.is_batched, "All layouts in data_list must be non-batched."
            assert len(data) == ref_length, (
                "All layouts in data_list must have the same length."
            )
            assert data.device == ref_device, (
                "All layouts in data_list must be on the same device."
            )
            for name, tensor in data.to_dict().items():
                field_dict[name].append(tensor)

        batched_fields = {
            name: torch.stack(tensors, dim=0) for name, tensors in field_dict.items()
        }
        return cls.from_dict(batched_fields)

    def to_list(self, deepcopy: bool = False) -> list[Self]:
        """Unpack a Batched Layout into a list of data."""
        assert self.is_batched, "Layout is not batched."
        if deepcopy:
            data_dict = {
                name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in self.to_dict().items()
            }
        else:
            data_dict = self.to_dict()

        data_list: list[Self] = []
        batch_size = self.batch_size
        for i in range(batch_size):
            fields = {
                name: tensor[i] if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in data_dict.items()
            }
            data_list.append(self.from_dict(fields))
        return data_list

from numpy import ndarray
from anicrop.spatial import Region
from typing import Any, Union


class Image:
    def __init__(self, image: ndarray):

        if image.ndim not in (2, 3):
            raise ValueError("image array must be 2D or 3D")

        elif image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image dimensions must be greater than zero")

        elif image.ndim == 3 and image.shape[2] == 0:
            raise ValueError("image must have at least one channel")

        self._data = image
        self._channels = image.shape[2] if image.ndim > 2 else 1

    def __region_to_slice(self, region: Region) -> slice:

        return (
            slice(region.x.start, region.x.end),
            slice(region.y.start, region.y.end)
        )

    def __to_indexer(self, key: Any) -> Any:
        if isinstance(key, Region):
            return self.__region_to_slice(key)

        elif isinstance(key, tuple):
            if any([isinstance(arg, Region) for arg in key[1:]]):
                raise TypeError("Region argument is only valid at the first position")

            elif isinstance(key[0], Region):
                return self.__region_to_slice(key[0]) + key[1:]

        return key

    def __getitem__(self, key: Region | Any) -> ndarray:
        return self._data[self.__to_indexer(key)]

    def __setitem__(self, key: Region | Any, value: Any) -> None:
        self._data[self.__to_indexer(key)] = value

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def width(self) -> int:
        return self._data.shape[1]

    @property
    def height(self) -> int:
        return self._data.shape[0]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def channels(self) -> int:
        return self._channels

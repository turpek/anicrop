from numpy import ndarray
from anicrop.spatial import Region
from typing import Any


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

    def __getitem__(self, key: Any):
        if isinstance(key, Region):
            region = (
                slice(key.x.start, key.x.end),
                slice(key.y.start, key.y.end),
            )
            return self._data[region]
        if isinstance(key, tuple):
            if isinstance(key[0], Region):
                reg = key[0]
                region = (
                    slice(reg.x.start, reg.x.end),
                    slice(reg.y.start, reg.y.end),
                )
                if any(isinstance(arg, Region) for arg in key[1:]):
                    raise TypeError("Region must be the first and only spatial argument")
                return self._data[region + key[1:]]
        if any(isinstance(arg, Region) for arg in key):
            raise TypeError("Region must be the first and only spatial argument")
        return self._data[key]

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

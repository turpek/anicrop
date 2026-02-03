from numpy import ndarray


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

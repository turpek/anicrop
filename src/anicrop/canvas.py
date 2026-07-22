from anicrop.spatial import Region


class Canvas:
    def __init__(self, width: int, height: int):
        self._region = Region.from_size(width, height)

    @property
    def size(self) -> tuple[int, int]:
        return self._region.size

    @property
    def region(self) -> Region:
        return self._region

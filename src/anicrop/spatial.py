from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self


class Span:
    def __init__(self, start: int, end: int):

        if start < 0:
            raise ValueError(f'start cannot be less than 0 (start={start})')

        if start >= end:
            raise ValueError(f'start must be < end (start={start}, end={end})')
        self._start = start
        self._end = end

    def __repr__(self):
        return f"{self.__class__.__name__}(start={self.start}, end={self.end})"

    @property
    def length(self) -> int:
        """The length of the span, calculated as `end - start`."""
        return self._end - self._start

    @property
    def start(self) -> int:
        """The starting point of the span."""
        return self._start

    @property
    def end(self) -> int:
        """The ending point of the span."""
        return self._end

    def expand(self, margin: int, bounds: Span) -> Span:

        end = min(bounds.end, self.end + margin)
        start = max(bounds.start, self.start - margin)
        if margin < 0:
            raise ValueError(
                "Margin for expand() must be non-negative. To contract "
                "the span, use the shrink() method with a positive margin."
            )

        return Span(start, end)

    def shrink(self, margin: int) -> Span:
        """Shrinks the span inwards purely mathematically."""
        if margin < 0:
            raise ValueError(
                "Margin for shrink() must be non-negative, but got -1. "
                "To expand the span, use the expand() method with a positive margin."
            )

        return Span(self.start + margin, self.end - margin)

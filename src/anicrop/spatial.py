from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self


class SpanBase(ABC):
    """Abstract base class for span-like objects.

    Provides common structure (start, end) and derived functionality
    like length, representation, and equality checking.

    class is not meant to be instantiated directly. Its subclasses
    define specific invariants and behaviors.
    """

    def __init__(self, start: int, end: int):
        """Initializes the base attributes for a span-like object."""

        if start >= end:
            raise ValueError(f"Start '({start})' cannot be greater than end '({end})'.")
        self._start = start
        self._end = end

    def __repr__(self):
        """Returns the official string representation of the object."""
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

    @abstractmethod
    def expand(self, value: int) -> Self:
        """Returns a new, expanded instance of the span."""
        ...

    @abstractmethod
    def shrink(self, margin: int) -> Self:
        """Returns a new, shrunken instance of the span."""
        ...


class RelativeSpan(SpanBase):
    """Represents a relative difference or offset. Can contain any integer."""

    def expand(self, margin: int) -> RelativeSpan:
        """Expands the span outwards, clamping the start at 0."""

        if margin < 0:
            raise ValueError(
                "Margin for expand() must be non-negative. To contract "
                "the span, use the shrink() method with a positive margin."
            )

        return RelativeSpan(self.start - margin, self.end + margin)

    def shrink(self, margin: int) -> RelativeSpan:
        """Shrinks the span inwards purely mathematically."""
        if margin < 0:
            raise ValueError(
                "Margin for shrink() must be non-negative, but got -1. "
                "To expand the span, use the expand() method with a positive margin."
            )

        return RelativeSpan(self.start + margin, self.end - margin)

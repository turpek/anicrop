from __future__ import annotations
from dataclasses import dataclass
from operator import add, sub
from ovld import ovld
from typing import Callable


class SpanError(Exception):
    pass


class Span:
    """Represents a continuous numerical interval (span) in one dimension.

    The class operates on non-negative coordinates and ensures that `start < end`.
    It is an immutable object.

    Attributes:
        start (int): The starting point of the span.
        end (int): The ending point of the span.
        length (int): The length of the span, calculated as `end - start`.
    """

    @ovld
    def __init__(self, end: int, /):
        """Initializes a Span with start=0.

        Args:
            start: The starting point of the span. Must be non-negative.
            end: The ending point of the span. Must be greater than start.

        Raises:
            ValueError: If `start >= end`.
        """
        self._setup(0, end)

    @ovld
    def __init__(self, start: int, end: int, /):  # noqa: F811
        """Initializes a Span.

        Args:
            start: The starting point of the span. Must be non-negative.
            end: The ending point of the span. Must be greater than start.

        Raises:
            ValueError: If `start < 0` or if `start >= end`.
        """
        self._setup(start, end)

    def _setup(self, start: int, end: int) -> None:
        if start < 0:
            raise ValueError(f'start cannot be less than 0 (start={start})')

        if start >= end:
            raise ValueError(f'start must be < end (start={start}, end={end})')

        self._start = start
        self._end = end

    def __repr__(self):
        return f"{self.__class__.__name__}(start={self.start}, end={self.end})"

    def __eq__(self, span: Span) -> bool:
        if not isinstance(span, Span):
            return NotImplemented

        return self.start == span.start and self.end == span.end

    def __add__(self, offset: int | Span) -> Span:
        """Shifts the span to the right.

        Implements the `Span + int` operation.

        Args:
            offset: The integer value to add to the start and end points.

        Returns:
            A new Span object shifted to the right.
        """
        if isinstance(offset, Span):
            offset = offset.start
        return Span(self.start + offset, self.end + offset)

    def __sub__(self, offset: int | Span) -> Span:
        """Shifts the span to the left.

        Implements the `Span - int` operation. Ensures the new start point
        is not less than zero.

        Args:
            offset: The integer value to subtract from the start and end points.

        Returns:
            A new Span object shifted to the left.
        """
        if isinstance(offset, Span):
            offset = offset.start

        start = max(0, self.start - offset)
        return Span(start, self.end - offset)

    def __or__(self, span: Span) -> Span:
        """Computes the union of this span and another.

        Implements the `|` operator.

        Args:
            span: The other Span object to union with.

        Returns:
            A new Span representing the union, running from the minimum start
            to the maximum end of both spans.
        """

        start = min(self.start, span.start)
        end = max(self.end, span.end)
        return Span(start, end)

    def __and__(self, span: Span) -> Span:
        """Computes the intersection of this span and another.

        Implements the `&` operator.

        Args:
            span: The other Span object to intersect with.

        Returns:
            A new Span representing the overlapping area.

        Raises:
            SpanError: If there is no overlaps between the spans.
        """

        if not self.overlaps(span):
            raise SpanError("no overlap between spans.")

        overlap_start = max(self.start, span.start)
        overlap_end = min(self.end, span.end)
        return Span(overlap_start, overlap_end)

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

    def overlaps(self, other: Span) -> bool:
        return self.end > other.start and other.end > self.start

    def expand(self, both: int = None, *, before: int = 0, after: int = 0) -> Span:
        """Expands the span outward.

        Args:
            margin: The number of units to expand on each side.

        Returns:
            A new Span object expanded.

        Raises:
            ValueError: If the margin is negative.
        """

        if both:
            before = after = both

        if after < 0 or before < 0:
            raise ValueError(
                "Margin for expand() must be non-negative. To contract "
                "the span, use the shrink() method with a positive value."
            )

        start_expand = max(0, self.start - before)

        return Span(start_expand, self.end + after)

    def shrink(self, both: int = None, *, before: int = 0, after: int = 0) -> Span:

        """Shrinks the span inward.

        Args:
            margin: The non-negative number of units to shrink on each side.

        Returns:
            A new, smaller Span object.

        Raises:
            ValueError: If the margin is negative.
        """
        if both:
            before = after = both

        if before < 0 or after < 0:
            raise ValueError(
                "Margin for shrink() must be non-negative. To contract "
                "To expand the span, use the expand() method with a positive margin."
            )

        return Span(self.start + before, self.end - after)

    # def offset(self, other: Span) -> Span:
    #    return Span(span.start )

    def offset_to(self, span: Span) -> int:
        """Calculates the offset between the start point of this span and another's.

        As per the `context.md` specification, this behavior corresponds to the
        `Span - Span` operation. The result can be negative, indicating the
        relative position.

        Args:
            span: The other Span object.

        Returns:
            An integer representing the offset from this span's start to the other's.
        """

        return span.start - self.start


@dataclass(frozen=True)
class Vector:
    x: int = 0
    y: int = 0

    def __abs__(self) -> Vector:
        return Vector(abs(self.x), abs(self.y))


@dataclass(frozen=True)
class Region:
    x: Span
    y: Span

    @classmethod
    def from_size(cls, width: int, height: int) -> Region:
        return cls(Span(width), Span(height))

    def __shift(self, operation: Callable, offset: int | tuple[int, int] | Region | Vector) -> Region:
        if isinstance(offset, (Vector, Region)):
            x, y = offset.x, offset.y

        elif isinstance(offset, tuple):
            x, y = offset[0], offset[1]

        elif isinstance(offset, int):
            x = y = offset

        else:
            raise TypeError(
                "offset must be an int, a (x, y) tuple, or a Vector instance "
                f"(got {type(offset).__name__})"
            )
        return Region(operation(self.x, x), operation(self.y, y))

    def __add__(self, offset: int | tuple[int, int] | Region | Vector) -> Region:
        return self.__shift(add, offset)

    def __sub__(self, offset: int | tuple[int, int] | Region | Vector) -> Region:
        return self.__shift(sub, offset)

    def __or__(self, other: Region) -> Region:
        return Region(self.x | other.x, self.y | other.y)

    def __and__(self, other: Region) -> Region:
        return Region(self.x & other.x, self.y & other.y)

    def _apply_margins(
        self,
        span_op_x: Callable,
        span_op_y: Callable,
        all: Vector | int | None = None,
        *,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0
    ) -> Region:

        if isinstance(all, Vector):
            left = right = all.x
            top = bottom = all.y

        elif isinstance(all, int):
            left = right = all
            top = bottom = all

        return Region(
            span_op_x(before=left, after=right),
            span_op_y(before=top, after=bottom)
        )

    @property
    def area(self) -> int:
        return self.x.length * self.y.length

    @property
    def width(self) -> int:
        return self.x.length

    @property
    def height(self) -> int:
        return self.y.length

    def expand(
        self,
        all: Vector | int | None = None,
        *,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0
    ) -> Region:

        return self._apply_margins(
            self.x.expand, self.y.expand, all,
            left=left, right=right,
            top=top, bottom=bottom
        )

    def shrink(
        self,
        all: Vector | int | None = None,
        *,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0
    ) -> Region:

        return self._apply_margins(
            self.x.shrink, self.y.shrink, all,
            left=left, right=right,
            top=top, bottom=bottom
        )

    def offset_to(self, other: Region) -> Vector:
        return Vector(self.x.offset_to(other.x), self.y.offset_to(other.y))

    def overlaps(self, other: Region) -> bool:
        return self.x.overlaps(other.x) and self.y.overlaps(other.y)

    def overlap_with(self, other: Region) -> Region:
        if not self.overlaps(other):
            raise ValueError("no overlap: 'other' out of bounds")
        intersection = self & other
        return intersection - self

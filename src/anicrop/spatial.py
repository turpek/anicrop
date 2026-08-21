from __future__ import annotations
from dataclasses import dataclass
from operator import add, sub
from ovld import ovld
from typing import Callable, Optional


class SpanError(Exception):
    pass


class Span:
    """Represents a continuous numerical interval (span) in one dimension.

    Defined by a `start` coordinate and a positive `length`.
    Supports negative coordinates. This class is immutable.

    Attributes:
        start (int): The starting point of the span.
        length (int): The length of the span (must be > 0).
        end (int): The ending point (start + length).
    """

    @ovld
    def __init__(self, length: int, /):
        """Initializes a Span with start=0.

        Args:
            length: The length of the span. Must be positive.

        Raises:
            ValueError: If `length <= 0`.
        """
        self._setup(0, length)

    @ovld  # type: ignore[no-redef]
    def __init__(self, start: int, length: int, /):  # noqa: F811
        """Initializes a Span.

        Args:
            start: The starting point of the span (can be negative).
            length: The length of the span. Must be positive.

        Raises:
            ValueError: If `length <= 0`.
        """
        self._setup(start, length)

    def _setup(self, start: int, length: int) -> None:
        if length <= 0:
            raise ValueError(f'length must be greater than 0 (length={length})')

        self._start = start
        self._length = length

    def __repr__(self):
        return f"{self.__class__.__name__}(start={self.start}, length={self.length})"

    def __eq__(self, span: object) -> bool:
        if not isinstance(span, Span):
            return NotImplemented

        return self.start == span.start and self.length == span.length

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
        return Span(self.start + offset, self.length)

    def __sub__(self, offset: int | Span) -> Span:
        """Shifts the span to the left.

        Implements the `Span - int` operation.

        Args:
            offset: The integer value to subtract from the start.

        Returns:
            A new Span object shifted to the left.
        """
        if isinstance(offset, Span):
            offset = offset.start

        return Span(self.start - offset, self.length)

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
        return Span(start, end - start)

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
        return Span(overlap_start, overlap_end - overlap_start)

    @property
    def length(self) -> int:
        """The length of the span."""
        return self._length

    @property
    def start(self) -> int:
        """The starting point of the span."""
        return self._start

    @property
    def end(self) -> int:
        """The ending point of the span (start + length)."""
        return self.start + self.length

    def overlaps(self, other: Span) -> bool:
        return self.end > other.start and other.end > self.start

    def expand(self, both: Optional[int] = None, *, before: int = 0, after: int = 0) -> Span:
        """Expands the span outward.

        Args:
            both: Amount to expand on both sides.
            before: Amount to expand the start (moves left).
            after: Amount to expand the end (moves right).

        Returns:
            A new Span object expanded.

        Raises:
            ValueError: If any margin is negative.
        """

        if both:
            before = after = both

        if after < 0 or before < 0:
            raise ValueError(
                "Margin for expand() must be non-negative. To contract "
                "the span, use the shrink() method with a positive value."
            )
        return Span(self.start - before, before + self.length + after)

    def shrink(self, both: Optional[int] = None, *, before: int = 0, after: int = 0) -> Span:
        """Shrinks the span inward.

        Guarantees that the resulting span stays within the original bounds
        (no drift) and has a minimum length of 1.

        Args:
            both: Amount to shrink on both sides.
            before: Amount to shrink from the start (moves right).
            after: Amount to shrink from the end (moves left).

        Returns:
            A new, smaller Span object.

        Raises:
            ValueError: If any margin is negative.
        """
        if both:
            before = after = both

        if before < 0 or after < 0:
            raise ValueError(
                "Margin for shrink() must be non-negative. To contract "
                "To expand the span, use the expand() method with a positive margin."
            )

        start = min(self.start + before, self.end)
        end = max(self.end - after, self.start)

        if start >= self.end:
            start -= 1
        if start >= end:
            end = start + 1

        return Span(start, end - start)

    def offset_to(self, span: Span, anchor_end: bool = False) -> int:
        """Calculates the offset distance between this span and another.

        When `anchor_end` is False (default), calculates the offset between
        start points (`span.start - self.start`).
        When `anchor_end` is True, calculates the offset between end points
        (`span.end - self.end`).

        Args:
            span: The target Span object.
            anchor_end: If True, calculates the offset between end points instead of start points.

        Returns:
            An integer representing the signed offset from this span to the target span.
        """

        if anchor_end:
            return span.end - self.end
        return span.start - self.start

    def slack(self, span: Span) -> int:
        """Calculates the available free space between this span and a reference span.

        The slack is the difference in length between the reference span and this span.
        A positive value means the reference span is larger.

        Args:
            span: The reference Span to compare against.

        Returns:
            An integer representing the difference in length (`span.length - self.length`).
        """
        return span.length - self.length

    def align(self, span: Span, factor: float = 0.5) -> Span:
        """Aligns this span relative to a reference span based on a given factor.

        Calculates a new starting position by distributing the slack (free space)
        according to the alignment factor. Returns a new Span with the updated
        start position and the original length.

        Args:
            span: The reference Span to align to.
            factor: A float between 0.0 and 1.0 representing the alignment position.
                0.0 aligns to the start (left/top), 0.5 to the center, and 1.0 to the end (right/bottom).

        Returns:
            A new Span instance with the translated position.
        """
        start = round(span.start + (self.slack(span)) * factor)
        return Span(start, self.length)

    def replace(self, value: int | Span | None) -> Span:
        """Replaces the start coordinate or the entire span.

        Args:
            value: An integer representing a new start position (preserving length),
                a replacement Span instance, or None to keep the current span.

        Returns:
            A new Span instance if replaced, or the current Span instance if value is None.
        """
        if isinstance(value, int):
            return Span(value, self.length)
        elif isinstance(value, Span):
            return value
        return self


@dataclass(frozen=True)
class Region:
    x: Span
    y: Span

    @classmethod
    def from_size(cls, width: int, height: int) -> Region:
        return cls(Span(width), Span(height))

    @classmethod
    def from_rect(cls, x: int, y: int, width: int, height: int) -> Region:
        return cls(Span(x, width), Span(y, height))

    def __repr__(self):
        start = f'start=({self.x.start},{self.y.start})'
        length = f'length=({self.x.length},{self.y.length})'
        return f'{type(self).__name__}({start}, {length})'

    def __shift(self, operation: Callable, offset: int | tuple[int, int] | Region) -> Region:
        if isinstance(offset, Region):
            x, y = offset.x.start, offset.y.start

        elif isinstance(offset, tuple):
            x, y = offset[0], offset[1]

        elif isinstance(offset, int):
            x = y = offset

        else:
            raise TypeError(
                "offset must be an int, a (x, y) tuple, or a Region instance "
                f"(got {type(offset).__name__})"
            )
        return Region(operation(self.x, x), operation(self.y, y))

    def __add__(self, offset: int | tuple[int, int] | Region) -> Region:
        return self.__shift(add, offset)

    def __sub__(self, offset: int | tuple[int, int] | Region) -> Region:
        return self.__shift(sub, offset)

    def __or__(self, other: Region) -> Region:
        return Region(self.x | other.x, self.y | other.y)

    def __and__(self, other: Region) -> Region:
        """Computes the intersection in global coordinates (Canvas space).

        Useful for determining the shared area on the canvas (Destination).
        """
        return Region(self.x & other.x, self.y & other.y)

    def _apply_margins(
        self,
        span_op_x: Callable,
        span_op_y: Callable,
        all: int | tuple[int, int] | None = None,
        *,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0
    ) -> Region:

        if isinstance(all, tuple):
            left = right = all[0]
            top = bottom = all[1]

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

    @property
    def size(self) -> tuple[int, int]:
        """Returns the dimensions of the region as a (width, height) tuple."""
        return (self.width, self.height)

    @property
    def top_left(self) -> tuple[int, int]:
        """Returns the top-left coordinates as an (x, y) tuple."""
        return (self.x.start, self.y.start)

    @property
    def bottom_right(self) -> tuple[int, int]:
        """Returns the bottom-right coordinates as an (x, y) tuple."""
        return (self.x.end, self.y.end)

    def expand(
        self,
        all: int | tuple[int, int] | None = None,
        *,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0
    ) -> Region:
        """Expands the region outward."""
        return self._apply_margins(
            self.x.expand, self.y.expand, all,
            left=left, right=right,
            top=top, bottom=bottom
        )

    def shrink(
        self,
        all: int | tuple[int, int] | None = None,
        *,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0
    ) -> Region:
        """Shrinks the region inward."""
        return self._apply_margins(
            self.x.shrink, self.y.shrink, all,
            left=left, right=right,
            top=top, bottom=bottom
        )

    def offset_to(self, other: Region, anchor_end: bool = False) -> tuple[int, int]:
        """Calculates the 2D offset (x, y) between this region and another.

        When `anchor_end` is False (default), calculates offsets between top-left start points.
        When `anchor_end` is True, calculates offsets between bottom-right end points.

        Args:
            other: The target Region object.
            anchor_end: If True, calculates offsets between end points instead of start points.

        Returns:
            A tuple (x, y) containing the offset values.
        """
        return (
            self.x.offset_to(other.x, anchor_end=anchor_end),
            self.y.offset_to(other.y, anchor_end=anchor_end)
        )

    def overlaps(self, other: Region) -> bool:
        return self.x.overlaps(other.x) and self.y.overlaps(other.y)

    def overlap_with(self, other: Region) -> Region:
        """Calculates the intersection relative to this region (Source Slice).

        Returns a Region expressed in local coordinates (0,0 is top-left of self).
        Useful for NumPy slicing: `slice = img.overlap_with(canvas)`.
        """
        if not self.overlaps(other):
            raise ValueError("no overlap: 'other' out of bounds")
        intersection = self & other
        return intersection - self

    def align(
        self,
        ref: Region,
        x_factor: float = 0.5,
        y_factor: float = 0.5,
    ) -> Region:
        """Aligns this region relative to a reference region.

        Uses the alignment factors to distribute the slack on both the X and Y axes,
        returning a newly translated Region that preserves its original dimensions.

        Args:
            ref: The reference Region to align to.
            x_factor: Alignment factor for the horizontal (X) axis (0.0 to 1.0).
            y_factor: Alignment factor for the vertical (Y) axis (0.0 to 1.0).

        Returns:
            A new Region instance aligned to the reference.
        """
        return Region(
            self.x.align(ref.x, x_factor),
            self.y.align(ref.y, y_factor),
        )

    def replace(self, *, x: int | Span | None = None, y: int | Span | None = None) -> Region:
        """Creates a new Region with updated horizontal (X) and/or vertical (Y) spans.

        Args:
            x: An integer for a new X start position, a replacement X Span, or None.
            y: An integer for a new Y start position, a replacement Y Span, or None.

        Returns:
            A new Region instance with the specified span updates.
        """
        return Region(self.x.replace(x), self.y.replace(y))


def rect_to_region(rect: tuple[int, int, int, int]):
    x, y, w, h = rect
    return Region(Span(x, w), Span(y, h))

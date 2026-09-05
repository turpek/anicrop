from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from operator import add, sub
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, overload

from ovld import ovld

DEFAULT_EPSILON: float = 1e-4


class SpanError(Exception):
    pass


class Point(NamedTuple):
    """Representa um ponto ou dimensão 2D contínua em float, compatível com tuple[float, float]."""

    x: float
    y: float

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (tuple, list)) and len(other) == 2:
            return math.isclose(
                self[0], other[0], abs_tol=DEFAULT_EPSILON
            ) and math.isclose(self[1], other[1], abs_tol=DEFAULT_EPSILON)
        return False

    def to_int(self, mode: str = "round") -> tuple[int, int]:
        """Converte as coordenadas do Point para uma tupla de inteiros discretos (x, y)."""
        if mode == "round":
            return (int(round(self.x)), int(round(self.y)))
        elif mode == "floor":
            return (int(math.floor(self.x)), int(math.floor(self.y)))
        elif mode == "ceil":
            return (int(math.ceil(self.x)), int(math.ceil(self.y)))
        return (int(self.x), int(self.y))

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"


class Span:
    """Represents a continuous numerical interval (span) in one dimension.

    Defined by a `start` coordinate and a positive `length`.
    Supports negative coordinates and continuous floating-point numbers.
    This class is immutable.

    Attributes:
        start (float): The starting point of the span.
        length (float): The length of the span (must be > 0).
        end (float): The ending point (start + length).
    """

    if TYPE_CHECKING:

        @overload
        def __init__(self, length: float, /) -> None:
            pass

        @overload
        def __init__(self, start: float, length: float, /) -> None:
            pass

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    else:

        @ovld
        def __init__(self, length: int | float, /):
            """Initializes a Span with start=0.

            Args:
                length: The length of the span. Must be positive.

            Raises:
                ValueError: If `length <= 0`.
            """
            self._setup(0, length)

        @ovld
        def __init__(self, start: int | float, length: int | float, /):  # noqa: F811
            """Initializes a Span.

            Args:
                start: The starting point of the span (can be negative).
                length: The length of the span. Must be positive.

            Raises:
                ValueError: If `length <= 0`.
            """
            self._setup(start, length)

    def _setup(self, start: float, length: float) -> None:
        if length <= 0:
            raise ValueError(f"length must be greater than 0 (length={length})")

        self._start = float(start)
        self._length = float(length)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(start={self.start}, length={self.length})"

    def __eq__(self, span: object) -> bool:
        if not isinstance(span, Span):
            return NotImplemented

        return math.isclose(
            self.start, span.start, abs_tol=DEFAULT_EPSILON
        ) and math.isclose(self.length, span.length, abs_tol=DEFAULT_EPSILON)

    def __add__(self, offset: int | float | Span) -> Span:
        """Shifts the span to the right.

        Implements the `Span + offset` operation.

        Args:
            offset: The value to add to the start and end points.

        Returns:
            A new Span object shifted to the right.
        """
        if isinstance(offset, Span):
            offset = offset.start
        return Span(self.start + offset, self.length)

    def __sub__(self, offset: int | float | Span) -> Span:
        """Shifts the span to the left.

        Implements the `Span - offset` operation.

        Args:
            offset: The value to subtract from the start.

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
    def length(self) -> int | float:
        """The length of the span."""
        return self._length

    @property
    def start(self) -> int | float:
        """The starting point of the span."""
        return self._start

    @property
    def end(self) -> int | float:
        """The ending point of the span (start + length)."""
        return self.start + self.length

    def overlaps(self, other: Span) -> bool:
        return (self.end > other.start + DEFAULT_EPSILON) and (
            other.end > self.start + DEFAULT_EPSILON
        )

    def expand(
        self,
        both: Optional[int | float] = None,
        *,
        before: int | float = 0,
        after: int | float = 0,
    ) -> Span:
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

    def shrink(
        self,
        both: Optional[int | float] = None,
        *,
        before: int | float = 0,
        after: int | float = 0,
    ) -> Span:
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
            start = self.end - 1
        if start >= end:
            end = start + 1

        return Span(start, end - start)

    def offset_to(self, span: Span, anchor_end: bool = False) -> int | float:
        """Calculates the offset distance between this span and another.

        When `anchor_end` is False (default), calculates the offset between
        start points (`span.start - self.start`).
        When `anchor_end` is True, calculates the offset between end points
        (`span.end - self.end`).

        Args:
            span: The target Span object.
            anchor_end: If True, calculates the offset between end points instead of start points.

        Returns:
            A signed offset from this span to the target span.
        """
        if anchor_end:
            return span.end - self.end
        return span.start - self.start

    def slack(self, span: Span) -> int | float:
        """Calculates the available free space between this span and a reference span.

        The slack is the difference in length between the reference span and this span.
        A positive value means the reference span is larger.

        Args:
            span: The reference Span to compare against.

        Returns:
            An offset representing the difference in length (`span.length - self.length`).
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
        start = span.start + (self.slack(span)) * factor
        return Span(start, self.length)

    def replace(self, value: float | Span | None) -> Span:
        """Replaces the start coordinate or the entire span.

        Args:
            value: A number representing a new start position (preserving length),
                a replacement Span instance, or None to keep the current span.

        Returns:
            A new Span instance if replaced, or the current Span instance if value is None.
        """
        if isinstance(value, (int, float, Real)):
            return Span(float(value), self.length)
        elif isinstance(value, Span):
            return value
        return self

    def to_int(self, mode: str = "round") -> Span:
        """Converts this span's start and length into discrete integers."""
        if mode == "round":
            return Span(int(round(self.start)), max(1, int(round(self.length))))
        elif mode == "floor":
            return Span(
                int(math.floor(self.start)), max(1, int(math.floor(self.length)))
            )
        elif mode == "ceil":
            return Span(int(math.ceil(self.start)), max(1, int(math.ceil(self.length))))
        return Span(int(self.start), max(1, int(self.length)))

    def to_slice(self, mode: str = "round") -> slice:
        """Converts this continuous span into a discrete Python slice.

        Uses start and length to guarantee that the discrete slice has an exact
        length of max(1, quantize(length)) without subpixel size drift.
        """
        int_span = self.to_int(mode=mode)
        start_idx = int(int_span.start)
        return slice(start_idx, start_idx + int(int_span.length))


@dataclass(frozen=True)
class Region:
    x: Span
    y: Span

    @classmethod
    def from_size(cls, width: float, height: float) -> Region:
        return cls(Span(width), Span(height))

    @classmethod
    def from_rect(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> Region:
        return cls(Span(x, width), Span(y, height))

    def __repr__(self) -> str:
        start = f"start=({self.x.start},{self.y.start})"
        length = f"length=({self.x.length},{self.y.length})"
        return f"{type(self).__name__}({start}, {length})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Region):
            return False
        return self.x == other.x and self.y == other.y

    def __shift(
        self,
        operation: Callable,
        offset: float | tuple[float, float] | Point | Region,
    ) -> Region:
        if isinstance(offset, Region):
            x, y = offset.x.start, offset.y.start

        elif isinstance(offset, (tuple, Point)):
            x, y = float(offset[0]), float(offset[1])

        elif isinstance(offset, (int, float, Real)):
            x = y = float(offset)

        else:
            raise TypeError(
                "offset must be a number, a (x, y) tuple, a Point, or a Region instance "
                f"(got {type(offset).__name__})"
            )
        return Region(operation(self.x, x), operation(self.y, y))

    def __add__(
        self,
        offset: float | tuple[float, float] | Point | Region,
    ) -> Region:
        return self.__shift(add, offset)

    def __sub__(
        self,
        offset: float | tuple[float, float] | Point | Region,
    ) -> Region:
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
        all: float | tuple[float, float] | None = None,
        *,
        left: float = 0.0,
        right: float = 0.0,
        top: float = 0.0,
        bottom: float = 0.0,
    ) -> Region:
        if isinstance(all, (tuple, list)):
            left = right = float(all[0])
            top = bottom = float(all[1])

        elif isinstance(all, (int, float, Real)):
            left = right = float(all)
            top = bottom = float(all)

        return Region(
            span_op_x(before=left, after=right),
            span_op_y(before=top, after=bottom),
        )

    @property
    def area(self) -> float:
        return self.x.length * self.y.length

    @property
    def width(self) -> float:
        return self.x.length

    @property
    def height(self) -> float:
        return self.y.length

    @property
    def size(self) -> Point:
        """Returns the dimensions of the region as a tuple-compatible Point(width, height)."""
        return Point(self.width, self.height)

    @property
    def top_left(self) -> Point:
        """Returns the top-left coordinates as a tuple-compatible Point(x, y)."""
        return Point(self.x.start, self.y.start)

    @property
    def bottom_right(self) -> Point:
        """Returns the bottom-right coordinates as a tuple-compatible Point(x, y)."""
        return Point(self.x.end, self.y.end)

    def expand(
        self,
        all: float | tuple[float, float] | None = None,
        *,
        left: float = 0.0,
        right: float = 0.0,
        top: float = 0.0,
        bottom: float = 0.0,
    ) -> Region:
        """Expands the region outward."""
        return self._apply_margins(
            self.x.expand,
            self.y.expand,
            all,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )

    def shrink(
        self,
        all: float | tuple[float, float] | None = None,
        *,
        left: float = 0.0,
        right: float = 0.0,
        top: float = 0.0,
        bottom: float = 0.0,
    ) -> Region:
        """Shrinks the region inward."""
        return self._apply_margins(
            self.x.shrink,
            self.y.shrink,
            all,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )

    def offset_to(self, other: Region, anchor_end: bool = False) -> Point:
        """Calculates the 2D offset (x, y) between this region and another.

        When `anchor_end` is False (default), calculates offsets between top-left start points.
        When `anchor_end` is True, calculates offsets between bottom-right end points.

        Args:
            other: The target Region object.
            anchor_end: If True, calculates offsets between end points instead of start points.

        Returns:
            A tuple-compatible Point containing the offset values.
        """
        return Point(
            self.x.offset_to(other.x, anchor_end=anchor_end),
            self.y.offset_to(other.y, anchor_end=anchor_end),
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

    def replace(
        self,
        *,
        x: float | Span | None = None,
        y: float | Span | None = None,
    ) -> Region:
        """Creates a new Region with updated horizontal (X) and/or vertical (Y) spans.

        Args:
            x: A number for a new X start position, a replacement X Span, or None.
            y: A number for a new Y start position, a replacement Y Span, or None.

        Returns:
            A new Region instance with the specified span updates.
        """
        return Region(self.x.replace(x), self.y.replace(y))

    def fit_contain(
        self,
        ref: Region,
        x_factor: float = 0.5,
        y_factor: float = 0.5,
    ) -> Region:
        """Scales this region proportionally to fit completely inside the reference region and aligns it.

        Args:
            ref: The reference Region boundaries.
            x_factor: Alignment factor for horizontal axis (0.0=left, 0.5=center, 1.0=right).
            y_factor: Alignment factor for vertical axis (0.0=top, 0.5=center, 1.0=bottom).

        Returns:
            A new Region scaled and aligned inside ref.
        """
        scale = min(ref.width / self.width, ref.height / self.height)
        new_w = self.width * scale
        new_h = self.height * scale
        scaled_region = Region.from_rect(self.x.start, self.y.start, new_w, new_h)
        return scaled_region.align(ref, x_factor, y_factor)

    def fit_cover(
        self,
        ref: Region,
        x_factor: float = 0.5,
        y_factor: float = 0.5,
    ) -> Region:
        """Scales this region proportionally to cover the entire reference region and aligns it.

        Args:
            ref: The reference Region boundaries.
            x_factor: Alignment factor for horizontal axis (0.0=left, 0.5=center, 1.0=right).
            y_factor: Alignment factor for vertical axis (0.0=top, 0.5=center, 1.0=bottom).

        Returns:
            A new Region scaled and aligned covering ref.
        """
        scale = max(ref.width / self.width, ref.height / self.height)
        new_w = self.width * scale
        new_h = self.height * scale
        scaled_region = Region.from_rect(self.x.start, self.y.start, new_w, new_h)
        return scaled_region.align(ref, x_factor, y_factor)

    def scale_width(self, width: float) -> Region:
        """Scales this region to a specific width, calculating proportional height and keeping position.

        Args:
            width: Target positive width.

        Returns:
            A new Region with the new width and proportional height.
        """
        if width <= 0:
            raise ValueError(f"Width must be positive, got {width}")
        height = self.height * (width / self.width)
        return Region.from_rect(self.x.start, self.y.start, width, height)

    def scale_height(self, height: float) -> Region:
        """Scales this region to a specific height, calculating proportional width and keeping position.

        Args:
            height: Target positive height.

        Returns:
            A new Region with the new height and proportional width.
        """
        if height <= 0:
            raise ValueError(f"Height must be positive, got {height}")
        width = self.width * (height / self.height)
        return Region.from_rect(self.x.start, self.y.start, width, height)

    def to_int(self, mode: str = "round") -> Region:
        """Converts both X and Y spans into discrete integer spans."""
        return Region(self.x.to_int(mode=mode), self.y.to_int(mode=mode))

    def to_slice(self, mode: str = "round") -> tuple[slice, slice]:
        """Returns a (slice_y, slice_x) tuple for direct 2D matrix indexing in NumPy."""
        return (self.y.to_slice(mode=mode), self.x.to_slice(mode=mode))


def rect_to_region(
    rect: tuple[float, float, float, float] | Sequence[float],
) -> Region:
    return Region.from_rect(rect[0], rect[1], rect[2], rect[3])

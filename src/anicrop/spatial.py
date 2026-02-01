from __future__ import annotations
from ovld import ovld


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

    def __add__(self, offset: int) -> Span:
        """Shifts the span to the right.

        Implements the `Span + int` operation.

        Args:
            offset: The integer value to add to the start and end points.

        Returns:
            A new Span object shifted to the right.
        """
        return Span(self.start + offset, self.end + offset)

    def __sub__(self, offset: int) -> Span:
        """Shifts the span to the left.

        Implements the `Span - int` operation. Ensures the new start point
        is not less than zero.

        Args:
            offset: The integer value to subtract from the start and end points.

        Returns:
            A new Span object shifted to the left.
        """

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
            SpanError: If there is no overlap between the spans.
        """

        if self.end <= span.start or span.end <= self.start:
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

    def expand(self, margin: int) -> Span:
        """Expands the span outward.

        Args:
            margin: The number of units to expand on each side.

        Returns:
            A new Span object expanded.

        Raises:
            ValueError: If the margin is negative.
        """

        start_expand = max(0, self.start - margin)

        if margin < 0:
            raise ValueError(
                "Margin for expand() must be non-negative. To contract "
                "the span, use the shrink() method with a positive margin."
            )

        return Span(start_expand, self.end + margin)

    def shrink(self, margin: int) -> Span:

        """Shrinks the span inward.

        Args:
            margin: The non-negative number of units to shrink on each side.

        Returns:
            A new, smaller Span object.

        Raises:
            ValueError: If the margin is negative.
        """

        if margin < 0:
            raise ValueError(
                f"Margin for shrink() must be non-negative, but got {margin}. "
                "To expand the span, use the expand() method with a positive margin."
            )

        return Span(self.start + margin, self.end - margin)

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

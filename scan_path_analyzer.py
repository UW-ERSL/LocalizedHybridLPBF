"""Preprocess LPBF scan descriptions into canonical lines and pre-patches.

The public entry point is :class:`ScanPathAnalyzer`.  It accepts one ordered
sequence containing explicit laser-on lines, compact patch specifications, or
a mixture of both.  All lengths are in metres, times in seconds, velocities
in metres/second, and scalar directions in degrees counter-clockwise about the
configured build direction.

This module deliberately does not perform time-dependent/dynamic partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np


Vec3: TypeAlias = tuple[float, float, float]


class ScanPathError(ValueError):
    """Raised when a scan description is invalid or internally inconsistent."""


class PatternType(str, Enum):
    UNIDIRECTIONAL = "unidirectional"
    HELIX = "helix"


@dataclass(frozen=True, slots=True)
class AnalyzerTolerances:
    """Numerical tolerances used for validation and raw-line recognition."""

    length_atol: float = 1.0e-9
    length_rtol: float = 1.0e-5
    time_atol: float = 1.0e-10
    time_rtol: float = 1.0e-7
    direction_atol: float = 1.0e-7
    plane_atol: float = 1.0e-9

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ClosedRange:
    """Inclusive lower and upper bounds for one inferred patch parameter."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("Range bounds must be finite")
        if minimum > maximum:
            raise ValueError("Range minimum cannot exceed range maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def contains(self, value: float, *, atol: float, rtol: float) -> bool:
        lower_ok = value >= self.minimum or math.isclose(
            value, self.minimum, abs_tol=atol, rel_tol=rtol
        )
        upper_ok = value <= self.maximum or math.isclose(
            value, self.maximum, abs_tol=atol, rel_tol=rtol
        )
        return lower_ok and upper_ok


@dataclass(frozen=True, slots=True)
class UnidirectionalTrainingDomain:
    """Training ranges used only when a patch is inferred from raw lines.

    An inferred group must contain strictly more than ``n_min`` consecutive
    lines to become a pre-patch.  There is deliberately no maximum
    ``n`` here: a large pre-patch may later be divided by the dynamic
    partitioner.  The time-dependent age ``tau`` is also not checked here.
    """

    hatch_spacing: ClosedRange
    line_length: ClosedRange
    velocity: ClosedRange
    off_time: ClosedRange
    n_min: int = 2

    def __post_init__(self) -> None:
        _validate_inferred_n_min(self.n_min, "unidirectional n_min")


@dataclass(frozen=True, slots=True)
class HelixTrainingDomain:
    """Training ranges used only when a helix is inferred from raw lines.

    ``n_min`` counts complete contours, not their four straight segments; an
    inferred helix must contain strictly more than ``n_min`` contours.
    There is deliberately no maximum ``n`` here, and the time-dependent age
    ``tau`` is not checked here.
    """

    outer_length: ClosedRange
    inner_length: ClosedRange
    velocity: ClosedRange
    off_time: ClosedRange
    n_min: int = 2

    def __post_init__(self) -> None:
        _validate_inferred_n_min(self.n_min, "helix n_min")


@dataclass(frozen=True, slots=True)
class InferredPatchTrainingDomains:
    """Available surrogate domains for automatic raw-line recognition.

    A missing domain disables automatic creation of that pre-patch type.  It
    does not affect a patch explicitly declared by the user.
    """

    unidirectional: UnidirectionalTrainingDomain | None = None
    helix: HelixTrainingDomain | None = None


@dataclass(frozen=True, slots=True)
class ScanLine:
    """One canonical, straight, constant-speed, laser-on scan segment."""

    line_id: int
    start_time: float
    start_point: Vec3
    end_point: Vec3
    velocity: float
    source_entry_index: int

    @property
    def length(self) -> float:
        return float(np.linalg.norm(_array(self.end_point) - _array(self.start_point)))

    @property
    def duration(self) -> float:
        return self.length / self.velocity

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def direction(self) -> Vec3:
        vector = (_array(self.end_point) - _array(self.start_point)) / self.length
        return _vec3(vector)


@dataclass(frozen=True, slots=True)
class PatchFrame:
    """Patch-local orthonormal frame expressed in global coordinates."""

    origin: Vec3
    e1: Vec3
    e2: Vec3
    e3: Vec3

    @property
    def rotation(self) -> np.ndarray:
        """Return R=[e1 e2 e3], mapping local vectors to global vectors."""
        return np.column_stack((_array(self.e1), _array(self.e2), _array(self.e3)))

    def point_to_local(self, point: Sequence[float]) -> np.ndarray:
        return self.rotation.T @ (_point(point, "point") - _array(self.origin))

    def point_to_global(self, point: Sequence[float]) -> np.ndarray:
        return _array(self.origin) + self.rotation @ _point(point, "point")

    def vector_to_local(self, vector: Sequence[float]) -> np.ndarray:
        return self.rotation.T @ _point(vector, "vector")

    def vector_to_global(self, vector: Sequence[float]) -> np.ndarray:
        return self.rotation @ _point(vector, "vector")


@dataclass(frozen=True, slots=True)
class UnidirectionalParameters:
    n: int
    hatch_spacing: float
    line_length: float
    velocity: float
    off_time: float

    @property
    def eta(self) -> tuple[float, ...]:
        return (float(self.n), self.hatch_spacing, self.line_length,
                self.velocity, self.off_time)


@dataclass(frozen=True, slots=True)
class HelixParameters:
    """Parameters for concentric closed square contours."""

    n: int
    outer_length: float
    inner_length: float
    velocity: float
    off_time: float

    @property
    def contour_spacing(self) -> float:
        if self.n == 1:
            return 0.0
        return (self.outer_length - self.inner_length) / (2.0 * (self.n - 1))

    @property
    def eta(self) -> tuple[float, ...]:
        return (float(self.n), self.outer_length, self.inner_length,
                self.velocity, self.off_time)


PatchParameters: TypeAlias = UnidirectionalParameters | HelixParameters


@dataclass(frozen=True, slots=True)
class PrePatch:
    """One maximal or explicitly declared static pattern-compatible patch."""

    patch_id: int
    pattern_type: PatternType
    parameters: PatchParameters
    frame: PatchFrame
    lines: tuple[ScanLine, ...]
    declared_by_user: bool

    @property
    def line_ids(self) -> tuple[int, ...]:
        return tuple(line.line_id for line in self.lines)

    @property
    def start_time(self) -> float:
        return self.lines[0].start_time

    @property
    def completion_time(self) -> float:
        return self.lines[-1].end_time

    @property
    def eta(self) -> tuple[float, ...]:
        return self.parameters.eta

    @property
    def logical_unit_count(self) -> int:
        """Number of hatch lines (unidirectional) or contours (helix)."""
        return self.parameters.n


@dataclass(frozen=True, slots=True)
class ScanPathAnalysisResult:
    """Complete static preprocessing result consumed by the hybrid solver."""

    lines: tuple[ScanLine, ...]
    prepatches: tuple[PrePatch, ...]
    analytical_only_lines: tuple[ScanLine, ...]

    def validate_ownership(self) -> None:
        all_ids = [line.line_id for line in self.lines]
        owned_ids = [line.line_id for patch in self.prepatches for line in patch.lines]
        owned_ids.extend(line.line_id for line in self.analytical_only_lines)
        if len(owned_ids) != len(set(owned_ids)):
            raise RuntimeError("A line is owned by more than one output category")
        if set(owned_ids) != set(all_ids):
            raise RuntimeError("Output ownership does not cover every input line")


@dataclass(frozen=True, slots=True)
class _Block:
    lines: tuple[ScanLine, ...]
    declared_type: PatternType | None = None
    parameters: PatchParameters | None = None
    frame: PatchFrame | None = None


class ScanPathAnalyzer:
    """Build canonical scan lines and static pre-patches from user input.

    Supported entries
    -----------------
    Explicit line mapping::

        {"kind": "line", "start_time": 0.0,
         "start_point": [0, 0, 0], "end_point": [0.006, 0, 0],
         "velocity": 1.0}

    Compact patch mapping::

        {"kind": "patch", "patch_type": "unidirectional",
         "eta": {"n": 80, "h": 80e-6, "L": 6e-3,
                 "V": 1.0, "t_off": 0.01},
         "start_time": 0.0, "start_point": [0, 0, 0],
         "direction": 30.0}

    A line may also use the legacy tuple ``(t_start, p_start, p_end, V)``
    or a flat row ``(t_start, xs, ys, zs, xe, ye, ze, V)``.
    """

    def __init__(
        self,
        tolerances: AnalyzerTolerances | None = None,
        build_direction: Sequence[float] = (0.0, 0.0, 1.0),
        inferred_patch_domains: InferredPatchTrainingDomains | None = None,
    ) -> None:
        self.tol = tolerances or AnalyzerTolerances()
        self._e3 = _unit(build_direction, "build_direction")
        self.inferred_patch_domains = (
            inferred_patch_domains or InferredPatchTrainingDomains()
        )

    def analyze(self, scan_description: Sequence[Any]) -> ScanPathAnalysisResult:
        """Validate and preprocess an ordered mixed scan description."""
        if isinstance(scan_description, (str, bytes, Mapping)):
            raise ScanPathError("scan_description must be an ordered sequence of entries")
        try:
            entries = list(scan_description)
        except TypeError as exc:
            raise ScanPathError("scan_description must be iterable") from exc
        if not entries:
            return ScanPathAnalysisResult((), (), ())

        blocks: list[_Block] = []
        next_line_id = 0
        for entry_index, entry in enumerate(entries):
            if self._entry_kind(entry) == "patch":
                block = self._generate_patch(entry, entry_index, next_line_id)
            else:
                line = self._build_line(entry, entry_index, next_line_id)
                block = _Block((line,))
            blocks.append(block)
            next_line_id += len(block.lines)

        lines = tuple(line for block in blocks for line in block.lines)
        self._validate_chronology(lines)

        prepatches: list[PrePatch] = []
        analytical: list[ScanLine] = []
        raw_run: list[ScanLine] = []

        def flush_raw_run() -> None:
            if not raw_run:
                return
            found, unsupported = self._identify_raw_prepatches(tuple(raw_run))
            prepatches.extend(found)
            analytical.extend(unsupported)
            raw_run.clear()

        for block in blocks:
            if block.declared_type is None:
                raw_run.extend(block.lines)
                continue
            flush_raw_run()
            assert block.parameters is not None and block.frame is not None
            prepatches.append(PrePatch(
                patch_id=-1,
                pattern_type=block.declared_type,
                parameters=block.parameters,
                frame=block.frame,
                lines=block.lines,
                declared_by_user=True,
            ))
        flush_raw_run()

        prepatches.sort(key=lambda patch: patch.lines[0].line_id)
        numbered = tuple(PrePatch(
            patch_id=index,
            pattern_type=patch.pattern_type,
            parameters=patch.parameters,
            frame=patch.frame,
            lines=patch.lines,
            declared_by_user=patch.declared_by_user,
        ) for index, patch in enumerate(prepatches))
        analytical.sort(key=lambda line: line.line_id)

        result = ScanPathAnalysisResult(lines, numbered, tuple(analytical))
        result.validate_ownership()
        return result

    def _entry_kind(self, entry: Any) -> str:
        if not isinstance(entry, Mapping):
            return "line"
        kind = str(entry.get("kind", "")).strip().lower()
        if kind in {"line", "patch"}:
            return kind
        if "patch_type" in entry or "eta" in entry:
            return "patch"
        if "start_point" in entry and "end_point" in entry:
            return "line"
        raise ScanPathError("Mapping entry must describe either a line or a patch")

    def _build_line(self, entry: Any, source_index: int, line_id: int) -> ScanLine:
        if isinstance(entry, Mapping):
            try:
                start_time = entry["start_time"]
                start_point = entry["start_point"]
                end_point = entry["end_point"]
                velocity = entry.get("velocity", entry.get("V"))
            except KeyError as exc:
                raise ScanPathError(f"Line entry {source_index} is missing {exc.args[0]!r}") from exc
            if velocity is None:
                raise ScanPathError(f"Line entry {source_index} is missing 'velocity'")
        else:
            try:
                values = list(entry)
            except TypeError as exc:
                raise ScanPathError(f"Line entry {source_index} is not a supported record") from exc
            if len(values) == 4:
                start_time, start_point, end_point, velocity = values
            elif len(values) == 8:
                start_time = values[0]
                start_point = values[1:4]
                end_point = values[4:7]
                velocity = values[7]
            else:
                raise ScanPathError(
                    f"Line entry {source_index} must have 4 legacy fields or 8 flat fields"
                )
        return self._make_line(line_id, source_index, start_time, start_point,
                               end_point, velocity)

    def _make_line(
        self,
        line_id: int,
        source_index: int,
        start_time: Any,
        start_point: Sequence[float],
        end_point: Sequence[float],
        velocity: Any,
    ) -> ScanLine:
        t = _finite_float(start_time, "start_time")
        v = _finite_float(velocity, "velocity")
        if t < 0.0:
            raise ScanPathError("start_time must be nonnegative")
        if v <= 0.0:
            raise ScanPathError("velocity must be positive")
        p0 = _point(start_point, "start_point")
        p1 = _point(end_point, "end_point")
        length = float(np.linalg.norm(p1 - p0))
        if not math.isfinite(length) or length <= self.tol.length_atol:
            raise ScanPathError("Every scan line must have nonzero finite length")
        return ScanLine(line_id, t, _vec3(p0), _vec3(p1), v, source_index)

    def _generate_patch(
        self, entry: Mapping[str, Any], source_index: int, first_line_id: int
    ) -> _Block:
        try:
            pattern = PatternType(str(entry["patch_type"]).strip().lower())
            eta_raw = entry["eta"]
            start_time = _finite_float(entry["start_time"], "start_time")
            start_point = _point(entry["start_point"], "start_point")
            direction = entry["direction"]
        except KeyError as exc:
            raise ScanPathError(f"Patch entry {source_index} is missing {exc.args[0]!r}") from exc
        except ValueError as exc:
            raise ScanPathError(
                f"Patch entry {source_index} has unsupported patch_type; use 'unidirectional' or 'helix'"
            ) from exc
        if start_time < 0.0:
            raise ScanPathError("start_time must be nonnegative")
        frame_at_start = self._frame_from_direction(start_point, direction)

        if pattern is PatternType.UNIDIRECTIONAL:
            params = self._parse_unidirectional_eta(eta_raw)
            lines = self._generate_unidirectional(
                params, frame_at_start, start_time, source_index, first_line_id
            )
            return _Block(lines, pattern, params, frame_at_start)

        params = self._parse_helix_eta(eta_raw)
        # The supplied point is the first laser point (top-left).  The helix
        # surrogate's local origin is the common square centre.
        half = 0.5 * params.outer_length
        centre = start_point + half * _array(frame_at_start.e1) - half * _array(frame_at_start.e2)
        frame = PatchFrame(_vec3(centre), frame_at_start.e1,
                           frame_at_start.e2, frame_at_start.e3)
        lines = self._generate_helix(
            params, frame, start_time, source_index, first_line_id
        )
        return _Block(lines, pattern, params, frame)

    def _frame_from_direction(self, origin: np.ndarray, direction: Any) -> PatchFrame:
        if np.isscalar(direction):
            angle = math.radians(_finite_float(direction, "direction angle"))
            reference = self._reference_axis()
            transverse = np.cross(self._e3, reference)
            e1 = math.cos(angle) * reference + math.sin(angle) * transverse
        else:
            e1 = _unit(direction, "direction")
            if abs(float(np.dot(e1, self._e3))) > self.tol.direction_atol:
                raise ScanPathError("Patch direction must lie in the plane normal to build_direction")
        e1 = _unit(e1, "direction")
        e2 = _unit(np.cross(self._e3, e1), "transverse direction")
        return PatchFrame(_vec3(origin), _vec3(e1), _vec3(e2), _vec3(self._e3))

    def _reference_axis(self) -> np.ndarray:
        x_axis = np.array((1.0, 0.0, 0.0))
        projected = x_axis - np.dot(x_axis, self._e3) * self._e3
        if np.linalg.norm(projected) <= self.tol.direction_atol:
            y_axis = np.array((0.0, 1.0, 0.0))
            projected = y_axis - np.dot(y_axis, self._e3) * self._e3
        return _unit(projected, "angle reference axis")

    def _parse_unidirectional_eta(self, eta: Any) -> UnidirectionalParameters:
        if isinstance(eta, Mapping):
            n = _pick(eta, "n")
            h = _pick(eta, "h", "hatch_spacing")
            length = _pick(eta, "L", "line_length")
            velocity = _pick(eta, "V", "velocity")
            off_time = _pick(eta, "t_off", "off_time")
        else:
            values = _eta_sequence(eta, "unidirectional")
            n, h, length, velocity, off_time = values
        params = UnidirectionalParameters(
            _positive_integer(n, "n"),
            _positive_float(h, "hatch_spacing"),
            _positive_float(length, "line_length"),
            _positive_float(velocity, "velocity"),
            _nonnegative_float(off_time, "off_time"),
        )
        return params

    def _parse_helix_eta(self, eta: Any) -> HelixParameters:
        if isinstance(eta, Mapping):
            n = _pick(eta, "n")
            outer = _pick(eta, "L_out", "outer_length")
            inner = _pick(eta, "L_in", "inner_length")
            velocity = _pick(eta, "V", "velocity")
            off_time = _pick(eta, "t_off", "off_time")
        else:
            values = _eta_sequence(eta, "helix")
            n, outer, inner, velocity, off_time = values
        params = HelixParameters(
            _positive_integer(n, "n"),
            _positive_float(outer, "outer_length"),
            _positive_float(inner, "inner_length"),
            _positive_float(velocity, "velocity"),
            _nonnegative_float(off_time, "off_time"),
        )
        if params.inner_length > params.outer_length and not self._length_close(
            params.inner_length, params.outer_length
        ):
            raise ScanPathError("Helix inner_length cannot exceed outer_length")
        if params.n == 1 and not self._length_close(
            params.inner_length, params.outer_length
        ):
            raise ScanPathError("For a one-contour helix, inner_length must equal outer_length")
        if params.n > 1 and params.inner_length >= params.outer_length:
            raise ScanPathError("For multiple helix contours, inner_length must be smaller than outer_length")
        return params

    def _generate_unidirectional(
        self,
        params: UnidirectionalParameters,
        frame: PatchFrame,
        start_time: float,
        source_index: int,
        first_line_id: int,
    ) -> tuple[ScanLine, ...]:
        origin = _array(frame.origin)
        e1, e2 = _array(frame.e1), _array(frame.e2)
        duration = params.line_length / params.velocity
        lines = []
        for i in range(params.n):
            p0 = origin + i * params.hatch_spacing * e2
            p1 = p0 + params.line_length * e1
            t0 = start_time + i * (duration + params.off_time)
            lines.append(self._make_line(first_line_id + i, source_index, t0,
                                         p0, p1, params.velocity))
        return tuple(lines)

    def _generate_helix(
        self,
        params: HelixParameters,
        frame: PatchFrame,
        start_time: float,
        source_index: int,
        first_line_id: int,
    ) -> tuple[ScanLine, ...]:
        centre = _array(frame.origin)
        e1, e2 = _array(frame.e1), _array(frame.e2)
        current_time = start_time
        lines: list[ScanLine] = []
        for contour in range(params.n):
            side = params.outer_length - 2.0 * contour * params.contour_spacing
            half = 0.5 * side
            corners = (
                centre - half * e1 + half * e2,  # top-left
                centre + half * e1 + half * e2,  # top-right
                centre + half * e1 - half * e2,  # bottom-right
                centre - half * e1 - half * e2,  # bottom-left
            )
            for edge in range(4):
                p0 = corners[edge]
                p1 = corners[(edge + 1) % 4]
                line_id = first_line_id + len(lines)
                line = self._make_line(line_id, source_index, current_time,
                                       p0, p1, params.velocity)
                lines.append(line)
                current_time = line.end_time
            if contour + 1 < params.n:
                current_time += params.off_time
        return tuple(lines)

    def _validate_chronology(self, lines: tuple[ScanLine, ...]) -> None:
        for previous, current in zip(lines, lines[1:]):
            if current.start_time < previous.end_time and not self._time_close(
                current.start_time, previous.end_time
            ):
                raise ScanPathError(
                    f"Chronology overlap: line {current.line_id} starts at "
                    f"{current.start_time:.12g} before line {previous.line_id} ends at "
                    f"{previous.end_time:.12g}"
                )

    def _identify_raw_prepatches(
        self, lines: tuple[ScanLine, ...]
    ) -> tuple[list[PrePatch], list[ScanLine]]:
        patches: list[PrePatch] = []
        analytical: list[ScanLine] = []
        i = 0
        while i < len(lines):
            helix = self._recognize_helix_at(lines, i)
            if helix is not None:
                end, params, frame = helix
                end, params = self._trim_helix_at_inner_limit(
                    lines, i, end, params
                )
                domain = self.inferred_patch_domains.helix
                if domain is None:
                    analytical.extend(lines[i:end])
                    i = end
                elif self._inferred_helix_parameters_are_in_training_domain(params):
                    if params.n > domain.n_min:
                        patches.append(PrePatch(-1, PatternType.HELIX, params, frame,
                                                lines[i:end], False))
                    else:
                        # A geometrically complete candidate with too few
                        # contours is one rejected candidate, not a new search
                        # starting at each of its contours.
                        analytical.extend(lines[i:end])
                    i = end
                else:
                    # Helix recognition advances in complete four-line units.
                    # Reconsider the next contour as the possible beginning of
                    # a valid candidate instead of consuming the full rejected
                    # candidate.
                    analytical.extend(lines[i:i + 4])
                    i += 4
                continue
            unidirectional = self._recognize_unidirectional_at(lines, i)
            if unidirectional is not None:
                end, params, frame = unidirectional
                domain = self.inferred_patch_domains.unidirectional
                if domain is None:
                    analytical.extend(lines[i:end])
                    i = end
                elif self._inferred_unidirectional_parameters_are_in_training_domain(params):
                    if params.n > domain.n_min:
                        patches.append(PrePatch(-1, PatternType.UNIDIRECTIONAL, params,
                                                frame, lines[i:end], False))
                    else:
                        # The candidate obeys the pattern rules but is too
                        # short, so all of it remains analytical.
                        analytical.extend(lines[i:end])
                    i = end
                else:
                    # Only the first line is rejected.  The next line may be
                    # the beginning of a valid consecutive pre-patch.
                    analytical.append(lines[i])
                    i += 1
                continue
            analytical.append(lines[i])
            i += 1
        return patches, analytical

    def _inferred_unidirectional_parameters_are_in_training_domain(
        self, params: UnidirectionalParameters
    ) -> bool:
        domain = self.inferred_patch_domains.unidirectional
        if domain is None:
            return False
        return (
            domain.hatch_spacing.contains(
                params.hatch_spacing,
                atol=self.tol.length_atol,
                rtol=self.tol.length_rtol,
            )
            and domain.line_length.contains(
                params.line_length,
                atol=self.tol.length_atol,
                rtol=self.tol.length_rtol,
            )
            and domain.velocity.contains(
                params.velocity,
                atol=self.tol.length_atol,
                rtol=self.tol.length_rtol,
            )
            and domain.off_time.contains(
                params.off_time,
                atol=self.tol.time_atol,
                rtol=self.tol.time_rtol,
            )
        )

    def _inferred_helix_parameters_are_in_training_domain(
        self, params: HelixParameters
    ) -> bool:
        domain = self.inferred_patch_domains.helix
        if domain is None:
            return False
        return (
            domain.outer_length.contains(
                params.outer_length,
                atol=self.tol.length_atol,
                rtol=self.tol.length_rtol,
            )
            and domain.inner_length.contains(
                params.inner_length,
                atol=self.tol.length_atol,
                rtol=self.tol.length_rtol,
            )
            and domain.velocity.contains(
                params.velocity,
                atol=self.tol.length_atol,
                rtol=self.tol.length_rtol,
            )
            and domain.off_time.contains(
                params.off_time,
                atol=self.tol.time_atol,
                rtol=self.tol.time_rtol,
            )
        )

    def _trim_helix_at_inner_limit(
        self,
        lines: tuple[ScanLine, ...],
        start: int,
        end: int,
        params: HelixParameters,
    ) -> tuple[int, HelixParameters]:
        """Stop before the first contour below the allowed inner length.

        Helix ``inner_length`` is the side length of the candidate's final
        contour.  As contours shrink monotonically, a contour below the lower
        bound is the first unit that breaks the allowed helix domain.  The
        valid prefix must remain available as a candidate.
        """
        domain = self.inferred_patch_domains.helix
        if domain is None:
            return end, params
        contour_count = params.n
        for contour in range(1, contour_count):
            side = lines[start + 4 * contour].length
            if not domain.inner_length.contains(
                side,
                atol=self.tol.length_atol,
                rtol=self.tol.length_rtol,
            ) and side < domain.inner_length.minimum:
                kept = contour
                if kept < 2:
                    return end, params
                trimmed_end = start + 4 * kept
                trimmed = HelixParameters(
                    kept,
                    params.outer_length,
                    lines[trimmed_end - 4].length,
                    params.velocity,
                    params.off_time,
                )
                return trimmed_end, trimmed
        return end, params

    def _recognize_unidirectional_at(
        self, lines: tuple[ScanLine, ...], start: int
    ) -> tuple[int, UnidirectionalParameters, PatchFrame] | None:
        if start + 1 >= len(lines):
            return None
        first, second = lines[start], lines[start + 1]
        e1 = _array(first.direction)
        if abs(float(np.dot(e1, self._e3))) > self.tol.direction_atol:
            return None
        e2 = _unit(np.cross(self._e3, e1), "transverse direction")
        delta = _array(second.start_point) - _array(first.start_point)
        h = float(np.dot(delta, e2))
        if h <= self.tol.length_atol:
            return None
        if abs(float(np.dot(delta, e1))) > self._length_limit(first.length):
            return None
        if abs(float(np.dot(delta, self._e3))) > self.tol.plane_atol:
            return None
        off_time = second.start_time - first.end_time
        if off_time < -self.tol.time_atol:
            return None
        if not self._same_line_process(first, second, e1):
            return None

        end = start + 2
        while end < len(lines):
            previous, candidate = lines[end - 1], lines[end]
            step = _array(candidate.start_point) - _array(previous.start_point)
            candidate_off = candidate.start_time - previous.end_time
            if not self._same_line_process(first, candidate, e1):
                break
            if not self._length_close(float(np.dot(step, e2)), h):
                break
            if abs(float(np.dot(step, e1))) > self._length_limit(first.length):
                break
            if abs(float(np.dot(step, self._e3))) > self.tol.plane_atol:
                break
            if not self._time_close(candidate_off, off_time):
                break
            end += 1

        frame = PatchFrame(first.start_point, _vec3(e1), _vec3(e2), _vec3(self._e3))
        params = UnidirectionalParameters(
            end - start, h, first.length, first.velocity, max(0.0, off_time)
        )
        return end, params, frame

    def _same_line_process(self, reference: ScanLine, candidate: ScanLine,
                           direction: np.ndarray) -> bool:
        return (
            self._length_close(reference.length, candidate.length)
            and self._length_close(reference.velocity, candidate.velocity)
            and np.linalg.norm(_array(candidate.direction) - direction)
            <= self.tol.direction_atol
        )

    def _recognize_helix_at(
        self, lines: tuple[ScanLine, ...], start: int
    ) -> tuple[int, HelixParameters, PatchFrame] | None:
        first_square = self._parse_square(lines, start)
        second_square = self._parse_square(lines, start + 4)
        if first_square is None or second_square is None:
            return None
        centre, outer, e1, velocity, end_time = first_square
        centre2, side2, e12, velocity2, end_time2 = second_square
        if not self._point_close(centre, centre2):
            return None
        if np.linalg.norm(e1 - e12) > self.tol.direction_atol:
            return None
        if not self._length_close(velocity, velocity2) or side2 >= outer:
            return None
        decrement = outer - side2
        off_time = lines[start + 4].start_time - end_time
        if off_time < -self.tol.time_atol:
            return None

        contour_count = 2
        cursor = start + 8
        previous_end_time = end_time2
        previous_side = side2
        while True:
            square = self._parse_square(lines, cursor)
            if square is None:
                break
            c, side, d, v, square_end = square
            gap = lines[cursor].start_time - previous_end_time
            if not self._point_close(c, centre):
                break
            if np.linalg.norm(d - e1) > self.tol.direction_atol:
                break
            if not self._length_close(v, velocity):
                break
            if not self._length_close(previous_side - side, decrement):
                break
            if not self._time_close(gap, off_time):
                break
            contour_count += 1
            cursor += 4
            previous_side = side
            previous_end_time = square_end

        inner = outer - (contour_count - 1) * decrement
        e2 = _unit(np.cross(self._e3, e1), "transverse direction")
        frame = PatchFrame(_vec3(centre), _vec3(e1), _vec3(e2), _vec3(self._e3))
        params = HelixParameters(contour_count, outer, inner, velocity,
                                 max(0.0, off_time))
        return start + 4 * contour_count, params, frame

    def _parse_square(
        self, lines: tuple[ScanLine, ...], start: int
    ) -> tuple[np.ndarray, float, np.ndarray, float, float] | None:
        if start + 4 > len(lines):
            return None
        edges = lines[start:start + 4]
        velocity = edges[0].velocity
        side = edges[0].length
        e1 = _array(edges[0].direction)
        if abs(float(np.dot(e1, self._e3))) > self.tol.direction_atol:
            return None
        e2 = _unit(np.cross(self._e3, e1), "transverse direction")
        for edge in edges:
            if not self._length_close(edge.length, side):
                return None
            if not self._length_close(edge.velocity, velocity):
                return None
            if abs(float(np.dot(_array(edge.direction), self._e3))) > self.tol.direction_atol:
                return None
        for previous, current in zip(edges, edges[1:]):
            if not self._point_close(_array(previous.end_point), _array(current.start_point)):
                return None
            if not self._time_close(previous.end_time, current.start_time):
                return None
        if not self._point_close(_array(edges[-1].end_point), _array(edges[0].start_point)):
            return None
        expected_directions = (e1, -e2, -e1, e2)
        if any(np.linalg.norm(_array(edge.direction) - expected) > self.tol.direction_atol
               for edge, expected in zip(edges, expected_directions)):
            return None
        centre = (
            _array(edges[0].start_point)
            + 0.5 * side * e1
            - 0.5 * side * e2
        )
        expected_corners = (
            centre - 0.5 * side * e1 + 0.5 * side * e2,
            centre + 0.5 * side * e1 + 0.5 * side * e2,
            centre + 0.5 * side * e1 - 0.5 * side * e2,
            centre - 0.5 * side * e1 - 0.5 * side * e2,
        )
        if any(not self._point_close(_array(edge.start_point), expected)
               for edge, expected in zip(edges, expected_corners)):
            return None
        return centre, side, e1, velocity, edges[-1].end_time

    def _length_limit(self, scale: float) -> float:
        return self.tol.length_atol + self.tol.length_rtol * abs(scale)

    def _length_close(self, a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=self.tol.length_rtol,
                            abs_tol=self.tol.length_atol)

    def _time_close(self, a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=self.tol.time_rtol,
                            abs_tol=self.tol.time_atol)

    def _point_close(self, a: np.ndarray, b: np.ndarray) -> bool:
        # Coincidence is a geometric, translation-invariant condition; using
        # global coordinate magnitudes here would make the result depend on
        # how far the part is from the global origin.
        return float(np.linalg.norm(a - b)) <= self.tol.length_atol


def _array(value: Sequence[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _vec3(value: Sequence[float]) -> Vec3:
    array = np.asarray(value, dtype=np.float64)
    return (float(array[0]), float(array[1]), float(array[2]))


def _point(value: Sequence[float], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ScanPathError(f"{name} must contain three finite numbers") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ScanPathError(f"{name} must contain exactly three finite numbers")
    return array


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    vector = _point(value, name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ScanPathError(f"{name} must be nonzero")
    return vector / norm


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScanPathError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ScanPathError(f"{name} must be a finite number")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ScanPathError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ScanPathError(f"{name} must be nonnegative")
    return result


def _positive_integer(value: Any, name: str) -> int:
    number = _finite_float(value, name)
    rounded = round(number)
    if rounded < 1 or not math.isclose(number, rounded, abs_tol=1.0e-12):
        raise ScanPathError(f"{name} must be a positive integer")
    return int(rounded)


def _validate_inferred_n_min(value: Any, name: str) -> None:
    """Validate the minimum size used for automatically inferred pre-patches."""
    number = _finite_float(value, name)
    rounded = round(number)
    if rounded < 2 or not math.isclose(number, rounded, abs_tol=1.0e-12):
        raise ValueError(f"{name} must be an integer greater than or equal to 2")


def _pick(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise ScanPathError(f"eta is missing one of the required keys: {keys}")


def _eta_sequence(eta: Any, pattern_name: str) -> list[Any]:
    if isinstance(eta, (str, bytes)):
        raise ScanPathError(f"{pattern_name} eta must be a mapping or five values")
    try:
        values = list(eta)
    except TypeError as exc:
        raise ScanPathError(f"{pattern_name} eta must be a mapping or five values") from exc
    if len(values) != 5:
        raise ScanPathError(f"{pattern_name} eta must contain exactly five values")
    return values


__all__ = [
    "AnalyzerTolerances",
    "ClosedRange",
    "HelixTrainingDomain",
    "HelixParameters",
    "InferredPatchTrainingDomains",
    "PatchFrame",
    "PatternType",
    "PrePatch",
    "ScanLine",
    "ScanPathAnalysisResult",
    "ScanPathAnalyzer",
    "ScanPathError",
    "UnidirectionalTrainingDomain",
    "UnidirectionalParameters",
]

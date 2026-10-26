"""Conventional and residual-consistent localized analytical--FEA LPBF hybrid.

This module intentionally contains no surrogate machinery and no scan-path
partitioner.  The modes use the same mesh, time
steps, material, scan path, and Eagar--Tsai engine:

Conventional hybrid
    T = T_A + T_F

Localized hybrid
    T = sum_l chi_l T_l + T_F,
    chi_l(x,t) = s_l(x) a_l(t).

The conventional mode evaluates every started scan line at every required
point and retains it for all later times.  The localized mode uses a smooth
spatial window around each finite scan segment and a smooth temporal transfer
window after the segment has completed.

Temporal truncation evaluates retained lines everywhere until a configurable
age after completion, then discards them without a localization residual or
transfer. Ordinary hybrid boundary corrections remain active.

The localized correction load is assembled from the WEAK FORM of the derived
residual.  This is important: it avoids the explicit Laplacian of s and only
requires T_l, grad(T_l), s, grad(s), and da/dt.

For one line l, after using the total-field Neumann condition, the additional
volume terms are

  integral [ -rho cp s adot T_l + k a grad(s).grad(T_l) ] N_i dV
  - integral k a T_l grad(s).grad(N_i) dV,

and the boundary correction term is

  - integral k s a grad(T_l).n N_i dGamma.

Tet4 centroid quadrature is used for the volume residual.  This keeps the
implementation small and makes the added residual cost explicit in the timing
breakdown.  If a later convergence study requires higher-order volume
quadrature, it can be replaced in one place without changing the formulation.
"""

from __future__ import annotations

import copy
import inspect
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np


from pyto_runtime import configure_pyto, environment_metadata
PYTO_SRC = configure_pyto()

import linear_solvers
from tet_transient_thermal import TetTransientThermalFEA

Array = np.ndarray


class HybridConfigurationError(ValueError):
    """Raised when the experiment inputs are mutually inconsistent."""


class HybridMode(str, Enum):
    CONVENTIONAL = "conventional"
    LOCALIZED = "localized"
    TEMPORAL_TRUNCATION = "temporal_truncation"


@dataclass(frozen=True)
class BoundaryQuadrature:
    """One centroid quadrature point on each triangular Neumann face."""

    face_nodes: Array
    points_global: Array
    normals_global: Array
    weights: Array
    shape_values: Array | None = None

    def __post_init__(self) -> None:
        faces = np.asarray(self.face_nodes, dtype=int)
        points = np.asarray(self.points_global, dtype=float)
        normals = np.asarray(self.normals_global, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise HybridConfigurationError("face_nodes must have shape (m, 3)")
        m = faces.shape[0]
        if points.shape != (m, 3) or normals.shape != (m, 3):
            raise HybridConfigurationError(
                "boundary points and normals must both have shape (m, 3)"
            )
        if weights.shape != (m,) or np.any(weights <= 0.0):
            raise HybridConfigurationError("boundary weights must be positive")
        lengths = np.linalg.norm(normals, axis=1)
        if not np.allclose(lengths, 1.0, rtol=1e-7, atol=1e-10):
            raise HybridConfigurationError("boundary normals must be unit vectors")
        shape = (
            np.full((m, 3), 1.0 / 3.0)
            if self.shape_values is None
            else np.asarray(self.shape_values, dtype=float)
        )
        if shape.shape != (m, 3) or not np.allclose(shape.sum(axis=1), 1.0):
            raise HybridConfigurationError(
                "shape_values must have shape (m, 3) with rows summing to one"
            )
        object.__setattr__(self, "face_nodes", faces)
        object.__setattr__(self, "points_global", points)
        object.__setattr__(self, "normals_global", normals)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "shape_values", shape)


@dataclass(frozen=True)
class VolumeQuadrature:
    """Tet4 centroid quadrature plus constant Tet4 shape-function gradients."""

    element_nodes: Array
    points_global: Array
    weights: Array
    shape_values: Array
    shape_gradients: Array

    @classmethod
    def from_mesh(cls, mesh: Any) -> "VolumeQuadrature":
        nodes = np.asarray(mesh.node_xyz, dtype=float)
        elements = np.asarray(mesh.elems, dtype=int)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise HybridConfigurationError("mesh.node_xyz must have shape (n, 3)")
        if elements.ndim != 2 or elements.shape[1] != 4:
            raise HybridConfigurationError("mesh.elems must have shape (m, 4)")

        xyz = nodes[elements]
        centroids = xyz.mean(axis=1)
        volumes = np.abs(
            np.einsum(
                "ij,ij->i",
                xyz[:, 0] - xyz[:, 3],
                np.cross(xyz[:, 1] - xyz[:, 3], xyz[:, 2] - xyz[:, 3]),
            )
        ) / 6.0
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise HybridConfigurationError("mesh contains invalid tetrahedra")

        # M_e [1, x, y, z]^T coefficients.  Column i of inv(M_e) contains
        # coefficients of N_i; rows 1: are therefore grad(N_i).
        matrices = np.concatenate(
            (np.ones((elements.shape[0], 4, 1)), xyz), axis=2
        )
        try:
            inverse = np.linalg.inv(matrices)
        except np.linalg.LinAlgError as exc:
            raise HybridConfigurationError(
                "could not construct Tet4 shape gradients; degenerate element"
            ) from exc
        gradients = np.transpose(inverse[:, 1:, :], (0, 2, 1))

        return cls(
            element_nodes=elements,
            points_global=centroids,
            weights=volumes,
            shape_values=np.full((elements.shape[0], 4), 0.25),
            shape_gradients=gradients,
        )


@dataclass(frozen=True)
class LocalizationParameters:
    """Parameters of chi_l(x,t)=s_l(x) a_l(t).

    core_radius
        s=1 for points whose shortest distance to the finite scan segment is
        <= core_radius.
    transition_width
        s decreases smoothly from 1 to 0 over this distance.
    retention_age
        Time after line completion for which a=1.
    transfer_duration
        Duration over which a decreases smoothly from 1 to 0.
    """

    core_radius: float
    transition_width: float
    retention_age: float
    transfer_duration: float

    def __post_init__(self) -> None:
        values = {
            "core_radius": self.core_radius,
            "transition_width": self.transition_width,
            "retention_age": self.retention_age,
            "transfer_duration": self.transfer_duration,
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise HybridConfigurationError(f"{name} must be finite")
        if self.core_radius < 0.0:
            raise HybridConfigurationError("core_radius must be nonnegative")
        if self.transition_width <= 0.0:
            raise HybridConfigurationError("transition_width must be positive")
        if self.retention_age < 0.0:
            raise HybridConfigurationError("retention_age must be nonnegative")
        if self.transfer_duration <= 0.0:
            raise HybridConfigurationError("transfer_duration must be positive")

    @property
    def support_radius(self) -> float:
        return self.core_radius + self.transition_width


class QuinticLineLocalization:
    """Quintic smootherstep spatial and temporal localization."""

    def __init__(self, parameters: LocalizationParameters) -> None:
        self.parameters = parameters

    @staticmethod
    def _decay(xi: Array | float) -> Array | float:
        # 1 -> 0 on xi in [0,1].
        return 1.0 - 10.0 * np.asarray(xi) ** 3 + 15.0 * np.asarray(xi) ** 4 - 6.0 * np.asarray(xi) ** 5

    @staticmethod
    def _decay_derivative(xi: Array | float) -> Array | float:
        return -30.0 * np.asarray(xi) ** 2 + 60.0 * np.asarray(xi) ** 3 - 30.0 * np.asarray(xi) ** 4

    def temporal(self, time_now: float, line_end_time: float) -> tuple[float, float]:
        """Return a(t) and da/dt for one completed/active line."""
        p = self.parameters
        transfer_start = float(line_end_time) + p.retention_age
        transfer_end = transfer_start + p.transfer_duration
        t = float(time_now)
        if t <= transfer_start:
            return 1.0, 0.0
        if t >= transfer_end:
            return 0.0, 0.0
        xi = (t - transfer_start) / p.transfer_duration
        a = float(self._decay(xi))
        adot = float(self._decay_derivative(xi) / p.transfer_duration)
        return a, adot

    def spatial(
        self,
        points: Array,
        line_start: Sequence[float],
        line_end: Sequence[float],
    ) -> tuple[Array, Array]:
        """Return s(x) and grad(s) for shortest distance to a finite segment."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise HybridConfigurationError("localization points must have shape (n,3)")
        start = np.asarray(line_start, dtype=float)
        end = np.asarray(line_end, dtype=float)
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 0.0:
            raise HybridConfigurationError("scan line must have positive length")

        relative = points - start
        u = np.clip((relative @ segment) / length_sq, 0.0, 1.0)
        closest = start + u[:, None] * segment
        offset = points - closest
        distance = np.linalg.norm(offset, axis=1)

        p = self.parameters
        s = np.ones(points.shape[0], dtype=float)
        grad_s = np.zeros((points.shape[0], 3), dtype=float)

        outside = distance >= p.support_radius
        transition = (distance > p.core_radius) & (~outside)
        s[outside] = 0.0
        if np.any(transition):
            d = distance[transition]
            xi = (d - p.core_radius) / p.transition_width
            s[transition] = self._decay(xi)
            ds_dd = self._decay_derivative(xi) / p.transition_width
            # Transition points have d > core_radius.  For core_radius=0,
            # d is still strictly positive because transition uses d>0.
            grad_distance = offset[transition] / d[:, None]
            grad_s[transition] = ds_dd[:, None] * grad_distance
        return s, grad_s


class VectorizedEagarTsaiAdapter:
    """Map analyzer ScanLine IDs to the supplied VectorizedEagarTsai engine."""

    def __init__(
        self,
        engine: Any,
        scan_analysis: Any,
        *,
        surface_normal: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> None:
        if not hasattr(engine, "evaluate"):
            raise HybridConfigurationError("analytical engine must expose evaluate")
        module = inspect.getmodule(type(engine))
        segment_type = getattr(module, "ScanSegment", None) if module else None
        if segment_type is None or not hasattr(segment_type, "from_endpoints"):
            raise HybridConfigurationError(
                "could not find ScanSegment.from_endpoints beside analytical engine"
            )
        normal = np.asarray(surface_normal, dtype=float)
        self.engine = engine
        self.lines: dict[int, Any] = {}
        self.segments: dict[int, Any] = {}
        for line in tuple(scan_analysis.lines):
            line_id = int(line.line_id)
            if line_id in self.lines:
                raise HybridConfigurationError(f"duplicate scan line ID {line_id}")
            self.lines[line_id] = line
            self.segments[line_id] = segment_type.from_endpoints(
                start=line.start_point,
                end=line.end_point,
                start_time=float(line.start_time),
                end_time=float(line.end_time),
                speed=float(line.velocity),
                surface_normal=normal,
            )
        self.line_ids = tuple(sorted(self.lines))
        self._start_times = np.asarray(
            [float(self.lines[i].start_time) for i in self.line_ids], dtype=float
        )
        self._end_times = np.asarray(
            [float(self.lines[i].end_time) for i in self.line_ids], dtype=float
        )
        self._expiration_times_cache: dict[float, Array] = {}
        if np.any(np.diff(self._start_times) < 0.0) or np.any(
            np.diff(self._end_times) < 0.0
        ):
            raise HybridConfigurationError(
                "scan lines must be ordered chronologically"
            )
        self._prepared_all = engine.prepare_lines(
            tuple(self.segments[i] for i in self.line_ids)
        )
        self._id_to_position = {line_id: j for j, line_id in enumerate(self.line_ids)}
        self._prepared_cache: dict[tuple[int, ...], Any] = {
            self.line_ids: self._prepared_all
        }

    def started_line_ids(self, time_now: float) -> tuple[int, ...]:
        t = float(time_now)
        count = int(np.searchsorted(self._start_times, t, side="left"))
        return self.line_ids[:count]

    def retained_line_ids(
        self, time_now: float, retention_horizon: float
    ) -> tuple[tuple[int, ...], int, int]:
        """Return retained IDs, number started, and number expired."""
        t = float(time_now)
        horizon = float(retention_horizon)
        if not np.isfinite(horizon) or horizon < 0.0:
            raise HybridConfigurationError(
                "retention_horizon must be finite and nonnegative"
            )
        started_count = int(np.searchsorted(self._start_times, t, side="left"))
        expiration_times = self._expiration_times_cache.get(horizon)
        if expiration_times is None:
            expiration_times = self._end_times + horizon
            self._expiration_times_cache[horizon] = expiration_times
        expired_count = int(np.searchsorted(expiration_times, t, side="right"))
        expired_count = min(expired_count, started_count)
        return (
            self.line_ids[expired_count:started_count],
            started_count,
            expired_count,
        )

    def _prepared_subset(self, line_ids: Sequence[int]) -> Any:
        ids = tuple(int(i) for i in line_ids)
        cached = self._prepared_cache.get(ids)
        if cached is not None:
            return cached
        try:
            positions = np.asarray(
                [self._id_to_position[i] for i in ids], dtype=int
            )
        except KeyError as exc:
            raise HybridConfigurationError(
                f"analytical evaluation requested unknown line ID {exc.args[0]}"
            ) from exc
        cls = type(self._prepared_all)
        prepared = cls(
            start=self._prepared_all.start[positions],
            rotation=self._prepared_all.rotation[positions],
            speed=self._prepared_all.speed[positions],
            start_time=self._prepared_all.start_time[positions],
            end_time=self._prepared_all.end_time[positions],
        )
        self._prepared_cache[ids] = prepared
        return prepared

    def evaluate_many(
        self,
        line_ids: Sequence[int],
        points: Array,
        time_now: float,
        *,
        need_gradient: bool = True,
    ) -> tuple[Array, Array | None]:
        ids = tuple(int(i) for i in line_ids)
        points = np.asarray(points, dtype=float)
        if not ids:
            return (
                np.zeros(points.shape[0], dtype=float),
                np.zeros((points.shape[0], 3), dtype=float) if need_gradient else None,
            )
        result = self.engine.evaluate(
            self._prepared_subset(ids),
            points,
            float(time_now),
            need_gradient=need_gradient,
        )
        gradient = (
            None
            if result.gradient is None
            else np.asarray(result.gradient, dtype=float)
        )
        return np.asarray(result.temperature, dtype=float), gradient

    def evaluate_by_line(
        self,
        line_ids: Sequence[int],
        points: Array,
        time_now: float,
        *,
        need_gradient: bool = True,
    ) -> tuple[Array, Array | None]:
        """Return a separate field for each line using the vectorized kernel."""
        ids = tuple(int(i) for i in line_ids)
        points = np.asarray(points, dtype=float)
        if not ids:
            return (
                np.zeros((0, points.shape[0]), dtype=float),
                (
                    np.zeros((0, points.shape[0], 3), dtype=float)
                    if need_gradient
                    else None
                ),
            )
        method = getattr(self.engine, "evaluate_by_line", None)
        if not callable(method):
            raise HybridConfigurationError(
                "VectorizedEagarTsai must provide evaluate_by_line(); use the "
                "optimized vectorized_eagar_tsai.py supplied with this solver"
            )
        temperature, gradient = method(
            self._prepared_subset(ids),
            points,
            float(time_now),
            need_gradient=need_gradient,
        )
        return np.asarray(temperature, dtype=float), (
            None if gradient is None else np.asarray(gradient, dtype=float)
        )

    def evaluate_one(
        self,
        line_id: int,
        points: Array,
        time_now: float,
        *,
        need_gradient: bool = True,
    ) -> tuple[Array, Array | None]:
        return self.evaluate_many(
            (int(line_id),), points, time_now, need_gradient=need_gradient
        )


@dataclass(frozen=True)
class LineLocalizationCache:
    """Sparse, time-independent localization data for one scan line.

    The expensive point-to-segment distances and spatial smootherstep values
    are computed once, on first use, then reused at every later time step.
    Only points inside the compact support are stored.
    """

    node_indices: Array
    node_s: Array
    boundary_indices: Array
    boundary_s: Array
    volume_indices: Array
    volume_s: Array
    volume_grad_s: Array
    volume_transition_positions: Array


@dataclass
class StepTiming:
    localization: float = 0.0
    analytical: float = 0.0
    residual_assembly: float = 0.0
    boundary_assembly: float = 0.0
    fea: float = 0.0


@dataclass
class EvaluationSnapshot:
    time: float
    started_line_count: int
    explicitly_retained_line_count: int
    transferred_line_count: int
    analytical_node_point_evaluations: int
    analytical_boundary_point_evaluations: int
    analytical_volume_point_evaluations: int
    discarded_line_count: int = 0


@dataclass
class HybridRunResult:
    mode: HybridMode
    times: Array
    saved_steps: Array
    saved_times: Array
    node_coordinates: Array
    total_temperature: Array
    analytical_temperature: Array
    correction_temperature: Array
    timing_history: list[StepTiming]
    evaluation_history: list[EvaluationSnapshot]
    wall_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


class _HybridPyTOFEA(TetTransientThermalFEA):
    """PyTO assembly with initialization for implicit backward Euler.

    Mirrors the data setup used by the pinned PyTO constructor, omitting its
    explicit-style CFL restriction and interactive prompt. Assembly is inherited.
    """

    def __init__(self, mesh, mat_prop, bc, deltaTime, solver):
        self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
        self.solver = solver
        self.deltaTime = float(deltaTime)
        self.initial_temp = np.zeros(len(mesh.node_xyz), dtype=float)
        self.edofMat = np.asarray(mesh.elems[:, :4], dtype=int)
        self.node_idx = np.stack((
            np.kron(self.edofMat, np.ones((4, 1))).flatten(),
            np.kron(self.edofMat, np.ones((1, 4))).flatten(),
        )).T.astype(int)
        self.num_dofs = bc.num_dofs


class _DefaultPyTOStepper:
    """Existing backward-Euler hybrid update using unmodified PyTO assembly/solve."""

    def __init__(self, solver: Any) -> None:
        self.solver = solver

    def advance(self, previous_correction: Array, force: Array,
                analytical_temperature: Array) -> Array:
        solver = self.solver
        previous = np.asarray(previous_correction, dtype=float).reshape(-1)
        force = np.asarray(force, dtype=float).reshape(-1)
        analytical = np.asarray(analytical_temperature, dtype=float).reshape(-1)
        for name, values in (("previous correction", previous), ("force", force),
                             ("analytical temperature", analytical)):
            if values.shape != (solver.num_dofs,) or not np.all(np.isfinite(values)):
                raise HybridConfigurationError(f"Invalid {name} vector")
        if not hasattr(solver, "K_mtrx") or not hasattr(solver, "C_mtrx"):
            solver.assemble_global_stiffness_matrices()
        # Exactly the beta=1 calculation in the original advance_one_step.
        capacity_over_dt = solver.C_mtrx / solver.deltaTime
        effective = solver.K_mtrx + capacity_over_dt
        rhs = capacity_over_dt @ previous + force
        original = solver.bc.dirichlet_values.copy()
        fixed = np.asarray(solver.bc.fixed_dofs, dtype=int)
        solver.bc.dirichlet_values = original - analytical[fixed]
        try:
            result = linear_solvers.solve(effective, rhs, solver.solver, solver.bc)
        finally:
            solver.bc.dirichlet_values = original
        result = np.asarray(result, dtype=float)
        if result.shape != (solver.num_dofs,) or not np.all(np.isfinite(result)):
            raise HybridConfigurationError("FE solve returned an invalid temperature field")
        return result


class HybridSimulation:
    """Run a conventional, localized, or hard temporal-truncation simulation."""

    def __init__(
        self,
        *,
        mode: HybridMode | str,
        mesh: Any,
        mat_prop: Any,
        bc: Any,
        delta_time: float,
        total_time: float,
        boundary_quadrature: BoundaryQuadrature,
        analytical_engine: Any,
        scan_analysis: Any,
        localization: LocalizationParameters | None = None,
        truncation_age: float | None = None,
        stepper_factory: Any | None = None,
        initial_temperature_c: float = 32.0,
        solver_name: str = "PARDISO",
        save_every: int = 1,
        progress_percent_interval: int = 5,
    ) -> None:
        self.mode = mode if isinstance(mode, HybridMode) else HybridMode(mode)
        self.mesh = mesh
        self.mat_prop = mat_prop
        self.bc = bc
        self.delta_time = float(delta_time)
        self.total_time = float(total_time)
        self.boundary = boundary_quadrature
        self.volume = (
            VolumeQuadrature.from_mesh(mesh)
            if self.mode is HybridMode.LOCALIZED
            else None
        )
        self.analytical = VectorizedEagarTsaiAdapter(analytical_engine, scan_analysis)
        self.truncation_age = None if truncation_age is None else float(truncation_age)
        if self.mode is HybridMode.TEMPORAL_TRUNCATION and (
            self.truncation_age is None
            or not np.isfinite(self.truncation_age)
            or self.truncation_age < 0.0
        ):
            raise HybridConfigurationError(
                "temporal_truncation requires a finite nonnegative truncation_age"
            )
        self.localization_parameters = localization
        self.localization = (
            QuinticLineLocalization(localization) if localization is not None else None
        )
        self.stepper_factory = stepper_factory or _DefaultPyTOStepper
        self.save_every = int(save_every)
        self.progress_percent_interval = int(progress_percent_interval)

        self.nodes = np.asarray(mesh.node_xyz, dtype=float)
        self.elements = np.asarray(mesh.elems, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.rho = float(mat_prop.mass_density)
        self.cp = float(mat_prop.specific_heat)
        self.k = float(mat_prop.thermal_conductivity)

        # Static point arrays reused at every time step.
        self._boundary_volume_points = (
            np.vstack((self.boundary.points_global, self.volume.points_global))
            if self.volume is not None
            else None
        )
        self._n_boundary_points = int(self.boundary.points_global.shape[0])
        self._line_localization_cache: dict[int, LineLocalizationCache] = {}
        self._expired_line_count = 0
        self._localized_line_batch_size = max(
            1, int(getattr(self.analytical.engine, "line_chunk_size", 1))
        )

        if self.mode is HybridMode.LOCALIZED and self.localization is None:
            raise HybridConfigurationError(
                "localized mode requires LocalizationParameters"
            )
        if (not np.isfinite(self.delta_time) or not np.isfinite(self.total_time)
                or self.delta_time <= 0.0 or self.total_time < 0.0):
            raise HybridConfigurationError("invalid time configuration")
        n_float = self.total_time / self.delta_time
        self.n_steps = int(round(n_float))
        if not np.isclose(n_float, self.n_steps, rtol=1e-10, atol=1e-12):
            raise HybridConfigurationError(
                "total_time must be an integer multiple of delta_time"
            )
        if self.save_every <= 0:
            raise HybridConfigurationError("save_every must be positive")
        if not 1 <= self.progress_percent_interval <= 100:
            raise HybridConfigurationError(
                "progress_percent_interval must lie in [1,100]"
            )

        if not np.isfinite(initial_temperature_c):
            raise HybridConfigurationError("initial_temperature_c must be finite")
        self.initial_temperature_c = float(initial_temperature_c)
        # The FE state remains relative to the uniform initial temperature.
        self.initial_temperature_rise = np.zeros(self.n_nodes, dtype=float)
        try:
            self.solver_kind = linear_solvers.Solvers[solver_name.upper()]
        except KeyError as exc:
            raise HybridConfigurationError("solver_name must be PARDISO, SPSOLVE or SPLU") from exc
        if self.solver_kind.name not in {"PARDISO", "SPSOLVE", "SPLU"}:
            raise HybridConfigurationError("Supported solvers: PARDISO, SPSOLVE, SPLU")
        if self.solver_kind.name == "PARDISO" and linear_solvers.pypardiso is None:
            raise HybridConfigurationError("PARDISO requested but pypardiso is unavailable. "
                                           "Install it or explicitly choose SPSOLVE/SPLU.")

        force = np.asarray(getattr(bc, "force"), dtype=float)
        if force.shape != (self.n_nodes,):
            raise HybridConfigurationError("bc.force has wrong shape")
        self.base_force = force.copy()

        fixed = np.asarray(getattr(bc, "fixed_dofs"), dtype=int)
        fixed_values = np.asarray(getattr(bc, "dirichlet_values"), dtype=float)
        if fixed.ndim != 1 or fixed_values.shape != fixed.shape:
            raise HybridConfigurationError("invalid Dirichlet BC arrays")
        if not np.all(np.isfinite(fixed_values)):
            raise HybridConfigurationError("Dirichlet values must be finite")

    def _make_stepper(self) -> Any:
        solver = _HybridPyTOFEA(
            mesh=self.mesh, mat_prop=self.mat_prop, bc=copy.deepcopy(self.bc),
            deltaTime=self.delta_time, solver=self.solver_kind,
        )
        return self.stepper_factory(solver)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(float(seconds))))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def run(self) -> HybridRunResult:
        stepper = self._make_stepper()
        wall_start = time.perf_counter()
        label = {
            HybridMode.CONVENTIONAL: "Conventional hybrid",
            HybridMode.LOCALIZED: "Localized hybrid",
            HybridMode.TEMPORAL_TRUNCATION: "Temporal truncation hybrid",
        }[self.mode]
        print(
            f"[{label}]   0.0% | step 0/{self.n_steps} | elapsed=00:00:00",
            flush=True,
        )

        times = np.arange(self.n_steps + 1, dtype=float) * self.delta_time
        saved_steps = np.unique(
            np.r_[np.arange(0, self.n_steps + 1, self.save_every), self.n_steps]
        ).astype(int)
        save_lookup = {int(step): i for i, step in enumerate(saved_steps)}
        n_saved = saved_steps.size

        total_saved = np.empty((n_saved, self.n_nodes), dtype=float)
        analytical_saved = np.empty_like(total_saved)
        correction_saved = np.empty_like(total_saved)
        timing_history: list[StepTiming] = []
        evaluation_history: list[EvaluationSnapshot] = []

        # At t=0 no Eagar--Tsai segment contributes because the evaluator uses
        # current_time > start_time.  Keep the initialization explicit.
        analytical_nodes = np.zeros(self.n_nodes, dtype=float)
        correction = self.initial_temperature_rise.copy()
        total_saved[0] = self.initial_temperature_c + analytical_nodes + correction
        analytical_saved[0] = analytical_nodes
        correction_saved[0] = self.initial_temperature_c + correction
        timing_history.append(StepTiming())
        evaluation_history.append(
            EvaluationSnapshot(0.0, 0, 0, 0, 0, 0, 0)
        )

        next_progress = self.progress_percent_interval
        for step in range(1, self.n_steps + 1):
            t_now = float(times[step])
            timing = StepTiming()

            if self.mode in {HybridMode.CONVENTIONAL, HybridMode.TEMPORAL_TRUNCATION}:
                (
                    analytical_nodes,
                    force,
                    evaluation,
                    local_timing,
                ) = self._conventional_step_fields(t_now)
            else:
                (
                    analytical_nodes,
                    force,
                    evaluation,
                    local_timing,
                ) = self._localized_step_fields(t_now)

            timing.localization = local_timing.localization
            timing.analytical = local_timing.analytical
            timing.residual_assembly = local_timing.residual_assembly
            timing.boundary_assembly = local_timing.boundary_assembly

            tic = time.perf_counter()
            correction = stepper.advance(correction, force, analytical_nodes)
            timing.fea = time.perf_counter() - tic
            if correction.shape != (self.n_nodes,):
                raise HybridConfigurationError("FEA step returned wrong field shape")
            total = self.initial_temperature_c + analytical_nodes + correction

            if step in save_lookup:
                j = save_lookup[step]
                total_saved[j] = total
                analytical_saved[j] = analytical_nodes
                correction_saved[j] = self.initial_temperature_c + correction

            timing_history.append(timing)
            evaluation_history.append(evaluation)

            percentage = 100.0 * step / self.n_steps if self.n_steps else 100.0
            if percentage >= next_progress or step == self.n_steps:
                elapsed = time.perf_counter() - wall_start
                print(
                    f"[{label}] {percentage:5.1f}% | step {step}/{self.n_steps} | "
                    f"sim t={t_now:.6f} s | elapsed={self._format_duration(elapsed)}",
                    flush=True,
                )
                while next_progress <= percentage:
                    next_progress += self.progress_percent_interval

        return HybridRunResult(
            mode=self.mode,
            times=times,
            saved_steps=saved_steps,
            saved_times=times[saved_steps].copy(),
            node_coordinates=self.nodes.copy(),
            total_temperature=total_saved,
            analytical_temperature=analytical_saved,
            correction_temperature=correction_saved,
            timing_history=timing_history,
            evaluation_history=evaluation_history,
            wall_time=time.perf_counter() - wall_start,
            metadata={
                "result_format_version": 2,
                "temperature": "total temperature",
                "initial_temperature_celsius": self.initial_temperature_c,
                "analytical_temperature_basis": "source-induced increment",
                "correction_temperature_basis": "FE correction including uniform initial temperature",
                "relative_error_basis": "physical Celsius temperature",
                "time_integrator": "backward Euler (beta=1)",
                "solver": self.solver_kind.name,
                "environment": environment_metadata(),
                "temperature_unit": "degree Celsius",
                "length_unit": "m",
                "time_unit": "s",
                "delta_time": self.delta_time,
                "total_time": self.total_time,
                "save_every": self.save_every,
                "conventional_definition": (
                    "every started line evaluated at all nodes and Neumann boundary points at every time; no expiration"
                ),
                "truncation_age": self.truncation_age,
                "temporal_truncation_definition": (
                    "started lines retained while t < line_end + truncation_age; "
                    "full spatial evaluation and ordinary boundary correction; "
                    "hard deletion without localization residual or FE transfer"
                ),
                "localized_definition": (
                    "line-wise chi=s(x)*a(t), quintic spatial and temporal windows, weak residual assembly"
                ),
                "localization_parameters": (
                    None
                    if self.localization_parameters is None
                    else {
                        "core_radius": self.localization_parameters.core_radius,
                        "transition_width": self.localization_parameters.transition_width,
                        "retention_age": self.localization_parameters.retention_age,
                        "transfer_duration": self.localization_parameters.transfer_duration,
                    }
                ),
                "volume_quadrature": (
                    "one centroid point per Tet4"
                    if self.volume is not None
                    else None
                ),
                "localized_implementation": (
                    "cached sparse spatial windows; line-batched Eagar--Tsai; "
                    "node temperature batching; merged boundary/volume gradient batching"
                ),
            },
        )

    def _conventional_step_fields(
        self, time_now: float
    ) -> tuple[Array, Array, EvaluationSnapshot, StepTiming]:
        timing = StepTiming()
        started = self.analytical.started_line_ids(time_now)
        retained = started
        discarded = 0
        if self.mode is HybridMode.TEMPORAL_TRUNCATION:
            retained, _, discarded = self.analytical.retained_line_ids(
                time_now, self.truncation_age
            )

        tic = time.perf_counter()
        theta_nodes, _ = self.analytical.evaluate_many(
            retained, self.nodes, time_now, need_gradient=False
        )
        _, grad_boundary = self.analytical.evaluate_many(
            retained,
            self.boundary.points_global,
            time_now,
            need_gradient=True,
        )
        timing.analytical = time.perf_counter() - tic
        assert grad_boundary is not None

        tic = time.perf_counter()
        boundary_force = self._assemble_boundary_force_from_gradient(grad_boundary)
        timing.boundary_assembly = time.perf_counter() - tic
        force = self.base_force + boundary_force

        n_started = len(started)
        evaluation = EvaluationSnapshot(
            time=float(time_now),
            started_line_count=n_started,
            explicitly_retained_line_count=len(retained),
            transferred_line_count=0,
            discarded_line_count=discarded,
            analytical_node_point_evaluations=len(retained) * self.n_nodes,
            analytical_boundary_point_evaluations=(
                len(retained) * self.boundary.points_global.shape[0]
            ),
            analytical_volume_point_evaluations=0,
        )
        return theta_nodes, force, evaluation, timing

    def _get_line_localization_cache(
        self, line_id: int
    ) -> LineLocalizationCache:
        """Compute static spatial localization once for one scan line."""
        cached = self._line_localization_cache.get(int(line_id))
        if cached is not None:
            return cached
        assert self.localization is not None
        line = self.analytical.lines[int(line_id)]

        s_nodes, _ = self.localization.spatial(
            self.nodes, line.start_point, line.end_point
        )
        node_indices = np.flatnonzero(s_nodes > 0.0)

        s_boundary, _ = self.localization.spatial(
            self.boundary.points_global, line.start_point, line.end_point
        )
        boundary_indices = np.flatnonzero(s_boundary > 0.0)

        s_volume, grad_s_volume = self.localization.spatial(
            self.volume.points_global, line.start_point, line.end_point
        )
        volume_indices = np.flatnonzero(s_volume > 0.0)
        volume_s = s_volume[volume_indices]
        volume_grad_s = grad_s_volume[volume_indices]
        transition_positions = np.flatnonzero(
            np.linalg.norm(volume_grad_s, axis=1) > 0.0
        )

        cached = LineLocalizationCache(
            node_indices=node_indices,
            node_s=s_nodes[node_indices],
            boundary_indices=boundary_indices,
            boundary_s=s_boundary[boundary_indices],
            volume_indices=volume_indices,
            volume_s=volume_s,
            volume_grad_s=volume_grad_s,
            volume_transition_positions=transition_positions,
        )
        self._line_localization_cache[int(line_id)] = cached
        return cached

    @staticmethod
    def _union_sorted(arrays: Sequence[Array]) -> Array:
        nonempty = [np.asarray(a, dtype=int) for a in arrays if len(a)]
        if not nonempty:
            return np.empty(0, dtype=int)
        if len(nonempty) == 1:
            return np.unique(nonempty[0])
        return np.unique(np.concatenate(nonempty))

    def _localized_step_fields(
        self, time_now: float
    ) -> tuple[Array, Array, EvaluationSnapshot, StepTiming]:
        """Optimized residual-consistent localized field assembly.

        Optimizations are purely implementation-level:
        * spatial windows/masks are cached once per line;
        * retained lines are processed in the analytical engine's existing
          ``line_chunk_size`` batches;
        * node requests are vectorized across a line batch;
        * boundary and volume requests are merged into one gradient call per
          line batch;
        * the vectorized Eagar--Tsai kernel keeps the line axis so each line's
          different s(x) and a(t) weights are applied after one batched call.
        """
        assert self.localization is not None
        assert self.volume is not None
        assert self._boundary_volume_points is not None
        timing = StepTiming()
        retention_horizon = (
            self.localization.parameters.retention_age
            + self.localization.parameters.transfer_duration
        )
        retained_ids, started_count, expired_count = (
            self.analytical.retained_line_ids(time_now, retention_horizon)
        )
        if expired_count > self._expired_line_count:
            for line_id in self.analytical.line_ids[
                self._expired_line_count : expired_count
            ]:
                self._line_localization_cache.pop(int(line_id), None)
            self._expired_line_count = expired_count
        theta_nodes = np.zeros(self.n_nodes, dtype=float)
        boundary_force = np.zeros(self.n_nodes, dtype=float)
        volume_force = np.zeros(self.n_nodes, dtype=float)

        retained_data: list[tuple[int, float, float, LineLocalizationCache]] = []
        for line_id in retained_ids:
            line = self.analytical.lines[line_id]
            a, adot = self.localization.temporal(time_now, float(line.end_time))
            if a == 0.0 and adot == 0.0:
                self._line_localization_cache.pop(int(line_id), None)
                continue
            tic = time.perf_counter()
            cache = self._get_line_localization_cache(line_id)
            timing.localization += time.perf_counter() - tic
            retained_data.append((line_id, a, adot, cache))

        node_evals = 0
        boundary_evals = 0
        volume_evals = 0
        batch_size = self._localized_line_batch_size

        for batch_start in range(0, len(retained_data), batch_size):
            batch = retained_data[batch_start : batch_start + batch_size]
            batch_ids = tuple(item[0] for item in batch)

            # ----------------------------------------------------------
            # 1) Nodal temperature: one temperature-only ET call for the
            #    union of all nodal supports in this line batch.
            # ----------------------------------------------------------
            node_union = self._union_sorted(
                [item[3].node_indices for item in batch]
            )
            if node_union.size:
                tic = time.perf_counter()
                t_by_line, _ = self.analytical.evaluate_by_line(
                    batch_ids,
                    self.nodes[node_union],
                    time_now,
                    need_gradient=False,
                )
                timing.analytical += time.perf_counter() - tic
                node_evals += len(batch) * int(node_union.size)
                for j, (_, a, _, cache) in enumerate(batch):
                    if cache.node_indices.size == 0:
                        continue
                    positions = np.searchsorted(node_union, cache.node_indices)
                    theta_nodes[cache.node_indices] += (
                        a * cache.node_s * t_by_line[j, positions]
                    )

            # ----------------------------------------------------------
            # 2) Boundary + volume: merge both gradient-requiring point sets
            #    and evaluate all lines in this batch together.
            # ----------------------------------------------------------
            bv_ids_by_line: list[Array] = []
            volume_positions_by_line: list[Array] = []
            for _, a, adot, cache in batch:
                if abs(adot) > 0.0:
                    vpos = np.arange(cache.volume_indices.size, dtype=int)
                elif a != 0.0:
                    vpos = cache.volume_transition_positions
                else:
                    vpos = np.empty(0, dtype=int)
                volume_positions_by_line.append(vpos)
                vglobal = (
                    self._n_boundary_points + cache.volume_indices[vpos]
                    if vpos.size
                    else np.empty(0, dtype=int)
                )
                bv_ids_by_line.append(
                    np.concatenate((cache.boundary_indices, vglobal))
                    if cache.boundary_indices.size or vglobal.size
                    else np.empty(0, dtype=int)
                )

            bv_union = self._union_sorted(bv_ids_by_line)
            if bv_union.size:
                tic = time.perf_counter()
                t_by_line, g_by_line = self.analytical.evaluate_by_line(
                    batch_ids,
                    self._boundary_volume_points[bv_union],
                    time_now,
                    need_gradient=True,
                )
                timing.analytical += time.perf_counter() - tic
                assert g_by_line is not None
                boundary_union_size = int(
                    np.count_nonzero(bv_union < self._n_boundary_points)
                )
                volume_union_size = int(bv_union.size) - boundary_union_size
                boundary_evals += len(batch) * boundary_union_size
                volume_evals += len(batch) * volume_union_size

                for j, (_, a, adot, cache) in enumerate(batch):
                    # Boundary contribution.
                    if cache.boundary_indices.size:
                        bpos = np.searchsorted(bv_union, cache.boundary_indices)
                        tic = time.perf_counter()
                        self._add_localized_boundary_force(
                            boundary_force,
                            cache.boundary_indices,
                            a * cache.boundary_s,
                            g_by_line[j, bpos],
                        )
                        timing.boundary_assembly += time.perf_counter() - tic

                    # Weak-form volume residual.
                    vpos_local = volume_positions_by_line[j]
                    if vpos_local.size:
                        element_indices = cache.volume_indices[vpos_local]
                        global_ids = self._n_boundary_points + element_indices
                        upos = np.searchsorted(bv_union, global_ids)
                        tic = time.perf_counter()
                        self._add_localized_volume_residual(
                            volume_force,
                            element_indices,
                            cache.volume_s[vpos_local],
                            cache.volume_grad_s[vpos_local],
                            a,
                            adot,
                            t_by_line[j, upos],
                            g_by_line[j, upos],
                        )
                        timing.residual_assembly += time.perf_counter() - tic

        evaluation = EvaluationSnapshot(
            time=float(time_now),
            started_line_count=started_count,
            explicitly_retained_line_count=len(retained_data),
            transferred_line_count=expired_count,
            analytical_node_point_evaluations=node_evals,
            analytical_boundary_point_evaluations=boundary_evals,
            analytical_volume_point_evaluations=volume_evals,
        )
        return (
            theta_nodes,
            self.base_force + boundary_force + volume_force,
            evaluation,
            timing,
        )

    def _assemble_boundary_force_from_gradient(self, analytical_gradient: Array) -> Array:
        if analytical_gradient.shape != (self.boundary.face_nodes.shape[0], 3):
            raise HybridConfigurationError("boundary gradient has wrong shape")
        force = np.zeros(self.n_nodes, dtype=float)
        normal_derivative = np.einsum(
            "ij,ij->i", analytical_gradient, self.boundary.normals_global
        )
        face_flux = -self.k * normal_derivative
        contributions = (
            face_flux[:, None]
            * self.boundary.weights[:, None]
            * self.boundary.shape_values
        )
        np.add.at(force, self.boundary.face_nodes.ravel(), contributions.ravel())
        return force

    def _add_localized_boundary_force(
        self,
        force: Array,
        boundary_indices: Array,
        chi: Array,
        analytical_gradient: Array,
    ) -> None:
        """Add -k chi grad(T_l).n to the Neumann load.

        The T_l grad(chi).n part cancels against the boundary term generated
        when the spatial residual is integrated by parts.  This is why only
        chi*grad(T_l) appears here.
        """
        normals = self.boundary.normals_global[boundary_indices]
        normal_derivative = np.einsum("ij,ij->i", analytical_gradient, normals)
        face_flux = -self.k * chi * normal_derivative
        contributions = (
            face_flux[:, None]
            * self.boundary.weights[boundary_indices, None]
            * self.boundary.shape_values[boundary_indices]
        )
        faces = self.boundary.face_nodes[boundary_indices]
        np.add.at(force, faces.ravel(), contributions.ravel())

    def _add_localized_volume_residual(
        self,
        force: Array,
        element_indices: Array,
        s: Array,
        grad_s: Array,
        a: float,
        adot: float,
        temperature: Array,
        analytical_gradient: Array,
    ) -> None:
        """Assemble the line-wise weak localization residual on Tet4 centroids."""
        weights = self.volume.weights[element_indices]
        shape_values = self.volume.shape_values[element_indices]
        shape_gradients = self.volume.shape_gradients[element_indices]
        elements = self.volume.element_nodes[element_indices]

        scalar = (
            -self.rho * self.cp * s * adot * temperature
            + self.k * a * np.einsum("ij,ij->i", grad_s, analytical_gradient)
        )
        first = weights[:, None] * scalar[:, None] * shape_values

        grad_dot = np.einsum("eij,ej->ei", shape_gradients, grad_s)
        second = -weights[:, None] * self.k * a * temperature[:, None] * grad_dot
        contributions = first + second
        np.add.at(force, elements.ravel(), contributions.ravel())


__all__ = [
    "BoundaryQuadrature",
    "EvaluationSnapshot",
    "HybridConfigurationError",
    "HybridMode",
    "HybridRunResult",
    "HybridSimulation",
    "LineLocalizationCache",
    "LocalizationParameters",
    "QuinticLineLocalization",
    "StepTiming",
    "VolumeQuadrature",
]

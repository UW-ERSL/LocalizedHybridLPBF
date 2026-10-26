"""Vectorized Eagar--Tsai temperature-rise and gradient evaluation.

This module is intentionally independent of PyTO, meshing, FEA, scan-path
partitioning, surrogate inference, and physical cutoff policies.  Both the
future full-analytical solver and the analytical fallback in the hybrid solver
should call this same evaluator.

The implementation evaluates a finite scan segment directly over its physical
exposure interval.  It does not use the older semi-infinite extend/subtract
construction.  With ``s = t - t'`` and ``r = sqrt(s)``, the apparent
``s**(-1/2)`` endpoint singularity is removed before Gauss--Legendre
quadrature is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class EagarTsaiParameters:
    """Constant material and Gaussian-source parameters in SI units.

    Parameters
    ----------
    thermal_conductivity:
        ``k`` in W/(m K).
    density:
        ``rho`` in kg/m^3.
    specific_heat:
        ``c_p`` in J/(kg K).
    sigma:
        Gaussian standard deviation in m.
    q_m:
        Absorbed laser power in W as it appears in Eq. (17):
        ``q_m = absorptivity * laser_power``.
    """

    thermal_conductivity: float
    density: float
    specific_heat: float
    sigma: float
    q_m: float

    def __post_init__(self) -> None:
        values = {
            "thermal_conductivity": self.thermal_conductivity,
            "density": self.density,
            "specific_heat": self.specific_heat,
            "sigma": self.sigma,
            "q_m": self.q_m,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value!r}")

    @property
    def diffusivity(self) -> float:
        return self.thermal_conductivity / (self.density * self.specific_heat)

    @property
    def prefactor(self) -> float:
        alpha = self.diffusivity
        return self.q_m / (
            np.pi * self.density * self.specific_heat * np.sqrt(4.0 * np.pi * alpha)
        )

    @classmethod
    def from_absorbed_power(
        cls,
        *,
        thermal_conductivity: float,
        density: float,
        specific_heat: float,
        sigma: float,
        laser_power: float,
        absorptivity: float,
    ) -> "EagarTsaiParameters":
        """Construct parameters using ``q_m = absorptivity * laser_power``."""

        if not np.isfinite(laser_power) or laser_power <= 0.0:
            raise ValueError("laser_power must be finite and positive")
        if not np.isfinite(absorptivity) or not 0.0 < absorptivity <= 1.0:
            raise ValueError("absorptivity must lie in (0, 1]")
        q_m = absorptivity * laser_power
        return cls(
            thermal_conductivity=thermal_conductivity,
            density=density,
            specific_heat=specific_heat,
            sigma=sigma,
            q_m=q_m,
        )


@dataclass(frozen=True)
class ScanSegment:
    """One straight, constant-speed laser exposure segment.

    ``rotation_global_to_local`` has local basis vectors as its rows, so

    ``x_local = R @ (x_global - start)``.

    Consequently, the first row must equal the scan direction.  The third row
    is the local surface-normal/depth direction used by the semi-infinite
    Eagar--Tsai solution.
    """

    start: np.ndarray
    end: np.ndarray
    start_time: float
    end_time: float
    speed: float
    rotation_global_to_local: np.ndarray

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=np.float64)
        end = np.asarray(self.end, dtype=np.float64)
        rotation = np.asarray(self.rotation_global_to_local, dtype=np.float64)

        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("start and end must each have shape (3,)")
        if rotation.shape != (3, 3):
            raise ValueError("rotation_global_to_local must have shape (3, 3)")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ValueError("segment coordinates must be finite")
        if not np.isfinite(self.start_time) or not np.isfinite(self.end_time):
            raise ValueError("segment times must be finite")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if not np.isfinite(self.speed) or self.speed <= 0.0:
            raise ValueError("speed must be finite and positive")

        length = float(np.linalg.norm(end - start))
        if length <= 0.0:
            raise ValueError("scan segment must have positive length")
        expected_length = self.speed * (self.end_time - self.start_time)
        tolerance = 1.0e-8 + 1.0e-5 * max(length, expected_length)
        if abs(length - expected_length) > tolerance:
            raise ValueError(
                "segment geometry and timing are inconsistent: "
                f"length={length:.12g}, speed*duration={expected_length:.12g}"
            )

        identity = rotation @ rotation.T
        if not np.allclose(identity, np.eye(3), rtol=1.0e-7, atol=1.0e-9):
            raise ValueError("rotation_global_to_local must be orthonormal")
        direction = (end - start) / length
        if not np.allclose(rotation[0], direction, rtol=1.0e-7, atol=1.0e-9):
            raise ValueError("the first row of the rotation must equal the scan direction")

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "rotation_global_to_local", rotation)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @classmethod
    def from_endpoints(
        cls,
        *,
        start: ArrayLike,
        end: ArrayLike,
        start_time: float,
        end_time: float,
        speed: Optional[float] = None,
        surface_normal: ArrayLike = (0.0, 0.0, 1.0),
    ) -> "ScanSegment":
        """Build the local frame from endpoints and a surface normal.

        The scan direction must be tangent to the supplied surface normal.
        For the usual horizontal LPBF surface, use the default global ``+z``
        normal.  Supplying a complete rotation directly is preferable if a
        future geometry uses a different local convention.
        """

        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        if start_array.shape != (3,) or end_array.shape != (3,):
            raise ValueError("start and end must each have shape (3,)")

        displacement = end_array - start_array
        length = float(np.linalg.norm(displacement))
        if length <= 0.0:
            raise ValueError("scan segment must have positive length")
        direction = displacement / length

        normal = np.asarray(surface_normal, dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("surface_normal must be a finite vector of shape (3,)")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("surface_normal must be nonzero")
        normal = normal / normal_norm

        normal_component = float(np.dot(direction, normal))
        if abs(normal_component) > 1.0e-7:
            raise ValueError("scan direction must be tangent to surface_normal")

        transverse = np.cross(normal, direction)
        transverse /= np.linalg.norm(transverse)
        rotation = np.stack((direction, transverse, normal), axis=0)

        duration = float(end_time - start_time)
        inferred_speed = length / duration if duration > 0.0 else np.nan
        return cls(
            start=start_array,
            end=end_array,
            start_time=float(start_time),
            end_time=float(end_time),
            speed=float(inferred_speed if speed is None else speed),
            rotation_global_to_local=rotation,
        )


@dataclass(frozen=True)
class EagarTsaiResult:
    """Superposed temperature rise and optional global spatial gradient."""

    temperature: np.ndarray
    gradient: Optional[np.ndarray]


@dataclass(frozen=True)
class PreparedScanSegments:
    """Validated structure-of-arrays representation prepared once per scan path."""

    start: np.ndarray
    rotation: np.ndarray
    speed: np.ndarray
    start_time: np.ndarray
    end_time: np.ndarray

    @property
    def number_of_lines(self) -> int:
        return int(self.start_time.size)


class VectorizedEagarTsai:
    """Chunked PyTorch-eager evaluator vectorized over lines, points, and quadrature.

    Notes
    -----
    * The returned temperature is a rise; no ambient ``T0`` is added.
    * No spatial or temporal physical cutoff is applied here.
    * Lines that have not started contribute zero.
    * Currently scanning lines are integrated only to the current beam time.
    * Completed lines are integrated over their complete finite exposure.
    """

    def __init__(
        self,
        parameters: EagarTsaiParameters,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Union[str, torch.dtype] = "float32",
        gauss_order: int = 64,
        point_chunk_size: int = 4096,
        line_chunk_size: int = 8,
    ) -> None:
        self.parameters = parameters
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        if gauss_order <= 0:
            raise ValueError("gauss_order must be positive")
        if point_chunk_size <= 0 or line_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")

        self.gauss_order = int(gauss_order)
        self.point_chunk_size = int(point_chunk_size)
        self.line_chunk_size = int(line_chunk_size)

        nodes, weights = np.polynomial.legendre.leggauss(self.gauss_order)
        self._gauss_nodes = torch.as_tensor(nodes, dtype=self.dtype, device=self.device)
        self._gauss_weights = torch.as_tensor(weights, dtype=self.dtype, device=self.device)
        self._alpha = torch.tensor(parameters.diffusivity, dtype=self.dtype, device=self.device)
        self._sigma_squared = torch.tensor(parameters.sigma**2, dtype=self.dtype, device=self.device)
        self._prefactor = torch.tensor(parameters.prefactor, dtype=self.dtype, device=self.device)

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return resolved

    @staticmethod
    def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
        if dtype in ("float32", np.float32, torch.float32):
            return torch.float32
        if dtype in ("float64", np.float64, torch.float64):
            return torch.float64
        raise ValueError("dtype must be float32 or float64")

    @staticmethod
    def _validate_points(points: ArrayLike) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("points must have shape (n_points, 3)")
        if not np.all(np.isfinite(array)):
            raise ValueError("points must contain only finite values")
        return array

    @staticmethod
    def prepare_lines(lines: Iterable[ScanSegment]) -> PreparedScanSegments:
        """Validate and pack a complete scan path for reuse at every time step."""

        materialized = tuple(lines)
        if not materialized:
            return PreparedScanSegments(
                start=np.empty((0, 3), dtype=np.float64),
                rotation=np.empty((0, 3, 3), dtype=np.float64),
                speed=np.empty(0, dtype=np.float64),
                start_time=np.empty(0, dtype=np.float64),
                end_time=np.empty(0, dtype=np.float64),
            )
        if not all(isinstance(line, ScanSegment) for line in materialized):
            raise TypeError("every line must be a ScanSegment")

        return PreparedScanSegments(
            start=np.stack([line.start for line in materialized]),
            rotation=np.stack(
                [line.rotation_global_to_local for line in materialized]
            ),
            speed=np.asarray([line.speed for line in materialized], dtype=np.float64),
            start_time=np.asarray(
                [line.start_time for line in materialized], dtype=np.float64
            ),
            end_time=np.asarray(
                [line.end_time for line in materialized], dtype=np.float64
            ),
        )

    @staticmethod
    def _active_line_arrays(
        lines: PreparedScanSegments, current_time: float
    ) -> Optional[dict[str, np.ndarray]]:
        if not np.isfinite(current_time):
            raise ValueError("current_time must be finite")

        active_mask = current_time > lines.start_time
        if not np.any(active_mask):
            return None

        active_indices = np.flatnonzero(active_mask)
        start_times = lines.start_time[active_indices]
        end_times = lines.end_time[active_indices]
        ages = current_time - start_times
        exposure_durations = np.minimum(current_time, end_times) - start_times
        positive = exposure_durations > 0.0
        if not np.any(positive):
            return None

        active_indices = active_indices[positive]
        ages = ages[positive]
        exposure_durations = exposure_durations[positive]

        # In lag coordinates s=t-t', the finite exposure maps to
        # s in [age-exposure_duration, age].  r=sqrt(s) removes s^(-1/2).
        lag_lower = np.maximum(ages - exposure_durations, 0.0)
        lag_upper = ages

        return {
            "start": lines.start[active_indices],
            "rotation": lines.rotation[active_indices],
            "speed": lines.speed[active_indices],
            "age": ages,
            "r_lower": np.sqrt(lag_lower),
            "r_upper": np.sqrt(lag_upper),
        }

    def evaluate(
        self,
        lines: Union[Iterable[ScanSegment], PreparedScanSegments],
        points: ArrayLike,
        current_time: float,
        *,
        need_gradient: bool = True,
    ) -> EagarTsaiResult:
        """Evaluate the superposed finite-line field at one global time."""

        points_array = self._validate_points(points)
        number_of_points = points_array.shape[0]
        temperature = np.zeros(number_of_points, dtype=np.float64)
        gradient = (
            np.zeros((number_of_points, 3), dtype=np.float64)
            if need_gradient
            else None
        )

        prepared_lines = (
            lines
            if isinstance(lines, PreparedScanSegments)
            else self.prepare_lines(lines)
        )
        line_arrays = self._active_line_arrays(prepared_lines, float(current_time))
        if line_arrays is None or number_of_points == 0:
            return EagarTsaiResult(temperature=temperature, gradient=gradient)

        number_of_lines = line_arrays["start"].shape[0]
        line_tensors = {
            name: torch.as_tensor(value, dtype=self.dtype, device=self.device)
            for name, value in line_arrays.items()
        }

        with torch.inference_mode():
            for point_start in range(0, number_of_points, self.point_chunk_size):
                point_stop = min(
                    point_start + self.point_chunk_size, number_of_points
                )
                point_tensor = torch.as_tensor(
                    points_array[point_start:point_stop],
                    dtype=self.dtype,
                    device=self.device,
                )
                temperature_chunk = torch.zeros(
                    point_stop - point_start, dtype=self.dtype, device=self.device
                )
                gradient_chunk = (
                    torch.zeros(
                        (point_stop - point_start, 3),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    if need_gradient
                    else None
                )

                for line_start in range(0, number_of_lines, self.line_chunk_size):
                    line_stop = min(
                        line_start + self.line_chunk_size, number_of_lines
                    )
                    self._accumulate_line_chunk(
                        line_tensors=line_tensors,
                        line_slice=slice(line_start, line_stop),
                        points=point_tensor,
                        temperature_accumulator=temperature_chunk,
                        gradient_accumulator=gradient_chunk,
                    )

                temperature[point_start:point_stop] = (
                    temperature_chunk.cpu().numpy().astype(np.float64, copy=False)
                )
                if need_gradient and gradient is not None and gradient_chunk is not None:
                    gradient[point_start:point_stop] = (
                        gradient_chunk.cpu().numpy().astype(np.float64, copy=False)
                    )

        return EagarTsaiResult(temperature=temperature, gradient=gradient)

    def evaluate_by_line(
        self,
        lines: Union[Iterable[ScanSegment], PreparedScanSegments],
        points: ArrayLike,
        current_time: float,
        *,
        need_gradient: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluate each finite scan line separately in one vectorized call.

        Unlike :meth:`evaluate`, this method does not superpose the line axis.
        The returned temperature has shape ``(n_lines, n_points)`` and the
        optional gradient has shape ``(n_lines, n_points, 3)``.  Keeping the
        line axis is useful for localized hybrid formulations in which every
        line is multiplied by a different spatial/temporal window.

        The numerical kernel, quadrature, dtype, device, point chunking and
        line chunking are identical to :meth:`evaluate`; this is only an
        output-layout/assembly optimization and introduces no new numerical
        parameter.
        """
        points_array = self._validate_points(points)
        prepared_lines = (
            lines
            if isinstance(lines, PreparedScanSegments)
            else self.prepare_lines(lines)
        )
        number_of_lines = prepared_lines.number_of_lines
        number_of_points = points_array.shape[0]
        temperature = np.zeros(
            (number_of_lines, number_of_points), dtype=np.float64
        )
        gradient = (
            np.zeros((number_of_lines, number_of_points, 3), dtype=np.float64)
            if need_gradient
            else None
        )
        if number_of_lines == 0 or number_of_points == 0:
            return temperature, gradient
        if not np.isfinite(current_time):
            raise ValueError("current_time must be finite")

        # Preserve the input-line ordering in the returned arrays.  Lines that
        # have not started remain exactly zero.
        active_mask = float(current_time) > prepared_lines.start_time
        active_indices = np.flatnonzero(active_mask)
        if active_indices.size == 0:
            return temperature, gradient

        start_times = prepared_lines.start_time[active_indices]
        end_times = prepared_lines.end_time[active_indices]
        ages = float(current_time) - start_times
        exposure = np.minimum(float(current_time), end_times) - start_times
        positive = exposure > 0.0
        if not np.any(positive):
            return temperature, gradient
        active_indices = active_indices[positive]
        ages = ages[positive]
        exposure = exposure[positive]
        lag_lower = np.maximum(ages - exposure, 0.0)

        line_arrays = {
            "start": prepared_lines.start[active_indices],
            "rotation": prepared_lines.rotation[active_indices],
            "speed": prepared_lines.speed[active_indices],
            "age": ages,
            "r_lower": np.sqrt(lag_lower),
            "r_upper": np.sqrt(ages),
        }
        line_tensors = {
            name: torch.as_tensor(value, dtype=self.dtype, device=self.device)
            for name, value in line_arrays.items()
        }

        with torch.inference_mode():
            for point_start in range(0, number_of_points, self.point_chunk_size):
                point_stop = min(
                    point_start + self.point_chunk_size, number_of_points
                )
                point_tensor = torch.as_tensor(
                    points_array[point_start:point_stop],
                    dtype=self.dtype,
                    device=self.device,
                )
                for line_start in range(
                    0, active_indices.size, self.line_chunk_size
                ):
                    line_stop = min(
                        line_start + self.line_chunk_size, active_indices.size
                    )
                    t_line, g_line = self._line_chunk_values(
                        line_tensors=line_tensors,
                        line_slice=slice(line_start, line_stop),
                        points=point_tensor,
                        need_gradient=need_gradient,
                    )
                    output_lines = active_indices[line_start:line_stop]
                    temperature[
                        output_lines, point_start:point_stop
                    ] = t_line.cpu().numpy().astype(np.float64, copy=False)
                    if need_gradient and gradient is not None and g_line is not None:
                        gradient[
                            output_lines, point_start:point_stop, :
                        ] = g_line.cpu().numpy().astype(np.float64, copy=False)

        return temperature, gradient

    def evaluate_temperature(
        self,
        lines: Union[Iterable[ScanSegment], PreparedScanSegments],
        points: ArrayLike,
        current_time: float,
    ) -> np.ndarray:
        return self.evaluate(
            lines, points, current_time, need_gradient=False
        ).temperature

    def evaluate_gradient(
        self,
        lines: Union[Iterable[ScanSegment], PreparedScanSegments],
        points: ArrayLike,
        current_time: float,
    ) -> np.ndarray:
        result = self.evaluate(lines, points, current_time, need_gradient=True)
        assert result.gradient is not None
        return result.gradient

    def _line_chunk_values(
        self,
        *,
        line_tensors: dict[str, torch.Tensor],
        line_slice: slice,
        points: torch.Tensor,
        need_gradient: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return one chunk with the line axis intact."""
        start = line_tensors["start"][line_slice]
        rotation = line_tensors["rotation"][line_slice]
        speed = line_tensors["speed"][line_slice]
        age = line_tensors["age"][line_slice]
        r_lower = line_tensors["r_lower"][line_slice]
        r_upper = line_tensors["r_upper"][line_slice]

        difference = points.unsqueeze(0) - start.unsqueeze(1)
        local = torch.einsum("bij,bnj->bni", rotation, difference)
        xi = local[:, :, 0]
        eta = local[:, :, 1]
        zeta = local[:, :, 2]

        half_width = 0.5 * (r_upper - r_lower)
        midpoint = 0.5 * (r_upper + r_lower)
        r = midpoint[:, None] + half_width[:, None] * self._gauss_nodes[None, :]
        quadrature_weight = half_width[:, None] * self._gauss_weights[None, :]

        lag = r.square()
        beam_coordinate = speed[:, None] * (age[:, None] - lag)
        lag_3d = lag[:, :, None]
        xi_offset = xi[:, None, :] - beam_coordinate[:, :, None]
        eta_3d = eta[:, None, :]
        zeta_3d = zeta[:, None, :]

        denominator_xy = 4.0 * self._alpha * lag_3d + 2.0 * self._sigma_squared
        denominator_prefactor = 2.0 * self._alpha * lag_3d + self._sigma_squared
        exponent = -(
            xi_offset.square() + eta_3d.square()
        ) / denominator_xy - zeta_3d.square() / (4.0 * self._alpha * lag_3d)

        kernel = 2.0 * torch.exp(exponent) / denominator_prefactor
        weighted_kernel = quadrature_weight[:, :, None] * kernel
        temperature_by_line = self._prefactor * weighted_kernel.sum(dim=1)
        if not need_gradient:
            return temperature_by_line, None

        derivative_xi = kernel * (-2.0 * xi_offset / denominator_xy)
        derivative_eta = kernel * (-2.0 * eta_3d / denominator_xy)
        derivative_zeta = kernel * (
            -2.0 * zeta_3d / (4.0 * self._alpha * lag_3d)
        )
        gradient_local = self._prefactor * torch.stack(
            (
                (quadrature_weight[:, :, None] * derivative_xi).sum(dim=1),
                (quadrature_weight[:, :, None] * derivative_eta).sum(dim=1),
                (quadrature_weight[:, :, None] * derivative_zeta).sum(dim=1),
            ),
            dim=2,
        )
        gradient_global = torch.einsum("bni,bij->bnj", gradient_local, rotation)
        return temperature_by_line, gradient_global

    def _accumulate_line_chunk(
        self,
        *,
        line_tensors: dict[str, torch.Tensor],
        line_slice: slice,
        points: torch.Tensor,
        temperature_accumulator: torch.Tensor,
        gradient_accumulator: Optional[torch.Tensor],
    ) -> None:
        temperature_by_line, gradient_by_line = self._line_chunk_values(
            line_tensors=line_tensors,
            line_slice=line_slice,
            points=points,
            need_gradient=gradient_accumulator is not None,
        )
        temperature_accumulator.add_(temperature_by_line.sum(dim=0))
        if gradient_accumulator is not None and gradient_by_line is not None:
            gradient_accumulator.add_(gradient_by_line.sum(dim=0))


__all__ = [
    "EagarTsaiParameters",
    "ScanSegment",
    "EagarTsaiResult",
    "PreparedScanSegments",
    "VectorizedEagarTsai",
]
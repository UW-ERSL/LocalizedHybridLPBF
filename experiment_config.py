"""Problem setup for the LPBF simulations.

Edit the directory settings, EXPERIMENT_LIBRARY and ACTIVE_RUN here.
Launch simulations with: python run_localized_hybrid_experiment.py
This module defines inputs only; it does not run a simulation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# DIRECTORY SETTINGS: change the PyTO address here if it is stored elsewhere.
PYTO_SOURCE_DIRECTORY = (
    SCRIPT_DIRECTORY.parent / "PyTO-main" / "src"
)
GEOMETRY_DIRECTORY = SCRIPT_DIRECTORY / "Models"
RESULTS_DIRECTORY = SCRIPT_DIRECTORY / "results_localized_hybrid"


@dataclass(frozen=True)
class ProjectPaths:
    component_directory: Path
    pyto_source_directory: Path
    results_directory: Path


@dataclass(frozen=True)
class MeshConfiguration:
    stl_path: Path
    element_size: float | tuple[float, float]
    merge_facets: bool = True
    element_size_function: Callable[[np.ndarray], float] | None = None
    maximum_desired_elements: int | None = 10_000_000
    allow_large_mesh: bool = False


@dataclass(frozen=True)
class MaterialConfiguration:
    name: str
    mass_density: float
    thermal_conductivity: float
    specific_heat: float
    youngs_modulus: float = 1.16e11
    poissons_ratio: float = 0.34
    thermal_expansion: float = 8.6e-6
    cost: float = 0.0
    yield_strength: float = 880e6


@dataclass(frozen=True)
class LaserConfiguration:
    laser_power: float
    absorptivity: float
    gaussian_sigma: float


@dataclass(frozen=True)
class BoundaryConfiguration:
    build_axis: int = 2
    bottom_is_minimum: bool = True
    plane_tolerance: float | None = None
    initial_temperature_c: float = 32.0
    bottom_temperature_c: float = 32.0


@dataclass(frozen=True)
class TimeConfiguration:
    delta_time: float
    post_scan_time: float
    save_every: int = 1


@dataclass(frozen=True)
class AnalyticalConfiguration:
    device: str | None = None
    dtype: str = "float32"
    gauss_order: int = 64
    point_chunk_size: int = 4096
    line_chunk_size: int = 8


@dataclass(frozen=True)
class LocalizationConfiguration:
    core_radius: float
    transition_width: float
    retention_age: float
    transfer_duration: float


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    paths: ProjectPaths
    mesh: MeshConfiguration
    scan_path: tuple[Mapping[str, Any], ...]
    material: MaterialConfiguration
    laser: LaserConfiguration
    boundary: BoundaryConfiguration
    time: TimeConfiguration
    analytical: AnalyticalConfiguration
    localization: LocalizationConfiguration


@dataclass(frozen=True)
class RunConfiguration:
    """User-edited configuration for one simulation run."""

    experiment: ExperimentDefinition
    mode: str = "both"
    case_label: str | None = None
    truncation_age: float = 0.1
    # Replay physical/time/scan settings from an existing run, and verify mesh.
    reference_folder: Path | None = None
    solver_name: str = "PARDISO"
    show_mesh_plots: bool = False
    create_result_plots: bool = True


def make_linear_z_size_function(
    element_size_min: float,
    element_size_max: float,
    z_min: float,
    z_max: float,
) -> Callable[[np.ndarray], float]:
    if element_size_min <= 0.0 or element_size_max < element_size_min:
        raise ValueError("invalid element-size limits")
    if z_max <= z_min:
        raise ValueError("z_max must be larger than z_min")

    def size_function(point: np.ndarray) -> float:
        z = float(point[2])
        fraction = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
        return float(
            element_size_max
            + (element_size_min - element_size_max) * fraction
        )

    size_function.configuration = {
        "name": "size_function", "kind": "linear_z",
        "element_size_min": element_size_min, "element_size_max": element_size_max,
        "z_min": z_min, "z_max": z_max,
    }
    return size_function


Fin_MASTER_LINE_COUNT = 500
LOCALIZATION_SCALE = 0.5

def make_Fin_scan_path(
    n_lines: int,
    *, master_line_count: int = Fin_MASTER_LINE_COUNT,
    y_start: float = 0.001, y_end: float = 0.009, z_scan: float = 0.0105,
    top_x_min: float = 0.0, top_x_max: float = 0.030,
    x_boundary_margin: float = 5e-4, velocity: float = 2.0,
    off_time: float = 0.0005,
) -> tuple[Mapping[str, Any], ...]:
    """Create a fixed-spacing prefix of a 500-line master scan path.

    Lines travel in the y-direction. Their y-coordinates are prescribed
    directly. In the x-direction, the 500-line master path is distributed
    uniformly between the specified boundary margins. Smaller cases use
    the first n_lines of this same master path.
    """

    if not 1 <= n_lines <= master_line_count:
        raise ValueError(
            f"n_lines must be between 1 and "
            f"{master_line_count}"
        )

    # ---------------------------------------------------------
    # Scan-line geometry: travel in the y-direction
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Placement of lines in the x-direction
    # ---------------------------------------------------------


    if (master_line_count < 2 or velocity <= 0 or off_time < 0 or y_end <= y_start
            or top_x_max - top_x_min <= 2*x_boundary_margin):
        raise ValueError("Invalid scan geometry, master count, velocity or delay")
    x_first = top_x_min + x_boundary_margin
    x_last = top_x_max - x_boundary_margin

    # Spacing of the 500-line master path
    hatch_spacing = (
        (x_last - x_first)
        / (master_line_count - 1)
    )

    x_positions = (
        x_first
        + np.arange(n_lines, dtype=float)
        * hatch_spacing
    )

    # ---------------------------------------------------------
    # Process parameters
    # ---------------------------------------------------------

    line_length = y_end - y_start
    line_duration = line_length / velocity

    # ---------------------------------------------------------
    # Construct scan lines
    # ---------------------------------------------------------
    lines = []

    for i, x in enumerate(x_positions):
        start_time = i * (
            line_duration + off_time
        )

        lines.append(
            {
                "kind": "line",
                "start_time": start_time,
                "start_point": [
                    float(x),
                    y_start,
                    z_scan,
                ],
                "end_point": [
                    float(x),
                    y_end,
                    z_scan,
                ],
                "velocity": velocity,
            }
        )

    return tuple(lines)

# Initial values only.  They are explicit experiment parameters, not hidden
# defaults in the solver.  Change them here for the actual parameter study.
EXPERIMENT_LIBRARY: dict[str, ExperimentDefinition] = {
    "Fin": ExperimentDefinition(
        name="Fin_500_lines",

        paths=ProjectPaths(
            component_directory=SCRIPT_DIRECTORY,
            pyto_source_directory=PYTO_SOURCE_DIRECTORY,
            results_directory=RESULTS_DIRECTORY,
        ),

        mesh=MeshConfiguration(
            stl_path=GEOMETRY_DIRECTORY / "Fin.STL",

            element_size=(200e-6, 1000e-6),

            element_size_function=make_linear_z_size_function(
                200e-6,
                1000e-6,
                0.0,
                0.01,
            ),
        ),

        scan_path=make_Fin_scan_path(
            n_lines=500,
        ),


        material=MaterialConfiguration(
            name="Fin material",
            mass_density=8440.0,
            thermal_conductivity=11.4,
            specific_heat=443.0,
        ),

        laser=LaserConfiguration(
            laser_power=300.0,
            absorptivity=1.0,
            gaussian_sigma=50e-6,
        ),

        boundary=BoundaryConfiguration(),

        time=TimeConfiguration(
            delta_time=0.5e-3,
            post_scan_time=1,
            save_every=1,
        ),

        analytical=AnalyticalConfiguration(
            device="cpu",
            dtype="float32",
            gauss_order=64,
            point_chunk_size=4096,
            line_chunk_size=8,
        ),

        localization=LocalizationConfiguration(
            core_radius=LOCALIZATION_SCALE * 1.0e-3,
            transition_width=LOCALIZATION_SCALE * 1.0e-3,
            retention_age=LOCALIZATION_SCALE * 0.05,
            transfer_duration=LOCALIZATION_SCALE * 0.05,
        ),
    ),

    "Overhang": ExperimentDefinition(
        name="Overhang_basic_oneline_test",

        paths=ProjectPaths(
            component_directory=SCRIPT_DIRECTORY,
            pyto_source_directory=PYTO_SOURCE_DIRECTORY,
            results_directory=RESULTS_DIRECTORY,
        ),

        # ---------------------------------------------------------
        # Geometry / mesh
        # Overhang model:
        # x ~ 0 ... 0.003 m
        # y ~ 0 ... 0.003 m
        # z ~ 0 ... 0.002 m
        # ---------------------------------------------------------
        mesh=MeshConfiguration(
            stl_path=GEOMETRY_DIRECTORY / "Overhang.STL",

            # Fine near top, coarser toward bottom.
            element_size=(150e-6, 350e-6),

            merge_facets=True,
            maximum_desired_elements=10_000_000,
            allow_large_mesh=False,

            element_size_function=make_linear_z_size_function(
                element_size_min=150e-6,
                element_size_max=350e-6,
                z_min=0.0,
                z_max=0.002,
            ),
        ),

        # ---------------------------------------------------------
        # One scan line on the top surface z = 0.002 m
        #
        # Length   = 1.0 mm
        # Velocity = 0.5 m/s
        # Duration = 0.001 / 0.5 = 0.002 s
        # ---------------------------------------------------------
        scan_path=tuple(
            {
                "kind": "line",
                "start_time": i * (
                    (0.0020 - 0.0010) / 0.5
                    + 0.0005
                ),
                "start_point": [
                    0.0010,
                    0.0003 + i * 100e-6,
                    0.002,
                ],
                "end_point": [
                    0.0020,
                    0.0003 + i * 100e-6,
                    0.002,
                ],
                "velocity": 0.5,
            }
            for i in range(25)
        ),

        # ---------------------------------------------------------
        # Material
        # ---------------------------------------------------------
        material=MaterialConfiguration(
            name="Overhang test material",
            mass_density=8440.0,
            thermal_conductivity=11.4,
            specific_heat=443.0,
        ),

        # ---------------------------------------------------------
        # Laser / Eagar--Tsai
        # ---------------------------------------------------------
        laser=LaserConfiguration(
            laser_power=300.0,
            absorptivity=1.0,
            gaussian_sigma=50e-6,
        ),

        # ---------------------------------------------------------
        # Physical boundary conditions
        #
        # z = 0       -> fixed physical bottom temperature
        # other faces -> adiabatic physical boundaries
        # laser       -> represented analytically
        # ---------------------------------------------------------
        boundary=BoundaryConfiguration(
            build_axis=2,
            bottom_is_minimum=True,
            initial_temperature_c=32.0,
            bottom_temperature_c=32.0,
        ),

        # ---------------------------------------------------------
        # Time
        #
        # Scan ends at t = 0.002 s.
        # post_scan_time = 0.048 s
        # final simulation time = 0.050 s.
        #
        # dt = 0.0005 -> 100 total time steps.
        # ---------------------------------------------------------
        time=TimeConfiguration(
            delta_time=5e-4,
            post_scan_time=0.048,
            save_every=1,
        ),

        # ---------------------------------------------------------
        # Vectorized Eagar--Tsai numerical settings
        # ---------------------------------------------------------
        analytical=AnalyticalConfiguration(
            device="cpu",
            dtype="float32",
            gauss_order=64,
            point_chunk_size=4096,
            line_chunk_size=8,
        ),

        # ---------------------------------------------------------
        # NEW HYBRID PARAMETERS
        #
        # chi(x,t) = s(x) * a(t)
        #
        # These are INITIAL TEST VALUES, not final calibrated values.
        # ---------------------------------------------------------
        localization=LocalizationConfiguration(

            # s = 1 within 0.30 mm of the finite scan segment.
            # This is 6 sigma for sigma = 50 um.
            core_radius=300e-6,

            # Smooth quintic decay from s=1 to s=0 over another
            # 0.60 mm.  The total analytical support radius is
            # therefore 0.90 mm.
            transition_width=600e-6,

            # Keep the completed line fully analytically represented
            # for 5 ms after the laser reaches the endpoint.
            retention_age=0.005,

            # Then smoothly transfer a: 1 -> 0 over 10 ms.
            transfer_duration=0.010,
        ),
    ),
}

# Separate configuration; the one-line validation and other experiments remain available.
EXPERIMENT_LIBRARY["Fin_500_temporal_truncation"] = replace(
    EXPERIMENT_LIBRARY["Fin"],
    name="Fin_500_temporal_truncation",
    scan_path=make_Fin_scan_path(n_lines=500),
    time=TimeConfiguration(
        delta_time=500e-6,
        post_scan_time=1.0,
        save_every=1,
    ),
    analytical=AnalyticalConfiguration(
        device="cpu",
        dtype="float32",
        gauss_order=64,
        point_chunk_size=4096,
        line_chunk_size=8,
    ),
)

DEFAULT_EXPERIMENT = "Fin"


# ---------------------------------------------------------------------------
# SELECT AND RUN AN EXPERIMENT
#
# Edit only this block to select the experiment and mode for the next run.
# ``case_label`` is optional; when omitted, ``experiment.name`` is used.
# ---------------------------------------------------------------------------

ACTIVE_RUN = RunConfiguration(
    experiment=EXPERIMENT_LIBRARY["Fin"],
    mode="localized",
    case_label=f"Fin_500_lines_scale_{LOCALIZATION_SCALE:g}",
    reference_folder=None,
)



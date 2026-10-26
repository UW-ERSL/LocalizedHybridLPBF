"""Prepare and execute LPBF hybrid simulations from experiment_config.py.

Required project modules beside this script (or in component_directory):

    stl_mesh_builder.py
    scan_path_analyzer.py
    vectorized_eagar_tsai.py
    localized_hybrid_simulation.py
    localized_hybrid_evaluator.py

Required PyTO modules in pyto_source_directory:

    bound_cond.py
    mat_lib.py
    tet_transient_thermal.py
    linear_solvers.py
    tet_mesher.py
    tet_thermal_fea.py
    tet_thermal_examples.py
    stl_reader.py

There is deliberately no surrogate model and no dynamic partitioner here.
The simulation modes use identical mesh/material/scan/time data; only the hybrid
formulation changes.
"""

from __future__ import annotations

import importlib
from pyto_runtime import configure_pyto, environment_metadata
import json
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Input definitions live in experiment_config.py; no problem presets are defined here.
from experiment_config import (
    ProjectPaths, MeshConfiguration, MaterialConfiguration, LaserConfiguration,
    BoundaryConfiguration, TimeConfiguration, AnalyticalConfiguration,
    LocalizationConfiguration, ExperimentDefinition, RunConfiguration,
)

# Base for relative paths passed to the execution API.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

_PARDISO_RUNTIME_INITIALIZED = False


def _initialize_pardiso_runtime(solver_name: str) -> None:
    """Initialize MKL/PARDISO before PyTorch on Windows.

    A real tiny solve initializes the runtime; importing pypardiso alone is
    insufficient. This preserves the configured threading layer and thread
    counts. No duplicate-runtime bypass is set and no DLL is modified.
    """
    global _PARDISO_RUNTIME_INITIALIZED
    if (sys.platform != "win32" or solver_name.upper() != "PARDISO"
            or _PARDISO_RUNTIME_INITIALIZED):
        return

    import pypardiso
    from scipy.sparse import csr_matrix

    matrix = csr_matrix([[2.0, 1.0], [1.0, 3.0]])
    try:
        solution = pypardiso.spsolve(matrix, np.ones(2))
        if not np.allclose(solution, [0.4, 0.2], rtol=1e-12, atol=1e-14):
            raise RuntimeError("PARDISO initialization solve returned an unexpected result")
    finally:
        pypardiso.ps.free_memory()

    _PARDISO_RUNTIME_INITIALIZED = True
    print("[Initialization] PARDISO runtime ready", flush=True)


@dataclass(frozen=True)
class ProjectAPI:
    STLMeshBuilder: Any
    ScanPathAnalyzer: Any
    EagarTsaiParameters: Any
    VectorizedEagarTsai: Any
    BC: Any
    Material: Any
    BoundaryQuadrature: Any
    HybridMode: Any
    HybridSimulation: Any
    LocalizationParameters: Any
    LocalizedHybridEvaluator: Any


@dataclass(frozen=True)
class BoundarySetup:
    bc: Any
    quadrature: Any
    bottom_node_ids: np.ndarray


def _absolute(path: Path, base: Path = SCRIPT_DIRECTORY) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _add_import_directory(path: Path, description: str) -> None:
    resolved = _absolute(path)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def load_project_api(paths: ProjectPaths) -> ProjectAPI:
    _add_import_directory(paths.component_directory, "hybrid component directory")
    configure_pyto(paths.pyto_source_directory)
    modules = {
        name: importlib.import_module(name)
        for name in (
            "stl_mesh_builder",
            "scan_path_analyzer",
            "vectorized_eagar_tsai",
            "bound_cond",
            "mat_lib",
            "localized_hybrid_simulation",
            "localized_hybrid_evaluator",
        )
    }
    return ProjectAPI(
        STLMeshBuilder=modules["stl_mesh_builder"].STLMeshBuilder,
        ScanPathAnalyzer=modules["scan_path_analyzer"].ScanPathAnalyzer,
        EagarTsaiParameters=modules["vectorized_eagar_tsai"].EagarTsaiParameters,
        VectorizedEagarTsai=modules["vectorized_eagar_tsai"].VectorizedEagarTsai,
        BC=modules["bound_cond"].BC,
        Material=modules["mat_lib"].Material,
        BoundaryQuadrature=modules["localized_hybrid_simulation"].BoundaryQuadrature,
        HybridMode=modules["localized_hybrid_simulation"].HybridMode,
        HybridSimulation=modules["localized_hybrid_simulation"].HybridSimulation,
        LocalizationParameters=modules["localized_hybrid_simulation"].LocalizationParameters,
        LocalizedHybridEvaluator=modules["localized_hybrid_evaluator"].LocalizedHybridEvaluator,
    )


def build_mesh(api: ProjectAPI, configuration: MeshConfiguration) -> Any:
    builder = api.STLMeshBuilder(
        stl_file=_absolute(configuration.stl_path),
        element_size=configuration.element_size,
        element_size_function=configuration.element_size_function,
        merge_facets=configuration.merge_facets,
        max_desired_elements=configuration.maximum_desired_elements,
        allow_large_mesh=configuration.allow_large_mesh,
    )
    result = builder.build()
    print(result.summary())
    return result


def build_material(api: ProjectAPI, configuration: MaterialConfiguration) -> Any:
    values = asdict(configuration)
    values["thermal_expansion_coefficient"] = values.pop("thermal_expansion")
    return api.Material(**values)


def build_analytical_engine(
    api: ProjectAPI,
    material: Any,
    laser: LaserConfiguration,
    analytical: AnalyticalConfiguration,
) -> Any:
    parameters = api.EagarTsaiParameters.from_absorbed_power(
        thermal_conductivity=material.thermal_conductivity,
        density=material.mass_density,
        specific_heat=material.specific_heat,
        sigma=laser.gaussian_sigma,
        laser_power=laser.laser_power,
        absorptivity=laser.absorptivity,
    )
    return api.VectorizedEagarTsai(
        parameters,
        device=analytical.device,
        dtype=analytical.dtype,
        gauss_order=analytical.gauss_order,
        point_chunk_size=analytical.point_chunk_size,
        line_chunk_size=analytical.line_chunk_size,
    )


def _default_plane_tolerance(nodes: np.ndarray, axis: int) -> float:
    scale = max(
        float(np.ptp(nodes[:, axis])),
        float(np.max(np.ptp(nodes, axis=0))),
        1.0e-12,
    )
    return max(1.0e-10 * scale, 128.0 * np.finfo(float).eps * scale)


def _orient_surface_normals_outward(
    nodes: np.ndarray,
    elements: np.ndarray,
    surface_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    opposite_node: dict[tuple[int, int, int], int] = {}
    local_faces = ((1, 2, 3, 0), (0, 3, 2, 1), (0, 1, 3, 2), (0, 2, 1, 3))
    for element in np.asarray(elements, dtype=int):
        for i, j, k, opposite in local_faces:
            key = tuple(sorted((int(element[i]), int(element[j]), int(element[k]))))
            if key in opposite_node:
                opposite_node.pop(key, None)
            else:
                opposite_node[key] = int(element[opposite])

    xyz = nodes[surface_faces]
    centers = xyz.mean(axis=1)
    raw = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    lengths = np.linalg.norm(raw, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("surface mesh contains a zero-area triangle")
    normals = raw / lengths[:, None]
    for face_index, face in enumerate(surface_faces):
        key = tuple(sorted(int(node_id) for node_id in face))
        if key not in opposite_node:
            raise ValueError("surface triangle is not a unique tetrahedron face")
        inward = nodes[opposite_node[key]] - centers[face_index]
        if float(np.dot(normals[face_index], inward)) > 0.0:
            normals[face_index] *= -1.0
    return centers, normals, 0.5 * lengths


def build_boundary_setup(
    api: ProjectAPI,
    mesh: Any,
    configuration: BoundaryConfiguration,
) -> BoundarySetup:
    if configuration.build_axis not in (0, 1, 2):
        raise ValueError("build_axis must be 0, 1, or 2")
    if not np.all(np.isfinite([configuration.initial_temperature_c, configuration.bottom_temperature_c])):
        raise ValueError("Initial and bottom temperatures must be finite Celsius values")

    nodes = np.asarray(mesh.node_xyz, dtype=float)
    elements = np.asarray(mesh.elems, dtype=int)
    surface_faces = np.asarray(mesh.surface_triangles, dtype=int)
    axis = configuration.build_axis
    tolerance = (
        _default_plane_tolerance(nodes, axis)
        if configuration.plane_tolerance is None
        else float(configuration.plane_tolerance)
    )

    coordinate = nodes[:, axis]
    minimum = float(np.min(coordinate))
    maximum = float(np.max(coordinate))
    face_coordinate = nodes[surface_faces, axis]
    minimum_faces = np.all(np.abs(face_coordinate - minimum) <= tolerance, axis=1)
    maximum_faces = np.all(np.abs(face_coordinate - maximum) <= tolerance, axis=1)
    bottom_mask = minimum_faces if configuration.bottom_is_minimum else maximum_faces
    top_mask = maximum_faces if configuration.bottom_is_minimum else minimum_faces
    side_mask = ~(bottom_mask | top_mask)

    bottom_indices = np.flatnonzero(bottom_mask)
    side_indices = np.flatnonzero(side_mask)
    if bottom_indices.size == 0 or side_indices.size == 0:
        raise ValueError("could not identify bottom and side boundaries")

    bottom_nodes = np.unique(surface_faces[bottom_indices].ravel())
    bc = api.BC(
        force=np.zeros(nodes.shape[0], dtype=float),
        fixed_dofs=bottom_nodes.astype(int),
        dirichlet_values=np.full(
            bottom_nodes.size,
            configuration.bottom_temperature_c - configuration.initial_temperature_c,
            dtype=float,
        ),
    )

    centers, normals, areas = _orient_surface_normals_outward(
        nodes, elements, surface_faces
    )
    quadrature = api.BoundaryQuadrature(
        face_nodes=surface_faces[side_indices],
        points_global=centers[side_indices],
        normals_global=normals[side_indices],
        weights=areas[side_indices],
    )
    return BoundarySetup(bc=bc, quadrature=quadrature, bottom_node_ids=bottom_nodes)


def scan_end_time(scan_analysis: Any) -> float:
    lines = tuple(scan_analysis.lines)
    return max((float(line.end_time) for line in lines), default=0.0)


def _configuration_to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        if hasattr(value, "configuration"):
            return value.configuration
        return getattr(value, "__name__", str(value))
    if isinstance(value, dict):
        return {str(k): _configuration_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_configuration_to_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            k: _configuration_to_jsonable(v)
            for k, v in asdict(value).items()
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_experiment_configuration(
    experiment: ExperimentDefinition, output_directory: Path
) -> None:

    (output_directory / "experiment_configuration.json").write_text(
        json.dumps(
            _configuration_to_jsonable(experiment), indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )


def _safe_label(value: str) -> str:
    label = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value).strip()
    ).strip("._")
    if not label:
        raise ValueError("case_label must contain at least one letter or number")
    return label


def _save_mesh_data(mesh: Any, output_directory: Path) -> None:
    np.savez_compressed(
        output_directory / "mesh_data.npz",
        node_coordinates=np.asarray(mesh.node_xyz, dtype=float),
        element_connectivity=np.asarray(mesh.elems, dtype=int),
        surface_triangles=np.asarray(mesh.surface_triangles, dtype=int),
    )


def _save_run_manifest(
    run_configuration: RunConfiguration,
    output_directory: Path,
    scan_analysis: Any,
    available_modes: Sequence[str],
) -> None:
    experiment = run_configuration.experiment
    manifest = {
        "result_format_version": 2,
        "environment": environment_metadata(),
        "solver": run_configuration.solver_name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "case_label": (
            run_configuration.case_label
            if run_configuration.case_label is not None
            else experiment.name
        ),
        "requested_mode": run_configuration.mode,
        "truncation_age": (run_configuration.truncation_age
                           if run_configuration.mode == "temporal_truncation" else None),
        "reference_folder": (str(run_configuration.reference_folder)
                             if run_configuration.reference_folder is not None else None),
        "available_modes": list(available_modes),
        "scan_line_count": len(tuple(scan_analysis.lines)),
        "experiment": _configuration_to_jsonable(experiment),
        "files": {
            "experiment_configuration": "experiment_configuration.json",
            "mesh": "mesh_data.npz",
        },
    }
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _warn_if_localization_core_is_small(
    experiment: ExperimentDefinition,
) -> None:
    minimum_core_radius = 6.0 * experiment.laser.gaussian_sigma
    core_radius = experiment.localization.core_radius
    if core_radius < minimum_core_radius and not np.isclose(
        core_radius, minimum_core_radius, rtol=1.0e-12, atol=0.0
    ):
        warnings.warn(
            "The spatial core radius is smaller than six Gaussian standard "
            "deviations; the localization may modify the active heat source.",
            RuntimeWarning,
            stacklevel=2,
        )


def _reference_experiment(run_configuration: RunConfiguration) -> ExperimentDefinition:
    """Replay saved settings without guessing serialized mesh-size callables."""
    experiment = run_configuration.experiment
    if run_configuration.reference_folder is None:
        return experiment
    folder = _absolute(run_configuration.reference_folder)
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("result_format_version") != 2:
        raise ValueError("reference_folder must contain results from the new code")
    saved = manifest["experiment"]
    current_mesh = _configuration_to_jsonable(experiment.mesh)
    # Paths may change when the project is moved; all mesh generation settings
    # must agree. The final node/connectivity check is also mandatory.
    for key, value in saved["mesh"].items():
        current_value = current_mesh.get(key)
        if key != "stl_path" and current_value != value:
            raise ValueError(
                f"reference mesh setting {key!r} differs; update the experiment mesh "
                "configuration to match the saved run before executing"
            )
    if int(manifest["scan_line_count"]) != len(experiment.scan_path):
        raise ValueError("reference scan-line count differs from the selected experiment")
    return replace(
        experiment,
        scan_path=tuple(saved["scan_path"]),
        material=MaterialConfiguration(**saved["material"]),
        laser=LaserConfiguration(**saved["laser"]),
        boundary=BoundaryConfiguration(**saved["boundary"]),
        time=TimeConfiguration(**saved["time"]),
        analytical=AnalyticalConfiguration(**saved["analytical"]),
        localization=LocalizationConfiguration(**saved["localization"]),
    )


def _verify_reference_mesh(mesh: Any, folder: Path) -> None:
    with np.load(_absolute(folder) / "mesh_data.npz", allow_pickle=False) as saved:
        for key, actual in (
            ("node_coordinates", np.asarray(mesh.node_xyz, dtype=float)),
            ("element_connectivity", np.asarray(mesh.elems, dtype=int)),
            ("surface_triangles", np.asarray(mesh.surface_triangles, dtype=int)),
        ):
            if not np.array_equal(saved[key], actual):
                raise ValueError(
                    f"generated mesh differs from reference {key}; cannot compare "
                    "nodal histories directly. Use the original mesh generation setup."
                )


def run_experiment(run_configuration: RunConfiguration) -> Path:
    experiment = _reference_experiment(run_configuration)
    run_configuration = replace(run_configuration, experiment=experiment)
    mode = run_configuration.mode
    if mode not in {"both", "conventional", "localized", "temporal_truncation"}:
        raise ValueError("invalid hybrid mode")
    if mode == "temporal_truncation" and (
        not np.isfinite(run_configuration.truncation_age)
        or run_configuration.truncation_age < 0.0
    ):
        raise ValueError("truncation_age must be finite and nonnegative")
    if mode in {"both", "localized"}:
        _warn_if_localization_core_is_small(experiment)
    # Must precede load_project_api(), which imports the PyTorch ET evaluator.
    _initialize_pardiso_runtime(run_configuration.solver_name)
    api = load_project_api(experiment.paths)
    mesh_result = build_mesh(api, experiment.mesh)
    mesh = mesh_result.mesh
    if run_configuration.reference_folder is not None:
        _verify_reference_mesh(mesh, run_configuration.reference_folder)

    if run_configuration.show_mesh_plots:
        mesh_result.plot_stl_geometry(show_edges=False, show_axes=True, show_bounding_box=True)
        mesh_result.plot_tetrahedral_mesh(title=f"Tet Mesh: {mesh_result.model_name}")
    material = build_material(api, experiment.material)
    boundary = build_boundary_setup(api, mesh, experiment.boundary)
    scan_analysis = api.ScanPathAnalyzer().analyze(experiment.scan_path)
    engine = build_analytical_engine(
        api, material, experiment.laser, experiment.analytical
    )

    total_time = scan_end_time(scan_analysis) + experiment.time.post_scan_time
    # HybridSimulation requires an integer number of time steps.  Extend only
    # by roundoff to the next exact grid point if needed.
    n_steps = int(np.ceil(total_time / experiment.time.delta_time - 1.0e-12))
    total_time = n_steps * experiment.time.delta_time

    localization = api.LocalizationParameters(**asdict(experiment.localization))
    common = dict(
        mesh=mesh,
        mat_prop=material,
        bc=boundary.bc,
        delta_time=experiment.time.delta_time,
        total_time=total_time,
        boundary_quadrature=boundary.quadrature,
        analytical_engine=engine,
        scan_analysis=scan_analysis,
        initial_temperature_c=experiment.boundary.initial_temperature_c,
        solver_name=run_configuration.solver_name,
        save_every=experiment.time.save_every,
    )

    conventional = None
    localized = None
    truncated = None
    if mode in {"both", "conventional"}:
        conventional = api.HybridSimulation(
            mode=api.HybridMode.CONVENTIONAL,
            localization=None,
            **common,
        ).run()
    if mode in {"both", "localized"}:
        localized = api.HybridSimulation(
            mode=api.HybridMode.LOCALIZED,
            localization=localization,
            **common,
        ).run()

    if mode == "temporal_truncation":
        truncated = api.HybridSimulation(
            mode=api.HybridMode.TEMPORAL_TRUNCATION,
            localization=None,
            truncation_age=run_configuration.truncation_age,
            **common,
        ).run()

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    case_label = _safe_label(
        run_configuration.case_label
        if run_configuration.case_label is not None
        else experiment.name
    )
    output = (
        _absolute(experiment.paths.results_directory)
        / f"{case_label}_{mode}_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=True)
    _save_experiment_configuration(experiment, output)
    _save_mesh_data(mesh, output)

    if mode == "both":
        evaluator = api.LocalizedHybridEvaluator(conventional, localized)
        evaluator.print_summary()
        evaluator.save_results(output)
        if run_configuration.create_result_plots:
            evaluator.create_plots(output)
    else:
        result = (truncated if truncated is not None else
                  conventional if conventional is not None else localized)
        api.LocalizedHybridEvaluator.save_single_run(result, output)
        timing = api.LocalizedHybridEvaluator.timing_summary(result)
        print(f"{mode.capitalize()} wall time: {timing['wall_time']:.6f} s")
    available_modes = (
        ("conventional", "localized") if mode == "both" else (mode,)
    )
    _save_run_manifest(
        run_configuration, output, scan_analysis, available_modes
    )
    print(f"Results saved to: {output}")
    return output


def main() -> None:
    from experiment_config import ACTIVE_RUN

    run_experiment(ACTIVE_RUN)


if __name__ == "__main__":
    main()

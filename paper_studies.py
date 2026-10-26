"""Generate paper studies from previously saved simulation folders.

This module never runs the thermal solver. Configure arbitrary saved result
folders below, then run this file directly from the IDE.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from localized_hybrid_evaluator import (HybridAccuracyEvaluator, LocalizedHybridEvaluator,
                                        validate_temperature_metadata)
from localized_hybrid_simulation import HybridMode, HybridRunResult

# Optional Abaqus reference studies.
from abaqus_paper_studies import (
    AbaqusCase, AbaqusConvergenceStudyConfiguration, AbaqusHybridStudyConfiguration,
    run_abaqus_convergence_study, run_abaqus_hybrid_study,
)


@dataclass(frozen=True)
class LineCountCase:
    line_count: int
    conventional_folder: Path
    localized_folder: Path
    label: str | None = None


@dataclass(frozen=True)
class LineCountStudyConfiguration:
    study_name: str
    cases: tuple[LineCountCase, ...]
    output_directory: Path


@dataclass(frozen=True)
class LocalizationScaleCase:
    scale: float
    label: str
    localized_folder: Path


@dataclass(frozen=True)
class LocalizationExtentStudyConfiguration:
    study_name: str
    conventional_folder: Path
    cases: tuple[LocalizationScaleCase, ...]
    output_directory: Path


@dataclass(frozen=True)
class PostScanTemporalStudyConfiguration:
    study_name: str
    line_count: int
    conventional_folder: Path
    localized_folder: Path
    post_scan_durations: tuple[float, ...]
    output_directory: Path


@dataclass(frozen=True)
class DetailedCaseStudyConfiguration:
    study_name: str
    line_count: int
    conventional_folder: Path
    localized_folder: Path
    scan_time_fractions: tuple[float, ...]
    post_scan_field_times: tuple[float, ...]
    output_directory: Path


@dataclass(frozen=True)
class TemporalTruncationStudyConfiguration:
    study_name: str
    line_count: int
    conventional_folder: Path
    localized_folder: Path
    truncation_folder: Path
    output_directory: Path


StudyConfiguration = (
    LineCountStudyConfiguration
    | LocalizationExtentStudyConfiguration
    | PostScanTemporalStudyConfiguration
    | DetailedCaseStudyConfiguration
    | TemporalTruncationStudyConfiguration
    | AbaqusConvergenceStudyConfiguration
    | AbaqusHybridStudyConfiguration
)


# ---------------------------------------------------------------------------
# USER CONFIGURATION
#
# Run simulations first, then replace REPLACE_WITH_* below with your actual
# saved folder names. Keep timestamped names as saved; do not rename your runs.
# RESULTS_DIRECTORY is the parent of those folders; change it if needed.
# An absolute Path(r"C:\path\to\run") also works.
# Configure only studies selected in ACTIVE_STUDIES (empty by default).
# Input folders must exist. output_directory is where study outputs are saved.
# A "both" run can supply both conventional_folder and localized_folder.
# ---------------------------------------------------------------------------
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = SCRIPT_DIRECTORY / "results_localized_hybrid"

LOCALIZATION_EXTENT_STUDY = LocalizationExtentStudyConfiguration(
    study_name="fin_500_localization_extent",

    conventional_folder=(
        RESULTS_DIRECTORY
        / "REPLACE_WITH_Fin_500_lines_scale_0.5_both_run"
    ),

    cases=(
        LocalizationScaleCase(
            scale=0.50,
            label="lambda_0.50",
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_500_lines_scale_0.5_both_run"
            ),
        ),

        LocalizationScaleCase(
            scale=0.75,
            label="lambda_0.75",
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_500_lines_scale_0.75_localized_run"
            ),
        ),

        LocalizationScaleCase(
            scale=1.00,
            label="lambda_1.00",
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_500_lines_scale_1_localized_run"
            ),
        ),

        LocalizationScaleCase(
            scale=1.25,
            label="lambda_1.25",
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_500_lines_scale_1.25_localized_run"
            ),
        ),

        LocalizationScaleCase(
            scale=1.50,
            label="lambda_1.50",
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_500_lines_scale_1.5_localized_run"
            ),
        ),

        LocalizationScaleCase(
            scale=1.75,
            label="lambda_1.75",
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_500_lines_scale_1.75_localized_run"
            ),
        ),

        LocalizationScaleCase(
            scale=2.00,
            label="lambda_2.00",
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_500_lines_scale_2_localized_run"
            ),
        ),
    ),

    output_directory=(
        RESULTS_DIRECTORY
        / "paper_localization_scale_study"
    ),
)


# Optional additional analysis; not selected by default.
LINE_COUNT_STUDY = LineCountStudyConfiguration(
    study_name="fin_line_count_scaling",
    cases=(
        LineCountCase(
            line_count=100,
            conventional_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_100_lines_both_run"
            ),
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_100_lines_both_run"
            ),
            label="100_lines",
        ),
        LineCountCase(
            line_count=200,
            conventional_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_200_lines_both_run"
            ),
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_200_lines_both_run"
            ),
            label="200_lines",
        ),
        LineCountCase(
            line_count=300,
            conventional_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_300_lines_both_run"
            ),
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_300_lines_both_run"
            ),
            label="300_lines",
        ),
        LineCountCase(
            line_count=400,
            conventional_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_400_lines_both_run"
            ),
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_400_lines_both_run"
            ),
            label="400_lines",
        ),
        LineCountCase(
            line_count=500,
            conventional_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_500_lines_both_run"
            ),
            localized_folder=(
                RESULTS_DIRECTORY
                / "REPLACE_WITH_Fin_500_lines_both_run"
            ),
            label="500_lines",
        ),
    ),
    output_directory=(
        RESULTS_DIRECTORY / "paper_line_count_study_paper"
    ),
)


# Optional additional analysis; not selected by default.
POST_SCAN_TEMPORAL_STUDY = PostScanTemporalStudyConfiguration(
    study_name="fin_100_post_scan_temporal",
    line_count=100,
    conventional_folder=(
        RESULTS_DIRECTORY
        / "REPLACE_WITH_Fin_100_lines_posttime_2s_both_run"
    ),
    localized_folder=(
        RESULTS_DIRECTORY
        / "REPLACE_WITH_Fin_100_lines_posttime_2s_both_run"
    ),
    post_scan_durations=(0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0),
    output_directory=RESULTS_DIRECTORY / "paper_post_scan_temporal_study_paper",
)


DETAILED_500_LINE_STUDY = DetailedCaseStudyConfiguration(
    study_name="fin_500_detailed_case",
    line_count=500,
    conventional_folder=(
        RESULTS_DIRECTORY
        / "REPLACE_WITH_Fin_500_lines_scale_0.5_both_run"
    ),
    localized_folder=(
        RESULTS_DIRECTORY
        / "REPLACE_WITH_Fin_500_lines_scale_1_localized_run"
    ),
    scan_time_fractions=(0.20, 0.50, 1.00),
    post_scan_field_times=(0.50,),
    output_directory=RESULTS_DIRECTORY / "paper_500_line_detailed_study",
)


# ---------------------------------------------------------------------------
# ABAQUS STUDIES
# Replace the parent path and case paths below with your own reference data:
# an ODB file, a folder with exactly one ODB, or a completed export folder.
# Use metres and physical Celsius temperatures, matching the hybrid inputs.
# Check case resolutions and set physical_setup_confirmed=True only after
# checking the physical setup. No temperature offsets are applied.
# ---------------------------------------------------------------------------
ABAQUS_RUNS = SCRIPT_DIRECTORY / "REPLACE_WITH_Abaqus_runs_directory"

ABAQUS_A1 = AbaqusCase(
    "A1",
    ABAQUS_RUNS / "REPLACE_WITH_A1_CoarseMesh_FineTime",
    1.25e-5,
    5e-4,
    (200e-6, 1000e-6),
)

ABAQUS_A2 = AbaqusCase(
    "A2",
    ABAQUS_RUNS / "REPLACE_WITH_A2_MediumMesh_FineTime",
    1.25e-5,
    5e-4,
    (150e-6, 750e-6),
)

ABAQUS_A3 = AbaqusCase(
    "A3",
    ABAQUS_RUNS / "REPLACE_WITH_A3_FineMesh_FineTime",
    1.25e-5,
    5e-4,
    (100e-6, 500e-6),
)

ABAQUS_A4 = AbaqusCase(
    "A4",
    ABAQUS_RUNS / "REPLACE_WITH_A4_FineMesh_MediumTime",
    2.5e-5,
    1e-3,
    (100e-6, 500e-6),
)

ABAQUS_A5 = AbaqusCase(
    "A5",
    ABAQUS_RUNS / "REPLACE_WITH_A5_FineMesh_CoarseTime",
    5e-5,
    2e-3,
    (100e-6, 500e-6),
)

ABAQUS_A6 = AbaqusCase(
    "A6",
    ABAQUS_RUNS / "REPLACE_WITH_A6_FinerMesh_FineTime",
    1.25e-5,
    5e-4,
    (50e-6, 250e-6),
)


ABAQUS_CONVERGENCE_STUDY = AbaqusConvergenceStudyConfiguration(
    study_name="abaqus_one_line_convergence",

    mesh_cases=(
        ABAQUS_A1,
        ABAQUS_A2,
        ABAQUS_A3,
        ABAQUS_A6,
    ),

    time_cases=(
        ABAQUS_A5,
        ABAQUS_A4,
        ABAQUS_A3,
    ),

    output_directory=(
        RESULTS_DIRECTORY / "paper_abaqus_convergence_A6"
    ),

    grid_spacing_m=0.5e-3,

    abaqus_command=(
        "abaqus",
    ),

    physical_setup_confirmed=False,
)


HYBRID_ONE_LINE_RUN = RESULTS_DIRECTORY / "REPLACE_WITH_Fin_one_line_both_run"


ABAQUS_HYBRID_STUDY = AbaqusHybridStudyConfiguration(
    study_name="abaqus_one_line_hybrid_comparison",

    abaqus_case=ABAQUS_A6,

    # Both results are stored in the same "both" folder
    conventional_folder=HYBRID_ONE_LINE_RUN,
    localized_folder=HYBRID_ONE_LINE_RUN,

    output_directory=(
        RESULTS_DIRECTORY / "paper_abaqus_hybrid_comparison_A6"
    ),

    grid_spacing_m=0.5e-3,

    # Reuse exactly the same points created by the convergence study
    points_file=(
        RESULTS_DIRECTORY
        / "paper_abaqus_convergence_A6"
        / "sample_points.npy"
    ),

    abaqus_command=(
        "abaqus",
    ),

    physical_setup_confirmed=False,
)


# Set the truncation run below. Reference paths reuse the detailed study;
# configure those paths too, or supply separate compatible folders here.
TEMPORAL_TRUNCATION_STUDY = TemporalTruncationStudyConfiguration(
    study_name="fin_500_temporal_truncation",
    line_count=500,
    conventional_folder=DETAILED_500_LINE_STUDY.conventional_folder,
    localized_folder=DETAILED_500_LINE_STUDY.localized_folder,
    truncation_folder=RESULTS_DIRECTORY / "REPLACE_WITH_Fin_500_lines_truncation_100ms_temporal_truncation_run",
    output_directory=RESULTS_DIRECTORY / "paper_500_temporal_truncation",
)

# Register every configured paper study in one place.
STUDY_LIBRARY: dict[str, StudyConfiguration] = {
    "fin_500_temporal_truncation": TEMPORAL_TRUNCATION_STUDY,
    "fin_500_localization_extent": LOCALIZATION_EXTENT_STUDY,
    "fin_line_count_scaling": LINE_COUNT_STUDY,
    "fin_100_post_scan_temporal": POST_SCAN_TEMPORAL_STUDY,
    "fin_500_detailed_case": DETAILED_500_LINE_STUDY,
    "abaqus_one_line_convergence": ABAQUS_CONVERGENCE_STUDY,
    "abaqus_one_line_hybrid_comparison": ABAQUS_HYBRID_STUDY,
}
# Select only studies whose inputs you have configured, for example:
# ACTIVE_STUDIES = ("fin_500_detailed_case", "fin_500_temporal_truncation")
# For both Abaqus studies, run convergence first to create sample_points.npy:
# ACTIVE_STUDIES = ("abaqus_one_line_convergence", "abaqus_one_line_hybrid_comparison")
ACTIVE_STUDIES: tuple[str, ...] = ()


def _check_study_input_paths(configuration: StudyConfiguration) -> None:
    """Explain unconfigured/missing inputs before creating study outputs."""
    problems = []

    def visit(value, location):
        if isinstance(value, Path):
            if any(part.startswith("REPLACE_WITH_") for part in value.parts):
                problems.append(f"{location}: replace the placeholder {value}")
            elif not value.exists():
                problems.append(f"{location}: input does not exist: {value}")
        elif is_dataclass(value):
            for field in fields(value):
                if field.name != "output_directory":
                    visit(getattr(value, field.name), f"{location}.{field.name}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{location}[{index}]")

    visit(configuration, configuration.study_name)
    if problems:
        raise ValueError(
            "Configure the selected study's input paths in paper_studies.py "
            "(RESULTS_DIRECTORY and the study configuration):\n"
            + "\n".join(problems)
        )


def _safe_label(value: str) -> str:
    label = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value).strip()
    ).strip("._")
    if not label:
        raise ValueError("study and case labels must contain a letter or number")
    return label


def _case_label(case: LineCountCase) -> str:
    return case.label if case.label is not None else str(case.line_count)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty results table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(folder: Path) -> dict[str, Any] | None:
    path = folder / "run_manifest.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid run manifest: {path}")
    return payload


def _nonlocalization_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    experiment = dict(manifest.get("experiment", {}))
    experiment.pop("name", None)
    experiment.pop("localization", None)
    return experiment


def _localization_parameters_from_manifest(
    folder: Path, manifest: dict[str, Any] | None
) -> dict[str, float]:
    if manifest is None:
        raise ValueError(
            f"localization-extent studies require run_manifest.json in {folder}"
        )
    try:
        localization = manifest["experiment"]["localization"]
        parameters = {
            "core_radius": float(localization["core_radius"]),
            "transition_width": float(localization["transition_width"]),
            "retention_age": float(localization["retention_age"]),
            "transfer_duration": float(localization["transfer_duration"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid localization parameters in {folder / 'run_manifest.json'}"
        ) from exc
    if not all(np.isfinite(value) and value > 0.0 for value in parameters.values()):
        raise ValueError(
            f"localization parameters must be finite and positive in {folder}"
        )
    return parameters


def _validate_pair_manifests(
    conventional_folder: Path,
    localized_folder: Path,
    expected_line_count: int | None = None,
) -> None:
    left = _read_manifest(conventional_folder)
    right = _read_manifest(localized_folder)
    for folder, manifest in (
        (conventional_folder, left),
        (localized_folder, right),
    ):
        if manifest is not None and expected_line_count is not None:
            actual = int(manifest.get("scan_line_count", -1))
            if actual != expected_line_count:
                raise ValueError(
                    f"configured line count {expected_line_count} does not match "
                    f"manifest value {actual} in {folder}"
                )
    if left is not None and right is not None:
        if _nonlocalization_signature(left) != _nonlocalization_signature(right):
            raise ValueError(
                "conventional and localized folders have incompatible "
                "non-localization experiment configurations"
            )


def _line_count_fixed_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        experiment = dict(manifest["experiment"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid experiment data in run_manifest.json") from exc
    experiment.pop("name", None)
    experiment.pop("scan_path", None)

    # Names are descriptive labels and may include the line count even when
    # the physical material properties are identical.
    if isinstance(experiment.get("material"), dict):
        material = dict(experiment["material"])
        material.pop("name", None)
        experiment["material"] = material
    return experiment


def _different_configuration_fields(
    left: Any,
    right: Any,
    prefix: str = "experiment",
) -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(
                    _different_configuration_fields(left[key], right[key], child)
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{prefix}.length"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _different_configuration_fields(
                    left_item,
                    right_item,
                    f"{prefix}[{index}]",
                )
            )
        return differences
    return [] if left == right else [prefix]


def _validate_line_count_family(
    cases: Sequence[LineCountCase],
) -> None:
    manifests: list[tuple[LineCountCase, Path, dict[str, Any]]] = []
    for case in cases:
        conventional_folder = case.conventional_folder.expanduser().resolve()
        localized_folder = case.localized_folder.expanduser().resolve()
        _validate_pair_manifests(
            conventional_folder,
            localized_folder,
            int(case.line_count),
        )
        manifest = _read_manifest(conventional_folder)
        if manifest is None:
            raise ValueError(
                "line-count studies require run_manifest.json in "
                f"{conventional_folder}"
            )
        localized_manifest = _read_manifest(localized_folder)
        if localized_manifest is None:
            raise ValueError(
                "line-count studies require run_manifest.json in "
                f"{localized_folder}"
            )
        try:
            conventional_localization = manifest["experiment"]["localization"]
            localized_localization = localized_manifest["experiment"][
                "localization"
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid localization data for line-count case {case.line_count}"
            ) from exc
        if conventional_localization != localized_localization:
            raise ValueError(
                f"line-count case {case.line_count} uses different localization "
                "parameters in its conventional and localized result folders"
            )
        manifests.append((case, conventional_folder, manifest))

    reference_case, _, reference_manifest = manifests[0]
    reference_signature = _line_count_fixed_signature(reference_manifest)
    for case, folder, manifest in manifests[1:]:
        case_signature = _line_count_fixed_signature(manifest)
        if case_signature != reference_signature:
            differing_fields = _different_configuration_fields(
                reference_signature,
                case_signature,
            )
            details = ", ".join(differing_fields[:8])
            if len(differing_fields) > 8:
                details += f", and {len(differing_fields) - 8} more"
            raise ValueError(
                f"line-count case {case.line_count} in {folder} changes a "
                f"fixed study setting relative to case {reference_case.line_count}; "
                "only the scan-path prefix and resulting line count may change. "
                f"Differing fields: {details}"
            )

    master_case, master_folder, master_manifest = max(
        manifests,
        key=lambda item: int(item[0].line_count),
    )
    try:
        master_path = list(master_manifest["experiment"]["scan_path"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"invalid scan path in {master_folder / 'run_manifest.json'}"
        ) from exc
    if len(master_path) != int(master_case.line_count):
        raise ValueError(
            f"scan path in {master_folder} contains {len(master_path)} entries, "
            f"but the configured line count is {master_case.line_count}"
        )

    for case, folder, manifest in manifests:
        try:
            scan_path = list(manifest["experiment"]["scan_path"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid scan path in {folder / 'run_manifest.json'}"
            ) from exc
        count = int(case.line_count)
        if len(scan_path) != count:
            raise ValueError(
                f"scan path in {folder} contains {len(scan_path)} entries, "
                f"but the configured line count is {count}"
            )
        if scan_path != master_path[:count]:
            raise ValueError(
                f"line-count case {count} in {folder} is not a prefix of the "
                f"{master_case.line_count}-line master scan path"
            )


def _scan_time_bounds(
    folder: Path,
    manifest: dict[str, Any] | None,
) -> tuple[float, float]:
    if manifest is None:
        raise ValueError(f"run_manifest.json is required in {folder}")
    try:
        scan_path = list(manifest["experiment"]["scan_path"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"invalid scan path in {folder / 'run_manifest.json'}"
        ) from exc
    if not scan_path:
        raise ValueError(f"scan path is empty in {folder / 'run_manifest.json'}")

    starts: list[float] = []
    ends: list[float] = []
    for index, segment in enumerate(scan_path):
        try:
            start_time = float(segment["start_time"])
            if "end_time" in segment:
                end_time = float(segment["end_time"])
            else:
                start = np.asarray(segment["start_point"], dtype=float)
                end = np.asarray(segment["end_point"], dtype=float)
                velocity = float(segment["velocity"])
                if start.shape != end.shape or start.ndim != 1 or velocity <= 0.0:
                    raise ValueError
                end_time = start_time + float(np.linalg.norm(end - start)) / velocity
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid scan-path segment {index} in "
                f"{folder / 'run_manifest.json'}"
            ) from exc
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError(
                f"non-finite scan time in segment {index} of {folder}"
            )
        starts.append(start_time)
        ends.append(end_time)
    return min(starts), max(ends)


def _nearest_time_index(times: np.ndarray, requested_time: float) -> int:
    values = np.asarray(times, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("simulation time array must be nonempty and one-dimensional")
    if requested_time < values[0] or requested_time > values[-1]:
        raise ValueError(
            f"requested time {requested_time:g} s is outside the saved simulation "
            f"interval [{values[0]:g}, {values[-1]:g}] s"
        )
    return int(np.argmin(np.abs(values - requested_time)))


_TIMING_COMPONENTS = (
    "localization",
    "analytical",
    "residual_assembly",
    "boundary_assembly",
    "fea",
)


def _cumulative_timing(run: HybridRunResult, component: str) -> np.ndarray:
    return np.cumsum(
        np.asarray(
            [float(getattr(item, component)) for item in run.timing_history],
            dtype=float,
        )
    )


def _cumulative_recorded_time(run: HybridRunResult) -> np.ndarray:
    values = np.zeros(len(run.timing_history), dtype=float)
    for component in _TIMING_COMPONENTS:
        values += np.asarray(
            [float(getattr(item, component)) for item in run.timing_history],
            dtype=float,
        )
    return np.cumsum(values)


def _cumulative_evaluations(run: HybridRunResult) -> np.ndarray:
    return np.cumsum(
        np.asarray(
            [
                int(item.analytical_node_point_evaluations)
                + int(item.analytical_boundary_point_evaluations)
                + int(item.analytical_volume_point_evaluations)
                for item in run.evaluation_history
            ],
            dtype=np.int64,
        )
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float("inf")


def _evaluation_summary(run: HybridRunResult) -> dict[str, int]:
    node = sum(
        int(item.analytical_node_point_evaluations)
        for item in run.evaluation_history
    )
    boundary = sum(
        int(item.analytical_boundary_point_evaluations)
        for item in run.evaluation_history
    )
    volume = sum(
        int(item.analytical_volume_point_evaluations)
        for item in run.evaluation_history
    )
    return {
        "node_evaluations": node,
        "boundary_evaluations": boundary,
        "volume_evaluations": volume,
        "total_evaluations": node + boundary + volume,
        "maximum_started_lines": max(
            (int(item.started_line_count) for item in run.evaluation_history),
            default=0,
        ),
        "maximum_retained_lines": max(
            (
                int(item.explicitly_retained_line_count)
                for item in run.evaluation_history
            ),
            default=0,
        ),
    }


def _load_paper_run(folder: Path, mode: HybridMode) -> HybridRunResult:
    """Load physical Celsius fields directly from a new simulation."""
    run = LocalizedHybridEvaluator.load_run(folder, mode)
    validate_temperature_metadata(run.metadata)
    return run


def _load_pair(
    conventional_folder: Path, localized_folder: Path
) -> tuple[HybridRunResult, HybridRunResult, LocalizedHybridEvaluator]:
    conventional = _load_paper_run(
        conventional_folder, HybridMode.CONVENTIONAL
    )
    localized = _load_paper_run(
        localized_folder, HybridMode.LOCALIZED
    )
    evaluator = LocalizedHybridEvaluator(conventional, localized)
    return conventional, localized, evaluator


def _get_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
    })
    return plt


def _finish_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    _get_pyplot().close(figure)


def _plot_two_curves(
    x: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_label: str,
    right_label: str,
    x_label: str,
    y_label: str,
    path: Path,
    scientific_y: bool = False,
) -> None:
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.plot(x, left, "o-", label=left_label)
    axis.plot(x, right, "s-", label=right_label)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if scientific_y:
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axis.grid(alpha=0.25)
    axis.legend()
    _finish_figure(figure, path)


def _plot_single_curve(
    x: np.ndarray,
    values: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    path: Path,
) -> None:
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.plot(x, values, "o-")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25)
    _finish_figure(figure, path)


def _plot_main_case_histories(
    evaluator: LocalizedHybridEvaluator, output: Path
) -> None:
    plt = _get_pyplot()
    errors = evaluator.error_vs_time()
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.plot(errors["time"], 100.0 * errors["relative_nodal_2_norm"])
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Relative nodal 2-norm difference (%)")
    axis.grid(alpha=0.25)
    _finish_figure(figure, output / "temperature_difference_vs_time.png")

    history = evaluator.localized.evaluation_history
    times = np.asarray([item.time for item in history], dtype=float)
    started = np.asarray([item.started_line_count for item in history], dtype=int)
    retained = np.asarray(
        [item.explicitly_retained_line_count for item in history], dtype=int
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.plot(times, started, label="Started scan lines")
    axis.plot(times, retained, linestyle="--", label="Retained analytical scan lines")
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Number of scan lines")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02),
                ncol=2, frameon=False, borderaxespad=0.0,
                fontsize=12, columnspacing=1.5)
    _finish_figure(figure, output / "started_vs_retained_lines.png")

    components = (
        "analytical",
        "fea",
        "localization",
        "residual_assembly",
        "boundary_assembly",
    )
    component_labels = (
        "Analytical\nevaluation",
        "FE solution",
        "Localization",
        "Residual\nassembly",
        "Boundary\nassembly",
    )
    conventional_timing = evaluator.timing_summary(evaluator.conventional)
    localized_timing = evaluator.timing_summary(evaluator.localized)
    x = np.arange(len(components))
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.bar(
        x - 0.2,
        [conventional_timing[name] for name in components],
        width=0.4,
        label="Conventional hybrid",
    )
    axis.bar(
        x + 0.2,
        [localized_timing[name] for name in components],
        width=0.4,
        label="Localized hybrid",
    )
    axis.set_xticks(x, component_labels)
    axis.set_ylabel("Accumulated time (s)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _finish_figure(figure, output / "computational_time_breakdown.png")

    selected_components = ("analytical", "fea")
    selected_labels = ("Analytical evaluation", "FE solution")
    x = np.arange(len(selected_components))
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    axis.bar(
        x - 0.2,
        [conventional_timing[name] for name in selected_components],
        width=0.4,
        label="Conventional hybrid",
    )
    axis.bar(
        x + 0.2,
        [localized_timing[name] for name in selected_components],
        width=0.4,
        label="Localized hybrid",
    )
    axis.set_xticks(x, selected_labels)
    axis.set_ylabel("Accumulated time (s)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _finish_figure(
        figure,
        output / "analytical_and_fe_time_comparison.png",
    )

    grouped_labels = (
        "Analytical evaluation",
        "FE solution",
        "Localization and assembly",
    )
    conventional_grouped = (
        conventional_timing["analytical"],
        conventional_timing["fea"],
        conventional_timing["localization"]
        + conventional_timing["residual_assembly"]
        + conventional_timing["boundary_assembly"],
    )
    localized_grouped = (
        localized_timing["analytical"],
        localized_timing["fea"],
        localized_timing["localization"]
        + localized_timing["residual_assembly"]
        + localized_timing["boundary_assembly"],
    )
    x = np.arange(len(grouped_labels))
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    axis.bar(
        x - 0.2,
        conventional_grouped,
        width=0.4,
        label="Conventional hybrid",
    )
    axis.bar(
        x + 0.2,
        localized_grouped,
        width=0.4,
        label="Localized hybrid",
    )
    axis.set_xticks(x, grouped_labels)
    axis.set_ylabel("Accumulated time (s)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _finish_figure(
        figure,
        output / "grouped_computational_time_comparison.png",
    )


def _load_mesh_data(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    path = folder / "mesh_data.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing mesh data required for field plots: {path}"
        )
    with np.load(path, allow_pickle=False) as data:
        required = {"node_coordinates", "surface_triangles"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return (
            data["node_coordinates"].copy(),
            data["surface_triangles"].copy(),
        )


def _set_equal_3d_aspect(axis: Any, nodes: np.ndarray) -> None:
    ranges = np.ptp(nodes, axis=0)
    ranges = np.maximum(ranges, np.max(ranges) * 1.0e-6)
    axis.set_box_aspect(ranges)


def _add_surface_field(
    axis: Any,
    nodes: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    show_coordinates: bool = True,
) -> Any:
    from matplotlib import colormaps, colors
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plot_nodes = 1.0e3 * nodes
    face_values = np.mean(values[triangles], axis=1)
    normalizer = colors.Normalize(vmin=vmin, vmax=vmax)
    colormap = colormaps.get_cmap(cmap)
    collection = Poly3DCollection(
        plot_nodes[triangles],
        facecolors=colormap(normalizer(face_values)),
        linewidths=0.0,
        antialiased=False,
    )
    axis.add_collection3d(collection)
    axis.auto_scale_xyz(
        plot_nodes[:, 0],
        plot_nodes[:, 1],
        plot_nodes[:, 2],
    )
    _set_equal_3d_aspect(axis, plot_nodes)
    axis.set_proj_type("ortho")
    axis.view_init(elev=22.0, azim=-62.0)
    axis.set_title(title, pad=8.0, fontsize=18)
    for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate_axis.set_major_locator(MaxNLocator(nbins=4))
    axis.tick_params(axis="both", which="major", labelsize=15, pad=0)
    if show_coordinates:
        axis.set_xlabel("x (mm)", labelpad=4.0, fontsize=18)
        axis.set_ylabel("y (mm)", labelpad=4.0, fontsize=18)
        axis.set_zlabel("z (mm)", labelpad=4.0, fontsize=18)
    else:
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_zlabel("")
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_zticklabels([])
    return ScalarMappable(norm=normalizer, cmap=colormap)


def _plot_field_comparisons(
    evaluator: LocalizedHybridEvaluator,
    mesh_folder: Path,
    requested_times: Sequence[float],
    output: Path,
) -> None:
    if not requested_times:
        return
    nodes, triangles = _load_mesh_data(mesh_folder)
    if not np.array_equal(nodes, evaluator.conventional.node_coordinates):
        raise ValueError("mesh_data.npz does not match the saved temperature nodes")
    plt = _get_pyplot()
    for requested_time in requested_times:
        index = int(
            np.argmin(np.abs(evaluator.conventional.saved_times - requested_time))
        )
        actual_time = float(evaluator.conventional.saved_times[index])
        conventional = evaluator.conventional.total_temperature[index]
        localized = evaluator.localized.total_temperature[index]
        absolute_error = np.abs(localized - conventional)
        temperature_min = float(min(np.min(conventional), np.min(localized)))
        temperature_max = float(max(np.max(conventional), np.max(localized)))
        if temperature_max <= temperature_min:
            temperature_max = temperature_min + 1.0
        error_max = max(float(np.max(absolute_error)), np.finfo(float).eps)

        figure = plt.figure(figsize=(17.0, 5.0))
        grid = figure.add_gridspec(
            1,
            6,
            width_ratios=(1.0, 1.0, 0.045, 0.14, 1.0, 0.045),
            wspace=0.10,
        )
        axes = [
            figure.add_subplot(grid[0, 0], projection="3d"),
            figure.add_subplot(grid[0, 1], projection="3d"),
            figure.add_subplot(grid[0, 4], projection="3d"),
        ]
        temperature_color_axis = figure.add_subplot(grid[0, 2])
        error_color_axis = figure.add_subplot(grid[0, 5])
        temperature_map = _add_surface_field(
            axes[0], nodes, triangles, conventional,
            title="Conventional", vmin=temperature_min, vmax=temperature_max,
            cmap="inferno",
        )
        _add_surface_field(
            axes[1], nodes, triangles, localized,
            title="Localized", vmin=temperature_min, vmax=temperature_max,
            cmap="inferno", show_coordinates=False,
        )
        error_map = _add_surface_field(
            axes[2], nodes, triangles, absolute_error,
            title="Absolute error", vmin=0.0, vmax=error_max,
            cmap="viridis", show_coordinates=False,
        )
        temperature_colorbar = figure.colorbar(
            temperature_map,
            cax=temperature_color_axis,
            label="Temperature (°C)",
        )
        temperature_colorbar.ax.yaxis.set_label_position("left")
        error_colorbar = figure.colorbar(
            error_map,
            cax=error_color_axis,
            label="Absolute error (°C)",
        )
        for colorbar in (temperature_colorbar, error_colorbar):
            colorbar.ax.yaxis.label.set_size(18)
            colorbar.ax.tick_params(labelsize=15)
        figure.suptitle(f"Simulation time = {actual_time:.6g} s", fontsize=18)
        figure.subplots_adjust(
            left=0.025,
            right=0.975,
            bottom=0.04,
            top=0.86,
        )
        time_label = _safe_label(f"{actual_time:.9g}")
        figure.savefig(
            output / f"temperature_fields_time_{time_label}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)


def run_line_count_study(
    configuration: LineCountStudyConfiguration,
) -> Path:
    if not configuration.cases:
        raise ValueError("line-count study requires at least one case")
    counts = [int(case.line_count) for case in configuration.cases]
    if any(count <= 0 for count in counts) or len(set(counts)) != len(counts):
        raise ValueError("line counts must be positive and unique")
    labels = [_case_label(case) for case in configuration.cases]
    if len(set(labels)) != len(labels):
        raise ValueError("line-count case labels must be unique")

    output = configuration.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ordered_cases = sorted(configuration.cases, key=lambda case: case.line_count)
    _validate_line_count_family(ordered_cases)
    rows: list[dict[str, Any]] = []

    for case in ordered_cases:
        conventional_folder = case.conventional_folder.expanduser().resolve()
        localized_folder = case.localized_folder.expanduser().resolve()
        conventional, localized, evaluator = _load_pair(
            conventional_folder, localized_folder
        )
        metrics = evaluator.compute_metrics()
        conventional_timing = evaluator.timing_summary(conventional)
        localized_timing = evaluator.timing_summary(localized)
        conventional_evaluations = _evaluation_summary(conventional)
        localized_evaluations = _evaluation_summary(localized)
        label = _case_label(case)
        rows.append(
            {
                "case_label": label,
                "line_count": int(case.line_count),
                "relative_nodal_difference_percent": (
                    100.0 * metrics.relative_nodal_2_norm_temperature_error
                ),
                "rmse_temperature_difference_celsius": (
                    metrics.rmse_temperature_error
                ),
                "maximum_absolute_difference_celsius": (
                    metrics.max_abs_temperature_error
                ),
                "conventional_overall_time_seconds": (
                    metrics.conventional_wall_time
                ),
                "localized_overall_time_seconds": metrics.localized_wall_time,
                "overall_speedup": metrics.total_wall_time_speedup,
                "conventional_analytical_time_seconds": (
                    metrics.conventional_analytical_time
                ),
                "localized_analytical_time_seconds": (
                    metrics.localized_analytical_time
                ),
                "analytical_speedup": metrics.analytical_time_speedup,
                "conventional_fea_time_seconds": metrics.conventional_fea_time,
                "localized_fea_time_seconds": metrics.localized_fea_time,
                "localized_localization_time_seconds": (
                    localized_timing["localization"]
                ),
                "localized_residual_assembly_time_seconds": (
                    localized_timing["residual_assembly"]
                ),
                "localized_boundary_assembly_time_seconds": (
                    localized_timing["boundary_assembly"]
                ),
                "conventional_analytical_evaluations": (
                    conventional_evaluations["total_evaluations"]
                ),
                "localized_analytical_evaluations": (
                    localized_evaluations["total_evaluations"]
                ),
                "maximum_retained_lines": (
                    localized_evaluations["maximum_retained_lines"]
                ),
                "conventional_folder": str(conventional_folder),
                "localized_folder": str(localized_folder),
            }
        )

    _write_csv(output / "line_count_study_results.csv", rows)
    _write_json(
        output / "line_count_study_configuration.json", configuration
    )
    line_counts = np.asarray([row["line_count"] for row in rows], dtype=float)
    _plot_two_curves(
        line_counts,
        np.asarray(
            [row["conventional_analytical_time_seconds"] for row in rows]
        ),
        np.asarray([row["localized_analytical_time_seconds"] for row in rows]),
        left_label="Conventional hybrid",
        right_label="Localized hybrid",
        x_label=r"Number of scan lines, $N$",
        y_label="Accumulated analytical time (s)",
        path=output / "analytical_time_vs_line_count.png",
    )
    _plot_two_curves(
        line_counts,
        np.asarray(
            [row["conventional_analytical_evaluations"] for row in rows],
            dtype=float,
        ),
        np.asarray(
            [row["localized_analytical_evaluations"] for row in rows],
            dtype=float,
        ),
        left_label="Conventional hybrid",
        right_label="Localized hybrid",
        x_label=r"Number of scan lines, $N$",
        y_label="Analytical line-point evaluations",
        path=output / "analytical_evaluations_vs_line_count.png",
        scientific_y=True,
    )
    _plot_single_curve(
        line_counts,
        np.asarray([row["analytical_speedup"] for row in rows]),
        x_label=r"Number of scan lines, $N$",
        y_label="Analytical speedup",
        path=output / "analytical_speedup_vs_line_count.png",
    )
    _plot_single_curve(
        line_counts,
        np.asarray([row["overall_speedup"] for row in rows]),
        x_label=r"Number of scan lines, $N$",
        y_label="Overall speedup",
        path=output / "overall_speedup_vs_line_count.png",
    )
    _plot_single_curve(
        line_counts,
        np.asarray([row["relative_nodal_difference_percent"] for row in rows]),
        x_label=r"Number of scan lines, $N$",
        y_label="Relative nodal 2-norm difference (%)",
        path=output / "temperature_difference_vs_line_count.png",
    )

    print(f"Line-count study saved to: {output}")
    return output


def run_localization_extent_study(
    configuration: LocalizationExtentStudyConfiguration,
) -> Path:
    if len(configuration.cases) < 2:
        raise ValueError("localization-extent studies require at least two cases")
    scales = [float(case.scale) for case in configuration.cases]
    labels = [case.label for case in configuration.cases]
    if (
        not all(np.isfinite(scale) and scale > 0.0 for scale in scales)
        or len(set(scales)) != len(scales)
    ):
        raise ValueError("localization scales must be positive, finite, and unique")
    if len(set(labels)) != len(labels):
        raise ValueError("localization-scale labels must be unique")

    output = configuration.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    conventional_folder = configuration.conventional_folder.expanduser().resolve()
    conventional = _load_paper_run(
        conventional_folder, HybridMode.CONVENTIONAL
    )
    conventional_manifest = _read_manifest(conventional_folder)
    if conventional_manifest is None:
        raise ValueError(
            "localization-extent studies require run_manifest.json in the "
            f"conventional folder: {conventional_folder}"
        )
    ordered_cases = sorted(configuration.cases, key=lambda case: case.scale)
    rows: list[dict[str, Any]] = []
    normalized_reference: np.ndarray | None = None

    for case in ordered_cases:
        localized_folder = case.localized_folder.expanduser().resolve()
        localized = _load_paper_run(
            localized_folder, HybridMode.LOCALIZED
        )
        evaluator = LocalizedHybridEvaluator(conventional, localized)
        localized_manifest = _read_manifest(localized_folder)
        if localized_manifest is None:
            raise ValueError(
                "localization-extent studies require run_manifest.json in "
                f"the localized folder: {localized_folder}"
            )
        if _nonlocalization_signature(conventional_manifest) != (
            _nonlocalization_signature(localized_manifest)
        ):
            raise ValueError(
                f"localization case {case.label} changes configuration outside "
                "the localization parameters"
            )
        parameters = _localization_parameters_from_manifest(
            localized_folder, localized_manifest
        )
        parameter_vector = np.asarray(
            [
                parameters["core_radius"],
                parameters["transition_width"],
                parameters["retention_age"],
                parameters["transfer_duration"],
            ],
            dtype=float,
        )
        normalized = parameter_vector / float(case.scale)
        if normalized_reference is None:
            normalized_reference = normalized
        elif not np.allclose(
            normalized,
            normalized_reference,
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"localization parameters in {localized_folder} are not "
                f"consistent with the declared scale {case.scale:g}"
            )
        metrics = evaluator.compute_metrics()
        timing = evaluator.timing_summary(localized)
        evaluations = _evaluation_summary(localized)
        rows.append(
            {
                "case_label": case.label,
                "localization_scale": float(case.scale),
                "core_radius_mm": 1.0e3 * parameters["core_radius"],
                "transition_width_mm": 1.0e3 * parameters["transition_width"],
                "retention_age_ms": 1.0e3 * parameters["retention_age"],
                "transfer_duration_ms": (
                    1.0e3 * parameters["transfer_duration"]
                ),
                "relative_nodal_difference_percent": (
                    100.0 * metrics.relative_nodal_2_norm_temperature_error
                ),
                "maximum_absolute_difference_celsius": (
                    metrics.max_abs_temperature_error
                ),
                "localized_analytical_time_seconds": (
                    metrics.localized_analytical_time
                ),
                "conventional_analytical_time_seconds": (
                    metrics.conventional_analytical_time
                ),
                "analytical_time_speedup": metrics.analytical_time_speedup,
                "localized_analytical_evaluations": (
                    evaluations["total_evaluations"]
                ),
                "maximum_retained_lines": (
                    evaluations["maximum_retained_lines"]
                ),
                "localized_simulation_loop_time_seconds": (
                    metrics.localized_wall_time
                ),
                "simulation_loop_speedup": metrics.total_wall_time_speedup,
                "localized_localization_time_seconds": timing["localization"],
                "localized_residual_assembly_time_seconds": (
                    timing["residual_assembly"]
                ),
                "localized_folder": str(localized_folder),
            }
        )

    _write_csv(output / "localization_extent_study_results.csv", rows)
    _write_json(
        output / "localization_extent_study_configuration.json", configuration
    )
    x = np.asarray([row["localization_scale"] for row in rows], dtype=float)
    plt = _get_pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.4))

    axes[0].plot(
        x,
        [row["relative_nodal_difference_percent"] for row in rows],
        "o-",
    )
    axes[0].set_xlabel(r"Localization scale, $\lambda$")
    axes[0].set_ylabel("Relative nodal 2-norm difference (%)")
    # axes[0].set_title("(a) Agreement with conventional hybrid")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        x,
        [row["localized_analytical_evaluations"] for row in rows],
        "o-",
    )
    axes[1].set_xlabel(r"Localization scale, $\lambda$")
    axes[1].set_ylabel("Analytical line-point evaluations")
    # axes[1].set_title("(b) Analytical work")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        x,
        [row["analytical_time_speedup"] for row in rows],
        "o-",
    )
    axes[2].set_xlabel(r"Localization scale, $\lambda$")
    axes[2].set_ylabel("Analytical-time speedup")
    # axes[2].set_title("(c) Analytical speedup")
    axes[2].grid(alpha=0.25)

    # This three-panel figure is reduced to the paper's text width.
    # Set text sizes here so its labels remain readable after reduction.
    for axis in axes:
        axis.xaxis.label.set_size(18)
        axis.yaxis.label.set_size(18)
        axis.tick_params(axis="both", which="major", labelsize=16)
        axis.yaxis.get_offset_text().set_fontsize(16)

    _finish_figure(
        figure,
        output / "localization_extent_sensitivity.png",
    )

    for key, y_label, filename, scientific_y in (
        (
            "relative_nodal_difference_percent",
            "Relative nodal 2-norm difference (%)",
            "localization_extent_difference.png",
            False,
        ),
        (
            "localized_analytical_evaluations",
            "Analytical line-point evaluations",
            "localization_extent_evaluations.png",
            True,
        ),
        (
            "analytical_time_speedup",
            "Analytical-time speedup",
            "localization_extent_speedup.png",
            False,
        ),
    ):
        figure, axis = plt.subplots(figsize=(7.4, 4.5))
        axis.plot(x, [row[key] for row in rows], "o-")
        axis.set_xlabel(r"Localization scale, $\lambda$")
        axis.set_ylabel(y_label)
        if scientific_y:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.grid(alpha=0.25)
        _finish_figure(figure, output / filename)

    print(f"Localization-extent study saved to: {output}")
    return output


def run_post_scan_temporal_study(
    configuration: PostScanTemporalStudyConfiguration,
) -> Path:
    if configuration.line_count <= 0:
        raise ValueError("post-scan study line count must be positive")
    durations = np.asarray(configuration.post_scan_durations, dtype=float)
    if (
        durations.ndim != 1
        or durations.size == 0
        or not np.all(np.isfinite(durations))
        or np.any(durations < 0.0)
        or len(set(durations.tolist())) != durations.size
    ):
        raise ValueError(
            "post-scan durations must be finite, nonnegative, and unique"
        )
    durations = np.sort(durations)

    conventional_folder = configuration.conventional_folder.expanduser().resolve()
    localized_folder = configuration.localized_folder.expanduser().resolve()
    _validate_pair_manifests(
        conventional_folder,
        localized_folder,
        configuration.line_count,
    )
    conventional, localized, _ = _load_pair(
        conventional_folder,
        localized_folder,
    )
    manifest = _read_manifest(conventional_folder)
    _, scan_completion_time = _scan_time_bounds(conventional_folder, manifest)

    if len(conventional.timing_history) != conventional.times.size or len(
        localized.timing_history
    ) != localized.times.size:
        raise ValueError("timing-history length does not match the simulation grid")
    if len(conventional.evaluation_history) != conventional.times.size or len(
        localized.evaluation_history
    ) != localized.times.size:
        raise ValueError("evaluation-history length does not match the simulation grid")

    conventional_analytical = _cumulative_timing(conventional, "analytical")
    localized_analytical = _cumulative_timing(localized, "analytical")
    conventional_overall = _cumulative_recorded_time(conventional)
    localized_overall = _cumulative_recorded_time(localized)
    conventional_evaluations = _cumulative_evaluations(conventional)
    localized_evaluations = _cumulative_evaluations(localized)

    rows: list[dict[str, Any]] = []
    for requested_duration in durations:
        requested_time = scan_completion_time + float(requested_duration)
        conventional_index = _nearest_time_index(
            conventional.times,
            requested_time,
        )
        localized_index = _nearest_time_index(localized.times, requested_time)
        conventional_time = float(conventional.times[conventional_index])
        localized_time = float(localized.times[localized_index])
        if not np.isclose(conventional_time, localized_time, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "conventional and localized post-scan checkpoints do not align"
            )
        rows.append(
            {
                "requested_post_scan_duration_seconds": float(requested_duration),
                "actual_post_scan_duration_seconds": (
                    conventional_time - scan_completion_time
                ),
                "simulation_time_seconds": conventional_time,
                "conventional_cumulative_analytical_time_seconds": float(
                    conventional_analytical[conventional_index]
                ),
                "localized_cumulative_analytical_time_seconds": float(
                    localized_analytical[localized_index]
                ),
                "analytical_speedup": _safe_ratio(
                    float(conventional_analytical[conventional_index]),
                    float(localized_analytical[localized_index]),
                ),
                "conventional_cumulative_overall_time_seconds": float(
                    conventional_overall[conventional_index]
                ),
                "localized_cumulative_overall_time_seconds": float(
                    localized_overall[localized_index]
                ),
                "overall_speedup": _safe_ratio(
                    float(conventional_overall[conventional_index]),
                    float(localized_overall[localized_index]),
                ),
                "conventional_cumulative_analytical_evaluations": int(
                    conventional_evaluations[conventional_index]
                ),
                "localized_cumulative_analytical_evaluations": int(
                    localized_evaluations[localized_index]
                ),
            }
        )

    output = configuration.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "post_scan_temporal_study_results.csv", rows)
    _write_json(
        output / "post_scan_temporal_study_configuration.json",
        configuration,
    )
    x = np.asarray(
        [row["actual_post_scan_duration_seconds"] for row in rows],
        dtype=float,
    )
    _plot_two_curves(
        x,
        np.asarray(
            [
                row["conventional_cumulative_analytical_time_seconds"]
                for row in rows
            ]
        ),
        np.asarray(
            [row["localized_cumulative_analytical_time_seconds"] for row in rows]
        ),
        left_label="Conventional hybrid",
        right_label="Localized hybrid",
        x_label="Post-scan duration (s)",
        y_label="Cumulative analytical time (s)",
        path=output / "cumulative_analytical_time_vs_post_scan_duration.png",
    )
    _plot_two_curves(
        x,
        np.asarray(
            [
                row["conventional_cumulative_analytical_evaluations"]
                for row in rows
            ],
            dtype=float,
        ),
        np.asarray(
            [
                row["localized_cumulative_analytical_evaluations"]
                for row in rows
            ],
            dtype=float,
        ),
        left_label="Conventional hybrid",
        right_label="Localized hybrid",
        x_label="Post-scan duration (s)",
        y_label="Cumulative analytical line-point evaluations",
        path=output / "cumulative_analytical_evaluations_vs_post_scan_duration.png",
        scientific_y=True,
    )
    _plot_single_curve(
        x,
        np.asarray([row["analytical_speedup"] for row in rows]),
        x_label="Post-scan duration (s)",
        y_label="Analytical speedup",
        path=output / "analytical_speedup_vs_post_scan_duration.png",
    )
    _plot_single_curve(
        x,
        np.asarray([row["overall_speedup"] for row in rows]),
        x_label="Post-scan duration (s)",
        y_label="Overall speedup",
        path=output / "overall_speedup_vs_post_scan_duration.png",
    )

    print(f"Post-scan temporal study saved to: {output}")
    return output


def run_detailed_case_study(
    configuration: DetailedCaseStudyConfiguration,
) -> Path:
    if configuration.line_count <= 0:
        raise ValueError("detailed-case line count must be positive")
    fractions = np.asarray(configuration.scan_time_fractions, dtype=float)
    post_scan_times = np.asarray(configuration.post_scan_field_times, dtype=float)
    if (
        fractions.ndim != 1
        or fractions.size == 0
        or not np.all(np.isfinite(fractions))
        or np.any(fractions < 0.0)
        or np.any(fractions > 1.0)
        or len(set(fractions.tolist())) != fractions.size
    ):
        raise ValueError(
            "scan-time fractions must be finite, unique, and between zero and one"
        )
    if (
        post_scan_times.ndim != 1
        or not np.all(np.isfinite(post_scan_times))
        or np.any(post_scan_times < 0.0)
        or len(set(post_scan_times.tolist())) != post_scan_times.size
    ):
        raise ValueError(
            "post-scan field times must be finite, nonnegative, and unique"
        )

    conventional_folder = configuration.conventional_folder.expanduser().resolve()
    localized_folder = configuration.localized_folder.expanduser().resolve()
    _validate_pair_manifests(
        conventional_folder,
        localized_folder,
        configuration.line_count,
    )
    conventional, localized, evaluator = _load_pair(
        conventional_folder,
        localized_folder,
    )
    manifest = _read_manifest(conventional_folder)
    scan_start_time, scan_completion_time = _scan_time_bounds(
        conventional_folder,
        manifest,
    )
    scan_duration = scan_completion_time - scan_start_time
    requested_field_times = [
        scan_start_time + float(fraction) * scan_duration
        for fraction in fractions
    ]
    requested_field_times.extend(
        scan_completion_time + float(duration) for duration in post_scan_times
    )

    output = configuration.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluator.compute_metrics()
    conventional_timing = evaluator.timing_summary(conventional)
    localized_timing = evaluator.timing_summary(localized)
    conventional_evaluations = _evaluation_summary(conventional)
    localized_evaluations = _evaluation_summary(localized)
    summary = {
        "line_count": configuration.line_count,
        "scan_start_time_seconds": scan_start_time,
        "scan_completion_time_seconds": scan_completion_time,
        "relative_nodal_difference_percent": (
            100.0 * metrics.relative_nodal_2_norm_temperature_error
        ),
        "rmse_temperature_difference_celsius": metrics.rmse_temperature_error,
        "maximum_absolute_difference_celsius": metrics.max_abs_temperature_error,
        "conventional_overall_time_seconds": metrics.conventional_wall_time,
        "localized_overall_time_seconds": metrics.localized_wall_time,
        "overall_speedup": metrics.total_wall_time_speedup,
        "conventional_analytical_time_seconds": (
            metrics.conventional_analytical_time
        ),
        "localized_analytical_time_seconds": metrics.localized_analytical_time,
        "analytical_speedup": metrics.analytical_time_speedup,
        "conventional_fea_time_seconds": metrics.conventional_fea_time,
        "localized_fea_time_seconds": metrics.localized_fea_time,
        "localized_localization_time_seconds": localized_timing["localization"],
        "localized_residual_assembly_time_seconds": (
            localized_timing["residual_assembly"]
        ),
        "localized_boundary_assembly_time_seconds": (
            localized_timing["boundary_assembly"]
        ),
        "conventional_analytical_evaluations": (
            conventional_evaluations["total_evaluations"]
        ),
        "localized_analytical_evaluations": (
            localized_evaluations["total_evaluations"]
        ),
        "maximum_retained_analytical_lines": (
            localized_evaluations["maximum_retained_lines"]
        ),
        "conventional_folder": str(conventional_folder),
        "localized_folder": str(localized_folder),
    }
    _write_csv(output / "detailed_case_summary.csv", [summary])
    _write_json(output / "detailed_case_configuration.json", configuration)

    errors = evaluator.error_vs_time()
    _write_csv(
        output / "temperature_difference_history.csv",
        [
            {
                "simulation_time_seconds": float(time_value),
                "relative_nodal_2_norm_difference_percent": float(
                    100.0 * relative
                ),
                "rmse_temperature_difference_celsius": float(rmse),
                "maximum_absolute_difference_celsius": float(maximum),
            }
            for time_value, relative, rmse, maximum in zip(
                errors["time"],
                errors["relative_nodal_2_norm"],
                errors["rmse"],
                errors["max_abs"],
            )
        ],
    )
    _write_csv(
        output / "retained_line_history.csv",
        [
            {
                "simulation_time_seconds": float(item.time),
                "started_scan_lines": int(item.started_line_count),
                "retained_analytical_scan_lines": int(
                    item.explicitly_retained_line_count
                ),
                "transferred_scan_lines": int(item.transferred_line_count),
            }
            for item in localized.evaluation_history
        ],
    )
    _write_csv(
        output / "computational_time_breakdown.csv",
        [
            {
                "component": component,
                "conventional_hybrid_seconds": conventional_timing[component],
                "localized_hybrid_seconds": localized_timing[component],
            }
            for component in _TIMING_COMPONENTS
        ],
    )
    field_rows: list[dict[str, Any]] = []
    for requested_time in requested_field_times:
        index = _nearest_time_index(conventional.saved_times, requested_time)
        field_rows.append(
            {
                "requested_simulation_time_seconds": requested_time,
                "actual_saved_time_seconds": float(conventional.saved_times[index]),
            }
        )
    _write_csv(output / "temperature_field_times.csv", field_rows)

    _plot_main_case_histories(evaluator, output)
    _plot_field_comparisons(
        evaluator,
        conventional_folder,
        requested_field_times,
        output,
    )

    print(f"Detailed case study saved to: {output}")
    return output


def _truncation_physical_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    # Working directories and STL locations can change without changing physics.
    signature = _nonlocalization_signature(manifest)
    signature.pop("paths", None)
    mesh = dict(signature.get("mesh", {}))
    mesh.pop("stl_path", None)
    signature["mesh"] = mesh
    return signature


def run_temporal_truncation_study(
    configuration: TemporalTruncationStudyConfiguration,
) -> Path:
    folders = (configuration.conventional_folder, configuration.localized_folder,
               configuration.truncation_folder)
    modes = (HybridMode.CONVENTIONAL, HybridMode.LOCALIZED,
             HybridMode.TEMPORAL_TRUNCATION)
    manifests = [_read_manifest(folder) for folder in folders]
    for folder, manifest in zip(folders, manifests):
        if manifest is None:
            raise ValueError(f"run_manifest.json is required in {folder}")
        if int(manifest.get("scan_line_count", -1)) != configuration.line_count:
            raise ValueError(f"scan-line count does not match study in {folder}")
        if _truncation_physical_signature(manifest) != _truncation_physical_signature(manifests[0]):
            raise ValueError(f"physical/scan/time/analytical configurations differ in {folder}")
    runs = [_load_paper_run(folder, mode)
            for folder, mode in zip(folders, modes)]
    conventional, localized, truncated = runs
    comparisons = [HybridAccuracyEvaluator(conventional, run)
                   for run in (localized, truncated)]
    # Coordinates are checked by the evaluator; also verify connectivity.
    with np.load(folders[0] / "mesh_data.npz", allow_pickle=False) as reference_mesh:
        for folder in folders[1:]:
            with np.load(folder / "mesh_data.npz", allow_pickle=False) as mesh:
                for key in ("element_connectivity", "surface_triangles"):
                    if not np.array_equal(reference_mesh[key], mesh[key]):
                        raise ValueError(f"mesh {key} differs in {folder}")
    cutoff = float(truncated.metadata["truncation_age"])
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("invalid saved truncation_age")
    _, scan_end = _scan_time_bounds(folders[0], manifests[0])
    scan_path = manifests[0]["experiment"]["scan_path"]
    ends = [float(line.get("end_time", float(line["start_time"]) +
                  np.linalg.norm(np.asarray(line["end_point"]) -
                                 np.asarray(line["start_point"])) / float(line["velocity"])))
            for line in scan_path]
    first_cutoff = min(ends) + cutoff
    first_discard_step = next((i for i, item in enumerate(truncated.evaluation_history)
                               if item.discarded_line_count > 0), None)
    if first_discard_step is None:
        raise ValueError("no line was truncated during this run; reduce the window or extend cooling")
    first_observed = float(truncated.times[first_discard_step])
    for item in truncated.evaluation_history:
        if (item.transferred_line_count != 0
            or item.discarded_line_count != item.started_line_count - item.explicitly_retained_line_count
            or item.analytical_volume_point_evaluations != 0
            or item.analytical_node_point_evaluations != item.explicitly_retained_line_count * truncated.node_coordinates.shape[0]):
            raise ValueError("saved truncation history does not match hard deletion with full nodal evaluation")
    if any(item.residual_assembly != 0.0 for item in truncated.timing_history):
        raise ValueError("truncation run unexpectedly assembled localization residuals")
    before = conventional.saved_steps < first_discard_step
    if not np.any(before):
        raise ValueError("no saved fields before the first truncation")
    if not np.allclose(truncated.total_temperature[before], conventional.total_temperature[before],
                       rtol=1e-7, atol=1e-6):
        raise ValueError("truncation differs from conventional before any line deletion; check setup")

    output = configuration.output_directory / _safe_label(configuration.study_name)
    output.mkdir(parents=True, exist_ok=True)
    labels = ("Localized", "Temporal truncation")
    metrics = [{"method": label, **comparison.compute_metrics()}
               for label, comparison in zip(labels, comparisons)]
    _write_csv(output / "accuracy_summary.csv", metrics)
    histories = [comparison.error_vs_time() for comparison in comparisons]
    rows = []
    for i, t in enumerate(conventional.saved_times):
        row = {"time_s": float(t)}
        for name, history in zip(("localized", "temporal_truncation"), histories):
            for field in ("relative_error_percent", "rmse", "mean_signed_difference"):
                row[f"{name}_{field}"] = float(history[field][i])
        rows.append(row)
    _write_csv(output / "accuracy_vs_time.csv", rows)
    event_rows = []
    for event, requested_time in (
        ("first_deletion", first_observed), ("time_0_5_s", 0.5),
        ("scan_completion", scan_end),
        ("last_removal_deadline", max(ends) + cutoff),
        ("simulation_end", float(conventional.saved_times[-1])),
    ):
        index = int(np.argmin(np.abs(conventional.saved_times - requested_time)))
        event_rows.append({"event": event,
                           "requested_time_s": requested_time, **rows[index]})
    _write_csv(output / "accuracy_at_key_times.csv", event_rows)
    _write_csv(output / "truncation_history.csv", [
        {"time_s": item.time, "started_lines": item.started_line_count,
         "retained_lines": item.explicitly_retained_line_count,
         "discarded_lines": item.discarded_line_count}
        for item in truncated.evaluation_history])
    parameters = localized.metadata.get("localization_parameters") or manifests[1]["experiment"]["localization"]
    _write_json(output / "study_summary.json", {
        "configuration": configuration, "truncation_age_s": cutoff,
        "localized_temporal_support_s": parameters["retention_age"] + parameters["transfer_duration"],
        "first_cutoff_time_s": first_cutoff,
        "first_observed_deletion_time_s": first_observed,
        "scan_end_time_s": scan_end, "metrics": metrics,
        "temperature": "total temperature in degrees Celsius",
        "initial_temperature_celsius": conventional.metadata["initial_temperature_celsius"],
        "aggregation": "unweighted relative 2-norm over all saved node-time values",
        "pre_deletion_agreement_verified": True,
    })
    table = ["| Method vs conventional | Relative error (%) | RMSE (degrees C) |",
             "|---|---:|---:|"]
    for row in metrics:
        table.append(f"| {row['method']} | {row['relative_error_percent']:.6g} | {row['rmse_temperature_error']:.6g} |")
    (output / "accuracy_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    plt = _get_pyplot()
    for field, ylabel, filename in (
        ("relative_error_percent", "Relative 2-norm error (%)", "relative_error_vs_time.png"),
        ("rmse", "RMSE (degrees C)", "rmse_vs_time.png"),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        for label, history, linestyle in zip(labels, histories, ("--", "-")):
            axis.plot(history["time"], history[field], label=label,
                      linestyle=linestyle)
        # Label events beside their lines, independently of the data scale.
        for event_time, event_label, color, linestyle, offset in (
            (first_cutoff, "First cutoff", "0.35", ":", -6),
            (scan_end, "Scan completion", "0.55", "--", 6),
        ):
            axis.axvline(event_time, color=color, linestyle=linestyle,
                         linewidth=1.2, zorder=1)
            axis.annotate(
                event_label,
                xy=(event_time, 0.50),
                xycoords=axis.get_xaxis_transform(),
                xytext=(offset, 0),
                textcoords="offset points",
                rotation=90,
                ha="right" if offset < 0 else "left",
                va="center",
                color=color,
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=1.5),
            )
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02),
                    ncol=2, frameon=False, borderaxespad=0.0,
                    columnspacing=1.5)
        _finish_figure(figure, output / filename)
    print("\n".join(table))
    print(f"Paper outputs saved to: {output}")
    return output


def main() -> None:
    if not ACTIVE_STUDIES:
        raise ValueError("Configure your saved input folders, then select at least one "
                         "study from STUDY_LIBRARY in ACTIVE_STUDIES.")
    if len(set(ACTIVE_STUDIES)) != len(ACTIVE_STUDIES):
        raise ValueError("ACTIVE_STUDIES contains duplicate names")

    for study_name in ACTIVE_STUDIES:
        if study_name not in STUDY_LIBRARY:
            available = ", ".join(sorted(STUDY_LIBRARY)) or "<none configured>"
            raise KeyError(
                f"unknown study {study_name!r}; configured studies: {available}"
            )
        configuration = STUDY_LIBRARY[study_name]
        _check_study_input_paths(configuration)
        print(f"\nRunning paper study: {study_name}")
        if isinstance(configuration, AbaqusConvergenceStudyConfiguration):
            run_abaqus_convergence_study(configuration)
            continue
        if isinstance(configuration, AbaqusHybridStudyConfiguration):
            run_abaqus_hybrid_study(configuration)
            continue
        if isinstance(configuration, TemporalTruncationStudyConfiguration):
            run_temporal_truncation_study(configuration)
            continue
        if isinstance(configuration, LineCountStudyConfiguration):
            run_line_count_study(configuration)
        elif isinstance(configuration, LocalizationExtentStudyConfiguration):
            run_localization_extent_study(configuration)
        elif isinstance(configuration, PostScanTemporalStudyConfiguration):
            run_post_scan_temporal_study(configuration)
        elif isinstance(configuration, DetailedCaseStudyConfiguration):
            run_detailed_case_study(configuration)
        else:
            raise TypeError(
                f"unsupported configuration type for study {study_name!r}: "
                f"{type(configuration).__name__}"
            )


if __name__ == "__main__":
    main()

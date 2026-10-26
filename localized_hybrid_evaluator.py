"""Post-process conventional and localized hybrid LPBF runs.

The evaluator is intentionally independent of the simulation algorithm.  It
receives two HybridRunResult objects, checks compatibility, computes accuracy
and speed metrics, saves machine-readable outputs, and creates a small set of
plots suitable for deciding whether the localized formulation is effective.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from localized_hybrid_simulation import (
    EvaluationSnapshot,
    HybridConfigurationError,
    HybridMode,
    HybridRunResult,
    StepTiming,
)


def validate_temperature_metadata(metadata):
    """Accept this package's physical-temperature outputs without conversion."""
    if metadata.get("temperature") != "total temperature":
        raise ValueError("Expected physical-temperature output from the new code. "
                         "Old result conversion is not part of this package.")
    initial = metadata.get("initial_temperature_celsius")
    if initial is None or not np.isfinite(initial):
        raise ValueError("Missing or invalid initial_temperature_celsius metadata")
    return float(initial)


def _validate_temperature_pair(left, right):
    if validate_temperature_metadata(left.metadata) != validate_temperature_metadata(right.metadata):
        raise ValueError("Runs have different initial temperatures")


@dataclass(frozen=True)
class HybridComparisonMetrics:
    relative_nodal_2_norm_temperature_error: float
    rmse_temperature_error: float
    max_abs_temperature_error: float
    total_wall_time_speedup: float
    analytical_time_speedup: float
    conventional_wall_time: float
    localized_wall_time: float
    conventional_analytical_time: float
    localized_analytical_time: float
    conventional_fea_time: float
    localized_fea_time: float
    localized_residual_assembly_time: float
    localized_boundary_assembly_time: float
    localized_localization_time: float


class HybridAccuracyEvaluator:
    """Accuracy only, comparing either candidate against conventional."""

    def __init__(self, conventional: HybridRunResult, candidate: HybridRunResult):
        _validate_temperature_pair(conventional, candidate)
        if conventional.mode is not HybridMode.CONVENTIONAL:
            raise HybridConfigurationError("reference must be conventional")
        if candidate.mode not in {HybridMode.LOCALIZED, HybridMode.TEMPORAL_TRUNCATION}:
            raise HybridConfigurationError("candidate must be localized or temporal_truncation")
        for name in ("times", "saved_steps", "saved_times", "node_coordinates"):
            if not np.array_equal(getattr(conventional, name), getattr(candidate, name)):
                raise HybridConfigurationError(f"runs have different {name}")
        shape = (conventional.saved_times.size, conventional.node_coordinates.shape[0])
        for run in (conventional, candidate):
            if run.total_temperature.shape != shape:
                raise HybridConfigurationError("invalid saved-field shape")
            if not np.all(np.isfinite(run.total_temperature)):
                raise HybridConfigurationError("temperature histories contain NaN or infinity")
        self.reference = conventional
        self.candidate = candidate

    @staticmethod
    def _relative(numerator, denominator):
        # Nonzero error against a zero reference is infinite, not zero.
        return np.divide(numerator, denominator,
                         out=np.where(numerator == 0.0, 0.0, np.inf),
                         where=denominator > 0.0)

    def compute_metrics(self) -> dict[str, float]:
        difference = self.candidate.total_temperature - self.reference.total_temperature
        return {
            "relative_error_percent": float(100.0 * self._relative(
                np.linalg.norm(difference.ravel()),
                np.linalg.norm(self.reference.total_temperature.ravel()))),
            "rmse_temperature_error": float(np.sqrt(np.mean(difference ** 2))),
        }

    def error_vs_time(self) -> dict[str, np.ndarray]:
        difference = self.candidate.total_temperature - self.reference.total_temperature
        return {
            "time": self.reference.saved_times.copy(),
            "relative_error_percent": 100.0 * self._relative(
                np.linalg.norm(difference, axis=1),
                np.linalg.norm(self.reference.total_temperature, axis=1)),
            "rmse": np.sqrt(np.mean(difference ** 2, axis=1)),
            "mean_signed_difference": np.mean(difference, axis=1),
        }


class LocalizedHybridEvaluator:
    """Compare a conventional hybrid run against a localized hybrid run."""

    def __init__(
        self,
        conventional: HybridRunResult,
        localized: HybridRunResult,
    ) -> None:
        self.conventional = conventional
        self.localized = localized
        _validate_temperature_pair(conventional, localized)
        self._validate_compatibility()

    @staticmethod
    def _timing_total(run: HybridRunResult, name: str) -> float:
        return float(sum(float(getattr(item, name)) for item in run.timing_history))

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0.0 else float("inf")

    def compute_metrics(self) -> HybridComparisonMetrics:
        difference = (
            self.localized.total_temperature - self.conventional.total_temperature
        )
        if not np.all(np.isfinite(difference)):
            raise HybridConfigurationError(
                "temperature histories contain NaN or infinity"
            )
        reference_norm = float(
            np.linalg.norm(self.conventional.total_temperature.ravel())
        )
        difference_norm = float(np.linalg.norm(difference.ravel()))
        relative_nodal_2_norm = (
            difference_norm / reference_norm
            if reference_norm > 0.0
            else (0.0 if difference_norm == 0.0 else float("inf"))
        )

        conv_analytical = self._timing_total(self.conventional, "analytical")
        loc_analytical = self._timing_total(self.localized, "analytical")
        conv_fea = self._timing_total(self.conventional, "fea")
        loc_fea = self._timing_total(self.localized, "fea")

        return HybridComparisonMetrics(
            relative_nodal_2_norm_temperature_error=relative_nodal_2_norm,
            rmse_temperature_error=float(np.sqrt(np.mean(difference**2))),
            max_abs_temperature_error=float(np.max(np.abs(difference))),
            total_wall_time_speedup=self._safe_ratio(
                self.conventional.wall_time, self.localized.wall_time
            ),
            analytical_time_speedup=self._safe_ratio(
                conv_analytical, loc_analytical
            ),
            conventional_wall_time=float(self.conventional.wall_time),
            localized_wall_time=float(self.localized.wall_time),
            conventional_analytical_time=conv_analytical,
            localized_analytical_time=loc_analytical,
            conventional_fea_time=conv_fea,
            localized_fea_time=loc_fea,
            localized_residual_assembly_time=self._timing_total(
                self.localized, "residual_assembly"
            ),
            localized_boundary_assembly_time=self._timing_total(
                self.localized, "boundary_assembly"
            ),
            localized_localization_time=self._timing_total(
                self.localized, "localization"
            ),
        )

    def error_vs_time(self) -> dict[str, np.ndarray]:
        difference = (
            self.localized.total_temperature - self.conventional.total_temperature
        )
        if not np.all(np.isfinite(difference)):
            raise HybridConfigurationError(
                "temperature histories contain NaN or infinity"
            )
        numerator = np.linalg.norm(difference, axis=1)
        denominator = np.linalg.norm(self.conventional.total_temperature, axis=1)
        relative_nodal_2_norm = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        return {
            "time": self.conventional.saved_times.copy(),
            "relative_nodal_2_norm": relative_nodal_2_norm,
            "rmse": np.sqrt(np.mean(difference**2, axis=1)),
            "max_abs": np.max(np.abs(difference), axis=1),
        }

    @classmethod
    def timing_summary(cls, run: HybridRunResult) -> dict[str, float]:
        return {
            "wall_time": float(run.wall_time),
            "localization": cls._timing_total(run, "localization"),
            "analytical": cls._timing_total(run, "analytical"),
            "residual_assembly": cls._timing_total(run, "residual_assembly"),
            "boundary_assembly": cls._timing_total(run, "boundary_assembly"),
            "fea": cls._timing_total(run, "fea"),
        }

    def save_results(self, output_directory: str | Path) -> Path:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)

        payload = {
            "metrics": asdict(self.compute_metrics()),
            "conventional_timing": self.timing_summary(self.conventional),
            "localized_timing": self.timing_summary(self.localized),
            "conventional_metadata": self._jsonable(self.conventional.metadata),
            "localized_metadata": self._jsonable(self.localized.metadata),
        }
        (output / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        np.savez_compressed(
            output / "temperature_histories.npz",
            times=self.conventional.times,
            saved_steps=self.conventional.saved_steps,
            saved_times=self.conventional.saved_times,
            node_coordinates=self.conventional.node_coordinates,
            conventional_total=self.conventional.total_temperature,
            localized_total=self.localized.total_temperature,
            conventional_analytical=self.conventional.analytical_temperature,
            localized_analytical=self.localized.analytical_temperature,
            conventional_correction=self.conventional.correction_temperature,
            localized_correction=self.localized.correction_temperature,
        )
        self.save_run(self.conventional, output)
        self.save_run(self.localized, output)
        return output

    @classmethod
    def save_run(
        cls, run: HybridRunResult, output_directory: str | Path
    ) -> Path:
        """Save one mode using mode-specific filenames."""
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        mode_name = run.mode.value
        payload = {
            "mode": mode_name,
            "timing": cls.timing_summary(run),
            "metadata": cls._jsonable(run.metadata),
        }
        (output / f"{mode_name}_run_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        cls._write_timing_csv(output / f"{mode_name}_timing.csv", run)
        cls._write_evaluation_csv(output / f"{mode_name}_evaluation.csv", run)
        np.savez_compressed(
            output / f"{mode_name}_run_data.npz",
            times=run.times,
            saved_steps=run.saved_steps,
            saved_times=run.saved_times,
            node_coordinates=run.node_coordinates,
            total_temperature=run.total_temperature,
            analytical_temperature=run.analytical_temperature,
            correction_temperature=run.correction_temperature,
        )
        return output

    @classmethod
    def save_single_run(
        cls, run: HybridRunResult, output_directory: str | Path
    ) -> Path:
        """Save one independently executed run."""
        return cls.save_run(run, output_directory)

    @classmethod
    def load_run(
        cls, input_directory: str | Path, mode: HybridMode | str
    ) -> HybridRunResult:
        """Load one requested mode, ignoring other modes in the folder."""
        input_path = Path(input_directory)
        selected_mode = mode if isinstance(mode, HybridMode) else HybridMode(mode)
        mode_name = selected_mode.value
        summary_path = input_path / f"{mode_name}_run_summary.json"
        data_path = input_path / f"{mode_name}_run_data.npz"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"folder does not contain saved {mode_name} results: {summary_path}"
            )
        if not data_path.is_file():
            raise FileNotFoundError(
                f"missing comparison data: {data_path}; rerun this mode using "
                "the updated evaluator"
            )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        try:
            saved_mode = HybridMode(summary["mode"])
            wall_time = float(summary["timing"]["wall_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HybridConfigurationError(
                f"invalid single-run summary: {summary_path}"
            ) from exc
        if saved_mode is not selected_mode:
            raise HybridConfigurationError(
                f"{summary_path} contains {saved_mode.value}, not {mode_name}"
            )

        timing_path = input_path / f"{mode_name}_timing.csv"
        evaluation_path = input_path / f"{mode_name}_evaluation.csv"
        timing_history = cls._read_timing_csv(timing_path)
        evaluation_history = cls._read_evaluation_csv(evaluation_path)

        with np.load(data_path, allow_pickle=False) as data:
            required = {
                "times",
                "saved_steps",
                "saved_times",
                "node_coordinates",
                "total_temperature",
                "analytical_temperature",
                "correction_temperature",
            }
            missing = required.difference(data.files)
            if missing:
                raise HybridConfigurationError(
                    f"{data_path} is missing arrays: {sorted(missing)}"
                )
            result = HybridRunResult(
                mode=selected_mode,
                times=data["times"].copy(),
                saved_steps=data["saved_steps"].copy(),
                saved_times=data["saved_times"].copy(),
                node_coordinates=data["node_coordinates"].copy(),
                total_temperature=data["total_temperature"].copy(),
                analytical_temperature=data["analytical_temperature"].copy(),
                correction_temperature=data["correction_temperature"].copy(),
                timing_history=timing_history,
                evaluation_history=evaluation_history,
                wall_time=wall_time,
                metadata=summary.get("metadata", {}),
            )

        if len(result.timing_history) != result.times.size:
            raise HybridConfigurationError(
                f"timing history length does not match the time grid in {input_path}"
            )
        if len(result.evaluation_history) != result.times.size:
            raise HybridConfigurationError(
                f"evaluation history length does not match the time grid in {input_path}"
            )
        return result

    @classmethod
    def load_single_run(cls, input_directory: str | Path) -> HybridRunResult:
        """Load a folder containing exactly one saved run mode."""
        input_path = Path(input_directory)
        available = [
            mode
            for mode in HybridMode
            if (input_path / f"{mode.value}_run_summary.json").is_file()
        ]
        if len(available) != 1:
            raise HybridConfigurationError(
                f"expected exactly one saved mode in {input_path}; found "
                f"{[mode.value for mode in available]}"
            )
        return cls.load_run(input_path, available[0])

    def create_plots(self, output_directory: str | Path) -> list[Path]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []

        errors = self.error_vs_time()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(errors["time"], errors["relative_nodal_2_norm"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative nodal $2$-norm temperature error")
        ax.grid(alpha=0.25)
        created.append(
            self._save_figure(
                fig, output / "relative_nodal_2_norm_error_vs_time.png"
            )
        )

        components = (
            "localization",
            "analytical",
            "residual_assembly",
            "boundary_assembly",
            "fea",
        )
        conv = [self._timing_total(self.conventional, name) for name in components]
        loc = [self._timing_total(self.localized, name) for name in components]
        x = np.arange(len(components))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(x - 0.2, conv, width=0.4, label="Conventional")
        ax.bar(x + 0.2, loc, width=0.4, label="Localized")
        ax.set_xticks(x, [name.replace("_", " ") for name in components])
        ax.set_ylabel("Accumulated time (s)")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        created.append(self._save_figure(fig, output / "timing_breakdown.png"))

        times = np.asarray([item.time for item in self.localized.evaluation_history])
        retained = np.asarray(
            [item.explicitly_retained_line_count for item in self.localized.evaluation_history]
        )
        started = np.asarray(
            [item.started_line_count for item in self.localized.evaluation_history]
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(times, started, label="Started lines")
        ax.plot(times, retained, label="Explicitly retained analytical lines")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Number of lines")
        ax.legend()
        ax.grid(alpha=0.25)
        created.append(self._save_figure(fig, output / "retained_line_count.png"))

        return created

    def print_summary(self) -> None:
        m = self.compute_metrics()
        print("\n=== Localized hybrid comparison ===")
        print(
            "Relative nodal 2-norm error : "
            f"{m.relative_nodal_2_norm_temperature_error:.6e}"
        )
        print(f"RMSE temperature error        : {m.rmse_temperature_error:.6e} °C")
        print(f"Maximum absolute error        : {m.max_abs_temperature_error:.6e} °C")
        print(f"Total wall-time speedup       : {m.total_wall_time_speedup:.3f}x")
        print(f"Analytical-time speedup       : {m.analytical_time_speedup:.3f}x")
        print(f"Conventional analytical time  : {m.conventional_analytical_time:.6f} s")
        print(f"Localized analytical time     : {m.localized_analytical_time:.6f} s")
        print(f"Conventional FEA time         : {m.conventional_fea_time:.6f} s")
        print(f"Localized FEA time            : {m.localized_fea_time:.6f} s")
        print(f"Localized localization time   : {m.localized_localization_time:.6f} s")
        print(f"Localized residual assembly   : {m.localized_residual_assembly_time:.6f} s")
        print(f"Localized boundary assembly   : {m.localized_boundary_assembly_time:.6f} s")
        print("===================================\n")

    def _validate_compatibility(self) -> None:
        if self.conventional.mode is not HybridMode.CONVENTIONAL:
            raise HybridConfigurationError(
                "first run must use HybridMode.CONVENTIONAL"
            )
        if self.localized.mode is not HybridMode.LOCALIZED:
            raise HybridConfigurationError(
                "second run must use HybridMode.LOCALIZED"
            )
        pairs = (
            ("times", self.conventional.times, self.localized.times),
            ("saved_steps", self.conventional.saved_steps, self.localized.saved_steps),
            ("saved_times", self.conventional.saved_times, self.localized.saved_times),
            (
                "node_coordinates",
                self.conventional.node_coordinates,
                self.localized.node_coordinates,
            ),
        )
        for name, left, right in pairs:
            if not np.array_equal(left, right):
                raise HybridConfigurationError(f"runs have different {name}")
        if self.conventional.total_temperature.shape != self.localized.total_temperature.shape:
            raise HybridConfigurationError("runs have different saved-field shapes")

    @staticmethod
    def _write_timing_csv(path: Path, run: HybridRunResult) -> None:
        rows = []
        for step, timing in enumerate(run.timing_history):
            row = asdict(timing)
            row.update(step=step, time=float(run.times[step]))
            rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_evaluation_csv(path: Path, run: HybridRunResult) -> None:
        rows = [asdict(item) for item in run.evaluation_history]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_timing_csv(path: Path) -> list[StepTiming]:
        if not path.is_file():
            raise FileNotFoundError(f"missing timing history: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        try:
            return [
                StepTiming(
                    localization=float(row["localization"]),
                    analytical=float(row["analytical"]),
                    residual_assembly=float(row["residual_assembly"]),
                    boundary_assembly=float(row["boundary_assembly"]),
                    fea=float(row["fea"]),
                )
                for row in rows
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise HybridConfigurationError(f"invalid timing history: {path}") from exc

    @staticmethod
    def _read_evaluation_csv(path: Path) -> list[EvaluationSnapshot]:
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation history: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        try:
            return [
                EvaluationSnapshot(
                    time=float(row["time"]),
                    started_line_count=int(row["started_line_count"]),
                    explicitly_retained_line_count=int(
                        row["explicitly_retained_line_count"]
                    ),
                    transferred_line_count=int(row["transferred_line_count"]),
                    discarded_line_count=int(row.get("discarded_line_count", 0)),
                    analytical_node_point_evaluations=int(
                        row["analytical_node_point_evaluations"]
                    ),
                    analytical_boundary_point_evaluations=int(
                        row["analytical_boundary_point_evaluations"]
                    ),
                    analytical_volume_point_evaluations=int(
                        row["analytical_volume_point_evaluations"]
                    ),
                )
                for row in rows
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise HybridConfigurationError(
                f"invalid evaluation history: {path}"
            ) from exc

    @staticmethod
    def _save_figure(figure: Any, path: Path) -> Path:
        figure.tight_layout()
        figure.savefig(path, dpi=220, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(figure)
        return path

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): LocalizedHybridEvaluator._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [LocalizedHybridEvaluator._jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)


__all__ = ["HybridComparisonMetrics", "HybridAccuracyEvaluator", "LocalizedHybridEvaluator"]

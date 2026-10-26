# Localized Hybrid Thermal Modeling for Laser Powder Bed Fusion

This repository provides Python code for scan-resolved thermal simulation of laser powder bed fusion (LPBF). It accompanies the paper **Localized Hybrid Analytical–Finite-Element Thermal Modeling for Laser Powder Bed Fusion**.

Hybrid analytical–finite-element methods combine an analytical description of laser heating with a finite-element correction for the finite part. In a conventional hybrid formulation, the analytical evaluation becomes increasingly expensive as scan-line contributions accumulate. This work introduces space–time localization to restrict where and how long each contribution is evaluated analytically. Residual terms and boundary conditions derived from the heat equation allow the finite-element correction to retain the thermal effect of the removed analytical contributions.

The code supports the **localized hybrid**, **conventional hybrid**, and **uncompensated temporal-truncation** formulations, with Fin and Overhang geometries and tools for analyzing saved results. The paper assesses the method against an independent finite-element benchmark and multiline conventional-hybrid simulations. Its reported 500-line case achieves approximately **25.5× lower analytical evaluation time** and **3.92× lower total simulation time**, with a **0.139% relative nodal 2-norm difference** from the conventional solution. These are the paper’s case-specific results; runtime and accuracy depend on the chosen setup.

**Paper:** *Localized Hybrid Analytical–Finite-Element Thermal Modeling for Laser Powder Bed Fusion*  
**Preprint:** arXiv link forthcoming.
<!-- Replace the line above with: **Preprint:** [Read on arXiv](ACTUAL_ARXIV_URL) -->

**Authors:** Mohammad Amin Mohammadi Golafshani — [mmohammadi4@wisc.edu](mailto:mmohammadi4@wisc.edu)  
Krishnan Sureah — [ksuresh@wisc.edu](mailto:ksuresh@wisc.edu)  
University of Wisconsin–Madison

## Getting started

The workflow is: **install dependencies → edit `experiment_config.py` → run `run_localized_hybrid_experiment.py`**. Use `paper_studies.py` afterward to analyze saved runs.

The documented installation uses **Python 3.12 on Windows**. A GPU is not required. Full runs may require several gigabytes of memory and disk space. Abaqus is needed only for optional reference-data export.

### 1. Obtain this repository and PyTO

Download or clone this repository. Obtain [PyTO](https://github.com/UW-ERSL/PyTO) separately; it supplies meshing and finite-element dependencies. From the directory where you want to keep PyTO, run:

```shell
git clone https://github.com/UW-ERSL/PyTO.git
git -C PyTO checkout 1d4ce75e8092429ee4ec31fd09c0ef738773a7ea
```

A ZIP download of that revision also works. Keep the complete PyTO `src` directory unchanged and outside this repository. The simulation does not download PyTO automatically.

### 2. Create the environment and install dependencies

Open PowerShell in this repository’s root folder:

```powershell
py -3.12 -m venv localized_hybrid_venv
.\localized_hybrid_venv\Scripts\python.exe -m pip install --upgrade pip
.\localized_hybrid_venv\Scripts\python.exe -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
.\localized_hybrid_venv\Scripts\python.exe -m pip install -r requirements.txt
.\localized_hybrid_venv\Scripts\python.exe -m pip install "mkl==2026.1.0" "intel-openmp==2026.1.1"
```

These commands use the project environment explicitly; activation is optional. The listed PyTorch build supports CPU execution: keep `device="cpu"` for CPU runs. `requirements.txt` includes PyPardiso and the other dependencies. Keep its TetGen and PyVista version pins for compatibility with the specified PyTO revision.

The full Fin run was checked on Windows. Linux/macOS require an appropriate PyTorch/native-solver installation; their environment executable is `localized_hybrid_venv/bin/python`.

**VS Code:** choose **Python: Select Interpreter** from the command palette, select `localized_hybrid_venv\Scripts\python.exe`, and open a new terminal. You can then use **Run Python File in Terminal**.

## Configure a simulation

Edit **`experiment_config.py`**. Ordinary problem setup does not require changes to the numerical modules.

### Paths and geometry

```python
PYTO_SOURCE_DIRECTORY = Path(r"C:\path\to\PyTO\src")
GEOMETRY_DIRECTORY = SCRIPT_DIRECTORY / "Models"
RESULTS_DIRECTORY = SCRIPT_DIRECTORY / "results_localized_hybrid"
```

Replace the PyTO path with your actual location. `SCRIPT_DIRECTORY` is the directory containing the configuration file; the geometry and results paths above are relative to it. Results directories are created when saving. If you use the `PYTO_SRC` environment variable, keep it consistent with the configured path.

| Geometry | File | Experiment entry |
|---|---|---|
| Fin | `Models/Fin.STL` | `EXPERIMENT_LIBRARY["Fin"]` |
| Overhang | `Models/overhang.STL` | `EXPERIMENT_LIBRARY["Overhang"]` |

STL coordinates must be in **metres**. Preserve filename capitalization. For a new geometry, also adapt the scan path, mesh settings, and boundary selection.

### Problem inputs

Edit the selected entry in `EXPERIMENT_LIBRARY`:

| Setting | Inputs |
|---|---|
| `mesh` | Geometry and element sizes in metres |
| `material` | Density (kg/m³), conductivity (W/(m·K)), specific heat (J/(kg·K)) |
| `laser` | Power (W), absorptivity, Gaussian beam radius (m) |
| `boundary` | Initial and bottom temperatures (°C), bottom-plane selection |
| `scan_path` | Start times (s), coordinates (m), velocities (m/s) |
| `time` | Time step, post-scan cooling time, output saving interval |
| `analytical` | Device and numerical evaluation settings |
| `localization` | Core radius, transition width (m), retention age and transfer duration (s) |

The supplied setup uses constant material properties, uniform initial temperature, and a fixed bottom temperature. Other surfaces are adiabatic apart from laser heating. Scan paths lie on a horizontal top surface.

The Fin scan generator takes a prefix of a fixed 500-line master path: reducing `n_lines` keeps the hatch spacing unchanged. For its localization preset, `LOCALIZATION_SCALE` multiplies all four baseline parameters: **1 mm core radius, 1 mm transition width, 50 ms retention age, and 50 ms transfer duration**.

### Select the run

```python
ACTIVE_RUN = RunConfiguration(
    experiment=EXPERIMENT_LIBRARY["Fin"],
    mode="localized",
    case_label=f"Fin_500_lines_scale_{LOCALIZATION_SCALE:g}",
    solver_name="PARDISO",
    reference_folder=None,
    show_mesh_plots=False,
    create_result_plots=True,
)
```

| Mode | Simulation |
|---|---|
| `"localized"` | Localized hybrid |
| `"conventional"` | Conventional hybrid |
| `"both"` | Both formulations and their comparison |
| `"temporal_truncation"` | Uncompensated temporal truncation; set `truncation_age` in seconds |

`case_label` names the output folder; it does not set the line count or physics. Keep `reference_folder=None` for an independent run. Available solver selections are `"PARDISO"`, `"SPSOLVE"`, and `"SPLU"`.

## Run and view the geometry

Save the configuration, then run:

```powershell
.\localized_hybrid_venv\Scripts\python.exe run_localized_hybrid_experiment.py
```

To see the **STL geometry and tetrahedral mesh before the simulation**, set `show_mesh_plots=True` in `ACTIVE_RUN`. Close each plot window to continue. Leave it `False` for unattended runs.

`create_result_plots=True` produces comparison figures for `mode="both"`. Single-mode runs save numerical results; use the study tools for subsequent analysis.

Wait for **`Results saved to:`** before treating the run as complete. Each run gets a new timestamped folder. Full 500-line simulations can take tens of minutes and produce several gigabytes of data. Stopping before completion can lose the temperature history; the runner does not save intermediate checkpoints.

## Saved results and temperatures

| Output | Contents |
|---|---|
| `experiment_configuration.json`, `run_manifest.json` | Inputs and run/environment information |
| `mesh_data.npz` | Mesh coordinates and connectivity |
| `localized_run_data.npz` | Saved times and temperature arrays |
| `localized_run_summary.json` | Timing totals and metadata |
| `localized_timing.csv`, `localized_evaluation.csv` | Per-step timing and analytical-history statistics |

Other modes use their corresponding names in place of `localized`.

Define the initial and bottom temperatures directly in Celsius, for example `initial_temperature_c=32.0` and `bottom_temperature_c=32.0`. **Saved total temperatures are already physical Celsius values; do not add an offset.**

- `total_temperature`: physical temperature in °C.
- `analytical_temperature`: source-induced temperature increment.
- `correction_temperature`: FE correction including the initial baseline.

The fields satisfy `total_temperature = analytical_temperature + correction_temperature`. Arrays have shape `(saved times, nodes)`. Reading a field from a compressed NPZ can load the entire field into memory. Increase `time.save_every` to save less frequently without changing the simulation time step.

Keep each run’s configuration, manifest, mesh, and temperature files together for later comparisons, and record the repository revision used.

## Analyze saved runs

`paper_studies.py` generates plots and metrics from completed runs; it does not run simulations.

1. Set its `RESULTS_DIRECTORY` to the parent folder containing your runs.
2. In the desired study, replace `REPLACE_WITH_*` paths with your actual saved folder names, including timestamps. Absolute paths also work. Set `output_directory` for the analysis outputs.
3. Select the study in `ACTIVE_STUDIES`, which is empty by default:

```python
ACTIVE_STUDIES = ("fin_500_detailed_case",)
```

4. Run:

```powershell
.\localized_hybrid_venv\Scripts\python.exe paper_studies.py
```

Configure only selected studies; unused placeholders can remain. A `"both"` run can supply both `conventional_folder` and `localized_folder`. Do not rename saved folders to match the placeholders. Review case labels, scales, line counts, and resolutions to match the data being analyzed.

| Study key | Analysis |
|---|---|
| `fin_500_localization_extent` | Compare localization scales for the 500-line Fin setup |
| `fin_500_detailed_case` | Detailed 500-line comparison |
| `fin_500_temporal_truncation` | Temporal truncation versus hybrid formulations |
| `abaqus_one_line_convergence` | Mesh and time-step convergence of the one-line Abaqus benchmark |
| `abaqus_one_line_hybrid_comparison` | Compare conventional and localized hybrid with the one-line Abaqus reference |

Optional additional analyses are `fin_line_count_scaling` (different scan-line counts) and `fin_100_post_scan_temporal` (post-scan cooling durations). Mesh and time-step convergence in the table refers to the one-line Abaqus benchmark. Compared runs must have compatible physical setups and numerical data. The current comparison tools normalize relative errors by physical Celsius reference temperatures; use the same normalization when comparing reported metrics.

### Abaqus reference data

Use Abaqus Python to export an ODB:

```shell
abaqus python abaqus_export_temperature.py --odb reference.odb --out reference_export
```

The output is a directory of arrays and metadata. Add `--instance INSTANCE_NAME` if needed. In `paper_studies.py`, set `ABAQUS_RUNS` and the selected case paths to your ODBs, single-ODB run folders, or completed export folders. Use metres and physical Celsius temperatures; no temperature conversion is applied.

Check matching geometry, material, laser, boundary conditions, and timing before setting `physical_setup_confirmed=True`. The hybrid/Abaqus preset reuses convergence-study sample points: run convergence first, or set `points_file=None` to generate points independently. Abaqus reference models and results are not included.

## Troubleshooting

| Problem | Check |
|---|---|
| Missing Python package | Use the project interpreter and install the requirements into that environment. |
| Invalid PyTO source | Point to the complete PyTO `src` directory. |
| Meshing reports “too many values to unpack” | Keep the TetGen 0.6.7 and PyVista 0.45.3 pins. |
| OpenMP duplicate-runtime error | Use the documented Windows dependencies and current runner in a fresh process. Do not suppress it with `KMP_DUPLICATE_LIB_OK`. |
| Run waits after meshing | Close the geometry/mesh plot windows. |
| Study reports missing inputs | Replace the indicated placeholders and select only configured studies. |

## Citation and contact

If you use this code in research, please cite *Localized Hybrid Analytical–Finite-Element Thermal Modeling for Laser Powder Bed Fusion*. The arXiv link and complete citation will be added when available.

For questions about the code, contact **Mohammad Amin Mohammadi Golafshani** at [mmohammadi4@wisc.edu](mailto:mmohammadi4@wisc.edu).

Refer to the repository’s `LICENSE` file, when available, for reuse terms. PyTO and other dependencies retain their own licenses.

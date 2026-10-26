"""Postprocessing for Abaqus convergence and hybrid comparisons.

Dependencies: numpy, scipy, matplotlib. ODB export separately uses Abaqus Python.
Abaqus temperatures are interpolated with their FE shape functions. Hybrid
temperatures are evaluated as direct analytical temperature plus an FE
interpolation of the saved correction field. Metrics use physical Celsius temperature;
both inputs must contain physical Celsius temperatures. No offsets are applied.
The reference is numerical, not an exact solution.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import subprocess
import hashlib
import numpy as np
from localized_hybrid_evaluator import validate_temperature_metadata


@dataclass(frozen=True)
class AbaqusCase:
    label: str
    source: Path  # ODB, a folder with exactly one ODB, or completed export folder
    scan_dt: float
    cooling_dt: float
    mesh_sizes_m: tuple[float, float]
    instance: str | None = None


@dataclass(frozen=True)
class AbaqusConvergenceStudyConfiguration:
    study_name: str
    mesh_cases: tuple[AbaqusCase, ...]  # coarse -> fine: A1, A2, A3, A6
    time_cases: tuple[AbaqusCase, ...]  # coarse -> fine: A5, A4, A3
    output_directory: Path
    grid_spacing_m: float = 0.0005
    points_file: Path | None = None  # optional Nx3 .npy, metres
    abaqus_command: tuple[str, ...] = ('abaqus',)
    # Must be explicitly confirmed after checking material, DFLUX, BCs, units,
    # geometry and scan/cooling durations. ODB cannot prove these match.
    physical_setup_confirmed: bool = False


@dataclass(frozen=True)
class AbaqusHybridStudyConfiguration:
    study_name: str
    abaqus_case: AbaqusCase
    conventional_folder: Path
    localized_folder: Path
    output_directory: Path
    grid_spacing_m: float = 0.0005
    points_file: Path | None = None
    abaqus_command: tuple[str, ...] = ('abaqus',)
    physical_setup_confirmed: bool = False


def _json(path, value):
    def convert(x):
        if isinstance(x, Path):
            return str(x)
        if hasattr(x, '__dataclass_fields__'):
            return asdict(x)
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def _csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SolidMesh:
    """Point location using actual connectivity; never bridge fin voids.

    Tet4 barycentric and Hex8 isoparametric shape functions. Hex8 inversion
    rejects singular maps. Unsupported higher-order elements fail explicitly.
    """
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float)

    def __init__(self, nodes, elements):
        from scipy.spatial import cKDTree
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements)
        if (self.nodes.ndim != 2 or self.nodes.shape[1] != 3
                or not np.all(np.isfinite(self.nodes))):
            raise ValueError('Invalid coordinates')
        if (self.elements.ndim != 2 or self.elements.shape[1] not in (4, 8)
                or not np.issubdtype(self.elements.dtype, np.integer)
                or not self.elements.size or self.elements.min() < 0
                or self.elements.max() >= len(self.nodes)):
            raise ValueError('Need valid Tet4 or Hex8 connectivity')
        self.xyz = self.nodes[self.elements]
        self.lower, self.upper = self.xyz.min(axis=1), self.xyz.max(axis=1)
        self.centers = self.xyz.mean(axis=1)
        radii = np.max(np.linalg.norm(self.xyz-self.centers[:, None], axis=2), axis=1)
        self.radius = float(radii.max())
        self.tree = cKDTree(self.centers)
        self.tol = max(float(np.ptp(self.nodes, axis=0).max())*1e-10, 1e-14)
        self.width = self.elements.shape[1]
        if self.width == 4:
            mat = np.transpose(self.xyz[:, 1:]-self.xyz[:, :1], (0,2,1))
            self.inverse = np.linalg.inv(mat)

    @classmethod
    def shape(cls, natural):
        factors = 1 + cls.signs * natural
        n = np.prod(factors, axis=1)/8
        dn = np.empty((8,3))
        for j in range(3):
            dn[:,j] = cls.signs[:,j]*np.prod(factors[:,[k for k in range(3) if k != j]],axis=1)/8
        return n, dn

    def locate(self, points, allow_outside=False):
        points = np.asarray(points, dtype=float)
        ids = np.full(len(points), -1, dtype=int)
        weights = np.zeros((len(points), self.width))
        # Chunk candidate queries so a large mesh does not create a huge list.
        for start in range(0, len(points), 256):
            candidates = self.tree.query_ball_point(points[start:start+256], self.radius+self.tol)
            for offset, candidates_i in enumerate(candidates):
                i = start+offset
                c = np.asarray(candidates_i, dtype=int)
                c = c[np.all(points[i] >= self.lower[c]-self.tol, axis=1)
                      & np.all(points[i] <= self.upper[c]+self.tol, axis=1)]
                if self.width == 4 and len(c):
                    b = np.einsum('kij,kj->ki', self.inverse[c], points[i]-self.xyz[c,0])
                    w = np.column_stack((1-b.sum(axis=1), b))
                    good = np.flatnonzero(np.all(w >= -1e-9,axis=1) & np.all(w <= 1+1e-9,axis=1))
                    if len(good):
                        j = good[np.argmax(w[good].min(axis=1))]
                        ids[i], weights[i] = c[j], w[j]
                elif self.width == 8:
                    for element in c:
                        natural = np.zeros(3)
                        xyz = self.xyz[element]
                        for _ in range(25):
                            n, dn = self.shape(natural)
                            residual = n @ xyz-points[i]
                            if np.linalg.norm(residual) <= self.tol:
                                break
                            try:
                                natural -= np.linalg.solve(xyz.T @ dn, residual)
                            except np.linalg.LinAlgError:
                                break
                            if not np.all(np.isfinite(natural)) or np.max(np.abs(natural)) > 10:
                                break
                        n, _ = self.shape(natural)
                        if (np.all(np.abs(natural) <= 1+1e-9)
                                and np.linalg.norm(n @ xyz-points[i]) <= self.tol):
                            ids[i], weights[i] = element, n
                            break
        if not allow_outside and np.any(ids < 0):
            raise ValueError('%d comparison points outside this mesh. Check geometry/units; '
                             'points will not be silently removed.' % np.sum(ids < 0))
        return ids, weights

    def interpolate(self, field, ids, weights):
        return np.einsum('pi,pi->p', field[self.elements[ids]], weights)


def _export_folder(case, command):
    source = Path(case.source).expanduser().resolve()
    if source.is_dir() and (source/'metadata.json').exists():
        return source
    if source.is_dir():
        found = list(source.glob('*.odb'))
        if len(found) != 1:
            raise ValueError('Run folder must contain exactly one ODB: '+str(source))
        source = found[0]
    if not source.is_file() or source.suffix.lower() != '.odb':
        raise FileNotFoundError('Supply an ODB, its run folder, or an export folder: '+str(source))
    tag = hashlib.sha256((case.instance or '').encode()).hexdigest()[:8]
    output = source.with_name(source.stem+'_temperature_export_'+tag)
    if (output/'metadata.json').exists():
        meta = json.loads((output/'metadata.json').read_text())
        stat = source.stat()
        if meta['source_size'] != stat.st_size or meta['source_mtime'] != stat.st_mtime:
            raise ValueError('ODB changed since export. Move the old export folder and rerun.')
        return output
    if output.exists():
        raise ValueError('Incomplete export exists. Move it aside before retrying: '+str(output))
    script = Path(__file__).with_name('abaqus_export_temperature.py')
    args = list(command)+['python', str(script), '--odb', str(source), '--out', str(output)]
    if case.instance:
        args += ['--instance', case.instance]
    print('Exporting ODB read-only using Abaqus Python:', source, flush=True)
    subprocess.run(args, check=True)
    return output


def _load_abaqus(case, command):
    folder = _export_folder(case, command)
    meta = json.loads((folder/'metadata.json').read_text())
    if meta.get('format_version') != 1:
        raise ValueError('Unsupported export format')
    if not set(meta.get('element_types', [])).issubset({'DC3D4','DC3D8','DC3D8R'}):
        raise ValueError('Unsupported Abaqus element types')
    mesh = SolidMesh(np.load(folder/'nodes.npy'), np.load(folder/'elements.npy'))
    field = np.load(folder/'temperature.npy', mmap_mode='r')
    times = np.load(folder/'times.npy')
    _validate_field(times, field, mesh)
    return dict(label=case.label, case=case, mesh=mesh, times=times, field=field,
                metadata=meta, source=str(folder))


def _validate_field(times, field, mesh):
    if (len(times) < 3 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0)
            or field.shape != (len(times),len(mesh.nodes))):
        raise ValueError('Invalid saved times or temperature field shape')


def _load_hybrid(folder, mode):
    folder = Path(folder)
    manifest = json.loads((folder/'run_manifest.json').read_text())
    summary = json.loads((folder/(mode+'_run_summary.json')).read_text())
    if summary.get('mode') != mode:
        raise ValueError('Expected '+mode+' output')
    validate_temperature_metadata(summary['metadata'])
    with np.load(folder/'mesh_data.npz', allow_pickle=False) as data:
        mesh = SolidMesh(data['node_coordinates'], data['element_connectivity'])
    if mesh.width != 4:
        raise ValueError('Hybrid reader supports Tet4 only')
    with np.load(folder/(mode+'_run_data.npz'), allow_pickle=False) as data:
        if not np.array_equal(mesh.nodes, data['node_coordinates']):
            raise ValueError('Hybrid field node order differs from mesh_data.npz')
        required = {'saved_times', 'correction_temperature'}
        missing = required.difference(data.files)
        if missing:
            raise ValueError('Hybrid run data is missing arrays: '+str(sorted(missing)))
        times = data['saved_times'].copy()
        # Only the FE correction is interpolated.  The analytical contribution
        # is reevaluated directly at the comparison points from the run manifest.
        correction_field = data['correction_temperature'].copy()
    _validate_field(times,correction_field,mesh)
    return dict(label=mode, mesh=mesh, times=times, correction_field=correction_field,
                metadata=summary, manifest=manifest, source=str(folder))


def _points(mesh, config):
    if config.points_file is not None:
        points = np.load(config.points_file, allow_pickle=False)
    else:
        spacing = config.grid_spacing_m
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError('Grid spacing must be positive')
        lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
        axes = [np.arange(lo[j]+spacing/2, hi[j], spacing) for j in range(3)]
        if any(len(a) == 0 for a in axes) or np.prod([len(a) for a in axes]) > 2_000_000:
            raise ValueError('Grid empty or too large; change grid_spacing_m')
        candidates = np.stack(np.meshgrid(*axes, indexing='ij'),axis=-1).reshape(-1,3)
        ids, _ = mesh.locate(candidates, allow_outside=True)
        points = candidates[ids >= 0]
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
            or not np.all(np.isfinite(points))):
        raise ValueError('Need at least two finite Nx3 sample points')
    mesh.locate(points)
    return points


def _common_times(runs, tol=5e-8):
    starts = [r['times'][0] for r in runs]
    ends = [r['times'][-1] for r in runs]
    ranges = ', '.join('%s: %.12g to %.12g s (%d frames)'
                       % (r['label'], r['times'][0], r['times'][-1], len(r['times']))
                       for r in runs)
    print('Saved-time ranges:', ranges, flush=True)
    # A solver may omit the initial frame from its saved output (for example,
    # when the hybrid run saves every nth increment), even though the physical
    # simulations cover the same duration.  Compare only the overlapping saved
    # interval.  Final saved times may differ by at most one terminal output
    # interval; a larger mismatch indicates different simulated durations.
    terminal_intervals = [r['times'][-1]-r['times'][-2] for r in runs]
    endpoint_tol = max(tol, max(terminal_intervals)+tol)
    if np.ptp(ends) > endpoint_tol:
        raise ValueError('Saved final times differ by more than one terminal output interval: '
                         +ranges)
    overlap_start = max(starts)
    overlap_end = min(ends)
    if overlap_end-overlap_start <= tol:
        raise ValueError('Runs have no usable overlapping saved-time interval')
    common = runs[0]['times'][(runs[0]['times'] >= overlap_start-tol)
                              & (runs[0]['times'] <= overlap_end+tol)].copy()
    def match(t, requested):
        pos = np.searchsorted(t,requested)
        a,b = np.clip(pos,0,len(t)-1),np.clip(pos-1,0,len(t)-1)
        return np.where(abs(t[a]-requested) <= abs(t[b]-requested),a,b)
    for r in runs[1:]:
        idx = match(r['times'],common)
        common = common[abs(r['times'][idx]-common) <= tol]
    if len(common) < 3 or abs(common[-1]-overlap_end) > tol:
        raise ValueError('Need >=3 common saved times, including the final time; export more frames')
    return common, [match(r['times'],common) for r in runs]


def metrics(a,b,times):
    """Unweighted sampled 2-norm plus time-weighted supplemental metrics.

    b is always the explicitly named reference. Undefined relative differences
    at zero reference norm are None (CSV blank), not artificially zero.
    """
    error = a-b
    numerator,denominator = float(np.sum(error**2)),float(np.sum(b**2))
    relative = 100*np.sqrt(numerator/denominator) if denominator > 0 else None
    result = dict(relative_2norm_percent=relative, rmse_K=float(np.sqrt(np.mean(error**2))),
                  max_abs_K=float(np.max(np.abs(error))))
    if len(times) > 1:
        w = np.zeros(len(times))
        w[:-1] += np.diff(times)/2
        w[1:] += np.diff(times)/2
        n,d = float(w @ np.mean(error**2,axis=1)),float(w @ np.mean(b*b,axis=1))
        result.update(time_weighted_relative_L2_percent=100*np.sqrt(n/d) if d > 0 else None,
                      time_weighted_rmse_K=float(np.sqrt(n/w.sum())))
    return result


def _hybrid_analytical_context(run, points):
    """Rebuild the exact analytical evaluator used by the saved hybrid run.

    The run manifest is the source of truth for material, laser, scan-path,
    analytical-kernel and localization settings.  The analytical temperature is
    then evaluated directly at the fixed comparison points instead of being
    reconstructed from hybrid FE nodal values.
    """
    from vectorized_eagar_tsai import EagarTsaiParameters, ScanSegment, VectorizedEagarTsai
    from localized_hybrid_simulation import LocalizationParameters, QuinticLineLocalization

    try:
        exp = run['manifest']['experiment']
        material = exp['material']
        laser = exp['laser']
        settings = exp['analytical']
        scan_path = tuple(exp['scan_path'])
    except (KeyError, TypeError) as exc:
        raise ValueError('Hybrid run manifest lacks analytical reconstruction data') from exc

    parameters = EagarTsaiParameters.from_absorbed_power(
        thermal_conductivity=float(material['thermal_conductivity']),
        density=float(material['mass_density']),
        specific_heat=float(material['specific_heat']),
        sigma=float(laser['gaussian_sigma']),
        laser_power=float(laser['laser_power']),
        absorptivity=float(laser['absorptivity']),
    )
    engine = VectorizedEagarTsai(
        parameters,
        device=settings.get('device'),
        dtype=settings.get('dtype', 'float32'),
        gauss_order=int(settings.get('gauss_order', 64)),
        point_chunk_size=int(settings.get('point_chunk_size', 4096)),
        line_chunk_size=int(settings.get('line_chunk_size', 8)),
    )

    segments = []
    end_times = []
    endpoints = []
    for index, line in enumerate(scan_path):
        try:
            start = np.asarray(line['start_point'], dtype=float)
            end = np.asarray(line['end_point'], dtype=float)
            start_time = float(line['start_time'])
            velocity = float(line['velocity'])
            if 'end_time' in line:
                end_time = float(line['end_time'])
            else:
                end_time = start_time + float(np.linalg.norm(end-start))/velocity
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid scan-path segment %d in hybrid manifest' % index) from exc
        segments.append(ScanSegment.from_endpoints(
            start=start, end=end, start_time=start_time, end_time=end_time,
            speed=velocity, surface_normal=(0.0, 0.0, 1.0)))
        end_times.append(end_time)
        endpoints.append((start, end))
    prepared = engine.prepare_lines(tuple(segments))

    localization = None
    spatial_windows = None
    if run['label'] == 'localized':
        try:
            loc = exp['localization']
            localization = QuinticLineLocalization(LocalizationParameters(
                core_radius=float(loc['core_radius']),
                transition_width=float(loc['transition_width']),
                retention_age=float(loc['retention_age']),
                transfer_duration=float(loc['transfer_duration']),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Localized hybrid manifest lacks valid localization data') from exc
        spatial_windows = np.stack([
            localization.spatial(points, start, end)[0]
            for start, end in endpoints
        ]) if segments else np.empty((0, len(points)), dtype=float)

    return dict(engine=engine, prepared=prepared, localization=localization,
                spatial_windows=spatial_windows, end_times=np.asarray(end_times, dtype=float))


def _sample_hybrid(run, points, indices):
    """Evaluate T_hybrid = T_A(point) + I_h[T_correction](point)."""
    ids, weights = run['mesh'].locate(points)
    correction = np.stack([
        run['mesh'].interpolate(run['correction_field'][i], ids, weights)
        for i in indices
    ])

    context = _hybrid_analytical_context(run, points)
    analytical = np.empty_like(correction)
    for j, i in enumerate(indices):
        time_now = float(run['times'][i])
        if run['label'] == 'conventional':
            analytical[j] = context['engine'].evaluate(
                context['prepared'], points, time_now, need_gradient=False
            ).temperature
        else:
            by_line, _ = context['engine'].evaluate_by_line(
                context['prepared'], points, time_now, need_gradient=False
            )
            temporal = np.asarray([
                context['localization'].temporal(time_now, end_time)[0]
                for end_time in context['end_times']
            ], dtype=float)
            analytical[j] = np.sum(
                by_line * context['spatial_windows'] * temporal[:, None], axis=0
            )
    return analytical + correction


def _sample(runs,points,times,indices):
    histories = []
    for r, ix in zip(runs,indices):
        print('Sampling',r['label'],'at',len(points),'fixed points',flush=True)
        if 'correction_field' in r:
            history = _sample_hybrid(r, points, ix)
        else:
            ids, weights = r['mesh'].locate(points)
            history = np.stack([
                r['mesh'].interpolate(r['field'][i],ids,weights)
                for i in ix
            ])
        if not np.all(np.isfinite(history)):
            raise ValueError('Nonfinite temperatures in '+r['label'])
        histories.append(history)
    return histories


def _report(output, name, runs, points, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times, indices = _common_times(runs)
    histories = _sample(runs,points,times,indices)
    _json(output/'temperature_basis.json', {
        'reported_temperature': 'total temperature in degrees Celsius',
        'relative_error_basis': 'physical Celsius temperature',
    })
    rows, histories_rows = [], []
    for a,b,kind in pairs:
        row = dict(comparison=kind, candidate=runs[a]['label'],reference=runs[b]['label'],
                   point_count=len(points),common_time_count=len(times))
        row.update(metrics(histories[a],histories[b],times))
        rows.append(row)
        for i,t in enumerate(times):
            histories_rows.append(dict(comparison=kind,candidate=runs[a]['label'],reference=runs[b]['label'],
                       time_s=float(t),**metrics(histories[a][i:i+1],histories[b][i:i+1],times[i:i+1])))
    _csv(output/(name+'_metrics.csv'),rows)
    _csv(output/(name+'_error_history.csv'),histories_rows)
    np.savez_compressed(output/(name+'_sampled_histories.npz'),times=times,points=points,
                        labels=np.array([r['label'] for r in runs]),temperature=np.stack(histories),
                        temperature_basis=np.array("physical Celsius temperature"))
    for key,title in [('relative_2norm_percent','Relative sampled 2-norm difference (%)'),
                      ('rmse_K','Sampled RMSE (K)'),('max_abs_K','Sampled maximum absolute difference (K)')]:
        fig,ax = plt.subplots(figsize=(8,4.5))
        for a,b,kind in pairs:
            selected = [r for r in histories_rows if r['comparison']==kind
                        and r['candidate']==runs[a]['label'] and r['reference']==runs[b]['label']]
            ax.plot(times,[r[key] if r[key] is not None else np.nan for r in selected],
                    label=runs[a]['label']+' vs '+runs[b]['label']+' ('+kind+')')
        ax.set(xlabel='Total time (s)',ylabel=title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output/(name+'_'+key+'.png'),dpi=180)
        plt.close(fig)
    _json(output/(name+'_runs.json'),[dict(label=r['label'],source=r['source'],metadata=r['metadata'],
          node_count=len(r['mesh'].nodes),element_count=len(r['mesh'].elements),
          saved_time_count=len(r['times']),**({'declared_case':r['case']} if 'case' in r else {})) for r in runs])


def _prepare(config):
    if not config.physical_setup_confirmed:
        raise ValueError('Check identical geometry, metre units, material, source power/width/path, '
                         'BCs, initial temperature and durations. Then set physical_setup_confirmed=True. '
                         'This is a manual declaration, not automatic verification.')
    output = Path(config.output_directory)
    # Preserve prior results rather than overwriting them.
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty/new study output directory: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    _json(output/'configuration.json',config)
    return output


def _save_points(output,points,hybrid_direct_analytical=False):
    np.save(output/'sample_points.npy',points)
    np.savetxt(output/'sample_points.csv',points,delimiter=',',header='x_m,y_m,z_m',comments='')
    if hybrid_direct_analytical:
        interpolation = ('Abaqus: element-shape interpolation of saved nodal NT11. '
                         'Hybrid: direct analytical evaluation at each comparison point plus '
                         'Tet4 interpolation of the saved FE correction field only.')
        limitation = ('Sampled maximum is not a full-domain or continuous-time maximum. '
                      'Abaqus remains a numerical comparator, and its spatial discretization '
                      'error is assessed separately by the convergence study.')
    else:
        interpolation = 'Element shape functions on saved nodal temperatures, including hybrid TOTAL nodal field.'
        limitation = ('Hybrid analytical field is not reevaluated at sample points. Coarse nodal interpolation '
                      'adds representation error. Sampled maximum is not a full-domain or continuous-time maximum.')
    _json(output/'metric_notes.json',dict(
        primary='Equal-weight Euclidean 2-norm, RMSE, sampled max over common points and saved times.',
        supplementary='Trapezoidal time-weighted L2 and RMSE avoid output-cadence weighting bias.',
        zero_reference='Relative difference undefined for zero reference norm; CSV cell blank.',
        interpolation=interpolation,
        limitation=limitation,
        sampling='Fixed grid in actual solid, no track targeting. Repeat with smaller spacing to assess sampling sensitivity.',
        verification='Reference is a numerical comparator, not exact truth. No automatic convergence pass/fail.'))


def run_abaqus_convergence_study(config):
    groups = {}
    for kind,cases in [('mesh',config.mesh_cases),('time',config.time_cases)]:
        if len(cases) < 3 or len({c.label for c in cases}) != len(cases):
            raise ValueError('Provide >=3 uniquely labelled coarse-to-fine '+kind+' cases')
        if len({str(Path(c.source).resolve()) for c in cases}) != len(cases):
            raise ValueError('Duplicate source within '+kind+' sequence')
        dt = np.array([[c.scan_dt,c.cooling_dt] for c in cases])
        sizes = np.array([c.mesh_sizes_m for c in cases])
        if not np.all(np.isfinite(dt)) or np.any(dt <= 0) or np.any(sizes <= 0):
            raise ValueError('Invalid declared resolutions')
        if kind == 'mesh':
            if not np.all(dt == dt[0]) or not np.all(np.diff(sizes,axis=0) < 0):
                raise ValueError('Mesh sequence needs fixed time increments and decreasing mesh sizes')
        else:
            if not np.all(sizes == sizes[0]) or not np.all(np.diff(dt,axis=0) < 0):
                raise ValueError('Time sequence needs fixed mesh sizes and decreasing time increments')
        groups[kind] = cases
    output = _prepare(config)
    cache = {}
    for cases in groups.values():
        for c in cases:
            if c not in cache:
                cache[c] = _load_abaqus(c,config.abaqus_command)
    source = cache[config.mesh_cases[-1]]
    points = _points(source['mesh'],config)
    _save_points(output,points)
    for kind,cases in groups.items():
        runs = [cache[c] for c in cases]
        if kind == 'time':
            for r in runs[1:]:
                if (not np.array_equal(r['mesh'].nodes,runs[0]['mesh'].nodes)
                        or not np.array_equal(r['mesh'].elements,runs[0]['mesh'].elements)):
                    raise ValueError('Time study requires identical actual meshes, including ordering')
        elif np.any(np.diff([len(r['mesh'].elements) for r in runs]) <= 0):
            raise ValueError('Mesh study must have increasing actual element counts')
        for r in runs:
            if r['metadata']['steps'] != source['metadata']['steps']:
                raise ValueError('Abaqus step names/start times/durations differ')
        pairs = [(i,len(runs)-1,'vs_finest') for i in range(len(runs)-1)]
        pairs += [(i,i+1,'successive') for i in range(len(runs)-1)]
        _report(output,kind,runs,points,pairs)
    print('Abaqus convergence results:',output)
    return output


def run_abaqus_hybrid_study(config):
    output = _prepare(config)
    abaqus = _load_abaqus(config.abaqus_case,config.abaqus_command)
    conv = _load_hybrid(config.conventional_folder,'conventional')
    loc = _load_hybrid(config.localized_folder,'localized')
    def signature(r):
        exp = dict(r['manifest']['experiment'])
        for key in ('name','paths','localization'):
            exp.pop(key,None)
        return exp
    if signature(conv) != signature(loc):
        raise ValueError('Conventional/localized physical and numerical configurations differ')
    if (not np.array_equal(conv['mesh'].nodes,loc['mesh'].nodes)
            or not np.array_equal(conv['mesh'].elements,loc['mesh'].elements)):
        raise ValueError('Conventional/localized must use the same actual hybrid mesh')
    points = _points(abaqus['mesh'],config)
    _save_points(output,points,hybrid_direct_analytical=True)
    _report(output,'abaqus_hybrid',[abaqus,conv,loc],points,
            [(1,0,'hybrid_vs_abaqus'),(2,0,'hybrid_vs_abaqus'),(2,1,'localized_vs_conventional')])
    print('Abaqus/hybrid comparison results:',output)
    return output

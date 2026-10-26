"""Run with Abaqus Python, not the hybrid Python environment.

abaqus python abaqus_export_temperature.py --odb JOB.odb --out EXPORT_DIR
Reads the ODB without modifying it. Only linear DC3D4/DC3D8/DC3D8R supported.
NT11 values are exported unchanged. For the comparison utilities, define the
Abaqus model using metres and physical Celsius temperatures.
"""
import argparse
import json
import os
import numpy as np


def export(odb_path, output, instance_name=None):
    from odbAccess import openOdb
    from abaqusConstants import NODAL
    if os.path.exists(output):
        raise ValueError('Output already exists. Choose a new export directory: ' + output)
    odb = openOdb(path=os.path.abspath(odb_path), readOnly=True)
    try:
        names = [name for name in odb.rootAssembly.instances.keys()
                 if len(odb.rootAssembly.instances[name].elements)]
        if instance_name is None:
            if len(names) != 1:
                raise ValueError('Specify --instance; available instances: ' + str(names))
            instance_name = names[0]
        inst = odb.rootAssembly.instances[instance_name]
        nodes = list(inst.nodes)
        labels = np.array([n.label for n in nodes], dtype=np.int64)
        coords = np.array([n.coordinates for n in nodes], dtype=float)
        lookup = dict((int(label), i) for i, label in enumerate(labels))
        types = set(e.type for e in inst.elements)
        if not types or not types.issubset(set(['DC3D4', 'DC3D8', 'DC3D8R'])):
            raise ValueError('Unsupported elements: ' + str(types))
        widths = set(len(e.connectivity) for e in inst.elements)
        if len(widths) != 1:
            raise ValueError('Mixed Tet4/Hex8 meshes are not supported by this exporter.')
        connectivity = np.array([[lookup[int(n)] for n in e.connectivity]
                                 for e in inst.elements], dtype=np.int64)
        records, steps = [], []
        for name in odb.steps.keys():
            step = odb.steps[name]
            steps.append(dict(name=name, start=float(step.totalTime), duration=float(step.timePeriod)))
            for frame in step.frames:
                if 'NT11' not in frame.fieldOutputs:
                    continue
                time = float(step.totalTime + frame.frameValue)
                if records and abs(time-records[-1][0]) <= 1e-12:
                    # Keep the end of the preceding step, not a duplicate t=0 frame.
                    continue
                if records and time <= records[-1][0]:
                    raise ValueError('ODB temperature frames are not chronological.')
                records.append((time, name, frame))
        if len(records) < 3:
            raise ValueError('Need NT11 field output at at least three times.')
        expected_end = max(s['start'] + s['duration'] for s in steps)
        if abs(records[-1][0]-expected_end) > 1e-10:
            raise ValueError('Final NT11 frame is not at the planned end time. '
                             'Check job completion and final-frame output.')
        os.makedirs(output)
        np.save(os.path.join(output, 'nodes.npy'), coords)
        np.save(os.path.join(output, 'elements.npy'), connectivity)
        np.save(os.path.join(output, 'node_labels.npy'), labels)
        np.save(os.path.join(output, 'times.npy'), np.array([r[0] for r in records]))
        field = np.lib.format.open_memmap(os.path.join(output, 'temperature.npy'),
                     mode='w+', dtype='float64', shape=(len(records), len(nodes)))
        for i, (_, name, frame) in enumerate(records):
            field[i, :] = np.nan
            values = frame.fieldOutputs['NT11'].getSubset(region=inst, position=NODAL).values
            for value in values:
                try:
                    data = value.data
                except Exception:
                    data = value.dataDouble
                field[i, lookup[int(value.nodeLabel)]] = float(np.asarray(data).reshape(-1)[0])
            if not np.all(np.isfinite(field[i])):
                raise ValueError('Missing or nonfinite NT11 in frame ' + str(i))
            print('Exported NT11 frame %d/%d' % (i+1, len(records)))
        field.flush()
        stat = os.stat(odb_path)
        metadata = dict(format_version=1, source_odb=os.path.abspath(odb_path),
                        source_size=stat.st_size, source_mtime=stat.st_mtime,
                        instance=instance_name, element_types=sorted(types), steps=steps,
                        node_count=len(nodes), element_count=len(connectivity),
                        note='Coordinates/temperatures retain model units. No unit conversion.')
        # Written last: its presence identifies a completed export.
        with open(os.path.join(output, 'metadata.json'), 'w') as handle:
            json.dump(metadata, handle, indent=2)
    finally:
        odb.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odb', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--instance')
    args = parser.parse_args()
    export(args.odb, args.out, args.instance)

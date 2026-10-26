"""STL ingestion and tetrahedral-mesh construction for the LPBF solver.

This module deliberately contains no scan-path, material, boundary-condition,
or FEA-solver logic.  It is the geometry/mesh preprocessing ingredient only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np


from pyto_runtime import configure_pyto


Number = Union[int, float, np.number]
ElementSizeInput = Union[Number, Sequence[Number]]
SizeFunction = Callable[[np.ndarray], float]


class STLMeshError(RuntimeError):
    """Base exception for STL reading and tetrahedral meshing failures."""


class STLInputError(STLMeshError, ValueError):
    """Raised when an STL path or meshing option is invalid."""


class MeshSafetyLimitError(STLMeshError):
    """Raised when the estimated mesh exceeds the configured safety limit."""


class MeshValidationError(STLMeshError):
    """Raised when a generated tetrahedral mesh is structurally invalid."""


@dataclass(frozen=True)
class STLMeshResult:
    """Complete output of :meth:`STLMeshBuilder.build`.

    ``stl_geometry`` is the original :class:`stl_reader.STLGeom` instance and
    ``mesh`` is the original :class:`tet_mesher.TetMesher` instance.  Keeping
    both objects makes every existing helper method available downstream.
    """

    stl_path: Path
    model_name: str
    stl_geometry: Any
    mesh: Any
    surface_area: float
    stl_volume: float
    stl_signed_volume: float
    center_of_mass: np.ndarray
    inertia_tensor: np.ndarray
    bounds: Tuple[float, float, float, float, float, float]
    stl_triangle_count: int
    requested_num_elements: int
    mesh_volume: float
    volume_error_percent: float
    element_sizes: np.ndarray

    @property
    def nodes(self) -> np.ndarray:
        """Node coordinates with shape ``(num_nodes, 3)``."""
        return self.mesh.node_xyz

    @property
    def elements(self) -> np.ndarray:
        """Zero-based Tet4 connectivity with shape ``(num_elements, 4)``."""
        return self.mesh.elems

    @property
    def surface_triangles(self) -> np.ndarray:
        """Zero-based exterior-triangle connectivity with shape ``(n, 3)``."""
        return self.mesh.surface_triangles

    @property
    def num_nodes(self) -> int:
        return int(self.mesh.num_nodes)

    @property
    def num_elements(self) -> int:
        return int(self.mesh.num_elems)

    @property
    def num_surface_triangles(self) -> int:
        return int(len(self.mesh.surface_triangles))

    @property
    def mean_element_size(self) -> float:
        return float(np.mean(self.element_sizes))

    def plot_stl_geometry(
        self,
        *,
        show_edges: bool = False,
        show_axes: bool = True,
        show_bounding_box: bool = True,
    ) -> Any:
        """Open PyTO's original interactive STL-geometry window.

        This delegates directly to ``STLGeom.plotGeometry``.  The PyVista
        window normally blocks execution until the user closes it.
        """
        plot_method = getattr(self.stl_geometry, "plotGeometry", None)
        if not callable(plot_method):
            raise STLMeshError(
                "The configured STL geometry object does not provide "
                "plotGeometry()."
            )
        return plot_method(
            show_edges=show_edges,
            show_axes=show_axes,
            show_bounding_box=show_bounding_box,
        )

    def plot_tetrahedral_mesh(self, title: str = "Tet Mesh") -> Any:
        """Open PyTO's original interactive tetrahedral-mesh window.

        This delegates directly to ``TetMesher.plot``. The PyVista window
        normally blocks execution until the user closes it.
        """
        plot_method = getattr(self.mesh, "plot", None)
        if not callable(plot_method):
            raise STLMeshError(
                "The configured tetrahedral mesh object does not provide plot()."
            )
        return plot_method(title=title)

    def summary(self) -> str:
        """Return a readable deterministic summary without printing implicitly."""
        return (
            "--------- STL / Tet Mesh Information ---------\n"
            f"STL file: {self.stl_path}\n"
            f"Requested elements: {self.requested_num_elements}\n"
            f"STL surface triangles: {self.stl_triangle_count}\n"
            f"STL volume: {self.stl_volume:.6e}\n"
            f"Tet mesh volume: {self.mesh_volume:.6e}\n"
            f"Volume error: {self.volume_error_percent:.4f}%\n"
            f"Mesh nodes: {self.num_nodes}\n"
            f"Mesh elements: {self.num_elements}\n"
            f"Mesh surface triangles: {self.num_surface_triangles}\n"
            f"Mean max-edge element size: {self.mean_element_size:.6e}\n"
            "----------------------------------------------"
        )


class STLMeshBuilder:
    """Read one STL file and generate the Tet4 mesh used by the LPBF solver.

    Parameters
    ----------
    stl_file:
        Path to an existing ``.stl`` file. Relative paths are resolved against
        ``base_directory`` when it is supplied, otherwise against the current
        working directory.
    element_size:
        Either one positive characteristic size or ``(minimum, maximum)``.
        As in the old simulation, their average estimates the desired number
        of tetrahedra as ``STL volume / average_size**3``. The pair is also
        retained for later solver configuration; actual spatial grading is
        controlled by ``element_size_function``.
    element_size_function:
        Optional callable ``h = f([x, y, z])`` passed unchanged to
        :class:`TetMesher`.
    merge_facets:
        Passed unchanged to ``TetMesher.createTetMeshFromSTLFile``.
    max_desired_elements:
        Safety cap for the estimated desired count. Set to ``None`` to disable.
    allow_large_mesh:
        Explicit opt-in to exceed ``max_desired_elements``.
    """

    def __init__(
        self,
        stl_file: Union[str, Path],
        element_size: ElementSizeInput,
        *,
        element_size_function: Optional[SizeFunction] = None,
        merge_facets: bool = True,
        base_directory: Optional[Union[str, Path]] = None,
        max_desired_elements: Optional[int] = 10_000_000,
        allow_large_mesh: bool = False,
        stl_geometry_factory: Optional[Callable[[str], Any]] = None,
        tet_mesher_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.stl_path = self._resolve_stl_path(stl_file, base_directory)
        self.element_size_min, self.element_size_max = self._parse_element_size(
            element_size
        )

        if element_size_function is not None and not callable(element_size_function):
            raise STLInputError("element_size_function must be callable or None.")
        if not isinstance(merge_facets, (bool, np.bool_)):
            raise STLInputError("merge_facets must be True or False.")
        if max_desired_elements is not None:
            if isinstance(max_desired_elements, (bool, np.bool_)):
                raise STLInputError("max_desired_elements must be a positive integer or None.")
            try:
                max_as_float = float(max_desired_elements)
                max_as_int = int(max_desired_elements)
            except (TypeError, ValueError, OverflowError) as exc:
                raise STLInputError(
                    "max_desired_elements must be a positive integer or None."
                ) from exc
            if (
                not np.isfinite(max_as_float)
                or max_as_float != max_as_int
                or max_as_int <= 0
            ):
                raise STLInputError(
                    "max_desired_elements must be a positive integer or None."
                )
            max_desired_elements = max_as_int

        self.element_size_function = element_size_function
        self.merge_facets = bool(merge_facets)
        self.max_desired_elements = max_desired_elements
        self.allow_large_mesh = bool(allow_large_mesh)
        self._stl_geometry_factory = stl_geometry_factory
        self._tet_mesher_factory = tet_mesher_factory
        self.result: Optional[STLMeshResult] = None

    def build(self) -> STLMeshResult:
        """Read, mesh, validate, and return the complete preprocessing result."""
        stl_factory, mesher_factory = self._load_factories()

        try:
            geometry = stl_factory(str(self.stl_path))
            area, signed_volume, center, inertia = geometry.compute_mass_properties()
        except STLMeshError:
            raise
        except Exception as exc:
            raise STLMeshError(f"Failed to read STL geometry: {self.stl_path}") from exc

        area = self._positive_finite(area, "STL surface area")
        signed_volume = self._finite_scalar(signed_volume, "STL signed volume")
        if signed_volume == 0.0:
            raise STLInputError(
                "The STL has zero computed volume. It may be open, degenerate, or invalid."
            )
        volume = abs(signed_volume)

        center = np.asarray(center, dtype=float)
        inertia = np.asarray(inertia, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise STLInputError("STL center of mass must be a finite three-vector.")
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise STLInputError("STL inertia tensor must be a finite 3-by-3 array.")
        # A consistently inward-oriented surface reverses all divergence
        # integrals. Convert the tensor to the physical orientation while also
        # retaining the original signed volume in the result.
        inertia = inertia * np.sign(signed_volume)

        desired = self.estimate_desired_elements(volume)
        if (
            self.max_desired_elements is not None
            and desired > self.max_desired_elements
            and not self.allow_large_mesh
        ):
            raise MeshSafetyLimitError(
                f"Estimated desired element count {desired:,} exceeds the safety "
                f"limit {self.max_desired_elements:,}. Increase element_size, "
                "raise the limit, or set allow_large_mesh=True explicitly."
            )

        try:
            mesh = mesher_factory()
            mesh.createTetMeshFromSTLFile(
                str(self.stl_path),
                mergeFacets=self.merge_facets,
                nElemsDesired=desired,
                elemSizeFunction=self.element_size_function,
            )
        except STLMeshError:
            raise
        except Exception as exc:
            raise STLMeshError(f"Tetrahedral meshing failed for: {self.stl_path}") from exc

        nodes, elements, surface_triangles = self._validate_mesh(mesh)
        element_sizes, tet_volumes = self._tet_metrics(nodes, elements)
        mesh_volume = float(np.sum(tet_volumes))
        if not np.isfinite(mesh_volume) or mesh_volume <= 0.0:
            raise MeshValidationError("Generated tetrahedral mesh has non-positive volume.")

        # Standardize attributes required by the later FEA/handoff code.
        mesh.node_xyz = nodes
        mesh.elems = elements
        mesh.surface_triangles = surface_triangles
        mesh.num_nodes = int(nodes.shape[0])
        mesh.num_elems = int(elements.shape[0])
        mesh.origin = np.min(nodes, axis=0)
        mesh.elem_sizes = element_sizes
        mesh.elem_size = float(np.mean(element_sizes))
        mesh.volume = mesh_volume

        bounds_raw = geometry.get_bounding_box()
        bounds_array = np.asarray(bounds_raw, dtype=float)
        if bounds_array.shape != (6,) or not np.all(np.isfinite(bounds_array)):
            raise STLInputError("STL bounding box must contain six finite values.")

        triangle_count = int(getattr(geometry, "stl_n_triangles", 0))
        if triangle_count <= 0:
            raise STLInputError("The STL contains no surface triangles.")

        result = STLMeshResult(
            stl_path=self.stl_path,
            model_name=self.stl_path.stem,
            stl_geometry=geometry,
            mesh=mesh,
            surface_area=area,
            stl_volume=volume,
            stl_signed_volume=signed_volume,
            center_of_mass=center.copy(),
            inertia_tensor=inertia.copy(),
            bounds=tuple(float(x) for x in bounds_array),
            stl_triangle_count=triangle_count,
            requested_num_elements=desired,
            mesh_volume=mesh_volume,
            volume_error_percent=abs(mesh_volume - volume) / volume * 100.0,
            element_sizes=element_sizes,
        )
        self.result = result
        return result

    def estimate_desired_elements(self, stl_volume: float) -> int:
        """Apply the element-count estimate used by the old LPBF simulation."""
        volume = self._positive_finite(stl_volume, "STL volume")
        average_size = 0.5 * (self.element_size_min + self.element_size_max)
        ratio = volume / average_size**3
        # Preserve floor semantics, while preventing a value that is only a few
        # floating-point ulps below an integer (for example 7.999999999999998)
        # from losing one requested element.
        roundoff = np.finfo(float).eps * max(1.0, abs(ratio)) * 8.0
        return max(1, int(np.floor(ratio + roundoff)))

    @staticmethod
    def compute_element_sizes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """Return each tetrahedron's characteristic size (maximum edge length)."""
        sizes, _ = STLMeshBuilder._tet_metrics(
            np.asarray(nodes, dtype=float), np.asarray(elements, dtype=np.int64)
        )
        return sizes

    @staticmethod
    def _tet_metrics(
        nodes: np.ndarray, elements: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        xyz = nodes[elements]
        edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        lengths = np.stack(
            [np.linalg.norm(xyz[:, i] - xyz[:, j], axis=1) for i, j in edge_pairs],
            axis=1,
        )
        sizes = np.max(lengths, axis=1)
        volumes = np.abs(
            np.einsum(
                "ij,ij->i",
                xyz[:, 0] - xyz[:, 3],
                np.cross(xyz[:, 1] - xyz[:, 3], xyz[:, 2] - xyz[:, 3]),
            )
        ) / 6.0

        # Use the mesh's own length scale. A fixed 1 m floor would incorrectly
        # reject valid micrometer-scale tetrahedra as degenerate.
        scale = max(float(np.max(sizes)), np.finfo(float).tiny)
        volume_tolerance = np.finfo(float).eps * scale**3 * 64.0
        if np.any(~np.isfinite(sizes)) or np.any(sizes <= 0.0):
            raise MeshValidationError("The mesh contains invalid tetrahedron edge lengths.")
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= volume_tolerance):
            bad = np.flatnonzero((~np.isfinite(volumes)) | (volumes <= volume_tolerance))
            raise MeshValidationError(
                f"The mesh contains {bad.size} degenerate tetrahedron(s); "
                f"first invalid element index: {int(bad[0])}."
            )
        return sizes, volumes

    @staticmethod
    def _validate_mesh(mesh: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for name in ("node_xyz", "elems", "surface_triangles"):
            if not hasattr(mesh, name):
                raise MeshValidationError(f"Generated mesh is missing '{name}'.")

        nodes = np.asarray(mesh.node_xyz, dtype=float)
        elements_raw = np.asarray(mesh.elems)
        surface_raw = np.asarray(mesh.surface_triangles)

        if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] < 4:
            raise MeshValidationError("Mesh nodes must have shape (N, 3), with N >= 4.")
        if not np.all(np.isfinite(nodes)):
            raise MeshValidationError("Mesh node coordinates must all be finite.")

        elements = STLMeshBuilder._connectivity_array(elements_raw, 4, "elements")
        surface = STLMeshBuilder._connectivity_array(
            surface_raw, 3, "surface_triangles"
        )
        if elements.shape[0] == 0:
            raise MeshValidationError("Generated mesh contains no tetrahedra.")
        if surface.shape[0] == 0:
            raise MeshValidationError("Generated mesh contains no exterior triangles.")

        max_node = nodes.shape[0] - 1
        if np.any(elements < 0) or np.any(elements > max_node):
            raise MeshValidationError("Tet connectivity contains an out-of-range node index.")
        if np.any(surface < 0) or np.any(surface > max_node):
            raise MeshValidationError(
                "Surface connectivity contains an out-of-range node index."
            )
        if np.any(np.diff(np.sort(elements, axis=1), axis=1) == 0):
            raise MeshValidationError("A tetrahedron repeats a node index.")
        if np.any(np.diff(np.sort(surface, axis=1), axis=1) == 0):
            raise MeshValidationError("A surface triangle repeats a node index.")
        xyz = nodes[elements]
        jac = np.stack((xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0],
                        xyz[:, 3]-xyz[:, 0]), axis=1)
        if np.any(np.linalg.det(jac) <= 0.0):
            raise MeshValidationError("PyTO requires positively oriented Tet4 elements. "
                                      "Check mesh node ordering before FE assembly.")
        return nodes, elements, surface

    @staticmethod
    def _connectivity_array(array: np.ndarray, width: int, name: str) -> np.ndarray:
        if array.ndim != 2 or array.shape[1] != width:
            raise MeshValidationError(f"Mesh {name} must have shape (N, {width}).")
        if not np.issubdtype(array.dtype, np.integer):
            try:
                as_float = np.asarray(array, dtype=float)
            except (TypeError, ValueError) as exc:
                raise MeshValidationError(f"Mesh {name} must contain integers.") from exc
            if not np.all(np.isfinite(as_float)) or not np.all(as_float == np.floor(as_float)):
                raise MeshValidationError(f"Mesh {name} must contain integers.")
        return np.asarray(array, dtype=np.int64)

    def _load_factories(self) -> Tuple[Callable[[str], Any], Callable[[], Any]]:
        stl_factory = self._stl_geometry_factory
        mesher_factory = self._tet_mesher_factory

        if stl_factory is None or mesher_factory is None:
            configure_pyto()

        try:
            if stl_factory is None:
                from stl_reader import STLGeom

                stl_factory = STLGeom
            if mesher_factory is None:
                from tet_mesher import TetMesher

                mesher_factory = TetMesher
        except ImportError as exc:
            raise STLMeshError(
                "Could not import STLGeom/TetMesher from the configured PyTO/src. "
                "Install numpy-stl, pyvista, "
                "and tetgen."
            ) from exc
        return stl_factory, mesher_factory

    @staticmethod
    def _resolve_stl_path(
        stl_file: Union[str, Path], base_directory: Optional[Union[str, Path]]
    ) -> Path:
        if not isinstance(stl_file, (str, Path)) or not str(stl_file).strip():
            raise STLInputError("stl_file must be a non-empty path.")
        path = Path(stl_file).expanduser()
        if not path.is_absolute() and base_directory is not None:
            path = Path(base_directory).expanduser() / path
        path = path.resolve()
        if path.suffix.lower() != ".stl":
            raise STLInputError(f"Expected an .stl file, received: {path.name}")
        if not path.exists():
            raise STLInputError(f"STL file does not exist: {path}")
        if not path.is_file():
            raise STLInputError(f"STL path is not a file: {path}")
        return path

    @staticmethod
    def _parse_element_size(value: ElementSizeInput) -> Tuple[float, float]:
        if isinstance(value, (str, bytes, bool, np.bool_)):
            raise STLInputError(
                "element_size must be a positive number or a (minimum, maximum) pair."
            )
        if np.isscalar(value):
            minimum = maximum = STLMeshBuilder._positive_finite(
                value, "element_size"
            )
        else:
            try:
                values = tuple(value)
            except TypeError as exc:
                raise STLInputError(
                    "element_size must be a positive number or a (minimum, maximum) pair."
                ) from exc
            if len(values) != 2:
                raise STLInputError("element_size pair must contain exactly two values.")
            minimum = STLMeshBuilder._positive_finite(values[0], "minimum element size")
            maximum = STLMeshBuilder._positive_finite(values[1], "maximum element size")
        if minimum > maximum:
            raise STLInputError("Minimum element size cannot exceed maximum element size.")
        return minimum, maximum

    @staticmethod
    def _finite_scalar(value: Number, name: str) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise STLInputError(f"{name} must be a finite number.")
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise STLInputError(f"{name} must be a finite number.") from exc
        if not np.isfinite(result):
            raise STLInputError(f"{name} must be a finite number.")
        return result

    @staticmethod
    def _positive_finite(value: Number, name: str) -> float:
        result = STLMeshBuilder._finite_scalar(value, name)
        if result <= 0.0:
            raise STLInputError(f"{name} must be positive.")
        return result

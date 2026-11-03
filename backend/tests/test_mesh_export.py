import io
import unittest

import trimesh

from backend.mesh_export import create_stepped_cornice


class ConceptMeshTests(unittest.TestCase):
    def test_mesh_is_watertight_and_matches_requested_bounds(self):
        mesh = create_stepped_cornice(90, 42, 36)

        self.assertTrue(mesh.is_watertight)
        self.assertEqual(mesh.extents.tolist(), [90, 42, 36])

    def test_mesh_exports_as_reloadable_stl(self):
        mesh = create_stepped_cornice(90, 42, 36)
        stl_bytes = mesh.export(file_type="stl")

        loaded = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")

        self.assertTrue(loaded.is_watertight)
        self.assertEqual(loaded.extents.tolist(), [90, 42, 36])

    def test_rejects_nonfinite_and_out_of_range_dimensions(self):
        for dimensions in ((float("nan"), 42, 36), (9, 42, 36), (90, 42, 2001)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                create_stepped_cornice(*dimensions)


if __name__ == "__main__":
    unittest.main()
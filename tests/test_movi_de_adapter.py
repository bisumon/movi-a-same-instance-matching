#!/usr/bin/env python3
"""Unit and synthetic-record tests for the MOVi-D/E data adapter."""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SOURCE))

import extract_movi_de_phase1 as extractor  # noqa: E402
import movi_de_dataset_adapter as adapter  # noqa: E402
import download_movi_de_pilot as pilot_download  # noqa: E402


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def synthetic_record(dataset: str = "movi_e") -> dict[str, object]:
    del dataset
    num_frames, num_instances, height, width = 2, 2, 8, 8
    segmentations = []
    videos = []
    depths = []
    for frame_index in range(num_frames):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., frame_index] = 100
        segmentation = np.zeros((height, width), dtype=np.uint8)
        segmentation[1:4, 1 + frame_index : 4 + frame_index] = 1
        segmentation[5:7, 5:7] = 2
        encoded_depth = np.full((height, width), round((5.0 - 1.0) / 9.0 * 65535), dtype=np.uint16)
        videos.append(png_bytes(rgb))
        segmentations.append(png_bytes(segmentation))
        depths.append(png_bytes(encoded_depth))
    positions = np.asarray(
        [
            [[-1.0, 1.0, -5.0], [-0.5, 1.0, -5.0]],
            [[1.0, -1.0, -5.0], [1.0, -1.0, -5.0]],
        ],
        dtype=np.float32,
    )
    bbox_offsets = np.asarray(
        [[x, y, z] for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)],
        dtype=np.float32,
    )
    bboxes = positions[:, :, None, :] + bbox_offsets[None, None, :, :]
    return {
        "metadata/video_name": np.asarray([b"17"]),
        "metadata/depth_range": np.asarray([1.0, 10.0], dtype=np.float32),
        "metadata/num_frames": np.asarray([num_frames]),
        "metadata/num_instances": np.asarray([num_instances]),
        "metadata/height": np.asarray([height]),
        "metadata/width": np.asarray([width]),
        "camera/focal_length": np.asarray([35.0], dtype=np.float32),
        "camera/sensor_width": np.asarray([32.0], dtype=np.float32),
        "camera/positions": np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32),
        "camera/quaternions": np.asarray([[1.0, 0.0, 0.0, 0.0]] * num_frames, dtype=np.float32),
        "instances/asset_id": np.asarray([b"asset-a", b"asset-b"]),
        "instances/category": np.asarray([b"Shoe", b"Toys"]),
        "instances/scale": np.asarray([1.0, 1.5], dtype=np.float32),
        "instances/is_dynamic": np.asarray([True, False]),
        "instances/positions": positions,
        "instances/velocities": np.zeros((num_instances, num_frames, 3), dtype=np.float32),
        "instances/bboxes_3d": bboxes,
        "instances/visibility": np.asarray([[9, 9], [4, 4]], dtype=np.int64),
        "video": videos,
        "segmentations": segmentations,
        "depth": depths,
        "background": np.asarray([b"studio_hdri"]),
    }


class DatasetAdapterTests(unittest.TestCase):
    def test_dataset_alias_and_dataset_specific_ids(self) -> None:
        self.assertEqual(adapter.get_dataset_spec("MOVi-D").name, "movi_d")
        d_id = adapter.stable_observation_id("movi_d", "17", 0, 0)
        e_id = adapter.stable_observation_id("movi_e", "17", 0, 0)
        self.assertNotEqual(d_id, e_id)
        self.assertEqual(len(d_id), 20)

    def test_discover_shards_validates_declared_complete_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                (root / f"movi_e-validation.tfrecord-{index:05d}-of-00003").touch()
            shards = adapter.discover_shards(root, "movi_e")
            self.assertEqual([path.name for path in shards], [
                "movi_e-validation.tfrecord-00000-of-00003",
                "movi_e-validation.tfrecord-00001-of-00003",
                "movi_e-validation.tfrecord-00002-of-00003",
            ])
            (root / "movi_e-validation.tfrecord-00001-of-00003").unlink()
            with self.assertRaises(FileNotFoundError):
                adapter.discover_shards(root, "movi_e")
            partial = adapter.discover_shards(root, "movi_e", require_complete=False)
            self.assertEqual(len(partial), 2)

    def test_schema_and_instance_metadata_normalization(self) -> None:
        record = synthetic_record()
        adapter.validate_record_schema(record, "movi_e")
        rows = adapter.normalize_instance_metadata(record, "movi_e", "17", "pilot")
        self.assertEqual(rows[0]["asset_id"], "asset-a")
        self.assertEqual(rows[0]["category"], "Shoe")
        self.assertTrue(rows[0]["is_dynamic"])
        self.assertEqual(rows[1]["background"], "studio_hdri")
        del record["instances/category"]
        with self.assertRaises(ValueError):
            adapter.validate_record_schema(record, "movi_e")

    def test_camera_world_roundtrip_and_reprojection(self) -> None:
        points = np.asarray([[0.5, -0.25, 4.0], [-0.5, 0.75, 6.0]])
        position = np.asarray([2.0, -1.0, 3.0])
        quaternion = np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        world = adapter.camera_cv_to_world(points, position, quaternion)
        recovered = adapter.world_to_camera_cv(world, position, quaternion)
        np.testing.assert_allclose(recovered, points, atol=1e-12)
        pixels = adapter.project_camera_points(points, 100.0, 100.0, 128, 128)
        expected = np.column_stack(
            (100.0 * points[:, 0] / points[:, 2] + 63.5, 100.0 * points[:, 1] / points[:, 2] + 63.5)
        )
        np.testing.assert_allclose(pixels, expected, atol=1e-12)
        self.assertLess(adapter.rotation_orthonormality_error(quaternion), 1e-12)

    def test_model_leakage_boundary_allows_camera_pose_not_identity(self) -> None:
        extractor.validate_model_record(
            {
                "camera_pose": {
                    "position_world_xyz": [0.0, 0.0, 0.0],
                    "camera_to_world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "pose_aligned_world_visible_surface_centroid_xyz": [0.0, 0.0, -5.0],
            }
        )
        with self.assertRaises(ValueError):
            extractor.validate_model_record({"asset_id": "leaked-object"})
        with self.assertRaises(ValueError):
            extractor.validate_model_record({"gt_world_position_xyz": [0.0, 0.0, 0.0]})

    def test_process_synthetic_movi_e_record(self) -> None:
        args = argparse.Namespace(
            crop_size=16,
            crop_padding=0.15,
            min_visibility_pixels=1,
            min_mask_area=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = extractor.process_video(
                synthetic_record(), "movi_e", "17", "pilot", "synthetic-shard", output, args
            )
            self.assertEqual(len(result.model_rows), 4)
            self.assertEqual(len(result.index_rows), 4)
            self.assertEqual(len(result.diagnostic_rows), 4)
            self.assertEqual(len(result.instance_rows), 2)
            self.assertEqual(len(result.frame_rows), 2)
            self.assertEqual(result.quality["dynamic_instance_count"], 1)
            self.assertEqual(result.quality["static_instance_count"], 1)
            self.assertLess(result.quality["max_pose_roundtrip_error_scene_units"], 1e-12)
            self.assertLess(result.quality["max_reprojection_error_pixels"], 1e-12)
            row = result.model_rows[0]
            self.assertIn("camera_space_visible_surface_centroid_xyz", row)
            self.assertIn("pose_aligned_world_visible_surface_centroid_xyz", row)
            self.assertIn("camera_pose", row)
            self.assertNotIn("asset_id", row)
            self.assertTrue((output / row["rgb_crop_path"]).is_file())
            self.assertTrue((output / row["mask_crop_path"]).is_file())

    def test_seeded_pilot_selection_is_deterministic_and_nonconfirmatory(self) -> None:
        inventory = []
        for index in range(25):
            inventory.append(
                {
                    "dataset": "movi_e",
                    "video_id": str(index),
                    "source_shard": f"shard-{index % 2}",
                    "num_instances": 10 + index % 11,
                    "dynamic_instance_count": 1 + index % 3,
                    "mean_visibility_pixels": 100.0 + index * 5,
                    "camera_translation_scene_units": index / 10,
                    "camera_rotation_degrees": index / 5,
                    "seeded_tie_break": pilot_download.seeded_rank(20260825, "movi_e", str(index)),
                }
            )
        first = pilot_download.select_diverse_pilot(inventory, "movi_e", 20260825, 20)
        second = pilot_download.select_diverse_pilot(inventory, "movi_e", 20260825, 20)
        self.assertEqual(first, second)
        self.assertEqual(len({row["video_id"] for row in first}), 20)
        self.assertTrue(all(row["split"] == "pilot" for row in first))
        self.assertTrue(all(row["confirmatory_test_eligible"] is False for row in first))


if __name__ == "__main__":
    unittest.main()

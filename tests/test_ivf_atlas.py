"""Unit and small-integration tests for the frozen IVF/atlas harness."""
from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import ivf_atlas_common as common


class PolicyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.corpus = rng.normal(size=(80, 6)).astype(np.float32)
        self.mu = self.corpus[:4].copy()
        labels = np.arange(80) % 4
        self.groups = [np.flatnonzero(labels == c).astype(np.uint32) for c in range(4)]
        self.query = self.corpus[0] + np.float32(0.01)
        self.plan = common.candidate_plan(self.query, self.mu, self.groups)

    def test_threshold_recall_uses_distance_not_ids(self):
        scores = np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        self.assertEqual(common.threshold_recall(scores, np.asarray([1.0]))[0], 1.0)
        self.assertEqual(common.threshold_recall(np.asarray([[2.0] * 10]), np.asarray([1.0]))[0], 0.0)

    def test_ratio_always_returns_at_least_one_cell(self):
        self.assertEqual(common.ratio_probe(self.plan, 0.0), 1)

    def test_policy_accepts_non_grid_ratio_probe(self):
        all_d2 = common.brute_d2(self.corpus, self.query[None, :])
        gt_kth = np.asarray([common.top_k(all_d2[0])[1][-1]])
        result = common.policy_result([self.plan], all_d2, gt_kth, [3], len(self.corpus))
        self.assertEqual(result["recall"].shape, (1,))
        self.assertGreater(result["scan"][0], 0.0)

    def test_productive_doubling_guards_and_full_scan(self):
        self.assertGreaterEqual(common.productive_probe(self.plan, 1.0), 1)
        self.assertEqual(common.productive_probe(self.plan, 0.0), 100)

    def test_fixed_and_config_selection_use_validation(self):
        grid = {1: {"recall": np.asarray([0.8]), "scan": np.asarray([0.1])}, 2: {"recall": np.asarray([0.95]), "scan": np.asarray([0.2])}, 4: {"recall": np.asarray([0.99]), "scan": np.asarray([0.4])}}
        self.assertEqual(common.select_fixed(grid, 0.95), 2)
        outcomes = {0.1: grid[2], 0.2: grid[1]}
        self.assertEqual(common.select_config((0.1, 0.2), outcomes, 0.95), 0.1)
        self.assertIsNone(common.select_config((0.2,), outcomes, 0.95))

    def test_target_pass_and_no_valid_comparator(self):
        boot = common.bootstrap_indices(100)
        policy = {"recall": np.full(100, 0.96), "scan": np.full(100, 0.10)}
        fixed = {"recall": np.full(100, 0.96), "scan": np.full(100, 0.20)}
        self.assertEqual(common.evaluate_target(policy, fixed, 0.95, boot)["status"], "PASS")
        invalid = common.evaluate_target(policy, fixed, 0.98, boot)
        self.assertEqual(invalid["status"], "INCONCLUSIVE_NO_VALID_COMPARATOR")

    def test_family_requires_every_target(self):
        passing = {"0.9": {"status": "PASS"}, "0.95": {"status": "PASS"}}
        mixed = {"0.9": {"status": "PASS"}, "0.95": {"status": "FAIL"}}
        self.assertEqual(common.family_verdict(passing), "GO")
        self.assertEqual(common.family_verdict(mixed), "NO_GO")

    def test_read_bin_roundtrip(self):
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.bin"
            path.write_bytes(struct.pack("<QQ", *values.shape) + values.tobytes())
            np.testing.assert_array_equal(common.read_bin(path), values)


class AtlasTests(unittest.TestCase):
    def test_pair_sample_has_no_self_pairs(self):
        left, right = common.pair_sample(20, 100, 3)
        self.assertTrue(np.all(left != right))

    def test_pair_metrics_detect_identical_and_changed_distances(self):
        distances = np.asarray([1.0, 2.0, 3.0, 4.0])
        identical = common.pair_metrics(distances, distances * 2)
        self.assertAlmostEqual(identical["spearman"], 1.0)
        self.assertAlmostEqual(identical["scale_invariant_stress"], 0.0)
        changed = common.pair_metrics(distances, distances[::-1])
        self.assertGreater(changed["scale_invariant_stress"], 0.0)

    def test_knn_overlap_is_perfect_for_identity(self):
        values = np.random.default_rng(9).normal(size=(20, 4))
        self.assertEqual(common.knn_overlap(values, values, 5), 1.0)

    def test_splat_volume_shape_and_clipping_receipt(self):
        coordinates = np.random.default_rng(11).normal(size=(100, 3))
        coordinates[0] = 1000.0
        mip, receipt = common.splat_volume(coordinates, grid=16, sigma=1.0)
        self.assertEqual(mip.shape, (16, 16))
        self.assertGreater(receipt["clipped_fraction"], 0.0)
        self.assertEqual(receipt["volume_bytes"], 16**3 * 4)
        self.assertEqual(len(receipt["volume_sha256"]), 64)


class ReportTests(unittest.TestCase):
    def test_pending_receipts_cannot_derive_metrics(self):
        import report_ivf_atlas as report

        for generator in (report.ivf_report, report.atlas_report):
            text = generator({"status": "PENDING"})
            self.assertIn("VALIDATING", text)
            self.assertNotIn("GO", text)


@unittest.skipUnless(importlib.util.find_spec("faiss"), "faiss-cpu is optional in local development")
class FaissIntegrationTests(unittest.TestCase):
    def test_external_centroids_and_list_sizes(self):
        import faiss

        faiss.omp_set_num_threads(1)
        rng = np.random.default_rng(12)
        corpus = rng.normal(size=(100, 8)).astype(np.float32)
        centroids = corpus[:5].copy()
        quantizer = faiss.IndexFlatL2(8)
        quantizer.add(centroids)
        index = faiss.IndexIVFFlat(quantizer, 8, 5, faiss.METRIC_L2)
        index.add(corpus)
        index.nprobe = 2
        distances, ids = index.search(corpus[:1], 10)
        self.assertEqual(len(ids[0]), 10)
        self.assertTrue(np.all(np.diff(distances[0]) >= -1e-6))
        cells = quantizer.search(corpus[:1], 2)[1][0]
        x2 = np.einsum("ij,ij->i", corpus, corpus)
        m2 = np.einsum("ij,ij->i", centroids, centroids)
        assignments = np.argmin(x2[:, None] - 2 * (corpus @ centroids.T) + m2[None, :], axis=1)
        expected_scan = sum(int(np.count_nonzero(assignments == int(c))) for c in cells)
        self.assertEqual(sum(index.get_list_size(int(c)) for c in cells), expected_scan)


if __name__ == "__main__":
    unittest.main()

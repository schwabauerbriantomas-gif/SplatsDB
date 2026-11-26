"""Integrity audit: unit tests for the provenance helpers and the exact-search
reference of benchmarks/benchmark_reproducible.py.

Pure helpers on synthetic data: no model, no network, no Faiss.
"""
import contextlib
import importlib.util
import inspect
import io
import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_reproducible", ROOT / "benchmarks" / "benchmark_reproducible.py")
BR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BR)


def capture(fn, *args):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args)
    return result, buffer.getvalue()


class AnnDownloadSources(unittest.TestCase):
    def test_every_dataset_is_downloaded_from_the_official_host(self):
        # A Hugging Face mirror that does not exist answered 401 for every dataset.
        for name, config in BR.ANN_DATASETS.items():
            self.assertEqual(config["url"], f"https://ann-benchmarks.com/{name}.hdf5", name)

    def test_ann_function_never_labels_a_numpy_result_as_splatsdb(self):
        source = inspect.getsource(BR.run_ann_benchmark)
        self.assertNotIn("splatsdb_cpu", source)
        self.assertIn('results["splatsdb_measured"] = False', source)
        self.assertIn("NumPy exact reference", source)
        self.assertNotIn("p50_ms", source)


class ExactTopK(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.train = self.rng.normal(size=(400, 12)).astype(np.float32)
        self.test = self.rng.normal(size=(37, 12)).astype(np.float32)

    def test_angular_matches_a_direct_cosine_argsort(self):
        found = BR.exact_topk_blockwise(BR._unit_rows(self.test), BR._unit_rows(self.train), 5, True)
        sims = BR._unit_rows(self.test) @ BR._unit_rows(self.train).T
        direct = np.argsort(-sims, axis=1)[:, :5]
        for row in range(len(self.test)):
            self.assertEqual(set(found[row]), set(direct[row]), row)

    def test_euclidean_matches_a_direct_distance_argsort(self):
        found = BR.exact_topk_blockwise(self.test, self.train, 5, False)
        d2 = ((self.test[:, None, :] - self.train[None, :, :]) ** 2).sum(axis=2)
        direct = np.argsort(d2, axis=1)[:, :5]
        for row in range(len(self.test)):
            self.assertEqual(set(found[row]), set(direct[row]), row)

    def test_block_size_does_not_change_the_answer(self):
        for angular in (True, False):
            reference = BR.exact_topk_blockwise(self.test, self.train, 4, angular, block=400)
            for block in (1, 7, 36):
                other = BR.exact_topk_blockwise(self.test, self.train, 4, angular, block=block)
                np.testing.assert_array_equal(reference, other, err_msg=f"block={block} angular={angular}")

    def test_blockwise_reference_allocates_one_block_of_scores_not_the_full_matrix(self):
        # Guard the memory claim with tracemalloc: the largest temporary must scale
        # with the block, not with queries x train.
        import tracemalloc
        big_train = self.rng.normal(size=(20000, 64)).astype(np.float32)
        big_test = self.rng.normal(size=(600, 64)).astype(np.float32)
        tracemalloc.start()
        BR.exact_topk_blockwise(big_test, big_train, 10, True, block=32)
        _, peak_block = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        full = big_test.shape[0] * big_train.shape[0] * 4
        self.assertLess(peak_block, 0.5 * full, f"peak {peak_block} vs full matrix {full}")


class RecallAndNormalisation(unittest.TestCase):
    def test_unit_rows_keep_a_zero_row_at_zero(self):
        rows = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        out = BR._unit_rows(rows)
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 0.0], atol=1e-6)

    def test_recall_at_k_is_the_mean_fraction_recovered(self):
        found = np.array([[0, 1, 2], [9, 8, 7]])
        truth = np.array([[1, 2, 3], [7, 6, 5]])
        self.assertAlmostEqual(BR._recall_at_k(found, truth), (2 / 3 + 1 / 3) / 2)


class Provenance(unittest.TestCase):
    def test_attestation_note_is_added_next_to_every_checklist_at_any_depth(self):
        result = {"suite": {"integrity_checklist": {"code_reproducible": True}}, "list": [{"integrity_checklist": {"seed": 42}}]}
        BR.annotate_attestations(result)
        self.assertEqual(result["suite"]["integrity_checklist_note"], BR.ATTESTATION_NOTE)
        self.assertEqual(result["list"][0]["integrity_checklist_note"], BR.ATTESTATION_NOTE)

    def test_an_existing_note_is_not_overwritten(self):
        result = {"integrity_checklist": {}, "integrity_checklist_note": "written by the suite"}
        BR.annotate_attestations(result)
        self.assertEqual(result["integrity_checklist_note"], "written by the suite")

    def test_measured_suite_records_elapsed_time_and_completion(self):
        results = {}
        out = BR.run_and_record(results, "suite", lambda: {"x": 1, "integrity_checklist": {"code_reproducible": True}})
        self.assertIs(results["suite"], out)
        provenance = out["run_provenance"]
        self.assertTrue(provenance["measured_in_this_run"])
        self.assertEqual(provenance["result_source"], "computed_by_this_function")
        self.assertIsInstance(provenance["elapsed_s"], float)
        self.assertGreaterEqual(provenance["elapsed_s"], 0.0)
        self.assertEqual(len(provenance["completed_at"]), 19)
        self.assertIn("integrity_checklist_note", out)

    def test_placeholder_suite_is_marked_as_stored_numbers(self):
        results = {}
        out = BR.run_and_record(results, "suite", lambda: {"measured_by_this_function": False})
        self.assertFalse(out["run_provenance"]["measured_in_this_run"])
        self.assertEqual(out["run_provenance"]["result_source"], "stored_result_files")
        self.assertNotIn("elapsed_s", out["run_provenance"])

    def test_a_suite_that_returns_nothing_leaves_a_trace(self):
        results = {}
        BR.run_and_record(results, "suite", lambda: None)
        self.assertEqual(sorted(results), ["suite_not_run"])

    def test_summary_names_measured_suites_and_never_calls_them_integrity_checked(self):
        results = {"tool": "x", "version": "1", "philosophy": "x", "hardware": {}, "timestamp": "x",
                   "a": {"run_provenance": {"elapsed_s": 1.5}}, "b_error": "boom", "c_not_run": "none",
                   "d": {"measured_by_this_function": False}}
        measured, text = capture(BR.summarize_provenance, results)
        self.assertEqual(measured, ["a"])
        self.assertIn("a: 1.5 s", text)
        self.assertIn("verifies none of the integrity flags", text)
        self.assertNotIn("NO hardcoding", text)

    def test_summary_says_so_when_nothing_was_measured(self):
        results = {"tool": "x", "version": "1", "philosophy": "x", "hardware": {}, "timestamp": "x",
                   "ann_x_error": "HTTP Error 401: Unauthorized"}
        measured, text = capture(BR.summarize_provenance, results)
        self.assertEqual(measured, [])
        self.assertIn("NOTHING WAS MEASURED", text)

    def test_hardware_is_detected_at_run_time(self):
        hardware = BR.detect_hardware()
        self.assertEqual(sorted(hardware), ["cpu", "detected_at_run_time", "gpu", "os", "ram"])
        self.assertIs(hardware["detected_at_run_time"], True)
        self.assertTrue(all(isinstance(hardware[key], str) for key in ("cpu", "gpu", "os", "ram")))


class ClaimsReconciler(unittest.TestCase):
    def setUp(self):
        import tempfile
        spec = importlib.util.spec_from_file_location("reconcile_claims", ROOT / "benchmarks" / "reconcile_claims.py")
        self.rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.rc)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rc.ROOT = self.root
        self.addCleanup(self.tmp.cleanup)

    def test_a_number_is_traceable_at_its_precision_directly_or_as_a_difference(self):
        values = self.rc.matchable([0.7729, 0.7372, 149.7, 0.0, 0.0])
        self.assertTrue(self.rc.cell_matches("77.3%", values))
        self.assertTrue(self.rc.cell_matches("3.6%", values), "a printed delta must trace to a difference")
        self.assertTrue(self.rc.cell_matches("149.7", values))
        self.assertFalse(self.rc.cell_matches("77.4%", values))
        self.assertFalse(self.rc.cell_matches("3,125", values))

    def test_only_tables_with_a_cited_result_file_are_checked(self):
        (self.root / "bench-data").mkdir()
        (self.root / "bench-data" / "x.json").write_text(json.dumps({"a": 0.5, "b": 25}))
        (self.root / "README.md").write_text(
            "| Metric | Value |\n|---|---|\n| unchecked | 99.9% |\n\n"
            + "\n" * 20  # further than the citation window: this table has no source and is not checked
            + "Source: `bench-data/x.json`\n\n"
            + "| Metric | Value |\n|---|---|\n| ok | 50.0% |\n| invented | 12.5% |\n")
        self.rc.DOCS = ("README.md",)
        self.assertEqual(self.rc.untraceable_cells("README.md"), ["README.md:12.5%"])


    def test_a_citation_is_assigned_to_the_nearest_table_only(self):
        # Regression: a citation under table A used to make table B, a few lines
        # below, be checked against A's file and report every cell of B as new.
        (self.root / "bench-data").mkdir()
        (self.root / "bench-data" / "x.json").write_text(json.dumps({"a": 0.5}))
        (self.root / "README.md").write_text(
            "Source: `bench-data/x.json`\n\n"
            "| Metric | Value |\n|---|---|\n| ok | 50.0% |\n\n"
            "| Metric | Value |\n|---|---|\n| from another run | 12.5% |\n")
        self.assertEqual(self.rc.untraceable_cells("README.md"), [])


if __name__ == "__main__":
    unittest.main()

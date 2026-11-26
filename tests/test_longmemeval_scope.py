"""Unit tests for benchmarks/longmemeval_scope.py — pure helpers on synthetic data."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import longmemeval_scope as lm  # noqa: E402
import report_longmemeval_scope as report  # noqa: E402


class StreamJsonArrayTest(unittest.TestCase):
    def test_yields_objects_with_small_read_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            p.write_text('[\n  {"i": 1, "s": "' + "x" * 5000 + '"},\n  {"i": 2},\n  {"i": 3}\n]')
            self.assertEqual([o["i"] for o in lm.stream_json_array(p, chunk=64)], [1, 2, 3])

    def test_rejects_non_array(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            p.write_text('{"i": 1}')
            with self.assertRaises(ValueError):
                list(lm.stream_json_array(p))

    def test_truncated_array_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            p.write_text('[{"i": 1}, {"i": 2}')
            with self.assertRaises(EOFError):
                list(lm.stream_json_array(p))


class SessionLinesTest(unittest.TestCase):
    def test_role_content_and_missing_fields(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant"}, {}]
        self.assertEqual(lm.session_lines(msgs), ["user: hi", "assistant: ", ": "])


class AssemblePiecesTest(unittest.TestCase):
    def test_packs_until_budget_and_splits_long_lines(self):
        line_session = [0, 0, 0, 1]
        encoded = [[1, 2], [3, 4, 5], list(range(10, 17)), [7, 7]]
        self.assertEqual(
            lm.assemble_pieces(line_session, encoded, budget=5),
            [(0, [1, 2, 3, 4, 5]), (0, [10, 11, 12, 13, 14]), (0, [15, 16]), (1, [7, 7])],
        )

    def test_empty_lines_are_skipped(self):
        self.assertEqual(lm.assemble_pieces([0, 0], [[], []], budget=5), [])

    def test_never_exceeds_budget(self):
        rng = np.random.default_rng(0)
        line_session = [int(i // 3) for i in range(30)]
        encoded = [list(range(int(n))) for n in rng.integers(1, 12, size=30)]
        for sess, ids in lm.assemble_pieces(line_session, encoded, budget=7):
            self.assertLessEqual(len(ids), 7)
            self.assertLessEqual(sess, 9)


class TopkTest(unittest.TestCase):
    def test_ties_break_by_ascending_index(self):
        scores = np.array([0.5, 0.9, 0.9, 0.1], dtype=np.float32)
        self.assertEqual(lm.topk(scores, None, 3).tolist(), [1, 2, 0])

    def test_restriction_keeps_scores(self):
        scores = np.array([0.5, 0.9, 0.9, 0.1], dtype=np.float32)
        self.assertEqual(lm.topk(scores, np.array([3, 0]), 2).tolist(), [0, 3])


class HitsAtTest(unittest.TestCase):
    def test_any_answer_in_prefix(self):
        self.assertEqual(lm.hits_at([5, 1, 2], {2, 9}, (1, 3)), {1: False, 3: True})
        self.assertEqual(lm.hits_at([], {2}, (1,)), {1: False})


class FtsQueryTest(unittest.TestCase):
    def test_or_of_quoted_tokens(self):
        self.assertEqual(lm.fts_query("What is my favorite cuisine?"), '"cuisine" OR "favorite" OR "is" OR "my" OR "what"')

    def test_no_tokens(self):
        self.assertEqual(lm.fts_query("?! ..."), "")


class RrfTest(unittest.TestCase):
    def test_lists_agreeing_win(self):
        self.assertEqual(lm.rrf([[0, 1], [1, 2]]), [1, 0, 2])

    def test_empty(self):
        self.assertEqual(lm.rrf([]), [])


class StatsTest(unittest.TestCase):
    def test_summarize_per_type(self):
        questions = [{"qtype": "a"}, {"qtype": "a"}, {"qtype": "b"}]
        s = lm.summarize(np.array([True, False, True]), questions)
        self.assertEqual((s["n"], s["hits"], s["rate"]), (3, 2, 2 / 3))
        self.assertEqual(s["per_type"]["a"]["hits"], 1)

    def test_ci_and_paired(self):
        hit = np.ones(50, dtype=bool)
        self.assertEqual(lm.ci95(hit, n_boot=50), [1.0, 1.0])
        d = lm.paired_delta(np.array([True, False]), np.array([True, False]), n_boot=20)
        self.assertEqual(d["delta"], 0.0)
        self.assertEqual(d["ci95"], [0.0, 0.0])


class KeywordRankTest(unittest.TestCase):
    def test_global_and_within_share_one_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            conn = lm.build_fts(["alpha beta", "gamma delta alpha", "epsilon zeta"], Path(d) / "fts.sqlite")
            glob, within = lm.keyword_rank(conn, "alpha", np.array([0, 2]))
            conn.close()
        self.assertEqual(sorted(glob), [0, 1])
        self.assertEqual(within, [0])

    def test_no_tokens_gives_no_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            conn = lm.build_fts(["alpha"], Path(d) / "fts.sqlite")
            self.assertEqual(lm.keyword_rank(conn, "?! ...", np.array([0])), ([], []))
            conn.close()


class BuildCorpusTest(unittest.TestCase):
    def test_dedupes_sessions_shared_across_questions(self):
        data = [
            {"question_id": "q1", "question_type": "t", "question": "a?", "answer_session_ids": ["s2"],
             "haystack_session_ids": ["s1", "s2"],
             "haystack_sessions": [[{"role": "user", "content": "one"}], [{"role": "assistant", "content": "two"}]]},
            {"question_id": "q2", "question_type": "t", "question": "b?", "answer_session_ids": ["s1"],
             "haystack_session_ids": ["s2", "s3"],
             "haystack_sessions": [[{"role": "user", "content": "two"}], [{"role": "user", "content": "three"}]]},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ds.json"
            p.write_text(json.dumps(data))
            texts, questions, n_slots = lm.build_corpus(p)
        self.assertEqual(n_slots, 4)
        self.assertEqual(texts, ["user: one", "assistant: two", "user: three"])
        self.assertEqual(questions[0]["slots"], [0, 1])
        self.assertEqual(questions[0]["answers"], [1])
        self.assertEqual(questions[1]["slots"], [1, 2])
        self.assertEqual(questions[1]["answers"], [0])
        self.assertEqual([q["answers_missing"] for q in questions], [0, 0])


class ReportTest(unittest.TestCase):
    def test_report_is_in_sync_with_receipt(self):
        receipt = json.loads((ROOT / "bench-data/longmemeval_scope_results.json").read_text())
        self.assertEqual(report.build(receipt), (ROOT / "docs/validation/longmemeval-scope.md").read_text())

    def test_r2_verdict_branches(self):
        self.assertEqual(report.r2_verdict({"delta": -0.1, "ci95": [-0.2, -0.05]})[0], "`DENSE_BELOW_KEYWORD`")
        self.assertEqual(report.r2_verdict({"delta": 0.1, "ci95": [0.05, 0.2]})[0], "`DENSE_ABOVE_KEYWORD`")
        self.assertEqual(report.r2_verdict({"delta": 0.0, "ci95": [-0.05, 0.05]})[0], "`NO_DIFFERENCE_RESOLVED`")


if __name__ == "__main__":
    unittest.main()

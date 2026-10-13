"""Offline tests for routing, masking, result filtering, and metrics."""
import json
import logging
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import run_masking_and_planning

from parallel_runner import AggregatedResult, ParallelExperimentRunner, SkillsManager, DockerImageCache
from run_masking_and_planning import _make_masked_build_skills_block
from tokenplan_client import build_chat_completions_url
from analyze_skill_substitution import loaded_names, cosine
import numpy as np
import pool_order

class RoutingTests(unittest.TestCase):
    def test_plan_first_alias_preserves_resume_identity(self):
        for name in ("plan-first", "nudge"):
            with self.subTest(name=name), patch("sys.argv", ["run_masking_and_planning.py", "--experiment", name, "--run"]), patch.object(run_masking_and_planning, "run_experiment") as run:
                run_masking_and_planning.main()
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0], "nudge")

    def test_masks_preserve_gold_and_unmasked_fields(self):
        fields = {"name","description","location"}
        gold = SimpleNamespace(name="gold",description="gold description",location="/skills/gold/SKILL.md")
        extra = SimpleNamespace(name="extra",description="extra description",location="/skills/extra/SKILL.md")
        for targets in [{"name"},{"description"},{"location"},{"name","location"},fields]:
            with self.subTest(targets=targets), tempfile.TemporaryDirectory() as d:
                agent = SimpleNamespace(_ablation_golden_names={"gold"},logs_dir=Path(d),_format_loaded_skills=lambda:"None")
                loader = SimpleNamespace(get_metadata=lambda:[gold,extra])
                block = _make_masked_build_skills_block(None,targets)(agent,loader)
                data,_ = json.JSONDecoder().raw_decode(block.split("available_skills:\n",1)[1])
                self.assertEqual(data[0],vars(gold))
                for f in fields:
                    if f in targets: self.assertIn("<mask>",data[1][f])
                    else: self.assertEqual(data[1][f],getattr(extra,f))
                self.assertTrue((Path(d)/"skills_context_preview.txt").is_file())

    def test_pass_at_k(self):
        for n in range(1,6):
            for c in range(n+1):
                for k in range(1,n+1):
                    r = AggregatedResult("model",0,task_results={"task":[True]*c+[False]*(n-c)})
                    expected = 1.0 if n-c<k else 1-math.comb(n-c,k)/math.comb(n,k)
                    self.assertAlmostEqual(r.pass_at_k(k),expected)

    def test_resume_accepts_terminal_failure_rejects_retry(self):
        runner = object.__new__(ParallelExperimentRunner)
        runner.logger = logging.getLogger("offline-test")
        identity = ("model","task",1000,0,42)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"result.json"
            p.write_text(json.dumps({"agent_result":{"error":"Max episodes reached"},"final_success":False}))
            self.assertIsNotNone(runner._load_completed_result(p,identity))
            marker=Path(d)/"retry_required.json"; marker.write_text('{"reason":"test"}')
            self.assertIsNone(runner._load_completed_result(p,identity))
            marker.unlink()
            p.write_text(json.dumps({"agent_result":{},"infrastructure_failure":True}))
            self.assertIsNone(runner._load_completed_result(p,identity))
            p.write_text('{"agent_result":null}')
            self.assertIsNone(runner._load_completed_result(p,identity))

    def test_pool_order_and_duplicate_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); pool=root/"pool"; pool.mkdir()
            for name in ["b","gold","a"]:
                (pool/name).mkdir(); (pool/name/"SKILL.md").write_text(name)
            manifest=root/"pool.json"
            manifest.write_text(json.dumps([{"directory":n} for n in ["gold","a","b"]]))
            with patch.object(pool_order,"MANIFEST",manifest):
                manager=SkillsManager(pool)
                self.assertEqual(manager.get_available_skills(),["gold","a","b"])
                self.assertEqual({p.name for p in manager.sample_skills(2,["gold"],42)},{"a","b"})
                dest=root/"target"; (dest/"a").mkdir(parents=True)
                (dest/"a/SKILL.md").write_text("preserve")
                manager.copy_skills_to_dir([pool/"a"],dest)
                self.assertEqual((dest/"a/SKILL.md").read_text(),"preserve")
                (pool/"a/SKILL.md").unlink()
                with self.assertRaises(ValueError): pool_order.ordered_pool_paths(pool)

    def test_chat_url(self):
        self.assertEqual(build_chat_completions_url("https://example.invalid/v1/"),"https://example.invalid/v1/chat/completions")
        self.assertEqual(build_chat_completions_url("https://example.invalid/v1/chat/completions"),"https://example.invalid/v1/chat/completions")

    def test_cosine(self):
        self.assertAlmostEqual(cosine(np.array([1,0]),np.array([1,0])),1.0)
        self.assertAlmostEqual(cosine(np.array([1,0]),np.array([0,1])),0.0)

    def test_strict_labels_override_legacy_result(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/"result.json").write_text(json.dumps({"agent_result":{"loaded_skills":[{"skill":"incorrect-legacy-label"}]}}))
            (p/"skills_loaded.strict-v3.json").write_text(json.dumps({"parser_version":"strict-v3-context-checked","skill_names":["validated"]}))
            self.assertEqual(loaded_names(p),{"validated"})

    def test_global_pruning_disabled(self):
        import asyncio
        with patch.dict("os.environ",{},clear=True), patch("subprocess.run") as run:
            asyncio.run(DockerImageCache._cleanup_failed_build_artifacts(object()))
            run.assert_not_called()

if __name__ == "__main__": unittest.main()

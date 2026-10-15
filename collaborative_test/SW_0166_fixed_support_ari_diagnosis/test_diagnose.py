import os
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from collaborative_test.SW_0166_fixed_support_ari_diagnosis import diagnose


class FixedSupportTests(unittest.TestCase):
    def fixture(self):
        gt=np.zeros((32,32),dtype=np.int64); gt.flat[:4]=[1,1,2,2]
        candidate=np.zeros_like(gt); candidate.flat[:3]=[1,2,2]
        source=np.zeros_like(gt); source.flat[:3]=[1,2,2]
        return gt,candidate,source

    def test_pair_counts_match_independent_pair_enumeration(self):
        gt,cand,src=self.fixture(); row=diagnose.diagnose_one(gt,cand,src)
        names=("foreground_pair_count","same_gt_pair_count","different_gt_pair_count",
               "covered_merge_pair_count","covered_split_pair_count","forced_bg_merge_pair_count",
               "forced_cross_support_split_pair_count")
        counts={k:0 for k in names}
        for i in range(gt.size):
            for j in range(i+1,gt.size):
                if gt.flat[i]==0 or gt.flat[j]==0: continue
                counts["foreground_pair_count"]+=1
                same=gt.flat[i]==gt.flat[j]
                counts["same_gt_pair_count" if same else "different_gt_pair_count"]+=1
                ci,cj=src.flat[i]>0,src.flat[j]>0
                if ci and cj:
                    if same and cand.flat[i]!=cand.flat[j]: counts["covered_split_pair_count"]+=1
                    if not same and cand.flat[i]==cand.flat[j]: counts["covered_merge_pair_count"]+=1
                elif not ci and not cj and not same: counts["forced_bg_merge_pair_count"]+=1
                elif same and ci!=cj: counts["forced_cross_support_split_pair_count"]+=1
        for key,value in counts.items(): self.assertEqual(row[key],value,key)
        self.assertEqual(row["covered_patch_count"],3); self.assertEqual(row["missed_patch_count"],1)
        self.assertEqual(row["cross_support_gt_object_count"],1)
        self.assertAlmostEqual(row["merge_correction_delta"]+row["split_correction_delta"],
                               row["fixed_support_ari_upper_bound"]-row["actual_fg_ari"],places=12)
        self.assertGreaterEqual(row["fixed_support_ari_upper_bound"]+1e-12,row["actual_fg_ari"])

    def test_idealized_attachment_preserves_covered_groups_and_assigns_missed(self):
        gt,cand,_=self.fixture(); attached=diagnose._idealized_attachment(gt,cand)
        np.testing.assert_array_equal(attached.flat[:3],cand.flat[:3])
        self.assertEqual(attached.flat[3],2); self.assertTrue(np.array_equal(attached>0,gt>0))

    def test_uncovered_only_has_no_empty_table_failure(self):
        gt=np.zeros((32,32),dtype=np.int64); gt.flat[:3]=[1,2,2]
        labels=np.zeros_like(gt); row=diagnose.diagnose_one(gt,labels,labels)
        self.assertEqual(row["covered_patch_count"],0); self.assertEqual(row["missed_patch_count"],3)
        self.assertEqual(row["forced_bg_merge_pair_count"],2); self.assertEqual(row["covered_merge_pair_count"],0)
        self.assertTrue(np.isfinite(row["actual_fg_ari"]))
        self.assertGreaterEqual(row["fixed_support_ari_upper_bound"]+1e-12,row["actual_fg_ari"])

    def test_rejects_support_change_or_shape_mismatch(self):
        gt,cand,src=self.fixture(); cand.flat[3]=8
        with self.assertRaisesRegex(ValueError,"membership"): diagnose.diagnose_one(gt,cand,src)
        with self.assertRaises(ValueError): diagnose.diagnose_one(gt[:2],cand[:2],src[:2])

    def test_fixed320_flags_are_deterministic_and_separate_from_bootstrap(self):
        gt=np.zeros((320,32,32),dtype=np.int64); gt[:,:,:2]=1; gt[:,:,2:4]=2
        src=np.zeros_like(gt); src[:,:,:4]=1
        candidate=np.zeros_like(gt); candidate[:,:,:4]=7
        slot=gt.copy()
        frozen={s:{"candidate":candidate,"source_T":src,"slot":slot,
                   "score_sha256":f"{s}"*64,"score_path":f"score{s}","predictions":{},"references_sha256":{},"queue_receipt":None}
                for s in diagnose.SEEDS}
        report=diagnose.build_diagnostic(frozen,gt,dataset_info={"path":"fixture"})
        self.assertEqual(report["fixed320_deterministic_feasibility"],{
            "fixed320_cannot_reach_0p87":False,
            "fixed320_cannot_strictly_beat_slot":True,
            "fixed320_joint_goal_impossible_under_support":True})
        self.assertAlmostEqual(report["three_seed_fixed320_means"]["fixed_support_ari_upper_bound"],1.0)
        self.assertAlmostEqual(report["three_seed_fixed320_means"]["actual_fg_ari"],0.0)
        self.assertEqual(report["paired_bootstrap"]["fixed_support_ari_upper_bound_minus_constant_0p87"]["replicates"],10_000)

    def test_invalid_gt_foreground_row_is_reported_and_shared_across_seed_bootstrap(self):
        gt=np.zeros((320,32,32),dtype=np.int64); gt[:,:,:2]=1; gt[:,:,2:4]=2
        gt[0].fill(0); gt[0,0,0]=1
        src=np.zeros_like(gt); src[:,:,:4]=1
        candidate=np.zeros_like(gt); candidate[:,:,:4]=7
        frozen={s:{"candidate":candidate,"source_T":src,"slot":gt.copy(),
                   "score_sha256":f"{s}"*64,"score_path":f"score{s}","predictions":{},"references_sha256":{},"queue_receipt":None}
                for s in diagnose.SEEDS}
        report=diagnose.build_diagnostic(frozen,gt,dataset_info={"path":"fixture"})
        self.assertEqual(report["per_seed"]["0"]["valid_count"],319)
        self.assertEqual(report["per_seed"]["0"]["per_image"][0]["actual_fg_ari"],None)
        self.assertEqual(report["paired_bootstrap_spec"]["valid_count"],319)
        self.assertEqual(report["paired_bootstrap_spec"]["valid_image_offsets"][0],1)

    def test_public_validation_order_all_seeds_before_any_score_validator(self):
        events=[]; labels=np.zeros((320,32,32),dtype=np.int64); labels[:,:,:16]=1
        base=diagnose.HERE/f"test_artifacts_{os.getpid()}"
        if base.exists(): raise RuntimeError(f"refusing to reuse fixture path {base}")
        base.mkdir(); self.addCleanup(lambda: shutil.rmtree(base))
        contexts={}
        for seed in diagnose.SEEDS:
            refs={}
            for name in ("source_T","slot"):
                npz=base/f"{seed}_{name}.npz"; proto=base/f"{seed}_{name}.json"
                npz.write_bytes(f"ref-{seed}-{name}".encode()); proto.write_text("{}",encoding="utf-8")
                refs[name]={"labels":labels.copy(),"ids":diagnose.IMAGE_IDS.copy(),"prediction_path":npz,"protocol_path":proto}
            prepared={"references":refs}; pred_results={}
            for arm in ("foreground_induced","source_T_control"):
                folder=base/f"{seed}_{arm}"; folder.mkdir()
                pp=folder/"predictions.npz"; pr=folder/"protocol.json"
                pp.write_bytes(f"pred-{seed}-{arm}".encode()); pr.write_text("{}",encoding="utf-8")
                pred_results[arm]={"labels":labels.copy(),"directory":folder,
                    "prediction_sha256":diagnose.sha256_file(pp),"protocol_sha256":diagnose.sha256_file(pr)}
            score=base/f"score{seed}.json"; score.write_text(f"score-{seed}",encoding="utf-8")
            contexts[seed]=(prepared,pred_results,score)

        class FakeEval:
            ARMS=("foreground_induced","source_T_control")
            def __init__(self,seed):
                self.seed=seed; self.SCORE_PATH=base/"score1.json"
            def prepare_prediction_cpu(self,**kwargs): return contexts[self.seed][0]
            def validate_prediction(self,arm,**kwargs):
                events.append(("prediction",self.seed,arm)); return contexts[self.seed][1][arm]
            def validate_score(self,path,**kwargs):
                events.append(("score",self.seed)); return {"score_sha256":diagnose.sha256_file(path)}
            def score_path(self,seed): return contexts[seed][2]
        eval164=FakeEval(1); eval165=FakeEval(0)
        def fake_stage(seed,*_):
            return {"score_sha256":diagnose.sha256_file(contexts[seed][2]),"state_path":"state","state_sha256":"a"*64}
        seed1_score=base/"score1.json"; seed1_score.write_text("score-1",encoding="utf-8")
        seed1_sha=diagnose.sha256_file(seed1_score)
        with mock.patch.object(diagnose,"SW0164_SEED1_SCORE_SHA256",seed1_sha):
            with mock.patch.object(diagnose,"_load_modules_and_prepare",return_value=(eval164,eval165,SimpleNamespace())):
                with mock.patch.object(diagnose,"_check165_stage",side_effect=fake_stage):
                    frozen=diagnose._freeze_predictions()
        pred=[i for i,e in enumerate(events) if e[0]=="prediction"]
        score=[i for i,e in enumerate(events) if e[0]=="score"]
        self.assertEqual(len(pred),6); self.assertEqual(len(score),3)
        self.assertLess(max(pred),min(score)); self.assertEqual(set(frozen),set(diagnose.SEEDS))


if __name__=="__main__": unittest.main()

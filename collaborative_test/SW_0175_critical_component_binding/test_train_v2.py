import tempfile
import unittest
from pathlib import Path
import numpy
import torch
from collaborative_test.SW_0175_critical_component_binding import train_v2 as train
from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint

class TrainingScheduleTests(unittest.TestCase):
    def test_registered_schedule_is_two_numpy_permutations(self):
        expected_rng=numpy.random.RandomState(171001)
        expected=[expected_rng.permutation(512).astype(int).tolist() for _ in range(2)]
        self.assertEqual(train._orders(),expected)
        batches=train._expected_batches()
        self.assertEqual(len(batches),64)
        self.assertTrue(all(len(x)==16 for x in batches))
        exposures=[i for batch in batches for i in batch]
        self.assertEqual(sorted(exposures[:512]),list(range(512)))
        self.assertEqual(sorted(exposures[512:]),list(range(512)))

    def test_cursor_accepts_only_committed_registered_order(self):
        batches=train._expected_batches()
        history=[{"update":n+1,"image_indices":batch} for n,batch in enumerate(batches[:3])]
        train._validate_cursor(3,history)
        history[1]={"update":2,"image_indices":list(reversed(batches[1]))}
        with self.assertRaises(ValueError): train._validate_cursor(3,history)

    def test_boundary_resume_restores_latest_completed_update(self):
        temp_parent=Path(__file__).resolve().parents[2]/"collaborative_testing"/"temporary"
        temp_parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sw175_resume_",dir=temp_parent) as td:
            root=Path(td)
            wrapped=torch.nn.Linear(2,2); encoder=torch.nn.Linear(2,1)
            optimizer=torch.optim.Adam([{"params":list(wrapped.parameters()),"name":"core"},
                {"params":list(encoder.parameters()),"name":"context"}],lr=.01)
            bindings={"experiment":train.EXPERIMENT,"arm":"native_Q","fingerprint":"a"*64}
            runtime=checkpoint.runtime_signature("cpu")
            (wrapped(torch.ones(1,2))+encoder(torch.ones(1,2))).square().sum().backward()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            expected={"wrapped":{k:v.detach().clone() for k,v in wrapped.state_dict().items()},
                "encoder":{k:v.detach().clone() for k,v in encoder.state_dict().items()}}
            history=[{"update":1,"image_indices":train._expected_batches()[0]}]
            checkpoint.save_boundary(root,completed_updates=1,wrapped=wrapped,encoder=encoder,
                optimizer=optimizer,history=history,bindings=bindings,runtime=runtime,device="cpu")
            with torch.no_grad():
                for p in wrapped.parameters(): p.add_(10)
                for p in encoder.parameters(): p.add_(10)
            resumed=checkpoint.load_latest(root,wrapped=wrapped,encoder=encoder,optimizer=optimizer,
                expected_bindings=bindings,expected_runtime=runtime,device="cpu",max_updates=train.UPDATES)
            self.assertEqual(resumed["completed_updates"],1)
            self.assertEqual(resumed["history"],history)
            for section,module in (("wrapped",wrapped),("encoder",encoder)):
                for name,value in module.state_dict().items():
                    self.assertTrue(torch.equal(value,expected[section][name]))
            train._validate_cursor(resumed["completed_updates"],resumed["history"])

    def test_actual_objective_dispatch_uses_two_no_auxiliary_arms(self):
        from collaborative_test.SW_0175_critical_component_binding import critical_component_objective as obj
        q=torch.full((4,4),.5,requires_grad=True); q.data.fill_diagonal_(1.)
        rho=torch.stack([torch.eye(4,requires_grad=True) for _ in range(4)])
        target=[[1,1],[1,1]]; support=[[1,1],[1,1]]
        native=train._loss_for_arm("native_Q",q,rho,target,support)
        candidate=train._loss_for_arm("critical_component",q,rho,target,support)
        self.assertTrue(torch.isfinite(native) and torch.isfinite(candidate))
        self.assertAlmostEqual(float(native.detach()),float(obj.native_q_loss(q,target,support).detach()),places=7)
        self.assertAlmostEqual(float(candidate.detach()),float(obj.critical_component_loss(q,rho,target,support).detach()),places=7)
        self.assertNotEqual(native.detach().item(),candidate.detach().item())
        with self.assertRaises(ValueError): train._loss_for_arm("other",q,rho,target,support)

    def test_multiclass_cached_source_labels_are_normalized_at_loss_adapter(self):
        from collaborative_test.SW_0175_critical_component_binding import critical_component_objective as obj
        support_labels = numpy.asarray([[13, 13], [7, 0]], dtype=numpy.int32)
        target = numpy.asarray([[1, 1], [2, 0]], dtype=numpy.int32)
        q = torch.tensor([[1., .72, .31, .2], [.72, 1., .28, .1],
                          [.31, .28, 1., .05], [.2, .1, .05, 1.]], dtype=torch.float64,
                         requires_grad=True)
        rho = torch.full((4, 4, 4), .4, dtype=torch.float64, requires_grad=True)
        expected_support = support_labels > 0
        self.assertTrue(numpy.array_equal(train._support_mask(expected_support), expected_support))
        expected_native = obj.native_q_loss(q, target, expected_support)
        actual_native = train._native_q_loss(q, target, support_labels)
        actual_candidate = train._loss_for_arm("critical_component", q, rho, target,
                                                support_labels)
        expected_candidate = obj.critical_component_loss(q, rho, target, expected_support)
        self.assertTrue(torch.equal(actual_native, expected_native))
        self.assertTrue(torch.equal(actual_candidate, expected_candidate))
        self.assertTrue(torch.isfinite(actual_native) and torch.isfinite(actual_candidate))
        (actual_native + actual_candidate).backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(rho.grad).all())

    def test_support_adapter_rejects_malformed_geometry(self):
        with self.assertRaises(ValueError):
            train._support_mask(numpy.ones((4,), dtype=numpy.int32))
        with self.assertRaises(ValueError):
            train._support_mask(numpy.ones((2, 2), dtype=numpy.float32))
        with self.assertRaises(ValueError):
            train._support_mask(-numpy.ones((2, 2), dtype=numpy.int32))

if __name__=="__main__": unittest.main()

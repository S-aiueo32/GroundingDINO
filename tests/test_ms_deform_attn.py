"""Regression checks for the standalone deformable-attention backend.

Run: python -m unittest discover -s tests -v
"""
import copy
import importlib.util
from itertools import product
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

# Load the attention module without importing unrelated detector dependencies.
MODULE_PATH = (Path(__file__).resolve().parents[1] / "groundingdino" / "models"
               / "GroundingDINO" / "ms_deform_attn.py")
spec = importlib.util.spec_from_file_location("attention", MODULE_PATH)
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)


def reference(value, shapes, starts, locations, weights, step):
    return attention.multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)


class AttentionTests(unittest.TestCase):
    def compare_reference(self, device):
        for batch_first in (False, True):
            for reference_dim in (2, 4):
                with self.subTest(device=device, batch_first=batch_first, reference_dim=reference_dim):
                    torch.manual_seed(12)
                    model = attention.MultiScaleDeformableAttention(
                        embed_dim=8, num_heads=2, num_levels=2, num_points=2,
                        batch_first=batch_first,
                    ).to(device=device, dtype=torch.float64)
                    baseline = copy.deepcopy(model)
                    self.assertEqual(set(model.state_dict()), {
                        f"{layer}.{part}" for layer in (
                            "sampling_offsets", "attention_weights", "value_proj", "output_proj"
                        ) for part in ("weight", "bias")
                    })
                    baseline.load_state_dict(model.state_dict(), strict=True)
                    qshape = (2, 3, 8) if batch_first else (3, 2, 8)
                    vshape = (2, 7, 8) if batch_first else (7, 2, 8)
                    query = torch.randn(qshape, device=device, dtype=torch.float64, requires_grad=True)
                    value = torch.randn(vshape, device=device, dtype=torch.float64, requires_grad=True)
                    refs = torch.rand(2, 3, 2, reference_dim, device=device,
                                      dtype=torch.float64, requires_grad=True)
                    kwargs = dict(
                        spatial_shapes=torch.tensor([[2, 3], [1, 1]], device=device),
                        level_start_index=torch.tensor([0, 6], device=device),
                        key_padding_mask=torch.tensor([[False] * 6 + [True]] * 2, device=device),
                    )
                    actual = model(query, value=value, reference_points=refs, **kwargs)
                    actual_grads = torch.autograd.grad(actual.square().sum(),
                                                       (query, value, refs, *model.parameters()))
                    with patch.object(attention, "ms_deform_attn", reference):
                        expected = baseline(query, value=value, reference_points=refs, **kwargs)
                        expected_grads = torch.autograd.grad(expected.square().sum(),
                                                            (query, value, refs, *baseline.parameters()))
                    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-9)
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-8)

    def test_cpu_output_and_gradients(self):
        self.compare_reference("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_output_and_gradients(self):
        self.compare_reference("cuda")

    def test_low_precision_and_autocast(self):
        for dtype, autocast, fp32_refs, reference_dim in product(
            (torch.float16, torch.bfloat16), (False, True), (False, True), (2, 4)
        ):
            with self.subTest(dtype=dtype, autocast=autocast, fp32_refs=fp32_refs,
                              reference_dim=reference_dim):
                model = attention.MultiScaleDeformableAttention(
                    embed_dim=8, num_heads=2, num_levels=1, num_points=2, batch_first=True,
                ).to(dtype=torch.float32 if autocast else dtype)
                query = torch.randn(1, 4, 8, dtype=next(model.parameters()).dtype,
                                    requires_grad=True)
                refs = torch.rand(1, 4, 1, reference_dim,
                                  dtype=torch.float32 if fp32_refs else dtype,
                                  requires_grad=True)
                with torch.autocast("cpu", dtype=dtype, enabled=autocast):
                    output = model(
                        query, reference_points=refs,
                        spatial_shapes=torch.tensor([[2, 2]]),
                        level_start_index=torch.tensor([0]),
                    )
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(torch.isfinite(output).all())
                output.float().square().mean().backward()
                self.assertTrue(torch.isfinite(query.grad).all())
                self.assertTrue(torch.isfinite(refs.grad).all())



if __name__ == "__main__":
    unittest.main()

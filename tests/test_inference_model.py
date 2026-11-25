"""
Offline unit tests for InferenceModel / build_percival checkpoint loading
and for the shipped weight-table dimension check.

No network, GPU, or real checkpoint needed: Percival is replaced by a small
stand-in with the same projection-head shapes and the real
`Percival.load_visual`. Run from the repo root:

    python -m unittest discover -s tests -v
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import extract_embeddings
from train_operations import inference_model
from train_operations.inference_model import (
    PROJECTION_DIM, PROJECTION_DIMS, InferenceModel,
)
from train_operations.percival import Percival

# ViT width per backbone size (timm vit_{tiny,small,base,large}_patch16_224).
_EMBED_DIM = {"tiny": 192, "small": 384, "base": 768, "large": 1024}


class _FakeVisual(nn.Module):
    def __init__(self, embed_dim, projection_dim):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.norm = nn.LayerNorm(embed_dim)
        self.base_model.head = nn.Linear(embed_dim, projection_dim, bias=False)


class _FakePercival(nn.Module):
    """Stand-in for Percival: same head shapes, real load_visual."""

    def __init__(self, projection_dim, vision_model_size, **_):
        super().__init__()
        self.visual = _FakeVisual(_EMBED_DIM[vision_model_size], projection_dim)

    load_visual = Percival.load_visual


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for module in (inference_model, extract_embeddings):
            patcher = mock.patch.object(module, "Percival", _FakePercival)
            patcher.start()
            self.addCleanup(patcher.stop)

    def checkpoint(self, size, projection_dim=None):
        """Write a visual-tower checkpoint as released for `size`."""
        if projection_dim is None:
            projection_dim = PROJECTION_DIMS[size]
        path = Path(self._tmp.name) / f"visual_{size}_{projection_dim}.pth"
        visual = _FakeVisual(_EMBED_DIM[size], projection_dim)
        nn.init.normal_(visual.base_model.norm.weight)   # differ from a fresh init
        torch.save(visual.state_dict(), path)
        return str(path), visual.state_dict()

    def build(self, size, **kwargs):
        path, state = self.checkpoint(size, kwargs.pop("ckpt_projection_dim", None))
        model = _quiet(InferenceModel, img_weights=path,
                       vision_model_size=size, device="cpu", **kwargs)
        return model, state


class TestProjectionDim(_Base):
    def test_matches_shipped_configs(self):
        import yaml
        self.assertEqual(PROJECTION_DIM, 768)
        self.assertEqual(PROJECTION_DIMS["base"], PROJECTION_DIM)
        for size, dim in PROJECTION_DIMS.items():
            with open(_ROOT / "configs" / f"augreg_{size}_v0.yaml") as f:
                mdl = yaml.safe_load(f)["model"]
            self.assertEqual(mdl["vision_model_size"], size)
            self.assertEqual(mdl["projection_dim"], dim)

    def test_resolved_from_model_size(self):
        for size, dim in PROJECTION_DIMS.items():
            with self.subTest(size=size):
                model, state = self.build(size)
                head = model.model.visual.base_model.head.weight
                self.assertEqual(model.projection_dim, dim)
                self.assertEqual(tuple(head.shape), (dim, _EMBED_DIM[size]))
                self.assertTrue(torch.equal(head, state["base_model.head.weight"]))

    def test_default_call_is_base(self):
        path, state = self.checkpoint("base")
        model = _quiet(InferenceModel, img_weights=path, device="cpu")
        self.assertEqual(model.projection_dim, PROJECTION_DIM)
        self.assertTrue(torch.equal(model.model.visual.base_model.head.weight,
                                    state["base_model.head.weight"]))

    def test_explicit_projection_dim_is_kept(self):
        model, state = self.build("base", projection_dim=512, ckpt_projection_dim=512)
        self.assertEqual(model.projection_dim, 512)
        self.assertTrue(torch.equal(model.model.visual.base_model.head.weight,
                                    state["base_model.head.weight"]))

    def test_unlisted_size_keeps_previous_default(self):
        model, _ = self.build("large", ckpt_projection_dim=PROJECTION_DIM)
        self.assertEqual(model.projection_dim, PROJECTION_DIM)


class TestStrictLoading(_Base):
    def test_mismatched_projection_dim_raises(self):
        with self.assertRaisesRegex(RuntimeError, "base_model.head.weight"):
            self.build("tiny", projection_dim=768)

    def test_strict_false_opt_out(self):
        model, state = self.build("tiny", projection_dim=768, strict=False)
        head = model.model.visual.base_model.head.weight
        self.assertEqual(tuple(head.shape), (768, 192))
        self.assertTrue(torch.equal(model.model.visual.base_model.norm.weight,
                                    state["base_model.norm.weight"]))

    def test_missing_tensor_raises(self):
        path, state = self.checkpoint("base")
        del state["base_model.norm.bias"]
        torch.save(state, path)
        with self.assertRaisesRegex(RuntimeError, "base_model.norm.bias"):
            _quiet(InferenceModel, img_weights=path, device="cpu")



class TestBuildPercival(_Base):
    TINY_CFG = str(_ROOT / "configs/augreg_tiny_v0.yaml")
    CPU = torch.device("cpu")

    def test_matching_checkpoint_loads(self):
        path, state = self.checkpoint("tiny")
        model = _quiet(extract_embeddings.build_percival, self.TINY_CFG, path, self.CPU)
        self.assertTrue(torch.equal(model.visual.base_model.head.weight,
                                    state["base_model.head.weight"]))

    def test_mismatched_checkpoint_raises(self):
        path, _ = self.checkpoint("tiny", projection_dim=768)
        with self.assertRaisesRegex(RuntimeError, "base_model.head.weight"):
            _quiet(extract_embeddings.build_percival, self.TINY_CFG, path, self.CPU)

    def test_strict_false_opt_out(self):
        path, state = self.checkpoint("tiny", projection_dim=768)
        model = _quiet(extract_embeddings.build_percival, self.TINY_CFG, path,
                       self.CPU, strict=False)
        self.assertEqual(tuple(model.visual.base_model.head.weight.shape), (192, 192))
        self.assertTrue(torch.equal(model.visual.base_model.norm.weight,
                                    state["base_model.norm.weight"]))


class TestShippedTables(_Base):
    def test_base_scores_with_shipped_tables(self):
        model, _ = self.build("base")
        emb = np.random.default_rng(0).standard_normal(PROJECTION_DIM)

        df, summary = _quiet(model.diagnostic_inference_all_conditions, embedding=emb)
        self.assertEqual(model._classifier_cache["CHEST"]["W"].shape[1], PROJECTION_DIM)
        self.assertEqual(model._classifier_cache["ABD_PEL"]["mean"].shape[1], PROJECTION_DIM)
        self.assertEqual(summary["n_codes_scored"], len(df))
        self.assertGreater(len(df), 0)
        self.assertTrue(df["prob"].between(0, 1).all())

        df, summary = _quiet(model.prognostic_inference_all_conditions, embedding=emb)
        self.assertEqual(model._cox_cache["CHEST"]["W"].shape[1], PROJECTION_DIM)
        self.assertEqual(summary["n_codes_scored"], len(df))
        self.assertGreater(len(df), 0)

    def test_non_base_embeddings_are_rejected(self):
        for size in ("tiny", "small"):
            model, _ = self.build(size)
            emb = np.zeros(PROJECTION_DIMS[size])
            for method in (model.diagnostic_inference,
                           model.diagnostic_inference_all_conditions,
                           model.prognostic_inference,
                           model.prognostic_inference_all_conditions):
                with self.subTest(size=size, method=method.__name__):
                    with self.assertRaisesRegex(ValueError, "has 768 'w_\\*' columns"):
                        _quiet(method, embedding=emb)
            self.assertEqual(model._classifier_cache, {})
            self.assertEqual(model._cox_cache, {})


if __name__ == "__main__":
    unittest.main()

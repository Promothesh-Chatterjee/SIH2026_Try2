"""Phase 15: model loading in ONNX export is FAIL-FAST.

An export must never silently continue with random-initialized weights. A
missing, corrupted, non-tensor, or architecture/state-dict-mismatched
checkpoint raises instead of producing an .onnx file.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from src.deployment.export_onnx import export_deinterleaver, export_scheduler

CONFIG_PATH = Path("configs/model_config.yaml")


def _fake_onnx_export(model, dummy, output_path, **kwargs):
    # Real export needs the optional `onnxscript` package; stub the write so
    # happy-path tests still prove the checkpoint is loaded strictly (no
    # random-init fallback) and the export continues past loading.
    with open(output_path, "wb") as f:
        f.write(b"onnx-stub")
    return None


def _deint_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["deinterleaver"]


def _sched_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["drqn_scheduler"]


def _make_deint(d_model=None):
    from src.models.deinterleaver import PDWTransformerEncoder
    cfg = _deint_cfg()
    return PDWTransformerEncoder(
        pdw_dim=cfg["pdw_dim"],
        d_model=d_model if d_model is not None else cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        embed_dim=cfg["embed_dim"],
    )


def _make_scheduler():
    from src.models.drqn_scheduler import DRQNScheduler
    cfg = _sched_cfg()
    return DRQNScheduler(
        obs_dim=cfg["obs_dim"],
        n_bands=cfg["n_bands"],
        n_actions=cfg["n_actions"],
        n_modes=cfg["n_modes"],
        lstm_hidden=cfg["lstm_hidden"],
        lstm_layers=cfg["lstm_layers"],
    )


def _save(ckpt: Path, model, prefixed=False):
    state = model.state_dict()
    if prefixed:
        state = {"oops." + k: v for k, v in state.items()}
    torch.save({"state_dict": state, "metadata": {"arch": "test", "n_bands": 36}}, ckpt)


class DeinterleaverFailFastTests(unittest.TestCase):
    def test_missing_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.onnx"
            with self.assertRaises(FileNotFoundError):
                export_deinterleaver(str(CONFIG_PATH), str(Path(td) / "nope.pt"), str(out))
            self.assertFalse(out.exists())

    def test_corrupted_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "bad.pt"
            ckpt.write_bytes(b"this is not a torch checkpoint \x00\x01")
            out = Path(td) / "out.onnx"
            with self.assertRaises(RuntimeError):
                export_deinterleaver(str(CONFIG_PATH), str(ckpt), str(out))
            self.assertFalse(out.exists())

    def test_non_state_dict_payload_raises(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "wrong.pt"
            torch.save(torch.randn(8), ckpt)  # a bare tensor, not a state dict
            out = Path(td) / "out.onnx"
            with self.assertRaises(RuntimeError):
                export_deinterleaver(str(CONFIG_PATH), str(ckpt), str(out))
            self.assertFalse(out.exists())

    def test_architecture_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "mismatched.pt"
            _save(ckpt, _make_deint(d_model=_deint_cfg()["d_model"] + 8))
            out = Path(td) / "out.onnx"
            with self.assertRaises(RuntimeError) as ctx:
                export_deinterleaver(str(CONFIG_PATH), str(ckpt), str(out))
            self.assertIn("State dict mismatch", str(ctx.exception))
            self.assertFalse(out.exists())

    def test_renamed_keys_state_dict_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "prefix.pt"
            _save(ckpt, _make_deint(), prefixed=True)
            out = Path(td) / "out.onnx"
            with self.assertRaises(RuntimeError):
                export_deinterleaver(str(CONFIG_PATH), str(ckpt), str(out))
            self.assertFalse(out.exists())

    def test_valid_checkpoint_exports(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "good.pt"
            _save(ckpt, _make_deint())
            out = Path(td) / "out.onnx"
            with mock.patch("torch.onnx.export", side_effect=_fake_onnx_export):
                exported = export_deinterleaver(str(CONFIG_PATH), str(ckpt), str(out))
            self.assertTrue(exported.exists())
            self.assertGreater(exported.stat().st_size, 0)


class SchedulerFailFastTests(unittest.TestCase):
    def test_missing_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.onnx"
            with self.assertRaises(FileNotFoundError):
                export_scheduler(str(CONFIG_PATH), str(Path(td) / "nope.pt"), str(out))
            self.assertFalse(out.exists())

    def test_corrupted_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "bad.pt"
            ckpt.write_bytes(b"corrupt checkpoint bytes!")
            out = Path(td) / "out.onnx"
            with self.assertRaises(RuntimeError):
                export_scheduler(str(CONFIG_PATH), str(ckpt), str(out))
            self.assertFalse(out.exists())

    def test_architecture_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            from src.models.drqn_scheduler import DRQNScheduler
            cfg = _sched_cfg()
            bad = DRQNScheduler(
                obs_dim=cfg["obs_dim"],
                n_bands=cfg["n_bands"],
                n_actions=cfg["n_actions"],
                n_modes=cfg["n_modes"],
                lstm_hidden=cfg["lstm_hidden"] + 7,  # different LSTM hidden size
                lstm_layers=cfg["lstm_layers"],
            )
            ckpt = Path(td) / "mismatched.pt"
            _save(ckpt, bad)
            out = Path(td) / "out.onnx"
            with self.assertRaises(RuntimeError):
                export_scheduler(str(CONFIG_PATH), str(ckpt), str(out))
            self.assertFalse(out.exists())

    def test_valid_checkpoint_exports(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "good.pt"
            _save(ckpt, _make_scheduler())
            out = Path(td) / "out.onnx"
            with mock.patch("torch.onnx.export", side_effect=_fake_onnx_export):
                exported = export_scheduler(str(CONFIG_PATH), str(ckpt), str(out))
            self.assertTrue(exported.exists())
            self.assertGreater(exported.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
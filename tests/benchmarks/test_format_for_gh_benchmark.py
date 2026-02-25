# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


def _load_format_module():
    comparative_dir = Path(__file__).resolve().parent / "comparative"
    if str(comparative_dir) not in sys.path:
        sys.path.insert(0, str(comparative_dir))
    module_path = comparative_dir / "format_for_gh_benchmark.py"
    spec = importlib.util.spec_from_file_location("format_for_gh_benchmark", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _load_format_module()
build_entries = MOD.build_entries
QUALITY_METRICS = MOD.QUALITY_METRICS
PERFORMANCE_METRICS = MOD.PERFORMANCE_METRICS


SAMPLE_PER_SCENE = {
    "bonsai": {
        "fvdb_default": {
            "PSNR": 27.2845,
            "SSIM": 0.89391234,
            "training_time": 188.5366,
            "peak_gpu_memory_gb": 2.07219,
        },
        "fvdb_mcmc": {
            "PSNR": 25.1234,
            "SSIM": 0.87970001,
            "training_time": 201.123,
            "peak_gpu_memory_gb": 3.456,
        },
    },
    "garden": {
        "fvdb_default": {
            "PSNR": 24.5,
            "SSIM": 0.8,
            "training_time": 300.0,
            "peak_gpu_memory_gb": 4.0,
        },
    },
}


class TestBuildEntries:
    def test_normal_quality(self):
        entries = build_entries(SAMPLE_PER_SCENE, QUALITY_METRICS, ("fvdb_default", "fvdb_mcmc"))
        names = [e["name"] for e in entries]
        assert "bonsai/fvdb_default - PSNR" in names
        assert "bonsai/fvdb_default - SSIM" in names
        assert "bonsai/fvdb_mcmc - PSNR" in names
        assert "garden/fvdb_default - PSNR" in names

    def test_normal_performance(self):
        entries = build_entries(SAMPLE_PER_SCENE, PERFORMANCE_METRICS, ("fvdb_default",))
        names = [e["name"] for e in entries]
        assert "bonsai/fvdb_default - training_time" in names
        assert "bonsai/fvdb_default - peak_gpu_memory_gb" in names
        assert "garden/fvdb_default - training_time" in names

    def test_units(self):
        entries = build_entries(SAMPLE_PER_SCENE, QUALITY_METRICS, ("fvdb_default",))
        by_name = {e["name"]: e for e in entries}
        assert by_name["bonsai/fvdb_default - PSNR"]["unit"] == "dB"
        assert by_name["bonsai/fvdb_default - SSIM"]["unit"] == ""

    def test_rounding(self):
        entries = build_entries(SAMPLE_PER_SCENE, QUALITY_METRICS, ("fvdb_default",))
        by_name = {e["name"]: e for e in entries}
        assert by_name["bonsai/fvdb_default - PSNR"]["value"] == 27.2845
        assert by_name["bonsai/fvdb_default - SSIM"]["value"] == 0.8939

    def test_scenes_sorted(self):
        entries = build_entries(SAMPLE_PER_SCENE, QUALITY_METRICS, ("fvdb_default",))
        scene_order = []
        for e in entries:
            scene = e["name"].split("/")[0]
            if not scene_order or scene_order[-1] != scene:
                scene_order.append(scene)
        assert scene_order == sorted(scene_order)

    def test_missing_metric(self):
        per_scene = {
            "bonsai": {
                "fvdb_default": {"PSNR": 27.0},
            },
        }
        entries = build_entries(per_scene, QUALITY_METRICS, ("fvdb_default",))
        names = [e["name"] for e in entries]
        assert "bonsai/fvdb_default - PSNR" in names
        assert "bonsai/fvdb_default - SSIM" not in names

    def test_nan_value_filtered(self):
        per_scene = {
            "bonsai": {
                "fvdb_default": {"PSNR": float("nan"), "SSIM": 0.89},
            },
        }
        entries = build_entries(per_scene, QUALITY_METRICS, ("fvdb_default",))
        names = [e["name"] for e in entries]
        assert "bonsai/fvdb_default - PSNR" not in names
        assert "bonsai/fvdb_default - SSIM" in names

    def test_empty_per_scene(self):
        entries = build_entries({}, QUALITY_METRICS, ("fvdb_default",))
        assert entries == []

    def test_config_filtering(self):
        entries = build_entries(SAMPLE_PER_SCENE, QUALITY_METRICS, ("fvdb_mcmc",))
        names = [e["name"] for e in entries]
        assert all("fvdb_mcmc" in n for n in names)
        assert not any("fvdb_default" in n for n in names)

    def test_config_not_in_scene(self):
        entries = build_entries(SAMPLE_PER_SCENE, QUALITY_METRICS, ("nonexistent_config",))
        assert entries == []


class TestMainCLI:
    def _write_summary(self, tmp_path: Path, per_scene: dict) -> Path:
        summary = {"per_scene": per_scene}
        p = tmp_path / "summary_data.json"
        p.write_text(json.dumps(summary))
        return p

    def test_normal_run(self, tmp_path, monkeypatch):
        summary_path = self._write_summary(tmp_path, SAMPLE_PER_SCENE)
        out_dir = tmp_path / "output"
        monkeypatch.setattr("sys.argv", ["prog", str(summary_path), "--output-dir", str(out_dir)])
        ret = MOD.main()
        assert ret == 0
        quality_path = out_dir / "benchmark_quality.json"
        perf_path = out_dir / "benchmark_performance.json"
        assert quality_path.exists()
        assert perf_path.exists()

        quality = json.loads(quality_path.read_text())
        assert isinstance(quality, list)
        assert len(quality) > 0
        assert all({"name", "unit", "value"} <= set(e.keys()) for e in quality)

        perf = json.loads(perf_path.read_text())
        assert isinstance(perf, list)
        assert len(perf) > 0

    def test_empty_per_scene_returns_error(self, tmp_path, monkeypatch):
        summary_path = self._write_summary(tmp_path, {})
        out_dir = tmp_path / "output"
        monkeypatch.setattr("sys.argv", ["prog", str(summary_path), "--output-dir", str(out_dir)])
        ret = MOD.main()
        assert ret == 1

    def test_custom_configs(self, tmp_path, monkeypatch):
        summary_path = self._write_summary(tmp_path, SAMPLE_PER_SCENE)
        out_dir = tmp_path / "output"
        monkeypatch.setattr(
            "sys.argv", ["prog", str(summary_path), "--output-dir", str(out_dir), "--configs", "fvdb_mcmc"]
        )
        ret = MOD.main()
        assert ret == 0
        quality = json.loads((out_dir / "benchmark_quality.json").read_text())
        assert all("fvdb_mcmc" in e["name"] for e in quality)

    def test_output_defaults_to_summary_dir(self, tmp_path, monkeypatch):
        summary_path = self._write_summary(tmp_path, SAMPLE_PER_SCENE)
        monkeypatch.setattr("sys.argv", ["prog", str(summary_path)])
        ret = MOD.main()
        assert ret == 0
        assert (tmp_path / "benchmark_quality.json").exists()
        assert (tmp_path / "benchmark_performance.json").exists()

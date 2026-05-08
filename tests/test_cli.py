"""Tests for the Mneme CLI."""

import numpy as np
import pytest
from click.testing import CliRunner

from mneme.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_info(runner):
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0


def test_generate_gaussian_blob(runner, tmp_path):
    out = tmp_path / "blob.npz"
    result = runner.invoke(
        cli,
        ["generate", "-o", str(out), "-t", "gaussian_blob", "-s", "16,16", "--seed", "1"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    loaded = np.load(out, allow_pickle=True)
    assert "data" in loaded.files
    assert loaded["data"].shape == (16, 16)


def test_generate_sinusoidal(runner, tmp_path):
    out = tmp_path / "sin.npz"
    result = runner.invoke(
        cli,
        ["generate", "-o", str(out), "-t", "sinusoidal", "-s", "16,16", "--seed", "2"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_generate_bioelectric(runner, tmp_path):
    out = tmp_path / "bio.npz"
    result = runner.invoke(
        cli,
        [
            "generate",
            "-o",
            str(out),
            "-t",
            "bioelectric",
            "-s",
            "16,16",
            "--timesteps",
            "5",
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    loaded = np.load(out, allow_pickle=True)
    assert loaded["data"].shape == (5, 16, 16)


def test_generate_invalid_shape(runner, tmp_path):
    out = tmp_path / "bad.npz"
    result = runner.invoke(
        cli,
        ["generate", "-o", str(out), "-t", "gaussian_blob", "-s", "abc", "--seed", "1"],
    )
    # Should not crash; prints error and returns
    assert result.exit_code == 0
    assert "Invalid shape" in result.output


def test_analyze_standard_pipeline(runner, tmp_path):
    # First, generate small dataset
    data_file = tmp_path / "data.npz"
    res1 = runner.invoke(
        cli,
        ["generate", "-o", str(data_file), "-t", "gaussian_blob", "-s", "16,16", "--seed", "1"],
    )
    assert res1.exit_code == 0

    out_dir = tmp_path / "results"
    res2 = runner.invoke(
        cli,
        [
            "analyze",
            str(data_file),
            "-o",
            str(out_dir),
            "-p",
            "standard",
            "-f",
            "pickle",
            "--attractor-method",
            "none",
        ],
    )
    assert res2.exit_code == 0, res2.output
    files = list(out_dir.glob("analysis_results.*"))
    assert len(files) >= 1


def test_analyze_unsupported_format(runner, tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("hi")
    out = tmp_path / "out"
    res = runner.invoke(cli, ["analyze", str(bad), "-o", str(out)])
    assert res.exit_code == 0
    assert "Unsupported" in res.output


def test_list_experiments_empty(runner, tmp_path):
    res = runner.invoke(cli, ["list-experiments", "-b", str(tmp_path)])
    # Either prints "No experiments" or exits cleanly
    assert res.exit_code == 0

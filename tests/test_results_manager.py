"""Tests for analysis.results.ResultManager and report functions."""

import json

import numpy as np
import pytest

from mneme.analysis.results import (
    ResultManager,
    create_results_report,
)
from mneme.types import (
    AnalysisResult,
    Attractor,
    AttractorType,
    Field,
    PersistenceDiagram,
    ReconstructionMethod,
    ReconstructionResult,
    TopologyResult,
)


def _result(eid="exp_a"):
    raw = Field(data=np.random.RandomState(0).randn(8, 8))
    recon = ReconstructionResult(
        field=Field(data=np.zeros((8, 8))),
        method=ReconstructionMethod.GAUSSIAN_PROCESS,
        computation_time=0.1,
    )
    diag = PersistenceDiagram(points=np.array([[0.0, 1.0], [0.2, 0.7]]), dimension=0)
    topo = TopologyResult(diagrams=[diag])
    attractor = Attractor(
        type=AttractorType.FIXED_POINT,
        center=np.array([0.5, 0.5]),
        basin_size=1.0,
        dimension=0.0,
    )
    return AnalysisResult(
        experiment_id=eid,
        timestamp="2026-01-01T00:00:00",
        raw_data=raw,
        reconstruction=recon,
        topology=topo,
        attractors=[attractor],
    )


def test_manager_init_creates_subdirs(tmp_path):
    rm = ResultManager(tmp_path / "rm")
    assert (rm.base_dir / "experiments").is_dir()
    assert (rm.base_dir / "summaries").is_dir()
    assert (rm.base_dir / "exports").is_dir()


def test_save_and_load_result(tmp_path):
    rm = ResultManager(tmp_path)
    r = _result("exp_1")
    p = rm.save_result(r)
    assert p.exists()
    summary = json.loads((p.parent / "summary.json").read_text())
    assert summary["experiment_id"] == "exp_1"
    assert "reconstruction" in summary["analysis_summary"]
    loaded = rm.load_result("exp_1")
    assert loaded.experiment_id == "exp_1"


def test_save_pickle_format(tmp_path):
    rm = ResultManager(tmp_path)
    p = rm.save_result(_result("p1"), format="pickle")
    assert p.suffix == ".pickle"
    loaded = rm.load_result("p1", format="pickle")
    assert loaded.experiment_id == "p1"


def test_load_missing_raises(tmp_path):
    rm = ResultManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        rm.load_result("nope")


def test_list_experiments(tmp_path):
    rm = ResultManager(tmp_path)
    rm.save_result(_result("a"))
    rm.save_result(_result("b"))
    assert sorted(rm.list_experiments()) == ["a", "b"]


def test_comparative_summary(tmp_path):
    rm = ResultManager(tmp_path)
    rm.save_result(_result("a"))
    rm.save_result(_result("b"))
    summary = rm.create_comparative_summary(["a", "b", "missing"])
    assert summary["experiments"] == ["a", "b", "missing"]
    assert len(summary["comparison"]["reconstruction_methods"]) == 2
    assert len(summary["comparison"]["attractor_counts"]) == 2


def test_export_to_csv(tmp_path):
    rm = ResultManager(tmp_path)
    rm.save_result(_result("e"))
    out = tmp_path / "x.csv"
    rm.export_to_csv("e", out)
    assert out.exists()
    text = out.read_text()
    assert "topology" in text or "attractor" in text


def test_export_to_matlab(tmp_path):
    rm = ResultManager(tmp_path)
    rm.save_result(_result("m"))
    out = tmp_path / "x.mat"
    rm.export_to_matlab("m", out)
    assert out.exists()


def test_create_html_report(tmp_path):
    out = tmp_path / "report.html"
    create_results_report([_result("r1"), _result("r2")], out, format="html")
    content = out.read_text()
    assert "Mneme Analysis Report" in content
    assert "Total experiments: 2" in content


def test_create_markdown_report(tmp_path):
    out = tmp_path / "report.md"
    create_results_report([_result("r1")], out, format="markdown")
    assert "# Mneme Analysis Report" in out.read_text()


def test_create_report_unsupported(tmp_path):
    with pytest.raises(ValueError):
        create_results_report([_result()], tmp_path / "x.pdf", format="pdf")

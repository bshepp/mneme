"""Tests for I/O utilities."""

import json
import numpy as np
import pytest

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
from mneme.utils.io import (
    create_experiment_directory,
    list_experiments,
    load_experiment_config,
    load_results,
    save_experiment_config,
    save_results,
)


def _make_analysis_result():
    raw = Field(data=np.random.RandomState(0).randn(8, 8))
    recon = ReconstructionResult(
        field=Field(data=np.random.RandomState(1).randn(8, 8)),
        uncertainty=np.ones((8, 8)) * 0.1,
        method=ReconstructionMethod.GAUSSIAN_PROCESS,
        computation_time=0.42,
    )
    diag = PersistenceDiagram(
        points=np.array([[0.0, 1.0], [0.2, 1.5]]), dimension=0, threshold=0.1
    )
    topo = TopologyResult(diagrams=[diag])
    attractor = Attractor(
        type=AttractorType.FIXED_POINT,
        center=np.array([0.0, 0.0]),
        basin_size=1.0,
        dimension=0.0,
    )
    return AnalysisResult(
        experiment_id="exp-1",
        timestamp="2026-01-01T00:00:00",
        raw_data=raw,
        reconstruction=recon,
        topology=topo,
        attractors=[attractor],
        metadata={"note": "test", "k": 3},
    )


def test_save_load_hdf5_roundtrip(tmp_path):
    result = _make_analysis_result()
    out = tmp_path / "r.h5"
    save_results(result, out, format="hdf5")
    assert out.exists()
    loaded = load_results(out)
    assert loaded.experiment_id == "exp-1"
    assert loaded.raw_data.shape == (8, 8)
    assert loaded.reconstruction is not None
    assert loaded.reconstruction.method == ReconstructionMethod.GAUSSIAN_PROCESS
    assert loaded.topology is not None
    assert len(loaded.topology.diagrams) == 1
    assert loaded.attractors and loaded.attractors[0].type == AttractorType.FIXED_POINT


def test_save_load_pickle_roundtrip(tmp_path):
    result = _make_analysis_result()
    out = tmp_path / "r.pkl"
    save_results(result, out, format="pickle")
    loaded = load_results(out)
    assert loaded.experiment_id == "exp-1"


def test_save_json_summary(tmp_path):
    result = _make_analysis_result()
    out = tmp_path / "r.json"
    save_results(result, out, format="json")
    data = json.loads(out.read_text())
    assert data["experiment_id"] == "exp-1"
    assert "topology_summary" in data
    assert "attractors_summary" in data


def test_save_dict_hdf5(tmp_path):
    out = tmp_path / "d.h5"
    save_results({"a": np.ones((4, 4)), "b": "hello", "c": 1.5}, out, format="hdf5")
    assert out.exists()


def test_save_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        save_results({}, tmp_path / "x.foo", format="foo")


def test_load_unknown_extension(tmp_path):
    p = tmp_path / "x.unknown"
    p.write_text("nope")
    with pytest.raises(ValueError):
        load_results(p)


def test_create_experiment_directory(tmp_path):
    exp_dir = create_experiment_directory(tmp_path, "myexp")
    assert exp_dir.exists()
    for sub in ["data", "results", "plots", "logs", "configs"]:
        assert (exp_dir / sub).is_dir()


def test_create_experiment_no_subdirs(tmp_path):
    exp_dir = create_experiment_directory(tmp_path, "bare", create_subdirs=False)
    assert exp_dir.exists()
    assert not (exp_dir / "data").exists()


def test_save_load_experiment_config(tmp_path):
    exp_dir = create_experiment_directory(tmp_path, "exp")
    cfg = {"a": 1, "b": {"c": 2}}
    save_experiment_config(cfg, exp_dir)
    loaded = load_experiment_config(exp_dir)
    assert loaded == cfg


def test_load_missing_config(tmp_path):
    exp_dir = create_experiment_directory(tmp_path, "exp")
    with pytest.raises(FileNotFoundError):
        load_experiment_config(exp_dir, filename="missing.yaml")


def test_list_experiments(tmp_path):
    create_experiment_directory(tmp_path, "a")
    create_experiment_directory(tmp_path, "b")
    result = list_experiments(tmp_path)
    assert sorted(result) == ["a", "b"]


def test_list_experiments_missing_dir(tmp_path):
    assert list_experiments(tmp_path / "does_not_exist") == []

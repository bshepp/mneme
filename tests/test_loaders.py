"""Tests for data.loaders."""

import h5py
import numpy as np
import pandas as pd
import pytest

from mneme.data.loaders import (
    BioelectricDataLoader,
    CSVDataLoader,
    HDF5DataLoader,
    NumpyDataLoader,
    SyntheticDataLoader,
    create_data_loader,
    load_bioelectric_measurements,
    save_field_data,
)
from mneme.types import Field


def _write_h5(path, data_key="data", extra=None, attrs=None):
    with h5py.File(path, "w") as f:
        f.create_dataset(data_key, data=np.ones((4, 4)))
        if extra:
            for k, v in extra.items():
                f.create_dataset(k, data=v)
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v


def test_hdf5_loader(tmp_path):
    _write_h5(tmp_path / "a.h5", attrs={"name": "alpha"})
    _write_h5(tmp_path / "b.h5")
    loader = HDF5DataLoader(tmp_path)
    assert len(loader) == 2
    field = loader.load_file(loader.file_list[0])
    assert isinstance(field, Field)
    assert field.data.shape == (4, 4)
    info = loader.get_info()
    assert info.n_samples == 2
    assert info.shape == (4, 4)
    assert "float" in info.dtype


def test_hdf5_loader_iterates(tmp_path):
    _write_h5(tmp_path / "a.h5")
    loader = HDF5DataLoader(tmp_path)
    items = list(loader)
    assert len(items) == 1


def test_hdf5_missing_key_raises(tmp_path):
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f.create_dataset("foo", data=np.ones(3))
        f.create_dataset("bar", data=np.ones(3))
    loader = HDF5DataLoader(tmp_path, data_key="not_present")
    with pytest.raises(RuntimeError):
        loader.load_file(loader.file_list[0])


def test_loader_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        HDF5DataLoader(tmp_path / "missing")


def test_bioelectric_loader(tmp_path):
    p = tmp_path / "v.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("voltage_field", data=np.zeros((3, 4, 4)))
        f.create_dataset("timestamps", data=np.arange(3))
    loader = BioelectricDataLoader(tmp_path)
    sample = loader.load_file(p)
    assert sample["voltage_field"].shape == (3, 4, 4)
    assert sample["timestamps"].shape == (3,)


def test_bioelectric_missing_voltage(tmp_path):
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("other", data=np.zeros(3))
    loader = BioelectricDataLoader(tmp_path)
    with pytest.raises(RuntimeError):
        loader.load_file(p)


def test_csv_loader(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(tmp_path / "x.csv", index=False)
    loader = CSVDataLoader(tmp_path)
    arr = loader.load_file(loader.file_list[0])
    assert arr.shape == (2, 2)


def test_numpy_loader_npy_and_npz(tmp_path):
    np.save(tmp_path / "a.npy", np.ones(5))
    loader = NumpyDataLoader(tmp_path, file_pattern="*.npy")
    arr = loader.load_file(loader.file_list[0])
    assert arr.shape == (5,)

    npz = tmp_path / "z.npz"
    np.savez(npz, x=np.ones(3))
    loader2 = NumpyDataLoader(tmp_path, file_pattern="*.npz")
    arr = loader2.load_file(loader2.file_list[0])
    assert arr.shape == (3,)


def test_synthetic_loader(tmp_path):
    p = tmp_path / "s.npz"
    np.savez(p, gaussian=np.ones((4, 4)), sinusoidal=np.zeros((4, 4)))
    loader = SyntheticDataLoader(p)
    types = loader.get_field_types()
    assert "gaussian" in types
    assert loader.get_field("gaussian").shape == (4, 4)
    with pytest.raises(KeyError):
        loader.get_field("nope")


def test_synthetic_loader_unsupported(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("hi")
    with pytest.raises(ValueError):
        SyntheticDataLoader(p)


def test_synthetic_loader_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        SyntheticDataLoader(tmp_path / "nope.npz")


def test_load_bioelectric_measurements(tmp_path):
    p = tmp_path / "m.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("voltage", data=np.array([0.1, 0.2, 0.3]))
        f.create_dataset("positions", data=np.array([[0, 0], [1, 0], [2, 0]]))
        f.create_dataset("timestamps", data=np.array([0.0, 1.0, 2.0]))
    measurements = load_bioelectric_measurements(p)
    assert len(measurements) == 3
    assert measurements[0].voltage == pytest.approx(0.1)


def test_load_bioelectric_unsupported(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("x")
    with pytest.raises(ValueError):
        load_bioelectric_measurements(p)


def test_create_data_loader_auto_numpy(tmp_path):
    np.save(tmp_path / "a.npy", np.ones(2))
    loader = create_data_loader(tmp_path)
    assert isinstance(loader, NumpyDataLoader)


def test_create_data_loader_auto_csv(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "x.csv", index=False)
    loader = create_data_loader(tmp_path)
    assert isinstance(loader, CSVDataLoader)


def test_create_data_loader_auto_hdf5(tmp_path):
    _write_h5(tmp_path / "x.h5")
    loader = create_data_loader(tmp_path)
    assert isinstance(loader, HDF5DataLoader)


def test_create_data_loader_auto_bioelectric(tmp_path):
    with h5py.File(tmp_path / "b.h5", "w") as f:
        f.create_dataset("voltage_field", data=np.ones((2, 4, 4)))
    loader = create_data_loader(tmp_path)
    assert isinstance(loader, BioelectricDataLoader)


def test_create_data_loader_auto_unknown(tmp_path):
    (tmp_path / "x.unknown").write_text("hi")
    with pytest.raises(ValueError):
        create_data_loader(tmp_path)


def test_create_data_loader_invalid_type(tmp_path):
    np.save(tmp_path / "a.npy", np.ones(2))
    with pytest.raises(ValueError):
        create_data_loader(tmp_path, loader_type="bogus")


def test_save_field_data_hdf5(tmp_path):
    field = Field(data=np.ones((3, 3)), metadata={"src": "test"})
    out = tmp_path / "field.h5"
    save_field_data(field, out, format="hdf5")
    with h5py.File(out, "r") as f:
        assert f["data"].shape == (3, 3)
        assert f.attrs["src"] == "test"


def test_save_field_data_numpy(tmp_path):
    field = Field(data=np.ones((3, 3)))
    out = tmp_path / "field.npy"
    save_field_data(field, out, format="numpy")
    assert out.exists() or (out.with_suffix(".npy")).exists()

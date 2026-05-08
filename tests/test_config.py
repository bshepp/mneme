"""Tests for utils.config."""

import json
import os

import pytest
import yaml

from mneme.utils.config import (
    Config,
    MNEME_CONFIG_SCHEMA,
    load_default_config,
    merge_configs,
    validate_config,
)


def test_get_set_dot_notation():
    cfg = Config({"a": {"b": {"c": 7}}})
    assert cfg.get("a.b.c") == 7
    assert cfg.get("missing.path", default=42) == 42
    cfg.set("a.b.d", 9)
    assert cfg.get("a.b.d") == 9
    cfg.set("new.path.x", 1)
    assert cfg.get("new.path.x") == 1


def test_bracket_and_contains():
    cfg = Config({"k": 1})
    assert cfg["k"] == 1
    cfg["new"] = 2
    assert cfg["new"] == 2
    assert "k" in cfg
    assert "missing" not in cfg


def test_yaml_roundtrip(tmp_path):
    cfg = Config({"a": 1, "b": {"c": "hi"}})
    p = tmp_path / "c.yaml"
    cfg.save(p, format="yaml")
    loaded = Config.from_yaml(p)
    assert loaded.to_dict() == cfg.to_dict()


def test_json_roundtrip(tmp_path):
    cfg = Config({"a": [1, 2, 3], "b": True})
    p = tmp_path / "c.json"
    cfg.save(p, format="json")
    loaded = Config.from_json(p)
    assert loaded.to_dict() == cfg.to_dict()


def test_save_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        Config({}).save(tmp_path / "x.foo", format="foo")


def test_yaml_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(tmp_path / "no.yaml")


def test_json_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_json(tmp_path / "no.json")


def test_from_env(monkeypatch):
    monkeypatch.setenv("MNEMETEST_FOO", "1")
    monkeypatch.setenv("MNEMETEST_BAR", '"hello"')
    cfg = Config.from_env(prefix="MNEMETEST_")
    assert cfg.get("foo") == 1
    assert cfg.get("bar") == "hello"


def test_update_and_deep_merge():
    a = Config({"x": {"y": 1, "z": 2}})
    a.update({"x": {"z": 99, "w": 3}, "new": True})
    assert a.get("x.y") == 1
    assert a.get("x.z") == 99
    assert a.get("x.w") == 3
    assert a.get("new") is True


def test_update_with_config_instance():
    a = Config({"x": 1})
    b = Config({"y": 2})
    a.update(b)
    assert a.get("y") == 2


def test_merge_configs():
    merged = merge_configs({"a": 1}, Config({"b": 2}), {"a": 10})
    assert merged.get("a") == 10
    assert merged.get("b") == 2


def test_keys_and_repr():
    cfg = Config({"a": 1, "b": 2})
    assert sorted(cfg.keys()) == ["a", "b"]
    assert "Config" in repr(cfg)


def test_validate_config_valid():
    cfg = Config(
        {
            "project": {"name": "p", "version": "0.1", "random_seed": 0},
            "data": {"raw_path": "/x", "processed_path": "/y"},
        }
    )
    ok, errors = validate_config(cfg, MNEME_CONFIG_SCHEMA)
    assert ok, errors


def test_validate_config_invalid():
    cfg = Config({"project": {}})  # missing required keys
    ok, errors = validate_config(cfg, MNEME_CONFIG_SCHEMA)
    assert not ok
    assert any("name" in e for e in errors)


def test_load_default_config_returns_config():
    cfg = load_default_config()
    assert isinstance(cfg, Config)

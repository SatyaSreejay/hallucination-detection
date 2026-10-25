"""The .env loader copes with files written on Windows."""

from __future__ import annotations

import os

import pytest

from src.config import _load_dotenv


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_dotenv_encodings(tmp_path, monkeypatch, encoding):
    monkeypatch.delenv("DOTENV_T_KEY", raising=False)
    monkeypatch.delenv("DOTENV_T_MODEL", raising=False)
    f = tmp_path / ".env"
    f.write_text("DOTENV_T_KEY=abc123\r\nDOTENV_T_MODEL=   # provider default\r\n", encoding=encoding)
    _load_dotenv(f)
    assert os.environ["DOTENV_T_KEY"] == "abc123"
    assert os.environ["DOTENV_T_MODEL"] == ""


def test_dotenv_empty_env_var_does_not_block(tmp_path, monkeypatch):
    monkeypatch.setenv("DOTENV_T_KEY", "")
    f = tmp_path / ".env"
    f.write_text('export DOTENV_T_KEY="quoted value"\n', encoding="utf-8")
    _load_dotenv(f)
    assert os.environ["DOTENV_T_KEY"] == "quoted value"


def test_dotenv_existing_value_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("DOTENV_T_KEY", "from-shell")
    f = tmp_path / ".env"
    f.write_text("DOTENV_T_KEY=from-file\n", encoding="utf-8")
    _load_dotenv(f)
    assert os.environ["DOTENV_T_KEY"] == "from-shell"

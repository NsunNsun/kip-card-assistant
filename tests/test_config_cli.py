import json

import pytest

from kip_card import cli
from kip_card.config import ConfigError, load_settings

from .conftest import EXAMPLES, FakeLLM, make_copy


def test_cloud_without_key_is_config_error():
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        load_settings()


def test_placeholder_key_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-or-...")
    with pytest.raises(ConfigError):
        load_settings()


def test_local_server_needs_no_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    settings = load_settings()
    assert settings.is_local and settings.api_key == "local"


def test_bad_values(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "xml")
    with pytest.raises(ConfigError, match="LLM_RESPONSE_FORMAT"):
        load_settings()
    monkeypatch.setenv("LLM_RESPONSE_FORMAT", "auto")
    monkeypatch.setenv("LLM_TEMPERATURE", "горячо")
    with pytest.raises(ConfigError, match="LLM_TEMPERATURE"):
        load_settings()


def test_cli_check_exit_codes(capsys):
    assert cli.main(["check", str(EXAMPLES / "testo-830-t1.json")]) == cli.EXIT_OK
    assert cli.main(["check", str(EXAMPLES.parent / "edge-cases" / "broken-syntax.json")]) == cli.EXIT_INPUT
    assert "Некорректный JSON" in capsys.readouterr().err


def test_expand_paths_like_powershell():
    # PowerShell передаёт '*.json' программе как есть — раскрываем сами
    files, unmatched = cli.expand_paths([str(EXAMPLES / "*.json"), "-", str(EXAMPLES / "nothing-*.json")])
    names = [p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for p in files]
    assert names == ["manometer-dm02-from-task.json", "owen-pd100-di1.6-111-0.5.json", "testo-830-t1.json", "-"]
    assert unmatched == [str(EXAMPLES / "nothing-*.json")]


def test_cli_check_with_glob(capsys):
    assert cli.main(["check", str(EXAMPLES / "*.json")]) == cli.EXIT_OK
    assert cli.main(["check", str(EXAMPLES / "nothing-*.json")]) == cli.EXIT_INPUT
    assert "не найдено ни одного файла" in capsys.readouterr().err


def test_cli_generate_with_glob(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(cli, "OpenAICompatibleLLM", lambda settings: FakeLLM(make_copy()))
    assert cli.main(["generate", str(EXAMPLES / "*.json"), "-o", str(tmp_path), "-f", "json"]) == cli.EXIT_OK
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "manometer-dm02-from-task.json", "owen-pd100-di1.6-111-0.5.json", "testo-830-t1.json"]


def test_cli_generate_without_key():
    assert cli.main(["generate", str(EXAMPLES / "testo-830-t1.json")]) == cli.EXIT_LLM


def test_cli_generate_writes_files(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(cli, "OpenAICompatibleLLM", lambda settings: FakeLLM(make_copy()))
    code = cli.main(["generate", str(EXAMPLES / "testo-830-t1.json"),
                     str(EXAMPLES.parent / "edge-cases" / "no-name.json"), "-o", str(tmp_path)])
    assert code == cli.EXIT_INPUT  # один файл с ошибкой, второй обработан
    card = json.loads((tmp_path / "testo-830-t1.json").read_text(encoding="utf-8"))
    assert card["status"] == "ok"
    assert (tmp_path / "testo-830-t1.md").read_text(encoding="utf-8").startswith("# testo 830-T1")

import pytest
from fastapi.testclient import TestClient

from kip_card import web
from kip_card.judge import Judge
from kip_card.llm import LLMError

from .conftest import FakeLLM, make_copy


@pytest.fixture
def client():
    def use(llm, judge=None):
        web.app.dependency_overrides[web._llm_dependency] = lambda: llm
        web.app.dependency_overrides[web._judge_dependency] = lambda: judge
        return TestClient(web.app)

    yield use
    web.app.dependency_overrides.clear()


def test_index_and_examples(client):
    c = client(FakeLLM(make_copy()))
    assert "Черновик карточки" in c.get("/").text
    names = c.get("/api/examples").json()
    assert "testo-830-t1" in names
    assert c.get("/api/examples/testo-830-t1").json()["name"] == "testo 830-T1"
    assert c.get("/api/examples/..%2F..%2Frequirements").status_code == 404


def test_generate_ok(client, testo_raw):
    response = client(FakeLLM(make_copy())).post("/api/generate", content=testo_raw)
    assert response.status_code == 200
    body = response.json()
    assert body["card"]["status"] == "ok"
    assert body["markdown"].startswith("# testo 830-T1")


def test_broken_json_is_422_and_does_not_call_model(client):
    llm = FakeLLM(make_copy())
    response = client(llm).post("/api/generate", content='{"name": "X",}')
    assert response.status_code == 422
    assert "Некорректный JSON" in response.json()["issues"][0]["message"]
    assert llm.calls == []


def test_llm_error_is_502(client, testo_raw):
    response = client(FakeLLM(LLMError("Сервер отклонил API-ключ"))).post("/api/generate", content=testo_raw)
    assert response.status_code == 502
    assert "ключ" in response.json()["error"]


def test_check_endpoint(client, manometer_raw):
    body = client(FakeLLM(make_copy())).post("/api/check", content=manometer_raw).json()
    assert body["profile"] == "manometer"
    assert body["missing_specs"] == ["Степень защиты"]


def test_health_without_key_reports_misconfiguration(client):
    web.get_settings.cache_clear()
    body = client(FakeLLM(make_copy())).get("/api/health").json()
    assert body["status"] == "misconfigured"
    web.get_settings.cache_clear()


def test_generate_with_judge(client, testo_raw):
    judge = Judge(FakeLLM({"issues": []}))
    body = client(FakeLLM(make_copy()), judge).post("/api/generate", content=testo_raw).json()
    assert body["card"]["meta"]["judge_status"] == "passed"
    assert body["card"]["meta"]["judge_model"] == "fake-model"

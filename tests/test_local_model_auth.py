"""Authentication regression tests use an in-memory gateway, never a network."""
from __future__ import annotations

import io
import base64
import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from lh_harness.local_models import EndpointAuthenticationError, EndpointUnavailableError, EndpointClient, check_endpoint, is_local
from lh_harness import eval_inspect_tb2_original as runner


class Response(io.BytesIO):
    def __init__(self, value, content_type="application/json"):
        super().__init__(value.encode())
        self.headers = {"Content-Type": content_type}


def vllm_gateway(monkeypatch):
    calls = []
    def open_request(request, **kwargs):
        calls.append(request)
        # Mirrors vLLM AuthenticationMiddleware: all /v1/* routes require
        # Bearer; an Anthropic x-api-key header alone is rejected.
        if request.get_header("Authorization") != "Bearer test-only-secret":
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO())
        if request.full_url.endswith("/models"):
            return Response(json.dumps({"data": [{"id": "local", "max_model_len": 32768, "owned_by": "vllm"}]}))
        if request.full_url.endswith("/version"):
            return Response('{"version":"test-only"}')
        assert request.full_url == "http://172.17.0.1:8010/v1/messages"
        payload = json.loads(request.data)
        if payload["model"] != "local":
            raise urllib.error.HTTPError(request.full_url, 404, "Missing model", {}, io.BytesIO())
        events = [{"type": "message_start", "message": {"usage": {"input_tokens": 9, "output_tokens": 0}}},
                  {"type": "content_block_start", "content_block": {"type": "tool_use", "name": "endpoint_probe", "input": {"value": "OK"}}},
                  {"type": "message_delta", "usage": {"output_tokens": 4}}, {"type": "message_stop"}]
        return Response("".join("data: " + json.dumps(event) + "\n\n" for event in events), "text/event-stream")
    monkeypatch.setattr("lh_harness.local_models.urllib.request.build_opener", lambda *a: SimpleNamespace(open=open_request))
    return calls


def test_native_messages_and_models_use_the_same_vllm_bearer_auth(monkeypatch):
    calls = vllm_gateway(monkeypatch)
    result = check_endpoint("http://172.17.0.1:8010", "claude_code", "local",
                            api_key="test-only-secret", auth_scheme="bearer")
    assert result["usage"]["total_tokens"] == 13
    assert result["auth_scheme"] == "bearer"
    assert result["invalid_model_rejected"]
    assert all(request.get_header("Authorization") == "Bearer test-only-secret" for request in calls)
    assert not any(request.get_header("X-api-key") for request in calls)


def test_messages_probe_includes_disabled_thinking_with_cli_high_effort(monkeypatch):
    calls = vllm_gateway(monkeypatch)
    result = check_endpoint("http://172.17.0.1:8010", "claude_code", "local",
                            {"reasoning": {"enable_thinking": False, "effort": "high"}},
                            api_key="test-only-secret", auth_scheme="bearer")
    payload = json.loads(next(request.data for request in calls if request.data))
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["output_config"] == {"effort": "high"}
    assert result["reasoning"] == {"enable_thinking": False, "effort": "high"}


def test_required_image_is_sent_with_the_same_tool_stream_probe(monkeypatch):
    calls = vllm_gateway(monkeypatch)
    result = check_endpoint("http://172.17.0.1:8010", "claude_code", "local",
                            {"require_image": True, "reasoning": {"enable_thinking": False}},
                            api_key="test-only-secret", auth_scheme="bearer")
    payload = json.loads(next(request.data for request in calls if request.data))
    content = payload["messages"][0]["content"]
    assert content[0]["source"]["media_type"] == "image/png"
    assert base64.b64decode(content[0]["source"]["data"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert content[1]["type"] == "text" and payload["stream"] is True
    assert payload["tool_choice"] == {"type": "tool", "name": "endpoint_probe"}
    assert result["image_probe"] == "verified"


@pytest.mark.parametrize("count", [5, 32])
def test_multi_image_probe_exercises_accumulated_images_and_records_count(monkeypatch, count):
    calls = vllm_gateway(monkeypatch)
    result = check_endpoint("http://172.17.0.1:8010", "claude_code", "local",
                            {"require_image": True, "image_probe_count": count},
                            api_key="test-only-secret", auth_scheme="bearer")
    payload = json.loads(next(request.data for request in calls if request.data))
    images = [item for item in payload["messages"][0]["content"] if item["type"] == "image"]
    assert len(images) == count
    assert result["image_probe_count"] == count
    assert result["tool_probe"] == "verified" and result["streaming"] == "verified"
    assert result["usage"]["status"] == "reported"


def test_four_image_server_is_rejected_by_multi_image_preflight(monkeypatch):
    from lh_harness.local_models import EndpointError
    vllm_gateway(monkeypatch)
    original = urllib.request.build_opener().open
    calls = []
    def limited(request, **kwargs):
        calls.append(request.full_url)
        if request.data:
            content = json.loads(request.data)["messages"][0]["content"]
            if sum(item["type"] == "image" for item in content) > 4:
                raise urllib.error.HTTPError(request.full_url, 500, "Internal Server Error", {},
                    io.BytesIO(b'{"error":{"message":"At most 4 image(s) may be provided in one prompt."}}'))
        return original(request, **kwargs)
    monkeypatch.setattr("lh_harness.local_models.urllib.request.build_opener", lambda *a: SimpleNamespace(open=limited))
    with pytest.raises(EndpointError, match="HTTP 500"):
        check_endpoint("http://172.17.0.1:8010", "claude_code", "local",
                       {"require_image": True, "image_probe_count": 32},
                       api_key="test-only-secret", auth_scheme="bearer")
    assert calls == ["http://172.17.0.1:8010/v1/models", "http://172.17.0.1:8010/v1/messages"]


@pytest.mark.parametrize("count", [0, -1, True])
def test_invalid_image_probe_counts_are_rejected_before_network(count):
    from lh_harness.local_models import EndpointError, validate_profile
    with pytest.raises(EndpointError, match="image_probe_count"):
        validate_profile({"require_image": True, "image_probe_count": count})


def test_old_x_api_key_mismatch_reports_authentication_not_protocol(monkeypatch):
    vllm_gateway(monkeypatch)
    with pytest.raises(EndpointAuthenticationError) as error:
        check_endpoint("http://172.17.0.1:8010", "claude_code", "local", api_key="test-only-secret")
    message = str(error.value)
    assert "HTTP 401" in message and "http://172.17.0.1:8010/v1" in message
    assert "auth_scheme=x-api-key" in message and "test-only-secret" not in message
    assert "protocol/model compatibility" not in message


def test_auth_none_does_not_send_inherited_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-test-secret")
    captured = []
    client = EndpointClient("http://172.17.0.1:8010", "claude_code", auth_scheme="none")
    client.opener = SimpleNamespace(open=lambda request, **kwargs: captured.append(request))
    client.request("/models")
    assert not captured[0].get_header("Authorization") and not captured[0].get_header("X-api-key")


def test_local_credentials_are_forwarded_to_original_runtime_without_serialization():
    config = runner.resolved_config(runner.DEFAULT_CONFIG, runner.DEFAULT_OUTPUT)
    env = {"LLH_LOCAL_API_KEY": "local-test-secret", "ANTHROPIC_API_KEY": "old-test-secret"}
    key, source = runner.configure_auth(env, config, "bearer", None)
    assert key == "local-test-secret" and source == "LLH_LOCAL_API_KEY"
    assert env["ANTHROPIC_API_KEY"] == key and env["LLH_LOCAL_BEARER_AUTH"] == "1"
    assert "local-test-secret" not in json.dumps(config)
    no_auth, source = runner.configure_auth(env, config, "none", None)
    assert no_auth is None and env["ANTHROPIC_API_KEY"] == "local"
    assert env["LLH_LOCAL_BEARER_AUTH"] == "0"


def test_named_credential_must_be_a_present_environment_variable():
    with pytest.raises(ValueError, match="unset or empty"):
        runner.local_credentials({}, "bearer", "MY_LOCAL_API_KEY")
    with pytest.raises(ValueError, match="name an environment variable"):
        runner.local_credentials({}, "bearer", "secret-value-do-not-pass")
    assert runner.local_credentials({"OPENAI_API_KEY": "test-value"}, "bearer") == ("test-value", "OPENAI_API_KEY")


def test_stale_public_endpoint_is_not_implicitly_used_for_local_run():
    with pytest.raises(ValueError, match="inherited ANTHROPIC_BASE_URL"):
        runner.model_url({"ANTHROPIC_BASE_URL": "https://old-provider.example/v1"})
    assert runner.model_url({"LLH_LOCAL_BASE_URL": "http://172.17.0.1:8010/v1",
                             "ANTHROPIC_BASE_URL": "https://old-provider.example/v1"}) == "http://172.17.0.1:8010"


@pytest.mark.parametrize("url,expected", [("http://host.docker.internal:8010", True),
                                         ("http://gpu-server:8010", True),
                                         ("http://gpu.lan:8010", True),
                                         ("https://api.example.com/v1", False)])
def test_known_local_names_bypass_shell_http_proxy(url, expected):
    assert is_local(url) is expected


@pytest.mark.parametrize("status", ["ok", "401"])
def test_model_only_check_prints_safe_diagnostics_and_never_starts_benchmark(monkeypatch, tmp_path, capsys, status):
    monkeypatch.setattr(runner, "prepare_dataset", lambda *a, **k: {"tasks": [{"name": "one"}]})
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/fake/harbor")
    monkeypatch.setenv("LLH_LOCAL_API_KEY", "private-test-value")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://old-provider.example/v1")
    def check(url, backend, model, profile, **kwargs):
        assert url == "http://172.17.0.1:8010"
        assert kwargs == {"api_key": "private-test-value", "auth_scheme": "bearer"}
        if status == "401":
            raise EndpointAuthenticationError("/models: HTTP 401; no fallback")
        return {"base_url": url + "/v1", "model_id": model, "usage": {"status": "reported"}}
    monkeypatch.setattr(runner, "check_endpoint", check)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("no Docker or Harbor call allowed"))
    jobs = tmp_path / "jobs"
    result = runner.main(["--check-model-only", "--auth-scheme", "bearer", "--base-url", "http://172.17.0.1:8010", "--jobs-dir", str(jobs)])
    assert result == (0 if status == "ok" else 2)
    capture = capsys.readouterr()
    assert "http://172.17.0.1:8010" in capture.out and "LLH_LOCAL_API_KEY" in capture.out
    assert "private-test-value" not in capture.out + capture.err
    assert not jobs.exists()


def test_connection_refused_is_service_unavailable_not_authentication(monkeypatch):
    client = EndpointClient("http://127.0.0.1:8010/v1", "claude_code", auth_scheme="none")
    def offline(*args, **kwargs):
        raise urllib.error.URLError("connection refused")
    client.opener = SimpleNamespace(open=offline)
    with pytest.raises(EndpointUnavailableError, match="start the model service") as error:
        client.json("/models")
    assert "401" not in str(error.value) and "authentication" not in str(error.value)


def test_loopback_is_allowed_for_host_only_checks_but_not_task_containers(monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("host-only check must not inspect Docker"))
    assert runner.model_url({}, allow_loopback=True) == "http://127.0.0.1:8010"
    assert runner.model_url({"LLH_LOCAL_BASE_URL": "http://127.0.0.1:8010/v1"}, allow_loopback=True) == "http://127.0.0.1:8010"
    with pytest.raises(ValueError, match="localhost refers to the task container"):
        runner.model_url({"LLH_LOCAL_BASE_URL": "http://127.0.0.1:8010/v1"})


@pytest.mark.parametrize("offline", [False, True])
def test_current_local_profile_defaults_to_no_auth_and_ignores_old_keys(monkeypatch, tmp_path, capsys, offline):
    monkeypatch.delenv("LLH_LOCAL_AUTH_SCHEME", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "old-provider-test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "old-openai-test-secret")
    monkeypatch.setattr(runner, "prepare_dataset", lambda *a, **k: {"tasks": [{"name": "one"}]})
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/fake/harbor")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("model-only check must not launch Docker/Harbor"))
    def check(url, backend, model, profile, **kwargs):
        assert url == "http://127.0.0.1:8010"
        assert kwargs == {"api_key": None, "auth_scheme": "none"}
        if offline:
            raise EndpointUnavailableError("endpoint unavailable; no fallback")
        return {"base_url": url + "/v1", "model_id": model, "usage": {"status": "reported"}}
    monkeypatch.setattr(runner, "check_endpoint", check)
    jobs = tmp_path / "jobs"
    code = runner.main(["--check-model-only", "--base-url", "http://127.0.0.1:8010/v1", "--jobs-dir", str(jobs)])
    assert code == (2 if offline else 0)
    captured = capsys.readouterr()
    assert "Authentication: none" in captured.out
    assert "old-provider-test-secret" not in captured.out + captured.err
    assert "old-openai-test-secret" not in captured.out + captured.err
    if offline:
        assert "start vLLM" in captured.err and "set LLH_LOCAL_API_KEY" not in captured.err
    assert not jobs.exists()

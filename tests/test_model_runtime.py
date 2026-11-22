import json
from pathlib import Path

from lh_harness.model_runtime import Monitor
from lh_harness.local_models import EndpointUnavailableError
from lh_harness import verifier_runtime


def test_transient_health_failure_does_not_abort_but_three_failures_do(tmp_path, monkeypatch):
    monitor = Monitor("http://127.0.0.1:8010", "local", interval=0, auth_scheme="none")
    def unavailable(*args):
        raise EndpointUnavailableError("unavailable")
    monkeypatch.setattr(monitor.client, "json", unavailable)
    assert monitor.check(tmp_path) is None
    assert monitor.check(tmp_path) is None
    assert monitor.check(tmp_path)["consecutive_failures"] == 3


def test_recovered_connection_resets_failures_and_model_change_is_rejected(tmp_path, monkeypatch):
    monitor = Monitor("http://127.0.0.1:8010", "local", interval=0)
    monitor.consecutive_failures = 2
    monkeypatch.setattr(monitor.client, "json", lambda *a: {"data": [{"id": "local"}]})
    assert monitor.check(tmp_path) is None and monitor.consecutive_failures == 0
    monkeypatch.setattr(monitor.client, "json", lambda *a: {"data": [{"id": "other"}]})
    for _ in range(3):
        failure = monitor.check(tmp_path)
    assert failure["kind"] == "model_endpoint_unavailable_or_changed"


def test_partial_proxy_event_is_not_lost_and_credentials_are_not_copied(tmp_path, monkeypatch):
    log = tmp_path / "trial/agent/episodes/manager/ep/request_proxy_requests.jsonl"
    log.parent.mkdir(parents=True)
    monitor = Monitor("http://127.0.0.1:8010", "local", interval=100)
    monkeypatch.setattr(monitor.client, "json", lambda *a: {"data": [{"id": "local"}]})
    event = json.dumps({"event": "upstream_http_error", "status": 401, "body_head": "secret-must-not-copy"})
    log.write_text('{"event":"request"}\n' + event[:20])
    assert monitor.check(tmp_path) is None
    with log.open("a") as stream:
        stream.write(event[20:] + "\n")
    failure = monitor.check(tmp_path)
    assert failure["http_status"] == 401 and "secret" not in json.dumps(failure)


def test_model_failure_stops_batch_and_preserves_scores(tmp_path):
    import sys
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text('{"reward":0}')
    class MonitorFixture:
        def check(self, job):
            return {"kind": "model_api_error", "no_fallback": True}
    assert verifier_runtime.run_harbor([sys.executable, "-c", "import time;time.sleep(10)"], {}, tmp_path, job,
                                      model_monitor=MonitorFixture()) == 2
    assert json.loads((job / "result.json").read_text()) == {"reward": 0}
    report = json.loads(job.with_suffix(".verification-status.json").read_text())
    assert not report["benchmark_valid"] and report["model_score"] is None

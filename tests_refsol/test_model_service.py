"""Transport/adapter tests independent of course TODOs and model weights."""

from contextlib import contextmanager
from http.client import HTTPConnection
import json
import sys
from threading import Thread
from types import ModuleType
from unittest.mock import Mock

import pytest

import model_service


class FakeGenerator:
    model_name = "test/model"
    device = "cpu"

    def __init__(self):
        self.calls = []
        self.error = None

    def generate(self, messages, max_tokens, enable_thinking):
        self.calls.append((messages, max_tokens, enable_thinking))
        if self.error:
            raise self.error
        return '  {"final":"完成"}\n'


@pytest.fixture
def service():
    generator = FakeGenerator()
    with model_service.make_server("127.0.0.1", 0, generator) as server:
        thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01))
        thread.start()
        try:
            yield server.server_port, generator
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def request(port, method="POST", path="/generate", payload=None, raw=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        body = raw if raw is not None else json.dumps(payload).encode("utf-8")
        connection.request(method, path, body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.getheader("Content-Type") == "application/json; charset=utf-8"
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_health_and_generation_preserve_transcript_and_defaults(service):
    port, generator = service
    assert request(port, "GET", "/health") == (
        200,
        {"status": "ok", "model": "test/model", "device": "cpu"},
    )
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "任务"},
    ]
    assert request(port, payload={"messages": messages}) == (
        200,
        {"response": '  {"final":"完成"}\n'},
    )
    messages = messages + [
        {"role": "assistant", "content": '{"tool":"read_file","path":"README.md"}'},
        {"role": "user", "content": "Tool result:\ncontent"},
    ]
    assert (
        request(
            port,
            payload={"messages": messages, "max_tokens": 17, "enable_thinking": True},
        )[0]
        == 200
    )
    assert generator.calls == [(messages[:2], 256, False), (messages, 17, True)]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"messages": []},
        {"messages": "bad"},
        {"messages": [None]},
        {"messages": [{"role": "user"}]},
        {"messages": [{"role": 1, "content": "x"}]},
        {"messages": [{"role": "user", "content": None}]},
        *(
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": value}
            for value in [0, -1, True, 1.5, "2", None]
        ),
        *(
            {"messages": [{"role": "user", "content": "x"}], "enable_thinking": value}
            for value in [1, "false", None]
        ),
    ],
)
def test_invalid_requests_are_400_without_generation(service, payload):
    port, generator = service
    status, body = request(port, payload=payload)
    assert status == 400
    assert isinstance(body["error"], str)
    assert not generator.calls


@pytest.mark.parametrize("raw", [b"not JSON", b"\xff", b""])
def test_malformed_json_is_400(service, raw):
    port, generator = service
    assert request(port, raw=raw)[0] == 400
    assert not generator.calls


def test_generation_failure_is_500_and_server_recovers(service):
    port, generator = service
    generator.error = RuntimeError("test failure")
    payload = {"messages": [{"role": "user", "content": "x"}]}
    assert request(port, payload=payload) == (
        500,
        {"error": "generation failed: test failure"},
    )
    generator.error = None
    assert request(port, payload=payload)[0] == 200
    for method in ["GET", "POST"]:
        assert request(port, method, "/missing")[0] == 404


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_mlx_loads_once_and_uses_agent_template_and_device(monkeypatch, device):
    active_streams = []
    stream_entries = []

    @contextmanager
    def stream(selected):
        active_streams.append(selected)
        stream_entries.append(selected)
        try:
            yield
        finally:
            active_streams.pop()

    mx = ModuleType("mlx.core")
    mx.cpu, mx.gpu, mx.stream = "cpu-stream", "gpu-stream", stream
    mlx = ModuleType("mlx")
    mlx.core = mx
    lm = ModuleType("mlx_lm")
    model = object()
    tokenizer = Mock()

    def template(*args, **kwargs):
        assert active_streams == [f"{device}-stream"]
        return "rendered prompt"

    def generate(*args, **kwargs):
        assert active_streams == [f"{device}-stream"]
        return " unmodified output\n"

    tokenizer.apply_chat_template.side_effect = template
    lm.load = Mock(return_value=(model, tokenizer))
    lm.generate = Mock(side_effect=generate)
    for name, module in [("mlx", mlx), ("mlx.core", mx), ("mlx_lm", lm)]:
        monkeypatch.setitem(sys.modules, name, module)

    adapter = model_service.MlxGenerator("qwen3-4b", device)
    messages = [{"role": "user", "content": "task"}]
    for thinking in [False, True]:
        assert adapter.generate(messages, 19, thinking) == " unmodified output\n"
        tokenizer.apply_chat_template.assert_called_with(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        lm.generate.assert_called_with(
            model, tokenizer, "rendered prompt", max_tokens=19, verbose=False
        )
    lm.load.assert_called_once_with("Qwen/Qwen3-4B-MLX-4bit")
    assert stream_entries == [f"{device}-stream"] * 2
    assert not active_streams


def test_load_failure_exits_before_binding(monkeypatch, capsys):
    monkeypatch.setattr(
        model_service, "MlxGenerator", Mock(side_effect=RuntimeError("load failed"))
    )
    make_server = Mock()
    monkeypatch.setattr(model_service, "make_server", make_server)
    with pytest.raises(SystemExit) as error:
        model_service.main([])
    assert error.value.code == 1
    assert "could not load local model" in capsys.readouterr().err
    make_server.assert_not_called()


def test_cli_defaults_and_invalid_port(monkeypatch):
    args = model_service.build_parser().parse_args([])
    assert (args.host, args.port, args.model, args.device) == (
        "127.0.0.1",
        8000,
        "qwen3-4b",
        "gpu",
    )
    load = Mock()
    monkeypatch.setattr(model_service, "MlxGenerator", load)
    for port in ["0", "65536"]:
        with pytest.raises(SystemExit) as error:
            model_service.main(["--port", port])
        assert error.value.code == 2
    load.assert_not_called()

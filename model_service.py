"""Serve the same MLX-LM generation path as agent.py over JSON HTTP."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from model_names import shortcut_name_to_full_name


class MlxGenerator:
    """One loaded model; called serially by HTTPServer on its serving thread."""

    def __init__(self, model_name: str, device: str):
        import mlx.core as mx
        from mlx_lm import generate, load

        self.model_name = shortcut_name_to_full_name(model_name)
        self.device = device
        self._mx = mx
        self._generate = generate
        self._model, self._tokenizer = load(self.model_name)

    def generate(self, messages, max_tokens: int, enable_thinking: bool) -> str:
        with self._mx.stream(self._mx.gpu if self.device == "gpu" else self._mx.cpu):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            return self._generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=max_tokens,
                verbose=False,
            )


def validate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict) or not all(
            isinstance(message.get(key), str) for key in ("role", "content")
        ):
            raise ValueError("each message must contain string role and content fields")
    max_tokens = payload.get("max_tokens", 256)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    enable_thinking = payload.get("enable_thinking", False)
    if type(enable_thinking) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    return messages, max_tokens, enable_thinking


def make_server(host: str, port: int, generator) -> HTTPServer:
    """Inject a generator so transport tests never need MLX or model weights."""

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_json(404, {"error": "unknown endpoint"})
                return
            self.send_json(
                200,
                {
                    "status": "ok",
                    "model": generator.model_name,
                    "device": generator.device,
                },
            )

        def do_POST(self) -> None:
            if self.path != "/generate":
                self.send_json(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("Content-Length must be positive")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                messages, max_tokens, enable_thinking = validate_request(payload)
            except (ValueError, UnicodeError) as error:
                self.send_json(400, {"error": str(error)})
                return
            try:
                response = generator.generate(messages, max_tokens, enable_thinking)
                if not isinstance(response, str):
                    raise TypeError("model response must be a string")
            except Exception as error:
                self.send_json(500, {"error": f"generation failed: {error}"})
                return
            self.send_json(200, {"response": response})

    return HTTPServer((host, port), Handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    model_name = shortcut_name_to_full_name(args.model)
    try:
        generator = MlxGenerator(model_name, args.device)
    except Exception as error:
        parser.exit(1, f"error: could not load local model {model_name}: {error}\n")
    try:
        server = make_server(args.host, args.port, generator)
    except OSError as error:
        parser.exit(1, f"error: could not start model service: {error}\n")
    with server:
        print(f"model> {generator.model_name}", flush=True)
        print(f"device> {generator.device}", flush=True)
        print(f"listening> http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Embed adapter :8788 — formato splatdb MCP -> ollama OpenAI-compat :8860.

IN:  POST {"texts": ["..."]}           (también acepta {"text": "..."})
OUT: 200 {"embeddings": [[...], ...]}  (768 dims, embeddinggemma-2, CUDA)

Mantiene el contrato de EMBEDDING_SERVICE_URL_DEFAULT del MCP de splatdb
(src/mcp_server.rs:26). Sin deps externas: stdlib only.
"""
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA_URL = "http://127.0.0.1:8860/v1/embeddings"
MODEL = "google/embeddinggemma-2"
PORT = 8788


def embed(texts):
    payload = json.dumps({"model": MODEL, "input": list(texts)}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    items = sorted(resp.get("data", []), key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in items]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[embed8788] %s\n" % (fmt % args))

    def do_GET(self):
        body = json.dumps({"status": "ok", "model": MODEL}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            texts = req.get("texts") or ([req["text"]] if req.get("text") else [])
            if not texts or not all(isinstance(t, str) and t for t in texts):
                raise ValueError("texts debe ser lista de strings no vacíos")
            embs = embed(texts)
            body = json.dumps({"embeddings": embs}).encode()
            self.send_response(200)
        except Exception as e:  # fail loud al cliente, no colgar
            body = json.dumps({"error": str(e)[:300]}).encode()
            self.send_response(502)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    sys.stderr.write(f"[embed8788] listening on 127.0.0.1:{PORT} -> {OLLAMA_URL}\n")
    sys.stderr.flush()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

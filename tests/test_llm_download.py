from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from llm.qwen3_1_7b import download


def test_range_resume_bypasses_proxy(monkeypatch, tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers["Range"])
            self.send_response(206)
            self.send_header("Content-Range", "bytes 6-7/8")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"gh")

        def log_message(self, *args):
            pass

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    target = tmp_path / "part"
    target.write_bytes(b"def")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            download.fetch(f"http://127.0.0.1:{server.server_port}/weights", target, 3, 7)
        finally:
            server.shutdown()
            thread.join()
    assert received == ["bytes=6-7"]
    assert target.read_bytes() == b"defgh"

"""Serve a viewer's browser client over plain http.

A viewer run in `WEB` mode starts this next to its websocket server: the page
it serves is the ordinary `CLIENT` running under Pyodide (see `web.py`), so
there is nothing to deploy and no second checkout to keep in sync. The page is
generic -- it is handed the viewer class to construct, the modules to import it
from, and the websocket port to dial -- so any `Viewer` subclass runs in a
browser without the page knowing about it.

Everything is generated per request from the live source tree, which is what
lets a checkout's own code (rather than an installed copy) reach the browser.
"""
import importlib.util
import inspect
import io
import json
import socket
import threading
import zipfile
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).resolve().parent / "client.html"
PACKAGE_DIR = Path(__file__).resolve().parent


def zip_python(*roots: Path) -> bytes:
    """Zip modules and packages so unpacking yields them at the import root."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root in roots:
            if root.is_dir():
                for path in sorted(root.rglob("*.py")):
                    zf.write(path, path.relative_to(root.parent))
            else:
                zf.write(root, root.name)
    return buf.getvalue()


def client_sources(viewer_cls: type) -> list[Path]:
    """The sources the browser needs besides graphdecoviewer: the module that
    defines `viewer_cls`, plus whatever its `web_client_modules` names."""
    sources = [Path(inspect.getsourcefile(viewer_cls)).resolve()]
    for name in viewer_cls.web_client_modules:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(
                f"{viewer_cls.__name__}.web_client_modules names {name!r}, "
                "which does not import here.")
        origin = Path(spec.origin).resolve()
        # A package is shipped whole; a plain module as its single file.
        sources.append(origin.parent if origin.name == "__init__.py" else origin)
    return sources


def client_spec(viewer_cls: type, ws_port: int) -> dict:
    """What the page needs to start the client: where the viewer class lives
    (its module name as it will import from `app.zip`) and the port to dial."""
    source = Path(inspect.getsourcefile(viewer_cls)).resolve()
    return {"module": source.stem, "class": viewer_cls.__name__, "port": ws_port}


class Handler(BaseHTTPRequestHandler):
    """Four generated routes; no directory is exposed."""

    def __init__(self, *args, viewer_cls: type, ws_port: int, **kwargs):
        self._viewer_cls = viewer_cls
        self._ws_port = ws_port
        super().__init__(*args, **kwargs)

    def _route(self, path: str):
        if path in ("/", "/index.html"):
            return "text/html", PAGE.read_bytes()
        if path == "/app.json":
            spec = client_spec(self._viewer_cls, self._ws_port)
            return "application/json", json.dumps(spec).encode()
        if path == "/graphdecoviewer.zip":
            return "application/zip", zip_python(PACKAGE_DIR)
        if path == "/app.zip":
            return "application/zip", zip_python(*client_sources(self._viewer_cls))
        return None

    def do_GET(self):
        served = self._route(self.path.split("?")[0])
        if served is None:
            self.send_error(404)
            return
        content_type, data = served
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        # Pyodide may use SharedArrayBuffer, which browsers only enable on
        # cross-origin-isolated pages; every CDN it loads from sends CORS.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        pass   # one line per asset fetch is noise next to the viewer's own output


def client_url(host: str, port: int) -> str:
    """The link to hand the user. A wildcard bind has no name of its own, so
    advertise the machine's instead."""
    if host in ("", "0.0.0.0", "::"):
        host = socket.getfqdn()
    return f"http://{host}:{port}/"


def serve_client(viewer_cls: type, host: str, port: int, ws_port: int) -> ThreadingHTTPServer:
    """Start the page server in a daemon thread and return it (call `shutdown`
    to stop it)."""
    server = ThreadingHTTPServer(
        (host, port), partial(Handler, viewer_cls=viewer_cls, ws_port=ws_port))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

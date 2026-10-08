"""CLIENT-mode networking for the browser (Pyodide).

The browser has no `websockets` package and no threads, so the client talks
to the server through the page's own `WebSocket`. Messages are queued as they
arrive and the GUI frame pulls one whole update at a time, keeping the last
frame on screen while the next one is still in flight. The wire format is the
native one (see `Viewer._send_update`), so a native SERVER needs no change.
"""
import json
from collections import deque


class ConnectionClosed(Exception):
    """Stands in for `websockets.exceptions.ConnectionClosed`."""


class BrowserWebSocket:
    """Wraps a JS `WebSocket` behind the subset of the `websockets` sync
    client API the viewer uses (`send`, `close`), plus a non-blocking
    `recv_update` in place of `recv`."""

    def __init__(self, url: str):
        import asyncio
        from js import WebSocket
        from pyodide.ffi import create_proxy

        self._messages = deque()
        self._closed = False
        self._opened = asyncio.get_event_loop().create_future()
        # True while the server owes us no reply (see `Viewer._main`).
        self.ready_to_send = True

        # The proxies must outlive the socket: JS holds only weak handles.
        self._proxies = [create_proxy(f) for f in
                         (self._on_open, self._on_message, self._on_close, self._on_error)]
        self._ws = WebSocket.new(url)
        self._ws.binaryType = "arraybuffer"
        self._ws.onopen, self._ws.onmessage, self._ws.onclose, self._ws.onerror = self._proxies

    async def wait_open(self):
        """Resolve once the socket is open; raise if it fails to connect."""
        await self._opened

    def _on_open(self, _event):
        if not self._opened.done():
            self._opened.set_result(None)

    def _on_message(self, event):
        data = event.data
        self._messages.append(data if isinstance(data, str) else data.to_bytes())

    def _on_close(self, _event):
        self._closed = True
        if not self._opened.done():
            self._opened.set_exception(ConnectionClosed("connection refused"))

    def _on_error(self, _event):
        # The browser reports the reason on the console; `onclose` follows.
        self._closed = True

    def send(self, data, text: bool = True):
        if self._closed:
            raise ConnectionClosed("connection closed")
        from pyodide.ffi import to_js
        self.ready_to_send = False
        self._ws.send(data if text else to_js(data))

    def recv_update(self):
        """Pop one complete update (metadata, mapping, binaries) if it has
        fully arrived, else None. Raises once the socket is closed."""
        if len(self._messages) >= 2:
            n_binaries = len(json.loads(self._messages[1]))
            if len(self._messages) >= 2 + n_binaries:
                self.ready_to_send = True
                return [self._messages.popleft() for _ in range(2 + n_binaries)]
        if self._closed:
            raise ConnectionClosed("connection closed")
        return None

    def close(self):
        self._closed = True
        self._ws.close()
        for proxy in self._proxies:
            proxy.destroy()
        self._proxies = []

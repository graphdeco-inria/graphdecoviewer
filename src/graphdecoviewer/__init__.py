import json
import time
import threading
from typing import Optional
from collections import defaultdict
from .types import *
from .widgets import Widget
from .gl_context import create_headless_context, NullContext
from abc import ABC, abstractmethod

# The browser build (see `types.IS_WEB`) has neither glfw nor `websockets`;
# both are only needed by the native SERVER / CLIENT / LOCAL paths.
try:
    import glfw
except ImportError:
    glfw = None
try:
    from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError
    from websockets.sync.server import serve, ServerConnection
    from websockets.sync.client import connect, ClientConnection
except ImportError:
    from .web import ConnectionClosed
    ConnectionClosedOK = ConnectionClosedError = ConnectionClosed
    ServerConnection = ClientConnection = None

class Viewer(ABC):
    """
    Base class for viewer. This class setups up the relevant ImGui callbacks.
    The child class must override the 'show_gui' function to build the GUI.
    It can can also override the 'step' function to perform any per frame
    computations required.
    The '(server|client)_(send|recv)' should be used for message passing between
    the server and client for remote viewer support.
    """
    def __init__(self, mode: ViewerMode):
        if not hasattr(self, "window_title"):
            self.window_title = "Viewer"
        self.should_exit = False
        self.num_connections = 0

        # Only used in client mode
        self.websocket = None

        # Mapping of widget_id to widget
        self.widget_id_to_widget = {}

        self.mode = mode

        # Import server specific modules
        if self.mode & LOCAL_SERVER:
            self.import_server_modules()

        # Import client specific modules
        self.parent_import_server_modules_called = False
        if self.mode & LOCAL_CLIENT:
            self.import_client_modules()
            assert self.parent_import_server_modules_called, \
                "Call to `super().import_client_modules()` missing."

    def _setup(self):
        """ Go over all of the widgets and initialize them """
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                widget.setup()
                self.widget_id_to_widget[widget.widget_id] = widget

    def _destroy(self):
        """ Go over all of the widgets and free any manually allocated objects """
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                widget.destroy()

    def _main(self, websocket=None):
        """
        TODO: Update
        Internal method which handles inputs, resize and calls
        backend computation and then creates the UI.
        """
        if self.mode is CLIENT and self.websocket is not None:
            try:
                # The exchange is lockstep: one client update, one server reply.
                # A native socket enforces that by blocking in recv; the browser
                # socket never blocks, so it must not send again until the
                # reply to the previous update has been consumed.
                if not IS_WEB or self.websocket.ready_to_send:
                    self._send_update(self.websocket, "client")
            except ConnectionClosed:
                print("INFO: Server disconnected")
                self.websocket.close()
                self.websocket = None

        if self.mode is SERVER:
            self._dispatch_update(self._recv_update(websocket), "server")

        if self.mode & LOCAL_SERVER:
            self.step()

        if self.mode is SERVER:
            self._send_update(websocket, "server")

        if self.mode is CLIENT and self.websocket is not None:
            try:
                if IS_WEB:
                    # None until a whole update has arrived; keep the last frame.
                    messages = self.websocket.recv_update()
                else:
                    messages = self._recv_update(self.websocket)
                if messages is not None:
                    self._dispatch_update(messages, "client")
            except ConnectionClosed:
                print("INFO: Server disconnected")
                self.websocket.close()
                self.websocket = None

        if self.mode & LOCAL_CLIENT:
            self.show_gui()

    def _server_loop(self, websocket: ServerConnection):
        """ Internal method which runs the server loop. """
        # We only allow one client to connect at a time (for now).
        if self.num_connections > 0:
            print("INFO: Client already connected. Only one client is allowed.")
            websocket.close()
            return
        self.num_connections += 1

        self._gl.make_current()
        self.onconnect(websocket)

        # Main Loop
        try:
            while True:
                self._main(websocket)
        except ConnectionClosedOK:
            print("INFO: Client disconnected.")
            self.num_connections -= 1
        except ConnectionClosedError as e:
            print(f"ERROR: Connection closed with error: {e}")
            self.num_connections -= 1

        self._gl.release()

    # Wire format of one update, in either direction: a JSON metadata message
    # keyed by widget id, a JSON list mapping each following binary message to
    # its widget id, then one binary message per entry of that list.

    def _send_update(self, websocket, side: str):
        """
        Compile the state of all registered widgets and the viewer itself and
        send it over `websocket`. `side` is "server" or "client" and selects
        the `<side>_send` hooks.
        """
        metadata = {}   # Metadata for each widget (Should be JSON serializable)
        all_binaries = []   # List of all binaries to be sent
        binary_to_widget = []   # Mapping of which binary is for which widget
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                binary, text = getattr(widget, f"{side}_send")()
                if text is not None:
                    metadata[widget.widget_id] = text
                if binary is not None:
                    all_binaries.append(binary)
                    binary_to_widget.append(widget.widget_id)

        # Add global state (viewer)
        binary, text = getattr(self, f"{side}_send")()
        if text is not None:
            metadata["viewer"] = text
        if binary is not None:
            all_binaries.append(binary)
            binary_to_widget.append("viewer")

        websocket.send(json.dumps(metadata), text=True)
        websocket.send(json.dumps(binary_to_widget), text=True)
        for binary in all_binaries:
            websocket.send(binary, text=False)

    @staticmethod
    def _recv_update(websocket) -> list:
        """ Block until one whole update has been received; return its messages. """
        metadata = websocket.recv()
        binary_to_widget = websocket.recv()
        binaries = [websocket.recv() for _ in json.loads(binary_to_widget)]
        return [metadata, binary_to_widget] + binaries

    def _dispatch_update(self, messages: list, side: str):
        """
        Hand the messages of one received update to the widgets they belong
        to, through their `<side>_recv` hooks ("server" or "client").
        """
        metadata = json.loads(messages[0])
        binary_to_widget = json.loads(messages[1])
        all_binaries = messages[2:]

        # Assemble the data
        all_data = defaultdict(dict)
        for widget_id, text in metadata.items():
            if widget_id == "viewer":
                all_data["viewer"]["metadata"] = text
            else:
                all_data[int(widget_id)]["metadata"] = text
        for widget_id, binary in zip(binary_to_widget, all_binaries):
            all_data[widget_id]["binary"] = binary

        # Update the widgets
        for widget_id, data in all_data.items():
            if widget_id == "viewer":
                target = self
            else:
                target = self.widget_id_to_widget[int(widget_id)]
            getattr(target, f"{side}_recv")(data.get("binary", None), data.get("metadata", None))

    def _connected(self, websocket):
        """ Finish a client connection: run `onconnect`, then publish the socket. """
        print("INFO: Connected to server.")
        self.onconnect(websocket)
        # Make websocket available after onconnect finishes to avoid
        # the main thread from using it
        self.websocket = websocket

    def _client_loop(self, ip: str, port: int):
        """
        Internal method which runs the client loop. This loop only deals with
        connecting to the server and handling reconnections. The '_main' method
        is run by the 'immapp.run' function.
        """
        while True:
            # Try to connect to the server
            if self.websocket is None:
                try:
                    self._connected(connect(f"ws://{ip}:{port}", max_size=None, compression=None))
                except Exception as e:
                    print(f"INFO: Failed to connect to server with error: {e}."
                        " Retrying in 2 seconds.")
                    self.websocket = None
            time.sleep(2)

    async def _web_client_loop(self, ip: str, port: int):
        """ `_client_loop` for the browser, where there are no threads. """
        import asyncio
        from .web import BrowserWebSocket
        while True:
            if self.websocket is None:
                try:
                    websocket = BrowserWebSocket(f"ws://{ip}:{port}")
                    await websocket.wait_open()
                    self._connected(websocket)
                except Exception as e:
                    print(f"INFO: Failed to connect to server with error: {e}."
                        " Retrying in 2 seconds.")
                    self.websocket = None
            await asyncio.sleep(2)

    def onconnect(self, websocket: ClientConnection|ServerConnection):
        """ Called when a new connection is made. """
        pass

    def run(self, ip: str = "localhost", port: int = 6009):
        self.create_widgets()
        self.running = True

        # Run the client connection in a different thread, the main thread runs the GUI.
        if self.mode is CLIENT:
            if IS_WEB:
                import asyncio
                asyncio.ensure_future(self._web_client_loop(ip, port))
            else:
                connect_thread = threading.Thread(target=self._client_loop, args=(ip, port))
                # Make the thread a daemon so that it exits when the main thread exits.
                connect_thread.daemon = True
                connect_thread.start()
        if self.mode & LOCAL_CLIENT:
            self._runner_params = hello_imgui.RunnerParams()
            self._runner_params.fps_idling.enable_idling = False
            self._runner_params.app_window_params.window_geometry.window_size_state = hello_imgui.WindowSizeState.maximized
            self._runner_params.app_window_params.window_title = self.window_title
            self._runner_params.imgui_window_params.show_status_bar = True
            self._runner_params.imgui_window_params.show_menu_bar = True
            self._runner_params.callbacks.post_init = self._setup
            # self._runner_params.callbacks.before_exit = self._before_exit
            self._runner_params.callbacks.show_gui = self._main
            self._runner_params.callbacks.show_status = self.show_status
            if glfw is not None:
                # Disable VSync
                self._runner_params.callbacks.post_init_add_platform_backend_callbacks = lambda: glfw.swap_interval(0)
                self._runner_params.platform_backend_type = hello_imgui.PlatformBackendType.glfw
            self._addon_params = immapp.AddOnsParams(with_implot=True)

            # This is required to make 'want_capture_*' work. The default value is to create a full screen window,
            # but that would mean the 'want_capture_mouse' variable will always be set.
            self._runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
            # In the browser imgui_bundle patches `run` to return at once and
            # drive the frames from the page's animation loop instead.
            immapp.run(self._runner_params, self._addon_params)
        if self.mode is SERVER:
            # SERVER renders the widgets' offscreen framebuffers and streams them,
            # so it needs a current OpenGL context but no visible window. Prefer
            # GLFW (the GUI's backend) and fall back to a windowless EGL context on
            # headless hosts with no display. Both request an OpenGL 3.3 core
            # profile (see gl_context.py): a compatibility context leaves point
            # sprites off, so `gl_PointCoord` reads (0,0) in fragment shaders and
            # widgets that rely on it (e.g. PointRenderer's disc test) discard every
            # fragment.
            try:
                self._gl = create_headless_context()
            except RuntimeError as e:
                # Only widgets that render need a context; a server made of
                # image/camera widgets can run on a host without OpenGL.
                print(f"WARNING: {e} OpenGL widgets will not work.")
                self._gl = NullContext()
            self._gl.make_current()
            self._setup()

            # Release the context so the server thread can make it current.
            self._gl.release()

            # Start server
            with serve(self._server_loop, ip, port, max_size=None, compression=None) as server:
                server_thread = threading.Thread(target=server.serve_forever)
                server_thread.start()
                while True:
                    try:
                        time.sleep(1)
                    except KeyboardInterrupt:
                        print("INFO: Shutting down server.")
                        server.shutdown()
                        server_thread.join()
                        break

            # Reacquire the context and free resources
            self._gl.make_current()
            self._destroy()
            self._gl.destroy()

        self.running = False

    def step(self):
        """ Your application logic goes here. """
        pass

    def create_widgets(self):
        """ Define stateful widgets here. """

    def server_send(self) -> tuple[Optional[bytes],Optional[dict]]:
        """ Send global viewer state to the client. """
        return None, None

    def server_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """ Receive and process global viewer state from the client. """

    def client_send(self) -> tuple[Optional[bytes],Optional[dict]]:
        """ Send global viewer state to the server. """
        return None, None

    def client_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """ Receive and process global viewer state from the server. """
        pass

    def show_status(self):
        """ Use this function to render status bar at the bottom. """

    def import_server_modules(self):
        """
        Import server specific modules here. We want the viewer to run without
        any additional dependancies so we only import the server modules when
        on the server. The modules imported here can only be used in `step` and
        `server*` methods. Don't forget to declare the variables as `global` to
        ensure that they are globally accesible.
        """

    def import_client_modules(self):
        """
        Import client specific modules here. We want the viewer to run without
        needing to install `imgui_bundle` on the server. The modules imported
        here can only be used in `show_gui` and `client*` methods. Don't forget
        to declare the variables as `global` to  ensure that they are globally
        accesible.
        """
        global immapp
        global hello_imgui
        from imgui_bundle import immapp, hello_imgui

        self.parent_import_server_modules_called = True

    @abstractmethod
    def show_gui(self) -> bool:
        """ Define the GUI here. """

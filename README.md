# GraphDeco Viewer
GraphDeco Viewer is a modern, Python-based 3D visualization tool designed as a replacement for the legacy [SIBRViewer](https://sibr.gitlabpages.inria.fr/?page=index.html), developed at INRIA Sophia-Antipolis. It provides a flexible and lightweight platform for visualizing outputs of 3D algorithms, making it particularly well-suited for research, development, and prototyping in computer graphics and vision. The viewer is written completely in Python to allow for easy installation and prototyping. It is based upon [ImGui](https://github.com/ocornut/imgui) and uses [imgui-bundle](https://github.com/pthom/imgui_bundle) for the Python bindings.

**NOTE**: The current release is in *alpha* mode, and there may be breaking changes in future updates as the project evolves.

## Installation
The package is available in two versions: **GUI** and **Non-GUI**.
The only difference is that the GUI version requires `imgui-bundle`.
If your system lacks the dependencies to build `imgui-bundle`, you can still use the viewer in `SERVER` mode (no GUI needed) and visualize the results locally with the **GUI** version.

### GUI Version
```bash
pip install git+https://github.com/graphdeco-inria/graphdecoviewer[gui]
```

### Non GUI Version
```bash
pip install git+https://github.com/graphdeco-inria/graphdecoviewer
```

## Citation
If you use this software in your publication, please consider citing it:
```
@software{shah_graphdecoviewer,
  author = {Shah, Ishaan and Meuleman, Andreas and Lanvin, Alexandre and Drettakis, George},
  title = {GraphDeco Viewer},
  url = {https://github.com/graphdeco-inria/graphdecoviewer},
}

```

## Usage
The viewer can run in four different modes:
- `LOCAL`: Opens a window on the same device.
- `SERVER`: Creates a websocket server on the device which would perform all of the computation to be displayed on another device.
- `CLIENT`: Opens a window to display the GUI and connects to the websocket server.
- `WEB`: A `SERVER` that also serves its own browser client, and prints the link to open.

### Browser client (`WEB` mode)

`CLIENT` also runs in a browser through [Pyodide](https://pyodide.org) and the
wasm build of `imgui-bundle`: the same viewer class, the same protocol, no
server change. `WEB` mode is that whole setup in one process -- it starts the
websocket server *and* an http server for the page, and nothing has to be
deployed:

```python
MyViewer(WEB).run("0.0.0.0", 6009, web_port=8000)
# INFO: Open the viewer at http://<this machine>:8000/
```

The page is generic over the viewer: `web_server.py` hands it the class to
construct (`app.json`), this package's source (`graphdecoviewer.zip`) and the
module that defines the viewer (`app.zip`), all generated per request from the
live source tree, so a checkout's own code is what runs in the browser. A
viewer whose browser client needs further modules of its own lists them in
`web_client_modules`; one whose constructor needs more than the mode overrides
the `web_client` classmethod. `?host=&port=` on the page's URL override the
server to dial, which is how you reach one through an SSH tunnel.

The browser has no PyOpenGL, `glfw`, `websockets` or threads;
`graphdecoviewer.types.IS_WEB` selects the substitutes (`graphdecoviewer/web.py`
wraps the page's `WebSocket`, `Texture2D.update_rgba` uploads images through
HelloImGui) and `immapp.run` returns at once, the page driving the frames.
Pyodide's Python/Emscripten ABI has to match the `imgui-bundle` wasm wheel the
page installs; both are pinned in `client.html`.

### Creating a new Viewer
To create a new viewer, you must inherit from the `Viewer` class and override the following function:
- `__init__`: If you override this function, you must necessarily call the `__init__` of the parent class using `super().__init__(mode)` to ensure the state variables and appropriate modules are imported.
- `create_widgets`: Overide this function to define widgets required by the viewer along with any additional state variables.
- `import_server_modules`: Import any modules required only on the server here. This typically would involve packages such as `torch` and other local code dependancies which might not be available in the client environment. The modules imported here are only available in `server_(send|recv)` and `step` functions.
- `import_client_modules`: Import any modules required only on the client here. This typically would be `imgui_bundle` since it might not be available in the server environment. The modules imported here are only available in `client_(send|recv)` and `client` functions.
- `step`: Define any computation in this function. This function will only be called on the server in network mode
- `show_gui`: Define the GUI in this function. This function will only be called on the client in network mode.
- `(client|server)_send`: Override this function to send any data from the client or server. The function should return a tuple with the first element being any binary data, and the second being a dictionary containing any text metadata. Either elements can be `None` if the viewer doesn't need to send that type of data. The default implementation returns `(None,None)`.
- `(client|server)_recv`: Override this function to define how to update the viewer state with received data. The functions is expected to receive two arguments, the first being the binary data and, the second being the text.

### Creating new Widgets
To create a new widget, you must inherit from the `Widget` class and override the following functions:
- `__init__`: If you override this function, you must necessarily call the `__init__` of the parent class using `super().__init__(mode)` to ensure an appropriate ID gets assigned the the widget which is required for correct functioning in network mode.
- `setup`: This function will be called after initializing `glfw` and `OpenGL`. Setup related to the above two should go here (For eg. creating an OpenGL Texture).
- `destroy`: This function will be called when the application exits. Override this function to deallocate any OpenGL objects or anything else which doesn't support automatic garbage collection.
- `step`: Same as `Viewer.step`.
- `show_gui`: Same as `Viewer.show_gui`.
- `import_client_module`: Same as `Viewer.import_client_module`.
- `(client|server)_send`: Same as `Viewer.(client|server)_send`.
- `(client|server)_recv`: Same as `Viewer.(client|server)_recv`.

## Widgets
Following is the list of widgets which are available/planned to be included in the viewer by default.
|Name|Local|Network|Description|Note|
|----|-----|-------|-----------|----|
|NumpyImage|x|x|Displays a `np.ndarray` as a 2D Texture||
|TorchImage|x|x|Displays a `torch.Tensor` as a 2D Texture w/o requiring a roundtrip to the CPU|Requires `cuda-python`|
|PerformanceMonitor|x|x|A widget to display timings for different phases of your algorithm||
|FPS Camera|x|x|A first person camera||
|EllipsoidViewer|x|x|A widget to visualize gaussians as Ellipsoids||
|RadioPicker|x|x|A simple radio selection widget||
|EllipseViewer|||A widget to visualize 2D gaussians as Ellipses||
|PixelInspector|||A widget which allows to zoom onto a texture and inspect pixels||
|ImageCompare|||A image slider to compare two images side by side||

## Release Notes
### v0.1.2 (WIP)
- [x] (BREAKING) Make `imgui-bundle` optional.

  `imgui-bundle` is no longer installed by default. All imports from `imgui-bundle` must happen in `import_client_module` and they can only be used in `show_gui` and `client_(send|recv)`. This allows installation of the package on systems which don't have packages to build `imgui-bundle` from scratch.

- [x] Fix bug causing issue in remote mode.
- [x] Run `CLIENT` in a browser, and add `WEB` mode to serve it.
- [ ] Add `PointRenderer`.
- [ ] Add `ImageCompare`
"""Headless OpenGL contexts for SERVER mode.

SERVER mode renders each widget's offscreen framebuffer and streams the result;
it needs a *current* OpenGL context but never a visible window. We prefer GLFW
(the same backend the GUI uses) and fall back to a windowless EGL pbuffer
context when GLFW cannot create one -- typically a headless host with no X
display.

All backends expose the same tiny interface:
    make_current()  -- bind the context to the calling thread
    release()       -- unbind it (so another thread can bind it)
    destroy()       -- free the context and its resources

`make_current`/`release` mirror `glfw.make_context_current`: the context is set
up on the main thread, released, then bound on the websocket server thread.

We request an OpenGL 3.3 *core* profile in both backends. A compatibility
context leaves point sprites off, so `gl_PointCoord` reads (0, 0) in fragment
shaders and widgets that rely on it (e.g. PointRenderer's disc test) discard
every fragment.
"""

OPENGL_VERSION = (3, 3)


class GLFWContext:
    """Hidden-window GLFW context (works when a display is available)."""

    def __init__(self):
        import glfw
        self._glfw = glfw
        self.window = None

        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, OPENGL_VERSION[0])
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, OPENGL_VERSION[1])
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        self.window = glfw.create_window(1920, 1080, "", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed (no display?)")

    def make_current(self):
        self._glfw.make_context_current(self.window)

    def release(self):
        self._glfw.make_context_current(None)

    def destroy(self):
        if self.window is not None:
            self._glfw.destroy_window(self.window)
            self.window = None
        self._glfw.terminate()


def _use_egl_platform():
    """Make PyOpenGL drive the EGL platform for context-data lookups.

    PyOpenGL binds a single platform plugin (GLX by default on Linux) the first
    time `OpenGL.GL` is imported, and that plugin's `GetCurrentContext` must
    match the kind of context we make current. By the time we choose the EGL
    fallback, `OpenGL.GL` has already pulled in the GLX plugin, so
    `glXGetCurrentContext()` reports no current context and context-data lookups
    inside calls like `glVertexAttribPointer` raise "no valid context running in
    EGL mode". Point `PYOPENGL_PLATFORM` at EGL and reload the platform module so
    `GetCurrentContext` re-resolves to `eglGetCurrentContext`.
    """
    import os
    import importlib
    import OpenGL.platform
    if os.environ.get("PYOPENGL_PLATFORM") == "egl":
        return
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    importlib.reload(OpenGL.platform)


class EGLContext:
    """Windowless EGL pbuffer context for headless rendering (no display)."""

    def __init__(self):
        import ctypes
        # Switch PyOpenGL to the EGL platform before any GL call so its
        # context-data tracking sees the EGL context we are about to make
        # current (see `_use_egl_platform`).
        _use_egl_platform()
        from OpenGL import EGL
        self._egl = EGL

        self.display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if self.display == EGL.EGL_NO_DISPLAY:
            raise RuntimeError("eglGetDisplay(EGL_DEFAULT_DISPLAY) returned EGL_NO_DISPLAY")
        major, minor = EGL.EGLint(), EGL.EGLint()
        if not EGL.eglInitialize(self.display, ctypes.byref(major), ctypes.byref(minor)):
            raise RuntimeError("eglInitialize failed")

        # Desktop GL (not GLES), to match the GLFW/GUI context.
        if not EGL.eglBindAPI(EGL.EGL_OPENGL_API):
            raise RuntimeError("eglBindAPI(EGL_OPENGL_API) failed")

        config_attribs = [
            EGL.EGL_SURFACE_TYPE,    EGL.EGL_PBUFFER_BIT,
            EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
            EGL.EGL_RED_SIZE,   8,
            EGL.EGL_GREEN_SIZE, 8,
            EGL.EGL_BLUE_SIZE,  8,
            EGL.EGL_DEPTH_SIZE, 24,
            EGL.EGL_NONE,
        ]
        configs = (EGL.EGLConfig * 1)()
        num = EGL.EGLint()
        ok = EGL.eglChooseConfig(
            self.display,
            (EGL.EGLint * len(config_attribs))(*config_attribs),
            configs, 1, ctypes.byref(num),
        )
        if not ok or num.value == 0:
            raise RuntimeError("eglChooseConfig found no matching config")
        config = configs[0]

        ctx_attribs = [
            EGL.EGL_CONTEXT_MAJOR_VERSION, OPENGL_VERSION[0],
            EGL.EGL_CONTEXT_MINOR_VERSION, OPENGL_VERSION[1],
            EGL.EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL.EGL_NONE,
        ]
        self.context = EGL.eglCreateContext(
            self.display, config, EGL.EGL_NO_CONTEXT,
            (EGL.EGLint * len(ctx_attribs))(*ctx_attribs),
        )
        if self.context == EGL.EGL_NO_CONTEXT:
            raise RuntimeError("eglCreateContext failed")

        # A 1x1 pbuffer satisfies eglMakeCurrent; all real rendering targets the
        # widgets' own FBOs, so the surface size is irrelevant.
        pbuffer_attribs = [EGL.EGL_WIDTH, 1, EGL.EGL_HEIGHT, 1, EGL.EGL_NONE]
        self.surface = EGL.eglCreatePbufferSurface(
            self.display, config,
            (EGL.EGLint * len(pbuffer_attribs))(*pbuffer_attribs),
        )
        if self.surface == EGL.EGL_NO_SURFACE:
            raise RuntimeError("eglCreatePbufferSurface failed")

    def make_current(self):
        if not self._egl.eglMakeCurrent(
            self.display, self.surface, self.surface, self.context
        ):
            raise RuntimeError("eglMakeCurrent failed")

    def release(self):
        EGL = self._egl
        EGL.eglMakeCurrent(
            self.display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT
        )

    def destroy(self):
        EGL = self._egl
        if getattr(self, "surface", None):
            EGL.eglDestroySurface(self.display, self.surface)
        if getattr(self, "context", None):
            EGL.eglDestroyContext(self.display, self.context)
        EGL.eglTerminate(self.display)


def create_headless_context():
    """Create a current-able OpenGL context for SERVER mode.

    Try GLFW first (the GUI's backend); if it cannot create a context (e.g. a
    headless host with no display), fall back to a windowless EGL context.
    """
    try:
        return GLFWContext()
    except Exception as glfw_err:
        try:
            ctx = EGLContext()
        except Exception as egl_err:
            raise RuntimeError(
                "Could not create a headless OpenGL context. "
                f"GLFW failed ({glfw_err}); EGL fallback failed ({egl_err})."
            ) from egl_err
        print(f"INFO: GLFW context unavailable ({glfw_err}); using headless EGL context.")
        return ctx

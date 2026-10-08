import sys
from enum import IntFlag, auto
import numpy as np

# In the browser (Pyodide) the viewer is the ordinary CLIENT on a platform
# without PyOpenGL, glfw, `websockets` or threads; this flag selects the
# browser substitutes for each.
IS_WEB = sys.platform == "emscripten"

try:
    from OpenGL import *
    from OpenGL.GL import glBindTexture, glPixelStorei, glTexImage2D, glTexSubImage2D, \
        GL_TEXTURE_2D, GL_UNPACK_ALIGNMENT, GL_RGBA8, GL_RGB, GL_RGBA, GL_UNSIGNED_BYTE
    HAS_GL = True
except ImportError:
    HAS_GL = False

try:
    from imgui_bundle import imgui, __version__ as imgui_version
    IMGUI_192 = tuple(map(int, imgui_version.split("."))) >= (1,92,0)
except ImportError:
    # Running on server without `imgui_bundle` installed
    IMGUI_192 = False

class ViewerMode(IntFlag):
    LOCAL = auto()
    SERVER = auto()
    CLIENT = auto()

# Aliases for easy access
LOCAL = ViewerMode.LOCAL
CLIENT = ViewerMode.CLIENT
SERVER = ViewerMode.SERVER
LOCAL_SERVER = ViewerMode.LOCAL | ViewerMode.SERVER
LOCAL_CLIENT = ViewerMode.LOCAL | ViewerMode.CLIENT

class Texture2D:
    """
    This is just a struct like class which holds the state of the texture which
    for now is only the resolution and OpenGL ID but more fields my be added as
    needed. It doesn't provide any methods to actually perform operations on the 
    exture, use the OpenGL API for that. It is the responsibility of the
    application to update the state of the texture when and if needed.
    """
    def __init__(self):
        self.res_x = -1
        self.res_y = -1
        # Owning handle of the texture on the web (see `update_rgba`).
        self._gpu = None
        # There was an API change in ImGui v1.92.0 where all texture related functions expect
        # ImTextureRef instead of ImTextureID.
        if IMGUI_192:
            self._id = imgui.ImTextureRef(-1)
        else:
            self._id = -1
    
    @property
    def tex_ref(self):
        """ Use this when passing to imgui.image """
        return self._id

    @property
    def id(self):
        """ Use this when passing to OpenGL related functions """
        if IMGUI_192:
            return self._id._tex_id
        else:
            return self._id

    @id.setter
    def id(self, tex_id):
        if IMGUI_192:
            self._id._tex_id = tex_id
        else:
            self._id = tex_id

    def update_rgba(self, img: np.ndarray):
        """
        Replace the texture contents with `img`, an (H, W, 3|4) uint8 array.
        With PyOpenGL this uploads into the texture `id` the owner created; in
        the browser HelloImGui creates (and owns) the texture, so `id` changes.
        """
        res_y, res_x, channels = img.shape
        if HAS_GL:
            glBindTexture(GL_TEXTURE_2D, self.id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            fmt = GL_RGB if channels == 3 else GL_RGBA
            if res_x != self.res_x or res_y != self.res_y:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, res_x, res_y, 0, fmt, GL_UNSIGNED_BYTE, img)
            else:
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, res_x, res_y, fmt, GL_UNSIGNED_BYTE, img)
            glBindTexture(GL_TEXTURE_2D, 0)
        else:
            from imgui_bundle import hello_imgui
            if channels == 3:
                alpha = np.full((res_y, res_x, 1), 255, dtype=np.uint8)
                img = np.concatenate([img, alpha], axis=-1)
            self._gpu = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(img))
            self.id = self._gpu.texture_id()
        self.res_x = res_x
        self.res_y = res_y

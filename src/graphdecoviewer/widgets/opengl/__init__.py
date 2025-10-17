import numpy as np
from .. import Widget
from ...types import ViewerMode, CLIENT, Texture2D
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

class OpenGLWidget(Widget):
    """
    Virtual class for widgets that require rendering with OpenGL. It handles creation and deletion
    of shaders and renderbuffer. It also implements sending and receiving the renders in remote mode.
    """
    def __init__(self, vertex_shader: str, fragment_shader: str, mode: ViewerMode):
    # If the device doesn't support OpenGL, this will be `False`.
        self.enabled = False

        # Used to avoid sending image if `step` hasn't been called. Make sure to set it `True` in
        # the `step` call, otherwise the render won't be sent to the client.
        self.step_called = False

        self._vert_shader = vertex_shader
        self._frag_shader = fragment_shader
        super().__init__(mode)

    def setup(self):
        """
        Create the textures required for the RenderBufferObject (RBO) and compiles shaders. The RBO
        must be created on the first call to `step` using `self._create_fbo(res_x, res_y)` and on
        each subsequent resolution update.
        """
        self._color_texture = Texture2D()
        self._color_texture.id = glGenTextures(1)
        self._depth_texture = Texture2D() # Technically its a RBO
        self._depth_texture.id = glGenRenderbuffers(1)

        # Compile shaders
        self._shader = compileProgram(
            compileShader(self._vert_shader, GL_VERTEX_SHADER),
            compileShader(self._frag_shader, GL_FRAGMENT_SHADER)
        )

        self._fbo = None # FBO should be created in first call to step

        # Create a dummy FBO because `step` is not called in CLIENT mode
        if self.mode is CLIENT:
            self._create_fbo(1, 1)

    def _create_fbo(self, res_x: int, res_y: int):
        # Create framebuffer
        if self._fbo is not None:
            glDeleteFramebuffers(self._fbo)
        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # Create texture to render to
        self._color_texture.res_x = res_x
        self._color_texture.res_y = res_y
        glBindTexture(GL_TEXTURE_2D, self._color_texture.id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8,
            self._color_texture.res_x, self._color_texture.res_y,
            0, GL_RGBA, GL_UNSIGNED_BYTE, None
        )
        glBindTexture(GL_TEXTURE_2D, 0)
        # Attach texture to framebuffer
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            self._color_texture.id, 0
        )

        # Create depth RBO
        self._depth_texture.res_x = res_x
        self._depth_texture.res_y = res_y
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_texture.id)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
            self._depth_texture.res_x, self._depth_texture.res_y
        )
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        # Attach it framebuffer
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER,
            self._depth_texture.id
        )

        # Verify framebuffer is complete
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "FBO not complete"

        # Unbind the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def destroy(self):
        """
        Deletes the textures, RBOs and shaders. Make sure to call this method from the child
        class after deleting any other OpenGL objects created by it.
        """
        glDeleteTextures(1, int(self._color_texture.id))
        glDeleteRenderbuffers(1, int(self._depth_texture.id))
        if self._fbo is not None:
            glDeleteFramebuffers(1, int(self._fbo))
        glDeleteProgram(self._shader)

    def server_send(self):
        if not self.step_called:
            return None, None
        glBindTexture(GL_TEXTURE_2D, self._color_texture.id)
        arr = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.step_called = False
        return arr, {"shape": (self._color_texture.res_y, self._color_texture.res_x, 3)}

    def client_recv(self, binary, text):
        img = np.frombuffer(binary, dtype=np.uint8).reshape(text["shape"])
        res_y = text["shape"][0]
        res_x = text["shape"][1]
        glBindTexture(GL_TEXTURE_2D, self._color_texture.id)
        if self._color_texture.res_x != res_x  or self._color_texture.res_y != res_y:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, res_x, res_y, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            self._color_texture.res_x = res_x
            self._color_texture.res_y = res_y
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, res_x, res_y, GL_RGB, GL_UNSIGNED_BYTE, img)
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def show_gui(self, draw_list: 'imgui.ImDrawList'=None):
        res_x = self._color_texture.res_x
        res_y = self._color_texture.res_y
        if draw_list is not None:
            # Figure out
            draw_list.add_image(self._color_texture.tex_ref, (0, 0), (res_x, res_y))
        else:
            imgui.image(self._color_texture.tex_ref, (res_x, res_y))

    def import_client_modules(self):
        global imgui
        from imgui_bundle import imgui
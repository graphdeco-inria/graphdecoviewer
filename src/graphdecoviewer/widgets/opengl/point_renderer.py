import ctypes
import numpy as np
from . import OpenGLWidget
from OpenGL.GL import *
from ...types import ViewerMode, Texture2D
from ..cameras import Camera

_vert_shader = """
#version 330 core
layout(location = 0) in vec3 position;   // Vertex position (world space)
layout(location = 1) in vec3 color;      // Per-vertex color

out vec3 fragColor;

uniform mat4 projection;   // Camera projection matrix (OpenCV-style)
uniform mat4 view;         // World -> camera matrix

// `view`/`projection` follow the OpenCV camera convention used by `Camera`:
// x right, y down, +z forward, and projection has clip w = +z. For such a
// camera, world-up maps to negative NDC y, i.e. the bottom of the OpenGL
// framebuffer (texel row 0). When the resulting FBO texture is displayed with
// `imgui.add_image` (uv (0,0) at the top-left), row 0 is sampled at the top of
// the image, so the render comes out upright with no manual Y flip required.
void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    fragColor = color;
}
"""

_frag_shader = """
#version 330 core
in vec3 fragColor;

out vec4 result;

uniform float point_size;     // Sprite diameter in pixels
uniform float border_width;   // Border ring thickness in pixels (0 = none)
uniform vec3  border_color;   // Border ring color

void main() {
    // Draw each point as a filled, camera-facing disc. Fragments outside the
    // disc are discarded so the points look round and the background of the
    // render target stays fully transparent for compositing.
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5)
        discard;
    // Outer `border_width` pixels of the disc are drawn in `border_color`.
    float inner = 0.5 - border_width / max(point_size, 1.0);
    vec3 color = d > inner ? border_color : clamp(fragColor, 0.0, 1.0);
    result = vec4(color, 1.0);
}
"""


class PointRenderer(OpenGLWidget):
    """
    Renders a colored point cloud into an offscreen RGBA texture.

    Each point is drawn as a flat, camera-facing disc colored by its
    per-vertex color, with depth testing so nearer points occlude farther
    ones. The render target's background is left fully transparent, so the
    resulting texture can be composited on top of another image (for example
    via `imgui.ImDrawList.add_image`).

    Coloring is application defined - the widget just draws whatever per-point
    color it is given. To visualize surface normals pass `(normal + 1) / 2`;
    to visualize an albedo, error heatmap, etc. pass that instead.
    """

    def __init__(
        self,
        mode: ViewerMode,
        positions: np.ndarray = None,
        colors: np.ndarray = None,
        border_width: float = 0.0,
        border_color: tuple = (0.0, 0.0, 0.0),
    ):
        self.point_size = 4
        # Border ring drawn around each disc, in pixels (0 disables it).
        self.border_width = border_width
        self.border_color = border_color
        self.num_points = 0
        self._positions = positions
        self._colors = colors
        # Whether the GPU buffer is out of sync with `_positions`/`_colors`.
        self._dirty = positions is not None
        super().__init__(_vert_shader, _frag_shader, mode)

    def setup(self):
        super().setup()

        # Cache uniform locations once (valid after the shader is linked).
        self._uniforms = {
            name: glGetUniformLocation(self._shader, name)
            for name in ("point_size", "border_width", "border_color",
                         "projection", "view")
        }

        # Create VBO / VAO
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        if self._dirty:
            self._upload()
        self.enabled = True

    def update(self, positions: np.ndarray, colors: np.ndarray):
        """
        Replace the point cloud. Safe to call before or after `setup`; if
        OpenGL is not ready yet the upload is deferred to the first `step`.
        """
        self._positions = positions
        self._colors = colors
        if getattr(self, "_vao", None) is not None:
            self._upload()
        else:
            self._dirty = True

    def _upload(self):
        assert self._positions is not None and self._colors is not None, \
            "No point cloud set; call `update(positions, colors)` first."
        positions = np.asarray(self._positions, dtype=np.float32)
        colors = np.asarray(self._colors, dtype=np.float32)
        assert positions.shape == colors.shape, \
            "`positions` and `colors` must have the same shape (N, 3)."
        self.num_points = positions.shape[0]

        # Interleave as [x, y, z, r, g, b] per vertex. `concatenate` returns a
        # fresh C-contiguous float32 array, ready to upload as-is.
        vertex_data = np.concatenate([positions, colors], axis=1)

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        stride = vertex_data.shape[1] * vertex_data.itemsize
        # Position attribute (location 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Color attribute (location 1)
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * vertex_data.itemsize)
        )
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        self._dirty = False

    def step(self, camera: Camera, res_x: int, res_y: int) -> Texture2D:
        """
        Render the point cloud from `camera` into the offscreen texture and
        return it. The texture has a transparent background and can be drawn
        on top of another image at the same resolution.
        """
        if (
            self._fbo is None
            or res_x != self._color_texture.res_x
            or res_y != self._color_texture.res_y
        ):
            self._create_fbo(res_x, res_y)
        if self._dirty:
            self._upload()

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, res_x, res_y)

        # Opaque discs over a transparent background. Depth testing resolves
        # occlusion between points; blending is left off so the disc's alpha is
        # written verbatim (1 inside the disc, the cleared 0 everywhere else).
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.num_points > 0:
            glPointSize(self.point_size)
            glUseProgram(self._shader)

            # Border uniforms
            u = self._uniforms
            glUniform1f(u["point_size"], float(self.point_size))
            glUniform1f(u["border_width"], float(self.border_width))
            glUniform3f(u["border_color"], *self.border_color)

            # Load matrices transposed (GL_TRUE) since NumPy is row-major while
            # OpenGL expects column-major.
            glUniformMatrix4fv(
                u["projection"], 1, GL_TRUE,
                np.ascontiguousarray(camera.projection, dtype=np.float32),
            )
            glUniformMatrix4fv(
                u["view"], 1, GL_TRUE,
                np.ascontiguousarray(camera.to_camera, dtype=np.float32),
            )

            glBindVertexArray(self._vao)
            glDrawArrays(GL_POINTS, 0, self.num_points)
            glBindVertexArray(0)

            glUseProgram(0)
            glPointSize(1)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.step_called = True
        return self._color_texture

    def destroy(self):
        if getattr(self, "_vao", None) is not None:
            glDeleteVertexArrays(1, [self._vao])
            glDeleteBuffers(1, [self._vbo])
        super().destroy()

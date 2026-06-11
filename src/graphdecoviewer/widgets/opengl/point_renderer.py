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

out vec3  fragColor;
out float cameraDist;      // Distance from the camera, for depth testing

uniform mat4 projection;   // Camera projection matrix (OpenCV-style)
uniform mat4 view;         // World -> camera matrix

// `view`/`projection` follow the OpenCV camera convention used by `Camera`:
// x right, y down, +z forward, and projection has clip w = +z. For such a
// camera, world-up maps to negative NDC y, i.e. the bottom of the OpenGL
// framebuffer (texel row 0). When the resulting FBO texture is displayed with
// `imgui.add_image` (uv (0,0) at the top-left), row 0 is sampled at the top of
// the image, so the render comes out upright with no manual Y flip required.
void main() {
    // `view` places the camera at the origin, so the length of the camera-space
    // position is the distance from the camera along the ray to this point. This
    // matches the per-pixel ray distance stored by a depth map, so the two can
    // be compared directly in the fragment shader.
    vec4 viewPos = view * vec4(position, 1.0);
    cameraDist = length(viewPos.xyz);
    gl_Position = projection * viewPos;
    fragColor = color;
}
"""

_frag_shader = """
#version 330 core
in vec3  fragColor;
in float cameraDist;

out vec4 result;

uniform float point_size;     // Sprite diameter in pixels
uniform float border_width;   // Border ring thickness in pixels (0 = none)
uniform vec3  border_color;   // Border ring color

uniform bool      use_depth_test; // Whether to occlude points behind the depth map
uniform sampler2D depth_map;      // Per-pixel scene depth (camera-ray distance)
uniform vec2      resolution;     // Render target size, for sampling depth_map
uniform float     depth_bias;     // Relative slack so on-surface points survive

void main() {
    // Draw each point as a filled, camera-facing disc. Fragments outside the
    // disc are discarded so the points look round and the background of the
    // render target stays fully transparent for compositing.
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5)
        discard;

    // Hide points that lie behind the visible scene surface. `depth_map` stores
    // the camera-ray distance to that surface at each pixel; a value <= 0 means
    // the ray hit nothing (background), so nothing occludes the point there.
    if (use_depth_test) {
        float sceneDist = texture(depth_map, gl_FragCoord.xy / resolution).r;
        if (sceneDist > 0.0 && cameraDist > sceneDist * (1.0 + depth_bias))
            discard;
    }

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

    An optional depth map (see `update_depth_map`) can occlude points that fall
    behind the visible scene surface. Set `depth_test` to enable it; the depth
    map must store the per-pixel camera-ray distance, rendered with the same
    camera as the points.
    """

    def __init__(
        self,
        mode: ViewerMode,
        positions: np.ndarray = None,
        colors: np.ndarray = None,
        border_width: float = 0.0,
        border_color: tuple = (0.0, 0.0, 0.0),
        depth_test: bool = False,
        depth_bias: float = 0.01,
    ):
        self.point_size = 4
        # Border ring drawn around each disc, in pixels (0 disables it).
        self.border_width = border_width
        self.border_color = border_color
        # Occlude points behind the depth map. Only takes effect once a depth
        # map has been supplied via `update_depth_map`. `depth_bias` is the
        # relative slack (fraction of the scene distance) that keeps points lying
        # on the visible surface from being culled by their own depth.
        self.depth_test = depth_test
        self.depth_bias = depth_bias
        self.num_points = 0
        self._positions = positions
        self._colors = colors
        # Whether the GPU buffer is out of sync with `_positions`/`_colors`.
        self._dirty = positions is not None
        # Pending depth map (numpy) and whether it needs (re)uploading.
        self._depth_map = None
        self._depth_dirty = False
        self._depth_res = (0, 0)
        super().__init__(_vert_shader, _frag_shader, mode)

    def setup(self):
        super().setup()

        # Cache uniform locations once (valid after the shader is linked).
        self._uniforms = {
            name: glGetUniformLocation(self._shader, name)
            for name in ("point_size", "border_width", "border_color",
                         "projection", "view",
                         "use_depth_test", "depth_map", "resolution", "depth_bias")
        }

        # Create VBO / VAO
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        if self._dirty:
            self._upload()

        # Texture holding the optional scene depth map (uploaded lazily).
        self._depth_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._depth_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)
        if self._depth_dirty:
            self._upload_depth()
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

    def update_depth_map(self, depth: np.ndarray):
        """
        Set (or clear, with `None`) the scene depth map used to occlude points
        that lie behind the visible surface. `depth` is an (H, W) array of the
        per-pixel camera-ray distance, rendered with the same camera as the
        points. Takes effect only while `depth_test` is set. Safe to call before
        `setup`; the upload is deferred to the first `step`.
        """
        self._depth_map = None if depth is None else \
            np.ascontiguousarray(depth, dtype=np.float32).reshape(depth.shape[:2])
        if getattr(self, "_depth_tex", None) is not None:
            self._upload_depth()
        else:
            self._depth_dirty = True

    def _upload_depth(self):
        glBindTexture(GL_TEXTURE_2D, self._depth_tex)
        if self._depth_map is None:
            self._depth_res = (0, 0)
        else:
            h, w = self._depth_map.shape
            if (w, h) != self._depth_res:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0,
                             GL_RED, GL_FLOAT, self._depth_map)
                self._depth_res = (w, h)
            else:
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                                GL_RED, GL_FLOAT, self._depth_map)
        glBindTexture(GL_TEXTURE_2D, 0)
        self._depth_dirty = False

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
        if self._depth_dirty:
            self._upload_depth()

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

            # Depth test uniforms: only enabled once a depth map is uploaded.
            use_depth = self.depth_test and self._depth_res != (0, 0)
            glUniform1i(u["use_depth_test"], int(use_depth))
            if use_depth:
                glUniform1f(u["depth_bias"], float(self.depth_bias))
                glUniform2f(u["resolution"], float(res_x), float(res_y))
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self._depth_tex)
                glUniform1i(u["depth_map"], 0)

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
        if getattr(self, "_depth_tex", None) is not None:
            glDeleteTextures(1, [self._depth_tex])
        super().destroy()

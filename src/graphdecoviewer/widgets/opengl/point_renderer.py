import ctypes
import numpy as np
from enum import Enum
from . import OpenGLWidget
from OpenGL.GL import *
from ...types import ViewerMode, Texture2D
from ..cameras import Camera

_vert_shader = """
#version 330 core
layout(location = 0) in vec3 position;   // Vertex position
layout(location = 1) in vec3 color; // Vertex color

out vec3 fragColor; // Pass color to fragment shader

uniform mat4 projection; // Projection matrix
uniform mat4 view;       // View matrix

void main() {
    gl_Position = projection * view * vec4(position, 1.0);
    gl_Position.y = -gl_Position.y;
    fragColor = (normal+1) / 2;
}
"""

_frag_shader = """
#version 330 core
in vec3 fragColor;

out vec4 result;

void main() {
    // Render points as circles
    if (length(gl_PointCoord - 0.5) <= 0.5) {
        vec3 output = clamp(fragColor, 0, 1);
        output = pow(output, vec3(1 / 2.2));
        result = vec4(output, 1.);
    } else {
        result = vec4(0.0);
    }
}
"""

class PointRenderMode(Enum):
    radiance = 0
    normal = 1

class PointRenderer(OpenGLWidget):
    def __init__(self, positions: np.ndarray, colors: np.ndarray, mode: ViewerMode):
        self.point_size = 5
        self._positions = positions
        self._colors = colors
        super().__init__(_vert_shader, _frag_shader, mode)

    def setup(self):
        try:
            super().setup()

            # Enable depth testing
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)

            # Create VBO / VAO
            self._vao = glGenVertexArrays(1)
            self._vbo = glGenBuffers(1)
            self._upload()
            self.enabled = True
        except Exception as e:
            print(f"Error setting up PointRenderer: {e}")


    def _upload(self):
        vertex_data = np.concatenate([self._positions, self._colors], axis=1)
        vertex_data = np.ascontiguousarray(vertex_data)
        self._vertex_data = vertex_data

        # Bind VAO / VBO
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # Upload data to buffer
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # Configure vertex attribute for position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_data[0].nbytes, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Configure vertex attribute for color
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_data[0].nbytes, ctypes.c_void_p(3 * vertex_data.itemsize))
        glEnableVertexAttribArray(1)

        # Unbind VAO / VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
    
    def update(self, positions: np.ndarray, colors: np.ndarray):
        self._positions = positions
        self._colors = colors
        self._upload()

    def step(self, camera: Camera, wi: np.ndarray, res_x: int, res_y: int) -> Texture2D:
        if res_x != self._color_texture.res_x or res_y != self._color_texture.res_y:
            # Recreate FBO
            self._create_fbo(res_x, res_y)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        glPointSize(self.point_size)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Use the shader program
        glUseProgram(self._shader)

        # Pass the matrices to the shader
        # Load the matrices as transpose because OpenGL expects them in column major
        glUniformMatrix4fv(glGetUniformLocation(self._shader, "projection"),
                           1, GL_TRUE, camera.projection(res_y/res_x))
        glUniformMatrix4fv(glGetUniformLocation(self._shader, "view"),
                           1, GL_TRUE, camera.to_camera)
        glUniform3fv(glGetUniformLocation(self._shader, "wi"),
                    1, wi)
        glUniform1i(glGetUniformLocation(self._shader, "mode"), self.mode.value)

        # Bind the VAO and draw the points
        glBindVertexArray(self._vao)
        glDrawArrays(GL_POINTS, 0, self.num_points)
        glBindVertexArray(0)

        # Unbind program
        glUseProgram(0)
        glPointSize(1)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
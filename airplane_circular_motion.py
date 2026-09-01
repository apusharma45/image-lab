"""
3D Airplane Flying in a Circular Path
--------------------------------------
Dependencies:
    pip install pygame PyOpenGL numpy

Controls:
    SPACE      Pause / resume
    UP         Increase speed
    DOWN       Decrease speed
    R          Reset
    ESC        Exit

The important transformation is:

    M_airplane = T(position) @ Ry(heading) @ Rz(bank)

A local airplane vertex becomes a world-space vertex by:

    P_world = M_airplane @ P_local
"""

import math
import sys

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *


# -----------------------------
# Matrix functions
# -----------------------------

def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def scaling_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def rotation_y_matrix(angle_degrees: float) -> np.ndarray:
    theta = math.radians(angle_degrees)
    c = math.cos(theta)
    s = math.sin(theta)

    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float32)


def rotation_z_matrix(angle_degrees: float) -> np.ndarray:
    theta = math.radians(angle_degrees)
    c = math.cos(theta)
    s = math.sin(theta)

    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=np.float32)


def normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / length


def look_at_matrix(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray
) -> np.ndarray:
    """
    Builds a world-to-camera viewing matrix.

    n = direction from target to eye
    u = camera-right direction
    v = camera-up direction
    """
    n = normalize(eye - target)
    u = normalize(np.cross(up, n))
    v = np.cross(n, u)

    return np.array([
        [u[0], u[1], u[2], -np.dot(u, eye)],
        [v[0], v[1], v[2], -np.dot(v, eye)],
        [n[0], n[1], n[2], -np.dot(n, eye)],
        [0,    0,    0,     1],
    ], dtype=np.float32)


def perspective_matrix(
    fov_degrees: float,
    aspect: float,
    near: float,
    far: float
) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)

    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far),
         (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


# -----------------------------
# Basic geometry
# -----------------------------

CUBE_VERTICES = [
    (-0.5, -0.5, -0.5),
    ( 0.5, -0.5, -0.5),
    ( 0.5,  0.5, -0.5),
    (-0.5,  0.5, -0.5),
    (-0.5, -0.5,  0.5),
    ( 0.5, -0.5,  0.5),
    ( 0.5,  0.5,  0.5),
    (-0.5,  0.5,  0.5),
]

CUBE_FACES = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (1, 2, 6, 5),
    (0, 3, 7, 4),
]

FACE_COLORS = [
    (0.80, 0.80, 0.86),
    (0.65, 0.68, 0.75),
    (0.90, 0.90, 0.95),
    (0.55, 0.58, 0.65),
    (0.72, 0.75, 0.82),
    (0.60, 0.63, 0.70),
]


def load_model_view(matrix: np.ndarray) -> None:
    """
    NumPy stores matrices in row-major order.
    OpenGL expects column-major order, so we pass matrix.T.
    """
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(matrix.T)


def draw_unit_cube() -> None:
    glBegin(GL_QUADS)
    for face_index, face in enumerate(CUBE_FACES):
        glColor3fv(FACE_COLORS[face_index])
        for vertex_index in face:
            glVertex3fv(CUBE_VERTICES[vertex_index])
    glEnd()


def draw_component(
    view_matrix: np.ndarray,
    airplane_matrix: np.ndarray,
    local_translation: tuple[float, float, float],
    local_scale: tuple[float, float, float],
    color: tuple[float, float, float] | None = None,
) -> None:
    local_matrix = (
        translation_matrix(*local_translation)
        @ scaling_matrix(*local_scale)
    )

    model_matrix = airplane_matrix @ local_matrix
    load_model_view(view_matrix @ model_matrix)

    if color is not None:
        glColor3fv(color)

    draw_unit_cube()


def draw_airplane(view_matrix: np.ndarray, airplane_matrix: np.ndarray) -> None:
    """
    The airplane points in the +Z direction in its local coordinate system.
    It is made from simple cuboids so the transformations stay easy to explain.
    """

    # Fuselage
    draw_component(
        view_matrix,
        airplane_matrix,
        local_translation=(0.0, 0.0, 0.0),
        local_scale=(0.75, 0.65, 4.0),
    )

    # Main wings
    draw_component(
        view_matrix,
        airplane_matrix,
        local_translation=(0.0, 0.0, 0.25),
        local_scale=(5.0, 0.16, 1.1),
    )

    # Horizontal tail
    draw_component(
        view_matrix,
        airplane_matrix,
        local_translation=(0.0, 0.15, -1.65),
        local_scale=(2.2, 0.12, 0.65),
    )

    # Vertical tail
    draw_component(
        view_matrix,
        airplane_matrix,
        local_translation=(0.0, 0.55, -1.65),
        local_scale=(0.16, 1.15, 0.65),
    )

    # Nose
    draw_component(
        view_matrix,
        airplane_matrix,
        local_translation=(0.0, 0.0, 2.2),
        local_scale=(0.48, 0.48, 0.55),
    )


def draw_axes(view_matrix: np.ndarray, length: float = 5.0) -> None:
    load_model_view(view_matrix)
    glLineWidth(2.0)

    glBegin(GL_LINES)

    # X axis
    glColor3f(1.0, 0.2, 0.2)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)

    # Y axis
    glColor3f(0.2, 1.0, 0.2)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)

    # Z axis
    glColor3f(0.2, 0.4, 1.0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, length)

    glEnd()


def draw_circular_path(view_matrix: np.ndarray, radius: float) -> None:
    load_model_view(view_matrix)
    glColor3f(0.8, 0.8, 0.2)
    glLineWidth(2.0)

    glBegin(GL_LINE_LOOP)
    for degree in range(360):
        theta = math.radians(degree)
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        glVertex3f(x, 0.0, z)
    glEnd()


# -----------------------------
# Main simulation
# -----------------------------

def main() -> None:
    pygame.init()

    width, height = 1000, 700
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(
        "3D Airplane Circular Motion — SPACE pause, UP/DOWN speed"
    )

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.07, 0.12, 1.0)

    projection = perspective_matrix(
        fov_degrees=55.0,
        aspect=width / height,
        near=0.1,
        far=100.0,
    )

    glMatrixMode(GL_PROJECTION)
    glLoadMatrixf(projection.T)

    eye = np.array([18.0, 13.0, 18.0], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view = look_at_matrix(eye, target, up)

    clock = pygame.time.Clock()

    radius = 8.0
    angle_degrees = 0.0
    angular_speed = 25.0       # degrees per second
    bank_angle = -18.0         # fixed roll while turning
    paused = False
    running = True
    print_timer = 0.0

    while running:
        delta_time = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    angular_speed += 5.0
                elif event.key == pygame.K_DOWN:
                    angular_speed = max(0.0, angular_speed - 5.0)
                elif event.key == pygame.K_r:
                    angle_degrees = 0.0
                    angular_speed = 25.0
                    paused = False

        if not paused:
            angle_degrees += angular_speed * delta_time
            angle_degrees %= 360.0

        theta = math.radians(angle_degrees)

        # Circular position:
        # x = r cos(theta)
        # z = r sin(theta)
        x = radius * math.cos(theta)
        y = 1.5
        z = radius * math.sin(theta)

        # The airplane model points toward local +Z.
        # The tangent direction of the circle is:
        # (-sin(theta), 0, cos(theta))
        #
        # A Y rotation of -theta aligns local +Z with that tangent.
        heading_degrees = -angle_degrees

        translation = translation_matrix(x, y, z)
        heading = rotation_y_matrix(heading_degrees)
        bank = rotation_z_matrix(bank_angle)

        # Right-to-left application:
        # 1. Bank the airplane
        # 2. Turn it toward the tangent direction
        # 3. Move it to the circular-path position
        airplane_model = translation @ heading @ bank

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_axes(view)
        draw_circular_path(view, radius)
        draw_airplane(view, airplane_model)

        pygame.display.flip()

        print_timer += delta_time
        if print_timer >= 1.0:
            print_timer = 0.0
            print("\n----------------------------------------")
            print(f"Orbit angle   : {angle_degrees:7.2f} degrees")
            print(f"Position      : ({x:6.2f}, {y:6.2f}, {z:6.2f})")
            print(f"Heading       : {heading_degrees:7.2f} degrees")
            print(f"Angular speed : {angular_speed:7.2f} degrees/second")
            print("Airplane model matrix:")
            print(np.round(airplane_model, 3))

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

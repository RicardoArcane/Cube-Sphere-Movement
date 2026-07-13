import cv2
import mediapipe as mp
import math
import numpy as np
import sys
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("Starting...")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

model_path = resource_path("hand_landmarker.task")

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("4D Cube Control", cv2.WINDOW_NORMAL)

frame_timestamp = 0
angle = 0

sphere_spawned = False

BOUNCE = 0.85
THROW_MULT = 2.8

cube_pos = [300.0, 300.0]
cube_size = 120.0

cube_grab = False
cube_offset = [0.0, 0.0]
cube_init_dist = None
cube_init_size = cube_size

cube_vel = [0.0, 0.0]

sphere = {
    "exists": False,
    "pos": [450.0, 300.0],
    "size": 60.0,
    "grab": False,
    "offset": [0.0, 0.0],
    "init_dist": None,
    "init_size": 60.0,
    "vel": [0.0, 0.0]
}

smooth = 0.35


def dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def is_open_hand(lm):
    return (
        finger_up(lm, 8, 6) and
        finger_up(lm, 12, 10) and
        finger_up(lm, 16, 14) and
        finger_up(lm, 20, 18)
    )


def inside(pos, size, p):
    x, y = pos
    s = size
    return (x - s/2 < p[0] < x + s/2 and y - s/2 < p[1] < y + s/2)


def apply_physics(pos, vel):
    pos[0] += vel[0]
    pos[1] += vel[1]
    vel[0] *= 0.92
    vel[1] *= 0.92


def bounce(pos, vel, w, h, r):
    if pos[0] - r < 0:
        pos[0] = r
        vel[0] *= -BOUNCE
    if pos[0] + r > w:
        pos[0] = w - r
        vel[0] *= -BOUNCE
    if pos[1] - r < 0:
        pos[1] = r
        vel[1] *= -BOUNCE
    if pos[1] + r > h:
        pos[1] = h - r
        vel[1] *= -BOUNCE


vertices = np.array([
    [x, y, z, w]
    for x in (-1, 1)
    for y in (-1, 1)
    for z in (-1, 1)
    for w in (-1, 1)
], dtype=float)

edges = []
for i in range(len(vertices)):
    for j in range(i + 1, len(vertices)):
        if np.sum(np.abs(vertices[i] - vertices[j])) == 2:
            edges.append((i, j))


def rotate4d(p, a, b, t):
    p = p.copy()
    c = math.cos(t)
    s = math.sin(t)
    i, j = a, b
    pi, pj = p[i], p[j]
    p[i] = pi * c - pj * s
    p[j] = pi * s + pj * c
    return p


def project(p):
    w = p[3]
    f = 3 / (3 - w)
    return np.array([p[0], p[1], p[2]]) * f


def to2d(p, scale, cx, cy):
    return int(p[0] * scale + cx), int(p[1] * scale + cy)


def draw_tesseract(frame, x, y, size, t):
    pts = []

    for v in vertices:
        v = rotate4d(v, 0, 3, t)
        v = rotate4d(v, 1, 3, t * 0.7)
        v = rotate4d(v, 0, 1, t * 0.4)
        v = project(v)
        pts.append(to2d(v, size / 4, x, y))

    glow = frame.copy()

    for i, j in edges:
        cv2.line(glow, pts[i], pts[j], (0, 180, 255), 8)
        cv2.line(glow, pts[i], pts[j], (0, 120, 255), 4)

    frame[:] = cv2.addWeighted(glow, 0.22, frame, 0.78, 0)

    for i, j in edges:
        cv2.line(frame, pts[i], pts[j], (0, 230, 255), 2)


def draw_sphere(frame, obj, t):
    cx, cy = int(obj["pos"][0]), int(obj["pos"][1])
    r = int(obj["size"])

    glow = frame.copy()

    lx, ly = math.cos(t), math.sin(t)

    for i in range(-3, 4):
        theta = i * math.pi / 8 + t * 0.2
        prev = None

        for j in range(0, 360, 15):
            phi = math.radians(j)

            x = cx + r * math.cos(phi) * math.cos(theta)
            y = cy + r * math.sin(theta)

            shade = max(0.2, (math.cos(phi) * lx + math.sin(theta) * ly + 1) / 2)
            col = int(120 + 135 * shade)

            pt = (int(x), int(y))

            if prev:
                cv2.line(glow, prev, pt, (col, col, 255), 1)
            prev = pt

    for i in range(0, 360, 20):
        phi = math.radians(i + t * 30)
        prev = None

        for j in range(-80, 81, 10):
            theta = math.radians(j)

            x = cx + r * math.cos(theta) * math.cos(phi)
            y = cy + r * math.sin(theta)

            shade = max(0.2, (math.cos(phi) * lx + math.sin(theta) * ly + 1) / 2)
            col = int(120 + 135 * shade)

            pt = (int(x), int(y))

            if prev:
                cv2.line(glow, prev, pt, (255, col, col), 1)
            prev = pt

    frame[:] = cv2.addWeighted(glow, 0.18, frame, 0.82, 0)

    cv2.circle(frame, (cx, cy), r, (255, 230, 180), 2)

    cv2.circle(frame, (cx, cy), int(r * 0.6), (255, 180, 120), 1)
def draw_hud(frame):
    h, w, _ = frame.shape
    x0 = w - 260

    cv2.putText(frame, "4D SYSTEM", (x0, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    cv2.putText(frame, f"CUBE: {int(cube_pos[0])},{int(cube_pos[1])}", (x0, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"SPHERE: {sphere['exists']}", (x0, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 100), 1)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = landmarker.detect_for_video(mp_image, frame_timestamp)
        frame_timestamp += 1

        hands = []

        if result.hand_landmarks:
            for i, lm in enumerate(result.hand_landmarks):

                index = [int(lm[8].x * w), int(lm[8].y * h)]
                thumb = [int(lm[4].x * w), int(lm[4].y * h)]

                pinch = dist(thumb, index) < 60
                openh = is_open_hand(lm)

                label = result.handedness[i][0].category_name

                hands.append({
                    "pos": index,
                    "pinch": pinch,
                    "open": openh,
                    "label": label
                })

                cv2.circle(frame, tuple(index), 6, (0, 255, 255), -1)

        active = [h for h in hands if h["pinch"]]

        left_open = None
        for hnd in hands:
            if hnd["label"] == "Left" and hnd["open"]:
                left_open = hnd["pos"]
                break

        if left_open is not None and not sphere_spawned:
            sphere_spawned = True
            sphere["exists"] = True
            sphere["pos"] = left_open[:]

        if len(active) == 1:
            hand = active[0]

            if not cube_grab:
                if inside(cube_pos, cube_size, hand["pos"]):
                    cube_grab = True
                    cube_offset[0] = cube_pos[0] - hand["pos"][0]
                    cube_offset[1] = cube_pos[1] - hand["pos"][1]

            if cube_grab:
                cube_vel[0] = (hand["pos"][0] + cube_offset[0] - cube_pos[0]) * 0.25
                cube_vel[1] = (hand["pos"][1] + cube_offset[1] - cube_pos[1]) * 0.25

            cube_init_dist = None

        elif len(active) >= 2:
            p1, p2 = active[0]["pos"], active[1]["pos"]

            if cube_init_dist is None:
                if inside(cube_pos, cube_size, p1) or inside(cube_pos, cube_size, p2):
                    cube_init_dist = dist(p1, p2)
                    cube_init_size = cube_size

            if cube_init_dist:
                scale = dist(p1, p2) / cube_init_dist
                cube_size = max(40, min(400, cube_init_size * scale))
                cube_pos[0] = (p1[0] + p2[0]) // 2
                cube_pos[1] = (p1[1] + p2[1]) // 2

            cube_grab = False

        else:
            cube_grab = False
            cube_init_dist = None

        apply_physics(cube_pos, cube_vel)

        if sphere["exists"]:

            if len(active) == 1:
                hand = active[0]

                if not sphere["grab"]:
                    if inside(sphere["pos"], sphere["size"], hand["pos"]):
                        sphere["grab"] = True
                        sphere["offset"][0] = sphere["pos"][0] - hand["pos"][0]
                        sphere["offset"][1] = sphere["pos"][1] - hand["pos"][1]

                if sphere["grab"]:
                    tx = hand["pos"][0] + sphere["offset"][0]
                    ty = hand["pos"][1] + sphere["offset"][1]

                    sphere["vel"][0] = (tx - sphere["pos"][0]) * 0.25
                    sphere["vel"][1] = (ty - sphere["pos"][1]) * 0.25

                sphere["init_dist"] = None

            elif len(active) >= 2:
                p1, p2 = active[0]["pos"], active[1]["pos"]

                if sphere["init_dist"] is None:
                    if inside(sphere["pos"], sphere["size"], p1) or inside(sphere["pos"], sphere["size"], p2):
                        sphere["init_dist"] = dist(p1, p2)
                        sphere["init_size"] = sphere["size"]

                if sphere["init_dist"]:
                    scale = dist(p1, p2) / sphere["init_dist"]
                    sphere["size"] = max(20, min(200, sphere["init_size"] * scale))
                    sphere["pos"][0] = (p1[0] + p2[0]) // 2
                    sphere["pos"][1] = (p1[1] + p2[1]) // 2

                sphere["grab"] = False

            else:
                sphere["grab"] = False
                sphere["init_dist"] = None

            apply_physics(sphere["pos"], sphere["vel"])

        h, w = frame.shape[:2]
        bounce(cube_pos, cube_vel, w, h, cube_size / 2)

        if sphere["exists"]:
            bounce(sphere["pos"], sphere["vel"], w, h, sphere["size"] / 2)

        angle += 0.03

        draw_tesseract(frame, int(cube_pos[0]), int(cube_pos[1]), int(cube_size), angle)

        if sphere["exists"]:
            draw_sphere(frame, sphere, angle)

        draw_hud(frame)

        cv2.imshow("4D Cube Control", frame)

        if cv2.waitKey(1) in [27, 13]:
            break

except Exception as e:
    print("CRASH:", e)

cap.release()
cv2.destroyAllWindows()

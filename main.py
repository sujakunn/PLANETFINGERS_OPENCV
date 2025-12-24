import cv2
import tkinter as tk
from PIL import Image, ImageTk
import math
import numpy as np
import random
from collections import deque

import mediapipe as _mp
mp_hands = _mp.solutions.hands
mp_draw = _mp.solutions.drawing_utils

class PlanetMultiverseApp:
    def __init__(self, window, title):
        self.window = window
        self.window.title(title)

        self.WIDTH, self.HEIGHT = 700, 520
        self.window.geometry(f"{self.WIDTH}x{self.HEIGHT}")

        # ===== MEDIAPIPE =====
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ===== PLANETS =====
        self.planet_names = [
            "Matahari", "Merkurius", "Venus", "Bumi",
            "Mars", "Jupiter", "Saturnus", "Uranus", "Neptunus"
        ]
        files = [
            "matahari.png", "merkurius.png", "venus.png",
            "bumi.png", "mars.png", "jupiter.png",
            "saturnus.png", "uranus.png", "neptunus.png"
        ]

        self.planets = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                self.planets.append(img)

        # ===== STATES =====
        self.angle = 0
        self.pulse = 0
        self.finger_history = deque(maxlen=8)

        # ===== STARS =====
        self.stars = [(random.randint(0, self.WIDTH),
                       random.randint(0, self.HEIGHT),
                       random.randint(1, 3)) for _ in range(120)]

        # ===== EXPLOSION & PORTAL =====
        self.explosions = []
        self.portal_angle = 0

        self.canvas = tk.Canvas(window, width=self.WIDTH, height=self.HEIGHT, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    # ================= FINGER COUNT =================
    def count_fingers(self, hand, handedness):
        tips = [4, 8, 12, 16, 20]
        count = 0
        if handedness == "Right":
            if hand.landmark[4].x < hand.landmark[3].x:
                count += 1
        else:
            if hand.landmark[4].x > hand.landmark[3].x:
                count += 1
        for i in range(1, 5):
            if hand.landmark[tips[i]].y < hand.landmark[tips[i] - 2].y:
                count += 1
        return count

    # ================= OVERLAY PNG =================
    def overlay_png(self, bg, png, x, y, size=90):
        png = cv2.resize(png, (size, size))
        h, w = png.shape[:2]
        if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
            return
        alpha = png[:, :, 3:] / 255.0
        bg[y:y + h, x:x + w] = (
            (1 - alpha) * bg[y:y + h, x:x + w] +
            alpha * png[:, :, :3]
        ).astype(np.uint8)

    # ================= VISUAL FX =================
    def draw_stars(self, frame, warp=False):
        speed = 6 if warp else 2
        for i, (x, y, s) in enumerate(self.stars):
            cv2.circle(frame, (x, y), s, (255, 255, 255), -1)
            y += s * speed
            if y > self.HEIGHT:
                y = 0
                x = random.randint(0, self.WIDTH)
            self.stars[i] = (x, y, s)

    def spawn_explosion(self, cx, cy):
        for _ in range(80):
            self.explosions.append([
                cx, cy,
                random.uniform(-8, 8),
                random.uniform(-8, 8),
                random.randint(6, 14)
            ])

    def draw_explosion(self, frame):
        for p in self.explosions[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 0.4
            if p[4] <= 0:
                self.explosions.remove(p)
                continue
            cv2.circle(frame, (int(p[0]), int(p[1])), int(p[4]), (255, 0, 255), -1)

    def draw_portal(self, frame, cx, cy):
        for i in range(10):
            r = 30 + i * 6
            angle = self.portal_angle + i
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            cv2.circle(frame, (x, y), 2, (180, 0, 255), -1)
        self.portal_angle += 0.2

    def draw_text_fx(self, frame, text):
        alpha = int(200 * abs(math.sin(self.angle)))
        overlay = frame.copy()
        cv2.putText(
            overlay, text, (220, 90),
            cv2.FONT_HERSHEY_DUPLEX, 1.6,
            (255, 0, 255), 3
        )
        cv2.addWeighted(overlay, alpha / 255, frame, 1 - alpha / 255, 0, frame)

    # ================= MAIN LOOP =================
    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(30, self.update)
            return

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        palms = {}
        total_fingers = 0

        if result.multi_hand_landmarks:
            for i, hand in enumerate(result.multi_hand_landmarks):
                handed = result.multi_handedness[i].classification[0].label
                fingers = self.count_fingers(hand, handed)
                total_fingers += fingers

                cx = int(hand.landmark[9].x * self.WIDTH)
                cy = int(hand.landmark[9].y * self.HEIGHT)
                palms[handed] = (cx, cy)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # ===== MULTIVERSE MODE =====
        if total_fingers == 10 and len(palms) == 2:
            self.draw_stars(frame, warp=True)
            (xL, yL), (xR, yR) = palms.values()
            mx, my = (xL + xR) // 2, (yL + yR) // 2
            a = abs(xR - xL) // 2

            self.draw_portal(frame, mx, my)
            if not self.explosions:
                self.spawn_explosion(mx, my)

            for i, planet in enumerate(self.planets):
                t = self.angle + i * (2 * math.pi / len(self.planets))
                x = int(mx + a * math.cos(t))
                y = int(my + (a / 2) * math.sin(2 * t))
                pulse = int(8 * abs(math.sin(self.pulse)))
                self.overlay_png(frame, planet, x - 45 - pulse // 2, y - 45 - pulse // 2, 90 + pulse)
                cv2.putText(frame, self.planet_names[i], (x - 35, y + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            self.draw_text_fx(frame, "MULTIVERSE GOD MODE")

        # ===== NORMAL MODE =====
        else:
            self.draw_stars(frame)
            self.finger_history.append(total_fingers)
            show = min(max(self.finger_history, default=0), len(self.planets))
            if palms:
                (cx, cy) = list(palms.values())[0]
                for i in range(show):
                    ang = self.angle + i * (2 * math.pi / show)
                    ox = int(cx + 90 * math.cos(ang))
                    oy = int(cy + 90 * math.sin(ang))
                    self.overlay_png(frame, self.planets[i], ox - 45, oy - 45)
                    cv2.putText(frame, self.planet_names[i], (ox - 30, oy + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        self.draw_explosion(frame)

        self.angle += 0.04
        self.pulse += 0.08

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        self.canvas.image = img
        self.window.after(30, self.update)

    def on_close(self):
        self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    PlanetMultiverseApp(tk.Tk(), "AR Planet Multiverse — GOD MODE")

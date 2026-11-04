"""
GTC RACING - juego de carreras pseudo-3D estilo Doom (raycasting) en Python.

Controles:
  Flechas / WASD : acelerar, frenar/marcha atras, girar
  ESPACIO        : freno de mano (derrape)
  M              : silenciar / activar sonido
  F11            : pantalla completa
  R              : reiniciar carrera
  ESC            : volver al menu / salir
"""
import math
import random
import sys

import numpy as np
import pygame

# ---------------------------------------------------------------- configuracion
W, H = 320, 168                 # resolucion interna del render 3D
SCALE = 3
VIEW_W, VIEW_H = W * SCALE, H * SCALE
BAR_H = 96
SW, SH = VIEW_W, VIEW_H + BAR_H
PLANE = W / (2 * H)             # plano de camara (pixeles cuadrados)
TEX = 64
LAPS = 3
MAP_N = 40
TRACK_R = 2.0
FOG = np.array([175, 195, 215], np.float32)
FOG_DIST = 22.0
TAU = math.tau

ACCEL, BRAKE, REVERSE_MAX = 8.0, 16.0, 3.5
COAST, DRAG, STEER_RATE = 0.5, 0.004, 2.5
PLAYER_VMAX = 12.0
KMH = 18.0

SPR_W, SPR_H = 64, 32
SPR_WORLD_W, SPR_WORLD_H = 0.9, 0.45

WAYPOINTS = [(14, 4.5), (22, 4), (31, 5), (35, 9), (35, 15), (29, 19), (28, 23),
             (33, 27), (34, 33), (30, 36), (20, 36), (14, 32), (12, 27), (7, 30),
             (4, 26), (4, 18), (8, 13), (4, 9), (6, 5)]

rng = np.random.default_rng(7)


def clamp(v, a, b):
    return a if v < a else b if v > b else v


def wrap_angle(a):
    return (a + math.pi) % TAU - math.pi


def fmt_time(t):
    if t is None:
        return "--:--.--"
    m = int(t // 60)
    return f"{m}:{t - m * 60:05.2f}"


# ---------------------------------------------------------------- pista
class Track:
    def __init__(self, pts):
        self.A = np.array(pts, float)
        self.D = np.roll(self.A, -1, axis=0) - self.A
        self.len = np.hypot(self.D[:, 0], self.D[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.len)[:-1]])
        self.L = float(self.len.sum())

    def project(self, x, y):
        """Devuelve la distancia recorrida (s) del punto mas cercano en la pista."""
        px, py = x - self.A[:, 0], y - self.A[:, 1]
        t = np.clip((px * self.D[:, 0] + py * self.D[:, 1]) / self.len ** 2, 0, 1)
        cx = self.D[:, 0] * t - px
        cy = self.D[:, 1] * t - py
        d = cx * cx + cy * cy
        i = int(np.argmin(d))
        return float(self.cum[i] + t[i] * self.len[i]), math.sqrt(d[i])

    def point(self, s):
        s %= self.L
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        t = (s - self.cum[i]) / self.len[i]
        p = self.A[i] + self.D[i] * t
        dv = self.D[i] / self.len[i]
        return p[0], p[1], dv[0], dv[1]

    def dist_field(self, gx, gy):
        best = np.full(gx.shape, 1e9)
        for a, dv, l in zip(self.A, self.D, self.len):
            t = np.clip(((gx - a[0]) * dv[0] + (gy - a[1]) * dv[1]) / l ** 2, 0, 1)
            best = np.minimum(best, np.hypot(gx - a[0] - dv[0] * t, gy - a[1] - dv[1] * t))
        return best


class World:
    def __init__(self, track):
        ys, xs = np.mgrid[0:MAP_N, 0:MAP_N]
        d = track.dist_field(xs + 0.5, ys + 0.5)
        open_ = d < TRACK_R
        open_[0, :] = open_[-1, :] = open_[:, 0] = open_[:, -1] = False
        r = rng.random((MAP_N, MAP_N))
        wall = np.ones((MAP_N, MAP_N), np.int32)          # 1 = neumaticos
        wall[r < 0.20] = 2                                # 2 = cartel GTC
        wall[(r >= 0.20) & (r < 0.32)] = 4                # 4 = muro rojo/blanco
        wall[(r >= 0.32) & (r < 0.40)] = 3                # 3 = ladrillo
        wall[open_] = 0
        floor = np.full((MAP_N, MAP_N), 3, np.int32)      # 3 = cesped
        floor[open_] = 0                                  # 0 = asfalto
        floor[open_ & (d > TRACK_R - 0.65)] = 2           # 2 = piano (curb)
        fx = int(track.A[0, 0])
        floor[open_ & (xs == fx) & (ys < 10)] = 1         # 1 = meta
        self.wall, self.floor, self.open = wall, floor, open_

    def solid(self, x, y, r):
        for ox in (-r, r):
            for oy in (-r, r):
                cx, cy = int(x + ox), int(y + oy)
                if cx < 0 or cy < 0 or cx >= MAP_N or cy >= MAP_N or self.wall[cy, cx] > 0:
                    return True
        return False


# ---------------------------------------------------------------- texturas
def make_textures():
    n = TEX
    yy, xx = np.mgrid[0:n, 0:n]

    def noise(s):
        return rng.normal(0, s, (n, n))[..., None]

    walls = np.zeros((5, n, n, 3), np.float32)

    # 1: barrera de neumaticos
    ry = yy // 16
    lx = (xx + 8 * (ry % 2)) % 16 - 7.5
    ly = yy % 16 - 7.5
    rr = np.hypot(lx, ly)
    idx = (xx + 8 * (ry % 2)) // 16 + ry
    paint = np.where((idx % 2)[..., None] == 0, [[[215, 215, 215]]], [[[200, 35, 35]]])
    tire = np.full((n, n, 3), 22.0)
    ring = (rr < 7.6) & (rr >= 3.3)
    tire[ring] = paint[ring] * (0.75 + 0.25 * (-ly[ring] / 8))[:, None]
    tire[rr < 3.3] = 8
    walls[1] = np.clip(tire + noise(6), 0, 255)

    # 2: cartel publicitario GTC
    s = pygame.Surface((n, n))
    s.fill((20, 45, 150))
    pygame.draw.rect(s, (250, 200, 30), (0, 0, n, n), 3)
    pygame.draw.rect(s, (250, 200, 30), (0, 46, n, 3))
    f = pygame.font.SysFont("arialblack,impact,arial", 22, bold=True)
    t = f.render("GTC", True, (255, 215, 40))
    s.blit(t, ((n - t.get_width()) // 2, 8))
    f2 = pygame.font.SysFont("arial", 10, bold=True)
    t2 = f2.render("RACING", True, (255, 255, 255))
    s.blit(t2, ((n - t2.get_width()) // 2, 50))
    walls[2] = pygame.surfarray.array3d(s).transpose(1, 0, 2)

    # 3: ladrillo
    row = yy // 16
    mort = (yy % 16 < 2) | ((xx + 16 * (row % 2)) % 32 < 2)
    brick = np.where(mort[..., None], [[[150, 140, 130]]], [[[150, 60, 40]]]).astype(np.float32)
    brick += noise(10)
    walls[3] = np.clip(brick, 0, 255)

    # 4: muro rojo/blanco
    stripe = ((xx // 16) % 2 == 0)[..., None]
    wallc = np.where(stripe, [[[220, 220, 220]]], [[[200, 30, 30]]]).astype(np.float32)
    band = ((yy < 6) | (yy > 57))[..., None]
    wallc = np.where(band, [[[90, 90, 95]]], wallc) + noise(5)
    walls[4] = np.clip(wallc, 0, 255)

    floors = np.zeros((4, n, n, 3), np.float32)
    asph = 72 + rng.normal(0, 7, (n, n))
    asph[rng.random((n, n)) < 0.03] += 30
    floors[0] = np.clip(asph[..., None] * [1.0, 1.0, 1.05], 0, 255)
    chk = ((xx // 8 + yy // 8) % 2 == 0)[..., None]
    floors[1] = np.where(chk, [[[235, 235, 235]]], [[[20, 20, 20]]])
    curb = (((xx + yy) // 16) % 2 == 0)[..., None]
    floors[2] = np.clip(np.where(curb, [[[225, 225, 225]]], [[[205, 30, 30]]]) + noise(6), 0, 255)
    grass = np.array([[[50, 120, 45]]]) + rng.normal(0, 12, (n, n))[..., None] * [0.5, 1, 0.4]
    floors[3] = np.clip(grass, 0, 255)
    return walls, floors


def make_sky():
    sky_w = int(round(TAU / (2 * math.atan(PLANE)) * W))
    sky_h = H // 2 + 8
    y = (np.arange(sky_h) / (sky_h - 1))[:, None, None]
    top = np.array([35, 85, 170], np.float32)
    sky = np.broadcast_to(top * (1 - y) + FOG * y, (sky_h, sky_w, 3)).copy()
    x = np.arange(sky_w) / sky_w * TAU
    # nubes
    yy = np.arange(sky_h)[:, None]
    cl = (np.sin(x * 6 + 1.3) * np.sin(x * 11 + yy * 0.25) + np.sin(x * 3 - yy * 0.15)) * 0.5
    cmask = (cl > 0.45) & (yy < sky_h * 0.55)
    sky[cmask] = sky[cmask] * 0.4 + 255 * 0.6
    # montanas (dos capas)
    far = sky_h - (14 + 8 * np.sin(3 * x + 1) + 5 * np.sin(7 * x + 2) + 2 * np.sin(19 * x))
    near = sky_h - (6 + 5 * np.sin(5 * x + 4) + 3 * np.sin(13 * x + 1))
    rows = np.arange(sky_h)[:, None]
    sky[rows >= far[None, :]] = [120, 140, 165]
    m = rows >= near[None, :]
    sky[m] = [70, 105, 80]
    return sky.astype(np.float32), sky_w, sky_h


def make_car_sprites(color):
    key = (255, 0, 255)
    dark = tuple(int(v * 0.55) for v in color)
    light = tuple(min(255, int(v * 1.2) + 30) for v in color)

    def surf():
        s = pygame.Surface((SPR_W, SPR_H))
        s.fill(key)
        return s

    back = surf()
    pygame.draw.ellipse(back, (25, 25, 25), (8, 26, 48, 6))
    for wx in (12, 43):
        pygame.draw.rect(back, (15, 15, 15), (wx, 20, 9, 11), border_radius=2)
    pygame.draw.rect(back, dark, (19, 4, 26, 10), border_top_left_radius=5, border_top_right_radius=5)
    pygame.draw.rect(back, (60, 80, 105), (22, 6, 20, 6))
    pygame.draw.rect(back, color, (11, 13, 42, 13), border_radius=3)
    pygame.draw.rect(back, light, (11, 13, 42, 3))
    pygame.draw.rect(back, (240, 20, 20), (13, 16, 7, 3))
    pygame.draw.rect(back, (240, 20, 20), (44, 16, 7, 3))
    pygame.draw.rect(back, (240, 240, 240), (28, 19, 8, 4))
    pygame.draw.rect(back, (40, 40, 40), (11, 24, 42, 3))
    pygame.draw.rect(back, (25, 25, 25), (8, 9, 48, 3))
    pygame.draw.rect(back, (25, 25, 25), (14, 11, 3, 4))
    pygame.draw.rect(back, (25, 25, 25), (47, 11, 3, 4))

    front = surf()
    pygame.draw.ellipse(front, (25, 25, 25), (8, 26, 48, 6))
    for wx in (12, 43):
        pygame.draw.rect(front, (15, 15, 15), (wx, 20, 9, 11), border_radius=2)
    pygame.draw.rect(front, dark, (18, 4, 28, 10), border_top_left_radius=5, border_top_right_radius=5)
    pygame.draw.rect(front, (90, 125, 160), (20, 6, 24, 7))
    pygame.draw.rect(front, color, (11, 13, 42, 13), border_radius=3)
    pygame.draw.rect(front, light, (11, 13, 42, 3))
    pygame.draw.rect(front, (255, 250, 200), (13, 16, 8, 4))
    pygame.draw.rect(front, (255, 250, 200), (43, 16, 8, 4))
    pygame.draw.rect(front, (30, 30, 30), (24, 18, 16, 5))
    pygame.draw.rect(front, (40, 40, 40), (8, 25, 48, 2))

    side = surf()
    pygame.draw.ellipse(side, (25, 25, 25), (2, 27, 60, 5))
    pygame.draw.polygon(side, dark, [(16, 14), (24, 4), (42, 4), (52, 14)])
    pygame.draw.polygon(side, (80, 110, 140), [(20, 13), (26, 6), (32, 6), (32, 13)])
    pygame.draw.polygon(side, (80, 110, 140), [(35, 13), (35, 6), (41, 6), (48, 13)])
    pygame.draw.rect(side, color, (2, 13, 60, 12), border_radius=4)
    pygame.draw.rect(side, light, (2, 13, 60, 3))
    pygame.draw.rect(side, (245, 245, 245), (2, 19, 60, 2))
    pygame.draw.rect(side, (255, 245, 180), (58, 15, 4, 3))
    pygame.draw.rect(side, (230, 20, 20), (2, 15, 3, 3))
    pygame.draw.rect(side, dark, (1, 8, 9, 3))
    pygame.draw.rect(side, dark, (4, 10, 2, 4))
    for wx in (15, 49):
        pygame.draw.circle(side, (12, 12, 12), (wx, 25), 7)
        pygame.draw.circle(side, (150, 150, 155), (wx, 25), 3)

    def to_rgba(s):
        a = pygame.surfarray.array3d(s).transpose(1, 0, 2).astype(np.float32)
        alpha = ~((a[..., 0] == 255) & (a[..., 1] == 0) & (a[..., 2] == 255))
        return np.dstack([a, alpha.astype(np.float32)])

    sd = to_rgba(side)
    return {"back": to_rgba(back), "front": to_rgba(front),
            "side_r": sd, "side_l": np.ascontiguousarray(sd[:, ::-1])}


# ---------------------------------------------------------------- render 3D
NFOG = 32


def fog_level(d):
    """Indice de niebla segun la distancia (0 = todo niebla, NFOG-1 = nitido)."""
    return (np.clip(1.15 - d / FOG_DIST, 0, 1) * (NFOG - 1) + 0.5).astype(np.int32)


def fog_lut(tex):
    """Precalcula la textura mezclada con la niebla en NFOG niveles (uint8)."""
    s = np.linspace(0, 1, NFOG, dtype=np.float32).reshape((NFOG,) + (1,) * tex.ndim)
    return np.clip(tex[None] * s + FOG * (1 - s), 0, 255).astype(np.uint8)


class Renderer:
    def __init__(self, world):
        self.world = world
        wall_tex, floor_tex = make_textures()
        sidef = np.array([1.0, 0.78], np.float32)[:, None, None, None, None]
        self.wall_lut = fog_lut(wall_tex[None] * sidef).transpose(1, 0, 2, 3, 4, 5).copy()
        self.floor_lut = fog_lut(floor_tex)
        sky, self.sky_w, self.sky_h = make_sky()
        self.sky = np.clip(sky, 0, 255).astype(np.uint8)
        self.camx = ((2 * (np.arange(W) + 0.5) / W) - 1).astype(np.float32)
        self.atan = np.arctan(self.camx * PLANE)
        self.Y = (np.arange(H, dtype=np.float32) + 0.5)[:, None]

    def cast(self, px, py, rdx, rdy):
        wall = self.world.wall
        mapx = np.full(W, int(px), np.int32)
        mapy = np.full(W, int(py), np.int32)
        with np.errstate(divide="ignore"):
            ddx = np.where(rdx == 0, 1e30, np.abs(1.0 / rdx))
            ddy = np.where(rdy == 0, 1e30, np.abs(1.0 / rdy))
        stepx = np.where(rdx < 0, -1, 1).astype(np.int32)
        stepy = np.where(rdy < 0, -1, 1).astype(np.int32)
        sidex = np.where(rdx < 0, (px - mapx) * ddx, (mapx + 1.0 - px) * ddx)
        sidey = np.where(rdy < 0, (py - mapy) * ddy, (mapy + 1.0 - py) * ddy)
        side = np.zeros(W, np.int8)
        hit = np.zeros(W, bool)
        for _ in range(2 * MAP_N + 4):
            act = ~hit
            if not act.any():
                break
            lt = sidex < sidey
            mx = act & lt
            my = act & ~lt
            sidex[mx] += ddx[mx]
            mapx[mx] += stepx[mx]
            side[mx] = 0
            sidey[my] += ddy[my]
            mapy[my] += stepy[my]
            side[my] = 1
            np.clip(mapx, 0, MAP_N - 1, out=mapx)
            np.clip(mapy, 0, MAP_N - 1, out=mapy)
            hit |= act & (wall[mapy, mapx] > 0)
        perp = np.where(side == 0, sidex - ddx, sidey - ddy)
        perp = np.maximum(perp, 0.02)
        return perp, side, wall[mapy, mapx], mapx, mapy

    def render(self, px, py, ang, bob, cars, viewer):
        dx, dy = math.cos(ang), math.sin(ang)
        plx, ply = -dy * PLANE, dx * PLANE
        rdx = dx + plx * self.camx
        rdy = dy + ply * self.camx
        horizon = H / 2 + bob
        frame = np.empty((H, W, 3), np.uint8)
        y0 = int(clamp(math.floor(horizon - 0.5) + 1, 0, H))

        # cielo (panorama que gira con la camara)
        if y0 > 0:
            rows = np.clip((self.Y[:y0, 0] + (self.sky_h - horizon)).astype(np.int32), 0, self.sky_h - 1)
            cols = (((ang + self.atan) / TAU) * self.sky_w).astype(np.int32) % self.sky_w
            frame[:y0] = self.sky[rows[:, None], cols[None, :]]

        # suelo (floor casting, una distancia por fila)
        if y0 < H:
            rowd = (0.5 * H) / np.maximum(self.Y[y0:] - horizon, 0.5)
            fx = px + rowd * rdx[None, :]
            fy = py + rowd * rdy[None, :]
            flx, fly = np.floor(fx), np.floor(fy)
            cx = np.clip(flx.astype(np.int32), 0, MAP_N - 1)
            cy = np.clip(fly.astype(np.int32), 0, MAP_N - 1)
            ft = self.world.floor[cy, cx]
            tx = ((fx - flx) * TEX).astype(np.int32) & (TEX - 1)
            ty = ((fy - fly) * TEX).astype(np.int32) & (TEX - 1)
            frame[y0:] = self.floor_lut[fog_level(rowd), ft, ty, tx]

        # paredes
        perp, side, tid, _, _ = self.cast(px, py, rdx, rdy)
        wallx = np.where(side == 0, py + perp * rdy, px + perp * rdx)
        wallx -= np.floor(wallx)
        txw = (wallx * TEX).astype(np.int32)
        flip = ((side == 0) & (rdx < 0)) | ((side == 1) & (rdy > 0))
        txw = np.clip(np.where(flip, TEX - 1 - txw, txw), 0, TEX - 1)
        line_h = H / perp
        top = horizon - line_h / 2
        inwall = (self.Y >= top[None, :]) & (self.Y < (top + line_h)[None, :])
        yy, xx = np.nonzero(inwall)
        tyw = np.clip(((self.Y[yy, 0] - top[xx]) / line_h[xx] * TEX).astype(np.int32), 0, TEX - 1)
        lvl = fog_level(perp)
        frame[yy, xx] = self.wall_lut[side[xx], lvl[xx], tid[xx], tyw, txw[xx]]

        # sprites (coches rivales)
        self.draw_sprites(frame, perp, px, py, dx, dy, plx, ply, horizon, cars, viewer)
        return frame

    def draw_sprites(self, frame, zbuf, px, py, dx, dy, plx, ply, horizon, cars, viewer):
        inv = 1.0 / (plx * dy - dx * ply)
        items = []
        for c in cars:
            if c is viewer:
                continue
            rx, ry = c.x - px, c.y - py
            tx = inv * (dy * rx - dx * ry)
            tz = inv * (-ply * rx + plx * ry)
            if tz > 0.2:
                items.append((tz, tx, c))
        items.sort(key=lambda i: -i[0])
        for depth, tx, c in items:
            sx = W / 2 * (1 + tx / depth)
            scale = H / depth
            w, h = SPR_WORLD_W * scale, SPR_WORLD_H * scale
            bottom = horizon + 0.5 * H / depth
            top = bottom - h
            x0 = sx - w / 2
            xa, xb = max(0, int(x0)), min(W, int(x0 + w) + 1)
            ya, yb = max(0, int(top)), min(H, int(bottom) + 1)
            if xa >= xb or ya >= yb:
                continue
            # vista segun el angulo (como los sprites de 8 direcciones de Doom)
            rel = wrap_angle(math.atan2(py - c.y, px - c.x) - c.ang)
            if abs(rel) < math.pi / 4:
                img = c.sprites["front"]
            elif abs(rel) > 3 * math.pi / 4:
                img = c.sprites["back"]
            else:
                img = c.sprites["side_r"] if rel > 0 else c.sprites["side_l"]
            xs = np.arange(xa, xb)
            ys = np.arange(ya, yb)
            u = np.clip(((xs + 0.5 - x0) / w * SPR_W).astype(np.int32), 0, SPR_W - 1)
            v = np.clip(((ys + 0.5 - top) / h * SPR_H).astype(np.int32), 0, SPR_H - 1)
            sub = img[v[:, None], u[None, :]]
            mask = (sub[..., 3] > 0) & (zbuf[xa:xb] > depth)[None, :]
            if not mask.any():
                continue
            s = clamp(1.15 - depth / FOG_DIST, 0, 1)
            col = sub[..., :3][mask] * s + FOG * (1 - s)
            frame[ya:yb, xa:xb][mask] = col.astype(np.uint8)


# ---------------------------------------------------------------- coches
class Car:
    RADIUS = 0.28

    def __init__(self, name, color, x, y, ang, vmax, lane, is_player, track):
        self.name, self.color = name, color
        self.x, self.y, self.ang = x, y, ang
        self.vx = self.vy = 0.0
        self.vmax, self.lane, self.is_player = vmax, lane, is_player
        self.sprites = make_car_sprites(color)
        self.s, _ = track.project(x, y)
        self.u = self.s - track.L if self.s > track.L * 0.5 else self.s
        self.lap_idx = math.floor(self.u / track.L)
        self.lap_start = 0.0
        self.best_lap = self.last_lap = None
        self.finished, self.finish_time = False, None
        self.hurt = self.wrong = self.steer = 0.0
        self.stuck = self.reverse_t = 0.0
        self.throttle = False
        self.phase = random.random() * 6
        self.on_curb = False

    @property
    def vf(self):
        return self.vx * math.cos(self.ang) + self.vy * math.sin(self.ang)

    def drive(self, throttle, brake, steer_in, handbrake, dt, world):
        self.throttle = throttle
        ca, sa = math.cos(self.ang), math.sin(self.ang)
        vf = self.vx * ca + self.vy * sa
        vr = -self.vx * sa + self.vy * ca
        self.steer += (steer_in - self.steer) * min(1.0, dt * 8)
        if throttle:
            vf += ACCEL * dt * max(0.0, 1 - vf / self.vmax)
        if brake:
            if vf > 0.3:
                vf -= BRAKE * dt
            else:
                vf = max(-REVERSE_MAX, vf - ACCEL * 0.6 * dt)
        if not throttle and not brake:
            vf -= vf * COAST * dt
        vf -= vf * abs(vf) * DRAG * dt
        if self.on_curb:
            vf -= vf * 0.3 * dt
        grip = 1.3 if handbrake else 9.0
        if handbrake:
            vf -= vf * 0.8 * dt
        vr -= vr * min(1.0, grip * dt)
        turn = self.steer * STEER_RATE * clamp(vf / 2.5, -1, 1) / (1 + abs(vf) * 0.035)
        if handbrake:
            turn *= 1.35
        vf -= abs(turn) * vf * 0.08 * dt
        self.ang = (self.ang + turn * dt) % TAU
        self.vx = vf * ca - vr * sa
        self.vy = vf * sa + vr * ca

        speed = math.hypot(self.vx, self.vy)
        hit = False
        nx = self.x + self.vx * dt
        if world.solid(nx, self.y, self.RADIUS):
            self.vx *= -0.3
            hit = True
        else:
            self.x = nx
        ny = self.y + self.vy * dt
        if world.solid(self.x, ny, self.RADIUS):
            self.vy *= -0.3
            hit = True
        else:
            self.y = ny
        if hit:
            self.vx *= 0.75
            self.vy *= 0.75
            if speed > 2.5:
                self.hurt = 0.5
                return speed
        self.hurt = max(0.0, self.hurt - dt)
        return 0.0


def ai_inputs(car, track, t, dt):
    vf = car.vf
    if car.reverse_t > 0:
        car.reverse_t -= dt
        return False, True, -car.steer_rec, False
    look = 2.2 + max(0.0, vf) * 0.32
    tx, ty, tdx, tdy = track.point(car.s + look)
    lane = car.lane + 0.5 * math.sin(t * 0.35 + car.phase)
    tx += -tdy * lane
    ty += tdx * lane
    diff = wrap_angle(math.atan2(ty - car.y, tx - car.x) - car.ang)
    steer = clamp(diff * 2.2, -1, 1)
    _, _, ax, ay = track.point(car.s + 1.0)
    _, _, bx, by = track.point(car.s + 4.0 + max(0.0, vf) * 0.6)
    bend = abs(wrap_angle(math.atan2(by, bx) - math.atan2(ay, ax)))
    target = car.vmax * (1 - min(0.55, bend * 0.45))
    # recuperacion si se atasca contra un muro
    if vf < 0.6:
        car.stuck += dt
        if car.stuck > 1.2:
            car.stuck = 0
            car.reverse_t = 0.8
            car.steer_rec = steer
    else:
        car.stuck = 0
    return vf < target, vf > target + 1.5, steer, False


# ---------------------------------------------------------------- sonido
class Sound:
    def __init__(self):
        self.ok = False
        self.muted = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(22050, -16, 1, 512)
            self.freq, _, self.ch = pygame.mixer.get_init()
            pygame.mixer.set_reserved(1)
            self.engine_ch = pygame.mixer.Channel(0)
            self.engine = [self._engine(45 + i * 9) for i in range(16)]
            self.crash = self._crash()
            self.beep = self._tone(660, 0.18)
            self.go = self._tone(990, 0.5)
            self.level = -1
            self.ok = True
        except Exception:
            self.ok = False

    def _make(self, wave, vol=0.5):
        a = (np.clip(wave, -1, 1) * 32767 * vol).astype(np.int16)
        if self.ch == 2:
            a = np.column_stack([a, a])
        return pygame.sndarray.make_sound(np.ascontiguousarray(a))

    def _engine(self, f):
        period = max(2, int(round(self.freq / f)))
        f = self.freq / period
        cycles = max(1, int(0.15 * f))
        t = np.arange(period * cycles) / self.freq
        ph = (t * f) % 1.0
        wave = 0.45 * (2 * ph - 1) + 0.3 * np.sin(TAU * 2 * f * t) + 0.25 * np.sign(np.sin(TAU * f * t))
        wave += np.tile(np.random.default_rng(1).normal(0, 0.08, period), cycles)
        return self._make(wave * 0.6, 0.35)

    def _crash(self):
        n = int(self.freq * 0.35)
        w = np.random.default_rng(2).normal(0, 1, n)
        w = np.convolve(w, np.ones(12) / 12, "same") * 3
        return self._make(w * np.exp(-np.arange(n) / (n * 0.25)), 0.6)

    def _tone(self, f, dur):
        t = np.arange(int(self.freq * dur)) / self.freq
        env = np.minimum(1, (dur - t) * 20)
        return self._make(np.sign(np.sin(TAU * f * t)) * 0.5 * env, 0.4)

    def play(self, name):
        if self.ok and not self.muted:
            getattr(self, name).play()

    def update_engine(self, speed, throttle, vmax):
        if not self.ok:
            return
        if self.muted:
            self.engine_ch.stop()
            self.level = -1
            return
        lvl = int(clamp(abs(speed) / vmax, 0, 1) * 13) + (2 if throttle else 0)
        lvl = clamp(lvl, 0, 15)
        if lvl != self.level:
            self.level = lvl
            self.engine_ch.play(self.engine[lvl], loops=-1)
            self.engine_ch.set_volume(0.5)

    def stop_engine(self):
        if self.ok:
            self.engine_ch.stop()
            self.level = -1


# ---------------------------------------------------------------- juego
class Game:
    def __init__(self):
        pygame.mixer.pre_init(22050, -16, 1, 512)
        pygame.init()
        self.screen = pygame.display.set_mode((SW, SH), pygame.SCALED)
        pygame.display.set_caption("GTC Racing")
        self.clock = pygame.time.Clock()
        self.track = Track(WAYPOINTS)
        self.world = World(self.track)
        self.renderer = Renderer(self.world)
        self.sound = Sound()
        self.f_big = pygame.font.SysFont("impact,arialblack,arial", 52)
        self.f_huge = pygame.font.SysFont("impact,arialblack,arial", 110)
        self.f_mid = pygame.font.SysFont("impact,arialblack,arial", 36)
        self.f_small = pygame.font.SysFont("arial", 16, bold=True)
        self.bar = self.make_bar()
        self.minimap = self.make_minimap()
        self.t = 0.0
        self.new_race(attract=True)

    # ------------------------------------------------------------ setup
    def new_race(self, attract=False):
        self.state = "menu" if attract else "countdown"
        self.countdown = 3.99
        self.last_count = 4
        self.race_time = 0.0
        self.message, self.msg_t = "", 0.0
        self.results_t = 0.0
        specs = [("ROJO", (205, 40, 40), 0.95, -0.4),
                 ("AZUL", (40, 95, 225), 0.97, 0.4),
                 ("TU", (245, 200, 30), 1.0, 0.0),
                 ("VERDE", (40, 175, 70), 0.92, -0.2)]
        self.cars = []
        for k, (name, col, skill, lane) in enumerate(specs):
            s = self.track.L - 1.3 - (k // 2) * 2.2
            x, y, dx, dy = self.track.point(s)
            off = -0.8 if k % 2 == 0 else 0.8
            x += -dy * off
            y += dx * off
            is_p = name == "TU"
            car = Car(name, col, x, y, math.atan2(dy, dx), PLAYER_VMAX * skill, lane, is_p, self.track)
            self.cars.append(car)
            if is_p:
                self.player = car

    def make_bar(self):
        bar = pygame.Surface((SW, BAR_H))
        n = rng.normal(0, 9, (SW, BAR_H))
        base = 95 + n + np.sin(np.arange(BAR_H) / 3.0)[None, :] * 4
        arr = np.clip(np.dstack([base, base * 0.97, base * 0.92]), 0, 255).astype(np.uint8)
        pygame.surfarray.blit_array(bar, arr)
        pygame.draw.line(bar, (150, 150, 150), (0, 0), (SW, 0), 3)
        self.panels = [("VELOCIDAD", pygame.Rect(8, 8, 190, 80)),
                       ("VUELTA", pygame.Rect(206, 8, 170, 80)),
                       ("", pygame.Rect(392, 8, 176, 80)),
                       ("POSICION", pygame.Rect(584, 8, 170, 80)),
                       ("TIEMPO", pygame.Rect(770, 8, 182, 80))]
        for label, r in self.panels:
            pygame.draw.rect(bar, (45, 42, 40), r)
            pygame.draw.line(bar, (30, 30, 30), r.topleft, r.topright, 2)
            pygame.draw.line(bar, (30, 30, 30), r.topleft, r.bottomleft, 2)
            pygame.draw.line(bar, (160, 160, 160), r.bottomleft, r.bottomright, 2)
            pygame.draw.line(bar, (160, 160, 160), r.topright, r.bottomright, 2)
            if label:
                t = self.f_small.render(label, True, (200, 200, 200))
                bar.blit(t, (r.centerx - t.get_width() // 2, r.bottom - 22))
        return bar

    def make_minimap(self):
        c = 4
        mm = pygame.Surface((MAP_N * c, MAP_N * c), pygame.SRCALPHA)
        mm.fill((0, 0, 0, 110))
        for y in range(MAP_N):
            for x in range(MAP_N):
                if self.world.open[y, x]:
                    col = (230, 230, 230, 200) if self.world.floor[y, x] == 1 else (120, 120, 130, 190)
                    mm.fill(col, (x * c, y * c, c, c))
        return mm

    # ------------------------------------------------------------ logica
    def player_inputs(self):
        k = pygame.key.get_pressed()
        thr = k[pygame.K_UP] or k[pygame.K_w]
        brk = k[pygame.K_DOWN] or k[pygame.K_s]
        st = (1 if (k[pygame.K_RIGHT] or k[pygame.K_d]) else 0) - (1 if (k[pygame.K_LEFT] or k[pygame.K_a]) else 0)
        return thr, brk, st, k[pygame.K_SPACE]

    def show(self, msg, dur=2.0):
        self.message, self.msg_t = msg, dur

    def update(self, dt):
        self.t += dt
        self.msg_t = max(0.0, self.msg_t - dt)
        if self.state == "countdown":
            self.countdown -= dt
            n = int(self.countdown)
            if n < self.last_count and n >= 1:
                self.sound.play("beep")
            self.last_count = n
            if self.countdown <= 0:
                self.state = "race"
                self.sound.play("go")
                self.show("¡YA!", 1.0)
        if self.state in ("race", "finished"):
            self.race_time += dt
        if self.state == "finished":
            self.results_t += dt

        for car in self.cars:
            if self.state == "countdown":
                inp = (False, True, 0, False)
            elif car.is_player and self.state == "race":
                inp = self.player_inputs()
            else:
                inp = ai_inputs(car, self.track, self.t, dt)
                if car.finished:
                    inp = (inp[0] and car.vf < car.vmax * 0.5, inp[1], inp[2], False)
            fx, fy = int(car.x), int(car.y)
            car.on_curb = self.world.floor[fy, fx] == 2
            impact = car.drive(*inp, dt, self.world)
            if impact and car.is_player and self.state != "menu":
                self.sound.play("crash")
        self.collide_cars()

        L = self.track.L
        for car in self.cars:
            s, _ = self.track.project(car.x, car.y)
            ds = s - car.s
            if ds > L / 2:
                ds -= L
            elif ds < -L / 2:
                ds += L
            car.s = s
            car.u += ds
            if ds < -1e-4 and abs(car.vf) > 0.5:
                car.wrong += dt
            else:
                car.wrong = max(0.0, car.wrong - 2 * dt)
            if self.state not in ("race", "finished"):
                continue
            li = math.floor(car.u / L)
            if li > car.lap_idx:
                if li >= 1 and not car.finished:
                    car.last_lap = self.race_time - car.lap_start
                    car.best_lap = car.last_lap if car.best_lap is None else min(car.best_lap, car.last_lap)
                    car.lap_start = self.race_time
                    if car.is_player and li < LAPS:
                        self.show("¡VUELTA FINAL!" if li == LAPS - 1 else f"VUELTA {li + 1}", 2.0)
                        self.sound.play("beep")
                car.lap_idx = li
            if not car.finished and car.u >= LAPS * L:
                car.finished, car.finish_time = True, self.race_time
                if car.is_player:
                    self.state = "finished"
                    self.results_t = 0.0
                    self.sound.play("go")
                    self.show("¡META!", 2.5)

        p = self.player
        if self.state == "menu":
            self.sound.stop_engine()
        else:
            self.sound.update_engine(p.vf, p.throttle, p.vmax)

    def collide_cars(self):
        md = 0.6
        cs = self.cars
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                if d >= md or d < 1e-6:
                    continue
                nx, ny = dx / d, dy / d
                push = (md - d) / 2
                if not self.world.solid(a.x - nx * push, a.y - ny * push, Car.RADIUS):
                    a.x -= nx * push
                    a.y -= ny * push
                if not self.world.solid(b.x + nx * push, b.y + ny * push, Car.RADIUS):
                    b.x += nx * push
                    b.y += ny * push
                rel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
                if rel < 0:
                    jimp = -(1.3) * rel / 2
                    a.vx -= jimp * nx
                    a.vy -= jimp * ny
                    b.vx += jimp * nx
                    b.vy += jimp * ny
                    if (a.is_player or b.is_player) and -rel > 2 and self.state != "menu":
                        self.player.hurt = 0.4
                        self.sound.play("crash")

    def standings(self):
        return sorted(self.cars, key=lambda c: (0, c.finish_time) if c.finished else (1, -c.u))

    # ------------------------------------------------------------ dibujo
    def draw(self):
        p = self.player
        speed = abs(p.vf)
        bob = math.sin(self.t * 28) * min(1.0, speed / 12) * 0.6
        if p.on_curb and speed > 2:
            bob += random.uniform(-1.2, 1.2)
        shake = (0, 0)
        if p.hurt > 0:
            bob += random.uniform(-2, 2)
            shake = (random.randint(-6, 6), random.randint(-4, 4))
        frame = self.renderer.render(p.x, p.y, p.ang, bob, self.cars, p)
        img = pygame.image.frombuffer(frame.tobytes(), (W, H), "RGB")
        self.screen.fill((0, 0, 0))
        self.screen.blit(pygame.transform.scale(img, (VIEW_W, VIEW_H)), shake)

        if self.state == "menu":
            self.draw_menu()
        else:
            self.draw_wheel(p)
            self.draw_bar(p)
            self.draw_minimap()
            self.draw_overlay(p)
        pygame.display.flip()

    def text(self, font, s, color, center, shadow=True):
        if shadow:
            t = font.render(s, True, (0, 0, 0))
            r = t.get_rect(center=(center[0] + 3, center[1] + 3))
            self.screen.blit(t, r)
        t = font.render(s, True, color)
        r = t.get_rect(center=center)
        self.screen.blit(t, r)

    def draw_wheel(self, p):
        cx, cy, R = SW // 2, VIEW_H + 120, 230
        rot = p.steer * math.radians(100)
        pygame.draw.circle(self.screen, (30, 30, 34), (cx, cy), R, 28)
        pygame.draw.circle(self.screen, (70, 70, 78), (cx, cy), R - 4, 3)
        a = rot
        mx, my = cx + (R - 14) * math.sin(a), cy - (R - 14) * math.cos(a)
        pygame.draw.circle(self.screen, (220, 30, 30), (int(mx), int(my)), 10)
        for side in (-1, 1):
            a = rot + side * math.radians(48)
            hx, hy = cx + (R - 14) * math.sin(a), cy - (R - 14) * math.cos(a)
            glove = tuple(int(v * 0.6) for v in p.color)
            pygame.draw.circle(self.screen, (15, 15, 15), (int(hx), int(hy)), 25)
            pygame.draw.circle(self.screen, glove, (int(hx), int(hy)), 22)
            pygame.draw.circle(self.screen, (240, 240, 240), (int(hx - side * 6), int(hy - 8)), 6)

    def draw_bar(self, p):
        self.screen.blit(self.bar, (0, VIEW_H))
        red = (220, 25, 20)
        vals = [str(int(abs(p.vf) * KMH)),
                f"{min(LAPS, max(0, p.lap_idx) + 1)}/{LAPS}",
                None,
                f"{self.standings().index(p) + 1}/{len(self.cars)}",
                fmt_time(self.race_time if not p.finished else p.finish_time)]
        for (label, r), v in zip(self.panels, vals):
            if v is None:
                continue
            font = self.f_big if len(v) < 6 else self.f_mid
            self.text(font, v, red, (r.centerx, VIEW_H + r.y + 30))
        self.draw_face(p, SW // 2, VIEW_H + 48)

    def draw_face(self, p, cx, cy):
        s = self.screen
        pygame.draw.circle(s, (15, 15, 15), (cx, cy), 39)
        pygame.draw.circle(s, p.color, (cx, cy), 37)
        pygame.draw.rect(s, (245, 245, 245), (cx - 5, cy - 37, 10, 22))
        pygame.draw.rect(s, (225, 175, 135), (cx - 25, cy - 12, 50, 36), border_radius=12)
        look = int(p.steer * 7)
        pos = self.standings().index(p)
        if p.hurt > 0:
            for ex in (cx - 11, cx + 11):
                pygame.draw.line(s, (0, 0, 0), (ex - 5, cy - 6), (ex + 5, cy + 2), 3)
                pygame.draw.line(s, (0, 0, 0), (ex - 5, cy + 2), (ex + 5, cy - 6), 3)
            pygame.draw.ellipse(s, (120, 20, 20), (cx - 7, cy + 8, 14, 12))
            return
        for ex in (cx - 11, cx + 11):
            pygame.draw.circle(s, (255, 255, 255), (ex + look, cy - 2), 6)
            pygame.draw.circle(s, (0, 0, 0), (ex + look + int(p.steer * 2), cy - 1), 3)
        if p.finished and pos == 0 or (self.state == "race" and pos == 0):
            pygame.draw.arc(s, (90, 20, 20), (cx - 12, cy + 2, 24, 16), math.pi, TAU, 3)
        elif p.throttle and abs(p.vf) > 8:
            pygame.draw.rect(s, (255, 255, 255), (cx - 10, cy + 10, 20, 7))
            pygame.draw.rect(s, (0, 0, 0), (cx - 10, cy + 10, 20, 7), 1)
            pygame.draw.line(s, (0, 0, 0), (cx - 16, cy - 11), (cx - 5, cy - 7), 3)
            pygame.draw.line(s, (0, 0, 0), (cx + 16, cy - 11), (cx + 5, cy - 7), 3)
        else:
            pygame.draw.line(s, (90, 20, 20), (cx - 8, cy + 13), (cx + 8, cy + 13), 3)

    def draw_minimap(self):
        ox, oy, c = SW - MAP_N * 4 - 10, 10, 4
        self.screen.blit(self.minimap, (ox, oy))
        for car in self.cars:
            if car is self.player:
                continue
            pygame.draw.circle(self.screen, car.color, (int(ox + car.x * c), int(oy + car.y * c)), 4)
        p = self.player
        a = p.ang
        pts = [(ox + (p.x + math.cos(a) * 1.6) * c, oy + (p.y + math.sin(a) * 1.6) * c),
               (ox + (p.x + math.cos(a + 2.5) * 1.1) * c, oy + (p.y + math.sin(a + 2.5) * 1.1) * c),
               (ox + (p.x + math.cos(a - 2.5) * 1.1) * c, oy + (p.y + math.sin(a - 2.5) * 1.1) * c)]
        pygame.draw.polygon(self.screen, p.color, pts)
        pygame.draw.polygon(self.screen, (0, 0, 0), pts, 1)

    def draw_overlay(self, p):
        if self.state == "countdown":
            n = int(self.countdown)
            for i in range(3):
                on = (3 - i) > n
                col = (230, 30, 30) if on else (60, 20, 20)
                cx = SW // 2 + (i - 1) * 80
                pygame.draw.circle(self.screen, (10, 10, 10), (cx, 70), 34)
                pygame.draw.circle(self.screen, col, (cx, 70), 28)
            if n >= 1:
                self.text(self.f_huge, str(n), (255, 230, 60), (SW // 2, VIEW_H // 2 - 20))
        if self.msg_t > 0 and self.message:
            self.text(self.f_huge, self.message, (255, 230, 60), (SW // 2, VIEW_H // 2 - 40))
        if self.state == "race" and p.wrong > 1.0 and int(self.t * 3) % 2 == 0:
            self.text(self.f_mid, "¡SENTIDO CONTRARIO!", (255, 60, 40), (SW // 2, 140))
        if self.state in ("race", "finished"):
            y = 14
            for label, val in (("ULTIMA", p.last_lap), ("MEJOR", p.best_lap)):
                t = self.f_small.render(f"{label}: {fmt_time(val)}", True, (255, 255, 255))
                sh = self.f_small.render(f"{label}: {fmt_time(val)}", True, (0, 0, 0))
                self.screen.blit(sh, (16, y + 2))
                self.screen.blit(t, (14, y))
                y += 22
        if self.state == "finished" and self.results_t > 2.0:
            panel = pygame.Surface((520, 330), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 190))
            self.screen.blit(panel, (SW // 2 - 260, 70))
            self.text(self.f_mid, "RESULTADOS", (255, 230, 60), (SW // 2, 105))
            for i, c in enumerate(self.standings()):
                y = 160 + i * 46
                col = (255, 230, 60) if c.is_player else (235, 235, 235)
                pygame.draw.rect(self.screen, c.color, (SW // 2 - 220, y - 12, 24, 24))
                t = self.f_mid.render(f"{i + 1}.  {c.name}", True, col)
                self.screen.blit(t, (SW // 2 - 185, y - 22))
                tt = self.f_mid.render(fmt_time(c.finish_time), True, col)
                self.screen.blit(tt, (SW // 2 + 230 - tt.get_width(), y - 22))
            self.text(self.f_small, "R: otra carrera     ESC: menu", (220, 220, 220), (SW // 2, 375))

    def draw_menu(self):
        shade = pygame.Surface((SW, SH), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 90))
        self.screen.blit(shade, (0, 0))
        self.text(self.f_huge, "GTC RACING", (255, 210, 40), (SW // 2, 150))
        self.text(self.f_mid, f"Carrera a {LAPS} vueltas contra 3 rivales", (240, 240, 240), (SW // 2, 240))
        if int(self.t * 2) % 2 == 0:
            self.text(self.f_mid, "Pulsa ENTER para empezar", (255, 255, 255), (SW // 2, 330))
        lines = ["Flechas / WASD: conducir    ESPACIO: freno de mano",
                 "M: sonido    F11: pantalla completa    ESC: salir"]
        for i, l in enumerate(lines):
            self.text(self.f_small, l, (230, 230, 230), (SW // 2, 430 + i * 26))

    # ------------------------------------------------------------ bucle
    def run(self):
        while True:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        if self.state == "menu":
                            pygame.quit()
                            sys.exit()
                        self.new_race(attract=True)
                    elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.state == "menu":
                        self.new_race()
                    elif e.key == pygame.K_r and self.state != "menu":
                        self.new_race()
                    elif e.key == pygame.K_m:
                        self.sound.muted = not self.sound.muted
                    elif e.key == pygame.K_F11:
                        pygame.display.toggle_fullscreen()
            self.update(dt)
            self.draw()


if __name__ == "__main__":
    Game().run()

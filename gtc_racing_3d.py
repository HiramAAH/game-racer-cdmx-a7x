"""
GTC RACING 3D - Gran Premio CDMX
Juego de carreras 3D con poligonos estilo Nintendo 64 (Python + pygame + moderngl).
Pista inspirada en el autodromo de la Ciudad de Mexico, con su seccion de estadio.
Pilotos parodia de mascotas de marcas mexicanas, cinematica previa y 4 dificultades.

Controles:
  Flechas / WASD : acelerar, frenar/reversa, girar
  ESPACIO        : freno de mano (derrape)
  SHIFT / X      : nitro
  CTRL           : cambiar de cancion (soundtrack de noche)
  C              : cambiar camara (persecucion / capo)
  N              : horario (dia / atardecer / noche)
  ENTER          : elegir piloto / saltar la cinematica
  M              : silenciar / activar sonido
  R              : reiniciar carrera
  ESC            : volver al menu / salir
"""
import math
import os
import random
import re
import sys

import numpy as np
import pygame
import moderngl

# ---------------------------------------------------------------- configuracion
WIN_W, WIN_H = 1280, 720
LAPS = 3
HW = 7.0                 # medio ancho del asfalto (m)
CURB_W = 1.4
LIMIT = HW + 11.0        # distancia del centro a las barreras
CAR_R = 1.2
WB = 2.6                 # distancia entre ejes
VMAX = 90.0              # m/s (limite teorico; la punta real ~260 km/h)
ACCEL, BRAKE, A_LAT = 11.0, 28.0, 26.0
TAU = math.tau
SPACING = 3.0
NITRO_TIME = 3.2         # segundos de nitro con el tanque lleno
NITRO_RECHARGE = 0.06    # recarga por segundo (tanque lleno en ~17 s)
NITRO_ACCEL, NITRO_VMAX = 1.8, 1.2

SKY_TOP = (0.25, 0.48, 0.85)
FOG_COL = (0.72, 0.80, 0.90)
SUN_DIR = (0.35, -0.85, 0.40)

TRACK_SCALE = 1.2        # pista mas grande (recta principal ~1.3 km)
STADIUM_SCALE = 0.75     # la seccion del estadio conserva su tamano
CONTROL = [
    (-150, 0), (250, 0), (650, 0), (950, 0),                       # recta principal
    (1010, 15), (1030, 60), (1000, 100), (1020, 150),              # curvas 1-3
    (990, 205), (900, 235), (700, 250), (520, 262),                # contrarrecta
    (440, 245), (395, 290), (335, 250), (275, 295), (215, 255),    # las eses
    (150, 290), (70, 320), (-10, 310), (-60, 260),                 # curvas 10-11
    (-110, 215), (-170, 250), (-250, 270), (-310, 235),            # estadio
    (-300, 170), (-235, 150), (-275, 95),
    (-320, 45), (-280, 5),                                         # curva final
]
STADIUM_PTS = slice(21, 28)


def clamp(v, a, b):
    return a if v < a else b if v > b else v


def wrap_angle(a):
    return (a + math.pi) % TAU - math.pi


def fmt_time(t):
    if t is None:
        return "--:--.--"
    m = int(t // 60)
    return f"{m}:{t - m * 60:05.2f}"


# ---------------------------------------------------------------- matrices
def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.array(v, np.float32) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


def model_mat(x, z, ang):
    c, s = math.cos(-ang), math.sin(-ang)
    return np.array([[c, 0, s, x], [0, 1, 0, 0], [-s, 0, c, z], [0, 0, 0, 1]], np.float32)


def gl_bytes(m):
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


# ---------------------------------------------------------------- pista
def centripetal_cr(P, spacing):
    P = np.asarray(P, float)
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        steps = max(4, int(np.linalg.norm(p2 - p1)))
        for t in np.linspace(t1, t2, steps, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    pts = np.array(out)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    n_out = int(cum[-1] / spacing)
    ss = np.linspace(0, cum[-1], n_out, endpoint=False)
    res = np.column_stack([np.interp(ss, cum, closed[:, 0]), np.interp(ss, cum, closed[:, 1])])
    for _ in range(2):  # suavizado
        res = sum(np.roll(res, k, axis=0) for k in range(-3, 4)) / 7.0
    return res


class Track:
    def __init__(self):
        raw = np.array(CONTROL, float)
        ctrl = raw * TRACK_SCALE
        st = raw[STADIUM_PTS]
        cst = st.mean(axis=0)
        ctrl[STADIUM_PTS] = cst * TRACK_SCALE + (st - cst) * STADIUM_SCALE
        self.ctrl = ctrl
        self.pts = centripetal_cr(ctrl, SPACING)
        self.n = len(self.pts)
        nxt = np.roll(self.pts, -1, axis=0)
        self.d = nxt - self.pts
        self.seg_len = np.linalg.norm(self.d, axis=1)
        self.cum = np.concatenate([[0], np.cumsum(self.seg_len)[:-1]])
        self.L = float(self.seg_len.sum())
        self.spacing = self.L / self.n
        self.inv_len2 = 1.0 / self.seg_len ** 2
        t = self.d / self.seg_len[:, None]
        self.tan = t
        self.nrm = np.column_stack([-t[:, 1], t[:, 0]])  # derecha
        head = np.arctan2(t[:, 1], t[:, 0])
        dh = (head - np.roll(head, 1) + np.pi) % TAU - np.pi
        k = dh / self.spacing
        self.curv = sum(np.roll(k, j) for j in range(-3, 4)) / 7.0
        # zonas de grava en el exterior de las curvas cerradas: sand[i, 0]=izq, sand[i, 1]=der
        self.sand = np.zeros((self.n, 2), bool)
        sharp = np.abs(self.curv) > 1 / 110
        self.sand[sharp & (self.curv > 0), 0] = True
        self.sand[sharp & (self.curv < 0), 1] = True
        self.center = self.pts.mean(axis=0)

    def project(self, x, z):
        px = x - self.pts[:, 0]
        pz = z - self.pts[:, 1]
        t = np.clip((px * self.d[:, 0] + pz * self.d[:, 1]) * self.inv_len2, 0, 1)
        ex = px - self.d[:, 0] * t
        ez = pz - self.d[:, 1] * t
        i = int(np.argmin(ex * ex + ez * ez))
        nx, nz = self.nrm[i]
        lat = ex[i] * nx + ez[i] * nz
        cx = self.pts[i, 0] + self.d[i, 0] * t[i]
        cz = self.pts[i, 1] + self.d[i, 1] * t[i]
        return float(self.cum[i] + t[i] * self.seg_len[i]), i, float(lat), cx, cz, nx, nz

    def point(self, s):
        s %= self.L
        i = int(s / self.spacing) % self.n
        f = (s - self.cum[i]) / self.seg_len[i]
        p = self.pts[i] + self.d[i] * f
        return p[0], p[1], self.tan[i, 0], self.tan[i, 1]

    def nearest_dist(self, q):
        """Distancia minima de cada punto q (M,2) a los puntos de la pista."""
        out = np.empty(len(q))
        for a in range(0, len(q), 512):
            c = q[a:a + 512]
            d2 = ((c[:, None, :] - self.pts[None, :, :]) ** 2).sum(-1)
            out[a:a + 512] = np.sqrt(d2.min(1))
        return out

    def speed_profile(self, a_lat, brake, vmax):
        v = np.minimum(vmax, np.sqrt(a_lat / np.maximum(np.abs(self.curv), 1e-5)))
        for _ in range(2):
            for i in range(self.n - 1, -1, -1):
                nxt = v[(i + 1) % self.n]
                v[i] = min(v[i], math.sqrt(nxt * nxt + 2 * brake * self.spacing))
        return v


# ---------------------------------------------------------------- geometria
class Mesh:
    """Acumula triangulos con normal plana: pos(3) normal(3) color(3) material(3)."""

    def __init__(self):
        self.data = []

    def tri(self, a, b, c, cols, mats):
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        nx, ny, nz = nx / ln, ny / ln, nz / ln
        ext = self.data.extend
        for p, cc, m in ((a, cols[0], mats[0]), (b, cols[1], mats[1]), (c, cols[2], mats[2])):
            ext((p[0], p[1], p[2], nx, ny, nz, cc[0], cc[1], cc[2], m[0], m[1], m[2]))

    def quad(self, a, b, c, d, col, mats=None):
        cols = col if isinstance(col[0], (tuple, list)) else (col,) * 4
        mats = mats or ((0, 0, 0),) * 4
        self.tri(a, b, c, (cols[0], cols[1], cols[2]), (mats[0], mats[1], mats[2]))
        self.tri(a, c, d, (cols[0], cols[2], cols[3]), (mats[0], mats[2], mats[3]))

    def hexa(self, p, col, face_cols=None, mat=0):
        """Hexaedro: p = [b0..b3, t0..t3] (atras-izq, frente-izq, frente-der, atras-der)."""
        fc = face_cols or {}
        mv = mat if isinstance(mat, tuple) else (mat, 0, 0)
        faces = {"bottom": (0, 3, 2, 1), "top": (4, 5, 6, 7), "left": (0, 1, 5, 4),
                 "right": (3, 7, 6, 2), "front": (1, 2, 6, 5), "back": (0, 4, 7, 3)}
        m = (mv,) * 4
        for name, idx in faces.items():
            self.quad(*[p[i] for i in idx], fc.get(name, col), m)

    def box(self, o, ax, ay, az, col, face_cols=None, mat=0):
        def P(i, j, k):
            return (o[0] + i * ax[0] + j * ay[0] + k * az[0],
                    o[1] + i * ax[1] + j * ay[1] + k * az[1],
                    o[2] + i * ax[2] + j * ay[2] + k * az[2])
        pts = [P(0, 0, 0), P(1, 0, 0), P(1, 0, 1), P(0, 0, 1),
               P(0, 1, 0), P(1, 1, 0), P(1, 1, 1), P(0, 1, 1)]
        self.hexa(pts, col, face_cols, mat)

    def aabb(self, x0, y0, z0, x1, y1, z1, col, face_cols=None, mat=0):
        self.box((x0, y0, z0), (x1 - x0, 0, 0), (0, y1 - y0, 0), (0, 0, z1 - z0), col, face_cols, mat)

    def array(self):
        return np.array(self.data, np.float32)


def rgb(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0)


def billboard(mesh, cx, cz, fx, fz, width, y0, y1, k):
    """Cartel con textura del atlas. (fx,fz) = hacia donde mira quien lo lee."""
    rx, rz = -fz, fx
    h = width / 2
    lx, lz, qx, qz = cx - rx * h, cz - rz * h, cx + rx * h, cz + rz * h
    vt, vb = k / 16 + 0.002, (k + 1) / 16 - 0.002
    mesh.quad((lx, y0, lz), (qx, y0, qz), (qx, y1, qz), (lx, y1, lz), (1, 1, 1),
              ((7, 0, vb), (7, 0.5, vb), (7, 0.5, vt), (7, 0, vt)))
    bx, bz = fx * 0.15, fz * 0.15
    g = rgb(90, 90, 95)
    mesh.quad((lx + bx, y0, lz + bz), (lx + bx, y1, lz + bz), (qx + bx, y1, qz + bz), (qx + bx, y0, qz + bz), g)
    for px, pz in ((lx, lz), (qx, qz)):
        mesh.aabb(px - 0.15, 0, pz - 0.15, px + 0.15, y0, pz + 0.15, g)


def build_scene(track):
    m = Mesh()
    rnd = random.Random(42)
    pts, nrm, n = track.pts, track.nrm, track.n
    cxm, czm = track.center

    # suelo de cesped
    G = 6000
    grass = rgb(80, 150, 60)
    m.quad((cxm - G, 0, czm - G), (cxm + G, 0, czm - G), (cxm + G, 0, czm + G), (cxm - G, 0, czm + G),
           grass, ((2, 0, 0),) * 4)

    # explanada del estadio
    sc = track.ctrl[STADIUM_PTS].mean(axis=0)
    st_pts = track.ctrl[STADIUM_PTS]
    r0 = float(np.max(np.linalg.norm(st_pts - sc, axis=1))) + LIMIT + 6
    conc = rgb(170, 165, 155)
    seg = 48
    for j in range(seg):
        a0, a1 = j / seg * TAU, (j + 1) / seg * TAU
        m.tri((sc[0], 0.03, sc[1]), (sc[0] + r0 * math.cos(a1), 0.03, sc[1] + r0 * math.sin(a1)),
              (sc[0] + r0 * math.cos(a0), 0.03, sc[1] + r0 * math.sin(a0)), (conc,) * 3, ((5, 0, 0),) * 3)

    def off(i, d):
        return pts[i, 0] + nrm[i, 0] * d, pts[i, 1] + nrm[i, 1] * d

    # asfalto
    asph = rgb(85, 85, 92)
    for i in range(n):
        j = (i + 1) % n
        s0, s1 = track.cum[i], track.cum[i] + track.seg_len[i]
        a, b = off(i, -HW), off(i, HW)
        c, d = off(j, HW), off(j, -HW)
        m.quad((a[0], 0.08, a[1]), (b[0], 0.08, b[1]), (c[0], 0.08, c[1]), (d[0], 0.08, d[1]), asph,
               ((1, s0, -HW), (1, s0, HW), (1, s1, HW), (1, s1, -HW)))

    # pianos (curbs) en las curvas
    curvy = np.abs(track.curv) > 1 / 350
    curvy = np.convolve(np.concatenate([curvy[-6:], curvy, curvy[:6]]).astype(float), np.ones(13), "same")[6:-6] > 0
    sand_col = rgb(215, 190, 130)
    for i in range(n):
        j = (i + 1) % n
        s0, s1 = track.cum[i], track.cum[i] + track.seg_len[i]
        for side in (-1, 1):
            if curvy[i]:
                a, b = off(i, side * HW), off(i, side * (HW + CURB_W))
                c, d = off(j, side * (HW + CURB_W)), off(j, side * HW)
                m.quad((a[0], 0.11, a[1]), (b[0], 0.11, b[1]), (c[0], 0.11, c[1]), (d[0], 0.11, d[1]),
                       (1, 1, 1), ((4, s0, 0), (4, s0, 0), (4, s1, 0), (4, s1, 0)))
            if track.sand[i, 0 if side < 0 else 1]:
                a, b = off(i, side * (HW + CURB_W)), off(i, side * (LIMIT - 1))
                c, d = off(j, side * (LIMIT - 1)), off(j, side * (HW + CURB_W))
                m.quad((a[0], 0.06, a[1]), (b[0], 0.06, b[1]), (c[0], 0.06, c[1]), (d[0], 0.06, d[1]),
                       sand_col, ((5, 0, 0),) * 4)

    # linea de meta y parrilla
    for k in range(-1, 2):
        i = k % n
        j = (i + 1) % n
        if k != 0:
            continue
        a, b = off(i, -HW), off(i, HW)
        ex = track.tan[i] * 2.5
        m.quad((a[0], 0.13, a[1]), (b[0], 0.13, b[1]), (b[0] + ex[0], 0.13, b[1] + ex[1]),
               (a[0] + ex[0], 0.13, a[1] + ex[1]), (1, 1, 1),
               ((3, 0, -HW), (3, 0, HW), (3, 2.5, HW), (3, 2.5, -HW)))

    # barreras (solo donde no invaden otro tramo de pista)
    bar_h = 1.2
    for side in (-1, 1):
        q = pts + nrm * side * LIMIT
        valid = track.nearest_dist(q) >= LIMIT - 1.0
        for i in range(n):
            j = (i + 1) % n
            if not (valid[i] and valid[j]) or np.linalg.norm(q[j] - q[i]) > 8:
                continue
            u0, u1 = track.cum[i], track.cum[i] + track.seg_len[i]
            m.quad((q[i, 0], 0, q[i, 1]), (q[j, 0], 0, q[j, 1]), (q[j, 0], bar_h, q[j, 1]), (q[i, 0], bar_h, q[i, 1]),
                   (1, 1, 1), ((6, u0, 0), (6, u1, 0), (6, u1, 1), (6, u0, 1)))
        # carteles publicitarios detras de la barrera
        q2 = pts + nrm * side * (LIMIT + 2.5)
        valid2 = track.nearest_dist(q2) >= LIMIT + 1.5
        step = int(70 / track.spacing)
        start = 0 if side > 0 else step // 2
        for i in range(start, n, step):
            if valid2[i]:
                fx, fz = nrm[i] * side
                billboard(m, q2[i, 0], q2[i, 1], fx, fz, 11, 1.0, 3.6, SPONSORS[(i // step) % len(SPONSORS)])

    # portico de meta
    i0 = 0
    p0 = pts[i0]
    nr = nrm[i0]
    tcol = rgb(60, 60, 70)
    for side in (-1, 1):
        px, pz = p0 + nr * side * (HW + 2)
        m.aabb(px - 0.4, 0, pz - 0.4, px + 0.4, 8.5, pz + 0.4, tcol)
    fx, fz = track.tan[i0]
    billboard(m, p0[0] - fx * 0.3, p0[1] - fz * 0.3, fx, fz, 2 * HW + 4.8, 6.5, 8.5, 3)
    billboard(m, p0[0] + fx * 0.3, p0[1] + fz * 0.3, -fx, -fz, 2 * HW + 4.8, 6.5, 8.5, 3)

    # ---- tribunas en la recta principal (lado exterior, z negativo)
    def tiers(base, along, out, length, rows, dd, dh, seat_col, roof=True):
        bx, bz = base
        axx, axz = along
        ox, oz = out
        conc_c = rgb(185, 180, 170)
        for r in range(rows):
            d0, d1, h = r * dd, (r + 1) * dd, (r + 1) * dh
            p = [(bx + ox * d0, bz + oz * d0), (bx + ox * d0 + axx * length, bz + oz * d0 + axz * length),
                 (bx + ox * d1 + axx * length, bz + oz * d1 + axz * length), (bx + ox * d1, bz + oz * d1)]
            m.quad((p[0][0], h, p[0][1]), (p[1][0], h, p[1][1]), (p[2][0], h, p[2][1]), (p[3][0], h, p[3][1]),
                   seat_col, ((8, 0, r), (8, length, r), (8, length, r + 1), (8, 0, r + 1)))
            m.quad((p[0][0], h - dh, p[0][1]), (p[1][0], h - dh, p[1][1]), (p[1][0], h, p[1][1]), (p[0][0], h, p[0][1]),
                   seat_col, ((8, 0, r - 0.35), (8, length, r - 0.35), (8, length, r), (8, 0, r)))
            for e in (0, 1):
                a, b = (p[0], p[3]) if e == 0 else (p[1], p[2])
                m.quad((a[0], 0, a[1]), (b[0], 0, b[1]), (b[0], h, b[1]), (a[0], h, a[1]), conc_c)
        D, Ht = rows * dd, rows * dh
        e0 = (bx + ox * D, bz + oz * D)
        e1 = (e0[0] + axx * length, e0[1] + axz * length)
        m.quad((e0[0], 0, e0[1]), (e1[0], 0, e1[1]), (e1[0], Ht + 1, e1[1]), (e0[0], Ht + 1, e0[1]), conc_c)
        if roof:
            rc = rgb(235, 235, 240)
            ry = Ht + 6
            a0 = (bx + ox * D * 0.35, bz + oz * D * 0.35)
            a1 = (bx + ox * (D + 1), bz + oz * (D + 1))
            m.box((a0[0], ry, a0[1]), (axx * length, 0, axz * length), (0, 0.6, 0),
                  (a1[0] - a0[0], 0, a1[1] - a0[1]), rc)
            for k in range(0, int(length) + 1, 25):
                px, pz = e0[0] + axx * k, e0[1] + axz * k
                m.aabb(px - 0.4, 0, pz - 0.4, px + 0.4, ry, pz + 0.4, rgb(150, 150, 155))

    st0 = track.ctrl[0]
    tiers((st0[0] + 60, st0[1] - (LIMIT + 4)), (1, 0), (0, -1), 760, 12, 2.6, 1.6, rgb(40, 90, 200))
    tiers((st0[0] + 840, st0[1] - (LIMIT + 4)), (1, 0), (0, -1), 160, 8, 2.6, 1.6, rgb(220, 120, 30), roof=False)

    # edificio de pits (interior de la recta)
    pz0 = st0[1] + LIMIT + 8
    px0 = st0[0] + 120
    m.aabb(px0, 0, pz0, px0 + 420, 9, pz0 + 16, rgb(235, 235, 230), {"top": rgb(120, 120, 125)})
    m.aabb(px0 + 2, 4.5, pz0 - 0.2, px0 + 418, 7.5, pz0, rgb(40, 70, 110))
    for k in range(20):
        gx = px0 + 6 + k * 20.6
        m.aabb(gx, 0, pz0 - 0.1, gx + 14, 3.6, pz0, rgb(60, 60, 65))
    for k in range(3):
        billboard(m, px0 + 70 + k * 140, pz0 - 0.3, 0, 1, 30, 9.5, 13.5, k)

    # ---- estadio: graderio circular alrededor de la seccion del estadio
    rows, dd, dh = 14, 3.0, 1.8
    seat_cols = [rgb(230, 110, 30), rgb(30, 80, 190), rgb(240, 200, 40), rgb(200, 40, 40)]
    nseg = 56
    track_polar_r = np.hypot(pts[:, 0] - sc[0], pts[:, 1] - sc[1])
    track_polar_a = np.arctan2(pts[:, 1] - sc[1], pts[:, 0] - sc[0])
    rmax = r0 + rows * dd
    conc_c = rgb(185, 180, 170)
    for j in range(nseg):
        a0, a1 = j / nseg * TAU - math.pi, (j + 1) / nseg * TAU - math.pi
        am = (a0 + a1) / 2
        near_ang = np.abs((track_polar_a - am + math.pi) % TAU - math.pi) < (a1 - a0) * 1.2 + 0.05
        near_rad = (track_polar_r > r0 - LIMIT - 4) & (track_polar_r < rmax + LIMIT + 4)
        if np.any(near_ang & near_rad):
            continue
        col = seat_cols[(j // 4) % 4]
        c0, s0_, c1, s1_ = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        arc0 = r0 * a0
        for r in range(rows):
            ri, ro, h = r0 + r * dd, r0 + (r + 1) * dd, (r + 1) * dh
            P = lambda rad, c, s, y: (sc[0] + rad * c, y, sc[1] + rad * s)
            u0, u1 = ri * a0 - arc0, ri * a1 - arc0
            m.quad(P(ri, c0, s0_, h), P(ri, c1, s1_, h), P(ro, c1, s1_, h), P(ro, c0, s0_, h), col,
                   ((8, u0, r), (8, u1, r), (8, u1, r + 1), (8, u0, r + 1)))
            m.quad(P(ri, c0, s0_, h - dh), P(ri, c1, s1_, h - dh), P(ri, c1, s1_, h), P(ri, c0, s0_, h), col,
                   ((8, u0, r - 0.35), (8, u1, r - 0.35), (8, u1, r), (8, u0, r)))
        H = rows * dh
        P = lambda rad, c, s, y: (sc[0] + rad * c, y, sc[1] + rad * s)
        m.quad(P(rmax, c0, s0_, 0), P(rmax, c1, s1_, 0), P(rmax, c1, s1_, H + 2), P(rmax, c0, s0_, H + 2), conc_c)
        # techo
        ry = H + 9
        rc = rgb(240, 240, 245)
        rin = r0 + rows * dd * 0.4
        m.quad(P(rin, c0, s0_, ry), P(rin, c1, s1_, ry), P(rmax + 2, c1, s1_, ry), P(rmax + 2, c0, s0_, ry), rc)
        m.quad(P(rin + 0.3, c0, s0_, ry - 0.85), P(rin + 0.3, c1, s1_, ry - 0.85), P(rin + 5, c1, s1_, ry - 0.85),
               P(rin + 5, c0, s0_, ry - 0.85), rgb(255, 140, 50), ((11, 0, 0),) * 4)
        for kk in range(3):
            rad = rin + 9 + kk * 9
            m.quad(P(rad, c0, s0_, ry - 0.6), P(rad, c1, s1_, ry - 0.6), P(rad + 0.6, c1, s1_, ry - 0.6),
                   P(rad + 0.6, c0, s0_, ry - 0.6), rgb(255, 170, 80), ((11, 0, 0),) * 4)
        m.quad(P(rin, c0, s0_, ry - 0.8), P(rin, c1, s1_, ry - 0.8), P(rin, c1, s1_, ry), P(rin, c0, s0_, ry), rc)
        if j % 2 == 0:
            x, _, z = P(rmax + 1, c0, s0_, 0)
            m.aabb(x - 0.5, 0, z - 0.5, x + 0.5, ry, z + 0.5, rgb(150, 150, 155))
        # tapas laterales de la grada (en los huecos de entrada/salida)
        for (cc, ss_), nb in (((c0, s0_), j - 1), ((c1, s1_), j + 1)):
            am2 = ((nb % nseg) + 0.5) / nseg * TAU - math.pi
            na = np.abs((track_polar_a - am2 + math.pi) % TAU - math.pi) < (a1 - a0) * 1.2 + 0.05
            if np.any(na & near_rad):
                for r in range(rows):
                    ri, ro, h = r0 + r * dd, r0 + (r + 1) * dd, (r + 1) * dh
                    m.quad(P(ri, cc, ss_, 0), P(ro, cc, ss_, 0), P(ro, cc, ss_, h), P(ri, cc, ss_, h), conc_c)
                m.quad(P(r0 + rows * dd * 0.4, cc, ss_, H), P(rmax, cc, ss_, H), P(rmax, cc, ss_, ry),
                       P(r0 + rows * dd * 0.4, cc, ss_, ry), rgb(160, 160, 165))

    keep_out, info = build_extras(m, track, rnd, sc, r0, rmax, st0)

    # ---- arboles low-poly
    def tree(x, z, s):
        trunk = rgb(110, 75, 40)
        leaf = [rgb(40, 120, 50), rgb(55, 140, 45), rgb(35, 105, 60)][rnd.randrange(3)]
        k = 6
        for a in range(k):
            t0, t1 = a / k * TAU, (a + 1) / k * TAU
            r = 0.4 * s
            m.quad((x + r * math.cos(t0), 0, z + r * math.sin(t0)), (x + r * math.cos(t1), 0, z + r * math.sin(t1)),
                   (x + r * math.cos(t1), 2.2 * s, z + r * math.sin(t1)), (x + r * math.cos(t0), 2.2 * s, z + r * math.sin(t0)),
                   trunk)
        for y0, y1, rr in ((1.8, 6.0, 2.6), (4.0, 8.5, 1.9)):
            for a in range(k):
                t0, t1 = a / k * TAU + y0, (a + 1) / k * TAU + y0
                lc = tuple(v * (0.85 + 0.3 * (a % 2)) for v in leaf)
                m.tri((x + rr * s * math.cos(t0), y0 * s, z + rr * s * math.sin(t0)),
                      (x, y1 * s, z),
                      (x + rr * s * math.cos(t1), y0 * s, z + rr * s * math.sin(t1)), (lc,) * 3, ((0, 0, 0),) * 3)

    lo = pts.min(axis=0) - 220
    hi = pts.max(axis=0) + 220
    cand = np.column_stack([np.array([rnd.uniform(lo[0], hi[0]) for _ in range(2600)]),
                            np.array([rnd.uniform(lo[1], hi[1]) for _ in range(2600)])])
    dist = track.nearest_dist(cand)
    placed = 0
    for (x, z), d in zip(cand, dist):
        if d < LIMIT + 12 or math.hypot(x - sc[0], z - sc[1]) < rmax + 15:
            continue
        if st0[0] - 20 < x < st0[0] + 1020 and st0[1] - 75 < z < st0[1] + LIMIT + 35:
            continue
        if any(math.hypot(x - kx, z - kz) < kr for kx, kz, kr in keep_out):
            continue
        tree(x, z, rnd.uniform(0.9, 1.6))
        placed += 1
        if placed > 950:
            break

    # ---- ciudad lejana
    pal = [rgb(200, 190, 175), rgb(170, 175, 185), rgb(215, 200, 170), rgb(150, 150, 160), rgb(190, 160, 140)]
    for _ in range(260):
        a = rnd.uniform(0, TAU)
        r = rnd.uniform(1500, 2700)
        x, z = cxm + r * math.cos(a), czm + r * math.sin(a)
        w, d, h = rnd.uniform(20, 60), rnd.uniform(20, 60), rnd.choice([12, 18, 25, 35, 50, 70, 110])
        m.aabb(x - w / 2, 0, z - d / 2, x + w / 2, h, z + d / 2, pal[rnd.randrange(len(pal))])

    # ---- montanas y volcanes
    def mountain(x, z, r, h, base, snow=None, k=9):
        top = (x, h, z)
        for a in range(k):
            t0, t1 = a / k * TAU, (a + 1) / k * TAU
            mid = 0.55
            p0 = (x + r * math.cos(t0), 0, z + r * math.sin(t0))
            p1 = (x + r * math.cos(t1), 0, z + r * math.sin(t1))
            q0 = (x + r * (1 - mid) * math.cos(t0 + 0.2), h * mid, z + r * (1 - mid) * math.sin(t0 + 0.2))
            q1 = (x + r * (1 - mid) * math.cos(t1 + 0.2), h * mid, z + r * (1 - mid) * math.sin(t1 + 0.2))
            m.quad(p0, p1, q1, q0, base)
            m.tri(q0, q1, top, (snow or base,) * 3, ((0, 0, 0),) * 3)

    for k in range(26):
        a = k / 26 * TAU + rnd.uniform(-0.1, 0.1)
        r = rnd.uniform(3300, 4200)
        mountain(cxm + r * math.cos(a), czm + r * math.sin(a), rnd.uniform(700, 1300), rnd.uniform(250, 600),
                 rgb(90, 120, 85))
    # dos volcanes nevados al oriente
    mountain(cxm + 4300, czm + 900, 1700, 1300, rgb(110, 105, 100), rgb(245, 245, 250), 11)
    mountain(cxm + 4000, czm - 900, 2000, 1050, rgb(105, 105, 100), rgb(245, 245, 250), 11)

    # ---- nubes
    for _ in range(40):
        a = rnd.uniform(0, TAU)
        r = rnd.uniform(500, 3000)
        x, z, y = cxm + r * math.cos(a), czm + r * math.sin(a), rnd.uniform(260, 420)
        for _b in range(3):
            w = rnd.uniform(60, 140)
            ox, oz = rnd.uniform(-60, 60), rnd.uniform(-40, 40)
            m.aabb(x + ox - w / 2, y, z + oz - w / 3, x + ox + w / 2, y + rnd.uniform(15, 30), z + oz + w / 3,
                   (1, 1, 1))
    return m.array(), info


# ---------------------------------------------------------------- pilotos y vehiculos
PILOTS = [
    {"name": "MAMÁ LUCHA", "brand": "Bodega Aurrera", "vehicle": "pickup", "vname": "Camioneta del mandado",
     "body": (35, 140, 65), "accent": (250, 210, 30), "map": (35, 160, 65), "sign": 4, "driver": "lucha",
     "vmax": 88.0, "accel": 11.2, "grip": 26.5, "mass": 1.15},
    {"name": "DR. SIMI", "brand": "Farmacias Similares", "vehicle": "van", "vname": "Van de farmacia",
     "body": (245, 245, 248), "accent": (30, 80, 190), "map": (60, 120, 255), "sign": 5, "driver": "simi",
     "vmax": 88.0, "accel": 11.4, "grip": 26.0, "mass": 1.1},
    {"name": "EL DEL OXXO", "brand": "OXXO", "vehicle": "hatch", "vname": "Hatchback de reparto",
     "body": (215, 30, 35), "accent": (250, 200, 20), "map": (230, 40, 40), "sign": 6, "driver": "oxxo",
     "vmax": 85.0, "accel": 12.2, "grip": 27.0, "mass": 0.9},
    {"name": "OSITO BIMBO", "brand": "Bimbo", "vehicle": "truck", "vname": "Camión panadero",
     "body": (245, 245, 248), "accent": (25, 70, 170), "map": (255, 255, 255), "sign": 7, "driver": "bear",
     "vmax": 92.0, "accel": 10.4, "grip": 25.0, "mass": 1.5},
    {"name": "EL GASERO", "brand": "Gas a domicilio", "vehicle": "gas", "vname": "Pipa de gas",
     "body": (235, 235, 238), "accent": (215, 35, 35), "map": (255, 140, 30), "sign": 8, "driver": "gas",
     "vmax": 91.0, "accel": 10.6, "grip": 25.5, "mass": 1.6},
    {"name": "PILOTO GTC", "brand": "GTC Racing", "vehicle": "sport", "vname": "Deportivo GTC",
     "body": (250, 205, 25), "accent": (25, 25, 25), "map": (250, 215, 30), "sign": 9, "driver": "helmet",
     "vmax": 90.0, "accel": 11.0, "grip": 26.0, "mass": 1.0},
]

# nombre, habilidad, agresividad en curva, (ayuda si van atras, freno si van adelante), descripcion, frenada
DIFFS = [
    ("FÁCIL", 0.82, 0.72, (0.00, 0.08), "Rivales tranquilos, ideal para aprender", 0.6),
    ("NORMAL", 0.92, 0.84, (0.02, 0.03), "Un buen reto", 0.7),
    ("DIFÍCIL", 0.97, 0.90, (0.04, 0.00), "No perdonan errores", 0.8),
    ("LEYENDA", 1.05, 0.95, (0.07, 0.00), "Solo para pilotos de verdad", 0.9),
]

VEH = {
    "sport": dict(length=4.5, width=1.9, radius=1.2, cam_d=9.5, cam_h=3.6, hood=1.5),
    "hatch": dict(length=3.9, width=1.8, radius=1.15, cam_d=9.0, cam_h=3.6, hood=1.75),
    "pickup": dict(length=4.9, width=1.96, radius=1.3, cam_d=10.5, cam_h=4.0, hood=2.05),
    "van": dict(length=4.8, width=2.0, radius=1.3, cam_d=11.0, cam_h=4.4, hood=2.05),
    "truck": dict(length=6.0, width=2.3, radius=1.5, cam_d=13.5, cam_h=5.4, hood=2.7),
    "gas": dict(length=6.0, width=2.2, radius=1.5, cam_d=13.5, cam_h=5.2, hood=2.7),
}

GLASS = (35 / 255, 50 / 255, 75 / 255)
SKIN = (228 / 255, 178 / 255, 138 / 255)
BLACK = (0.08, 0.08, 0.09)


def sphere(m, c, r, col, sx=1.0, sy=1.0, sz=1.0, seg=8, rings=6, colfn=None):
    cx, cy, cz = c

    def P(t, j):
        ph = j / seg * TAU
        return (cx + r * sx * math.sin(t) * math.cos(ph), cy + r * sy * math.cos(t),
                cz + r * sz * math.sin(t) * math.sin(ph))
    for i in range(rings):
        t0, t1 = math.pi * i / rings, math.pi * (i + 1) / rings
        for j in range(seg):
            tm, pm = (t0 + t1) / 2, (j + 0.5) / seg * TAU
            d = (math.sin(tm) * math.cos(pm), math.cos(tm), math.sin(tm) * math.sin(pm))
            m.quad(P(t0, j), P(t0, j + 1), P(t1, j + 1), P(t1, j), colfn(d) if colfn else col)


def prism_x(m, x0, x1, cy, cz, r, col, k=10):
    for a in range(k):
        t0, t1 = a / k * TAU, (a + 1) / k * TAU
        y0, z0 = cy + r * math.cos(t0), cz + r * math.sin(t0)
        y1, z1 = cy + r * math.cos(t1), cz + r * math.sin(t1)
        m.quad((x0, y0, z0), (x1, y0, z0), (x1, y1, z1), (x0, y1, z1), col)
        for x in (x0, x1):
            m.tri((x, cy, cz), (x, y0, z0), (x, y1, z1), (col,) * 3, ((0, 0, 0),) * 3)


def prism_y(m, cx, cz, y0, y1, r, col, k=8):
    for a in range(k):
        t0, t1 = a / k * TAU, (a + 1) / k * TAU
        p0 = (cx + r * math.cos(t0), cz + r * math.sin(t0))
        p1 = (cx + r * math.cos(t1), cz + r * math.sin(t1))
        m.quad((p0[0], y0, p0[1]), (p1[0], y0, p1[1]), (p1[0], y1, p1[1]), (p0[0], y1, p0[1]), col)
        m.tri((cx, y1, cz), (p0[0], y1, p0[1]), (p1[0], y1, p1[1]), (col,) * 3, ((0, 0, 0),) * 3)


def wheels(m, xs, z_in, z_out, r):
    k = 8
    tire, hub = rgb(20, 20, 22), rgb(150, 150, 155)
    for wx in xs:
        for side in (-1, 1):
            z0, z1 = side * z_in, side * z_out
            for a in range(k):
                t0, t1 = a / k * TAU, (a + 1) / k * TAU
                p0 = (wx + r * math.cos(t0), r + r * math.sin(t0))
                p1 = (wx + r * math.cos(t1), r + r * math.sin(t1))
                m.quad((p0[0], p0[1], z0), (p1[0], p1[1], z0), (p1[0], p1[1], z1), (p0[0], p0[1], z1), tire)
                m.tri((wx, r, z1), (p0[0], p0[1], z1), (p1[0], p1[1], z1), (tire,) * 3, ((0, 0, 0),) * 3)
                h0 = (wx + 0.5 * r * math.cos(t0), r + 0.5 * r * math.sin(t0))
                h1 = (wx + 0.5 * r * math.cos(t1), r + 0.5 * r * math.sin(t1))
                zh = z1 + side * 0.01
                m.tri((wx, r, zh), (h0[0], h0[1], zh), (h1[0], h1[1], zh), (hub,) * 3, ((0, 0, 0),) * 3)


def decal(m, cx, cz, fx, fz, width, y0, y1, k):
    """Rectangulo con un letrero del atlas. (fx,fz) = hacia donde mira quien lo lee."""
    rx, rz = -fz, fx
    h = width / 2
    lx, lz, qx, qz = cx - rx * h, cz - rz * h, cx + rx * h, cz + rz * h
    vt, vb = k / 16 + 0.002, (k + 1) / 16 - 0.002
    m.quad((lx, y0, lz), (qx, y0, qz), (qx, y1, qz), (lx, y1, lz), (1, 1, 1),
           ((7, 0, vb), (7, 0.5, vb), (7, 0.5, vt), (7, 0, vt)))


def side_decals(m, xc, zhalf, width, y0, y1, k):
    decal(m, xc, -zhalf, 0, 1, width, y0, y1, k)
    decal(m, xc, zhalf, 0, -1, width, y0, y1, k)


def lights(m, xf, xr, yf, yr, zs=(0.5, 0.85)):
    for side in (-1, 1):
        z0, z1 = sorted((side * zs[0], side * zs[1]))
        m.aabb(xf - 0.05, yf, z0, xf + 0.05, yf + 0.14, z1, rgb(255, 250, 200))
        m.aabb(xr - 0.05, yr, z0, xr + 0.05, yr + 0.14, z1, rgb(230, 20, 20))


def eyes(m, x, y, z, col=BLACK, gap=0.1, s=0.05):
    for zz in (-gap, gap):
        m.aabb(x, y - s, z + zz - s, x + 0.04, y + s, z + zz + s, col)


def driver(m, kind, x, y, z):
    """Personaje sentado: (x, y) = posicion del asiento en coordenadas del vehiculo."""
    hc = (x, y + 0.85, z)
    hy = hc[1]

    def torso(col, arms=None):
        arms = arms or col
        m.aabb(x - 0.25, y, z - 0.3, x + 0.2, y + 0.55, z + 0.3, col)
        m.aabb(x + 0.05, y + 0.3, z - 0.38, x + 0.45, y + 0.45, z - 0.26, arms)
        m.aabb(x + 0.05, y + 0.3, z + 0.26, x + 0.45, y + 0.45, z + 0.38, arms)
        m.aabb(x + 0.45, y + 0.22, z - 0.2, x + 0.5, y + 0.55, z + 0.2, rgb(30, 30, 30))

    if kind == "lucha":
        mask, trim = rgb(225, 40, 140), rgb(250, 210, 30)
        torso(rgb(40, 140, 70), SKIN)
        m.aabb(x + 0.2, y + 0.02, z - 0.22, x + 0.23, y + 0.5, z + 0.22, trim)

        def f(d):
            dx, dy, dz = d
            if dx > 0.45 and -0.05 < dy < 0.5 and 0.1 < abs(dz) < 0.62:
                return (1, 1, 1)
            if dx > 0.5 and dy < -0.3:
                return SKIN
            if abs(dz) < 0.2 and dy > -0.1 and dx < 0.9:
                return trim
            return mask
        sphere(m, hc, 0.3, mask, colfn=f)
        eyes(m, x + 0.27, hy + 0.08, z, gap=0.11, s=0.035)
    elif kind == "simi":
        torso(rgb(245, 245, 245))
        white = (0.97, 0.97, 0.97)
        sphere(m, hc, 0.31, SKIN, colfn=lambda d: white if (d[1] > 0.3 and d[0] < 0.5) or d[0] < -0.35 else SKIN)
        m.aabb(x + 0.24, hy - 0.17, z - 0.18, x + 0.34, hy - 0.06, z + 0.18, white)           # bigote
        for zz in (-0.11, 0.11):                                                             # lentes
            m.aabb(x + 0.26, hy - 0.01, z + zz - 0.07, x + 0.33, hy + 0.11, z + zz + 0.07, rgb(30, 30, 30))
        m.aabb(x + 0.29, hy - 0.06, z - 0.035, x + 0.37, hy + 0.02, z + 0.035, rgb(235, 150, 120))
    elif kind == "oxxo":
        red, yel = rgb(215, 30, 35), rgb(250, 200, 20)
        torso(red, SKIN)
        m.aabb(x + 0.2, y + 0.18, z - 0.3, x + 0.215, y + 0.3, z + 0.3, yel)

        def f(d):
            if d[1] > 0.45:
                return red
            if d[0] < -0.1 and d[1] > -0.35:
                return rgb(30, 22, 15)
            return SKIN
        sphere(m, hc, 0.3, SKIN, colfn=f)
        m.aabb(x + 0.12, hy + 0.2, z - 0.22, x + 0.5, hy + 0.25, z + 0.22, red)
        eyes(m, x + 0.27, hy + 0.05, z)
    elif kind == "bear":
        fur = rgb(245, 245, 240)
        torso(fur)
        m.aabb(x + 0.2, y + 0.02, z - 0.2, x + 0.22, y + 0.5, z + 0.2, (1, 1, 1))
        sphere(m, hc, 0.32, fur)
        for zz in (-0.22, 0.22):
            sphere(m, (x - 0.02, hy + 0.25, z + zz), 0.11, fur, seg=6, rings=4)
        sphere(m, (x + 0.26, hy - 0.08, z), 0.13, rgb(250, 232, 210), seg=6, rings=4)
        m.aabb(x + 0.36, hy - 0.05, z - 0.05, x + 0.41, hy + 0.02, z + 0.05, BLACK)
        eyes(m, x + 0.26, hy + 0.09, z, gap=0.11)
        prism_y(m, x, z, hy + 0.2, hy + 0.48, 0.19, (1, 1, 1))                             # gorro de chef
        sphere(m, (x, hy + 0.52, z), 0.25, (1, 1, 1), sy=0.55, seg=8, rings=4)
    elif kind == "gas":
        blue = rgb(40, 80, 160)
        torso(blue, SKIN)
        sphere(m, hc, 0.3, SKIN, colfn=lambda d: blue if d[1] > 0.42 else SKIN)
        m.aabb(x + 0.12, hy + 0.18, z - 0.22, x + 0.46, hy + 0.23, z + 0.22, blue)
        m.aabb(x + 0.25, hy - 0.15, z - 0.15, x + 0.33, hy - 0.07, z + 0.15, rgb(45, 28, 15))
        eyes(m, x + 0.27, hy + 0.05, z)
    else:  # casco de piloto
        yel = rgb(250, 205, 25)
        torso(rgb(25, 25, 30))

        def f(d):
            if d[0] > 0.35 and -0.2 < d[1] < 0.35:
                return rgb(20, 25, 35)
            if abs(d[2]) < 0.14 and d[0] < 0.4:
                return (1, 1, 1)
            return yel
        sphere(m, hc, 0.32, yel, colfn=f)


def build_vehicle(p):
    m = Mesh()
    kind = p["vehicle"]
    c, a = rgb(*p["body"]), rgb(*p["accent"])
    dark = tuple(v * 0.5 for v in c)
    k = p["sign"]
    d = p["driver"]
    if kind == "sport":
        m.hexa([(-2.2, 0.28, -0.95), (2.25, 0.28, -0.85), (2.25, 0.28, 0.85), (-2.2, 0.28, 0.95),
                (-2.2, 0.78, -0.95), (2.25, 0.5, -0.8), (2.25, 0.5, 0.8), (-2.2, 0.78, 0.95)], c,
               {"bottom": dark, "back": dark})
        m.hexa([(-2.21, 0.5, -0.22), (2.26, 0.4, -0.22), (2.26, 0.4, 0.22), (-2.21, 0.5, 0.22),
                (-2.21, 0.8, -0.22), (2.26, 0.52, -0.22), (2.26, 0.52, 0.22), (-2.21, 0.8, 0.22)], a)
        m.quad((0.75, 0.66, -0.75), (0.75, 0.66, 0.75), (0.3, 1.15, 0.7), (0.3, 1.15, -0.7), GLASS)
        m.aabb(-1.05, 0.75, -0.55, -0.9, 1.45, 0.55, a)
        m.aabb(-2.35, 1.05, -1.0, -1.85, 1.15, 1.0, a)
        for zz in (-0.6, 0.5):
            m.aabb(-2.15, 0.7, zz, -2.0, 1.05, zz + 0.1, a)
        side_decals(m, -0.6, 0.94, 2.0, 0.32, 0.7, k)
        lights(m, 2.24, -2.21, 0.36, 0.5)
        wheels(m, (-1.45, 1.45), 0.72, 1.02, 0.38)
        driver(m, d, -0.35, 0.6, 0.0)
    elif kind == "hatch":
        m.hexa([(-1.9, 0.3, -0.9), (1.95, 0.3, -0.86), (1.95, 0.3, 0.86), (-1.9, 0.3, 0.9),
                (-1.9, 1.0, -0.9), (1.95, 0.78, -0.84), (1.95, 0.78, 0.84), (-1.9, 1.0, 0.9)], c, {"bottom": dark})
        m.aabb(-1.92, 0.5, -0.93, 1.97, 0.6, 0.93, a)
        m.quad((1.05, 0.86, -0.8), (1.05, 0.86, 0.8), (0.6, 1.88, 0.75), (0.6, 1.88, -0.75), GLASS)
        for px in (-1.65, 0.55):
            for sz in (-0.8, 0.8):
                m.aabb(px - 0.06, 0.95, sz - 0.06, px + 0.06, 1.9, sz + 0.06, c)
        m.aabb(-1.75, 1.88, -0.85, 0.65, 1.98, 0.85, c)
        m.aabb(-1.0, 1.98, -0.5, 0.0, 2.35, 0.5, a)
        side_decals(m, -0.5, 0.51, 1.0, 2.01, 2.32, k)
        side_decals(m, -0.55, 0.9, 1.7, 0.62, 0.9, k)
        lights(m, 1.94, -1.91, 0.45, 0.7, (0.45, 0.8))
        wheels(m, (-1.25, 1.25), 0.68, 0.96, 0.36)
        driver(m, d, -0.4, 0.6, 0.0)
    elif kind == "pickup":
        m.hexa([(-2.45, 0.38, -0.98), (2.45, 0.38, -0.95), (2.45, 0.38, 0.95), (-2.45, 0.38, 0.98),
                (-2.45, 1.0, -0.98), (2.45, 0.9, -0.95), (2.45, 0.9, 0.95), (-2.45, 1.0, 0.98)], c, {"bottom": dark})
        m.aabb(-2.47, 0.58, -1.0, 2.47, 0.68, 1.0, a)
        m.quad((1.45, 0.92, -0.88), (1.45, 0.92, 0.88), (1.05, 2.0, 0.85), (1.05, 2.0, -0.85), GLASS)
        for px in (-0.15, 1.05):
            for sz in (-0.88, 0.88):
                m.aabb(px - 0.06, 0.95, sz - 0.06, px + 0.06, 2.02, sz + 0.06, c)
        m.aabb(-0.25, 2.0, -0.95, 1.12, 2.1, 0.95, c)
        for sz in (-0.94, 0.94):
            m.aabb(-2.45, 1.0, sz - 0.04, -0.3, 1.4, sz + 0.04, c)
        m.aabb(-2.45, 1.0, -0.98, -2.37, 1.4, 0.98, c)
        m.aabb(-0.38, 1.0, -0.98, -0.3, 1.4, 0.98, c)
        m.aabb(-2.2, 1.0, -0.8, -1.6, 1.5, -0.15, rgb(250, 210, 30))     # despensa
        m.aabb(-1.5, 1.0, -0.75, -0.9, 1.42, -0.1, rgb(245, 245, 245))
        m.aabb(-2.2, 1.0, 0.1, -1.55, 1.35, 0.8, rgb(200, 120, 50))
        sphere(m, (-1.05, 1.3, 0.45), 0.3, rgb(40, 130, 50), seg=8, rings=5)
        side_decals(m, 0.45, 0.975, 1.3, 0.7, 0.96, k)
        lights(m, 2.44, -2.46, 0.6, 0.75)
        wheels(m, (-1.6, 1.6), 0.72, 1.04, 0.42)
        driver(m, d, 0.45, 0.75, 0.0)
    elif kind == "van":
        m.aabb(-2.4, 0.4, -1.0, 0.75, 2.3, 1.0, c)
        m.aabb(-2.42, 0.85, -1.02, 0.77, 1.1, 1.02, a)
        m.hexa([(0.75, 0.4, -1.0), (2.4, 0.4, -0.95), (2.4, 0.4, 0.95), (0.75, 0.4, 1.0),
                (0.75, 1.1, -1.0), (2.4, 0.95, -0.95), (2.4, 0.95, 0.95), (0.75, 1.1, 1.0)], c)
        m.quad((2.05, 1.0, -0.92), (2.05, 1.0, 0.92), (1.7, 2.22, 0.9), (1.7, 2.22, -0.9), GLASS)
        for sz in (-0.93, 0.93):
            m.aabb(1.6, 1.05, sz - 0.06, 1.72, 2.25, sz + 0.06, c)
        m.aabb(0.75, 2.2, -1.0, 1.78, 2.3, 1.0, c)
        m.aabb(1.0, 2.3, -0.6, 1.4, 2.45, 0.0, rgb(230, 30, 30))
        m.aabb(1.0, 2.3, 0.0, 1.4, 2.45, 0.6, rgb(40, 90, 230))
        side_decals(m, -0.85, 1.01, 2.6, 1.25, 2.15, k)
        decal(m, -2.41, 0, 1, 0, 1.6, 1.3, 1.9, k)
        lights(m, 2.4, -2.41, 0.6, 0.6)
        wheels(m, (-1.55, 1.55), 0.74, 1.06, 0.42)
        driver(m, d, 1.15, 0.8, 0.0)
    else:  # camiones: panadero o pipa de gas
        m.aabb(-3.0, 0.45, -1.0, 3.0, 0.8, 1.0, rgb(40, 40, 45))
        m.aabb(1.6, 0.8, -1.1, 3.0, 1.55, 1.1, c)
        m.aabb(2.98, 0.85, -0.9, 3.04, 1.3, 0.9, rgb(30, 30, 35))
        m.quad((3.0, 1.55, -1.0), (3.0, 1.55, 1.0), (2.85, 2.85, 1.0), (2.85, 2.85, -1.0), GLASS)
        for px in (1.68, 2.85):
            for sz in (-1.03, 1.03):
                m.aabb(px - 0.07, 1.5, sz - 0.07, px + 0.07, 2.88, sz + 0.07, c)
        m.aabb(1.6, 2.85, -1.1, 2.97, 3.0, 1.1, a)
        if kind == "truck":
            m.aabb(-3.0, 0.8, -1.15, 1.5, 3.3, 1.15, c)
            m.aabb(-3.02, 0.8, -1.17, 1.52, 1.1, 1.17, a)
            side_decals(m, -0.75, 1.17, 4.0, 1.4, 3.0, k)
            decal(m, -3.01, 0, 1, 0, 2.0, 1.6, 2.6, k)
        else:
            prism_x(m, -3.0, 1.4, 1.75, 0, 0.95, c, k=12)
            for x0 in (-2.4, -0.4, 1.0):
                prism_x(m, x0, x0 + 0.2, 1.75, 0, 0.98, a, k=12)
            for x0 in (-2.5, 0.6):
                m.aabb(x0, 0.8, -0.6, x0 + 0.3, 1.0, 0.6, rgb(60, 60, 65))
            side_decals(m, -1.4, 0.99, 1.7, 1.45, 2.05, k)
        lights(m, 3.02, -3.01, 1.0, 0.9, (0.6, 0.95))
        wheels(m, (2.2, -1.4, -2.3), 0.75, 1.1, 0.48)
        driver(m, d, 2.25, 1.2, 0.0)

    v = VEH[kind]
    sh = Mesh()
    hl, hw = v["length"] / 2 + 0.5, v["width"] / 2 + 0.4
    sh.quad((-hl, 0.15, -hw), (hl, 0.15, -hw), (hl, 0.15, hw), (-hl, 0.15, hw), (0, 0, 0),
            ((9, -1, -1), (9, 1, -1), (9, 1, 1), (9, -1, 1)))
    return m.array(), sh.array()


SPONSORS = [0, 4, 11, 5, 6, 13, 7, 8, 14, 15, 1, 2]


# ---------------------------------------------------------------- ambiente mexicano y concierto
# Horarios: cielo, niebla, sol, color del sol, luz ambiente, luz directa, tinte, luces de noche, faro, estrellas
TODS = [
    ("DÍA", dict(top=(0.25, 0.48, 0.85), fog=(0.72, 0.80, 0.90), sun=(0.35, -0.85, 0.40), sunc=(1.0, 0.9, 0.6),
                 amb=0.5, diff=0.55, tint=(1.0, 1.0, 1.0), night=0.0, head=0.0, stars=0.0)),
    ("ATARDECER", dict(top=(0.14, 0.10, 0.36), fog=(0.86, 0.52, 0.42), sun=(-0.74, -0.22, 0.63),
                       sunc=(1.0, 0.55, 0.25), amb=0.42, diff=0.6, tint=(1.0, 0.8, 0.64), night=1.0, head=0.15,
                       stars=0.0)),
    ("NOCHE", dict(top=(0.01, 0.01, 0.05), fog=(0.06, 0.06, 0.13), sun=(0.3, -0.6, -0.7), sunc=(0.6, 0.65, 0.8),
                   amb=0.28, diff=0.3, tint=(0.7, 0.78, 1.0), night=1.0, head=0.75, stars=1.0)),
]

SHIRTS = [rgb(220, 30, 40), rgb(30, 120, 60), rgb(245, 245, 245), rgb(30, 60, 160), rgb(250, 200, 30),
          rgb(20, 20, 25), rgb(240, 120, 20), rgb(150, 40, 160), rgb(230, 0, 124), rgb(60, 170, 220)]
PANTS = [rgb(30, 40, 80), rgb(25, 25, 28), rgb(90, 70, 50), rgb(60, 60, 70)]
SKINS = [rgb(228, 178, 138), rgb(198, 140, 100), rgb(160, 110, 75), rgb(120, 80, 55)]
HAIRS = [rgb(20, 15, 10), rgb(60, 35, 20), rgb(30, 25, 20), rgb(140, 100, 50)]
FLAG_G, FLAG_W, FLAG_R = rgb(0, 104, 71), rgb(245, 245, 245), rgb(206, 17, 38)
GUADA_UV = (0.5, 1.0, 0.0, 384 / 1024)
FLAG_UV = (0.5, 1.0, 400 / 1024, 546 / 1024)


def image_quad(m, cx, cz, fx, fz, width, y0, y1, uv):
    rx, rz = -fz, fx
    h = width / 2
    lx, lz, qx, qz = cx - rx * h, cz - rz * h, cx + rx * h, cz + rz * h
    u0, u1, vt, vb = uv
    m.quad((lx, y0, lz), (qx, y0, qz), (qx, y1, qz), (lx, y1, lz), (1, 1, 1),
           ((7, u0, vb), (7, u1, vb), (7, u1, vt), (7, u0, vt)))


def person(m, x, z, rnd, y0=0.0, s=1.0, arms_up=True, amp=None, flag=False, light=False, shirt=None, matid=10):
    """Aficionado low-poly que brinca (el shader lo anima con material 10)."""
    ph = rnd.uniform(0, TAU)
    am = rnd.uniform(0.12, 0.45) if amp is None else amp
    mv = (matid, ph, am)
    sh = shirt or rnd.choice(SHIRTS)
    pa, sk, ha = rnd.choice(PANTS), rnd.choice(SKINS), rnd.choice(HAIRS)

    def b(x0, ya, z0, x1, yb, z1, col, mat=mv):
        m.aabb(x + x0 * s, y0 + ya * s, z + z0 * s, x + x1 * s, y0 + yb * s, z + z1 * s, col, mat=mat)
    b(-0.2, 0, -0.12, 0.2, 0.8, 0.12, pa)
    b(-0.26, 0.8, -0.16, 0.26, 1.45, 0.16, sh)
    b(-0.14, 1.45, -0.14, 0.14, 1.75, 0.14, sk)
    b(-0.15, 1.72, -0.15, 0.15, 1.82, 0.15, ha)
    if arms_up:
        b(-0.38, 1.2, -0.07, -0.26, 2.05, 0.07, sh)
        b(0.26, 1.2, -0.07, 0.38, 2.05, 0.07, sh)
        b(-0.38, 2.05, -0.07, -0.26, 2.17, 0.07, sk)
        b(0.26, 2.05, -0.07, 0.38, 2.17, 0.07, sk)
    else:
        b(-0.36, 0.85, -0.07, -0.26, 1.42, 0.07, sh)
        b(0.26, 0.85, -0.07, 0.36, 1.42, 0.07, sh)
    if flag:
        b(0.3, 2.0, -0.02, 0.34, 2.9, 0.02, rgb(120, 90, 60))
        for k, col in enumerate((FLAG_G, FLAG_W, FLAG_R)):
            b(0.34 + k * 0.25, 2.45, -0.02, 0.59 + k * 0.25, 2.9, 0.02, col)
    if light:  # celular / encendedor (amplitud negativa = emisivo que parpadea)
        b(0.28, 2.17, -0.05, 0.38, 2.32, 0.05, (1.0, 0.97, 0.8), mat=(10, ph, -am))


def stand(m, base, along, out, length, rows, dd, dh, seat_col):
    bx, bz = base
    axx, axz = along
    ox, oz = out
    conc_c = rgb(185, 180, 170)
    for r in range(rows):
        d0, d1, h = r * dd, (r + 1) * dd, (r + 1) * dh
        p = [(bx + ox * d0, bz + oz * d0), (bx + ox * d0 + axx * length, bz + oz * d0 + axz * length),
             (bx + ox * d1 + axx * length, bz + oz * d1 + axz * length), (bx + ox * d1, bz + oz * d1)]
        m.quad((p[0][0], h, p[0][1]), (p[1][0], h, p[1][1]), (p[2][0], h, p[2][1]), (p[3][0], h, p[3][1]),
               seat_col, ((8, 0, r), (8, length, r), (8, length, r + 1), (8, 0, r + 1)))
        m.quad((p[0][0], h - dh, p[0][1]), (p[1][0], h - dh, p[1][1]), (p[1][0], h, p[1][1]), (p[0][0], h, p[0][1]),
               seat_col, ((8, 0, r - 0.35), (8, length, r - 0.35), (8, length, r), (8, 0, r)))
        for a, b in ((p[0], p[3]), (p[1], p[2])):
            m.quad((a[0], 0, a[1]), (b[0], 0, b[1]), (b[0], h, b[1]), (a[0], h, a[1]), conc_c)
    D, Ht = rows * dd, rows * dh
    e0 = (bx + ox * D, bz + oz * D)
    e1 = (e0[0] + axx * length, e0[1] + axz * length)
    m.quad((e0[0], 0, e0[1]), (e1[0], 0, e1[1]), (e1[0], Ht + 1, e1[1]), (e0[0], Ht + 1, e0[1]), conc_c)


def angel_monument(m, x, z):
    stone, dstone = rgb(235, 229, 214), rgb(200, 195, 180)
    gold, bronze = rgb(240, 190, 50), rgb(70, 60, 45)
    k = 24
    for j in range(k):
        a0, a1 = j / k * TAU, (j + 1) / k * TAU
        m.tri((x, 0.05, z), (x + 24 * math.cos(a1), 0.05, z + 24 * math.sin(a1)),
              (x + 24 * math.cos(a0), 0.05, z + 24 * math.sin(a0)), (rgb(60, 130, 50),) * 3, ((0, 0, 0),) * 3)
        fx, fz = x + 21 * math.cos(a0), z + 21 * math.sin(a0)
        fcol = (FLAG_G, FLAG_W, FLAG_R)[j % 3]
        m.aabb(fx - 1.2, 0, fz - 1.2, fx + 1.2, 0.6, fz + 1.2, fcol)
    m.aabb(x - 12, 0, z - 12, x + 12, 1.5, z + 12, dstone)
    m.aabb(x - 9, 1.5, z - 9, x + 9, 4, z + 9, stone)
    m.aabb(x - 5.5, 4, z - 5.5, x + 5.5, 10, z + 5.5, stone)
    for sx in (-1, 1):
        for sz in (-1, 1):
            px, pz = x + sx * 7.5, z + sz * 7.5
            m.aabb(px - 1.2, 4, pz - 1.2, px + 1.2, 5, pz + 1.2, dstone)
            m.aabb(px - 0.5, 5, pz - 0.4, px + 0.5, 7.2, pz + 0.4, bronze)
            sphere(m, (px, 7.6, pz), 0.4, bronze, seg=6, rings=4)
    m.aabb(x - 4, 10, z - 4, x + 4, 11.5, z + 4, dstone)
    prism_y(m, x, z, 11.5, 46, 2.0, stone, k=12)
    for yy in (18, 26, 34, 42):
        prism_y(m, x, z, yy, yy + 0.6, 2.25, dstone, k=12)
    m.aabb(x - 3, 46, z - 3, x + 3, 47.5, z + 3, dstone)
    prism_y(m, x, z, 47.5, 49.5, 1.4, gold, k=10)
    ay = 49.5
    m.aabb(x - 0.5, ay, z - 0.35, x + 0.5, ay + 1.6, z + 0.35, gold)
    m.aabb(x - 0.55, ay + 1.6, z - 0.4, x + 0.55, ay + 3.4, z + 0.4, gold)
    sphere(m, (x, ay + 3.9, z), 0.45, gold, seg=6, rings=4)
    m.aabb(x + 0.45, ay + 3.0, z - 0.15, x + 0.75, ay + 5.2, z + 0.15, gold)
    for j in range(8):
        a = j / 8 * TAU
        wx, wy = x + 0.6 + 0.55 * math.cos(a), ay + 5.8 + 0.55 * math.sin(a)
        m.aabb(wx - 0.12, wy - 0.12, z - 0.12, wx + 0.12, wy + 0.12, z + 0.12, gold)
    m.aabb(x - 1.4, ay + 2.6, z - 0.15, x - 0.5, ay + 2.9, z + 0.15, gold)
    for side in (-1, 1):
        m.tri((x - 0.2, ay + 3.2, z + side * 0.3), (x - 1.2, ay + 6.8, z + side * 2.8),
              (x - 0.6, ay + 1.4, z + side * 1.9), (gold,) * 3, ((0, 0, 0),) * 3)
        m.tri((x - 0.2, ay + 3.2, z + side * 0.3), (x - 1.7, ay + 5.0, z + side * 3.4),
              (x - 1.2, ay + 6.8, z + side * 2.8), (gold,) * 3, ((0, 0, 0),) * 3)


def flagpole(m, x, z, height=50.0):
    prism_y(m, x, z, 0, height, 0.45, rgb(205, 205, 210), k=8)
    m.aabb(x - 2, 0, z - 2, x + 2, 1, z + 2, rgb(190, 185, 175))
    sphere(m, (x, height + 0.5, z), 0.7, rgb(240, 190, 50), seg=6, rings=4)
    fw, fh, nx, ny = 26.0, 14.8, 16, 6
    u0, u1, v0, v1 = FLAG_UV
    top = height - 1
    for i in range(nx):
        for j in range(ny):
            xa, xb = x + 0.5 + fw * i / nx, x + 0.5 + fw * (i + 1) / nx
            yb, ya = top - fh * (j + 1) / ny, top - fh * j / ny
            ua, ub = u0 + (u1 - u0) * i / nx, u0 + (u1 - u0) * (i + 1) / nx
            vt, vb = v0 + (v1 - v0) * j / ny, v0 + (v1 - v0) * (j + 1) / ny
            m.quad((xa, yb, z), (xb, yb, z), (xb, ya, z), (xa, ya, z), (1, 1, 1),
                   ((12, ua, vb), (12, ub, vb), (12, ub, vt), (12, ua, vt)))


def papel_picado(m, pa, pb, h=7.5, rows=3):
    cols = [rgb(230, 0, 124), rgb(250, 140, 0), rgb(40, 170, 70), rgb(40, 110, 230), rgb(250, 215, 0),
            rgb(150, 50, 200)]
    for p in (pa, pb):
        prism_y(m, p[0], p[1], 0, h + 0.6, 0.18, rgb(120, 90, 60), k=6)
    dx, dz = pb[0] - pa[0], pb[1] - pa[1]
    ln = math.hypot(dx, dz)
    ux, uz = dx / ln, dz / ln
    n = int(ln / 0.9)
    for r in range(rows):
        yb = h - r * 1.0
        for i in range(n):
            t = (i + 0.5) / n
            px, pz = pa[0] + dx * t, pa[1] + dz * t
            y = yb - 1.4 * 4 * t * (1 - t)
            col = cols[(i + r) % len(cols)]
            mv = ((17, i * 0.7 + r, 0),) * 4
            m.quad((px - ux * 0.3, y - 0.75, pz - uz * 0.3), (px + ux * 0.3, y - 0.75, pz + uz * 0.3),
                   (px + ux * 0.3, y, pz + uz * 0.3), (px - ux * 0.3, y, pz - uz * 0.3), col, mv)


def draw_guadalupe(s, R):
    x0, y0, w, h = R
    cx, cy = x0 + w / 2, y0 + h * 0.5
    s.fill((150, 60, 40), R)
    for i in range(72):
        a = i / 72 * TAU
        col = (255, 200, 40) if i % 2 == 0 else (250, 150, 20)
        pygame.draw.line(s, col, (cx, cy), (cx + w * 0.47 * math.cos(a), cy + h * 0.47 * math.sin(a)), 4)
    pygame.draw.ellipse(s, (255, 225, 120), (cx - w * 0.3, cy - h * 0.38, w * 0.6, h * 0.76))
    mant = [(cx, y0 + h * 0.14), (cx + w * 0.13, y0 + h * 0.2), (cx + w * 0.2, y0 + h * 0.45),
            (cx + w * 0.18, y0 + h * 0.75), (cx + w * 0.1, y0 + h * 0.82), (cx - w * 0.1, y0 + h * 0.82),
            (cx - w * 0.18, y0 + h * 0.75), (cx - w * 0.2, y0 + h * 0.45), (cx - w * 0.13, y0 + h * 0.2)]
    pygame.draw.polygon(s, (20, 110, 100), mant)
    pygame.draw.polygon(s, (215, 110, 100), [(cx - w * 0.07, y0 + h * 0.3), (cx + w * 0.07, y0 + h * 0.3),
                                             (cx + w * 0.11, y0 + h * 0.8), (cx - w * 0.11, y0 + h * 0.8)])
    rnd = random.Random(5)
    for _ in range(40):
        px, py = cx + rnd.uniform(-w * 0.18, w * 0.18), y0 + rnd.uniform(h * 0.2, h * 0.78)
        if abs(px - cx) > w * 0.1:
            pygame.draw.circle(s, (255, 215, 60), (px, py), 2.5)
    for i in range(6):
        pygame.draw.line(s, (170, 70, 70), (cx - w * 0.06 + i * w * 0.025, y0 + h * 0.5),
                         (cx - w * 0.08 + i * w * 0.033, y0 + h * 0.79), 1)
    pygame.draw.ellipse(s, (200, 150, 110), (cx - w * 0.05, y0 + h * 0.19, w * 0.1, h * 0.08))
    pygame.draw.polygon(s, (215, 165, 125), [(cx - w * 0.035, y0 + h * 0.35), (cx + w * 0.035, y0 + h * 0.35),
                                             (cx, y0 + h * 0.44)])
    pygame.draw.rect(s, (40, 30, 40), (cx - w * 0.07, y0 + h * 0.47, w * 0.14, h * 0.012))
    pygame.draw.arc(s, (30, 30, 30), (cx - w * 0.17, y0 + h * 0.74, w * 0.34, h * 0.12), math.pi, TAU, 7)
    pygame.draw.polygon(s, (200, 40, 40), [(cx - w * 0.03, y0 + h * 0.87), (cx - w * 0.24, y0 + h * 0.83),
                                           (cx - w * 0.13, y0 + h * 0.91)])
    pygame.draw.polygon(s, (40, 140, 60), [(cx + w * 0.03, y0 + h * 0.87), (cx + w * 0.24, y0 + h * 0.83),
                                           (cx + w * 0.13, y0 + h * 0.91)])
    pygame.draw.ellipse(s, (210, 160, 120), (cx - w * 0.045, y0 + h * 0.845, w * 0.09, h * 0.05))
    for i in range(9):
        rx = x0 + 16 + i * (w - 32) / 8
        pygame.draw.circle(s, (30, 120, 40), (rx + 7, y0 + h - 8), 5)
        pygame.draw.circle(s, (210, 20, 50), (rx, y0 + h - 15), 10)
        pygame.draw.circle(s, (160, 10, 35), (rx, y0 + h - 15), 4)
    pygame.draw.rect(s, (255, 200, 40), R, 5)


def draw_flag(s, R):
    x0, y0, w, h = R
    t = w / 3
    s.fill((0, 104, 71), (x0, y0, t, h))
    s.fill((245, 245, 245), (x0 + t, y0, t, h))
    s.fill((206, 17, 38), (x0 + 2 * t, y0, w - 2 * t, h))
    cx, cy = x0 + w / 2, y0 + h / 2
    pygame.draw.arc(s, (40, 130, 60), (cx - 34, cy - 30, 68, 64), math.pi * 1.05, math.pi * 1.95, 4)
    pygame.draw.ellipse(s, (60, 140, 70), (cx - 14, cy + 6, 28, 18))
    pygame.draw.ellipse(s, (120, 75, 35), (cx - 10, cy - 18, 22, 28))
    pygame.draw.polygon(s, (100, 60, 25), [(cx - 6, cy - 10), (cx - 26, cy - 26), (cx - 16, cy + 2)])
    pygame.draw.polygon(s, (100, 60, 25), [(cx + 8, cy - 10), (cx + 26, cy - 28), (cx + 18, cy + 2)])
    pygame.draw.circle(s, (120, 75, 35), (cx - 8, cy - 20), 6)
    pygame.draw.line(s, (60, 120, 40), (cx - 12, cy - 18), (cx + 4, cy - 26), 3)


def make_sign_atlas():
    s = pygame.Surface((512, 1024))
    s.fill((0, 0, 0))
    defs = [((20, 45, 150), "GTC RACING", (255, 215, 40)),
            ((200, 25, 25), "GTC", (255, 255, 255)),
            ((20, 130, 60), "GRAN PREMIO CDMX", (255, 255, 255)),
            ((15, 15, 15), "META  -  GTC", (255, 215, 40)),
            ((25, 120, 55), "BODEGA AURRERA", (250, 210, 30)),
            ((255, 255, 255), "SIMI", (25, 70, 170)),
            ((215, 30, 35), "OXXO", (250, 200, 20)),
            ((255, 255, 255), "BIMBO", (25, 70, 170)),
            ((255, 255, 255), "¡EL GAAAS!", (210, 30, 30)),
            ((250, 205, 25), "GTC", (20, 20, 20)),
            ((10, 10, 10), "AVENGED SEVENFOLD", (235, 40, 40)),
            (None, "¡VIVA MÉXICO!", (20, 20, 20)),
            ((15, 30, 90), "ESTADIO GNP SEGUROS", (255, 255, 255)),
            ((228, 0, 124), "CDMX", (255, 255, 255)),
            ((10, 10, 10), "HOY EN VIVO · A7X", (255, 210, 40)),
            ((240, 120, 20), "TACOS EL GÜERO", (255, 255, 255))]
    for k, (bg, txt, fg) in enumerate(defs):
        r = pygame.Rect(0, k * 64, 256, 64)
        if bg is None:
            draw_flag(s, r)
        else:
            s.fill(bg, r)
        pygame.draw.rect(s, fg, r.inflate(-6, -6), 3)
        size = 44
        while True:
            f = pygame.font.SysFont("arialblack,impact,arial", size, bold=True)
            t = f.render(txt, True, fg)
            if t.get_width() < 236 or size < 12:
                break
            size -= 2
        s.blit(t, (128 - t.get_width() // 2, k * 64 + 32 - t.get_height() // 2))
    draw_guadalupe(s, pygame.Rect(256, 0, 256, 384))
    draw_flag(s, pygame.Rect(256, 400, 256, 146))
    return pygame.image.tobytes(s, "RGBA", False)


def build_extras(m, track, rnd, sc, r0, rmax, st0):
    pts, nrm, tan, n = track.pts, track.nrm, track.tan, track.n
    keep = []
    info = {}

    # ---------------- escenario del concierto dentro del estadio
    xs, zs = np.meshgrid(np.arange(-r0, r0, 3.0), np.arange(-r0, r0, 3.0))
    P = np.column_stack([xs.ravel() + sc[0], zs.ravel() + sc[1]])
    P = P[np.hypot(P[:, 0] - sc[0], P[:, 1] - sc[1]) < r0 - 5]
    best = P[int(np.argmax(track.nearest_dist(P)))]
    dvec = (best - sc) / np.linalg.norm(best - sc)
    S = sc + dvec * (r0 - 14)
    ff = -dvec
    rr = np.array([-ff[1], ff[0]])
    info["stage"] = (S, ff, rr)
    keep.append((S[0], S[1], 30))
    m_main = m
    m = Mesh()          # todo lo del concierto va en su propia malla: solo se dibuja de noche

    def L(a, b, y=0.0):
        return (S[0] + rr[0] * a + ff[0] * b, y, S[1] + rr[1] * a + ff[1] * b)

    def lbox(a0, a1, b0, b1, y0, y1, col, mat=0):
        m.box(L(a0, b0, y0), (rr[0] * (a1 - a0), 0, rr[1] * (a1 - a0)), (0, y1 - y0, 0),
              (ff[0] * (b1 - b0), 0, ff[1] * (b1 - b0)), col, mat=mat)

    def lquad(a0, a1, b, y0, y1, col, mats=None):
        m.quad(L(a0, b, y0), L(a1, b, y0), L(a1, b, y1), L(a0, b, y1), col, mats)

    blk, gray = rgb(25, 25, 30), rgb(90, 90, 100)
    scr = ((13, 0, 0), (13, 1, 0), (13, 1, 1), (13, 0, 1))
    scr_fire = ((13, 0, 2), (13, 1, 2), (13, 1, 3), (13, 0, 3))
    # escenario principal
    lbox(-15, 15, -6, 6, 0, 2.2, blk)
    lbox(-3, 3, -5, -1, 2.2, 3.2, rgb(40, 40, 48))
    lbox(-13, 13, -6.4, -5.6, 2.2, 15.4, blk)
    lquad(-12, 12, -5.5, 3.0, 15.0, (1, 1, 1), scr)
    # techo grande del escenario (como en las fotos)
    lbox(-19, 19, -8, 7, 20.6, 21.4, rgb(215, 215, 222))
    lbox(-16, 16, -7, 5, 21.4, 23.2, rgb(200, 200, 210))
    for a in (-18.5, 18.5):
        for bb in (-7.5, 6.5):
            lbox(a - 0.35, a + 0.35, bb - 0.35, bb + 0.35, 0, 20.6, gray)
    # barra de luces rojas sobre la pantalla
    for yy in (15.5, 16.4):
        for i in range(12):
            a = -11 + i * 2
            lbox(a - 0.6, a + 0.6, -5.4, -5.0, yy, yy + 0.55, rgb(255, 30, 30), mat=11)
    bx, _, bz = L(0, -5.3)
    decal(m, bx, bz, -ff[0], -ff[1], 24, 17.6, 20.3, 10)
    # bocinas a los lados del escenario
    for sgn in (-1, 1):
        lbox(sgn * 16 - 2.5, sgn * 16 + 2.5, -1, 3, 0, 9, blk)
        for yy in (1.5, 4.5, 7.5):
            lquad(sgn * 16 - 2.2, sgn * 16 + 2.2, 3.02, yy - 1.2, yy + 1.2, gray)
        # dos pantallas verticales por lado (con llamas)
        for a in (21.5, 27.5):
            lbox(sgn * a - 2.8, sgn * a + 2.8, 3.4, 4.0, 0, 16, blk)
            lquad(sgn * a - 2.5, sgn * a + 2.5, 4.05, 5.0, 15.5, (1, 1, 1), scr_fire)
        # torres de fuego
        lbox(sgn * 33 - 0.6, sgn * 33 + 0.6, 1.5, 2.7, 0, 15, gray)
        lbox(sgn * 33 - 1.0, sgn * 33 + 1.0, 1.1, 3.1, 15, 15.6, blk)
    lbox(-17, 17, 8.0, 8.3, 0, 1.2, gray)
    movers = []
    for i in range(10):
        a = -15 + i * 30 / 9
        lbox(a - 0.4, a + 0.4, -0.4, 0.4, 19.8, 20.6, rgb(255, 255, 240), mat=11)
        movers.append(L(a, 0, 20.0))
    info["movers"] = movers
    info["flames"] = [L(a, 6.3, 2.3) for a in (-13, -7.5, 7.5, 13)]
    info["fireballs"] = [L(-33, 2.1, 16.2), L(33, 2.1, 16.2), L(-12, 5.5, 23.6), L(-5, 5.5, 23.6),
                         L(5, 5.5, 23.6), L(12, 5.5, 23.6), L(-24.5, 3.7, 16.4), L(24.5, 3.7, 16.4)]
    info["fountain"] = [L(a, -7.5, 23.5) for a in (-14, -9, -4, 0, 4, 9, 14)]
    for a in (-13, -7.5, 7.5, 13):
        lbox(a - 0.5, a + 0.5, 5.8, 6.8, 2.2, 2.6, rgb(60, 60, 60))
    # torres de sonido (delay) entre el publico y carpa de audio al centro
    for sgn in (-1, 1):
        lbox(sgn * 24 - 0.3, sgn * 24 + 0.3, 32, 32.6, 0, 13, blk)
        lbox(sgn * 24 - 1.2, sgn * 24 + 1.2, 31.4, 33.2, 8.5, 13, blk)
        lbox(sgn * 24 - 0.5, sgn * 24 + 0.5, 31.9, 32.7, 13, 13.6, rgb(255, 255, 240), mat=11)
    lbox(-4, 4, 36, 42, 0, 3.0, rgb(225, 225, 230))
    lbox(-4.6, 4.6, 35.4, 42.6, 3.0, 3.4, rgb(240, 240, 245))
    lbox(-3, 3, 35.9, 36.0, 1.0, 2.2, rgb(90, 160, 255), mat=11)
    # la banda
    band = [(0, 3.6, rgb(20, 20, 20)), (-6, 2.6, rgb(30, 30, 30)), (6, 2.6, rgb(120, 10, 10)),
            (10, 1.2, rgb(20, 20, 25)), (0, -3.0, rgb(25, 25, 25))]
    for i, (a, b, col) in enumerate(band):
        x, _, z = L(a, b)
        y0 = 3.2 if i == 4 else 2.2
        person(m, x, z, rnd, y0=y0, s=1.15, arms_up=(i == 0), amp=0.18, shirt=col)
        if i in (1, 2, 3):
            lbox(a - 0.6, a + 0.6, b + 0.25, b + 0.4, y0 + 1.0, y0 + 1.3, [rgb(180, 20, 20), rgb(20, 20, 20),
                                                                          rgb(200, 200, 210)][i - 1])
    x, _, z = L(0, 4.6)
    prism_y(m, x, z, 2.2, 4.0, 0.04, gray, k=4)
    for a, b, r in ((-1.2, -2.0, 0.5), (1.2, -2.0, 0.5), (-0.6, -1.4, 0.35), (0.6, -1.4, 0.35)):
        x, _, z = L(a, b)
        prism_y(m, x, z, 3.2, 3.9, r, rgb(200, 30, 30), k=8)
    # el publico del concierto (playeras negras de metalero, celulares y banderas)
    metal = [rgb(15, 15, 18), rgb(22, 22, 26), rgb(30, 30, 35), rgb(15, 15, 18), rgb(110, 10, 10),
             rgb(40, 40, 70)]

    def in_crowd(a, b):
        return 10 <= b <= 60 and abs(a) < 16 + b * 0.5

    # circle pit: una rueda de metaleros corriendo en circulo (el shader la hace girar)
    pit_a, pit_b = -14.0, 26.0
    for ca, cb in ((-14, 26), (14, 26), (0, 30), (-10, 40), (10, 40)):
        x, _, z = L(ca, cb)
        if track.nearest_dist(np.array([[x, z]]))[0] > LIMIT + 14 and math.hypot(x - sc[0], z - sc[1]) < r0 - 12:
            pit_a, pit_b = ca, cb
            break
    px_, _, pz_ = L(pit_a, pit_b)
    info["pit"] = (float(px_), float(pz_))

    for a in np.arange(-46, 46, 1.55):
        for b in np.arange(10, 60, 1.55):
            if not in_crowd(a, b) or (abs(a) < 5.2 and 34.8 < b < 43.2):
                continue
            if abs(abs(a) - 24) < 1.6 and abs(b - 32.3) < 1.6:
                continue
            if math.hypot(a - pit_a, b - pit_b) < 10:
                continue
            x, _, z = L(a + rnd.uniform(-0.5, 0.5), b + rnd.uniform(-0.5, 0.5))
            if math.hypot(x - sc[0], z - sc[1]) > r0 - 3:
                continue
            if track.nearest_dist(np.array([[x, z]]))[0] < LIMIT + 4:
                continue
            person(m, x, z, rnd, arms_up=rnd.random() < 0.7, amp=rnd.uniform(0.25, 0.6),
                   flag=rnd.random() < 0.04, light=rnd.random() < 0.35,
                   shirt=rnd.choice(metal) if rnd.random() < 0.75 else None)
    for rad in (4.0, 5.5, 7.0, 8.5):
        cnt = int(TAU * rad / 1.35)
        for j in range(cnt):
            an = j / cnt * TAU + rnd.uniform(-0.1, 0.1)
            x, _, z = L(pit_a + rad * math.cos(an), pit_b + rad * math.sin(an))
            person(m, x, z, rnd, arms_up=rnd.random() < 0.4, amp=rnd.uniform(0.2, 0.45), matid=19,
                   shirt=rnd.choice(metal))
    # el resto de la cancha: "alfombra" de cabezas y celulares (barato de dibujar, se ve lleno de lejos)
    cell = 2.5
    for gx in np.arange(sc[0] - r0, sc[0] + r0, cell):
        for gz in np.arange(sc[1] - r0, sc[1] + r0, cell):
            cx, cz = gx + cell / 2, gz + cell / 2
            if math.hypot(cx - sc[0], cz - sc[1]) > r0 - 2:
                continue
            a = (cx - S[0]) * rr[0] + (cz - S[1]) * rr[1]
            b = (cx - S[0]) * ff[0] + (cz - S[1]) * ff[1]
            if b < 9 or in_crowd(a, b):
                continue
            if track.nearest_dist(np.array([[cx, cz]]))[0] < LIMIT + 3.5:
                continue
            y = 1.6
            m.quad((gx, y, gz), (gx + cell, y, gz), (gx + cell, y, gz + cell), (gx, y, gz + cell), rgb(30, 28, 32),
                   ((8, gx, gz), (8, gx + cell, gz), (8, gx + cell, gz + cell), (8, gx, gz + cell)))
    # puntos de lanzamiento de fuegos artificiales
    launch = [L(a, -5, 19.5) for a in (-12, 0, 12)]
    H = 14 * 1.8 + 9
    for j in range(10):
        a = j / 10 * TAU
        launch.append((sc[0] + (rmax - 3) * math.cos(a), H + 1, sc[1] + (rmax - 3) * math.sin(a)))
    info["launch"] = launch
    info["stadium"] = (sc, rmax)
    info["concert_mesh"] = m.array()
    m = m_main

    # ---------------- letrero del estadio (mirando a la recta principal)
    dS = (st0 - sc) / np.linalg.norm(st0 - sc)
    base_a = math.atan2(dS[1], dS[0])
    for off in (0, 0.2, -0.2, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8):
        a = base_a + off
        c = sc + np.array([math.cos(a), math.sin(a)]) * (rmax - 4)
        fx, fz = -math.cos(a), -math.sin(a)
        legs = np.array([[c[0] + fz * 23, c[1] - fx * 23], [c[0] - fz * 23, c[1] + fx * 23], c])
        if track.nearest_dist(legs).min() > LIMIT + 3:
            billboard(m, c[0], c[1], fx, fz, 46, 34.5, 43.5, 12)
            break

    # ---------------- Angel de la Independencia y bandera monumental
    def spot(cands, clear):
        cands = np.array(cands, float)
        d = track.nearest_dist(cands)
        for c, dd in zip(cands, d):
            if dd >= clear and math.hypot(c[0] - sc[0], c[1] - sc[1]) > rmax + 30:
                return c
        return cands[int(np.argmax(d))]
    ang = spot([(st0[0] + 400, st0[1] + 95), (st0[0] + 520, st0[1] + 100), (st0[0] + 300, st0[1] + 110),
                (st0[0] + 450, st0[1] - 140)], 45)
    angel_monument(m, ang[0], ang[1])
    info["angel"] = ang
    keep.append((ang[0], ang[1], 30))
    fl = spot([(st0[0] + 25, st0[1] + 70), (st0[0] + 60, st0[1] + 85), (st0[0] + 640, st0[1] + 80)], 34)
    flagpole(m, fl[0], fl[1])
    keep.append((fl[0], fl[1], 30))

    # ---------------- gradas extra en las rectas
    placed = []
    for i in range(0, n, 6):
        if len(placed) >= 3:
            break
        if np.max(np.abs(track.curv[[(i + k) % n for k in range(-12, 13)]])) > 1 / 900:
            continue
        if any(min(abs(i - j), n - abs(i - j)) < 60 for j in placed):
            continue
        for side in (1, -1):
            o = pts[i] + nrm[i] * side * (LIMIT + 4)
            out = nrm[i] * side
            along = tan[i]
            q = np.array([o + along * t_ + out * d_ for t_ in (0, 30, 60) for d_ in (0, 12, 24)])
            if track.nearest_dist(q).min() < LIMIT + 3 or min(math.hypot(*(c - sc)) for c in q) < rmax + 10:
                continue
            if abs(o[1] - st0[1]) < 70 and st0[0] - 30 < o[0] < st0[0] + 1030:
                continue
            stand(m, o, along, out, 60, 8, 2.6, 1.6, rnd.choice([rgb(30, 80, 190), rgb(220, 120, 30),
                                                                 rgb(200, 40, 40), rgb(40, 150, 70)]))
            c = o + along * 30 + out * 10
            keep.append((c[0], c[1], 45))
            placed.append(i)
            break

    # ---------------- bardas con la Virgen de Guadalupe
    bb_idx = set()
    for frac in (0.12, 0.3, 0.47, 0.68, 0.86):
        i = int(frac * n)
        for side in (1, -1):
            c = pts[i] + nrm[i] * side * (LIMIT + 6)
            if track.nearest_dist(c[None])[0] < LIMIT + 5 or math.hypot(*(c - sc)) < rmax + 8:
                continue
            fx, fz = nrm[i] * side
            rx, rz = -fz, fx
            wall = rgb(170, 80, 50)
            m.box((c[0] - rx * 6 + fx * 0.1, 0, c[1] - rz * 6 + fz * 0.1), (rx * 12, 0, rz * 12), (0, 13, 0),
                  (fx * 0.6, 0, fz * 0.6), wall)
            image_quad(m, c[0] - fx * 0.02, c[1] - fz * 0.02, fx, fz, 8.4, 0.6, 13.2, GUADA_UV)
            info.setdefault("bardas", []).append((c, (fx, fz)))
            for k in range(-5, 6):
                px, pz = c[0] + rx * k * 0.9 - fx * 0.8, c[1] + rz * k * 0.9 - fz * 0.8
                if k % 2 == 0:
                    m.aabb(px - 0.12, 0, pz - 0.12, px + 0.12, 0.45, pz + 0.12, rgb(250, 140, 0), mat=(14, k, 3.0))
                else:
                    m.aabb(px - 0.25, 0, pz - 0.25, px + 0.25, 0.5, pz + 0.25,
                           [rgb(250, 150, 0), rgb(220, 20, 40), rgb(250, 250, 250)][k % 3])
            keep.append((c[0], c[1], 12))
            bb_idx.update(range(i - 8, i + 9))
            break

    # ---------------- papel picado sobre la pista
    for frac in (0.02, 0.26, 0.55, 0.78):
        i = int(frac * n)
        a = pts[i] + nrm[i] * -(HW + 2.5)
        b = pts[i] + nrm[i] * (HW + 2.5)
        papel_picado(m, a, b)

    # ---------------- torres de iluminacion
    for i in range(0, n, 30):
        side = 1 if (i // 30) % 2 == 0 else -1
        c = pts[i] + nrm[i] * side * (LIMIT + 4)
        if track.nearest_dist(c[None])[0] < LIMIT + 3:
            continue
        prism_y(m, c[0], c[1], 0, 20, 0.3, rgb(150, 150, 155), k=6)
        fx, fz = -nrm[i] * side
        rx, rz = -fz, fx
        m.box((c[0] - rx * 1.8, 20, c[1] - rz * 1.8), (rx * 3.6, 0, rz * 3.6), (0, 1.6, 0), (fx * 0.6, 0, fz * 0.6),
              rgb(255, 250, 225), mat=11)

    # ---------------- aficionados echando porras junto a las barreras
    for i in range(0, n, 8):
        if i in bb_idx:
            continue
        for side in (1, -1):
            if rnd.random() > 0.7:
                continue
            base = pts[i] + nrm[i] * side * (LIMIT + 1.8)
            if track.nearest_dist(base[None])[0] < LIMIT + 1.2:
                continue
            if math.hypot(base[0] - S[0], base[1] - S[1]) < 60:
                continue
            for _ in range(rnd.randint(3, 7)):
                t_ = rnd.uniform(-7, 7)
                d_ = rnd.uniform(-0.5, 1.2)
                x = base[0] + tan[i, 0] * t_ + nrm[i, 0] * side * d_
                z = base[1] + tan[i, 1] * t_ + nrm[i, 1] * side * d_
                person(m, x, z, rnd, arms_up=rnd.random() < 0.65, flag=rnd.random() < 0.15)
    return keep, info


def concert_beams(t, info, strength):
    S, ff, rr = info["stage"]
    cols = [(1.0, 0.2, 0.2), (0.3, 0.5, 1.0), (1.0, 1.0, 1.0), (0.9, 0.2, 1.0), (0.2, 1.0, 0.4), (1.0, 0.7, 0.1)]
    beams = []
    for i, (x, y, z) in enumerate(info["movers"]):
        if i % 2:   # estos barren las gradas
            yaw = math.sin(t * 0.5 + i) * 1.45
            pitch = 0.2 + 0.15 * math.sin(t * 1.1 + i)
        else:
            yaw = math.sin(t * 0.9 + i * 0.8) * 0.9
            pitch = 0.75 + 0.5 * math.sin(t * 0.6 + i * 1.3)
        fx = ff[0] * math.cos(yaw) + rr[0] * math.sin(yaw)
        fz = ff[1] * math.cos(yaw) + rr[1] * math.sin(yaw)
        d = (fx * math.cos(pitch), math.sin(pitch), fz * math.cos(pitch))
        c = cols[(i + int(t / 2)) % len(cols)]
        beams.append(((x, y, z), (x + d[0] * 170, y + d[1] * 170, z + d[2] * 170), 0.3, 8.0,
                      tuple(v * strength for v in c)))
    for i in range(6):
        a = -0.8 + 1.6 * i / 5 + 0.35 * math.sin(t * 1.5)
        dx = ff[0] * math.cos(a) + rr[0] * math.sin(a)
        dz = ff[1] * math.cos(a) + rr[1] * math.sin(a)
        el = 0.12 + 0.1 * math.sin(t * 3 + i)
        o = (S[0] + ff[0] * 5.5, 3.0, S[1] + ff[1] * 5.5)
        lc = (0.1, 1.0, 0.2) if i % 2 else (0.15, 0.4, 1.0)
        beams.append((o, (o[0] + dx * 130, o[1] + el * 130, o[2] + dz * 130), 0.05, 0.5,
                      tuple(v * strength for v in lc)))
    return beams


class FX:
    """Fuegos artificiales, abanicos dorados, bolas de fuego, humo y haces de luz (geometria dinamica aditiva).
    Tipos: 0 chispa, 1 cohete, 2 fuego, 3 estela, 4 destello, 5 humo, 6 cometa dorado,
    7 chispa de choque, 8 fuego azul del nitro."""
    MAXV = 100000
    GRAV = np.array([5.5, 9.8, -2.0, 1.5, 0.0, -0.15, 7.0, 9.8, -1.0], np.float32)
    DRAG = np.array([0.9, 0.0, 2.5, 3.0, 0.0, 0.25, 0.15, 1.2, 4.0], np.float32)
    PALETTE = np.array([[1.0, 0.25, 0.2], [0.3, 0.6, 1.0], [1.0, 0.85, 0.2], [0.8, 0.3, 1.0], [0.3, 1.0, 0.5],
                        [1.0, 0.5, 0.1], [1.0, 1.0, 1.0], [1.0, 0.3, 0.7]], np.float32)
    TRICOLOR = np.array([[0.0, 0.9, 0.4], [1.0, 1.0, 1.0], [1.0, 0.1, 0.15]], np.float32)

    def __init__(self, ctx, prog, info):
        self.buf = ctx.buffer(reserve=self.MAXV * 48, dynamic=True)
        self.vao = ctx.vertex_array(prog, [(self.buf, "3f 3f 3f 3f", "in_pos", "in_norm", "in_col", "in_mat")])
        self.info = info
        self.rng = np.random.default_rng(3)
        self.p = np.zeros((0, 3), np.float32)
        self.v = np.zeros((0, 3), np.float32)
        self.c = np.zeros((0, 3), np.float32)
        self.life = np.zeros(0, np.float32)
        self.maxl = np.zeros(0, np.float32)
        self.size = np.zeros(0, np.float32)
        self.kind = np.zeros(0, np.int8)
        self.next_launch = 0.0
        self.next_flame = 0.0
        self.next_fireball = 2.0
        self.next_fountain = 3.0
        self.count = 0
        self.bursts = 0
        self.concert = True

    def spawn(self, p, v, c, life, size, kind):
        p = np.asarray(p, np.float32).reshape(-1, 3)
        k = len(p)
        self.p = np.concatenate([self.p, p])
        self.v = np.concatenate([self.v, np.broadcast_to(np.asarray(v, np.float32), (k, 3))])
        self.c = np.concatenate([self.c, np.broadcast_to(np.asarray(c, np.float32), (k, 3))])
        life = np.broadcast_to(np.asarray(life, np.float32), (k,))
        self.life = np.concatenate([self.life, life])
        self.maxl = np.concatenate([self.maxl, life])
        self.size = np.concatenate([self.size, np.broadcast_to(np.asarray(size, np.float32), (k,))])
        self.kind = np.concatenate([self.kind, np.full(k, kind, np.int8)])

    def _concert(self, t):
        """Fuegos artificiales, fuego, cometas y humo del concierto (solo de noche)."""
        rng = self.rng
        info = self.info
        S, ff, rr = info["stage"]
        # cohetes que explotan en el cielo
        lp = info["launch"]
        if self.next_launch < t - 2:
            self.next_launch = t
        while self.next_launch < t:
            o = np.array(lp[rng.integers(len(lp))], np.float32)
            v = [rng.normal(0, 5), rng.uniform(28, 40), rng.normal(0, 5)]
            self.spawn(o, v, [1.0, 0.8, 0.5], 3.5, 1.4, 1)
            self.next_launch += rng.uniform(0.05, 0.3)
        # chorros de fuego al frente del escenario
        if t > self.next_flame:
            for f in info["flames"]:
                k = 22
                v = np.column_stack([rng.normal(0, 1.2, k), rng.uniform(9, 18, k), rng.normal(0, 1.2, k)])
                c = np.column_stack([np.ones(k), rng.uniform(0.35, 0.75, k), rng.uniform(0.0, 0.15, k)])
                self.spawn(np.tile(f, (k, 1)), v, c, rng.uniform(0.35, 0.7, k), rng.uniform(1.0, 1.8, k), 2)
            self.next_flame = t + rng.choice([0.9, 1.4, 2.2, 3.0])
        # bolas de fuego en torres y techo (todas a la vez, como en las fotos)
        if t > self.next_fireball:
            for f in info["fireballs"]:
                k = 24
                v = np.column_stack([rng.normal(0, 2.5, k), rng.uniform(4, 12, k), rng.normal(0, 2.5, k)])
                c = np.column_stack([np.full(k, 0.75), rng.uniform(0.25, 0.45, k), rng.uniform(0.02, 0.08, k)])
                self.spawn(np.tile(f, (k, 1)), v, c * 0.7, rng.uniform(0.5, 1.0, k), rng.uniform(2.5, 4.5, k), 2)
            self.next_fireball = t + rng.choice([2.5, 4.0, 5.5, 7.0])
        # abanico de cometas dorados detras del escenario
        if t > self.next_fountain:
            for o in info["fountain"]:
                for _ in range(2):
                    ang = rng.uniform(-0.55, 0.55)
                    hx, hz = rr * math.sin(ang) * 16
                    v = [hx + rng.normal(0, 1), rng.uniform(34, 46), hz + rng.normal(0, 1)]
                    self.spawn(o, v, [1.0, 0.82, 0.4], rng.uniform(1.6, 2.3), 1.4, 6)
            self.next_fountain = t + rng.uniform(5, 9)
        # humo morado sobre el escenario y el publico
        nsmoke = int((self.kind == 5).sum())
        if nsmoke < 45:
            k = min(3, 45 - nsmoke)
            a = rng.uniform(-38, 38, k)
            b = rng.uniform(-2, 45, k)
            pos = np.column_stack([S[0] + rr[0] * a + ff[0] * b, rng.uniform(2, 9, k), S[1] + rr[1] * a + ff[1] * b])
            v = np.column_stack([rng.normal(0, 0.6, k), rng.uniform(0, 0.4, k), rng.normal(0, 0.6, k)])
            col = np.array([0.08, 0.025, 0.1], np.float32) * rng.uniform(0.6, 1.2, (k, 1))
            self.spawn(pos, v, col, rng.uniform(9, 16, k), rng.uniform(12, 22, k), 5)

    def sparks(self, x, z, nx, nz, strength):
        rng = self.rng
        k = int(clamp(strength * 1.6, 2, 34))
        d = rng.normal(0, 1, (k, 3))
        d[:, 1] = np.abs(d[:, 1]) * 1.3 + 0.4
        d[:, 0] -= nx * 1.2
        d[:, 2] -= nz * 1.2
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        sp = rng.uniform(4, 9 + min(strength, 20) * 0.6, (k, 1))
        col = np.column_stack([np.ones(k), rng.uniform(0.6, 0.9, k), rng.uniform(0.15, 0.4, k)]) * 2.0
        self.spawn(np.tile([x, 0.6, z], (k, 1)), d * sp, col, rng.uniform(0.3, 0.7, k), rng.uniform(0.3, 0.55, k), 7)
        self.spawn([x, 0.7, z], (0, 0, 0), (0.9, 0.6, 0.25), 0.08, 1.6 + min(strength, 15) * 0.12, 4)

    def nitro_flame(self, car):
        rng = self.rng
        fx, fz = math.cos(car.ang), math.sin(car.ang)
        rx, rz = -fz, fx
        back = car.size * 2.25 + 0.15
        y = 0.45 + 0.35 * max(0.0, car.size - 1.0) / 0.33
        for side in (-0.38, 0.38):
            ox, oz = car.x - fx * back + rx * side, car.z - fz * back + rz * side
            k = 5
            v = np.column_stack([car.vx * 0.6 - fx * rng.uniform(6, 13, k) + rng.normal(0, 0.6, k),
                                 rng.normal(0.3, 0.4, k),
                                 car.vz * 0.6 - fz * rng.uniform(6, 13, k) + rng.normal(0, 0.6, k)])
            white = rng.random(k) < 0.3
            col = np.where(white[:, None], [[0.75, 0.85, 1.0]], [[0.2, 0.45, 1.0]]) * 1.5
            self.spawn(np.tile([ox, y, oz], (k, 1)), v, col, rng.uniform(0.15, 0.32, k), rng.uniform(0.4, 0.8, k), 8)
        self.spawn([car.x - fx * back, y + 0.2, car.z - fz * back], (0, 0, 0), (0.1, 0.25, 0.7), 0.06, 2.4, 4)

    def update(self, dt, t):
        if self.concert:
            self._concert(t)
        elif len(self.kind):   # sin concierto: solo quedan chispas, nitro y destellos
            keep = (self.kind == 7) | (self.kind == 8) | (self.kind == 4)
            if not keep.all():
                for name in ("p", "v", "c", "life", "maxl", "size", "kind"):
                    setattr(self, name, getattr(self, name)[keep])
        rng = self.rng
        if not len(self.p):
            return
        k = self.kind
        self.v[:, 1] -= self.GRAV[k] * dt
        self.v *= (1 - self.DRAG[k] * dt)[:, None]
        self.p += self.v * dt
        self.life -= dt
        # estelas de cohetes y cometas
        for kind, col, life, size in ((1, (1.0, 0.6, 0.25), 0.5, 0.9), (6, (1.0, 0.78, 0.35), 0.8, 1.0)):
            idx = np.nonzero(k == kind)[0]
            if len(idx):
                self.spawn(self.p[idx], (0, 0, 0), col, life, size, 3)
        k = self.kind
        burst = np.nonzero((k == 1) & ((self.v[:, 1] < 5) | (self.life < 0.15)))[0]
        for i in burst:
            self.life[i] = 0
            nb = 170
            d = rng.normal(0, 1, (nb, 3))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            sp = rng.uniform(22, 34) * rng.uniform(0.85, 1.0, nb)[:, None]
            if rng.random() < 0.35:
                sp[nb // 2:] *= 0.5
            if rng.random() < 0.3:
                col = self.TRICOLOR[rng.integers(3, size=nb)]
            else:
                a, b = self.PALETTE[rng.integers(len(self.PALETTE), size=2)]
                col = np.where((rng.random(nb) < 0.7)[:, None], a, b)
            self.spawn(np.tile(self.p[i], (nb, 1)), d * sp, col * 1.4, rng.uniform(1.6, 2.8, nb), 2.6, 0)
            self.spawn(self.p[i], (0, 0, 0), col[0] * 0.6 + 0.6, 0.3, 16.0, 4)
            self.bursts += 1
        alive = self.life > 0
        if len(alive) > 12000:
            alive[: len(alive) - 12000] = False
        for name in ("p", "v", "c", "life", "maxl", "size", "kind"):
            setattr(self, name, getattr(self, name)[alive])

    def build(self, view, eye, beams):
        right = view[0, :3]
        up = view[1, :3]
        n = len(self.p)
        parts = []
        if n:
            f = np.clip(self.life / self.maxl, 0, 1)
            k = self.kind
            bright = np.select([k == 0, k == 3, k == 2, k == 4, k == 5, k == 7, k == 8],
                               [f ** 0.6 * self.rng.uniform(0.6, 1.0, n), f * 0.6, f, f * f,
                                np.sin(np.pi * f) * 0.9, f, f ** 0.8], 1.0).astype(np.float32)
            size = self.size * np.select([k == 0, k == 2, k == 5, k == 7, k == 8],
                                         [0.5 + 0.5 * f, 1 + (1 - f) * 1.5, 1 + (1 - f) * 0.5, 0.6 + 0.4 * f,
                                          0.4 + 0.9 * f], 1.0)
            col = self.c * bright[:, None]
            verts = np.zeros((n, 6, 12), np.float32)
            for j, (a, b) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1))):
                verts[:, j, 0:3] = self.p + (right * a + up * b)[None, :] * size[:, None]
                verts[:, j, 6:9] = col
                verts[:, j, 9] = 15
                verts[:, j, 10] = a
                verts[:, j, 11] = b
            parts.append(verts.reshape(-1, 12))
        if beams:
            bv = []
            eye = np.asarray(eye, np.float32)
            for A, B, w0, w1, col in beams:
                A, B = np.asarray(A, np.float32), np.asarray(B, np.float32)
                sd = np.cross(B - A, eye - A)
                ln = np.linalg.norm(sd)
                if ln < 1e-6:
                    continue
                sd /= ln
                a0, a1, b0, b1 = A - sd * w0, A + sd * w0, B - sd * w1, B + sd * w1
                for p, u, v in ((a0, -1, 0), (a1, 1, 0), (b1, 1, 1), (a0, -1, 0), (b1, 1, 1), (b0, -1, 1)):
                    bv.append((p[0], p[1], p[2], 0, 0, 0, col[0], col[1], col[2], 16, u, v))
            if bv:
                parts.append(np.array(bv, np.float32))
        if not parts:
            self.count = 0
            return
        data = np.concatenate(parts)[: self.MAXV]
        self.buf.write(data.tobytes())
        self.count = len(data)

    def render(self):
        if self.count:
            self.vao.render(vertices=self.count)


# ---------------------------------------------------------------- shaders
SCENE_VS = """
#version 330
uniform mat4 u_vp;
uniform mat4 u_model;
uniform float u_time;
uniform vec2 u_pit;
in vec3 in_pos; in vec3 in_norm; in vec3 in_col; in vec3 in_mat;
out vec3 v_world; out vec3 v_norm; out vec3 v_col; out vec3 v_mat;
void main() {
    vec3 p = in_pos;
    int m = int(in_mat.x + 0.5);
    if (m == 10) {                      // gente brincando
        p.y += abs(sin(u_time * 6.5 + in_mat.y)) * abs(in_mat.z);
    } else if (m == 12) {               // bandera ondeando
        float f = clamp((in_mat.y - 0.5) * 2.0, 0.0, 1.0);
        p += in_norm * sin(u_time * 2.6 - f * 7.0) * f * 1.4;
    } else if (m == 17) {               // papel picado
        p += in_norm * sin(u_time * 2.0 + in_mat.y) * 0.25;
    } else if (m == 19) {               // circle pit: giran alrededor del centro
        float an = u_time * 0.55;
        vec2 d = p.xz - u_pit;
        p.xz = u_pit + vec2(d.x * cos(an) - d.y * sin(an), d.x * sin(an) + d.y * cos(an));
        p.y += abs(sin(u_time * 7.0 + in_mat.y)) * abs(in_mat.z);
    }
    vec4 w = u_model * vec4(p, 1.0);
    v_world = w.xyz;
    v_norm = mat3(u_model) * in_norm;
    v_col = in_col;
    v_mat = in_mat;
    gl_Position = u_vp * w;
}
"""

SCENE_FS = """
#version 330
uniform vec3 u_cam; uniform vec3 u_fog; uniform vec2 u_fogr; uniform vec3 u_sun;
uniform sampler2D u_tex; uniform float u_hw; uniform float u_time;
uniform float u_amb; uniform float u_diff; uniform vec3 u_tint; uniform float u_night; uniform float u_head;
uniform vec2 u_stage; uniform float u_concert;
in vec3 v_world; in vec3 v_norm; in vec3 v_col; in vec3 v_mat;
out vec4 f_color;
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
void main() {
    int m = int(v_mat.x + 0.5);
    float u = v_mat.y, v = v_mat.z;
    vec3 c = v_col;
    float d = length(v_world - u_cam);
    float fg = clamp((d - u_fogr.x) / (u_fogr.y - u_fogr.x), 0.0, 1.0);
    if (m == 9) {
        float r = length(vec2(u, v));
        f_color = vec4(0.0, 0.0, 0.0, 0.5 * (1.0 - smoothstep(0.5, 1.0, r)));
        return;
    }
    if (m == 15) {                      // particula (fuegos artificiales / fuego)
        float r = length(vec2(u, v));
        float a = 1.0 - smoothstep(0.0, 1.0, r);
        f_color = vec4(c * a * a * (1.0 - 0.6 * fg), 1.0);
        return;
    }
    if (m == 16) {                      // haz de luz
        float a = (1.0 - smoothstep(0.2, 1.0, abs(u))) * (1.0 - v) * 0.45;
        f_color = vec4(c * a * (1.0 - 0.6 * fg), 1.0);
        return;
    }
    bool emis = false;
    vec3 extra = vec3(0.0);
    if (m == 1) {
        c *= 0.9 + 0.18 * hash(floor(v_world.xz * 2.0));
        float av = abs(v);
        if (av > u_hw - 0.75 && av < u_hw - 0.4) c = vec3(0.92);
    } else if (m == 2) {
        float st = mod(floor(v_world.x / 14.0) + floor(v_world.z / 14.0), 2.0);
        c *= (0.9 + 0.12 * st) * (0.9 + 0.2 * hash(floor(v_world.xz * 0.8)));
    } else if (m == 3) {
        float ch = mod(floor(u * 1.25) + floor(v * 1.25), 2.0);
        c = mix(vec3(0.07), vec3(0.95), ch);
    } else if (m == 4) {
        c = mod(floor(u / 3.0), 2.0) < 1.0 ? vec3(0.85, 0.08, 0.08) : vec3(0.95);
    } else if (m == 5) {
        c *= 0.85 + 0.3 * hash(floor(v_world.xz * 3.0));
    } else if (m == 6) {
        c = mod(floor(u / 4.0), 2.0) < 1.0 ? vec3(0.1, 0.25, 0.75) : vec3(0.95);
        if (v > 0.85) c = vec3(0.8);
    } else if (m == 7 || m == 12) {
        c = texture(u_tex, vec2(u, v)).rgb;
    } else if (m == 8) {                // publico: siluetas (cuerpo + cabeza), celulares y luz del escenario
        vec2 q = vec2(u * 2.2, v * 3.0 + v_world.y * 2.5);
        vec2 cell = floor(q);
        vec2 fc = fract(q);
        float h = hash(cell);
        float h2 = hash(cell + 7.0);
        vec3 body = h < 0.55 ? vec3(0.04 + 0.08 * h2) : h < 0.63 ? vec3(0.45, 0.06, 0.06) : h < 0.69 ? vec3(0.7)
                  : h < 0.77 ? vec3(0.1, 0.15, 0.36) : h < 0.83 ? vec3(0.18, 0.35, 0.2) : v_col * 0.7;
        vec3 head = h2 < 0.75 ? vec3(0.62, 0.45, 0.33) : vec3(0.08, 0.06, 0.05);
        bool is_head = fc.y > 0.58 && abs(fc.x - 0.5) < 0.24;
        bool gap = fc.y > 0.58 && !is_head;
        c = is_head ? head : (gap ? vec3(0.03) : body);
        if (u_night > 0.5) {
            vec2 ph = fc - vec2(0.78, 0.8);
            if (hash(cell + 13.0) > 0.8 && dot(ph, ph) < 0.02) {   // celular encendido
                c = vec3(0.95, 0.95, 1.0) * (0.6 + 0.4 * sin(u_time * 5.0 + h * 60.0));
                emis = true;
            }
            float sd = length(v_world.xz - u_stage);
            float wash = clamp(1.0 - sd / 260.0, 0.0, 1.0);
            vec3 wc = mix(vec3(0.7, 0.1, 0.62), 0.5 + 0.5 * cos(u_time * 0.8 + vec3(0.0, 2.5, 4.0)), 0.3);
            extra = wc * wash * 0.2 * (0.6 + 0.4 * sin(u_time * 3.0 + v_world.x * 0.05)) * u_concert;
            if (hash(cell + floor(u_time * 3.0) * 1.37) > 0.9975) { c = vec3(1.0); emis = true; }   // flash de foto
        }
    } else if (m == 10) {
        if (v < 0.0) { c = v_col * (0.5 + 0.6 * step(0.4, fract(u_time * 1.7 + u))); emis = true; }
    } else if (m == 11) {
        emis = true;
    } else if (m == 13 && v > 1.5) {    // pantallas laterales: llamas
        vec2 q = vec2(u, v - 2.0);
        float nz = hash(floor(vec2(q.x * 14.0, q.y * 22.0 - u_time * 9.0)));
        float fl = clamp((1.0 - q.y) * (0.55 + 0.7 * nz) + 0.25 * sin(q.x * 18.0 + u_time * 5.0), 0.0, 1.0);
        c = mix(vec3(0.08, 0.0, 0.0), mix(vec3(1.0, 0.3, 0.0), vec3(1.0, 0.92, 0.5), fl * fl), fl);
        emis = true;
    } else if (m == 13) {               // pantallas LED del escenario
        vec3 a = 0.5 + 0.5 * cos(u_time * 1.7 + u * 9.0 + vec3(0.0, 2.1, 4.2));
        float bars = step(0.5, fract(u * 12.0 - u_time * 2.0));
        float ring = 0.5 + 0.5 * sin(length(vec2(u, v) - 0.5) * 30.0 - u_time * 8.0);
        float pulse = pow(0.5 + 0.5 * sin(u_time * 12.0), 6.0);
        float mode = mod(floor(u_time / 4.0), 3.0);
        c = mode < 0.5 ? a * (0.3 + 0.7 * bars) : (mode < 1.5 ? vec3(1.0, 0.15, 0.1) * ring : mix(a, vec3(1.0), pulse));
        emis = true;
    } else if (m == 14) {               // veladoras
        c = v_col * (0.35 + 0.9 * step(0.4, fract(u_time * v + u * 0.37)));
        emis = true;
    }
    if (emis) {
        f_color = vec4(mix(c, u_fog, fg * 0.4), 1.0);
        return;
    }
    vec3 n = normalize(v_norm);
    float lit = u_amb + u_diff * abs(dot(n, -u_sun)) + u_head * clamp(1.0 - d / 90.0, 0.0, 1.0);
    c = c * lit * u_tint + extra;
    f_color = vec4(mix(c, u_fog, fg), 1.0);
}
"""

SKY_VS = """
#version 330
in vec2 in_pos; out vec2 v_ndc;
void main() { v_ndc = in_pos; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

SKY_FS = """
#version 330
uniform mat4 u_inv; uniform vec3 u_cam; uniform vec3 u_fog; uniform vec3 u_top; uniform vec3 u_sun;
uniform vec3 u_sunc; uniform float u_stars;
in vec2 v_ndc; out vec4 f_color;
void main() {
    vec4 p = u_inv * vec4(v_ndc, 1.0, 1.0);
    vec3 d = normalize(p.xyz / p.w - u_cam);
    float e = max(d.y, 0.0);
    vec3 c = mix(u_fog, u_top, pow(smoothstep(0.0, 0.6, e), 0.8));
    float s = max(dot(d, -u_sun), 0.0);
    c += u_sunc * (pow(s, 400.0) * 2.0 + pow(s, 12.0) * 0.25);
    if (u_stars > 0.5 && d.y > 0.05) {
        vec3 q = floor(d * 300.0);
        float h = fract(sin(dot(q, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
        if (h > 0.997) c += vec3(0.9);
    }
    f_color = vec4(c, 1.0);
}
"""

HUD_VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main() { v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

HUD_FS = """
#version 330
uniform sampler2D u_hud; in vec2 v_uv; out vec4 f_color;
void main() { f_color = texture(u_hud, v_uv); }
"""


# ---------------------------------------------------------------- coches
class Car:
    def __init__(self, pilot, x, z, ang, track, skill, corner, lane, is_player, slot, brake_f=0.7):
        self.pilot = pilot
        self.name = pilot["name"]
        self.color = pilot["body"]
        self.map_color = pilot["map"]
        v = VEH[pilot["vehicle"]]
        self.radius, self.size = v["radius"], v["length"] / 4.5
        self.cam_d, self.cam_h, self.hood = v["cam_d"], v["cam_h"], v["hood"]
        self.x, self.z, self.ang = x, z, ang
        self.vx = self.vz = 0.0
        self.is_player, self.lane, self.slot = is_player, lane, slot
        self.grid_side = -1 if lane < 0 else 1
        self.vmax = pilot["vmax"] * (1.0 if is_player else skill)
        self.accel = pilot["accel"] * (1.0 if is_player else skill)
        self.grip = pilot["grip"]
        self.mass = pilot["mass"]
        self.prof = track.speed_profile(self.grip * corner, BRAKE * brake_f, self.vmax)
        self.rb = 1.0
        self.nitro = 1.0
        self.boosting = self.boost_start = False
        self.nitro_ai = (not is_player) and skill >= 0.9
        self.hit = None
        self.s, self.idx, self.lat = track.project(x, z)[:3]
        self.s_new = self.s
        self.u = self.s - track.L if self.s > track.L * 0.5 else self.s
        self.lap_idx = math.floor(self.u / track.L)
        self.lap_start = 0.0
        self.best_lap = self.last_lap = None
        self.finished, self.finish_time = False, None
        self.steer = self.hurt = self.wrong = 0.0
        self.stuck = self.reverse_t = self.rec_steer = 0.0
        self.throttle = False
        self.surface = 0
        self.phase = random.random() * 6
        self.vao = self.shadow = None

    @property
    def vf(self):
        return self.vx * math.cos(self.ang) + self.vz * math.sin(self.ang)

    def drive(self, thr, brk, steer_in, hb, nitro, dt, track):
        self.throttle = thr
        was = self.boosting
        self.boosting = bool(nitro) and bool(thr) and self.nitro > 0.01
        self.boost_start = self.boosting and not was
        if self.boosting:
            self.nitro = max(0.0, self.nitro - dt / NITRO_TIME)
        else:
            self.nitro = min(1.0, self.nitro + dt * NITRO_RECHARGE)
        nb = NITRO_ACCEL if self.boosting else 1.0
        ca, sa = math.cos(self.ang), math.sin(self.ang)
        vf = self.vx * ca + self.vz * sa
        vr = -self.vx * sa + self.vz * ca
        self.steer += (steer_in - self.steer) * min(1.0, dt * (5 if self.is_player else 10))
        surf = self.surface
        vmax = self.vmax * self.rb * (NITRO_VMAX if self.boosting else 1.0)
        if thr:
            vf += self.accel * self.rb * nb * dt * max(0.0, 1 - vf / vmax) * (0.6 if surf else 1.0)
        if brk:
            if vf > 0.5:
                vf -= BRAKE * dt * (0.5 if surf else 1.0)
            else:
                vf = max(-10.0, vf - 6.0 * dt)
        if not thr and not brk:
            vf -= vf * 0.12 * dt
        vf -= 0.0004 * vf * abs(vf) * dt
        if surf == 1:
            vf -= vf * 0.55 * dt
        elif surf == 2:
            vf -= vf * 1.4 * dt
        delta = self.steer * 0.6 / (1 + abs(vf) / 16)
        yaw = vf * math.tan(delta) / WB
        alat = self.grip * (0.5 if surf else 1.0) * (1.6 if hb else 1.0)
        maxyaw = alat / max(abs(vf), 1.0)
        yaw = clamp(yaw, -maxyaw, maxyaw)
        if hb:
            vf -= vf * 0.6 * dt
            grip = 1.5
        else:
            grip = 3.5 if surf else 10.0
        vr -= vr * min(1.0, grip * dt)
        self.ang = (self.ang + yaw * dt) % TAU
        self.vx = vf * ca - vr * sa
        self.vz = vf * sa + vr * ca
        self.x += self.vx * dt
        self.z += self.vz * dt

        s, i, lat, cx, cz, nx, nz = track.project(self.x, self.z)
        impact = 0.0
        self.hit = None
        lim = LIMIT - self.radius
        if abs(lat) > lim:
            sg = 1.0 if lat > 0 else -1.0
            self.x, self.z = cx + nx * sg * lim, cz + nz * sg * lim
            ox, oz = nx * sg, nz * sg
            vn = self.vx * ox + self.vz * oz
            speed = math.hypot(self.vx, self.vz)
            if speed > 4:   # chispas al chocar o rozar el muro
                self.hit = (self.x + ox * self.radius, self.z + oz * self.radius, ox, oz,
                            max(vn, 0.0) + 0.1 * speed)
            if vn > 0:
                self.vx -= 1.3 * vn * ox
                self.vz -= 1.3 * vn * oz
                self.vx *= 0.92
                self.vz *= 0.92
                impact = vn
            lat = sg * lim
        self.s_new, self.idx, self.lat = s, i, lat
        a = abs(lat)
        self.surface = 0 if a < HW + CURB_W else (2 if track.sand[i, 0 if lat < 0 else 1] else 1)
        if impact > 4:
            self.hurt = 0.4
        else:
            self.hurt = max(0.0, self.hurt - dt)
        return impact


def ai_inputs(car, track, t, dt):
    vf = car.vf
    if car.reverse_t > 0:
        car.reverse_t -= dt
        return False, True, -car.rec_steer, False, False
    ld = (10 + max(vf, 0) * 0.55) * max(1.0, car.size * 0.9)
    tx, tz, tdx, tdz = track.point(car.s + ld)
    lane = clamp(car.lane + 1.2 * math.sin(t * 0.2 + car.phase), -HW + 1.8, HW - 1.8)
    tx += -tdz * lane
    tz += tdx * lane
    alpha = wrap_angle(math.atan2(tz - car.z, tx - car.x) - car.ang)
    delta = math.atan(2 * math.sin(alpha) / ld * WB)
    steer = clamp(delta / (0.6 / (1 + abs(vf) / 16)), -1, 1)
    ahead = int(max(vf, 0) * 0.35 / track.spacing) + 2
    target = car.prof[(car.idx + ahead) % track.n] * car.rb
    if vf < 1.0 and abs(alpha) < 2.5:
        car.stuck += dt
        if car.stuck > 1.5:
            car.stuck = 0
            car.reverse_t = 1.0
            car.rec_steer = steer
    else:
        car.stuck = 0
    nitro = (car.nitro_ai and vf > 25 and car.nitro > 0.3 and vf < target - 1
             and car.prof[(car.idx + ahead * 4) % track.n] > car.vmax * 0.75)
    return vf < target - 0.3, vf > target + 2.0, steer, False, nitro


# ---------------------------------------------------------------- sonido
class Sound:
    def __init__(self):
        self.ok = False
        self.muted = False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(44100, -16, 2, 1024)
            self.freq, _, self.ch = pygame.mixer.get_init()
            pygame.mixer.set_reserved(2)
            self.engine_ch = pygame.mixer.Channel(0)
            self.engine = [self._engine(50 + i * 10) for i in range(18)]
            self.crash = self._crash()
            self.beep = self._tone(660, 0.18)
            self.go = self._tone(990, 0.5)
            self.whoosh = self._whoosh()
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
        return self._make(wave * 0.6, 0.3)

    def _crash(self):
        n = int(self.freq * 0.35)
        w = np.random.default_rng(2).normal(0, 1, n)
        w = np.convolve(w, np.ones(12) / 12, "same") * 3
        return self._make(w * np.exp(-np.arange(n) / (n * 0.25)), 0.6)

    def _whoosh(self):
        n = int(self.freq * 0.8)
        t = np.arange(n) / self.freq
        w = np.convolve(np.random.default_rng(5).normal(0, 1, n), np.ones(6) / 6, "same")
        rumble = np.sin(TAU * (60 * t + 40 * t * t)) * 0.5
        return self._make((w * 0.8 + rumble) * np.minimum(1, t * 12) * np.exp(-t * 3), 0.55)

    def _tone(self, f, dur):
        t = np.arange(int(self.freq * dur)) / self.freq
        env = np.minimum(1, (dur - t) * 20)
        return self._make(np.sign(np.sin(TAU * f * t)) * 0.5 * env, 0.4)

    def play(self, name):
        if self.ok and not self.muted:
            getattr(self, name).play()

    def update_engine(self, speed, throttle):
        if not self.ok:
            return
        if self.muted:
            self.stop_engine()
            return
        lvl = int(clamp(abs(speed) / 75.0, 0, 1) * 15) + (2 if throttle else 0)
        lvl = clamp(lvl, 0, 17)
        if lvl != self.level:
            self.level = lvl
            self.engine_ch.play(self.engine[lvl], loops=-1)
            self.engine_ch.set_volume(0.5)

    def stop_engine(self):
        if self.ok:
            self.engine_ch.stop()
            self.level = -1


# ---------------------------------------------------------------- musica (chiptune propia + soundtrack de noche)
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


H = -1  # en las melodias: sostener la nota anterior
# Menu: Do mayor, tranquila (112 BPM). Acordes (raiz del bajo, menor?) y melodia en octavos por compas.
MENU_CHORDS = [(36, False), (33, True), (29, False), (31, False)] * 2
MENU_LEAD = [
    [76, H, 79, H, 84, H, 79, 76], [72, H, 76, H, 81, H, 79, 76],
    [77, H, 81, H, 84, H, 81, 77], [79, H, H, 74, 71, H, 74, H],
    [76, 79, 84, 79, 76, H, 72, H], [81, H, 79, 76, 72, H, 76, 79],
    [77, 81, 84, 86, 84, H, 81, 77], [79, H, 74, H, 71, H, H, H],
]
# Carrera (dia / atardecer): Mi menor, rapida (160 BPM)
RACE_CHORDS = [(40, True), (36, False), (38, False), (35, False)] * 2 + \
              [(33, True), (40, True), (36, False), (38, False), (33, True), (40, True), (36, False), (35, False)]
RACE_LEAD = [
    [76, H, 79, 76, 83, H, 81, 79], [79, H, 76, 72, 76, H, 79, H],
    [78, H, 74, 78, 81, H, 79, 78], [78, H, 75, 71, 75, 78, H, H],
    [76, 76, 79, 83, 88, H, 86, 83], [84, H, 83, 79, 76, H, 79, 84],
    [86, H, 81, 78, 74, 78, 81, H], [83, H, 78, 75, 71, H, 0, 0],
    [81, H, 84, 81, 76, H, 72, 76], [79, H, 76, 71, 76, 79, 83, H],
    [84, H, 79, 76, 72, 76, 79, 84], [86, H, 81, 78, 81, 83, 86, H],
    [88, H, 84, 81, 84, H, 81, 76], [83, H, 79, 76, 79, H, 76, 71],
    [76, 79, 84, H, 83, H, 79, 76], [78, H, H, 75, 71, H, H, H],
]


def synth_song(sr, bpm, chords, lead, style):
    """Sintetiza una cancion chiptune (onda cuadrada, triangular y ruido) que se repite sin cortes."""
    beat = 60.0 / bpm
    bar = 4 * beat
    n = int(len(chords) * bar * sr)
    out = np.zeros(n, np.float32)
    rng = np.random.default_rng(11)
    race = style == "race"

    def put(t0, w):
        s = int(t0 * sr)
        if s >= n:
            return
        e = min(n, s + len(w))
        out[s:e] += w[:e - s]

    def note(t0, f, dur, kind, vol, duty=0.5, decay=2.0):
        m = max(2, int(dur * sr))
        t = np.arange(m) / sr
        ph = (t * f) % 1.0
        w = np.where(ph < duty, 1.0, -1.0) if kind == "sq" else 4 * np.abs(ph - 0.5) - 1
        e = np.minimum(1.0, t / 0.004) * np.exp(-t * decay)
        r = min(m, int(0.015 * sr))
        e[m - r:] *= np.linspace(1, 0, r)
        put(t0, (w * e * vol).astype(np.float32))

    tk = np.arange(int(0.12 * sr)) / sr
    kick = np.sin(TAU * (50 * tk + 12 * (1 - np.exp(-tk * 35)))) * np.exp(-tk * 22)
    ts = np.arange(int(0.12 * sr)) / sr
    snare = rng.normal(0, 1, len(ts)) * np.exp(-ts * 28)
    th = np.arange(int(0.03 * sr)) / sr
    hat = np.diff(rng.normal(0, 1, len(th)) * np.exp(-th * 120), prepend=0)
    for b, (root, minor) in enumerate(chords):
        t0 = b * bar
        tones = [0, 3 if minor else 4, 7, 12]
        if race:
            for k in range(8):
                note(t0 + k * beat / 2, midi(root + (12 if k % 2 else 0)), beat / 2 * 0.9, "tri", 0.38, decay=3)
        else:
            for k in range(4):
                note(t0 + k * beat, midi(root + (7 if k % 2 else 0)), beat * 0.9, "tri", 0.34, decay=1.5)
        steps = 16 if race else 8
        for k in range(steps):
            d = bar / steps
            note(t0 + k * d, midi(root + 36 + tones[k % 4]), d * 0.9, "sq", 0.06 if race else 0.07, 0.125, 10)
        line = lead[b]
        k = 0
        while k < 8:
            mn = line[k]
            if mn and mn > 0:
                ln = 1
                while k + ln < 8 and line[k + ln] == H:
                    ln += 1
                note(t0 + k * beat / 2, midi(mn), beat / 2 * ln * 0.95, "sq", 0.16 if race else 0.14,
                     0.25 if race else 0.5, 1.2 if race else 1.8)
                k += ln
            else:
                k += 1
        if race:
            for q in range(4):
                put(t0 + q * beat, kick * 0.8 if q in (0, 2) else snare * 0.45)
                if q == 2:
                    put(t0 + q * beat + beat / 2, kick * 0.6)
            for k in range(8):
                put(t0 + k * beat / 2, hat * 0.18)
        else:
            put(t0, kick * 0.55)
            put(t0 + 2 * beat, snare * 0.25)
            for k in range(4):
                put(t0 + k * beat + beat / 2, hat * 0.1)
    out /= np.max(np.abs(out)) + 1e-6
    return out * 0.9


def crowd_ambience(sr, secs=8.0):
    """Murmullo de publico con gritos, en loop."""
    n = int(sr * secs)
    rng = np.random.default_rng(9)
    t = np.arange(n) / sr
    w = np.convolve(rng.normal(0, 1, n), np.ones(25) / 25, "same")
    out = w * (0.6 + 0.4 * np.sin(TAU * t / secs * 2))
    for _ in range(6):
        s = rng.integers(0, n - sr)
        m = int(sr * rng.uniform(0.8, 1.5))
        burst = np.diff(rng.normal(0, 1, m + 1)) * np.sin(np.linspace(0, np.pi, m)) * 0.25
        out[s:s + m] += burst
    f = int(0.02 * sr)
    out[:f] *= np.linspace(0, 1, f)
    out[-f:] *= np.linspace(1, 0, f)
    return out / (np.max(np.abs(out)) + 1e-6)


def clean_song_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"\[.*?\]|\(.*?\)", " ", name)
    name = re.sub(r"(?i)avenged\s+sevenfold|8[\s-]?bit(\s+(version|remake|cover))?!?|\s-\s|^-|-$", " ", name)
    return re.sub(r"\s+", " ", name).strip(" -") or os.path.basename(path)


class Music:
    """Elige que suena: menu (propia), ambiente de publico, carrera (propia) o el soundtrack de noche (carpeta music/)."""

    def __init__(self, sound):
        self.sound = sound
        self.ok = getattr(sound, "ok", False)
        self.mode = None
        self.now = None
        self.np_t = 99.0
        self.vol = 0.6
        self.order = []
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music")
        try:
            self.playlist = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                             if f.lower().endswith((".mp3", ".ogg", ".wav", ".flac"))]
        except OSError:
            self.playlist = []
        if not self.ok:
            return
        sr = sound.freq
        self.ch = pygame.mixer.Channel(1)
        self.songs = {"menu": sound._make(synth_song(sr, 112, MENU_CHORDS, MENU_LEAD, "menu"), 0.8),
                      "race": sound._make(synth_song(sr, 160, RACE_CHORDS, RACE_LEAD, "race"), 0.8),
                      "ambience": sound._make(crowd_ambience(sr), 0.6)}

    def set_mode(self, mode):
        if not self.ok or mode == self.mode:
            return
        if mode == "playlist" and not self.playlist:
            mode = "race"
        old, self.mode = self.mode, mode
        if old == "playlist":
            pygame.mixer.music.fadeout(700)
        else:
            self.ch.fadeout(500)
        if mode in self.songs:
            self.ch.play(self.songs[mode], loops=-1, fade_ms=700)
        elif mode == "playlist":
            self.order = []
            self.next_song()
        self.apply_volume()

    def next_song(self):
        """Canción al azar del soundtrack de noche, nunca la misma que acaba de sonar."""
        tried = set()
        for _ in range(len(self.playlist)):
            pool = [p for p in self.playlist if p != getattr(self, "cur", None) and p not in tried] or \
                   [p for p in self.playlist if p not in tried]
            if not pool:
                break
            path = random.choice(pool)
            tried.add(path)
            self.cur = path
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play(fade_ms=500)
                self.now = clean_song_name(path)
                self.np_t = 0.0
                return
            except pygame.error:
                continue
        self.now = None

    def apply_volume(self):
        v = 0.0 if self.sound.muted else self.vol
        self.ch.set_volume(v)
        pygame.mixer.music.set_volume(v)

    def update(self, dt):
        if not self.ok:
            return
        self.np_t += dt
        if self.mode == "playlist" and not pygame.mixer.music.get_busy():
            self.next_song()
        self.apply_volume()


# ---------------------------------------------------------------- juego
def smooth(f):
    f = clamp(f, 0.0, 1.0)
    return f * f * (3 - 2 * f)


def lerp(a, b, f):
    return tuple(x + (y - x) * f for x, y in zip(a, b))


class Game:
    def __init__(self, headless=False):
        self.headless = headless
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.init()
        if headless:
            self.ctx = moderngl.create_standalone_context()
            self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer((WIN_W, WIN_H))],
                                            depth_attachment=self.ctx.depth_renderbuffer((WIN_W, WIN_H)))
        else:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
            try:
                pygame.display.set_mode((WIN_W, WIN_H), pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
            except pygame.error:
                pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
                pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
                pygame.display.set_mode((WIN_W, WIN_H), pygame.OPENGL | pygame.DOUBLEBUF)
            pygame.display.set_caption("GTC Racing 3D - Gran Premio CDMX")
            self.ctx = moderngl.create_context()
            self.fbo = self.ctx.screen
        self.clock = pygame.time.Clock()
        self.track = Track()
        if headless:
            self.sound = Sound.__new__(Sound)
            self.sound.ok, self.sound.muted = False, True
        else:
            self.sound = Sound()

        self.music = Music(self.sound)
        ctx = self.ctx
        self.prog = ctx.program(vertex_shader=SCENE_VS, fragment_shader=SCENE_FS)
        self.sky_prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        self.hud_prog = ctx.program(vertex_shader=HUD_VS, fragment_shader=HUD_FS)
        scene_arr, self.info = build_scene(self.track)
        self.scene_vao = self.make_vao(scene_arr)
        self.concert_vao = self.make_vao(self.info.pop("concert_mesh"))
        self.fx = FX(ctx, self.prog, self.info)
        self.tod = 0          # arranca de dia (N cambia a atardecer / noche)
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self.sky_vao = ctx.vertex_array(self.sky_prog, [(quad, "2f", "in_pos")])
        self.hud_vao = ctx.vertex_array(self.hud_prog, [(quad, "2f", "in_pos")])
        self.sign_tex = ctx.texture((512, 1024), 4, make_sign_atlas())
        self.sign_tex.build_mipmaps()
        self.sign_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.hud_tex = ctx.texture((WIN_W, WIN_H), 4)
        self.hud = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        self.vehicle_vaos = {}
        for p in PILOTS:
            body, shadow = build_vehicle(p)
            self.vehicle_vaos[p["name"]] = (self.make_vao(body), self.make_vao(shadow))

        self.f_huge = pygame.font.SysFont("impact,arialblack,arial", 120)
        self.f_big = pygame.font.SysFont("impact,arialblack,arial", 64)
        self.f_mid = pygame.font.SysFont("impact,arialblack,arial", 38)
        self.f_small = pygame.font.SysFont("arial", 20, bold=True)
        self.f_tiny = pygame.font.SysFont("arial", 16, bold=True)
        self.minimap_pts = self.make_minimap()
        self.cam_mode = 0
        self.t = 0.0
        self.sel = len(PILOTS) - 1
        self.diff = 1
        self.new_race(attract=True)

    def make_vao(self, arr):
        buf = self.ctx.buffer(arr.tobytes())
        return self.ctx.vertex_array(self.prog, [(buf, "3f 3f 3f 3f", "in_pos", "in_norm", "in_col", "in_mat")])

    # ------------------------------------------------------------ setup
    def new_race(self, attract=False, intro=True):
        self.state = "menu" if attract else ("intro" if intro else "countdown")
        self.countdown = 3.99
        self.last_count = 4
        self.race_time = 0.0
        self.message, self.msg_t = "", 0.0
        self.results_t = 0.0
        me = PILOTS[self.sel] if not attract else PILOTS[-1]
        others = [p for p in PILOTS if p is not me]
        random.shuffle(others)
        player_slot = 3
        order = others[:player_slot] + [me] + others[player_slot:]
        _, skill, corner, _, _, brake_f = DIFFS[self.diff if not attract else 2]
        jitter = [0.015, 0.01, 0.005, 0.0, -0.005, -0.01]
        self.cars = []
        tr = self.track
        for slot, pilot in enumerate(order):
            s = tr.L - 10 - (slot // 2) * 12
            x, z, dx, dz = tr.point(s)
            off = -3.3 if slot % 2 == 0 else 3.3
            x += -dz * off
            z += dx * off
            is_p = pilot is me
            car = Car(pilot, x, z, math.atan2(dz, dx), tr, skill + jitter[slot], corner, off * 0.5, is_p, slot,
                      brake_f)
            car.vao, car.shadow = self.vehicle_vaos[pilot["name"]]
            self.cars.append(car)
            if is_p:
                self.player = car
        self.cam_yaw = self.player.ang
        self.cam_pos = None
        if self.state == "intro":
            self.build_intro()
        self.update_camera(0.0)
        # de noche cada carrera arranca con una cancion nueva al azar (tambien al reiniciar con R)
        if not attract and self.tod == 2 and self.music.mode == "playlist":
            self.music.next_song()

    def build_intro(self):
        tr = self.track
        self.gc = np.array([np.mean([c.x for c in self.cars]), np.mean([c.z for c in self.cars])])
        _, _, fx, fz = tr.point(tr.L - 20)
        self.gfwd = np.array([fx, fz])
        self.gright = np.array([-fz, fx])
        order = sorted(self.cars, key=lambda c: -c.slot)
        self.shots = [("aerial", 4.5, None)] + ([("concert", 4.5, None)] if self.tod == 2 else []) + [("car", 1.9, c) for c in order] + [("final", 2.2, self.player)]
        self.intro_total = sum(s[1] for s in self.shots)
        self.intro_t = 0.0

    def end_intro(self):
        self.state = "countdown"
        self.countdown = 3.99
        self.last_count = 4
        self.cam_yaw = self.player.ang
        self.cam_pos = None
        self.update_camera(0.0)

    def make_minimap(self):
        pts = self.track.pts
        lo, hi = pts.min(0), pts.max(0)
        size = 230
        sc = size / max(hi - lo)
        self.mm_lo, self.mm_sc = lo, sc
        self.mm_rect = pygame.Rect(WIN_W - size - 30, 20, size + 10, int((hi[1] - lo[1]) * sc) + 10)
        return [(self.mm_rect.x + 5 + (p[0] - lo[0]) * sc, self.mm_rect.y + 5 + (p[1] - lo[1]) * sc) for p in pts[::3]]

    # ------------------------------------------------------------ logica
    def player_inputs(self):
        k = pygame.key.get_pressed()
        thr = k[pygame.K_UP] or k[pygame.K_w]
        brk = k[pygame.K_DOWN] or k[pygame.K_s]
        st = (1 if (k[pygame.K_RIGHT] or k[pygame.K_d]) else 0) - (1 if (k[pygame.K_LEFT] or k[pygame.K_a]) else 0)
        nitro = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_x]
        return thr, brk, st, k[pygame.K_SPACE], nitro

    def show(self, msg, dur=2.0):
        self.message, self.msg_t = msg, dur

    def update(self, dt):
        self.t += dt
        self.msg_t = max(0.0, self.msg_t - dt)
        self.fx.concert = self.tod == 2
        self.fx.update(dt, self.t)
        if self.state == "select":
            self.sound.stop_engine()
            self.update_camera(dt)
            self.update_music(dt)
            return
        if self.state == "intro":
            self.intro_t += dt
            if self.intro_t >= self.intro_total:
                self.end_intro()
        if self.state == "countdown":
            self.countdown -= dt
            n = int(self.countdown)
            if n < self.last_count and n >= 1:
                self.sound.play("beep")
            self.last_count = n
            if self.countdown <= 0:
                self.state = "race"
                self.sound.play("go")
                self.show("¡ARRANCA!", 1.2)
        if self.state in ("race", "finished"):
            self.race_time += dt
        if self.state == "finished":
            self.results_t += dt

        tr = self.track
        # dificultad dinamica: los bots se ayudan o se frenan segun la distancia al jugador
        boost, slow = DIFFS[self.diff][3] if self.state != "menu" else (0.0, 0.0)
        for car in self.cars:
            car.rb = 1.0
            if not car.is_player and self.state == "race":
                gap = self.player.u - car.u
                if gap > 120:
                    car.rb = 1.0 + boost
                elif gap < -120:
                    car.rb = 1.0 - slow

        for car in self.cars:
            if self.state in ("countdown", "intro"):
                # congelados en la parrilla hasta el arranque (antes el freno los echaba en reversa)
                car.vx = car.vz = 0.0
                car.throttle = False
                car.s_new = car.s
                continue
            if car.is_player and self.state == "race":
                inp = self.player_inputs()
            else:
                inp = ai_inputs(car, tr, self.t, dt)
                if car.finished:
                    inp = (inp[0] and car.vf < 30, inp[1] or car.vf > 32, inp[2], False, False)
            impact = car.drive(*inp, dt, tr)
            if car.hit is not None:
                hx, hz, ox, oz, strength = car.hit
                self.fx.sparks(hx, hz, ox, oz, strength)
            if car.boosting:
                self.fx.nitro_flame(car)
                if car.is_player and car.boost_start and self.state == "race":
                    self.sound.play("whoosh")
            if impact > 4 and car.is_player and self.state != "menu":
                self.sound.play("crash")
        self.collide_cars()

        L = tr.L
        for car in self.cars:
            ds = car.s_new - car.s
            if ds > L / 2:
                ds -= L
            elif ds < -L / 2:
                ds += L
            car.s = car.s_new
            car.u += ds
            if ds < -1e-3 and abs(car.vf) > 1:
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

        self.update_camera(dt)
        self.update_music(dt)
        if self.state == "menu":
            self.sound.stop_engine()
        elif self.state == "intro":
            self.sound.update_engine(25 + 18 * math.sin(self.t * 2.3), True)
        else:
            self.sound.update_engine(self.player.vf, self.player.throttle)

    def music_mode(self):
        """Menu: cancion propia. De noche el soundtrack de A7X arranca desde la cinematica.
        Dia/tarde: publico antes de arrancar y luego la cancion propia de carrera."""
        if self.state in ("menu", "select"):
            return "menu"
        if self.tod == 2:
            return "playlist"
        if self.state in ("intro", "countdown"):
            return "ambience"
        return "race"

    def update_music(self, dt):
        self.music.set_mode(self.music_mode())
        self.music.update(dt)

    def collide_cars(self):
        cs = self.cars
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                md = a.radius + b.radius
                dx, dz = b.x - a.x, b.z - a.z
                d = math.hypot(dx, dz)
                if d >= md or d < 1e-6:
                    continue
                nx, nz = dx / d, dz / d
                ia, ib = 1 / a.mass, 1 / b.mass
                push = (md - d) / (ia + ib)
                a.x -= nx * push * ia
                a.z -= nz * push * ia
                b.x += nx * push * ib
                b.z += nz * push * ib
                rel = (b.vx - a.vx) * nx + (b.vz - a.vz) * nz
                if rel < -1.5:
                    self.fx.sparks((a.x + b.x) / 2, (a.z + b.z) / 2, -nz, nx, -rel * 1.5)
                if rel < 0:
                    jimp = -1.3 * rel / (ia + ib)
                    a.vx -= jimp * ia * nx
                    a.vz -= jimp * ia * nz
                    b.vx += jimp * ib * nx
                    b.vz += jimp * ib * nz
                    if (a.is_player or b.is_player) and -rel > 3 and self.state not in ("menu", "intro"):
                        self.player.hurt = 0.3
                        self.sound.play("crash")

    def standings(self):
        return sorted(self.cars, key=lambda c: (0, c.finish_time) if c.finished else (1, -c.u))

    # ------------------------------------------------------------ camaras
    def update_camera(self, dt):
        self.cam = self.camera(dt)

    def chase_pose(self, p, yaw):
        dx, dz = math.cos(yaw), math.sin(yaw)
        eye = (p.x - dx * p.cam_d, p.cam_h, p.z - dz * p.cam_d)
        target = (p.x + dx * 6, 1.2 * p.size, p.z + dz * 6)
        return eye, target

    def showroom_pose(self):
        x, z, dx, dz = self.track.point(160)
        return x, z, -dz, dx

    def camera(self, dt):
        p = self.player
        if self.state == "select":
            x, z, rx, rz = self.showroom_pose()
            sz = VEH[PILOTS[self.sel]["vehicle"]]["length"] / 4.5
            d = 5.0 + 3.6 * sz
            return (x + rx * d, 1.3 + 1.1 * sz, z + rz * d), (x, -0.25 + 0.75 * sz, z), 50
        if self.state == "intro":
            cam = self.intro_cam()
            if cam:
                return cam
        self.cam_yaw += wrap_angle(p.ang - self.cam_yaw) * min(1.0, dt * 4.5)
        if self.cam_mode == 0:
            eye, target = self.chase_pose(p, self.cam_yaw)
            eye = np.array(eye)
            if self.cam_pos is not None:
                eye = self.cam_pos + (eye - self.cam_pos) * min(1.0, dt * 12)
            self.cam_pos = eye
        else:
            dx, dz = math.cos(p.ang), math.sin(p.ang)
            eye = np.array([p.x + dx * 0.2 * p.size, p.hood, p.z + dz * 0.2 * p.size])
            self.cam_pos = None
            target = (eye[0] + dx * 20, p.hood - 0.6, eye[2] + dz * 20)
        if p.hurt > 0:
            eye = eye + np.random.uniform(-0.15, 0.15, 3)
        self.fov_b = getattr(self, "fov_b", 0.0)
        self.fov_b += ((1.0 if p.boosting else 0.0) - self.fov_b) * min(1.0, dt * 4)
        fov = 62 + min(18.0, abs(p.vf) * 0.2) + 10 * self.fov_b
        return tuple(eye), target, fov

    def current_shot(self):
        t = self.intro_t
        for shot in self.shots:
            if t < shot[1]:
                return shot, t
            t -= shot[1]
        return None, 0.0

    def intro_cam(self):
        shot, lt = self.current_shot()
        if shot is None:
            return None
        kind, dur, car = shot
        e = smooth(lt / dur)
        g, fw, rt = self.gc, self.gfwd, self.gright
        if kind == "aerial":
            e0 = g + fw * 190 + rt * 110
            e1 = g + fw * 30 + rt * 22
            ex, ez = e0 + (e1 - e0) * e
            tg = (g + fw * 70) + ((g - fw * 4) - (g + fw * 70)) * e
            return (ex, 95 + (5 - 95) * e, ez), (tg[0], 1.0, tg[1]), 55
        if kind == "concert":
            S, ff, rr = self.info["stage"]
            a = -0.7 + 1.4 * e
            dx = ff[0] * math.cos(a) + rr[0] * math.sin(a)
            dz = ff[1] * math.cos(a) + rr[1] * math.sin(a)
            dd = 54 - 14 * e
            return (S[0] + dx * dd, 16 - 7 * e, S[1] + dz * dd), (S[0], 16.0 + 6 * e, S[1]), 66
        if kind == "car":
            th = math.radians(28 + 55 * e)
            d = (6.8 - 1.4 * e) * max(1.0, car.size)
            fx, fz = math.cos(car.ang), math.sin(car.ang)
            rx, rz = -fz, fx
            side = car.grid_side
            ex = car.x + (fx * math.cos(th) + rx * side * math.sin(th)) * d
            ez = car.z + (fz * math.cos(th) + rz * side * math.sin(th)) * d
            ey = (0.9 + 0.6 * e) * max(1.0, car.size * 0.9)
            return (ex, ey, ez), (car.x + fx * 0.4, 1.0 * max(1.0, car.size * 0.9), car.z + fz * 0.4), 48
        # final: de lo alto a la camara de persecucion
        eye1, tgt1 = self.chase_pose(car, car.ang)
        fx, fz = math.cos(car.ang), math.sin(car.ang)
        eye0 = (car.x - fx * 45 + fz * 14, 24, car.z - fz * 45 - fx * 14)
        tgt0 = (car.x + fx * 20, 0.5, car.z + fz * 20)
        return lerp(eye0, eye1, e), lerp(tgt0, tgt1, e), 62

    # ------------------------------------------------------------ dibujo 3D
    def drawables(self):
        if self.state == "select":
            x, z, _, _ = self.showroom_pose()
            vao, sh = self.vehicle_vaos[PILOTS[self.sel]["name"]]
            return [(vao, sh, x, z, self.t * 0.7)]
        out = []
        for c in self.cars:
            hide = c is self.player and self.cam_mode == 1 and self.state in ("countdown", "race", "finished")
            out.append((None if hide else c.vao, c.shadow, c.x, c.z, c.ang))
        return out

    def render3d(self):
        ctx = self.ctx
        self.fbo.use()
        eye, target, fov = self.cam
        tod = TODS[self.tod][1]
        sun = np.array(tod["sun"], np.float32)
        sun = tuple(sun / np.linalg.norm(sun))
        proj = perspective(fov, WIN_W / WIN_H, 0.5, 9000.0)
        view = look_at(eye, target)
        vp = proj @ view
        ident = gl_bytes(np.identity(4, np.float32))
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.clear(*tod["fog"], 1.0)
        sp = self.sky_prog
        sp["u_inv"].write(gl_bytes(np.linalg.inv(vp)))
        for name, val in (("u_cam", tuple(eye)), ("u_fog", tod["fog"]), ("u_top", tod["top"]), ("u_sun", sun),
                          ("u_sunc", tod["sunc"]), ("u_stars", tod["stars"])):
            if sp.get(name, None) is not None:
                sp[name].value = val
        self.sky_vao.render(moderngl.TRIANGLE_STRIP)

        ctx.enable(moderngl.DEPTH_TEST)
        pr = self.prog
        pr["u_vp"].write(gl_bytes(vp))
        pr["u_model"].write(ident)
        for name, val in (("u_cam", tuple(eye)), ("u_fog", tod["fog"]), ("u_fogr", (250.0, 5200.0)), ("u_sun", sun),
                          ("u_hw", HW), ("u_tex", 0), ("u_time", self.t), ("u_amb", tod["amb"]),
                          ("u_diff", tod["diff"]), ("u_tint", tod["tint"]), ("u_night", tod["night"]),
                          ("u_head", tod["head"]),
                          ("u_stage", (float(self.info["stage"][0][0]), float(self.info["stage"][0][1]))),
                          ("u_pit", self.info["pit"]), ("u_concert", 1.0 if self.tod == 2 else 0.0)):
            if pr.get(name, None) is not None:
                pr[name].value = val
        self.sign_tex.use(0)
        self.scene_vao.render()
        if self.tod == 2:
            self.concert_vao.render()

        items = self.drawables()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for vao, sh, x, z, ang in items:
            pr["u_model"].write(gl_bytes(model_mat(x, z, ang)))
            sh.render()
        ctx.disable(moderngl.BLEND)
        for vao, sh, x, z, ang in items:
            if vao is not None:
                pr["u_model"].write(gl_bytes(model_mat(x, z, ang)))
                vao.render()

        # fuegos artificiales, llamaradas y luces del concierto (mezcla aditiva, sin escribir profundidad)
        strength = 0.35 if self.tod == 0 else 1.0
        self.fx.build(view, eye, concert_beams(self.t, self.info, strength) if self.tod == 2 else [])
        pr["u_model"].write(ident)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.fbo.depth_mask = False
        self.fx.render()
        self.fbo.depth_mask = True
        ctx.disable(moderngl.BLEND)

    # ------------------------------------------------------------ HUD
    def text(self, font, s, color, center=None, topleft=None, topright=None, outline=3):
        img = font.render(s, True, color)
        sh = font.render(s, True, (0, 0, 0))
        if center:
            r = img.get_rect(center=center)
        elif topright:
            r = img.get_rect(topright=topright)
        else:
            r = img.get_rect(topleft=topleft)
        if outline:
            for ox, oy in ((-outline, 0), (outline, 0), (0, -outline), (0, outline), (outline, outline)):
                self.hud.blit(sh, r.move(ox, oy))
        self.hud.blit(img, r)
        return r

    def draw_hud(self):
        self.hud.fill((0, 0, 0, 0))
        if self.state == "menu":
            self.draw_menu()
        elif self.state == "select":
            self.draw_select()
        elif self.state == "intro":
            self.draw_intro()
        else:
            self.draw_race_hud()

    def draw_race_hud(self):
        h = self.hud
        p = self.player
        inten = clamp((abs(p.vf) * 3.6 - 170) / 120, 0, 1) + (0.8 if p.boosting else 0)
        if inten > 0.02:   # lineas de velocidad
            rr_ = random.Random(int(self.t * 30))
            ccx, ccy = WIN_W / 2, WIN_H / 2 - 40
            lc = (150, 190, 255, int(80 * min(inten, 1.2))) if p.boosting else (255, 255, 255, int(60 * min(inten, 1.2)))
            for _ in range(int(26 * min(inten, 1.5))):
                a = rr_.uniform(0, TAU)
                r1 = rr_.uniform(380, 520)
                r2 = r1 + rr_.uniform(80, 220)
                pygame.draw.line(h, lc, (ccx + math.cos(a) * r1, ccy + math.sin(a) * r1 * 0.6),
                                 (ccx + math.cos(a) * r2, ccy + math.sin(a) * r2 * 0.6), 2)
        order = self.standings()
        pos = order.index(p) + 1
        r = self.text(self.f_big, f"{pos}º", (255, 215, 40), topleft=(30, 14))
        self.text(self.f_small, f"de {len(self.cars)}", (255, 255, 255), topleft=(r.right + 10, 58), outline=2)
        lap = min(LAPS, max(0, p.lap_idx) + 1)
        self.text(self.f_mid, f"VUELTA {lap}/{LAPS}", (255, 255, 255), topleft=(30, 96))
        self.text(self.f_mid, fmt_time(self.race_time if not p.finished else p.finish_time), (255, 255, 255),
                  topleft=(30, 142))
        self.text(self.f_small, f"ULTIMA  {fmt_time(p.last_lap)}", (230, 230, 230), topleft=(32, 190), outline=2)
        self.text(self.f_small, f"MEJOR   {fmt_time(p.best_lap)}", (230, 230, 230), topleft=(32, 214), outline=2)

        # minimapa + tabla de posiciones
        mm = pygame.Surface(self.mm_rect.size, pygame.SRCALPHA)
        mm.fill((0, 0, 0, 100))
        h.blit(mm, self.mm_rect)
        pygame.draw.lines(h, (40, 40, 40), True, self.minimap_pts, 7)
        pygame.draw.lines(h, (230, 230, 230), True, self.minimap_pts, 3)
        for c in self.cars:
            x = self.mm_rect.x + 5 + (c.x - self.mm_lo[0]) * self.mm_sc
            y = self.mm_rect.y + 5 + (c.z - self.mm_lo[1]) * self.mm_sc
            rr = 7 if c.is_player else 5
            pygame.draw.circle(h, (0, 0, 0), (int(x), int(y)), rr + 2)
            pygame.draw.circle(h, c.map_color, (int(x), int(y)), rr)
        y0 = self.mm_rect.bottom + 10
        board = pygame.Surface((self.mm_rect.w, 26 * len(order) + 8), pygame.SRCALPHA)
        board.fill((0, 0, 0, 110))
        h.blit(board, (self.mm_rect.x, y0))
        for i, c in enumerate(order):
            yy = y0 + 6 + i * 26
            pygame.draw.rect(h, c.map_color, (self.mm_rect.x + 8, yy + 3, 14, 14))
            col = (255, 215, 40) if c.is_player else (240, 240, 240)
            self.text(self.f_tiny, f"{i + 1}. {c.name}", col, topleft=(self.mm_rect.x + 30, yy), outline=0)

        # velocimetro
        cx, cy, R = WIN_W - 150, WIN_H - 140, 115
        pygame.draw.circle(h, (0, 0, 0, 150), (cx, cy), R)
        pygame.draw.circle(h, (255, 255, 255, 200), (cx, cy), R, 3)
        kmh = abs(p.vf) * 3.6
        for v in range(0, 301, 20):
            a = math.radians(225 - v / 300 * 270)
            r0 = R - (18 if v % 60 == 0 else 10)
            col = (255, 60, 40) if v >= 240 else (255, 255, 255)
            pygame.draw.line(h, col, (cx + r0 * math.cos(a), cy - r0 * math.sin(a)),
                             (cx + (R - 4) * math.cos(a), cy - (R - 4) * math.sin(a)), 3)
            if v % 60 == 0:
                t = self.f_small.render(str(v), True, (255, 255, 255))
                rr = R - 36
                h.blit(t, t.get_rect(center=(cx + rr * math.cos(a), cy - rr * math.sin(a))))
        a = math.radians(225 - min(kmh, 300) / 300 * 270)
        pygame.draw.line(h, (255, 40, 30), (cx, cy), (cx + (R - 14) * math.cos(a), cy - (R - 14) * math.sin(a)), 5)
        pygame.draw.circle(h, (30, 30, 30), (cx, cy), 10)
        self.text(self.f_mid, f"{int(kmh)}", (255, 255, 255), center=(cx, cy + 52), outline=2)
        self.text(self.f_small, "km/h", (220, 220, 220), center=(cx, cy + 82), outline=2)
        gx, gy, gh = cx - R - 34, cy - 100, 200
        pygame.draw.rect(h, (0, 0, 0, 150), (gx - 4, gy - 4, 22, gh + 8), border_radius=8)
        fill = int(gh * p.nitro)
        ncol = (200, 230, 255) if p.boosting else ((90, 170, 255) if p.nitro > 0.3 else (60, 90, 150))
        pygame.draw.rect(h, ncol, (gx, gy + gh - fill, 14, fill), border_radius=5)
        self.text(self.f_tiny, "NITRO", (150, 200, 255), center=(gx + 7, gy + gh + 18), outline=2)
        self.text(self.f_small, f"{p.name}  ·  {DIFFS[self.diff][0]}  ·  {TODS[self.tod][0]}", (255, 255, 255),
                  topleft=(30, WIN_H - 40), outline=2)

        self.draw_song_banner(WIN_H - 52)
        if self.state == "countdown":
            n = int(self.countdown)
            box = pygame.Rect(WIN_W // 2 - 150, 40, 300, 100)
            pygame.draw.rect(h, (20, 20, 20, 230), box, border_radius=18)
            for i in range(3):
                on = (3 - i) > n
                pygame.draw.circle(h, (230, 30, 30) if on else (70, 20, 20), (box.x + 60 + i * 90, box.centery), 34)
            if n >= 1:
                self.text(self.f_huge, str(n), (255, 220, 50), center=(WIN_W // 2, WIN_H // 2 - 40))
        if self.msg_t > 0 and self.message:
            self.text(self.f_huge, self.message, (255, 220, 50), center=(WIN_W // 2, WIN_H // 2 - 80))
        if self.state == "race" and p.wrong > 1.0 and int(self.t * 3) % 2 == 0:
            self.text(self.f_big, "¡SENTIDO CONTRARIO!", (255, 60, 40), center=(WIN_W // 2, 200))
        if self.state == "finished" and self.results_t > 2.0:
            panel = pygame.Rect(WIN_W // 2 - 330, 100, 660, 470)
            pygame.draw.rect(h, (0, 0, 0, 210), panel, border_radius=16)
            self.text(self.f_mid, "RESULTADOS", (255, 215, 40), center=(WIN_W // 2, 140))
            for i, c in enumerate(order):
                y = 185 + i * 56
                col = (255, 215, 40) if c.is_player else (240, 240, 240)
                pygame.draw.rect(h, c.map_color, (panel.x + 30, y + 8, 26, 26), border_radius=4)
                self.text(self.f_mid, f"{i + 1}. {c.name}", col, topleft=(panel.x + 70, y - 2), outline=2)
                self.text(self.f_tiny, c.pilot["brand"], (200, 200, 200), topleft=(panel.x + 74, y + 38), outline=0)
                t = fmt_time(c.finish_time) if c.finished else "en pista"
                self.text(self.f_mid, t, col, topright=(panel.right - 30, y - 2), outline=2)
            self.text(self.f_small, "R: otra carrera      ESC: menú", (230, 230, 230),
                      center=(WIN_W // 2, panel.bottom - 22), outline=2)

    def draw_menu(self):
        h = self.hud
        h.fill((0, 0, 0, 70))
        self.text(self.f_huge, "GTC RACING", (255, 210, 40), center=(WIN_W // 2, 170), outline=5)
        self.text(self.f_mid, "GRAN PREMIO CDMX", (255, 255, 255), center=(WIN_W // 2, 265))
        if int(self.t * 2) % 2 == 0:
            self.text(self.f_mid, "Pulsa ENTER para elegir piloto", (255, 255, 255), center=(WIN_W // 2, 400))
        lines = [f"{LAPS} vueltas  -  6 pilotos",
                 "Flechas / WASD: conducir     ESPACIO: freno de mano     SHIFT: nitro",
                 "C: cámara     N: horario     CTRL: cambiar canción (noche)     M: sonido     R: reiniciar     ESC: salir"]
        for i, l in enumerate(lines):
            self.text(self.f_small, l, (240, 240, 240), center=(WIN_W // 2, 500 + i * 32), outline=2)

    def draw_select(self):
        h = self.hud
        p = PILOTS[self.sel]
        self.text(self.f_big, "ELIGE A TU PILOTO", (255, 215, 40), center=(WIN_W // 2, 50))
        n = len(PILOTS)
        cw = 190
        x0 = WIN_W // 2 - n * cw // 2
        for i, q in enumerate(PILOTS):
            r = pygame.Rect(x0 + i * cw + 5, 92, cw - 10, 40)
            sel = i == self.sel
            pygame.draw.rect(h, (0, 0, 0, 200 if sel else 120), r, border_radius=8)
            pygame.draw.rect(h, q["map"], (r.x, r.bottom - 6, r.w, 6), border_bottom_left_radius=8,
                             border_bottom_right_radius=8)
            if sel:
                pygame.draw.rect(h, (255, 215, 40), r, 3, border_radius=8)
            self.text(self.f_tiny, q["name"], (255, 255, 255) if sel else (180, 180, 180),
                      center=(r.centerx, r.centery - 3), outline=0)
        # flechas
        for sgn, cx in ((-1, 120), (1, WIN_W - 120)):
            pts = [(cx + sgn * 30, 330), (cx - sgn * 20, 290), (cx - sgn * 20, 370)]
            pygame.draw.polygon(h, (0, 0, 0), [(x + 3, y + 3) for x, y in pts])
            pygame.draw.polygon(h, (255, 215, 40), pts)

        panel = pygame.Rect(90, 470, WIN_W - 180, 235)
        pygame.draw.rect(h, (0, 0, 0, 200), panel, border_radius=16)
        pygame.draw.rect(h, p["map"], (panel.x, panel.y, 10, panel.h), border_top_left_radius=16,
                         border_bottom_left_radius=16)
        self.text(self.f_big, p["name"], (255, 255, 255), topleft=(panel.x + 35, panel.y + 6))
        self.text(self.f_mid, p["brand"], p["map"], topleft=(panel.x + 38, panel.y + 80), outline=2)
        self.text(self.f_small, p["vname"], (220, 220, 220), topleft=(panel.x + 40, panel.y + 128), outline=0)
        stats = [("VELOCIDAD", p["vmax"], 84, 93), ("ACELERACIÓN", p["accel"], 10.0, 12.5),
                 ("AGARRE", p["grip"], 24.5, 27.5), ("PESO", p["mass"], 0.8, 1.7)]
        bx = panel.x + 560
        for i, (lab, v, lo, hi) in enumerate(stats):
            y = panel.y + 22 + i * 34
            self.text(self.f_tiny, lab, (230, 230, 230), topleft=(bx, y), outline=0)
            bar = pygame.Rect(bx + 140, y + 2, 300, 16)
            pygame.draw.rect(h, (60, 60, 60, 220), bar, border_radius=6)
            fw = int(bar.w * clamp((v - lo) / (hi - lo), 0.12, 1.0))
            pygame.draw.rect(h, p["map"] if p["map"] != (255, 255, 255) else (200, 200, 210),
                             (bar.x, bar.y, fw, bar.h), border_radius=6)
        dname, _, _, _, ddesc, _ = DIFFS[self.diff]
        dcol = [(90, 220, 90), (255, 215, 40), (255, 140, 30), (255, 50, 50)][self.diff]
        y = panel.y + 165
        self.text(self.f_small, "DIFICULTAD:", (230, 230, 230), topleft=(panel.x + 40, y + 8), outline=0)
        r = self.text(self.f_mid, f"‹ {dname} ›", dcol, topleft=(panel.x + 180, y - 2), outline=2)
        self.text(self.f_small, ddesc, (200, 200, 200), topleft=(r.right + 25, y + 8), outline=0)
        self.text(self.f_tiny, f"← →  piloto     ↑ ↓  dificultad     N  horario: {TODS[self.tod][0]}     "
                  "ENTER  correr     ESC  volver",
                  (200, 200, 200), center=(WIN_W // 2, panel.bottom - 18), outline=0)

    def draw_intro(self):
        h = self.hud
        shot, lt = self.current_shot()
        if shot is None:
            return
        kind, dur, car = shot
        bars = int(80 * smooth(self.intro_t / 0.6))
        pygame.draw.rect(h, (0, 0, 0), (0, 0, WIN_W, bars))
        pygame.draw.rect(h, (0, 0, 0), (0, WIN_H - bars, WIN_W, bars))
        if kind == "aerial":
            a = int(255 * smooth((lt - 0.6) / 0.8))
            img = self.f_huge.render("GRAN PREMIO CDMX", True, (255, 215, 40))
            img.set_alpha(a)
            h.blit(img, img.get_rect(center=(WIN_W // 2, WIN_H // 2 + 120)))
            sub = self.f_mid.render(f"AUTÓDROMO  ·  {self.track.L / 1000:.1f} KM  ·  {LAPS} VUELTAS  ·  "
                                    f"{DIFFS[self.diff][0]}", True, (255, 255, 255))
            sub.set_alpha(a)
            h.blit(sub, sub.get_rect(center=(WIN_W // 2, WIN_H // 2 + 205)))
        elif kind == "concert":
            a = int(255 * smooth((lt - 0.3) / 0.6))
            for txt, font, col, y in (("ESTADIO GNP SEGUROS", self.f_mid, (255, 255, 255), WIN_H // 2 + 150),
                                      ("AVENGED SEVENFOLD  ·  EN VIVO", self.f_big, (255, 60, 40), WIN_H // 2 + 215)):
                img = font.render(txt, True, col)
                img.set_alpha(a)
                h.blit(img, img.get_rect(center=(WIN_W // 2, y)))
        elif kind == "car":
            slide = smooth(lt / 0.35)
            card = pygame.Rect(int(-520 + 580 * slide), WIN_H - 250, 520, 140)
            pygame.draw.rect(h, (0, 0, 0, 210), card, border_radius=12)
            pygame.draw.rect(h, car.map_color, (card.x, card.y, 12, card.h), border_top_left_radius=12,
                             border_bottom_left_radius=12)
            self.text(self.f_big, car.name, (255, 255, 255), topleft=(card.x + 30, card.y + 4), outline=2)
            self.text(self.f_small, f"{car.pilot['brand']}  ·  {car.pilot['vname']}", car.map_color,
                      topleft=(card.x + 34, card.y + 84), outline=2)
            self.text(self.f_tiny, f"SALIDA  P{car.slot + 1}", (220, 220, 220), topleft=(card.x + 34, card.y + 112),
                      outline=0)
            if car.is_player:
                self.text(self.f_mid, "TÚ", (255, 215, 40), topright=(card.right - 20, card.y + 10))
        else:
            a = int(255 * smooth(lt / 0.5))
            img = self.f_huge.render("¡PREPÁRATE!", True, (255, 215, 40))
            img.set_alpha(a)
            h.blit(img, img.get_rect(center=(WIN_W // 2, WIN_H // 2 - 60)))
        if lt < 0.3:
            fade = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            fade.fill((0, 0, 0, int(255 * (1 - lt / 0.3))))
            h.blit(fade, (0, 0))
        self.text(self.f_tiny, "ENTER: saltar", (200, 200, 200), topright=(WIN_W - 20, WIN_H - 30), outline=0)
        self.draw_song_banner(22)

    def draw_song_banner(self, y):
        """Letrero con la cancion de A7X que esta sonando (unos segundos al empezar cada una)."""
        h = self.hud
        mu = self.music
        if mu.mode == "playlist" and mu.now and mu.np_t < 7 and not self.sound.muted:
            a = int(255 * clamp(min(mu.np_t / 0.5, (7 - mu.np_t) / 1.0), 0, 1))
            img = self.f_small.render(f"♪  AVENGED SEVENFOLD  -  {mu.now}  (8-bit)  ♪", True, (255, 220, 60))
            img.set_alpha(a)
            bar = pygame.Rect(WIN_W // 2 - img.get_width() // 2 - 14, y, img.get_width() + 28, 36)
            pygame.draw.rect(h, (0, 0, 0, int(a * 0.6)), bar, border_radius=10)
            h.blit(img, img.get_rect(center=bar.center))

    def draw(self):
        self.render3d()
        self.draw_hud()
        self.hud_tex.write(pygame.image.tobytes(self.hud, "RGBA", True))
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.hud_tex.use(1)
        self.hud_prog["u_hud"].value = 1
        self.hud_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

    def screenshot(self, path):
        data = self.fbo.read(components=3)
        img = pygame.image.frombuffer(data, (WIN_W, WIN_H), "RGB")
        pygame.image.save(pygame.transform.flip(img, False, True), path)

    # ------------------------------------------------------------ bucle
    def handle_key(self, key):
        st = self.state
        if key == pygame.K_ESCAPE:
            if st == "menu":
                return False
            if st == "select":
                self.new_race(attract=True)
            else:
                self.new_race(attract=True)
        elif st == "menu":
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.state = "select"
        elif st == "select":
            if key in (pygame.K_LEFT, pygame.K_a):
                self.sel = (self.sel - 1) % len(PILOTS)
            elif key in (pygame.K_RIGHT, pygame.K_d):
                self.sel = (self.sel + 1) % len(PILOTS)
            elif key in (pygame.K_UP, pygame.K_w):
                self.diff = min(len(DIFFS) - 1, self.diff + 1)
            elif key in (pygame.K_DOWN, pygame.K_s):
                self.diff = max(0, self.diff - 1)
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.new_race()
        elif st == "intro":
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.end_intro()
        else:
            if key == pygame.K_r:
                self.new_race(intro=False)
        if key == pygame.K_c:
            self.cam_mode = 1 - self.cam_mode
        elif key == pygame.K_n:
            self.tod = (self.tod + 1) % len(TODS)
        elif key in (pygame.K_LCTRL, pygame.K_RCTRL):
            # solo cambia el soundtrack de A7X (noche, en carrera); dia y tarde mantienen su cancion
            if self.music.mode == "playlist":
                self.music.next_song()
        elif key == pygame.K_m:
            self.sound.muted = not self.sound.muted
        return True

    def run(self):
        while True:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and not self.handle_key(e.key)):
                    pygame.quit()
                    sys.exit()
            self.update(dt)
            self.draw()
            pygame.display.flip()


if __name__ == "__main__":
    Game().run()

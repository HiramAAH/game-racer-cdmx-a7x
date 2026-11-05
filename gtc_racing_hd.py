"""
GTC RACING HD - Gran Premio CDMX  (edicion de graficos ultra)

Juego de carreras 3D con render fisico (PBR), sombras, reflejos en el asfalto, iluminacion HDR,
bloom, profundidad de campo y miles de aficionados animados. Estilo Burnout / Need for Speed.

Reutiliza de gtc_racing_3d.py (sin modificarlo) la fisica, la pista, la IA y los pilotos:
ese archivo debe estar en la misma carpeta.

Controles:
  Flechas / WASD : acelerar, frenar/reversa, girar
  ESPACIO        : freno de mano (derrapar recarga el nitro)
  SHIFT / X      : nitro
  C              : cambiar camara
  N              : horario (dia / atardecer / noche)
  CTRL           : cambiar de cancion (soundtrack de noche)
  F1             : calidad grafica
  M              : silenciar / activar sonido
  R              : reiniciar carrera
  ESC            : volver al menu / salir
"""
import json
import math
import os
import random
import re
import socket
import sys
import time
import wave

import numpy as np
import pygame
import moderngl

import gtc_racing_3d as base
from gtc_racing_3d import (Track, Car, ai_inputs, Sound, PILOTS, DIFFS, VEH, LAPS, HW, CURB_W, LIMIT, TAU,
                           STADIUM_PTS, BRAKE, clamp, wrap_angle, fmt_time, perspective, look_at, gl_bytes,
                           smooth, lerp, crowd_ambience, clean_song_name, draw_guadalupe, draw_flag)

HERE = os.path.dirname(os.path.abspath(__file__))
WIN_W, WIN_H = 1280, 720
F4 = np.float32
MAXL = 24                     # luces puntuales por cuadro
VSYNC = 0                     # 1 = sincronia vertical (sin "tearing" pero puede bajar a 30 FPS)

# ---------------------------------------------------------------- horarios (iluminacion HDR)
# sun = direccion HACIA el sol/luna
TOD_HD = [
    ("DÍA", dict(sun=(-0.42, 0.78, 0.46), sun_col=(5.6, 5.2, 4.6), sky_top=(0.13, 0.32, 0.75),
                 sky_hor=(0.62, 0.74, 0.9), ground=(0.16, 0.17, 0.13), fog_col=(0.58, 0.68, 0.82),
                 fog_den=0.00032, night=0.0, wet=0.0, lamps=0.0, exposure=0.95, clouds=0.5, shadows=1.0,
                 grade=(1.04, 1.0, 0.96), sat=1.08, stars=0.0, moon=0.0)),
    ("ATARDECER", dict(sun=(0.78, 0.12, -0.61), sun_col=(4.6, 2.1, 0.85), sky_top=(0.1, 0.1, 0.32),
                       sky_hor=(1.05, 0.5, 0.26), ground=(0.12, 0.08, 0.07), fog_col=(0.8, 0.46, 0.32),
                       fog_den=0.00045, night=1.0, wet=0.3, lamps=0.6, exposure=1.0, clouds=0.6, shadows=1.0,
                       grade=(1.08, 0.97, 0.9), sat=1.15, stars=0.0, moon=0.0)),
    ("NOCHE", dict(sun=(0.32, 0.55, -0.77), sun_col=(0.16, 0.2, 0.3), sky_top=(0.004, 0.006, 0.018),
                   sky_hor=(0.05, 0.04, 0.08), ground=(0.015, 0.015, 0.025), fog_col=(0.035, 0.03, 0.055),
                   fog_den=0.00055, night=1.0, wet=0.75, lamps=1.0, exposure=1.15, clouds=0.35, shadows=0.0,
                   grade=(0.95, 0.98, 1.08), sat=1.2, stars=1.0, moon=1.0)),
]

# calidad: tamano de sombras, reflejos (divisor de resolucion, 0 = sin), distancias del publico
QUALITY = [
    ("BAJA", dict(shadow=0, refl=0, near=40, far=200, dof=False, streak=False, scale=0.6)),
    ("MEDIA", dict(shadow=2048, refl=0, near=45, far=250, dof=True, streak=True, scale=0.75)),
    ("ALTA", dict(shadow=2048, refl=3, near=70, far=330, dof=True, streak=True, scale=0.85)),
    ("ULTRA", dict(shadow=4096, refl=2, near=120, far=520, dof=True, streak=True, scale=1.0)),
]

# ids de material (v_mat.x en los shaders)
M_DEF, M_ASPH, M_GRASS, M_CHECK, M_CURB, M_GRAVEL, M_CONC, M_SIGN, M_CROWD = 0, 1, 2, 3, 4, 5, 6, 7, 8
M_METAL, M_LAMP, M_FLAG, M_LED, M_CANDLE, M_PAPEL, M_GLASS, M_PAINT, M_FENCE = 10, 11, 12, 13, 14, 17, 18, 19, 20
M_WINDOWS, M_LEAF, M_ADS, M_STARTL, M_SEATS, M_EMIS, M_GOLD, M_RIBBON, M_FACADE = 21, 23, 25, 26, 27, 28, 29, 31, 32
M_TERRAIN, M_STUDIO, M_HEAD, M_TAIL, M_TIRE, M_PERSON, M_BARK, M_LINE = 33, 34, 35, 36, 37, 30, 38, 39


def rgb(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0)


def mat(mid, rough=0.6, metal=0.0, emis=0.0):
    return (float(mid), rough, metal, emis)


def norm(v):
    v = np.asarray(v, F4)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


# ---------------------------------------------------------------- geometria (normales suaves, PBR)
class Geo:
    """Acumula triangulos: pos(3) normal(3) color(3) material(4: id, rugosidad, metal, emision) uv(2)."""
    STRIDE = 15

    def __init__(self):
        self.parts = []

    def tris(self, P, N, C, M, UV=None):
        P = np.asarray(P, F4).reshape(-1, 3)
        n = len(P)
        if n == 0:
            return
        a = np.empty((n, 15), F4)
        a[:, 0:3] = P
        a[:, 3:6] = np.asarray(N, F4).reshape(-1, 3) if np.size(N) > 3 else np.asarray(N, F4)
        a[:, 6:9] = np.asarray(C, F4).reshape(-1, 3) if np.size(C) > 3 else np.asarray(C, F4)
        a[:, 9:13] = np.asarray(M, F4).reshape(-1, 4) if np.size(M) > 4 else np.asarray(M, F4)
        a[:, 13:15] = 0 if UV is None else np.asarray(UV, F4).reshape(-1, 2)
        self.parts.append(a)

    def merge(self, other):
        self.parts.extend(other.parts)

    def array(self):
        return np.concatenate(self.parts) if self.parts else np.zeros((0, 15), F4)

    def count(self):
        return sum(len(p) for p in self.parts)


def quad(g, a, b, c, d, col, m, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    a, b, c, d = (np.asarray(v, F4) for v in (a, b, c, d))
    nn = np.cross(b - a, c - a)
    ln = np.linalg.norm(nn)
    nn = nn / ln if ln > 1e-9 else np.array([0, 1, 0], F4)
    uv = np.asarray(uv, F4)
    g.tris([a, b, c, a, c, d], nn, col, m, uv[[0, 1, 2, 0, 2, 3]])


def grid(g, P, col, m, UV=None, wrap=False, flip=False, C=None, autoorient=False, orient=None):
    """Superficie a partir de una malla de puntos P (R, Cn, 3) con normales suaves.
    wrap=True cierra la superficie en la direccion de las columnas."""
    P = np.asarray(P, F4)
    R, Cn = P.shape[:2]
    dR = np.gradient(P, axis=0) if R > 1 else np.zeros_like(P)
    if wrap:
        dC = (np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)) * 0.5
    else:
        dC = np.gradient(P, axis=1)
    N = np.cross(dC, dR)
    ln = np.linalg.norm(N, axis=-1, keepdims=True)
    bad = ln[..., 0] < 1e-9
    N = N / np.maximum(ln, 1e-9)
    if bad.any():
        cen = P.reshape(-1, 3).mean(0)
        N[bad] = norm(P[bad] - cen)
    if autoorient:
        cen = P.reshape(-1, 3).mean(0)
        if np.mean(np.sum(N * (P - cen), axis=-1)) < 0:
            N = -N
    if orient is not None:          # orientar cada normal hacia 'orient' (vector o arreglo de vectores)
        o = np.broadcast_to(np.asarray(orient, F4), N.shape)
        N = np.where((np.sum(N * o, axis=-1) < 0)[..., None], -N, N)
    if flip:
        N = -N
    if UV is None:
        UV = np.zeros((R, Cn, 2), F4)
    UV = np.asarray(UV, F4)
    if C is not None:
        C = np.broadcast_to(np.asarray(C, F4), (R, Cn, 3))
    if wrap:
        P = np.concatenate([P, P[:, :1]], axis=1)
        N = np.concatenate([N, N[:, :1]], axis=1)
        UV = np.concatenate([UV, UV[:, :1]], axis=1)
        if C is not None:
            C = np.concatenate([C, C[:, :1]], axis=1)
        Cn += 1
    i, j = np.meshgrid(np.arange(R - 1), np.arange(Cn - 1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    v00, v01, v10, v11 = (i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)
    idx_r = np.stack([v00[0], v10[0], v11[0], v00[0], v11[0], v01[0]], 1).ravel()
    idx_c = np.stack([v00[1], v10[1], v11[1], v00[1], v11[1], v01[1]], 1).ravel()
    g.tris(P[idx_r, idx_c], N[idx_r, idx_c], col if C is None else C[idx_r, idx_c], m, UV[idx_r, idx_c])


def obox(g, o, ax, ay, az, col, m, uvscale=1.0):
    """Caja orientada: o + i*ax + j*ay + k*az, caras planas, uv en metros."""
    o, ax, ay, az = (np.asarray(v, F4) for v in (o, ax, ay, az))
    cen = o + (ax + ay + az) * 0.5

    def P(i, j, k):
        return o + ax * i + ay * j + az * k
    faces = [((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), ax, ay), ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1), ay, ax),
             ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), ay, az), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0), az, ay),
             ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1), ax, az), ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0), az, ax)]
    for a, b, c, d, e1, e2 in faces:
        pa, pb, pc, pd = P(*a), P(*b), P(*c), P(*d)
        nn = np.cross(pb - pa, pc - pa)
        if np.dot(nn, (pa + pc) * 0.5 - cen) < 0:
            pb, pd = pd, pb
        l1, l2 = np.linalg.norm(e1) * uvscale, np.linalg.norm(e2) * uvscale
        quad(g, pa, pb, pc, pd, col, m, ((0, 0), (l1, 0), (l1, l2), (0, l2)))


def box(g, x0, y0, z0, x1, y1, z1, col, m):
    obox(g, (x0, y0, z0), (x1 - x0, 0, 0), (0, y1 - y0, 0), (0, 0, z1 - z0), col, m)


def frame_box(g, center, fwd, half, col, m, y0=0.0, y1=1.0):
    """Caja apoyada en el piso orientada con fwd (2D): half = (medio largo, medio ancho)."""
    fx, fz = fwd
    rx, rz = -fz, fx
    hl, hw = half
    o = (center[0] - fx * hl - rx * hw, y0, center[1] - fz * hl - rz * hw)
    obox(g, o, (fx * 2 * hl, 0, fz * 2 * hl), (0, y1 - y0, 0), (rx * 2 * hw, 0, rz * 2 * hw), col, m)


def sgnpow(x, e):
    return np.sign(x) * np.abs(x) ** e


def superellipsoid(g, c, half, e, col, m, seg=24, rings=14, deform=None, colfn=None, uvfn=None):
    """Elipsoide 'cuadrado' de bordes redondeados (e < 1 = mas cuadrado). Normales suaves."""
    th = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)
    ph = np.linspace(-np.pi, np.pi, seg, endpoint=False)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    ct, st = np.cos(T), np.sin(T)
    x = half[0] * sgnpow(ct, e) * sgnpow(np.cos(Ph), e)
    y = half[1] * sgnpow(st, e)
    z = half[2] * sgnpow(ct, e) * sgnpow(np.sin(Ph), e)
    P = np.stack([x, y, z], -1)
    if deform is not None:
        P = deform(P)
    P = P + np.asarray(c, F4)
    C = colfn(P) if colfn is not None else None
    UV = uvfn(P) if uvfn is not None else None
    grid(g, P, col, m, UV=UV, wrap=True, C=C, autoorient=True)


def sphere(g, c, r, col, m, seg=16, rings=10, scale=(1, 1, 1), colfn=None):
    superellipsoid(g, c, (r * scale[0], r * scale[1], r * scale[2]), 1.0, col, m, seg, rings, colfn=colfn)


def revolve(g, profile, center, axis, col, m, seg=24, uaxis=None, UVv=None, C=None):
    """Revoluciona un perfil [(radio, altura)] alrededor de 'axis' (normalizado) desde 'center'."""
    axis = norm(axis)
    tmp = np.array([0, 1, 0], F4) if abs(axis[1]) < 0.9 else np.array([1, 0, 0], F4)
    u = norm(np.cross(axis, tmp)) if uaxis is None else norm(uaxis)
    v = np.cross(axis, u)
    prof = np.asarray(profile, F4)
    ang = np.linspace(0, TAU, seg, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    P = (np.asarray(center, F4)[None, None, :] + prof[:, 1, None, None] * axis[None, None, :]
         + prof[:, 0, None, None] * (ca[None, :, None] * u[None, None, :] + sa[None, :, None] * v[None, None, :]))
    UV = None
    if UVv is not None:
        UV = np.stack(np.meshgrid(UVv, ang / TAU, indexing="ij"), -1)[..., ::-1]
    grid(g, P, col, m, UV=UV, wrap=True, C=C, autoorient=False, flip=False)


def cylinder(g, c0, c1, r, col, m, seg=16, caps=True, r1=None):
    c0, c1 = np.asarray(c0, F4), np.asarray(c1, F4)
    ax = c1 - c0
    h = float(np.linalg.norm(ax))
    r1 = r if r1 is None else r1
    prof = [(r, 0.0), (r1, h)]
    revolve(g, prof, c0, ax / h, col, m, seg)
    if caps:
        axn = ax / h
        tmp = np.array([0, 1, 0], F4) if abs(axn[1]) < 0.9 else np.array([1, 0, 0], F4)
        u = norm(np.cross(axn, tmp))
        v = np.cross(axn, u)
        a = np.linspace(0, TAU, seg + 1)
        for cc, rr, sg in ((c0, r, -1), (c1, r1, 1)):
            ring = cc + rr * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
            P = np.stack([np.repeat(cc[None], seg, 0), ring[:-1], ring[1:]], 1).reshape(-1, 3)
            g.tris(P, axn * sg, col, m)


# ---------------------------------------------------------------- texturas (atlas HD 2048x2048)
SIGNS = [((20, 45, 150), (60, 110, 230), "GTC RACING", (255, 215, 40)),
         ((170, 15, 15), (230, 40, 40), "GTC", (255, 255, 255)),
         ((10, 100, 45), (30, 160, 80), "GRAN PREMIO CDMX", (255, 255, 255)),
         ((10, 10, 10), (40, 40, 40), "META  -  GTC", (255, 215, 40)),
         ((20, 110, 50), (40, 160, 70), "BODEGA AURRERA", (250, 210, 30)),
         ((235, 240, 250), (255, 255, 255), "SIMI", (25, 70, 170)),
         ((190, 20, 25), (235, 45, 45), "OXXO", (250, 200, 20)),
         ((235, 240, 250), (255, 255, 255), "BIMBO", (25, 70, 170)),
         ((235, 235, 235), (255, 255, 255), "¡EL GAAAS!", (210, 30, 30)),
         ((230, 180, 10), (255, 215, 40), "GTC", (20, 20, 20)),
         ((5, 5, 5), (45, 5, 5), "AVENGED SEVENFOLD", (240, 40, 40)),
         (None, None, "¡VIVA MÉXICO!", (20, 20, 20)),
         ((10, 25, 80), (25, 60, 150), "ESTADIO GNP SEGUROS", (255, 255, 255)),
         ((200, 0, 110), (240, 40, 150), "CDMX", (255, 255, 255)),
         ((5, 5, 5), (40, 30, 0), "HOY EN VIVO · A7X", (255, 210, 40)),
         ((220, 100, 10), (255, 150, 40), "TACOS EL GÜERO", (255, 255, 255))]
SPONSORS_HD = [0, 4, 11, 5, 6, 13, 7, 8, 14, 15, 1, 2, 9]
ATLAS = 2048


def ad_uv(k):
    return (0.0, 1.0, k * 64 / ATLAS, (k + 1) * 64 / ATLAS)


def bb_uv(k):
    u0 = (k % 2) * 0.25
    v0 = (1024 + (k // 2) * 128) / ATLAS
    return (u0, u0 + 0.25, v0, v0 + 128 / ATLAS)


GUADA_UV_HD = (0.5, 0.75, 1024 / ATLAS, 1792 / ATLAS)
FLAG_UV_HD = (0.75, 1.0, 1024 / ATLAS, 1316 / ATLAS)


def _sign_panel(s, rect, bg, bg2, txt, fg, big=True):
    x, y, w, h = rect
    if bg is None:
        draw_flag(s, pygame.Rect(rect))
    else:
        for i in range(h):
            f = i / max(1, h - 1)
            col = [int(bg[c] + (bg2[c] - bg[c]) * (1 - abs(f - 0.35) * 1.4)) for c in range(3)]
            s.fill([max(0, min(255, v)) for v in col], (x, y + i, w, 1))
        for k in range(-h, w, max(24, h)):
            pygame.draw.polygon(s, [min(255, v + 18) for v in bg2],
                                [(x + k, y + h), (x + k + h // 2, y), (x + k + h // 2 + 10, y), (x + k + 10, y + h)])
    pygame.draw.rect(s, fg, (x, y, w, max(3, h // 16)))
    pygame.draw.rect(s, fg, (x, y + h - max(3, h // 16), w, max(3, h // 16)))
    size = int(h * 0.62)
    while True:
        f = pygame.font.SysFont("bahnschrift,arialblack,impact,arial", size, bold=True, italic=True)
        t = f.render(txt, True, fg)
        if t.get_width() < w * 0.9 or size < 10:
            break
        size -= 2
    sh = f.render(txt, True, (0, 0, 0))
    cx, cy = x + w // 2, y + h // 2
    s.blit(sh, sh.get_rect(center=(cx + max(2, h // 30), cy + max(2, h // 30))))
    s.blit(t, t.get_rect(center=(cx, cy)))


def make_atlas_hd():
    pygame.font.init()
    s = pygame.Surface((ATLAS, ATLAS))
    s.fill((40, 40, 40))
    for k, (bg, bg2, txt, fg) in enumerate(SIGNS):
        _sign_panel(s, (0, k * 64, 1024, 64), bg, bg2, txt, fg)          # publicidad de muro (izquierda)
        _sign_panel(s, (1024, k * 64, 1024, 64), bg, bg2, txt, fg)       # (se repite a la derecha)
        _sign_panel(s, ((k % 2) * 512, 1024 + (k // 2) * 128, 512, 128), bg, bg2, txt, fg)   # espectaculares
    draw_guadalupe(s, pygame.Rect(1024, 1024, 512, 768))
    draw_flag(s, pygame.Rect(1536, 1024, 512, 292))
    return pygame.image.tobytes(s, "RGBA", False)


# ---------------------------------------------------------------- mundo HD
SHIRTS_HD = [rgb(200, 25, 35), rgb(25, 110, 55), rgb(235, 235, 235), rgb(25, 55, 150), rgb(240, 190, 30),
             rgb(18, 18, 22), rgb(230, 110, 20), rgb(130, 35, 150), rgb(220, 0, 110), rgb(50, 150, 210),
             rgb(18, 18, 22), rgb(30, 30, 36), rgb(245, 245, 245)]
METAL_SHIRTS = [rgb(12, 12, 14), rgb(18, 18, 20), rgb(25, 25, 30), rgb(12, 12, 14), rgb(100, 10, 10),
                rgb(35, 35, 60), rgb(60, 60, 64)]


class Crowd:
    """Lista de aficionados instanciados: x, y, z, yaw, color de playera, fase, tipo, tono de piel, escala."""

    def __init__(self):
        self.rows = []

    def add(self, x, y, z, face_x, face_z, rnd, shirt=None, typ=None, concert=False, palette=None):
        if typ is None:
            r = rnd.random()
            typ = 0 if r < 0.4 else 1 if r < 0.62 else 2 if r < 0.85 else 3
        sh = shirt or rnd.choice(palette or SHIRTS_HD)
        sh = tuple(clamp(c * rnd.uniform(0.85, 1.12), 0, 1) for c in sh)
        self.rows.append((x, y, z, math.atan2(face_x, face_z), sh[0], sh[1], sh[2], rnd.uniform(0, TAU), typ,
                          rnd.random(), rnd.uniform(0.92, 1.08), 1.0 if concert else 0.0))

    def array(self):
        return np.array(self.rows, F4).reshape(-1, 12)


def runs(mask):
    """Tramos contiguos True de una mascara circular (cada tramo incluye el punto siguiente para cerrar)."""
    n = len(mask)
    if mask.all():
        return [np.append(np.arange(n), 0)]
    if not mask.any():
        return []
    start = int(np.argmin(mask))
    idx = (np.arange(n) + start) % n
    out, cur = [], []
    for k in idx:
        if mask[k]:
            cur.append(k)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return [np.append(r, (r[-1] + 1) % n) for r in out if len(r) >= 1]


def offset_grid(track, idx, offs, ys, side):
    """Malla (len(offs), len(idx), 3) de puntos desplazados lateralmente de la pista."""
    pts, nrm = track.pts[idx], track.nrm[idx]
    offs = np.asarray(offs, F4)
    ys = np.broadcast_to(np.asarray(ys, F4), offs.shape)
    P = np.zeros((len(offs), len(idx), 3), F4)
    P[..., 0] = pts[None, :, 0] + nrm[None, :, 0] * side * offs[:, None]
    P[..., 2] = pts[None, :, 1] + nrm[None, :, 1] * side * offs[:, None]
    P[..., 1] = ys[:, None]
    return P


def s_unwrap(track, idx):
    s = track.cum[idx].astype(F4).copy()
    for k in range(1, len(s)):
        if s[k] < s[k - 1] - track.L / 2:
            s[k:] += track.L
    return s


def build_track_hd(g, track, info, lights):
    n = track.n
    pts, nrm, tan = track.pts, track.nrm, track.tan
    allidx = np.append(np.arange(n), 0)
    s_all = s_unwrap(track, allidx)
    # asfalto (5 franjas laterales para que la luz y los reflejos se vean finos)
    lat = np.array([-HW, -HW * 0.5, 0, HW * 0.5, HW], F4)
    P = offset_grid(track, allidx, lat, 0.06, 1)
    UV = np.stack(np.broadcast_arrays(s_all[None, :], lat[:, None]), -1)
    grid(g, P, rgb(58, 58, 62), mat(M_ASPH, 0.7), UV=UV, orient=(0, 1, 0))
    # orillas de asfalto hasta el pasto (banqueta oscura)
    for side in (-1, 1):
        P = offset_grid(track, allidx, [HW, HW + CURB_W + 0.6], 0.045, side)
        grid(g, P, rgb(48, 48, 50), mat(M_CONC, 0.85), orient=(0, 1, 0),
             UV=np.stack(np.broadcast_arrays(s_all[None, :], np.array([0, 1], F4)[:, None]), -1))
    # pianos con relieve
    curvy = np.abs(track.curv) > 1 / 350
    curvy = np.convolve(np.concatenate([curvy[-8:], curvy, curvy[:8]]).astype(float), np.ones(17), "same")[8:-8] > 0
    offs = [0.0, 0.15, 0.55, 1.0, CURB_W]
    ys = [0.065, 0.13, 0.16, 0.13, 0.05]
    for side in (-1, 1):
        for r in runs(curvy):
            P = offset_grid(track, r, [HW + o for o in offs], ys, side)
            ss = s_unwrap(track, r)
            UV = np.stack(np.broadcast_arrays(ss[None, :], np.array(offs, F4)[:, None]), -1)
            grid(g, P, (1, 1, 1), mat(M_CURB, 0.45), UV=UV, orient=(0, 1, 0))
    # grava en el exterior de las curvas cerradas
    for si, side in ((0, -1), (1, 1)):
        for r in runs(track.sand[:, si]):
            P = offset_grid(track, r, [HW + CURB_W + 0.6, LIMIT - 0.6], 0.04, side)
            grid(g, P, rgb(190, 170, 130), mat(M_GRAVEL, 0.95), orient=(0, 1, 0))
    # linea de meta y parrilla de salida
    p0, n0, t0 = pts[0], nrm[0], tan[0]
    a, b = p0 - n0 * HW, p0 + n0 * HW
    quad(g, (a[0], 0.07, a[1]), (b[0], 0.07, b[1]), (b[0] + t0[0] * 2.5, 0.07, b[1] + t0[1] * 2.5),
         (a[0] + t0[0] * 2.5, 0.07, a[1] + t0[1] * 2.5), (1, 1, 1), mat(M_CHECK, 0.5),
         ((0, -HW), (0, HW), (2.5, HW), (2.5, -HW)))
    for slot in range(6):
        s = track.L - 10 - (slot // 2) * 12
        x, z, dx, dz = track.point(s)
        off = -3.3 if slot % 2 == 0 else 3.3
        cx, cz = x - dz * off, z + dx * off
        rx, rz = -dz, dx
        for (u0, u1, w0, w1) in ((2.3, 2.5, -1.2, 1.2), (-0.8, 2.5, -1.2, -1.05), (-0.8, 2.5, 1.05, 1.2)):
            q = [(cx + dx * u + rx * w, 0.072, cz + dz * u + rz * w) for u, w in ((u0, w0), (u1, w0), (u1, w1), (u0, w1))]
            quad(g, *q, (0.92, 0.92, 0.92), mat(M_LINE, 0.55))
    # muros con publicidad + malla de proteccion
    for side in (-1, 1):
        q = pts + nrm * side * LIMIT
        valid = track.nearest_dist(q) >= LIMIT - 1.0
        segok = valid & np.roll(valid, -1) & (np.linalg.norm(np.roll(q, -1, 0) - q, axis=1) < 8)
        for r in runs(segok):
            ss = s_unwrap(track, r)
            inward = np.zeros((2, len(r), 3), F4)
            inward[..., 0] = -nrm[r, 0] * side
            inward[..., 2] = -nrm[r, 1] * side
            P = offset_grid(track, r, [LIMIT, LIMIT], [0.0, 1.05], side)
            UV = np.stack(np.broadcast_arrays(ss[None, :], np.array([0.0, 1.0], F4)[:, None]), -1)
            grid(g, P, (1, 1, 1), mat(M_ADS, 0.45), UV=UV, orient=inward[:1])
            P = offset_grid(track, r, [LIMIT, LIMIT + 0.4], 1.1, side)
            grid(g, P, rgb(150, 150, 150), mat(M_CONC, 0.8), orient=(0, 1, 0))
            P = offset_grid(track, r, [LIMIT + 0.4, LIMIT + 0.4], [1.1, 0.0], side)
            grid(g, P, rgb(130, 130, 130), mat(M_CONC, 0.8), orient=-inward[:1])
            P = offset_grid(track, r, [LIMIT + 0.2, LIMIT + 0.05, LIMIT - 0.35], [1.12, 3.2, 4.2], side)
            UV = np.stack(np.broadcast_arrays(ss[None, :], np.array([1.12, 3.2, 4.2], F4)[:, None]), -1)
            grid(g, P, rgb(150, 155, 160), mat(M_FENCE, 0.4, 0.8), UV=UV, orient=inward[:1])
            for k in r[::2]:
                base_ = pts[k] + nrm[k] * side * (LIMIT + 0.2)
                top = pts[k] + nrm[k] * side * (LIMIT - 0.35)
                cylinder(g, (base_[0], 1.1, base_[1]), (top[0], 4.25, top[1]), 0.05, rgb(120, 125, 130),
                         mat(M_METAL, 0.35, 1.0), seg=6, caps=False)
        # espectaculares detras del muro
        q2 = pts + nrm * side * (LIMIT + 3.5)
        valid2 = track.nearest_dist(q2) >= LIMIT + 2.5
        step = int(80 / track.spacing)
        start = 0 if side > 0 else step // 2
        for i in range(start, n, step):
            if not valid2[i]:
                continue
            fx, fz = nrm[i] * side
            billboard_hd(g, q2[i, 0], q2[i, 1], fx, fz, 12, 2.2, 5.4, SPONSORS_HD[(i // step) % len(SPONSORS_HD)],
                         lights)
    # torres de iluminacion
    for i in range(0, n, 25):
        side = 1 if (i // 25) % 2 == 0 else -1
        c = pts[i] + nrm[i] * side * (LIMIT + 6)
        if track.nearest_dist(c[None])[0] < LIMIT + 4:
            continue
        light_tower(g, c, -nrm[i] * side, lights)
    # portico de salida con semaforo
    gantry_hd(g, p0, n0, t0, info, lights)


def billboard_hd(g, cx, cz, fx, fz, width, y0, y1, k, lights=None):
    rx, rz = -fz, fx
    h = width / 2
    lx, lz, qx, qz = cx - rx * h, cz - rz * h, cx + rx * h, cz + rz * h
    u0, u1, v0, v1 = bb_uv(k)
    quad(g, (lx, y0, lz), (qx, y0, qz), (qx, y1, qz), (lx, y1, lz), (1, 1, 1), mat(M_SIGN, 0.35),
         ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))
    bx, bz = fx * 0.25, fz * 0.25
    frame_box(g, (cx + bx, cz + bz), (rx, rz), (h + 0.2, 0.18), rgb(60, 62, 68), mat(M_METAL, 0.5, 0.6), y0 - 0.2,
              y1 + 0.2)
    for px, pz in ((lx, lz), (qx, qz), (cx, cz)):
        cylinder(g, (px + bx, 0, pz + bz), (px + bx, y0, pz + bz), 0.16, rgb(70, 72, 78), mat(M_METAL, 0.45, 0.8),
                 seg=8, caps=False)
    # lamparas sobre el espectacular
    box(g, cx - rx * h * 0.8 - 0.2, y1 + 0.25, cz - rz * h * 0.8 - 0.2, cx + rx * h * 0.8 + 0.2, y1 + 0.45,
        cz + rz * h * 0.8 + 0.2, (1.0, 0.95, 0.85), mat(M_LAMP, 0.5, 0, 6.0))


def light_tower(g, c, toward, lights):
    cylinder(g, (c[0], 0, c[1]), (c[0], 22, c[1]), 0.38, rgb(140, 145, 150), mat(M_METAL, 0.4, 0.9), seg=10,
             r1=0.25)
    fx, fz = toward
    rx, rz = -fz, fx
    obox(g, (c[0] - rx * 2.5, 21.5, c[1] - rz * 2.5), (rx * 5, 0, rz * 5), (0, 2.2, 0), (fx * 0.6, 0, fz * 0.6),
         rgb(60, 60, 66), mat(M_METAL, 0.5, 0.7))
    for k in range(4):
        for row in range(2):
            px = c[0] - rx * 2.0 + rx * k * 1.33 + fx * 0.62
            pz = c[1] - rz * 2.0 + rz * k * 1.33 + fz * 0.62
            quad(g, (px - rx * 0.5, 21.7 + row, pz - rz * 0.5), (px + rx * 0.5, 21.7 + row, pz + rz * 0.5),
                 (px + rx * 0.5, 22.55 + row, pz + rz * 0.5), (px - rx * 0.5, 22.55 + row, pz - rz * 0.5),
                 (1.0, 0.93, 0.8), mat(M_LAMP, 0.4, 0, 7.0))
    lights.append((c[0] + fx * 10, 16.0, c[1] + fz * 10, 58.0, 1.0, 0.9, 0.75, 4.5))


def gantry_hd(g, p0, n0, t0, info, lights):
    steel = rgb(40, 42, 48)
    for side in (-1, 1):
        c = p0 + n0 * side * (HW + 2.6)
        for dx, dz in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)):
            cx, cz = c[0] + t0[0] * dx + n0[0] * dz, c[1] + t0[1] * dx + n0[1] * dz
            cylinder(g, (cx, 0, cz), (cx, 10.2, cz), 0.12, steel, mat(M_METAL, 0.4, 0.9), seg=6, caps=False)
        frame_box(g, c, t0, (0.7, 0.7), steel, mat(M_METAL, 0.45, 0.8), 0, 0.6)
    a = p0 - n0 * (HW + 3.3)
    obox(g, (a[0] - t0[0] * 0.6, 7.4, a[1] - t0[1] * 0.6), (n0[0] * (2 * HW + 6.6), 0, n0[1] * (2 * HW + 6.6)),
         (0, 2.8, 0), (t0[0] * 1.2, 0, t0[1] * 1.2), steel, mat(M_METAL, 0.45, 0.8))
    u0, u1, v0, v1 = bb_uv(3)
    w = 2 * HW + 3
    for sgn in (-1, 1):
        cc = p0 + t0 * 0.62 * sgn
        rx, rz = (-t0[1] * sgn, t0[0] * sgn)
        l, r = (cc[0] - rx * w / 2, cc[1] - rz * w / 2), (cc[0] + rx * w / 2, cc[1] + rz * w / 2)
        quad(g, (l[0], 8.6, l[1]), (r[0], 8.6, r[1]), (r[0], 10.0, r[1]), (l[0], 10.0, l[1]), (1, 1, 1),
             mat(M_SIGN, 0.3), ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))
    # semaforo: 5 columnas de luces frente a los coches (lado -t0)
    for k in range(5):
        off = (k - 2) * 2.2
        cx, cz = p0[0] - t0[0] * 0.63 + n0[0] * off, p0[1] - t0[1] * 0.63 + n0[1] * off
        for row in range(2):
            y = 7.65 + row * 0.55
            quad(g, (cx - n0[0] * 0.5, y, cz - n0[1] * 0.5), (cx + n0[0] * 0.5, y, cz + n0[1] * 0.5),
                 (cx + n0[0] * 0.5, y + 0.45, cz + n0[1] * 0.5), (cx - n0[0] * 0.5, y + 0.45, cz - n0[1] * 0.5),
                 (1, 1, 1), mat(M_STARTL, 0.3), ((k, 0), (k, 0), (k, 1), (k, 1)))
    lights.append((p0[0], 9.0, p0[1], 30.0, 1.0, 0.95, 0.85, 3.0))
    info["gantry"] = (float(p0[0]), float(p0[1]))


def grandstand_hd(g, crowd, rnd, base_, along, out, length, rows, dd, dh, seat_col, inst_rows, lights,
                  roof=True, spacing=1.05, concert=False):
    """Tribuna recta gigante con escalones, pasillos, publicidad, techo de armadura y publico instanciado."""
    bx, bz = base_
    ax, az = along
    ox, oz = out
    conc = rgb(150, 148, 142)
    face = (-ox, -oz)
    aisle = 40.0

    def W(d, a, y):
        return (bx + ox * d + ax * a, y, bz + oz * d + az * a)
    for r in range(rows):
        d0, d1, h = r * dd, (r + 1) * dd, (r + 1) * dh + 0.8
        crowdtex = r >= inst_rows
        m_ = mat(M_CROWD, 0.85) if crowdtex else mat(M_SEATS, 0.55)
        quad(g, W(d0, 0, h), W(d0, length, h), W(d1, length, h), W(d1, 0, h), seat_col, m_,
             ((0, r), (length, r), (length, r + 1), (0, r + 1)))
        quad(g, W(d0, 0, h - dh), W(d0, length, h - dh), W(d0, length, h), W(d0, 0, h),
             seat_col if crowdtex else conc, m_ if crowdtex else mat(M_CONC, 0.8),
             ((0, r - 0.35), (length, r - 0.35), (length, r), (0, r)))
        for e in (0, length):
            quad(g, W(d0, e, 0), W(d1, e, 0), W(d1, e, h), W(d0, e, h), conc, mat(M_CONC, 0.85))
        # pasillos (escaleras)
        for a in np.arange(aisle, length - 5, aisle):
            quad(g, W(d0, a - 0.7, h + 0.012), W(d0, a + 0.7, h + 0.012), W(d1, a + 0.7, h + 0.012),
                 W(d1, a - 0.7, h + 0.012), rgb(110, 110, 108), mat(M_CONC, 0.8))
        if r < inst_rows:
            for a in np.arange(1.0, length - 0.5, spacing):
                if abs(((a + aisle / 2) % aisle) - aisle / 2) < 1.2 or rnd.random() < 0.06:
                    continue
                x, y, z = W((r + 0.55) * dd, a + rnd.uniform(-0.15, 0.15), h)
                crowd.add(x, y, z, face[0], face[1], rnd, concert=concert)
    D, H = rows * dd, rows * dh + 0.8
    # pared frontal con publicidad, pared trasera y fachada
    for k, a0 in enumerate(np.arange(0, length, 24.0)):
        a1 = min(length, a0 + 24.0)
        quad(g, W(-0.05, a0, 0), W(-0.05, a1, 0), W(-0.05, a1, 1.6), W(-0.05, a0, 1.6), (1, 1, 1), mat(M_ADS, 0.4),
             ((a0 + 24 * (k * 5 % 13), 0), (a1 + 24 * (k * 5 % 13), 0), (a1 + 24 * (k * 5 % 13), 1),
              (a0 + 24 * (k * 5 % 13), 1)))
    quad(g, W(D, 0, 0), W(D, length, 0), W(D, length, H + 2.5), W(D, 0, H + 2.5), rgb(120, 122, 128),
         mat(M_FACADE, 0.35, 0.2), ((0, 0), (length, 0), (length, H + 2.5), (0, H + 2.5)))
    quad(g, W(-0.05, 0, 1.6), W(-0.05, length, 1.6), W(0, length, dh + 0.8), W(0, 0, dh + 0.8), conc,
         mat(M_CONC, 0.8))
    if roof:
        ry0, ry1 = H + 9.5, H + 7.0
        steel, white = rgb(70, 72, 80), rgb(225, 228, 235)
        quad(g, W(D * 0.15, 0, ry0), W(D * 0.15, length, ry0), W(D + 2, length, ry1), W(D + 2, 0, ry1), white,
             mat(M_METAL, 0.35, 0.3))
        quad(g, W(D * 0.15, 0, ry0 - 0.6), W(D * 0.15, length, ry0 - 0.6), W(D * 0.15, length, ry0),
             W(D * 0.15, 0, ry0), steel, mat(M_METAL, 0.4, 0.8))
        for a in np.arange(0, length + 0.1, 12.0):
            cylinder(g, W(D + 1.2, a, 0), W(D + 1.2, a, ry1), 0.35, steel, mat(M_METAL, 0.4, 0.85), seg=8)
            cylinder(g, W(D * 0.15, a, ry0 - 0.5), W(D + 1.2, a, ry1 - 0.5), 0.22, steel, mat(M_METAL, 0.4, 0.85),
                     seg=6, caps=False)
            cylinder(g, W(D * 0.55, a, ry0 - 3.5), W(D + 1.2, a, ry1 - 0.5), 0.16, steel, mat(M_METAL, 0.4, 0.85),
                     seg=6, caps=False)
            cylinder(g, W(D * 0.55, a, ry0 - 3.5), W(D * 0.15, a, ry0 - 0.5), 0.16, steel, mat(M_METAL, 0.4, 0.85),
                     seg=6, caps=False)
        quad(g, W(D * 0.17, 0, ry0 - 0.65), W(D * 0.17, length, ry0 - 0.65), W(D * 0.3, length, ry0 - 0.85),
             W(D * 0.3, 0, ry0 - 0.85), (1.0, 0.85, 0.65), mat(M_LAMP, 0.5, 0, 5.0))
        for a in np.arange(30, length, 90.0):
            x, y, z = W(D * 0.3, a, ry0 - 2)
            lights.append((x, y, z, 45.0, 1.0, 0.85, 0.65, 4.0))


def bowl_hd(g, crowd, rnd, track, sc, r0, lights, rows=40, dd=1.0, dh=0.5, inst_rows=16):
    """Graderio circular del estadio con techo de armadura, listones LED y publico."""
    pts = track.pts
    nseg = 96
    tr_r = np.hypot(pts[:, 0] - sc[0], pts[:, 1] - sc[1])
    tr_a = np.arctan2(pts[:, 1] - sc[1], pts[:, 0] - sc[0])
    rmax = r0 + rows * dd
    near_rad = (tr_r > r0 - LIMIT - 4) & (tr_r < rmax + LIMIT + 4)
    seat_cols = [rgb(225, 105, 25), rgb(25, 75, 180), rgb(235, 190, 35), rgb(190, 35, 35)]
    conc = rgb(150, 148, 142)
    steel, white = rgb(70, 72, 80), rgb(230, 232, 238)
    H = rows * dh + 0.8
    built = []
    for j in range(nseg):
        a0, a1 = j / nseg * TAU - math.pi, (j + 1) / nseg * TAU - math.pi
        am = (a0 + a1) / 2
        na = np.abs((tr_a - am + math.pi) % TAU - math.pi) < (a1 - a0) * 1.3 + 0.04
        built.append(not np.any(na & near_rad))
    for j in range(nseg):
        if not built[j]:
            continue
        a0, a1 = j / nseg * TAU - math.pi, (j + 1) / nseg * TAU - math.pi
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        col = seat_cols[(j // 6) % 4]

        def P(rad, c, s, y):
            return (sc[0] + rad * c, y, sc[1] + rad * s)
        for r in range(rows):
            ri, ro, h = r0 + r * dd, r0 + (r + 1) * dd, (r + 1) * dh + 0.8
            u0, u1 = ri * a0, ri * a1
            crowdtex = r >= inst_rows
            m_ = mat(M_CROWD, 0.85) if crowdtex else mat(M_SEATS, 0.55)
            quad(g, P(ri, c0, s0, h), P(ri, c1, s1, h), P(ro, c1, s1, h), P(ro, c0, s0, h), col, m_,
                 ((u0, r), (u1, r), (u1, r + 1), (u0, r + 1)))
            quad(g, P(ri, c0, s0, h - dh), P(ri, c1, s1, h - dh), P(ri, c1, s1, h), P(ri, c0, s0, h),
                 col if crowdtex else conc, m_ if crowdtex else mat(M_CONC, 0.8),
                 ((u0, r - 0.35), (u1, r - 0.35), (u1, r), (u0, r)))
            if r < inst_rows:
                rr = (ri + ro) * 0.5
                arc = rr * (a1 - a0)
                k = max(1, int(arc / 1.05))
                for q in range(k):
                    if rnd.random() < 0.06 or (j % 8 == 0 and q < 2):
                        continue
                    a = a0 + (q + 0.5 + rnd.uniform(-0.15, 0.15)) / k * (a1 - a0)
                    x, z = sc[0] + rr * math.cos(a), sc[1] + rr * math.sin(a)
                    crowd.add(x, h, z, -math.cos(a), -math.sin(a), rnd)
        # pared exterior (fachada), listón LED interior y techo
        quad(g, P(rmax, c0, s0, 0), P(rmax, c1, s1, 0), P(rmax, c1, s1, H + 3), P(rmax, c0, s0, H + 3),
             rgb(120, 124, 132), mat(M_FACADE, 0.35, 0.2), ((rmax * a0, 0), (rmax * a1, 0), (rmax * a1, H + 3),
                                                             (rmax * a0, H + 3)))
        quad(g, P(r0 - 0.1, c0, s0, 0), P(r0 - 0.1, c1, s1, 0), P(r0 - 0.1, c1, s1, 1.3), P(r0 - 0.1, c0, s0, 1.3),
             (1, 1, 1), mat(M_RIBBON, 0.3, 0, 2.5), ((r0 * a0, 0), (r0 * a1, 0), (r0 * a1, 1), (r0 * a0, 1)))
        rin, ry_in, ry_out = r0 + rows * dd * 0.3, H + 12.5, H + 9.0
        quad(g, P(rin, c0, s0, ry_in), P(rin, c1, s1, ry_in), P(rmax + 3, c1, s1, ry_out), P(rmax + 3, c0, s0, ry_out),
             white, mat(M_METAL, 0.35, 0.25))
        quad(g, P(rin + 0.3, c0, s0, ry_in - 0.9), P(rin + 0.3, c1, s1, ry_in - 0.9), P(rin + 6, c1, s1, ry_in - 1.3),
             P(rin + 6, c0, s0, ry_in - 1.3), (1.0, 0.6, 0.25), mat(M_LAMP, 0.5, 0, 6.0))
        if j % 2 == 0:
            xo, _, zo = P(rmax + 1.5, c0, s0, 0)
            cylinder(g, (xo, 0, zo), (xo, ry_out, zo), 0.45, steel, mat(M_METAL, 0.4, 0.85), seg=8)
            cylinder(g, P(rin, c0, s0, ry_in - 0.4), P(rmax + 1.5, c0, s0, ry_out - 0.4), 0.3, steel,
                     mat(M_METAL, 0.4, 0.85), seg=6, caps=False)
            cylinder(g, P(rin + 12, c0, s0, ry_in - 5), P(rmax + 1.5, c0, s0, ry_out - 0.4), 0.2, steel,
                     mat(M_METAL, 0.4, 0.85), seg=6, caps=False)
            cylinder(g, P(rin + 12, c0, s0, ry_in - 5), P(rin, c0, s0, ry_in - 0.4), 0.2, steel,
                     mat(M_METAL, 0.4, 0.85), seg=6, caps=False)
        if j % 8 == 0:
            lx, ly, lz = P(rin + 3, c0, s0, ry_in - 3)
            lights.append((lx, ly, lz, 70.0, 1.0, 0.75, 0.45, 5.0))
        # tapas laterales donde la pista entra al estadio
        for (cc, ss_), nb in (((c0, s0), j - 1), ((c1, s1), j + 1)):
            if not built[nb % nseg]:
                for r in range(rows):
                    ri, ro, h = r0 + r * dd, r0 + (r + 1) * dd, (r + 1) * dh + 0.8
                    quad(g, P(ri, cc, ss_, 0), P(ro, cc, ss_, 0), P(ro, cc, ss_, h), P(ri, cc, ss_, h), conc,
                         mat(M_CONC, 0.85))
    return rmax, H


def pit_building_hd(g, px0, pz0, lights):
    L_, Dp, Hp = 420.0, 18.0, 11.0
    white, dark = rgb(235, 236, 238), rgb(28, 30, 34)
    box(g, px0, 0, pz0, px0 + L_, 4.2, pz0 + Dp, white, mat(M_CONC, 0.6))
    for k in range(20):
        gx = px0 + 6 + k * 20.6
        box(g, gx, 0, pz0 - 0.05, gx + 15, 3.6, pz0 + 0.3, dark, mat(M_DEF, 0.7))
        quad(g, (gx + 15, 0.2, pz0 - 0.06), (gx, 0.2, pz0 - 0.06), (gx, 3.4, pz0 - 0.06), (gx + 15, 3.4, pz0 - 0.06),
             (1.0, 0.82, 0.6), mat(M_LAMP, 0.6, 0, 1.6))
        u0, u1, v0, v1 = bb_uv(SPONSORS_HD[k % len(SPONSORS_HD)])
        quad(g, (gx + 14, 3.7, pz0 - 0.08), (gx + 1, 3.7, pz0 - 0.08), (gx + 1, 4.15, pz0 - 0.08),
             (gx + 14, 4.15, pz0 - 0.08), (1, 1, 1), mat(M_SIGN, 0.35), ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))
    obox(g, (px0, 4.2, pz0 - 1.0), (L_, 0, 0), (0, Hp - 4.2, 0), (0, 0, Dp + 1.0), rgb(60, 80, 100),
         mat(M_FACADE, 0.12, 0.6))
    box(g, px0 - 2, Hp, pz0 - 3, px0 + L_ + 2, Hp + 0.6, pz0 + Dp + 1, white, mat(M_METAL, 0.4, 0.2))
    # torre de control con pantalla
    tx = px0 + L_ + 6
    box(g, tx, 0, pz0, tx + 14, 26, pz0 + 14, white, mat(M_CONC, 0.55))
    obox(g, (tx - 0.5, 18, pz0 - 0.5), (15, 0, 0), (0, 7, 0), (0, 0, 15), rgb(60, 80, 100), mat(M_FACADE, 0.1, 0.6))
    quad(g, (tx + 13, 6, pz0 - 0.06), (tx + 1, 6, pz0 - 0.06), (tx + 1, 15, pz0 - 0.06), (tx + 13, 15, pz0 - 0.06),
         (1, 1, 1), mat(M_LED, 0.3, 0, 2.0), ((0, 0), (1, 0), (1, 1), (0, 1)))
    for k in range(3):
        lights.append((px0 + 70 + k * 140, 6.0, pz0 - 6, 40.0, 1.0, 0.85, 0.7, 3.0))


def bridge_hd(g, track, i, k_ad, lights):
    p, nr, t = track.pts[i], track.nrm[i], track.tan[i]
    half = LIMIT + 2.6
    steel, white = rgb(60, 62, 70), rgb(225, 228, 235)
    a = p - nr * half
    obox(g, (a[0] - t[0] * 2, 7.2, a[1] - t[1] * 2), (nr[0] * 2 * half, 0, nr[1] * 2 * half), (0, 1.2, 0),
         (t[0] * 4, 0, t[1] * 4), white, mat(M_METAL, 0.35, 0.3))
    for side in (-1, 1):
        c = p + nr * side * half
        frame_box(g, c, t, (2.3, 1.2), steel, mat(M_METAL, 0.4, 0.8), 0, 9.6)
        for dz in (-2, 2):
            rp = p + t * dz
            ra, rb = rp - nr * half, rp + nr * half
            cylinder(g, (ra[0], 9.4, ra[1]), (rb[0], 9.4, rb[1]), 0.06, steel, mat(M_METAL, 0.3, 1.0), seg=6,
                     caps=False)
    u0, u1, v0, v1 = ad_uv(k_ad)
    w = 2 * half
    for sgn in (-1, 1):
        cc = p + t * 2.02 * sgn
        rx, rz = (-t[1] * sgn, t[0] * sgn)
        l, r = (cc[0] - rx * w / 2, cc[1] - rz * w / 2), (cc[0] + rx * w / 2, cc[1] + rz * w / 2)
        quad(g, (l[0], 7.25, l[1]), (r[0], 7.25, r[1]), (r[0], 8.35, r[1]), (l[0], 8.35, l[1]), (1, 1, 1),
             mat(M_SIGN, 0.3), ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))
    for side in (-0.5, 0.5):
        c = p + nr * side * half
        box(g, c[0] - 1, 7.1, c[1] - 1, c[0] + 1, 7.2, c[1] + 1, (1.0, 0.9, 0.75), mat(M_LAMP, 0.5, 0, 8.0))
        lights.append((c[0], 6.5, c[1], 28.0, 1.0, 0.9, 0.75, 4.0))


def tree_hd(g, x, z, s, rnd):
    bark = rgb(90, 62, 40)
    if rnd.random() < 0.5:          # pino
        cylinder(g, (x, 0, z), (x, 2.5 * s, z), 0.28 * s, bark, mat(M_BARK, 0.9), seg=6, caps=False, r1=0.2 * s)
        leaf = rgb(*[int(v * rnd.uniform(0.85, 1.15)) for v in (35, 95, 45)])
        for y0, y1, rr in ((1.6, 6.5, 2.7), (3.8, 8.4, 2.1), (6.0, 10.2, 1.4)):
            revolve(g, [(rr * s, y0 * s), (rr * s * 0.55, (y0 + y1) * 0.5 * s), (0.02, y1 * s)], (x, 0, z),
                    (0, 1, 0), leaf, mat(M_LEAF, 0.8), seg=9)
    else:                           # frondoso
        cylinder(g, (x, 0, z), (x, 3.4 * s, z), 0.3 * s, bark, mat(M_BARK, 0.9), seg=6, caps=False, r1=0.22 * s)
        leaf = rgb(*[int(v * rnd.uniform(0.85, 1.15)) for v in (55, 110, 40)])
        ph = rnd.uniform(0, 6)
        for k in range(3):
            cx, cy, cz = x + rnd.uniform(-1, 1) * s, (4.5 + k * 1.1 + rnd.uniform(-0.3, 0.3)) * s, \
                z + rnd.uniform(-1, 1) * s
            r = (2.6 - k * 0.5) * s

            def deform(P, ph=ph + k):
                d = norm(P)
                f = 1 + 0.16 * np.sin(d[..., 0] * 5 + ph) * np.sin(d[..., 2] * 4 - ph) + 0.08 * np.sin(d[..., 1] * 7)
                return P * f[..., None]
            superellipsoid(g, (cx, cy, cz), (r, r * 0.8, r), 1.0, leaf, mat(M_LEAF, 0.8), seg=9, rings=6,
                           deform=deform)


def angel_hd(g, x, z, lights):
    stone, dstone = rgb(232, 226, 210), rgb(196, 190, 176)
    gold, bronze = rgb(255, 190, 60), rgb(90, 70, 45)
    ang = np.linspace(0, TAU, 33)
    for j in range(32):
        a0, a1 = ang[j], ang[j + 1]
        quad(g, (x, 0.06, z), (x, 0.06, z), (x + 26 * math.cos(a1), 0.06, z + 26 * math.sin(a1)),
             (x + 26 * math.cos(a0), 0.06, z + 26 * math.sin(a0)), rgb(70, 120, 50), mat(M_GRASS, 0.9))
    for j in range(24):
        a = j / 24 * TAU
        fx, fz = x + 22 * math.cos(a), z + 22 * math.sin(a)
        sphere(g, (fx, 0.4, fz), 1.1, (rgb(0, 120, 70), rgb(240, 240, 240), rgb(210, 20, 40))[j % 3],
               mat(M_DEF, 0.8), seg=8, rings=5, scale=(1, 0.5, 1))
    box(g, x - 12, 0, z - 12, x + 12, 1.5, z + 12, dstone, mat(M_CONC, 0.7))
    box(g, x - 9, 1.5, z - 9, x + 9, 4, z + 9, stone, mat(M_CONC, 0.6))
    box(g, x - 5.5, 4, z - 5.5, x + 5.5, 10, z + 5.5, stone, mat(M_CONC, 0.55))
    for sx in (-1, 1):
        for sz in (-1, 1):
            px, pz = x + sx * 7.5, z + sz * 7.5
            box(g, px - 1.2, 4, pz - 1.2, px + 1.2, 5, pz + 1.2, dstone, mat(M_CONC, 0.6))
            superellipsoid(g, (px, 6.0, pz), (0.55, 1.0, 0.45), 0.7, bronze, mat(M_METAL, 0.35, 1.0), 10, 8)
            sphere(g, (px, 7.4, pz), 0.38, bronze, mat(M_METAL, 0.35, 1.0), 10, 6)
    box(g, x - 4, 10, z - 4, x + 4, 11.5, z + 4, dstone, mat(M_CONC, 0.6))
    # columna con estrias (24 lados) y anillos
    revolve(g, [(2.1, 11.5), (2.0, 20), (1.95, 30), (1.9, 46)], (x, 0, z), (0, 1, 0), stone, mat(M_CONC, 0.5), seg=24)
    for yy in (18, 26, 34, 42):
        revolve(g, [(2.0, yy), (2.3, yy + 0.3), (2.0, yy + 0.6)], (x, 0, z), (0, 1, 0), dstone, mat(M_CONC, 0.5),
                seg=24)
    box(g, x - 3, 46, z - 3, x + 3, 47.5, z + 3, dstone, mat(M_CONC, 0.5))
    revolve(g, [(1.5, 47.5), (1.3, 49.5), (0.0, 49.6)], (x, 0, z), (0, 1, 0), gold, mat(M_GOLD, 0.25, 1.0), seg=16)
    ay = 49.5
    gm = mat(M_GOLD, 0.22, 1.0)
    superellipsoid(g, (x, ay + 1.0, z), (0.55, 1.0, 0.45), 0.8, gold, gm, 12, 8)
    superellipsoid(g, (x, ay + 2.6, z), (0.6, 0.95, 0.42), 0.8, gold, gm, 12, 8)
    sphere(g, (x, ay + 3.95, z), 0.45, gold, gm, 12, 8)
    cylinder(g, (x + 0.45, ay + 3.1, z), (x + 0.75, ay + 5.3, z), 0.14, gold, gm, seg=8)
    revolve(g, [(0.5, -0.08), (0.62, 0.0), (0.5, 0.08)], (x + 0.75, ay + 5.75, z), (0, 0, 1), gold, gm, seg=16)
    cylinder(g, (x - 0.5, ay + 2.8, z), (x - 1.5, ay + 2.5, z), 0.12, gold, gm, seg=8)
    for side in (-1, 1):            # alas (superficie curva)
        uu, vv = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 7), indexing="ij")
        P = np.stack([x - 0.2 - uu * 1.6 - vv * 0.4, ay + 1.6 + vv * 5.0 - uu * 1.0,
                      z + side * (0.35 + uu * 3.0 * (0.4 + 0.6 * vv))], -1)
        P[..., 0] -= np.sin(uu * np.pi) * 0.3
        grid(g, P, gold, gm, orient=(0, 0, side))
        grid(g, P + np.array([0.04, 0, 0], F4), gold, gm, orient=(0, 0, -side))
    lights.append((x + 8, 12.0, z + 8, 55.0, 1.0, 0.8, 0.5, 6.0))


def flagpole_hd(g, x, z, height=50.0):
    revolve(g, [(0.5, 0), (0.32, height)], (x, 0, z), (0, 1, 0), rgb(210, 212, 218), mat(M_METAL, 0.25, 1.0), seg=12)
    box(g, x - 2.5, 0, z - 2.5, x + 2.5, 1.2, z + 2.5, rgb(190, 185, 175), mat(M_CONC, 0.7))
    sphere(g, (x, height + 0.5, z), 0.7, rgb(255, 190, 60), mat(M_GOLD, 0.25, 1.0), 10, 6)
    fw, fh, nx, ny = 26.0, 14.8, 28, 12
    u0, u1, v0, v1 = FLAG_UV_HD
    xs = np.linspace(0, 1, nx + 1)
    ys = np.linspace(0, 1, ny + 1)
    U, Vv = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([x + 0.5 + fw * U, height - 1 - fh * Vv, np.full_like(U, z)], -1)
    UV = np.stack([u0 + (u1 - u0) * U, v0 + (v1 - v0) * Vv], -1)
    grid(g, P, (1, 1, 1), mat(M_FLAG, 0.75), UV=UV, orient=(0, 0, 1))
    grid(g, P + np.array([0, 0, -0.02], F4), (1, 1, 1), mat(M_FLAG, 0.75), UV=UV, orient=(0, 0, -1))


def guadalupe_wall_hd(g, c, f, lights, rnd):
    fx, fz = f
    rx, rz = -fz, fx
    obox(g, (c[0] - rx * 6.5 + fx * 0.1, 0, c[1] - rz * 6.5 + fz * 0.1), (rx * 13, 0, rz * 13), (0, 14, 0),
         (fx * 0.7, 0, fz * 0.7), rgb(165, 75, 45), mat(M_CONC, 0.9))
    obox(g, (c[0] - rx * 6.8 + fx * 0.05, 14, c[1] - rz * 6.8 + fz * 0.05), (rx * 13.6, 0, rz * 13.6), (0, 0.5, 0),
         (fx * 0.8, 0, fz * 0.8), rgb(120, 50, 30), mat(M_CONC, 0.8))
    u0, u1, v0, v1 = GUADA_UV_HD
    w = 8.8
    lx, lz, qx, qz = c[0] - rx * w / 2 - fx * 0.02, c[1] - rz * w / 2 - fz * 0.02, c[0] + rx * w / 2 - fx * 0.02, \
        c[1] + rz * w / 2 - fz * 0.02
    quad(g, (lx, 0.6, lz), (qx, 0.6, qz), (qx, 13.6, qz), (lx, 13.6, lz), (1, 1, 1), mat(M_SIGN, 0.55),
         ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))
    for k in range(-6, 7):
        px, pz = c[0] + rx * k * 0.85 - fx * 0.9, c[1] + rz * k * 0.85 - fz * 0.9
        if k % 2 == 0:
            cylinder(g, (px, 0, pz), (px, 0.4, pz), 0.1, rgb(200, 40, 30), mat(M_DEF, 0.6), seg=8)
            sphere(g, (px, 0.48, pz), 0.06, (1.0, 0.6, 0.2), mat(M_CANDLE, 0.5, 0, 9.0), 6, 4)
        else:
            sphere(g, (px, 0.3, pz), 0.32, [rgb(250, 140, 0), rgb(215, 20, 40), rgb(250, 250, 250)][k % 3],
                   mat(M_DEF, 0.8), 8, 5)
    lights.append((c[0] - fx * 2.5, 1.5, c[1] - fz * 2.5, 14.0, 1.0, 0.55, 0.2, 3.0))


def papel_picado_hd(g, pa, pb, h=7.5, rows=4):
    cols = [rgb(230, 0, 124), rgb(250, 140, 0), rgb(40, 170, 70), rgb(40, 110, 230), rgb(250, 215, 0),
            rgb(150, 50, 200)]
    for p in (pa, pb):
        cylinder(g, (p[0], 0, p[1]), (p[0], h + 0.8, p[1]), 0.16, rgb(120, 90, 60), mat(M_BARK, 0.8), seg=8)
    dx, dz = pb[0] - pa[0], pb[1] - pa[1]
    ln = math.hypot(dx, dz)
    ux, uz = dx / ln, dz / ln
    nn = int(ln / 0.7)
    for r in range(rows):
        yb = h - r * 0.9
        for i in range(nn):
            t = (i + 0.5) / nn
            px, pz = pa[0] + dx * t, pa[1] + dz * t
            y = yb - 1.6 * 4 * t * (1 - t)
            col = cols[(i + r) % len(cols)]
            m_ = (float(M_PAPEL), 0.7, 0.0, i * 0.7 + r)
            quad(g, (px - ux * 0.26, y - 0.62, pz - uz * 0.26), (px + ux * 0.26, y - 0.62, pz + uz * 0.26),
                 (px + ux * 0.26, y, pz + uz * 0.26), (px - ux * 0.26, y, pz - uz * 0.26), col, m_)


def city_hd(g, cxm, czm, rnd, lights):
    pal = [rgb(190, 180, 168), rgb(160, 168, 178), rgb(205, 192, 165), rgb(140, 142, 150), rgb(180, 150, 132),
           rgb(110, 125, 140)]
    for _ in range(320):
        a = rnd.uniform(0, TAU)
        r = rnd.uniform(1500, 2900)
        x, z = cxm + r * math.cos(a), czm + r * math.sin(a)
        w, d = rnd.uniform(20, 60), rnd.uniform(20, 60)
        h = rnd.choice([12, 18, 25, 35, 50, 70, 90, 120])
        obox(g, (x - w / 2, 0, z - d / 2), (w, 0, 0), (0, h, 0), (0, 0, d), pal[rnd.randrange(len(pal))],
             mat(M_WINDOWS, 0.4, 0.1))
        if h >= 70:
            box(g, x - 0.8, h, z - 0.8, x + 0.8, h + 6, z + 0.8, (1.0, 0.1, 0.05), mat(M_EMIS, 0.5, 0, 6.0))
    # rascacielos tipo Reforma al poniente
    for k in range(7):
        x, z = cxm - 1900 + k * 70 + rnd.uniform(-20, 20), czm - 1300 + rnd.uniform(-120, 120)
        h = rnd.uniform(150, 245)
        w = rnd.uniform(28, 40)
        obox(g, (x - w / 2, 0, z - w / 2), (w, 0, 0), (0, h, 0), (0, 0, w), rgb(90, 115, 140),
             mat(M_WINDOWS, 0.15, 0.5))
        box(g, x - 1, h, z - 1, x + 1, h + 14, z + 1, (1.0, 0.1, 0.05), mat(M_EMIS, 0.5, 0, 8.0))


def mountain_hd(g, x, z, r, h, rnd, snow=0.0, seg=28, rings=10):
    ang = np.linspace(0, TAU, seg, endpoint=False)
    rad = np.linspace(1, 0, rings + 1) ** 0.9
    A, Rr = np.meshgrid(ang, rad, indexing="ij")
    ph = rnd.uniform(0, 6)
    wob = 1 + 0.18 * np.sin(A * 3 + ph) + 0.1 * np.sin(A * 7 + ph * 2) + 0.05 * np.sin(A * 13)
    Hh = h * (1 - Rr) ** 1.25 * (1 + 0.08 * np.sin(A * 5 + Rr * 6))
    P = np.stack([x + r * Rr * wob * np.cos(A), Hh, z + r * Rr * wob * np.sin(A)], -1)
    P = np.transpose(P, (1, 0, 2))
    UV = np.stack([np.full(P.shape[:2], snow, F4), (P[..., 1] / max(h, 1)).astype(F4)], -1)
    grid(g, P, rgb(70, 95, 65), mat(M_TERRAIN, 0.9), UV=UV, wrap=True, orient=(0, 1, 0))


def build_world(track):
    """Construye todo el escenario estatico HD. Devuelve mallas, publico, luces e informacion del circuito."""
    rnd = random.Random(42)
    g = Geo()            # escenario (siempre)
    gc = Geo()           # concierto (solo de noche)
    crowd = Crowd()
    lights = []          # (x, y, z, radio, r, g, b, intensidad)
    info = {}
    pts, nrm, tan, n = track.pts, track.nrm, track.tan, track.n
    cxm, czm = track.center
    # suelo
    G = 6500
    xs = np.linspace(cxm - G, cxm + G, 41)
    zs = np.linspace(czm - G, czm + G, 41)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    P = np.stack([X, np.zeros_like(X), Z], -1)
    grid(g, P, rgb(70, 120, 45), mat(M_GRASS, 0.95), orient=(0, 1, 0))
    build_track_hd(g, track, info, lights)
    # estadio
    sc = track.ctrl[STADIUM_PTS].mean(axis=0)
    st_pts = track.ctrl[STADIUM_PTS]
    r0 = float(np.max(np.linalg.norm(st_pts - sc, axis=1))) + LIMIT + 6
    ang = np.linspace(0, TAU, 65)
    for j in range(64):
        quad(g, (sc[0], 0.03, sc[1]), (sc[0], 0.03, sc[1]), (sc[0] + r0 * math.cos(ang[j + 1]), 0.03,
                                                            sc[1] + r0 * math.sin(ang[j + 1])),
             (sc[0] + r0 * math.cos(ang[j]), 0.03, sc[1] + r0 * math.sin(ang[j])), rgb(150, 146, 138),
             mat(M_CONC, 0.75))
    rmax, Hb = bowl_hd(g, crowd, rnd, track, sc, r0, lights)
    info["stadium"] = (sc, rmax, r0)
    st0 = track.ctrl[0]
    # tribunas principales y pits
    grandstand_hd(g, crowd, rnd, (st0[0] + 60, st0[1] - (LIMIT + 4)), (1, 0), (0, -1), 760, 24, 0.9, 0.5,
                  rgb(30, 80, 190), 14, lights)
    grandstand_hd(g, crowd, rnd, (st0[0] + 840, st0[1] - (LIMIT + 4)), (1, 0), (0, -1), 160, 16, 0.9, 0.5,
                  rgb(220, 120, 30), 10, lights)
    pit_building_hd(g, st0[0] + 120, st0[1] + LIMIT + 8, lights)
    keep = [(sc[0], sc[1], rmax + 25)]
    # tribunas extra en las rectas
    placed = []
    for i in range(0, n, 6):
        if len(placed) >= 4:
            break
        if np.max(np.abs(track.curv[[(i + k) % n for k in range(-14, 15)]])) > 1 / 900:
            continue
        if any(min(abs(i - j), n - abs(i - j)) < 70 for j in placed):
            continue
        for side in (1, -1):
            o = pts[i] + nrm[i] * side * (LIMIT + 4)
            out = nrm[i] * side
            q = np.array([o + tan[i] * t_ + out * d_ for t_ in (0, 35, 70) for d_ in (0, 10, 20)])
            if track.nearest_dist(q).min() < LIMIT + 3 or min(math.hypot(*(c - sc)) for c in q) < rmax + 12:
                continue
            if abs(o[1] - st0[1]) < 75 and st0[0] - 30 < o[0] < st0[0] + 1030:
                continue
            grandstand_hd(g, crowd, rnd, o, tan[i], out, 70, 16, 0.9, 0.5,
                          rnd.choice([rgb(30, 80, 190), rgb(220, 120, 30), rgb(190, 40, 40), rgb(40, 150, 70)]), 9,
                          lights)
            c = o + tan[i] * 35 + out * 10
            keep.append((c[0], c[1], 50))
            placed.append(i)
            break
    # puentes sobre la pista
    bridges = 0
    for i in list(range(int(n * 0.12), int(n * 0.2), 4)) + list(range(int(n * 0.3), n, 6)):
        if bridges >= 2:
            break
        if np.max(np.abs(track.curv[[(i + k) % n for k in range(-8, 9)]])) > 1 / 600:
            continue
        side_pts = np.array([pts[i] + nrm[i] * s * (LIMIT + 2.6) for s in (-1, 1)])
        if track.nearest_dist(side_pts).min() < LIMIT + 2:
            continue
        if bridges == 1 and abs(i - bridges_i) < n * 0.15:
            continue
        bridge_hd(g, track, i, [0, 12][bridges], lights)
        bridges_i = i
        bridges += 1

    def spot(cands, clear):
        cands = np.array(cands, float)
        d = track.nearest_dist(cands)
        for c, dd in zip(cands, d):
            if dd >= clear and math.hypot(c[0] - sc[0], c[1] - sc[1]) > rmax + 30:
                return c
        return cands[int(np.argmax(d))]
    an = spot([(st0[0] + 400, st0[1] + 95), (st0[0] + 520, st0[1] + 100), (st0[0] + 300, st0[1] + 110)], 45)
    angel_hd(g, an[0], an[1], lights)
    keep.append((an[0], an[1], 32))
    fl = spot([(st0[0] + 25, st0[1] + 70), (st0[0] + 60, st0[1] + 85)], 34)
    flagpole_hd(g, fl[0], fl[1])
    keep.append((fl[0], fl[1], 30))
    info["angel"], info["flag"] = an, fl
    # bardas de la Virgen de Guadalupe
    bb = set()
    for frac in (0.12, 0.3, 0.47, 0.68, 0.86):
        i = int(frac * n)
        for side in (1, -1):
            c = pts[i] + nrm[i] * side * (LIMIT + 7)
            if track.nearest_dist(c[None])[0] < LIMIT + 6 or math.hypot(*(c - sc)) < rmax + 8:
                continue
            guadalupe_wall_hd(g, c, nrm[i] * side, lights, rnd)
            keep.append((c[0], c[1], 12))
            bb.update(range(i - 8, i + 9))
            break
    for frac in (0.02, 0.26, 0.55, 0.78, 0.4):
        i = int(frac * n)
        papel_picado_hd(g, pts[i] + nrm[i] * -(HW + 2.5), pts[i] + nrm[i] * (HW + 2.5))
    # aficionados detras de las mallas
    for i in range(0, n, 7):
        if i in bb:
            continue
        for side in (1, -1):
            if rnd.random() > 0.75:
                continue
            base_ = pts[i] + nrm[i] * side * (LIMIT + 1.6)
            if track.nearest_dist(base_[None])[0] < LIMIT + 1.2:
                continue
            if math.hypot(base_[0] - sc[0], base_[1] - sc[1]) < r0:
                continue
            for _ in range(rnd.randint(4, 9)):
                t_ = rnd.uniform(-8, 8)
                d_ = rnd.uniform(0.0, 1.6)
                x = base_[0] + tan[i, 0] * t_ + nrm[i, 0] * side * d_
                z = base_[1] + tan[i, 1] * t_ + nrm[i, 1] * side * d_
                crowd.add(x, 0.0, z, -nrm[i, 0] * side, -nrm[i, 1] * side, rnd)
    # arboles
    lo = pts.min(axis=0) - 240
    hi = pts.max(axis=0) + 240
    placed_t = 0
    for _ in range(4000):
        x, z = rnd.uniform(lo[0], hi[0]), rnd.uniform(lo[1], hi[1])
        if math.hypot(x - sc[0], z - sc[1]) < rmax + 18:
            continue
        if st0[0] - 20 < x < st0[0] + 1030 and st0[1] - 80 < z < st0[1] + LIMIT + 40:
            continue
        if any(math.hypot(x - kx, z - kz) < kr for kx, kz, kr in keep):
            continue
        if track.nearest_dist(np.array([[x, z]]))[0] < LIMIT + 12:
            continue
        tree_hd(g, x, z, rnd.uniform(0.9, 1.5), rnd)
        placed_t += 1
        if placed_t >= 700:
            break
    city_hd(g, cxm, czm, rnd, lights)
    for k in range(26):
        a = k / 26 * TAU + rnd.uniform(-0.1, 0.1)
        r = rnd.uniform(3500, 4400)
        mountain_hd(g, cxm + r * math.cos(a), czm + r * math.sin(a), rnd.uniform(800, 1400), rnd.uniform(300, 650),
                    rnd)
    mountain_hd(g, cxm + 4600, czm + 1000, 1900, 1350, rnd, snow=1.0, seg=36, rings=14)
    mountain_hd(g, cxm + 4300, czm - 1000, 2300, 1100, rnd, snow=1.0, seg=36, rings=14)
    # camaras de TV para el modo demo
    cams = []
    for i in range(0, n, 45):
        side = 1 if (i // 45) % 2 else -1
        c = pts[i] + nrm[i] * side * (LIMIT + 9)
        if track.nearest_dist(c[None])[0] > LIMIT + 6:
            cams.append((float(c[0]), 6.5, float(c[1])))
    info["tvcams"] = cams
    build_concert_hd(gc, crowd, rnd, track, sc, r0, rmax, Hb, info)
    return g, gc, crowd, lights, info
# ---------------------------------------------------------------- personajes, vehiculos y escenario del concierto
def transform_geo(src, yaw, off, scale=1.0):
    """Gira una geometria alrededor de Y (yaw: +x local -> direccion (cos, sin) en xz) y la traslada."""
    out = Geo()
    c, s = math.cos(yaw), math.sin(yaw)
    Rm = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], F4)
    for p in src.parts:
        q = p.copy()
        q[:, 0:3] = (p[:, 0:3] * scale) @ Rm + np.asarray(off, F4)
        q[:, 3:6] = p[:, 3:6] @ Rm
        out.parts.append(q)
    return out


SKIN_HD = rgb(226, 176, 136)


def character(g, kind, x, y, z, standing=False):
    """Personaje (mirando a +x). y = asiento (sentado) o piso (de pie)."""
    sk = SKIN_HD
    if standing:
        yb = y + 0.95
    else:
        yb = y
    hy = yb + 0.82
    hc = (x, hy, z)
    cloth = {"lucha": rgb(40, 140, 70), "simi": rgb(245, 245, 245), "oxxo": rgb(210, 28, 34),
             "bear": rgb(245, 245, 240), "gas": rgb(40, 80, 160), "helmet": rgb(25, 25, 30)}[kind]
    m_cl = mat(M_DEF, 0.75)
    superellipsoid(g, (x - 0.02, yb + 0.32, z), (0.2, 0.32, 0.27), 0.7, cloth, m_cl, 14, 10)
    if standing:
        pants = rgb(35, 40, 70) if kind not in ("bear", "simi") else cloth
        for sz in (-0.11, 0.11):
            cylinder(g, (x, yb + 0.02, z + sz), (x, y + 0.08, z + sz * 1.1), 0.085, pants, m_cl, seg=10)
            superellipsoid(g, (x + 0.06, y + 0.05, z + sz * 1.1), (0.13, 0.05, 0.07), 0.6, rgb(25, 25, 28),
                           mat(M_DEF, 0.5), 10, 6)
        for sz in (-1, 1):
            cylinder(g, (x, yb + 0.55, z + sz * 0.29), (x + 0.05, yb + 0.05, z + sz * 0.34), 0.065, cloth, m_cl, seg=8)
            sphere(g, (x + 0.05, yb + 0.02, z + sz * 0.34), 0.07, sk if kind != "bear" else cloth, m_cl, 8, 6)
    else:
        for sz in (-1, 1):
            cylinder(g, (x, yb + 0.52, z + sz * 0.29), (x + 0.42, yb + 0.36, z + sz * 0.22), 0.065, cloth, m_cl,
                     seg=8)
            sphere(g, (x + 0.44, yb + 0.36, z + sz * 0.22), 0.07, sk if kind != "bear" else cloth, m_cl, 8, 6)
        revolve(g, [(0.2, -0.02), (0.21, 0.0), (0.2, 0.02)], (x + 0.5, yb + 0.4, z), (1, -0.5, 0),
                rgb(25, 25, 28), mat(M_DEF, 0.5), seg=14)
    cylinder(g, (x, yb + 0.6, z), (x, hy - 0.15, z), 0.08, sk if kind != "bear" else cloth, m_cl, seg=8)
    skin_m = mat(M_DEF, 0.6)

    def dirs(P):
        return norm(P - np.asarray(hc, F4))
    if kind == "lucha":
        mask, trim = rgb(225, 40, 140), rgb(250, 210, 30)

        def cf(P):
            d = dirs(P)
            C = np.empty(P.shape, F4)
            C[:] = mask
            eye = (d[..., 0] > 0.45) & (d[..., 1] > -0.05) & (d[..., 1] < 0.5) & (np.abs(d[..., 2]) > 0.1) & \
                (np.abs(d[..., 2]) < 0.62)
            C[eye] = (1, 1, 1)
            C[(d[..., 0] > 0.5) & (d[..., 1] < -0.3)] = sk
            C[(np.abs(d[..., 2]) < 0.2) & (d[..., 1] > -0.1) & (d[..., 0] < 0.9) & ~eye] = trim
            return C
        sphere(g, hc, 0.28, mask, skin_m, 18, 12, colfn=cf)
        superellipsoid(g, (x + 0.2, yb + 0.25, z), (0.03, 0.25, 0.22), 0.6, trim, m_cl, 8, 6)
    elif kind == "simi":
        def cf(P):
            d = dirs(P)
            C = np.empty(P.shape, F4)
            C[:] = sk
            C[((d[..., 1] > 0.3) & (d[..., 0] < 0.5)) | (d[..., 0] < -0.35)] = (0.97, 0.97, 0.97)
            return C
        sphere(g, hc, 0.29, sk, skin_m, 18, 12, colfn=cf)
        superellipsoid(g, (x + 0.25, hy - 0.11, z), (0.06, 0.05, 0.19), 0.7, (0.98, 0.98, 0.98), m_cl, 10, 6)
        for sz in (-0.11, 0.11):
            revolve(g, [(0.07, -0.012), (0.085, 0.0), (0.07, 0.012)], (x + 0.27, hy + 0.04, z + sz), (1, 0, 0),
                    rgb(20, 20, 22), mat(M_METAL, 0.3, 0.6), seg=12)
        sphere(g, (x + 0.29, hy - 0.02, z), 0.05, rgb(235, 150, 120), skin_m, 8, 6)
    elif kind == "oxxo":
        red = rgb(210, 28, 34)

        def cf(P):
            d = dirs(P)
            C = np.empty(P.shape, F4)
            C[:] = sk
            C[(d[..., 0] < -0.1) & (d[..., 1] > -0.35)] = rgb(30, 22, 15)
            C[d[..., 1] > 0.42] = red
            return C
        sphere(g, hc, 0.28, sk, skin_m, 18, 12, colfn=cf)
        superellipsoid(g, (x + 0.24, hy + 0.2, z), (0.17, 0.02, 0.2), 0.5, red, m_cl, 10, 6)
        superellipsoid(g, (x + 0.2, yb + 0.25, z), (0.025, 0.04, 0.27), 0.6, rgb(250, 200, 20), m_cl, 8, 6)
    elif kind == "bear":
        fur = rgb(245, 245, 240)
        sphere(g, hc, 0.3, fur, mat(M_DEF, 0.9), 18, 12)
        for sz in (-0.22, 0.22):
            sphere(g, (x - 0.02, hy + 0.24, z + sz), 0.1, fur, mat(M_DEF, 0.9), 10, 6)
        sphere(g, (x + 0.24, hy - 0.07, z), 0.12, rgb(250, 232, 210), mat(M_DEF, 0.9), 10, 6)
        sphere(g, (x + 0.35, hy - 0.04, z), 0.045, rgb(20, 20, 20), mat(M_DEF, 0.3), 8, 5)
        revolve(g, [(0.19, 0.0), (0.2, 0.25), (0.27, 0.32), (0.25, 0.45), (0.0, 0.5)], (x, hy + 0.18, z),
                (0, 1, 0), (1, 1, 1), mat(M_DEF, 0.8), seg=14)
    elif kind == "gas":
        blue = rgb(40, 80, 160)

        def cf(P):
            d = dirs(P)
            C = np.empty(P.shape, F4)
            C[:] = sk
            C[d[..., 1] > 0.4] = blue
            return C
        sphere(g, hc, 0.28, sk, skin_m, 18, 12, colfn=cf)
        superellipsoid(g, (x + 0.24, hy + 0.19, z), (0.16, 0.02, 0.2), 0.5, blue, m_cl, 10, 6)
        superellipsoid(g, (x + 0.25, hy - 0.11, z), (0.05, 0.035, 0.16), 0.7, rgb(45, 28, 15), m_cl, 10, 6)
    else:
        yel = rgb(250, 205, 25)

        def cf(P):
            d = dirs(P)
            C = np.empty(P.shape, F4)
            C[:] = yel
            C[(np.abs(d[..., 2]) < 0.14) & (d[..., 0] < 0.4)] = (1, 1, 1)
            return C
        sphere(g, hc, 0.3, yel, mat(M_PAINT, 0.2), 18, 12, colfn=cf)
        superellipsoid(g, (x + 0.2, hy + 0.02, z), (0.12, 0.12, 0.22), 0.6, rgb(15, 20, 30), mat(M_GLASS, 0.05),
                       12, 8)
    if kind in ("lucha", "oxxo", "gas", "simi"):
        for sz in (-0.1, 0.1):
            sphere(g, (x + 0.26, hy + 0.05, z + sz), 0.035, rgb(15, 15, 15), mat(M_DEF, 0.2), 8, 5)


def build_vehicle_hd(p):
    kind = p["vehicle"]
    col, acc = rgb(*p["body"]), rgb(*p["accent"])
    body, glass, wheel = Geo(), Geo(), Geo()
    paint = mat(M_PAINT, 0.26)
    trim = mat(M_DEF, 0.55)
    dark = rgb(18, 18, 20)
    chrome = mat(M_METAL, 0.12, 1.0)
    uvf = lambda P: P[..., [0, 2]]
    info = {}

    def lights_(xf, xr, yf, yr, zin, zout, h=0.08):
        for sz in (-1, 1):
            zc = sz * (zin + zout) / 2
            superellipsoid(body, (xf, yf, zc), (0.07, h, (zout - zin) / 2), 0.4, (1.0, 0.97, 0.9), mat(M_HEAD, 0.2),
                           10, 6)
            superellipsoid(body, (xr, yr, zc), (0.06, h, (zout - zin) / 2), 0.4, (1.0, 0.08, 0.05),
                           mat(M_TAIL, 0.2), 10, 6)
        info["head"] = [(xf + 0.1, yf, sz * (zin + zout) / 2) for sz in (-1, 1)]

    def decal(xc, zh, width, y0, y1, k):
        u0, u1, v0, v1 = bb_uv(k)
        for sz in (-1, 1):
            z_ = sz * zh
            a, b = (xc + width / 2, xc - width / 2) if sz < 0 else (xc - width / 2, xc + width / 2)
            quad(body, (a, y0, z_), (b, y0, z_), (b, y1, z_), (a, y1, z_), (1, 1, 1), mat(M_SIGN, 0.2),
                 ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))

    if kind == "sport":
        def dfb(P):
            P = P.copy()
            x, y = P[..., 0], P[..., 1]
            top = y > 0
            P[..., 1] = np.where(top, y * (1 - 0.45 * np.clip((x + 0.4) / 2.7, 0, 1)), y * 0.85)
            P[..., 2] *= 1 - 0.06 * np.clip(y / 0.33, 0, 1)
            return P
        superellipsoid(body, (0, 0.6, 0), (2.3, 0.33, 0.98), 0.3, col, paint, 28, 14, deform=dfb, uvfn=uvf)

        def dfc(P):
            P = P.copy()
            P[..., 0] -= (P[..., 1] + 0.3) * 0.9
            P[..., 2] *= 1 - 0.22 * np.clip((P[..., 1] + 0.3) / 0.6, 0, 1)
            return P
        superellipsoid(glass, (-0.2, 0.98, 0), (1.05, 0.3, 0.76), 0.45, (0.04, 0.05, 0.07), mat(M_GLASS, 0.04),
                       28, 12, deform=dfc)
        superellipsoid(body, (-0.72, 1.26, 0), (0.6, 0.045, 0.58), 0.3, col, paint, 20, 6, uvfn=uvf)
        superellipsoid(body, (-2.18, 1.18, 0), (0.26, 0.035, 1.0), 0.35, acc, paint, 16, 6, uvfn=uvf)
        for sz in (-0.6, 0.6):
            box(body, -2.25, 0.75, sz - 0.04, -2.08, 1.16, sz + 0.04, dark, trim)
        superellipsoid(body, (0.0, 0.32, 0), (2.25, 0.06, 1.0), 0.4, dark, trim, 20, 6)
        lights_(2.24, -2.28, 0.62, 0.68, 0.45, 0.85)
        decal(-0.5, 0.995, 2.0, 0.42, 0.72, p["sign"])
        info["wheels"] = [(1.45, 0.86), (-1.45, 0.86)]
        r, w = 0.37, 0.3
        info["exhaust"] = [(-2.32, 0.42, -0.42), (-2.32, 0.42, 0.42)]
        for e in info["exhaust"]:
            cylinder(body, (e[0] + 0.25, e[1], e[2]), (e[0], e[1], e[2]), 0.07, rgb(200, 200, 205), chrome, seg=10)
        character(body, p["driver"], -0.42, 0.42, 0.0)
    elif kind == "hatch":
        def dfb(P):
            P = P.copy()
            x, y = P[..., 0], P[..., 1]
            P[..., 1] = np.where(y > 0, y * (1 - 0.35 * np.clip((x - 0.6) / 1.4, 0, 1)), y * 0.85)
            return P
        superellipsoid(body, (0, 0.72, 0), (1.95, 0.42, 0.9), 0.26, col, paint, 26, 13, deform=dfb, uvfn=uvf)

        def dfc(P):
            P = P.copy()
            yy = np.clip((P[..., 1] + 0.42) / 0.84, 0, 1)
            P[..., 0] = np.where(P[..., 0] > 0, P[..., 0] * (1 - 0.45 * yy), P[..., 0] * (1 - 0.1 * yy))
            P[..., 2] *= 1 - 0.12 * yy
            return P
        superellipsoid(glass, (-0.3, 1.36, 0), (1.3, 0.42, 0.84), 0.35, (0.04, 0.05, 0.07), mat(M_GLASS, 0.04),
                       28, 12, deform=dfc)
        superellipsoid(body, (-0.45, 1.79, 0), (0.95, 0.05, 0.76), 0.3, col, paint, 20, 6, uvfn=uvf)
        superellipsoid(body, (-0.5, 2.0, 0), (0.5, 0.16, 0.42), 0.25, acc, paint, 14, 8)
        for sz in (-1, 1):
            u0, u1, v0, v1 = bb_uv(p["sign"])
            z_ = sz * 0.425
            a, b = (0.0, -1.0) if sz < 0 else (-1.0, 0.0)
            quad(body, (a, 1.88, z_), (b, 1.88, z_), (b, 2.12, z_), (a, 2.12, z_), (1, 1, 1), mat(M_SIGN, 0.3),
                 ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))
        superellipsoid(body, (0.0, 0.42, 0), (1.92, 0.07, 0.93), 0.4, acc, paint, 20, 6)
        lights_(1.94, -1.96, 0.78, 0.95, 0.4, 0.8)
        decal(-0.4, 0.905, 1.7, 0.6, 0.92, p["sign"])
        info["wheels"] = [(1.25, 0.78), (-1.25, 0.78)]
        r, w = 0.35, 0.26
        info["exhaust"] = [(-2.0, 0.35, -0.5)]
        character(body, p["driver"], -0.4, 0.62, 0.0)
    elif kind == "pickup":
        superellipsoid(body, (0, 0.8, 0), (2.45, 0.38, 0.98), 0.22, col, paint, 28, 12, uvfn=uvf)

        def dfc(P):
            P = P.copy()
            yy = np.clip((P[..., 1] + 0.45) / 0.9, 0, 1)
            P[..., 0] = np.where(P[..., 0] > 0, P[..., 0] * (1 - 0.4 * yy), P[..., 0])
            P[..., 2] *= 1 - 0.08 * yy
            return P
        superellipsoid(glass, (0.42, 1.55, 0), (0.85, 0.45, 0.92), 0.3, (0.04, 0.05, 0.07), mat(M_GLASS, 0.04),
                       24, 12, deform=dfc)
        superellipsoid(body, (0.3, 2.02, 0), (0.62, 0.05, 0.86), 0.3, col, paint, 16, 6, uvfn=uvf)
        for sz in (-0.93, 0.93):
            box(body, -2.42, 1.1, sz - 0.05, -0.4, 1.5, sz + 0.05, col, paint)
        box(body, -2.45, 1.1, -0.98, -2.36, 1.5, 0.98, col, paint)
        box(body, -0.45, 1.1, -0.98, -0.36, 1.55, 0.98, col, paint)
        superellipsoid(body, (0, 0.62, 0), (2.47, 0.06, 1.0), 0.5, acc, paint, 20, 6)
        box(body, -2.2, 1.18, -0.8, -1.6, 1.68, -0.15, rgb(250, 210, 30), mat(M_DEF, 0.8))
        box(body, -1.5, 1.18, -0.75, -0.9, 1.6, -0.1, rgb(245, 245, 245), mat(M_DEF, 0.8))
        box(body, -2.2, 1.18, 0.1, -1.55, 1.52, 0.8, rgb(200, 120, 50), mat(M_DEF, 0.85))
        sphere(body, (-1.05, 1.48, 0.45), 0.3, rgb(40, 130, 50), mat(M_DEF, 0.4), 14, 10)
        lights_(2.44, -2.46, 0.82, 0.95, 0.45, 0.85)
        decal(0.45, 0.99, 1.3, 0.62, 0.98, p["sign"])
        info["wheels"] = [(1.6, 0.86), (-1.6, 0.86)]
        r, w = 0.43, 0.32
        info["exhaust"] = [(-2.5, 0.45, 0.6)]
        character(body, p["driver"], 0.4, 0.82, 0.0)
    elif kind == "van":
        superellipsoid(body, (-0.8, 1.38, 0), (1.62, 0.96, 1.0), 0.18, col, paint, 30, 16, uvfn=uvf)

        def dfh(P):
            P = P.copy()
            P[..., 1] = np.where(P[..., 1] > 0, P[..., 1] * (1 - 0.3 * np.clip(P[..., 0] / 0.8, 0, 1)), P[..., 1])
            return P
        superellipsoid(body, (1.55, 0.82, 0), (0.85, 0.4, 0.97), 0.25, col, paint, 20, 12, deform=dfh, uvfn=uvf)

        def dfc(P):
            P = P.copy()
            yy = np.clip((P[..., 1] + 0.55) / 1.1, 0, 1)
            P[..., 0] = np.where(P[..., 0] > 0, P[..., 0] * (1 - 0.55 * yy), P[..., 0])
            return P
        superellipsoid(glass, (1.1, 1.65, 0), (0.6, 0.58, 0.95), 0.28, (0.04, 0.05, 0.07), mat(M_GLASS, 0.04),
                       20, 12, deform=dfc)
        superellipsoid(body, (0.9, 2.26, 0), (0.45, 0.06, 0.92), 0.3, col, paint, 14, 6, uvfn=uvf)
        superellipsoid(body, (-0.8, 1.05, 0), (1.64, 0.14, 1.02), 0.3, acc, paint, 24, 6)
        box(body, 0.9, 2.32, -0.6, 1.3, 2.47, 0.0, (1.0, 0.1, 0.08), mat(M_EMIS, 0.3, 0, 4.0))
        box(body, 0.9, 2.32, 0.0, 1.3, 2.47, 0.6, (0.1, 0.3, 1.0), mat(M_EMIS, 0.3, 0, 4.0))
        lights_(2.38, -2.42, 0.85, 0.8, 0.5, 0.85)
        decal(-0.85, 1.015, 2.6, 1.35, 2.2, p["sign"])
        info["wheels"] = [(1.55, 0.86), (-1.55, 0.86)]
        r, w = 0.43, 0.3
        info["exhaust"] = [(-2.45, 0.42, 0.6)]
        character(body, p["driver"], 1.1, 0.85, 0.0)
    else:
        box(body, -3.0, 0.48, -0.95, 3.0, 0.85, 0.95, dark, trim)
        superellipsoid(body, (2.3, 1.2, 0), (0.72, 0.42, 1.12), 0.22, col, paint, 22, 12, uvfn=uvf)

        def dfc(P):
            P = P.copy()
            yy = np.clip((P[..., 1] + 0.55) / 1.1, 0, 1)
            P[..., 0] = np.where(P[..., 0] > 0, P[..., 0] * (1 - 0.18 * yy), P[..., 0])
            return P
        superellipsoid(glass, (2.3, 2.1, 0), (0.66, 0.55, 1.06), 0.28, (0.04, 0.05, 0.07), mat(M_GLASS, 0.04),
                       20, 12, deform=dfc)
        superellipsoid(body, (2.25, 2.68, 0), (0.7, 0.07, 1.1), 0.3, acc, paint, 16, 6, uvfn=uvf)
        cylinder(body, (1.5, 1.0, 0.95), (1.5, 3.2, 0.95), 0.09, rgb(200, 200, 205), chrome, seg=10)
        if kind == "truck":
            superellipsoid(body, (-0.75, 2.1, 0), (2.25, 1.25, 1.18), 0.12, col, paint, 30, 14, uvfn=uvf)
            superellipsoid(body, (-0.75, 0.98, 0), (2.27, 0.14, 1.2), 0.3, acc, paint, 24, 6)
            decal(-0.75, 1.185, 4.0, 1.35, 3.05, p["sign"])
        else:
            prof = [(0.0, -2.2), (0.6, -2.2), (0.92, -2.05), (0.98, -1.8), (0.98, 1.8), (0.92, 2.05), (0.6, 2.2),
                    (0.0, 2.2)]
            revolve(body, prof, (-0.85, 1.9, 0), (1, 0, 0), col, mat(M_PAINT, 0.18, 0.4), seg=28)
            for x0 in (-2.3, -0.85, 0.6):
                revolve(body, [(0.97, -0.1), (1.0, 0.0), (0.97, 0.1)], (x0, 1.9, 0), (1, 0, 0), acc, paint, seg=28)
            for x0 in (-2.6, 0.9):
                box(body, x0, 0.85, -0.6, x0 + 0.35, 1.05, 0.6, rgb(60, 60, 66), mat(M_METAL, 0.4, 0.8))
            decal(-1.3, 1.0, 1.8, 1.55, 2.25, p["sign"])
        lights_(3.02, -3.02, 1.05, 0.95, 0.6, 0.95, 0.1)
        info["wheels"] = [(2.2, 0.92), (-1.4, 0.92), (-2.3, 0.92)]
        r, w = 0.5, 0.36
        info["exhaust"] = [(1.5, 3.2, 0.95)]
        character(body, p["driver"], 2.3, 1.25, 0.0)
    # rueda: llanta con flanco y rin con rayos (eje = z local, cara exterior hacia +z)
    rim_col = {"sport": rgb(20, 20, 22), "hatch": rgb(200, 200, 205), "pickup": rgb(170, 170, 175),
               "van": rgb(190, 190, 195), "truck": rgb(200, 200, 205), "gas": rgb(200, 200, 205)}[kind]
    tire_prof = [(0.62 * r, -w / 2), (0.9 * r, -w / 2 - 0.01), (0.985 * r, -w / 2 + 0.035), (r, 0.0),
                 (0.985 * r, w / 2 - 0.035), (0.9 * r, w / 2 + 0.01), (0.62 * r, w / 2)]
    revolve(wheel, tire_prof, (0, 0, 0), (0, 0, 1), rgb(22, 22, 24), mat(M_TIRE, 0.85), seg=28)
    revolve(wheel, [(0.0, w / 2 - 0.06), (0.14 * r, w / 2 - 0.03), (0.6 * r, w / 2 - 0.05), (0.64 * r, w / 2 + 0.0)],
            (0, 0, 0), (0, 0, 1), rgb(70, 70, 75), mat(M_METAL, 0.35, 1.0), seg=24)
    for k in range(6):
        a = k / 6 * TAU
        d = np.array([math.cos(a), math.sin(a), 0], F4)
        pp = np.array([-math.sin(a), math.cos(a), 0], F4)
        obox(wheel, d * 0.1 * r - pp * 0.035 + np.array([0, 0, w / 2 - 0.045], F4), d * 0.52 * r, pp * 0.07,
             np.array([0, 0, 0.04], F4), rim_col, mat(M_METAL, 0.15, 1.0))
    sphere(wheel, (0, 0, w / 2 - 0.02), 0.06 * r / 0.37, rim_col, mat(M_METAL, 0.15, 1.0), 10, 6)
    info["radius"], info["width"] = r, w
    return body, glass, wheel, info


def build_person_meshes():
    """Malla de aficionado para instancing: (pos, normal, parte, ranura de color). Mirando a +z."""
    def collect(g_parts):
        arrs = []
        for geo, part, slot in g_parts:
            a = geo.array()
            out = np.zeros((len(a), 8), F4)
            out[:, 0:3] = a[:, 0:3]
            out[:, 3:6] = a[:, 3:6]
            out[:, 6] = part
            out[:, 7] = slot
            arrs.append(out)
        return np.concatenate(arrs)
    # detalle cercano
    legs, torso, head, hair, arm_l, arm_r, phone = (Geo() for _ in range(7))
    for sx in (-0.1, 0.1):
        superellipsoid(legs, (sx, 0.45, 0), (0.09, 0.45, 0.1), 0.6, (1, 1, 1), mat(0), 5, 3)
    superellipsoid(torso, (0, 1.15, 0), (0.23, 0.32, 0.14), 0.55, (1, 1, 1), mat(0), 6, 4)
    sphere(head, (0, 1.66, 0.01), 0.13, (1, 1, 1), mat(0), 6, 4)
    superellipsoid(hair, (0, 1.73, -0.02), (0.135, 0.08, 0.13), 0.8, (1, 1, 1), mat(0), 5, 3)
    superellipsoid(arm_l, (-0.29, 1.12, 0), (0.06, 0.3, 0.06), 0.7, (1, 1, 1), mat(0), 4, 3)
    superellipsoid(arm_r, (0.29, 1.12, 0), (0.06, 0.3, 0.06), 0.7, (1, 1, 1), mat(0), 4, 3)
    box(phone, 0.25, 0.78, -0.02, 0.33, 0.92, 0.02, (1, 1, 1), mat(0))
    near = collect([(legs, 0, 1), (torso, 0, 0), (head, 3, 2), (hair, 3, 3), (arm_l, 1, 0), (arm_r, 2, 0),
                    (phone, 2, 4)])
    # lejos: cajas simples
    legs, torso, head, arm_l, arm_r = (Geo() for _ in range(5))
    box(legs, -0.18, 0, -0.1, 0.18, 0.9, 0.1, (1, 1, 1), mat(0))
    box(torso, -0.25, 0.88, -0.13, 0.25, 1.47, 0.13, (1, 1, 1), mat(0))
    box(head, -0.12, 1.5, -0.12, 0.12, 1.8, 0.12, (1, 1, 1), mat(0))
    far = collect([(legs, 0, 1), (torso, 0, 0), (head, 3, 2)])
    return near, far


def build_concert_hd(g, crowd, rnd, track, sc, r0, rmax, Hb, info):
    """Escenario de Avenged Sevenfold dentro del estadio (solo de noche)."""
    xs, zs = np.meshgrid(np.arange(-r0, r0, 3.0), np.arange(-r0, r0, 3.0))
    P = np.column_stack([xs.ravel() + sc[0], zs.ravel() + sc[1]])
    P = P[np.hypot(P[:, 0] - sc[0], P[:, 1] - sc[1]) < r0 - 5]
    best = P[int(np.argmax(track.nearest_dist(P)))]
    dvec = (best - sc) / np.linalg.norm(best - sc)
    S = sc + dvec * (r0 - 16)
    ff = -dvec
    rr = np.array([-ff[1], ff[0]])
    info["stage"] = (S, ff, rr)

    def L(a, b, y=0.0):
        return (S[0] + rr[0] * a + ff[0] * b, y, S[1] + rr[1] * a + ff[1] * b)

    def lbox(a0, a1, b0, b1, y0, y1, col, m_):
        obox(g, L(a0, b0, y0), (rr[0] * (a1 - a0), 0, rr[1] * (a1 - a0)), (0, y1 - y0, 0),
             (ff[0] * (b1 - b0), 0, ff[1] * (b1 - b0)), col, m_)

    def lquad(a0, a1, b, y0, y1, col, m_, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
        quad(g, L(a0, b, y0), L(a1, b, y0), L(a1, b, y1), L(a0, b, y1), col, m_, uv)
    blk, steel = rgb(18, 18, 22), rgb(55, 58, 66)
    sm = mat(M_METAL, 0.45, 0.85)
    lbox(-18, 18, -7, 7, 0, 2.4, blk, mat(M_DEF, 0.6))
    lbox(-4, 4, -6, -1.5, 2.4, 3.6, rgb(35, 35, 42), mat(M_DEF, 0.5))
    lbox(-2.5, 2.5, 7, 22, 0, 1.6, blk, mat(M_DEF, 0.6))                      # pasarela al publico
    lbox(-17, 17, -6.6, -5.8, 2.4, 18.0, blk, mat(M_DEF, 0.6))
    lquad(-16, 16, -5.7, 3.0, 17.4, (1, 1, 1), mat(M_LED, 0.3, 0, 2.2))
    for sgn in (-1, 1):
        for a in (23.0, 29.5):
            lbox(sgn * a - 3.1, sgn * a + 3.1, 3.3, 4.0, 0, 18.5, blk, mat(M_DEF, 0.6))
            lquad(sgn * a - 2.8, sgn * a + 2.8, 4.05, 4.5, 18.0, (1, 1, 1), mat(M_LED, 0.3, 0, 3.0),
                  ((0, 2), (1, 2), (1, 3), (0, 3)))
        for k in range(9):                                                   # line arrays colgantes
            y = 21 - k * 1.05
            lbox(sgn * 19 - 1.2, sgn * 19 + 1.2, -0.6 + k * 0.08, 0.7 + k * 0.08, y - 1.0, y, blk, mat(M_DEF, 0.5))
        lbox(sgn * 36 - 0.7, sgn * 36 + 0.7, 1.5, 2.9, 0, 16, steel, sm)
        lbox(sgn * 36 - 1.1, sgn * 36 + 1.1, 1.1, 3.3, 16, 16.7, blk, mat(M_DEF, 0.5))
        for bb in (-7.5, 6.5):
            lbox(sgn * 21 - 0.4, sgn * 21 + 0.4, bb - 0.4, bb + 0.4, 0, 24, steel, sm)
    lbox(-22, 22, -8.5, 7.5, 24, 24.8, rgb(220, 222, 230), mat(M_METAL, 0.35, 0.3))
    lbox(-19, 19, -7.5, 5.5, 24.8, 26.8, rgb(200, 202, 212), mat(M_METAL, 0.35, 0.3))
    for bb in (-6.5, -1.5, 3.5):
        lbox(-21, 21, bb - 0.25, bb + 0.25, 23.2, 23.7, steel, sm)
    for yy in (17.9, 18.8):
        for i in range(14):
            a = -13 + i * 2
            lbox(a - 0.6, a + 0.6, -5.5, -5.1, yy, yy + 0.55, (1.0, 0.1, 0.08), mat(M_EMIS, 0.3, 0, 9.0))
    bx, _, bz = L(0, -5.4)
    u0, u1, v0, v1 = bb_uv(10)
    lquad(-13, 13, -5.4, 19.6, 23.0, (1, 1, 1), mat(M_SIGN, 0.3, 0, 0), ((u1, v1), (u0, v1), (u0, v0), (u1, v0)))
    movers = []
    for i in range(16):
        a = -19 + i * 38 / 15
        lbox(a - 0.45, a + 0.45, -0.45, 0.45, 22.6, 23.4, (1.0, 1.0, 0.95), mat(M_EMIS, 0.3, 0, 10.0))
        movers.append(L(a, 0, 22.8))
    info["movers"] = movers
    info["flames"] = [L(a, 6.6, 2.5) for a in (-15, -9, 9, 15)]
    info["fireballs"] = [L(-36, 2.2, 17.2), L(36, 2.2, 17.2), L(-14, 5.5, 27.2), L(-5, 5.5, 27.2),
                         L(5, 5.5, 27.2), L(14, 5.5, 27.2), L(-26, 3.7, 19.0), L(26, 3.7, 19.0)]
    info["fountain"] = [L(a, -8.5, 27) for a in (-16, -10, -5, 0, 5, 10, 16)]
    for a in (-15, -9, 9, 15):
        lbox(a - 0.5, a + 0.5, 6.0, 7.0, 2.4, 2.8, rgb(60, 60, 60), sm)
    # la banda
    for i, (a, b, kind) in enumerate([(0, 4.0, "gas"), (-7, 3.0, "helmet"), (7, 3.0, "oxxo"), (11, 1.5, "helmet"),
                                      (0, -3.5, "helmet")]):
        tmp = Geo()
        character(tmp, kind, 0, 0, 0, standing=True)
        x, _, z = L(a, b)
        g.merge(transform_geo(tmp, math.atan2(ff[1], ff[0]), (x, 3.6 if i == 4 else 2.4, z), 1.15))
    for a, b in ((-1.3, -2.2), (1.3, -2.2), (-0.7, -1.4), (0.7, -1.4)):
        x, _, z = L(a, b)
        cylinder(g, (x, 3.6, z), (x, 4.3, z), 0.5, rgb(180, 20, 20), mat(M_PAINT, 0.25), seg=14)
    for sgn in (-1, 1):                                                       # torres delay y FOH
        lbox(sgn * 26 - 0.35, sgn * 26 + 0.35, 32, 32.7, 0, 15, steel, sm)
        lbox(sgn * 26 - 1.3, sgn * 26 + 1.3, 31.4, 33.3, 10, 15, blk, mat(M_DEF, 0.5))
        lbox(sgn * 26 - 0.5, sgn * 26 + 0.5, 31.9, 32.8, 15, 15.6, (1, 1, 0.95), mat(M_EMIS, 0.3, 0, 8.0))
    lbox(-5, 5, 38, 45, 0, 3.2, rgb(225, 225, 230), mat(M_DEF, 0.6))
    lbox(-5.6, 5.6, 37.4, 45.6, 3.2, 3.6, rgb(240, 240, 245), mat(M_DEF, 0.6))
    lbox(-18, 18, 8.4, 8.7, 0, 1.2, steel, sm)
    # publico del concierto + circle pit
    pit_a, pit_b = -14.0, 28.0
    for ca, cb in ((-14, 28), (14, 28), (0, 33), (-11, 42), (11, 42)):
        x, _, z = L(ca, cb)
        if track.nearest_dist(np.array([[x, z]]))[0] > LIMIT + 14 and math.hypot(x - sc[0], z - sc[1]) < r0 - 12:
            pit_a, pit_b = ca, cb
            break
    px_, _, pz_ = L(pit_a, pit_b)
    info["pit"] = (float(px_), float(pz_))

    def in_crowd(a, b):
        return 10 <= b <= 64 and abs(a) < 18 + b * 0.55
    for a in np.arange(-55, 55, 1.2):
        for b in np.arange(10, 64, 1.2):
            if not in_crowd(a, b) or (abs(a) < 6.2 and 36.8 < b < 46.2) or (abs(a) < 3.2 and b < 23):
                continue
            if abs(abs(a) - 26) < 1.6 and abs(b - 32.3) < 1.6:
                continue
            if math.hypot(a - pit_a, b - pit_b) < 10:
                continue
            x, _, z = L(a + rnd.uniform(-0.4, 0.4), b + rnd.uniform(-0.4, 0.4))
            if math.hypot(x - sc[0], z - sc[1]) > r0 - 3:
                continue
            if track.nearest_dist(np.array([[x, z]]))[0] < LIMIT + 4:
                continue
            typ = 1 if rnd.random() < 0.55 else (2 if rnd.random() < 0.6 else 0)
            crowd.add(x, 0.0, z, -ff[0] * 0 + (S[0] - x), (S[1] - z), rnd, typ=typ, concert=True,
                      palette=METAL_SHIRTS if rnd.random() < 0.8 else None)
    for rad in (4.0, 5.5, 7.0, 8.5):
        cnt = int(TAU * rad / 1.25)
        for j in range(cnt):
            an = j / cnt * TAU + rnd.uniform(-0.1, 0.1)
            x, _, z = L(pit_a + rad * math.cos(an), pit_b + rad * math.sin(an))
            tx, tz = -math.sin(an), math.cos(an)
            wx, wz = rr[0] * tx + ff[0] * tz, rr[1] * tx + ff[1] * tz
            crowd.add(x, 0.0, z, wx, wz, rnd, typ=4, concert=True, palette=METAL_SHIRTS)
    # el resto de la cancha: alfombra de cabezas y celulares
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
            quad(g, (gx, 1.6, gz), (gx + cell, 1.6, gz), (gx + cell, 1.6, gz + cell), (gx, 1.6, gz + cell),
                 rgb(30, 28, 32), mat(M_CROWD, 0.85), ((gx, gz), (gx + cell, gz), (gx + cell, gz + cell),
                                                       (gx, gz + cell)))
    launch = [L(a, -6, 26) for a in (-14, 0, 14)]
    for j in range(12):
        a = j / 12 * TAU
        launch.append((sc[0] + (rmax - 3) * math.cos(a), Hb + 13, sc[1] + (rmax - 3) * math.sin(a)))
    info["launch"] = launch


def build_studio():
    """Garage de exhibicion para elegir piloto: piso brillante, ciclorama y tiras de luz."""
    g = Geo()
    ang = np.linspace(0, TAU, 65)
    R = 40.0
    rings = np.array([0, 3.8, 4.0, 4.3, 12, 40], F4)
    for k in range(len(rings) - 1):
        for j in range(64):
            a0, a1 = ang[j], ang[j + 1]
            r_a, r_b = rings[k], rings[k + 1]
            mid = M_STUDIO if k != 1 and k != 2 else M_EMIS
            col = rgb(14, 14, 18) if mid == M_STUDIO else (0.3, 0.55, 1.0)
            em = 0.0 if mid == M_STUDIO else 1.6
            y = 0.0 if k >= 2 else 0.05
            quad(g, (r_a * math.cos(a0), y, r_a * math.sin(a0)), (r_b * math.cos(a0), y, r_b * math.sin(a0)),
                 (r_b * math.cos(a1), y, r_b * math.sin(a1)), (r_a * math.cos(a1), y, r_a * math.sin(a1)), col,
                 mat(mid, 0.25, 0.0, em), ((r_a * math.cos(a0), r_a * math.sin(a0)),
                                           (r_b * math.cos(a0), r_b * math.sin(a0)),
                                           (r_b * math.cos(a1), r_b * math.sin(a1)),
                                           (r_a * math.cos(a1), r_a * math.sin(a1))))
    # ciclorama curvo de fondo
    th = np.linspace(-1.9, 1.9, 40) + math.pi
    ys = np.array([0, 2, 6, 14], F4)
    rad = np.array([22, 22.6, 23.5, 24], F4)
    P = np.zeros((len(ys), len(th), 3), F4)
    P[..., 0] = rad[:, None] * np.cos(th)[None, :]
    P[..., 2] = rad[:, None] * np.sin(th)[None, :]
    P[..., 1] = ys[:, None]
    grid(g, P, rgb(20, 22, 30), mat(M_STUDIO, 0.6), orient=(1, 0, 0))
    for k in range(7):
        a = math.pi + (k - 3) * 0.45
        x, z = 21.8 * math.cos(a), 21.8 * math.sin(a)
        rx, rz = -math.sin(a), math.cos(a)
        quad(g, (x - rx * 0.12, 0.5, z - rz * 0.12), (x + rx * 0.12, 0.5, z + rz * 0.12),
             (x + rx * 0.12, 9.0, z + rz * 0.12), (x - rx * 0.12, 9.0, z - rz * 0.12), (1.0, 0.85, 0.6),
             mat(M_EMIS, 0.4, 0, 2.5))
    for k in range(3):
        z = -6 + k * 6
        box(g, -8, 11.5, z - 0.15, 8, 11.7, z + 0.15, (1.0, 1.0, 1.0), mat(M_EMIS, 0.4, 0, 3.0))
    return g
# ---------------------------------------------------------------- shaders
GLSL_COMMON = """
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    float a = hash(i), b = hash(i + vec2(1, 0)), c = hash(i + vec2(0, 1)), d = hash(i + vec2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) { float v = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { v += a * noise(p); p = p * 2.03 + 17.1; a *= 0.5; } return v; }
"""

VS_SCENE = """
#version 330
uniform mat4 u_vp; uniform mat4 u_model; uniform float u_time; uniform float u_clip;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col; in vec4 in_mat; in vec2 in_uv;
out vec3 v_world; out vec3 v_nrm; out vec3 v_col; out vec4 v_mat; out vec2 v_uv;
invariant gl_Position;
void main() {
    vec3 p = in_pos;
    int m = int(in_mat.x + 0.5);
    if (m == 12) {
        float f = clamp((in_uv.x - 0.75) * 4.0, 0.0, 1.0);
        p.z += (sin(u_time * 2.6 - f * 7.0) * 1.4 + sin(u_time * 4.3 - f * 13.0) * 0.35) * f;
        p.y += sin(u_time * 1.9 - f * 5.0) * 0.25 * f;
    } else if (m == 17) {
        p += in_nrm * sin(u_time * 2.0 + in_mat.w) * 0.22 + vec3(0.0, sin(u_time * 1.7 + in_mat.w) * 0.07, 0.0);
    } else if (m == 23) {
        float h = max(p.y - 2.0, 0.0);
        p.x += sin(u_time * 1.3 + p.z * 0.15) * 0.04 * h;
        p.z += cos(u_time * 1.1 + p.x * 0.13) * 0.035 * h;
    }
    vec4 w = u_model * vec4(p, 1.0);
    v_world = w.xyz;
    v_nrm = mat3(u_model) * in_nrm;
    v_col = in_col;
    v_mat = in_mat;
    v_uv = in_uv;
    gl_ClipDistance[0] = u_clip > 0.5 ? (w.y - 0.09) : 1.0;
    gl_Position = u_vp * w;
}
"""

VS_CROWD = """
#version 330
uniform mat4 u_vp; uniform float u_time; uniform float u_clip; uniform vec2 u_pit; uniform float u_night;
in vec3 in_pos; in vec3 in_nrm; in float in_part; in float in_slot;
in vec3 i_pos; in float i_yaw; in vec3 i_shirt; in vec4 i_misc;
out vec3 v_world; out vec3 v_nrm; out vec3 v_col; out vec4 v_mat; out vec2 v_uv;
float h1(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
vec3 rotz(vec3 q, vec3 piv, float a) {
    vec3 d = q - piv; float c = cos(a), s = sin(a);
    return piv + vec3(d.x * c - d.y * s, d.x * s + d.y * c, d.z);
}
void main() {
    float ph = i_misc.x, typ = i_misc.y, tone = i_misc.z, sc = i_misc.w, t = u_time;
    float jump = 0.0, aL = 0.15, aR = 0.15;
    if (typ < 0.5) { float w = 0.5 + 0.5 * sin(t * 5.0 + ph); aL = 2.1 + 0.8 * w; aR = 2.1 + 0.8 * (1.0 - w);
                     jump = abs(sin(t * 2.5 + ph)) * 0.05; }
    else if (typ < 1.5) { jump = abs(sin(t * 6.0 + ph)) * 0.38; aL = aR = 2.65 + 0.25 * sin(t * 12.0 + ph); }
    else if (typ < 2.5) { aR = 2.8; aL = 0.2 + 0.15 * sin(t + ph); jump = abs(sin(t * 2.0 + ph)) * 0.03; }
    else if (typ < 3.5) { aL = aR = 0.12 + 0.08 * sin(t * 1.3 + ph); }
    else { jump = abs(sin(t * 7.0 + ph)) * 0.3; aL = aR = 1.0 + 0.9 * sin(t * 8.0 + ph); }
    vec3 p = in_pos, n = in_nrm;
    if (in_part > 0.5 && in_part < 1.5) { p = rotz(p, vec3(-0.27, 1.42, 0.0), -aL); n = rotz(n, vec3(0.0), -aL); }
    else if (in_part > 1.5 && in_part < 2.5) { p = rotz(p, vec3(0.27, 1.42, 0.0), aR); n = rotz(n, vec3(0.0), aR); }
    if (in_slot > 3.5 && (abs(typ - 2.0) > 0.5 || u_night < 0.5)) p = vec3(0.0, 1.0, 0.0);
    p *= sc;
    float yaw = i_yaw;
    vec3 base = i_pos;
    if (typ > 3.5) {
        float an = t * 0.55; vec2 d = base.xz - u_pit; float c = cos(an), s = sin(an);
        base.xz = u_pit + vec2(d.x * c - d.y * s, d.x * s + d.y * c); yaw -= an;
    }
    float c = cos(yaw), s = sin(yaw);
    vec3 wp = vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c) + base + vec3(0.0, jump, 0.0);
    v_nrm = vec3(n.x * c + n.z * s, n.y, -n.x * s + n.z * c);
    vec3 skin = mix(vec3(0.93, 0.8, 0.7), vec3(0.55, 0.42, 0.32), tone);
    vec3 pants = vec3(0.25, 0.3, 0.42) * (0.5 + 0.8 * h1(ph * 3.1));
    vec3 hair = vec3(0.2, 0.15, 0.1) * (0.5 + 1.0 * h1(ph * 7.7));
    vec3 col = in_slot < 0.5 ? i_shirt : in_slot < 1.5 ? pants : in_slot < 2.5 ? skin : in_slot < 3.5 ? hair : vec3(0.1);
    float emis = 0.0, mid = 30.0;
    if (in_slot > 3.5 && u_night > 0.5) { col = vec3(0.85, 0.9, 1.0); emis = 3.0 * (0.6 + 0.4 * sin(t * 3.0 + ph * 5.0)); mid = 28.0; }
    v_world = wp; v_col = col; v_mat = vec4(mid, 0.8, 0.0, emis); v_uv = vec2(0.0);
    gl_ClipDistance[0] = u_clip > 0.5 ? (wp.y - 0.09) : 1.0;
    gl_Position = u_vp * vec4(wp, 1.0);
}
"""

FS_MAIN = ("""
#version 330
#define MAXL __MAXL__
in vec3 v_world; in vec3 v_nrm; in vec3 v_col; in vec4 v_mat; in vec2 v_uv;
out vec4 f_color;
uniform vec3 u_cam; uniform float u_time;
uniform vec3 u_sun; uniform vec3 u_sun_col; uniform vec3 u_top; uniform vec3 u_hor; uniform vec3 u_ground;
uniform vec3 u_fog; uniform float u_fog_den;
uniform float u_night; uniform float u_wet; uniform float u_lamps; uniform float u_concert; uniform float u_studio;
uniform float u_shadow_on; uniform mat4 u_lvp_s; uniform mat4 u_lvp_d; uniform float u_ts; uniform float u_td;
uniform sampler2DShadow u_shs; uniform sampler2DShadow u_shd;
uniform sampler2D u_atlas; uniform sampler2D u_refl; uniform vec2 u_res; uniform float u_refl_on; uniform float u_reflpass;
uniform int u_nl; uniform vec4 u_lpos[MAXL]; uniform vec4 u_lcol[MAXL];
uniform vec2 u_stage; uniform float u_hw; uniform float u_startl; uniform float u_go;
uniform vec3 u_paint2; uniform float u_stripe; uniform float u_brake; uniform float u_headl; uniform float u_glass_alpha;
const float PI = 3.14159265;
""" + GLSL_COMMON + """
vec3 lin(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
vec3 env(vec3 d, float rough) {
    float e = d.y;
    vec3 c = mix(u_hor, u_top, pow(clamp(e, 0.0, 1.0), 0.45));
    if (e < 0.0) c = mix(u_hor * 0.6, u_ground * 1.5, clamp(-e * 2.5, 0.0, 1.0));
    float sd = max(dot(d, u_sun), 0.0);
    c += u_sun_col * pow(sd, mix(900.0, 6.0, rough)) * mix(2.0, 0.04, rough);
    c += u_night * vec3(1.0, 0.5, 0.25) * exp(-abs(e) * 10.0) * 0.06;
    if (u_studio > 0.5) {
        c = vec3(0.012, 0.013, 0.02) + vec3(1.0, 0.85, 0.6) * pow(max(d.y, 0.0), 5.0) * 2.5 * (1.0 - rough * 0.7)
          + vec3(0.25, 0.45, 1.0) * 0.35 * exp(-abs(e) * 5.0) * (1.0 - rough * 0.5);
    }
    return c;
}
float D_GGX(float NdH, float a) { float a2 = a * a; float d = NdH * NdH * (a2 - 1.0) + 1.0; return a2 / (PI * d * d + 1e-5); }
float G_S(float NdV, float NdL, float r) { float k = (r + 1.0) * (r + 1.0) / 8.0; return NdV / (NdV * (1.0 - k) + k) * NdL / (NdL * (1.0 - k) + k); }
vec3 F_S(float c, vec3 F0) { return F0 + (1.0 - F0) * pow(1.0 - c, 5.0); }
vec3 brdf(vec3 N, vec3 V, vec3 L, vec3 alb, float rough, float metal, vec3 F0) {
    vec3 H = normalize(L + V);
    float NdL = max(dot(N, L), 0.0), NdV = max(dot(N, V), 1e-3), NdH = max(dot(N, H), 0.0);
    float a = max(rough * rough, 0.002);
    vec3 F = F_S(max(dot(H, V), 0.0), F0);
    vec3 spec = D_GGX(NdH, a) * G_S(NdV, NdL, rough) * F / (4.0 * NdV * NdL + 1e-4);
    return ((1.0 - F) * (1.0 - metal) * alb / PI + spec) * NdL;
}
float shadow1(sampler2DShadow s, mat4 m, vec3 P, float texel) {
    vec4 lp = m * vec4(P, 1.0);
    vec3 c = lp.xyz / lp.w * 0.5 + 0.5;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0) return 1.0;
    float sum = 0.0;
    sum += texture(s, vec3(c.xy + vec2(-0.7, -0.7) * texel, c.z - 0.0006));
    sum += texture(s, vec3(c.xy + vec2(0.7, -0.7) * texel, c.z - 0.0006));
    sum += texture(s, vec3(c.xy + vec2(-0.7, 0.7) * texel, c.z - 0.0006));
    sum += texture(s, vec3(c.xy + vec2(0.7, 0.7) * texel, c.z - 0.0006));
    return sum * 0.25;
}
vec3 led(vec2 q) {
    if (q.y > 1.5) {
        q.y -= 2.0;
        float nz = noise(vec2(q.x * 9.0, q.y * 7.0 - u_time * 6.0));
        float fl = clamp((1.0 - q.y) * (0.5 + 0.8 * nz) + 0.2 * sin(q.x * 18.0 + u_time * 5.0), 0.0, 1.0);
        return mix(vec3(0.05, 0.0, 0.0), mix(vec3(1.0, 0.22, 0.0), vec3(1.0, 0.85, 0.4), fl * fl), fl) * 1.4;
    }
    float mode = mod(floor(u_time / 5.0), 4.0);
    vec3 a = 0.5 + 0.5 * cos(u_time * 1.7 + q.x * 9.0 + vec3(0.0, 2.1, 4.2));
    if (mode < 0.5) return a * (0.25 + 0.75 * step(0.5, fract(q.x * 14.0 - u_time * 2.0)));
    if (mode < 1.5) return vec3(1.0, 0.12, 0.08) * (0.5 + 0.5 * sin(length(q - 0.5) * 34.0 - u_time * 8.0));
    if (mode < 2.5) {
        float nz = noise(vec2(q.x * 6.0, q.y * 5.0 - u_time * 4.0));
        float fl = clamp((1.0 - q.y) * (0.6 + 0.7 * nz), 0.0, 1.0);
        return mix(vec3(0.0), mix(vec3(1.0, 0.25, 0.0), vec3(1.0, 0.9, 0.5), fl), fl);
    }
    float sk = smoothstep(0.03, 0.0, abs(length((q - vec2(0.5, 0.55)) * vec2(1.6, 1.0)) - 0.28))
             + smoothstep(0.04, 0.0, abs(q.x - 0.5)) * step(abs(q.y - 0.55), 0.35);
    return vec3(0.9, 0.95, 1.0) * sk * (0.6 + 0.4 * sin(u_time * 10.0)) + vec3(0.25, 0.0, 0.05) * 0.4;
}
void main() {
    int id = int(v_mat.x + 0.5);
    vec3 P = v_world;
    vec3 alb = lin(v_col);
    float rough = v_mat.y, metal = v_mat.z, emis = v_mat.w;
    vec3 N = normalize(v_nrm);
    vec3 V = normalize(u_cam - P);
    float dist = length(u_cam - P);
    if (dot(N, V) < 0.0 && (id == 20 || id == 12 || id == 17 || id == 7 || id == 23 || id == 18)) N = -N;
    vec3 E = vec3(0.0);
    float alpha = 1.0, reflm = 0.0, clear = 0.0;
    vec2 wp = P.xz;
    vec3 extra = vec3(0.0);
    if (id == 1) {
        float n1 = fbm(wp * 0.25), n2 = noise(wp * 5.0), n3 = noise(wp * 31.0);
        alb = vec3(0.055, 0.055, 0.06) * (0.7 + 0.5 * n1) * (0.8 + 0.35 * n2);
        float lat = v_uv.y;
        float rl = exp(-pow((lat - 2.0 * sin(v_uv.x * 0.0035 + 1.0)) / 2.3, 2.0));
        alb *= 1.0 - 0.3 * rl;
        rough = clamp(mix(0.8, 0.6, rl) - n3 * 0.12, 0.05, 1.0);
        float av = abs(lat);
        if (av > u_hw - 0.75 && av < u_hw - 0.45) { alb = vec3(0.6); rough = 0.55; }
        N = normalize(N + vec3(n2 - 0.5, 0.0, n3 - 0.5) * 0.12);
        float puddle = smoothstep(0.5, 0.68, fbm(wp * 0.06 + 7.3));
        float wet = clamp(u_wet * (0.5 + 0.7 * puddle), 0.0, 1.0);
        alb *= mix(1.0, 0.45, wet); rough = mix(rough, 0.03, wet);
        N = normalize(mix(N, vec3(0.0, 1.0, 0.0), wet * 0.85));
        reflm = wet;
    } else if (id == 2) {
        float n = fbm(wp * 0.05), m = noise(wp * 1.3), k = noise(wp * 9.0);
        alb = mix(vec3(0.03, 0.09, 0.015), vec3(0.09, 0.17, 0.03), n) * (0.8 + 0.3 * m) * (0.9 + 0.2 * k);
        alb *= 0.93 + 0.12 * mod(floor(wp.x / 10.0) + floor(wp.y / 10.0), 2.0);
        rough = 0.92; reflm = u_wet * 0.12;
    } else if (id == 3) {
        alb = vec3(mod(floor(v_uv.x * 1.25) + floor(v_uv.y * 1.25), 2.0) * 0.75 + 0.02);
        rough = 0.5; reflm = u_wet * 0.6;
    } else if (id == 4) {
        alb = mod(floor(v_uv.x / 3.0), 2.0) < 1.0 ? vec3(0.6, 0.03, 0.025) : vec3(0.72);
        alb *= 0.85 + 0.2 * noise(wp * 4.0);
        rough = mix(0.45, 0.1, u_wet); reflm = u_wet * 0.5;
    } else if (id == 5) {
        float k = noise(wp * 14.0), l = noise(wp * 3.0);
        alb = mix(vec3(0.35, 0.3, 0.22), vec3(0.55, 0.48, 0.38), k) * (0.8 + 0.3 * l);
        rough = 0.95;
    } else if (id == 6) {
        alb *= 0.8 + 0.35 * noise(wp * 3.0 + P.y * 2.0);
        rough = 0.85;
    } else if (id == 7) {
        alb = lin(texture(u_atlas, v_uv).rgb);
        E = alb * 0.7 * u_lamps;
    } else if (id == 8) {
        vec2 q = vec2(v_uv.x * 2.2, v_uv.y * 3.0 + P.y * 2.5);
        vec2 cell = floor(q), fc = fract(q);
        float h = hash(cell), h2 = hash(cell + 7.0);
        vec3 body = h < 0.55 ? vec3(0.02 + 0.05 * h2) : h < 0.63 ? vec3(0.35, 0.03, 0.03) : h < 0.69 ? vec3(0.5)
                  : h < 0.77 ? vec3(0.03, 0.06, 0.2) : h < 0.83 ? vec3(0.05, 0.15, 0.05) : alb * 0.6;
        vec3 head = h2 < 0.75 ? vec3(0.45, 0.28, 0.18) : vec3(0.02);
        bool ishead = fc.y > 0.58 && abs(fc.x - 0.5) < 0.24;
        alb = ishead ? head : (fc.y > 0.58 ? vec3(0.01) : body);
        rough = 0.85;
        if (u_night > 0.5) {
            vec2 ph = fc - vec2(0.78, 0.8);
            if (hash(cell + 13.0) > 0.82 && dot(ph, ph) < 0.02) E = vec3(0.85, 0.9, 1.0) * 3.0 * (0.6 + 0.4 * sin(u_time * 5.0 + h * 60.0));
            float sd = length(wp - u_stage);
            vec3 wc = mix(vec3(0.8, 0.1, 0.7), 0.5 + 0.5 * cos(u_time * 0.8 + vec3(0.0, 2.5, 4.0)), 0.3);
            extra = wc * clamp(1.0 - sd / 280.0, 0.0, 1.0) * 0.12 * (0.6 + 0.4 * sin(u_time * 3.0 + P.x * 0.05)) * u_concert;
            if (hash(cell + floor(u_time * 3.0) * 1.37) > 0.9975) E = vec3(6.0);
        }
    } else if (id == 11) {
        E = alb * emis * mix(0.03, 1.0, u_lamps);
    } else if (id == 12) {
        alb = lin(texture(u_atlas, v_uv).rgb);
        N = normalize(cross(dFdx(P), dFdy(P)));
        if (dot(N, V) < 0.0) N = -N;
        rough = 0.8;
    } else if (id == 13) {
        E = led(v_uv) * emis * mix(0.7, 1.6, u_night);
        alb = vec3(0.01); rough = 0.3;
    } else if (id == 14) {
        E = alb * emis * (0.6 + 0.4 * noise(vec2(u_time * 9.0, P.x * 3.0))) * mix(0.3, 1.0, u_lamps);
    } else if (id == 17) {
        E = alb * 0.08 * u_lamps;
    } else if (id == 18) {
        alpha = u_glass_alpha; rough = 0.03;
    } else if (id == 19) {
        if (u_stripe > 0.5 && abs(v_uv.y) < 0.24) alb = lin(u_paint2);
        clear = 1.0;
        rough = clamp(rough + (noise(P.xz * 90.0 + P.y * 80.0) - 0.5) * 0.08, 0.05, 1.0);
    } else if (id == 20) {
        vec2 q = vec2(v_uv.x + v_uv.y, v_uv.x - v_uv.y) * 4.0;
        float l = min(fract(q.x), fract(q.y));
        if (l > 0.15 || dist > 120.0 || hash(gl_FragCoord.xy) < smoothstep(50.0, 120.0, dist)) discard;
    } else if (id == 21 || id == 32) {
        vec2 g = v_uv / (id == 21 ? vec2(3.2, 3.6) : vec2(2.4, 3.2));
        vec2 cell = floor(g), f = fract(g);
        bool win = f.x > 0.15 && f.x < 0.85 && f.y > 0.22 && f.y < 0.82 && abs(N.y) < 0.5;
        if (id == 32) win = abs(N.y) < 0.5 && f.y > 0.08;
        if (win) {
            alb = vec3(0.03, 0.04, 0.05); rough = 0.06; metal = 0.0; reflm = 0.0;
            float on = step(0.42, hash(cell + floor(P.xz / 41.0) * 3.1));
            E = vec3(1.0, 0.75, 0.42) * on * (id == 21 ? 1.6 : 0.9) * u_lamps;
            clear = 1.0;
        }
    } else if (id == 23) {
        alb *= 0.75 + 0.45 * noise(P.xz * 2.2 + P.y * 3.0);
        E += alb * u_sun_col * 0.05 * pow(max(dot(-V, u_sun), 0.0), 4.0);
    } else if (id == 25) {
        float blk = floor(v_uv.x / 24.0);
        float k = mod(blk * 7.0 + floor(hash(vec2(blk, 3.0)) * 3.0), 16.0);
        vec2 tuv = vec2(fract(v_uv.x / 24.0) * 0.5, (k + clamp(1.0 - v_uv.y, 0.0, 1.0)) / 32.0);
        alb = lin(texture(u_atlas, tuv).rgb);
        E = alb * 0.35 * u_lamps;
        rough = 0.45;
    } else if (id == 26) {
        float on = v_uv.x + 0.5 < u_startl ? 1.0 : 0.0;
        vec3 lc = u_go > 0.5 ? vec3(0.1, 1.0, 0.2) : vec3(1.0, 0.04, 0.02);
        if (u_go > 0.5) on = 1.0;
        alb = vec3(0.02); E = lc * on * 25.0 + vec3(0.02);
    } else if (id == 27) {
        vec2 f = fract(vec2(v_uv.x * 1.9, v_uv.y));
        alb *= 0.45 + 0.55 * step(0.1, f.x) * step(0.25, f.y);
        rough = 0.45;
    } else if (id == 28) {
        E = alb * emis;
    } else if (id == 31) {
        float u = v_uv.x + u_time * 9.0;
        float k = mod(floor(u / 30.0), 16.0);
        vec2 tuv = vec2(fract(u / 30.0) * 0.5, (k + clamp(1.0 - v_uv.y, 0.0, 1.0)) / 32.0);
        E = lin(texture(u_atlas, tuv).rgb) * emis * mix(0.6, 1.3, u_night);
        alb = vec3(0.01);
    } else if (id == 33) {
        float h = v_uv.y, nz = noise(wp * 0.004), nz2 = noise(wp * 0.03);
        alb = mix(vec3(0.03, 0.07, 0.025), vec3(0.12, 0.11, 0.09), smoothstep(0.25, 0.7, h + nz * 0.2)) * (0.7 + 0.5 * nz2);
        if (v_uv.x > 0.5 && h > 0.6 + nz * 0.12) alb = vec3(0.75);
        rough = 0.9;
    } else if (id == 34) {
        alb = vec3(0.008); rough = 0.06; reflm = 0.9;
        vec2 gq = abs(fract(v_uv * 0.5) - 0.5);
        if (min(gq.x, gq.y) < 0.006) E = vec3(0.05, 0.08, 0.15);
    } else if (id == 35) {
        E = vec3(1.0, 0.95, 0.85) * (0.4 + u_headl * 16.0); rough = 0.1;
    } else if (id == 36) {
        E = vec3(1.0, 0.05, 0.02) * (0.6 + u_headl * 2.5 + u_brake * 14.0); rough = 0.1;
    } else if (id == 37) {
        alb = vec3(0.015); rough = 0.85;
    } else if (id == 38) {
        alb *= 0.7 + 0.5 * noise(vec2(atan(P.z, P.x) * 6.0, P.y * 6.0));
    } else if (id == 39) {
        rough = mix(0.55, 0.15, u_wet); reflm = u_wet * 0.5;
    }
    vec3 F0 = mix(vec3(0.04), alb, metal);
    float sh = 1.0;
    if (u_shadow_on > 0.5 && u_reflpass < 0.5) {
        vec3 Ps = P + N * 0.12;
        sh = min(shadow1(u_shs, u_lvp_s, Ps, u_ts), shadow1(u_shd, u_lvp_d, Ps, u_td));
    }
    vec3 col = brdf(N, V, u_sun, alb, rough, metal, F0) * u_sun_col * sh;
    vec3 amb = mix(u_ground, u_top * 1.3 + u_hor * 0.4, N.y * 0.5 + 0.5);
    col += alb * (1.0 - metal) * amb + alb * metal * (amb * 0.8 + u_sun_col * 0.04);
    float NdV = max(dot(N, V), 0.0);
    vec3 R = reflect(-V, N);
    vec3 Fr = F0 + (max(vec3(1.0 - rough), F0) - F0) * pow(1.0 - NdV, 5.0);
    float occ = mix(sh, 1.0, 0.4);
    col += env(R, rough) * Fr * occ;
    if (clear > 0.5) {
        vec3 Fc = F_S(NdV, vec3(0.04));
        col += env(R, 0.02) * Fc * occ + brdf(N, V, u_sun, vec3(0.0), 0.05, 0.0, vec3(0.04)) * u_sun_col * sh;
    }
    for (int i = 0; i < u_nl; i++) {
        vec3 Lv = u_lpos[i].xyz - P;
        float d2 = dot(Lv, Lv), r = u_lpos[i].w;
        if (d2 > r * r) continue;
        vec3 L = Lv * inversesqrt(d2);
        float att = pow(clamp(1.0 - d2 / (r * r), 0.0, 1.0), 2.0) / (1.0 + d2 * 0.01);
        col += brdf(N, V, L, alb, max(rough, clear > 0.5 ? 0.05 : 0.08), metal, F0) * u_lcol[i].rgb * (u_lcol[i].a * att * 7.0);
    }
    if (reflm > 0.0 && u_refl_on > 0.5 && u_reflpass < 0.5) {
        vec2 suv = gl_FragCoord.xy / u_res + N.xz * 0.035;
        vec3 rc = texture(u_refl, suv).rgb;
        float fr = 0.04 + 0.96 * pow(1.0 - NdV, 5.0);
        col = mix(col, rc, reflm * clamp(fr * 1.4 + 0.2, 0.0, 1.0));
    }
    col += E + extra;
    float fogd = 1.0 - exp(-dist * u_fog_den * (0.35 + 0.65 * exp(-max(P.y, 0.0) * 0.004)));
    vec3 fogc = u_fog + u_sun_col * 0.03 * pow(max(dot(-V, u_sun), 0.0), 6.0);
    if (u_studio > 0.5) fogd = 0.0;
    f_color = vec4(mix(col, fogc, fogd), alpha);
}
""").replace("__MAXL__", str(MAXL))

VS_SHADOW = """
#version 330
uniform mat4 u_lvp; uniform mat4 u_model;
in vec3 in_pos; in vec4 in_mat; in vec2 in_uv;
out vec4 v_mat; out vec2 v_uv;
void main() { v_mat = in_mat; v_uv = in_uv; gl_Position = u_lvp * u_model * vec4(in_pos, 1.0); }
"""
FS_SHADOW = """
#version 330
in vec4 v_mat; in vec2 v_uv;
void main() {
    int id = int(v_mat.x + 0.5);
    if (id == 20) { vec2 q = vec2(v_uv.x + v_uv.y, v_uv.x - v_uv.y) * 4.0; if (min(fract(q.x), fract(q.y)) > 0.15) discard; }
    if (id == 18 || id == 17 || id == 13) discard;
}
"""

VS_QUAD = """
#version 330
in vec2 in_pos; out vec2 v_uv; out vec2 v_ndc;
void main() { v_uv = in_pos * 0.5 + 0.5; v_ndc = in_pos; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FS_SKY = """
#version 330
uniform mat4 u_inv; uniform vec3 u_cam; uniform vec3 u_sun; uniform vec3 u_sun_col; uniform vec3 u_top; uniform vec3 u_hor;
uniform float u_time; uniform float u_clouds; uniform float u_stars; uniform float u_moon; uniform float u_night; uniform float u_studio;
in vec2 v_uv; in vec2 v_ndc; out vec4 f_color;
""" + GLSL_COMMON + """
void main() {
    vec4 p = u_inv * vec4(v_ndc, 1.0, 1.0);
    vec3 d = normalize(p.xyz / p.w - u_cam);
    if (u_studio > 0.5) { f_color = vec4(mix(vec3(0.004), vec3(0.015, 0.017, 0.028), d.y * 0.5 + 0.5), 1.0); return; }
    float e = d.y;
    vec3 c = mix(u_hor, u_top, pow(clamp(e, 0.0, 1.0), 0.45));
    if (e < 0.0) c = u_hor * 0.7;
    float sd = max(dot(d, u_sun), 0.0);
    c += u_sun_col * (pow(sd, 8.0) * 0.05 + pow(sd, 90.0) * 0.25);
    float disc = smoothstep(0.99955, 0.9998, sd);
    c += u_sun_col * disc * (u_moon > 0.5 ? 40.0 : 12.0);
    c += u_night * vec3(1.0, 0.45, 0.2) * exp(-max(e, 0.0) * 14.0) * 0.09;
    if (e > 0.005) {
        vec2 cp = d.xz / max(d.y, 0.02) * 1800.0 + u_time * vec2(7.0, 2.5);
        float cl = fbm(cp * 0.0006) * 0.7 + fbm(cp * 0.0024) * 0.3;
        cl = smoothstep(1.02 - u_clouds, 1.02 - u_clouds + 0.32, cl);
        vec3 lit = mix(u_hor * 1.15 + vec3(0.08), u_sun_col * 0.3 + u_hor * 0.5, pow(sd, 3.0));
        vec3 ccol = mix(lit, u_hor * 0.9, 0.35 * cl) * mix(1.0, 0.25, u_night * (1.0 - u_moon * 0.3));
        if (u_stars > 0.5) ccol = u_hor * 1.6 + vec3(0.03, 0.02, 0.01);
        c = mix(c, ccol, cl * smoothstep(0.005, 0.12, e));
        if (u_stars > 0.5) {
            vec3 q = floor(d * 420.0);
            float h = fract(sin(dot(q, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
            if (h > 0.9975) c += vec3(1.0) * (0.5 + 0.5 * sin(u_time * 3.0 + h * 90.0)) * smoothstep(0.0, 0.2, e) * (1.0 - cl) * 1.5;
        }
    }
    f_color = vec4(c, 1.0);
}
"""

VS_FX = """
#version 330
uniform mat4 u_vp;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col; in vec3 in_uvk;
out vec3 v_col; out vec3 v_uvk; out vec3 v_world;
void main() { v_col = in_col; v_uvk = in_uvk; v_world = in_pos; gl_Position = u_vp * vec4(in_pos, 1.0); }
"""
FS_FX = """
#version 330
uniform vec3 u_cam; uniform float u_fog_den; uniform float u_time;
in vec3 v_col; in vec3 v_uvk; in vec3 v_world; out vec4 f_color;
""" + GLSL_COMMON + """
void main() {
    float k = v_uvk.z;
    float fade = exp(-length(u_cam - v_world) * u_fog_den * 0.7);
    if (k < 15.5 || k > 16.5) {
        float r = length(v_uvk.xy);
        float a = 1.0 - smoothstep(0.0, 1.0, r);
        if (k > 16.5) a *= smoothstep(6.0, 30.0, length(u_cam - v_world));
        f_color = vec4(v_col * a * a * fade, 1.0);
    } else {
        float a = (1.0 - smoothstep(0.15, 1.0, abs(v_uvk.x))) * pow(1.0 - v_uvk.y, 1.5) * 0.5;
        a *= 0.65 + 0.35 * noise(vec2(v_uvk.y * 30.0 - u_time * 2.0, v_uvk.x * 3.0));
        f_color = vec4(v_col * a * fade, 1.0);
    }
}
"""

FS_DOWN = """
#version 330
uniform sampler2D u_src; uniform vec2 u_texel; uniform float u_thresh;
in vec2 v_uv; out vec4 f_color;
vec3 s(vec2 o) { return texture(u_src, v_uv + o * u_texel).rgb; }
void main() {
    vec3 a = s(vec2(-2, -2)), b = s(vec2(0, -2)), c = s(vec2(2, -2));
    vec3 d = s(vec2(-1, -1)), e = s(vec2(1, -1));
    vec3 f = s(vec2(-2, 0)), g = s(vec2(0, 0)), h = s(vec2(2, 0));
    vec3 i = s(vec2(-1, 1)), j = s(vec2(1, 1));
    vec3 k = s(vec2(-2, 2)), l = s(vec2(0, 2)), m = s(vec2(2, 2));
    vec3 col = (d + e + i + j) * 0.125 + (a + c + k + m) * 0.03125 + (b + f + h + l) * 0.0625 + g * 0.125;
    if (u_thresh > 0.0) {
        float br = max(col.r, max(col.g, col.b));
        float soft = clamp(br - u_thresh + 0.5, 0.0, 1.0);
        soft = soft * soft * 0.5;
        col *= max(soft, br - u_thresh) / max(br, 1e-4);
        col = min(col, vec3(60.0));
    }
    f_color = vec4(col, 1.0);
}
"""
FS_UP = """
#version 330
uniform sampler2D u_src; uniform vec2 u_texel; uniform float u_k;
in vec2 v_uv; out vec4 f_color;
void main() {
    vec3 c = vec3(0.0);
    c += texture(u_src, v_uv + vec2(-1, -1) * u_texel).rgb * 1.0;
    c += texture(u_src, v_uv + vec2(0, -1) * u_texel).rgb * 2.0;
    c += texture(u_src, v_uv + vec2(1, -1) * u_texel).rgb * 1.0;
    c += texture(u_src, v_uv + vec2(-1, 0) * u_texel).rgb * 2.0;
    c += texture(u_src, v_uv).rgb * 4.0;
    c += texture(u_src, v_uv + vec2(1, 0) * u_texel).rgb * 2.0;
    c += texture(u_src, v_uv + vec2(-1, 1) * u_texel).rgb * 1.0;
    c += texture(u_src, v_uv + vec2(0, 1) * u_texel).rgb * 2.0;
    c += texture(u_src, v_uv + vec2(1, 1) * u_texel).rgb * 1.0;
    f_color = vec4(c / 16.0 * u_k, 1.0);
}
"""
FS_BLUR1D = """
#version 330
uniform sampler2D u_src; uniform vec2 u_dir;
in vec2 v_uv; out vec4 f_color;
void main() {
    vec3 c = vec3(0.0); float wsum = 0.0;
    for (int i = -8; i <= 8; i++) { float w = exp(-float(i * i) / 28.0); c += texture(u_src, v_uv + u_dir * float(i)).rgb * w; wsum += w; }
    f_color = vec4(c / wsum, 1.0);
}
"""
FS_COMP = """
#version 330
uniform sampler2D u_hdr; uniform sampler2D u_bloom; uniform sampler2D u_streak; uniform sampler2D u_blur; uniform sampler2D u_depth;
uniform float u_exposure; uniform float u_bloom_k; uniform float u_streak_k; uniform float u_ca; uniform float u_radial;
uniform float u_vig; uniform float u_grain; uniform float u_time; uniform float u_sat; uniform vec3 u_grade;
uniform float u_dof; uniform float u_focus; uniform float u_aperture; uniform float u_near; uniform float u_far;
uniform float u_fade; uniform float u_flash;
in vec2 v_uv; out vec4 f_color;
float h(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
float lz(float d) { float z = d * 2.0 - 1.0; return 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near)); }
void main() {
    vec2 uv = v_uv;
    vec2 dc = uv - vec2(0.5, 0.52);
    vec3 c;
    if (u_radial > 0.001) {
        vec3 acc = vec3(0.0);
        for (int i = 0; i < 10; i++) { float t = float(i) / 9.0; acc += texture(u_hdr, uv - dc * t * u_radial).rgb; }
        c = acc / 10.0;
    } else c = texture(u_hdr, uv).rgb;
    if (u_ca > 0.0) {
        vec2 o = dc * u_ca;
        c.r = mix(c.r, texture(u_hdr, uv + o).r, 0.8);
        c.b = mix(c.b, texture(u_hdr, uv - o).b, 0.8);
    }
    if (u_dof > 0.5) {
        float z = lz(texture(u_depth, uv).r);
        float coc = clamp(abs(z - u_focus) / max(u_focus, 0.1) * u_aperture, 0.0, 1.0);
        c = mix(c, texture(u_blur, uv).rgb, coc);
    }
    c += texture(u_bloom, uv).rgb * u_bloom_k + texture(u_streak, uv).rgb * u_streak_k;
    c *= u_exposure;
    c = aces(c);
    c = pow(c, vec3(1.0 / 2.2));
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(l), c, u_sat) * u_grade;
    c += vec3(-0.015, 0.005, 0.025) * (1.0 - l) + vec3(0.02, 0.006, -0.018) * l;
    c *= 1.0 - u_vig * dot(dc, dc) * 2.4;
    c += (h(uv * vec2(1931.0, 1093.0) + fract(u_time)) - 0.5) * u_grain;
    c = mix(c, vec3(1.0), u_flash);
    c *= u_fade;
    f_color = vec4(c, dot(c, vec3(0.299, 0.587, 0.114)));
}
"""
FS_FXAA = """
#version 330
uniform sampler2D u_src; uniform vec2 u_texel; uniform float u_on;
in vec2 v_uv; out vec4 f_color;
void main() {
    vec3 rgbM = texture(u_src, v_uv).rgb;
    if (u_on < 0.5) { f_color = vec4(rgbM, 1.0); return; }
    float lNW = texture(u_src, v_uv + vec2(-1, -1) * u_texel).a, lNE = texture(u_src, v_uv + vec2(1, -1) * u_texel).a;
    float lSW = texture(u_src, v_uv + vec2(-1, 1) * u_texel).a, lSE = texture(u_src, v_uv + vec2(1, 1) * u_texel).a;
    float lM = dot(rgbM, vec3(0.299, 0.587, 0.114));
    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE))), lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
    vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
    float red = max((lNW + lNE + lSW + lSE) * 0.03125, 1.0 / 128.0);
    float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + red);
    dir = clamp(dir * rcp, -8.0, 8.0) * u_texel;
    vec3 a = 0.5 * (texture(u_src, v_uv + dir * (1.0 / 3.0 - 0.5)).rgb + texture(u_src, v_uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(u_src, v_uv - dir * 0.5).rgb + texture(u_src, v_uv + dir * 0.5).rgb);
    float lB = dot(b, vec3(0.299, 0.587, 0.114));
    f_color = vec4((lB < lMin || lB > lMax) ? a : b, 1.0);
}
"""
FS_DEPTH = """
#version 330
in vec4 v_mat; in vec2 v_uv; in vec3 v_world;
uniform vec3 u_cam;
float hh(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
void main() {
    int id = int(v_mat.x + 0.5);
    if (id == 20) {
        vec2 q = vec2(v_uv.x + v_uv.y, v_uv.x - v_uv.y) * 4.0;
        float dist = length(u_cam - v_world);
        if (min(fract(q.x), fract(q.y)) > 0.15 || dist > 120.0 || hh(gl_FragCoord.xy) < smoothstep(50.0, 120.0, dist)) discard;
    }
    if (id == 18) discard;
}
"""
FS_DEPTH = """
#version 330
in vec4 v_mat; in vec2 v_uv; in vec3 v_world;
uniform vec3 u_cam;
float hh(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
void main() {
    int id = int(v_mat.x + 0.5);
    if (id == 20) {
        vec2 q = vec2(v_uv.x + v_uv.y, v_uv.x - v_uv.y) * 4.0;
        float dist = length(u_cam - v_world);
        if (min(fract(q.x), fract(q.y)) > 0.15 || dist > 120.0 || hh(gl_FragCoord.xy) < smoothstep(50.0, 120.0, dist)) discard;
    }
    if (id == 18) discard;
}
"""
FS_HUD = """
#version 330
uniform sampler2D u_hud; in vec2 v_uv; out vec4 f_color;
void main() { f_color = texture(u_hud, vec2(v_uv.x, 1.0 - v_uv.y)); }
"""


# ---------------------------------------------------------------- renderizador HDR
def ortho(l, r, b, t, n, f):
    m = np.identity(4, F4)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    m[0, 3], m[1, 3], m[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    return m


def model_full(x, y, z, ang, roll=0.0, pitch=0.0):
    c, s = math.cos(-ang), math.sin(-ang)
    Ry = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], F4)
    cr, sr = math.cos(roll), math.sin(roll)
    Rx = np.array([[1, 0, 0, 0], [0, cr, -sr, 0], [0, sr, cr, 0], [0, 0, 0, 1]], F4)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cp, -sp, 0, 0], [sp, cp, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], F4)
    T = np.identity(4, F4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T @ Ry @ Rx @ Rz


def setu(prog, **kw):
    for k, v in kw.items():
        u = prog.get(k, None)
        if u is None:
            continue
        if isinstance(v, (bytes, bytearray)):
            u.write(v)
        elif isinstance(v, np.ndarray):
            u.write(np.ascontiguousarray(v, F4).tobytes() if v.ndim == 1 or v.shape != (4, 4) else gl_bytes(v))
        else:
            u.value = v


LAYOUT = [("in_pos", 3), ("in_nrm", 3), ("in_col", 3), ("in_mat", 4), ("in_uv", 2)]


def indexed(arr):
    """Convierte triangulos sueltos en vertices unicos + indices (cada vertice se procesa una sola vez)."""
    a = np.ascontiguousarray(arr, F4)
    if len(a) == 0:
        return a, np.zeros(0, np.uint32)
    key = np.ascontiguousarray(np.round(a, 4)).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return a[first], inv.astype(np.uint32).ravel()


def vao_for(ctx, prog, buf, ibo=None):
    """VAO con el formato de 15 floats, saltando los atributos que el shader no usa."""
    fmt, names = [], []
    for nm, k in LAYOUT:
        if prog.get(nm, None) is not None:
            fmt.append(f"{k}f")
            names.append(nm)
        else:
            fmt.append(f"{k * 4}x")
    return ctx.vertex_array(prog, [(buf, " ".join(fmt), *names)], index_buffer=ibo, index_element_size=4)


class Renderer:
    def __init__(self, ctx, W, H, atlas_bytes):
        self.ctx, self.W, self.H = ctx, W, H
        self.p_scene = ctx.program(vertex_shader=VS_SCENE, fragment_shader=FS_MAIN)
        self.p_crowd = ctx.program(vertex_shader=VS_CROWD, fragment_shader=FS_MAIN)
        self.p_shadow = ctx.program(vertex_shader=VS_SHADOW, fragment_shader=FS_SHADOW)
        self.p_depth = ctx.program(vertex_shader=VS_SCENE, fragment_shader=FS_DEPTH)
        self.p_sky = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_SKY)
        self.p_fx = ctx.program(vertex_shader=VS_FX, fragment_shader=FS_FX)
        self.p_down = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_DOWN)
        self.p_up = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_UP)
        self.p_blur = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_BLUR1D)
        self.p_comp = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_COMP)
        self.p_fxaa = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_FXAA)
        self.p_hud = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_HUD)
        qb = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], F4).tobytes())
        self.quads = {p: ctx.vertex_array(p, [(qb, "2f", "in_pos")]) for p in
                      (self.p_sky, self.p_down, self.p_up, self.p_blur, self.p_comp, self.p_fxaa, self.p_hud)}
        self.atlas = ctx.texture((ATLAS, ATLAS), 4, atlas_bytes)
        self.atlas.build_mipmaps()
        self.atlas.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        try:
            self.atlas.anisotropy = 8.0
        except Exception:
            pass
        self.scale = None
        self._alloc(1.0)
        self.ldr = ctx.texture((W, H), 4)
        self.ldr.repeat_x = self.ldr.repeat_y = False
        self.fbo_ldr = ctx.framebuffer([self.ldr])
        self.hud_tex = ctx.texture((W, H), 4)
        self.hud_tex.swizzle = "BGRA"
        self.dummy_shadow = self._shadow_tex(1)
        self.dummy_refl = ctx.texture((1, 1), 4, dtype="f2")
        self.quality = None
        self.shs = self.shd = None
        self.refl = None
        self.static_key = None
        self.lvp_s = self.lvp_d = np.identity(4, F4)

    def _alloc(self, scale):
        """Buferes de la escena HDR a la escala de render elegida (la interfaz siempre va a resolucion completa)."""
        if self.scale == scale:
            return
        ctx = self.ctx
        if self.scale is not None:
            for t, f in self.bloom + self.streak + self.blurtex:
                f.release()
                t.release()
            self.fbo_hdr.release()
            self.hdr.release()
            self.depth.release()
        self.scale = scale
        W, H = max(16, int(self.W * scale)), max(16, int(self.H * scale))
        self.RW, self.RH = W, H
        self.hdr = ctx.texture((W, H), 4, dtype="f2")
        self.hdr.repeat_x = self.hdr.repeat_y = False
        self.depth = ctx.depth_texture((W, H))
        self.depth.compare_func = ""
        self.fbo_hdr = ctx.framebuffer([self.hdr], self.depth)
        self.bloom = []
        w, h = W, H
        for _ in range(6):
            w, h = max(1, w // 2), max(1, h // 2)
            t = ctx.texture((w, h), 4, dtype="f2")
            t.repeat_x = t.repeat_y = False
            self.bloom.append((t, ctx.framebuffer([t])))
        sw, sh = self.bloom[1][0].size
        self.streak = []
        for _ in range(2):
            t = ctx.texture((sw, sh), 4, dtype="f2")
            t.repeat_x = t.repeat_y = False
            self.streak.append((t, ctx.framebuffer([t])))
        self.blurtex = []
        for _ in range(2):
            t = ctx.texture((sw, sh), 4, dtype="f2")
            t.repeat_x = t.repeat_y = False
            self.blurtex.append((t, ctx.framebuffer([t])))

    def _shadow_tex(self, S):
        t = self.ctx.depth_texture((S, S))
        t.compare_func = "<="
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        return t, self.ctx.framebuffer(depth_attachment=t)

    def set_quality(self, q):
        self.quality = q
        self._alloc(q.get("scale", 1.0))
        for res in (self.shs, self.shd):
            if res:
                res[1].release()
                res[0].release()
        self.shs = self._shadow_tex(q["shadow"]) if q["shadow"] else None
        self.shd = self._shadow_tex(2048) if q["shadow"] else None
        self.static_key = None
        if self.refl:
            for r in self.refl:
                r.release()
        self.refl = None
        if q["refl"]:
            rw, rh = self.RW // q["refl"], self.RH // q["refl"]
            t = self.ctx.texture((rw, rh), 4, dtype="f2")
            t.repeat_x = t.repeat_y = False
            d = self.ctx.depth_renderbuffer((rw, rh))
            self.refl = (t, d, self.ctx.framebuffer([t], d))

    def vao_scene(self, arr):
        verts, idx = indexed(arr)
        buf = self.ctx.buffer(verts.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        return (vao_for(self.ctx, self.p_scene, buf, ibo), vao_for(self.ctx, self.p_shadow, buf, ibo),
                vao_for(self.ctx, self.p_depth, buf, ibo))

    # -------------------------------------------------- uniformes comunes
    def common(self, prog, vp, eye, tod, t, refl_pass, lights_bytes, nl, extra):
        setu(prog, u_vp=gl_bytes(vp), u_cam=tuple(map(float, eye)), u_time=float(t), u_clip=1.0 if refl_pass else 0.0,
             u_sun=tod["sun_n"], u_sun_col=tod["sun_col"], u_top=tod["sky_top"], u_hor=tod["sky_hor"],
             u_ground=tod["ground"], u_fog=tod["fog_col"], u_fog_den=tod["fog_den"], u_night=tod["night"],
             u_wet=tod["wet"], u_lamps=tod["lamps"], u_concert=extra.get("concert", 0.0),
             u_studio=extra.get("studio", 0.0), u_reflpass=1.0 if refl_pass else 0.0,
             u_shadow_on=1.0 if (self.shs and tod["shadows"] > 0.5 and not refl_pass) else 0.0,
             u_lvp_s=gl_bytes(self.lvp_s), u_lvp_d=gl_bytes(self.lvp_d),
             u_ts=1.0 / (self.shs[0].size[0] if self.shs else 1), u_td=1.0 / (self.shd[0].size[0] if self.shd else 1),
             u_shs=1, u_shd=2, u_atlas=0, u_refl=3, u_res=(float(self.RW), float(self.RH)),
             u_refl_on=1.0 if (self.refl and not refl_pass) else 0.0, u_nl=nl, u_lpos=lights_bytes[0],
             u_lcol=lights_bytes[1], u_stage=extra.get("stage", (0.0, 0.0)), u_hw=HW,
             u_startl=extra.get("startl", 0.0), u_go=extra.get("go", 0.0), u_pit=extra.get("pit", (0.0, 0.0)),
             u_paint2=(0.0, 0.0, 0.0), u_stripe=0.0, u_brake=0.0, u_headl=extra.get("headl", 0.0),
             u_glass_alpha=0.5, u_model=gl_bytes(np.identity(4, F4)))

    def bind_textures(self, refl_pass):
        self.atlas.use(0)
        (self.shs or self.dummy_shadow)[0].use(1)
        (self.shd or self.dummy_shadow)[0].use(2)
        if self.refl and not refl_pass:
            self.refl[0].use(3)
        else:
            self.dummy_refl.use(3)

    def sky(self, vp, eye, tod, t, studio=0.0):
        setu(self.p_sky, u_inv=gl_bytes(np.linalg.inv(vp)), u_cam=tuple(map(float, eye)), u_sun=tod["sun_n"],
             u_sun_col=tod["sun_col"], u_top=tod["sky_top"], u_hor=tod["sky_hor"], u_time=float(t),
             u_clouds=tod["clouds"], u_stars=tod["stars"], u_moon=tod["moon"], u_night=tod["night"], u_studio=studio)
        self.quads[self.p_sky].render(moderngl.TRIANGLE_STRIP)

    # -------------------------------------------------- sombras
    def light_matrix(self, center, sun, half, depth=500.0):
        sun = norm(sun)
        eye = np.asarray(center, F4) + sun * depth * 0.5
        up = (0, 0, 1) if abs(sun[1]) > 0.99 else (0, 1, 0)
        return ortho(-half, half, -half, half, 1.0, depth) @ look_at(eye, center, up)

    def shadow_pass(self, fbo_tex, lvp, draw):
        tex, fbo = fbo_tex
        fbo.use()
        fbo.clear(depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        setu(self.p_shadow, u_lvp=gl_bytes(lvp), u_model=gl_bytes(np.identity(4, F4)))
        draw(self.p_shadow)

    # -------------------------------------------------- post
    def post(self, P):
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        src = self.hdr
        for i, (t, f) in enumerate(self.bloom):
            f.use()
            src.use(0)
            setu(self.p_down, u_src=0, u_texel=(1.0 / src.size[0], 1.0 / src.size[1]),
                 u_thresh=P.get("thresh", 1.2) if i == 0 else 0.0)
            self.quads[self.p_down].render(moderngl.TRIANGLE_STRIP)
            src = t
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for i in range(len(self.bloom) - 1, 0, -1):
            lo, hi = self.bloom[i][0], self.bloom[i - 1]
            hi[1].use()
            lo.use(0)
            setu(self.p_up, u_src=0, u_texel=(1.0 / lo.size[0], 1.0 / lo.size[1]), u_k=1.0)
            self.quads[self.p_up].render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)
        q = self.quality
        if q["streak"]:
            src = self.bloom[1][0]
            for k, (t, f) in enumerate(self.streak):
                f.use()
                (src if k == 0 else self.streak[0][0]).use(0)
                setu(self.p_blur, u_src=0, u_dir=((3.5 if k == 0 else 9.0) / src.size[0], 0.0))
                self.quads[self.p_blur].render(moderngl.TRIANGLE_STRIP)
        if q["dof"] and P.get("dof", 0) > 0.5:
            self.blurtex[0][1].use()
            self.hdr.use(0)
            setu(self.p_down, u_src=0, u_texel=(2.0 / self.RW, 2.0 / self.RH), u_thresh=0.0)
            self.quads[self.p_down].render(moderngl.TRIANGLE_STRIP)
            for k, d in ((1, (1.0, 0.0)), (0, (0.0, 1.0))):
                self.blurtex[k][1].use()
                self.blurtex[1 - k][0].use(0)
                sz = self.blurtex[0][0].size
                setu(self.p_blur, u_src=0, u_dir=(d[0] * 1.6 / sz[0], d[1] * 1.6 / sz[1]))
                self.quads[self.p_blur].render(moderngl.TRIANGLE_STRIP)
        self.fbo_ldr.use()
        self.hdr.use(0)
        self.bloom[0][0].use(1)
        self.streak[1][0].use(2)
        self.blurtex[0][0].use(3)
        self.depth.use(4)
        setu(self.p_comp, u_hdr=0, u_bloom=1, u_streak=2, u_blur=3, u_depth=4, u_exposure=P["exposure"],
             u_bloom_k=P.get("bloom", 0.08), u_streak_k=P.get("streak", 0.05) if q["streak"] else 0.0,
             u_ca=P.get("ca", 0.0), u_radial=P.get("radial", 0.0), u_vig=P.get("vig", 0.35), u_grain=P.get("grain", 0.025),
             u_time=float(P.get("time", 0.0)), u_sat=P.get("sat", 1.1), u_grade=P.get("grade", (1.0, 1.0, 1.0)),
             u_dof=1.0 if (q["dof"] and P.get("dof", 0) > 0.5) else 0.0, u_focus=P.get("focus", 10.0),
             u_aperture=P.get("aperture", 0.6), u_near=0.5, u_far=9000.0, u_fade=P.get("fade", 1.0),
             u_flash=P.get("flash", 0.0))
        self.quads[self.p_comp].render(moderngl.TRIANGLE_STRIP)

    def present(self, screen_fbo, hud_surface):
        ctx = self.ctx
        screen_fbo.use()
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ldr.use(0)
        setu(self.p_fxaa, u_src=0, u_texel=(1.0 / self.W, 1.0 / self.H), u_on=1.0)
        self.quads[self.p_fxaa].render(moderngl.TRIANGLE_STRIP)
        self.hud_tex.write(hud_surface.get_buffer().raw)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.hud_tex.use(0)
        setu(self.p_hud, u_hud=0)
        self.quads[self.p_hud].render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)


class StaticChunks:
    """Escenario estatico partido en bloques: se dibuja solo lo que esta frente a la camara, de cerca a lejos."""
    SIZE = 170.0

    def __init__(self, R, arr, track_center, keep_radius=1300.0):
        tri = arr.reshape(-1, 3, 15)
        cen = tri[:, :, 0:3].mean(1)
        ext = np.ptp(tri[:, :, 0:3], axis=1).max(1)
        far = (ext > 120) | (np.hypot(cen[:, 0] - track_center[0], cen[:, 2] - track_center[1]) > keep_radius)
        self.always = R.vao_scene(tri[far].reshape(-1, 15)) if far.any() else None
        tri, cen = tri[~far], cen[~far]
        keys = np.floor(cen[:, [0, 2]] / self.SIZE).astype(int)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        tri, cen, keys = tri[order], cen[order], keys[order]
        starts = np.r_[0, np.nonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1, len(tri)]
        self.chunks = []
        for a, b in zip(starts[:-1], starts[1:]):
            part = tri[a:b]
            pts = part[:, :, 0:3].reshape(-1, 3)
            c = (pts.min(0) + pts.max(0)) / 2
            r = float(np.linalg.norm(pts.max(0) - pts.min(0)) / 2)
            v, s, d = R.vao_scene(part.reshape(-1, 15))
            self.chunks.append((c, r, v, s, d))
        self.vertices = len(arr)
        self.C = np.array([c for c, *_ in self.chunks], F4).reshape(-1, 3)
        self.Rr = np.array([r for _, r, *_ in self.chunks], F4)

    def visible(self, eye, fwd, cosl, maxd):
        eye = np.asarray(eye, F4)
        out = []
        D = self.C - eye
        dist = np.sqrt((D * D).sum(1))
        ok = (dist - self.Rr <= maxd) & ~((dist > self.Rr) & (D @ np.asarray(fwd, F4) < dist * cosl - self.Rr))
        for i in np.nonzero(ok)[0]:
            c, r, v, s, dp = self.chunks[i]
            out.append((float(dist[i]), v, dp, i, r))
        out.sort(key=lambda x: x[0])
        return out

    def draw(self, eye, fwd, cosl, maxd, depth=False, vis=None, pre=None):
        for _, v, dp, c, r in (vis if vis is not None else self.visible(eye, fwd, cosl, maxd)):
            if pre is not None:
                pre(c, r)
            (dp if depth else v).render()
        if self.always:
            if pre is not None:
                pre(None, 0)
            self.always[2 if depth else 0].render()

    def draw_shadow(self, center, half):
        c0 = np.asarray(center, F4)
        for c, r, v, s, dp in self.chunks:
            if np.linalg.norm((c - c0)[[0, 2]]) - r < half * 1.5:
                s.render()


class CrowdGPU:
    """Publico instanciado en bloques con nivel de detalle y descarte por distancia/direccion."""
    CHUNK = 70.0

    def __init__(self, ctx, prog, crowd_arr):
        near, far = build_person_meshes()
        nv, ni = indexed(near)
        fv, fi = indexed(far)
        self.vb_near, self.ib_near = ctx.buffer(nv.tobytes()), ctx.buffer(ni.tobytes())
        self.vb_far, self.ib_far = ctx.buffer(fv.tobytes()), ctx.buffer(fi.tobytes())
        self.tris_near, self.tris_far = len(ni) // 3, len(fi) // 3
        self.chunks = []
        if len(crowd_arr) == 0:
            return
        keys = np.floor(crowd_arr[:, [0, 2]] / self.CHUNK).astype(int)
        flag = crowd_arr[:, 11].astype(int)
        order = np.lexsort((keys[:, 1], keys[:, 0], flag))
        arr, keys, flag = crowd_arr[order], keys[order], flag[order]
        starts = np.r_[0, np.nonzero(np.any(np.diff(np.c_[keys, flag], axis=0) != 0, axis=1))[0] + 1, len(arr)]
        fmt = "3f 1f 3f 4f 4x/i"
        names = ("i_pos", "i_yaw", "i_shirt", "i_misc")
        for a, b in zip(starts[:-1], starts[1:]):
            part = np.ascontiguousarray(arr[a:b], F4)
            part[:, 4:7] = part[:, 4:7]
            ib = ctx.buffer(part.tobytes())
            cen = part[:, 0:3].mean(0)
            rad = float(np.max(np.linalg.norm(part[:, 0:3] - cen, axis=1))) + 2.0
            vn = ctx.vertex_array(prog, [(self.vb_near, "3f 3f 1f 1f", "in_pos", "in_nrm", "in_part", "in_slot"),
                                         (ib, fmt, *names)], index_buffer=self.ib_near, index_element_size=4)
            vf = ctx.vertex_array(prog, [(self.vb_far, "3f 3f 1f 1f", "in_pos", "in_nrm", "in_part", "in_slot"),
                                         (ib, fmt, *names)], index_buffer=self.ib_far, index_element_size=4)
            self.chunks.append(dict(c=cen, r=rad, n=b - a, near=vn, far=vf, concert=bool(flag[a])))
        self.total = len(arr)

    def draw(self, eye, fwd, q, night_concert, tan_half, pre=None):
        eye = np.asarray(eye, F4)
        drawn = 0
        cosl = math.cos(min(1.5, math.atan(tan_half * 1.9) + 0.25))
        if not hasattr(self, "C"):
            self.C = np.array([ch["c"] for ch in self.chunks], F4).reshape(-1, 3)
            self.Rr = np.array([ch["r"] for ch in self.chunks], F4)
            self.flag = np.array([ch["concert"] for ch in self.chunks], bool)
        D = self.C - eye
        dist = np.sqrt((D * D).sum(1))
        dd = dist - self.Rr
        ok = (dd <= q["far"]) & ~((dd > 5) & (D @ np.asarray(fwd, F4) < dist * cosl - self.Rr))
        if not night_concert:
            ok &= ~self.flag
        for i in np.nonzero(ok)[0]:
            ch = self.chunks[i]
            if pre is not None:
                pre(i, ch["r"])
            (ch["near"] if dd[i] < q["near"] else ch["far"]).render(instances=ch["n"])
            drawn += ch["n"]
        return drawn


# ---------------------------------------------------------------- efectos HDR
class FXHD:
    """Particulas HDR: 0 chispa de cohete, 1 cohete, 2 fuego, 3 estela, 4 destello, 5 humo, 6 cometa dorado,
    7 chispa de choque, 8 fuego azul del nitro, 9 confeti."""
    MAXV = 120000
    GRAV = np.array([5.5, 9.8, -2.0, 1.5, 0.0, -0.15, 7.0, 9.8, -1.0, 1.6], F4)
    DRAG = np.array([0.9, 0.0, 2.5, 3.0, 0.0, 0.25, 0.15, 1.2, 4.0, 1.8], F4)
    PALETTE = np.array([[1.0, 0.25, 0.2], [0.3, 0.6, 1.0], [1.0, 0.85, 0.2], [0.8, 0.3, 1.0], [0.3, 1.0, 0.5],
                        [1.0, 0.5, 0.1], [1.0, 1.0, 1.0], [1.0, 0.3, 0.7]], F4)
    TRICOLOR = np.array([[0.0, 0.9, 0.4], [1.0, 1.0, 1.0], [1.0, 0.1, 0.15]], F4)

    def __init__(self, ctx, prog, info):
        self.buf = ctx.buffer(reserve=self.MAXV * 48, dynamic=True)
        self.vao = ctx.vertex_array(prog, [(self.buf, "3f 12x 3f 3f", "in_pos", "in_col", "in_uvk")])
        self.info = info
        self.rng = np.random.default_rng(3)
        self.p = np.zeros((0, 3), F4)
        self.v = np.zeros((0, 3), F4)
        self.c = np.zeros((0, 3), F4)
        self.life = np.zeros(0, F4)
        self.maxl = np.zeros(0, F4)
        self.size = np.zeros(0, F4)
        self.kind = np.zeros(0, np.int8)
        self.next_launch = self.next_flame = 0.0
        self.next_fireball, self.next_fountain = 2.0, 3.0
        self.count = 0
        self.concert = False
        self.flashes = []          # (x, y, z, color, t0) para iluminar el mundo
        self.fireball_t = -10.0

    def spawn(self, p, v, c, life, size, kind):
        p = np.asarray(p, F4).reshape(-1, 3)
        k = len(p)
        self.p = np.concatenate([self.p, p])
        self.v = np.concatenate([self.v, np.broadcast_to(np.asarray(v, F4), (k, 3))])
        self.c = np.concatenate([self.c, np.broadcast_to(np.asarray(c, F4), (k, 3))])
        life = np.broadcast_to(np.asarray(life, F4), (k,))
        self.life = np.concatenate([self.life, life])
        self.maxl = np.concatenate([self.maxl, life])
        self.size = np.concatenate([self.size, np.broadcast_to(np.asarray(size, F4), (k,))])
        self.kind = np.concatenate([self.kind, np.full(k, kind, np.int8)])

    def _concert(self, t):
        rng, info = self.rng, self.info
        S, ff, rr = info["stage"]
        lp = info["launch"]
        if self.next_launch < t - 2:
            self.next_launch = t
        while self.next_launch < t:
            o = np.array(lp[rng.integers(len(lp))], F4)
            self.spawn(o, [rng.normal(0, 5), rng.uniform(30, 42), rng.normal(0, 5)], [3.0, 2.2, 1.2], 3.5, 1.4, 1)
            self.next_launch += rng.uniform(0.05, 0.3)
        if t > self.next_flame:
            for f in info["flames"]:
                k = 24
                v = np.column_stack([rng.normal(0, 1.2, k), rng.uniform(9, 18, k), rng.normal(0, 1.2, k)])
                c = np.column_stack([np.ones(k), rng.uniform(0.35, 0.7, k), rng.uniform(0.0, 0.12, k)]) * 3.0
                self.spawn(np.tile(f, (k, 1)), v, c, rng.uniform(0.35, 0.7, k), rng.uniform(1.0, 1.8, k), 2)
            self.next_flame = t + rng.choice([0.9, 1.4, 2.2, 3.0])
        if t > self.next_fireball:
            for f in info["fireballs"]:
                k = 26
                v = np.column_stack([rng.normal(0, 2.5, k), rng.uniform(4, 12, k), rng.normal(0, 2.5, k)])
                c = np.column_stack([np.full(k, 0.8), rng.uniform(0.25, 0.45, k), rng.uniform(0.02, 0.08, k)]) * 2.4
                self.spawn(np.tile(f, (k, 1)), v, c, rng.uniform(0.5, 1.0, k), rng.uniform(2.6, 4.6, k), 2)
            self.fireball_t = t
            self.next_fireball = t + rng.choice([2.5, 4.0, 5.5, 7.0])
        if t > self.next_fountain:
            for o in info["fountain"]:
                for _ in range(2):
                    ang = rng.uniform(-0.55, 0.55)
                    hx, hz = rr * math.sin(ang) * 16
                    v = [hx + rng.normal(0, 1), rng.uniform(34, 46), hz + rng.normal(0, 1)]
                    self.spawn(o, v, [3.0, 2.3, 1.0], rng.uniform(1.6, 2.3), 1.4, 6)
            self.next_fountain = t + rng.uniform(5, 9)
        nsmoke = int((self.kind == 5).sum())
        if nsmoke < 50:
            k = min(3, 50 - nsmoke)
            a, b = rng.uniform(-40, 40, k), rng.uniform(-2, 48, k)
            pos = np.column_stack([S[0] + rr[0] * a + ff[0] * b, rng.uniform(2, 10, k), S[1] + rr[1] * a + ff[1] * b])
            v = np.column_stack([rng.normal(0, 0.6, k), rng.uniform(0, 0.4, k), rng.normal(0, 0.6, k)])
            col = np.array([0.07, 0.02, 0.09], F4) * rng.uniform(0.6, 1.2, (k, 1))
            self.spawn(pos, v, col, rng.uniform(9, 16, k), rng.uniform(13, 24, k), 5)

    def sparks(self, x, z, nx, nz, strength):
        rng = self.rng
        k = int(clamp(strength * 1.8, 3, 40))
        d = rng.normal(0, 1, (k, 3))
        d[:, 1] = np.abs(d[:, 1]) * 1.3 + 0.4
        d[:, 0] -= nx * 1.2
        d[:, 2] -= nz * 1.2
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        sp = rng.uniform(4, 10 + min(strength, 20) * 0.6, (k, 1))
        col = np.column_stack([np.ones(k), rng.uniform(0.55, 0.85, k), rng.uniform(0.15, 0.35, k)]) * 7.0
        self.spawn(np.tile([x, 0.6, z], (k, 1)), d * sp, col, rng.uniform(0.3, 0.7, k), rng.uniform(0.12, 0.24, k), 7)
        self.spawn([x, 0.7, z], (0, 0, 0), (3.0, 1.8, 0.7), 0.08, 1.4 + min(strength, 15) * 0.1, 4)
        self.flashes.append((x, 1.0, z, (1.0, 0.6, 0.25), 0.15, 2.0))

    def nitro_flame(self, car, exhausts):
        rng = self.rng
        fx, fz = math.cos(car.ang), math.sin(car.ang)
        rx, rz = -fz, fx
        for ex, ey, ez in exhausts:
            ox, oz = car.x + fx * ex + rx * ez, car.z + fz * ex + rz * ez
            k = 6
            v = np.column_stack([car.vx * 0.6 - fx * rng.uniform(6, 13, k) + rng.normal(0, 0.6, k),
                                 rng.normal(0.3, 0.4, k),
                                 car.vz * 0.6 - fz * rng.uniform(6, 13, k) + rng.normal(0, 0.6, k)])
            white = rng.random(k) < 0.3
            col = np.where(white[:, None], [[1.6, 1.9, 2.6]], [[0.35, 0.8, 3.2]])
            self.spawn(np.tile([ox, ey, oz], (k, 1)), v, col, rng.uniform(0.12, 0.3, k), rng.uniform(0.3, 0.65, k), 8)
            self.spawn([ox, ey, oz], (0, 0, 0), (0.3, 0.6, 2.2), 0.05, 1.6, 4)

    def confetti(self, x, z, n=400):
        rng = self.rng
        cols = np.array([[1, 0.1, 0.1], [0.1, 0.8, 0.3], [1, 1, 1], [1, 0.8, 0.1], [0.2, 0.5, 1.0]], F4)
        v = np.column_stack([rng.normal(0, 5, n), rng.uniform(8, 20, n), rng.normal(0, 5, n)])
        self.spawn(np.tile([x, 6.0, z], (n, 1)) + rng.normal(0, 2, (n, 3)), v, cols[rng.integers(5, size=n)] * 1.2,
                   rng.uniform(3, 5, n), rng.uniform(0.12, 0.2, n), 9)

    def update(self, dt, t):
        if self.concert:
            self._concert(t)
        elif len(self.kind):
            keep = (self.kind >= 7) | (self.kind == 4)
            if not keep.all():
                for name in ("p", "v", "c", "life", "maxl", "size", "kind"):
                    setattr(self, name, getattr(self, name)[keep])
        self.flashes = [f for f in self.flashes if t - f[4] < 0.5] if False else \
            [(x, y, z, c, life - dt, inten) for x, y, z, c, life, inten in self.flashes if life - dt > 0]
        if not len(self.p):
            return
        rng = self.rng
        k = self.kind
        self.v[:, 1] -= self.GRAV[k] * dt
        self.v *= (1 - self.DRAG[k] * dt)[:, None]
        conf = k == 9
        if conf.any():
            self.v[conf, 0] += np.sin(t * 3 + self.p[conf, 1]) * dt * 3
        self.p += self.v * dt
        self.life -= dt
        for kind, col, life, size in ((1, (2.5, 1.5, 0.6), 0.5, 0.9), (6, (3.0, 2.2, 0.9), 0.8, 1.0)):
            idx = np.nonzero(k == kind)[0]
            if len(idx):
                self.spawn(self.p[idx], (0, 0, 0), col, life, size, 3)
        k = self.kind
        burst = np.nonzero((k == 1) & ((self.v[:, 1] < 5) | (self.life < 0.15)))[0]
        for i in burst[:6]:
            self.life[i] = 0
            nb = 190
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
            self.spawn(np.tile(self.p[i], (nb, 1)), d * sp, col * 6.0, rng.uniform(1.6, 2.8, nb), 2.4, 0)
            self.spawn(self.p[i], (0, 0, 0), col[0] * 4 + 4, 0.3, 18.0, 4)
            x, y, z = self.p[i]
            self.flashes.append((float(x), float(y), float(z), tuple(map(float, col[0] * 0.7 + 0.3)), 0.6, 30.0))
        alive = self.life > 0
        if len(alive) > 14000:
            alive[: len(alive) - 14000] = False
        for name in ("p", "v", "c", "life", "maxl", "size", "kind"):
            setattr(self, name, getattr(self, name)[alive])

    def build(self, view, eye, beams):
        right, up = view[0, :3], view[1, :3]
        n = len(self.p)
        parts = []
        if n:
            f = np.clip(self.life / self.maxl, 0, 1)
            k = self.kind
            bright = np.select([k == 0, k == 3, k == 2, k == 4, k == 5, k == 7, k == 8, k == 9],
                               [f ** 0.6 * self.rng.uniform(0.6, 1.0, n), f * 0.6, f, f * f,
                                np.sin(np.pi * f) * 0.9, f, f ** 0.8, np.minimum(1, f * 3)], 1.0).astype(F4)
            size = self.size * np.select([k == 0, k == 2, k == 5, k == 7, k == 8],
                                         [0.5 + 0.5 * f, 1 + (1 - f) * 1.5, 1 + (1 - f) * 0.5, 0.6 + 0.4 * f,
                                          0.4 + 0.9 * f], 1.0)
            col = self.c * bright[:, None]
            corners = np.array([(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)], F4)
            off = corners[:, 0:1] * right[None, :] + corners[:, 1:2] * up[None, :]
            verts = np.zeros((n, 6, 12), F4)
            verts[:, :, 0:3] = self.p[:, None, :] + off[None, :, :] * size[:, None, None]
            verts[:, :, 6:9] = col[:, None, :]
            verts[:, :, 9:11] = corners[None, :, :]
            verts[:, :, 11] = np.where(k == 5, 17.0, 15.0)[:, None]
            parts.append(verts.reshape(-1, 12))
        if beams:
            eye = np.asarray(eye, F4)
            A = np.array([b[0] for b in beams], F4)
            B = np.array([b[1] for b in beams], F4)
            w0 = np.array([b[2] for b in beams], F4)[:, None]
            w1 = np.array([b[3] for b in beams], F4)[:, None]
            colb = np.array([b[4] for b in beams], F4)
            sd = np.cross(B - A, eye[None, :] - A)
            sd /= np.maximum(np.linalg.norm(sd, axis=1, keepdims=True), 1e-6)
            P6 = np.stack([A - sd * w0, A + sd * w0, B + sd * w1, A - sd * w0, B + sd * w1, B - sd * w1], 1)
            uv = np.array([(-1, 0), (1, 0), (1, 1), (-1, 0), (1, 1), (-1, 1)], F4)
            bv = np.zeros((len(beams), 6, 12), F4)
            bv[:, :, 0:3] = P6
            bv[:, :, 6:9] = colb[:, None, :]
            bv[:, :, 9:11] = uv[None]
            bv[:, :, 11] = 16
            parts.append(bv.reshape(-1, 12))
        if not parts:
            self.count = 0
            return
        data = np.concatenate(parts)[: self.MAXV]
        self.buf.write(data.tobytes())
        self.count = len(data)

    def render(self):
        if self.count:
            self.vao.render(vertices=self.count)


def concert_beams_hd(t, info):
    S, ff, rr = info["stage"]
    cols = [(1.0, 0.15, 0.15), (0.25, 0.45, 1.0), (1.0, 1.0, 1.0), (0.8, 0.2, 1.0), (0.2, 1.0, 0.4), (1.0, 0.6, 0.1)]
    beams = []
    for i, (x, y, z) in enumerate(info["movers"]):
        if i % 2:
            yaw, pitch = math.sin(t * 0.5 + i) * 1.45, 0.2 + 0.15 * math.sin(t * 1.1 + i)
        else:
            yaw, pitch = math.sin(t * 0.9 + i * 0.8) * 0.9, 0.75 + 0.5 * math.sin(t * 0.6 + i * 1.3)
        fx = ff[0] * math.cos(yaw) + rr[0] * math.sin(yaw)
        fz = ff[1] * math.cos(yaw) + rr[1] * math.sin(yaw)
        d = (fx * math.cos(pitch), math.sin(pitch), fz * math.cos(pitch))
        c = cols[(i + int(t / 2)) % len(cols)]
        beams.append(((x, y, z), (x + d[0] * 190, y + d[1] * 190, z + d[2] * 190), 0.35, 9.0,
                      tuple(v * 1.4 for v in c)))
    for i in range(8):
        a = -0.9 + 1.8 * i / 7 + 0.35 * math.sin(t * 1.5)
        dx = ff[0] * math.cos(a) + rr[0] * math.sin(a)
        dz = ff[1] * math.cos(a) + rr[1] * math.sin(a)
        el = 0.12 + 0.1 * math.sin(t * 3 + i)
        o = (S[0] + ff[0] * 6.5, 3.2, S[1] + ff[1] * 6.5)
        lc = (0.2, 3.0, 0.5) if i % 2 else (0.3, 0.9, 3.0)
        beams.append((o, (o[0] + dx * 140, o[1] + el * 140, o[2] + dz * 140), 0.05, 0.55, lc))
    return beams


# ---------------------------------------------------------------- coche HD (balanceo, ruedas, derrape, takedowns)
class HDCar(Car):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.spin = 0.0
        self.wheel_rot = 0.0
        self.roll = self.pitch = 0.0
        self.braking = False
        self.drifting = False
        self.wheel_r = 0.4
        self.vis = None
        self.crash_t = 0.0

    def drive(self, thr, brk, steer_in, hb, nitro, dt, track):
        if abs(self.spin) > 0.05:
            self.ang = (self.ang + self.spin * dt) % TAU
            self.spin *= max(0.0, 1 - 1.8 * dt)
            thr, nitro = False, False
        else:
            self.spin = 0.0
        vf0 = self.vf
        imp = super().drive(thr, brk, steer_in, hb, nitro, dt, track)
        vf = self.vf
        self.braking = bool(brk) and vf > 1
        self.wheel_rot = (self.wheel_rot + vf * dt / self.wheel_r) % TAU
        lat_v = -self.vx * math.sin(self.ang) + self.vz * math.cos(self.ang)
        self.drifting = abs(lat_v) > 3.5 and abs(vf) > 12
        if self.drifting:
            self.nitro = min(1.0, self.nitro + dt * 0.1)
        acc = (vf - vf0) / max(dt, 1e-4)
        self.pitch += (clamp(-acc * 0.0035, -0.045, 0.045) - self.pitch) * min(1.0, dt * 6)
        self.roll += (clamp(self.steer * abs(vf) * 0.0022 + lat_v * 0.006, -0.06, 0.06) - self.roll) * min(1.0, dt * 6)
        self.crash_t = max(0.0, self.crash_t - dt)
        return imp


# ---------------------------------------------------------------- sonido HD
class SoundHD(Sound):
    def _engine(self, f):
        """Motor con armonicos y aspereza (en lugar de la onda 'chiptune')."""
        period = max(2, int(round(self.freq / f)))
        f = self.freq / period
        cycles = max(1, int(0.2 * f))
        t = np.arange(period * cycles) / self.freq
        rng = np.random.default_rng(7)
        w = np.zeros_like(t)
        for k, amp in ((1, 1.0), (2, 0.65), (3, 0.42), (4, 0.3), (6, 0.18), (8, 0.1), (0.5, 0.35)):
            w += amp * np.sin(TAU * f * k * t + rng.uniform(0, TAU))
        w = np.tanh(w * 1.4)
        w += np.tile(np.convolve(rng.normal(0, 1, period), np.ones(3) / 3, "same"), cycles) * 0.12
        return self._make(w * 0.5, 0.26)

    def _crash(self):
        n = int(self.freq * 0.6)
        rng = np.random.default_rng(2)
        t = np.arange(n) / self.freq
        w = rng.normal(0, 1, n)
        w = np.convolve(w, np.ones(8) / 8, "same") * 2 + np.sin(TAU * 55 * t) * np.exp(-t * 9) * 1.5
        w += np.diff(rng.normal(0, 1, n + 1)) * np.exp(-t * 18) * 0.6
        return self._make(w * np.exp(-t * 5), 0.6)


# ---------------------------------------------------------------- musica HD (synthwave / electro-rock propia)
def _saw(f, t, cutoff=4000.0, voices=1, detune=0.0):
    out = np.zeros_like(t)
    for v in range(voices):
        fv = f * (1 + detune * (v - (voices - 1) / 2))
        nh = int(max(1, min(40, cutoff / fv)))
        ph = np.random.default_rng(v).uniform(0, TAU)
        for k in range(1, nh + 1):
            out += ((-1) ** (k + 1)) * np.sin(TAU * k * fv * t + ph * k) / k
    return out * (2 / np.pi) / max(1, voices) ** 0.5


def _reverb(x, sr, secs=1.6, mix=0.25):
    n = int(sr * secs)
    ir = np.random.default_rng(5).normal(0, 1, n) * np.exp(-np.linspace(0, 7, n))
    ir[0] = 0
    m = 1 << int(np.ceil(np.log2(len(x) + n)))
    y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[: len(x)]
    y /= np.max(np.abs(y)) + 1e-9
    return x * (1 - mix) + y * mix * np.max(np.abs(x))


def synth_hd(sr, style):
    """Genera una cancion completa: 'menu' (synthwave lento) o 'race' (electro-rock energico)."""
    if style == "menu":
        bpm, prog = 96, [(57, True), (53, False), (48, False), (55, False)] * 4
    else:
        bpm, prog = 128, ([(52, True), (48, False), (55, False), (50, False)] * 2 +
                          [(45, True), (52, True), (48, False), (50, False)] * 2) * 2
    beat = 60.0 / bpm
    bar = 4 * beat
    n = int(len(prog) * bar * sr)
    dry = np.zeros(n)
    wetbus = np.zeros(n)
    rng = np.random.default_rng(11)
    cache = {}

    def note(key, f, dur, gen):
        if key not in cache:
            m = int(dur * sr)
            t = np.arange(m) / sr
            cache[key] = gen(f, t)
        return cache[key]

    def put(bus, t0, w, g=1.0):
        s = int(t0 * sr)
        if s >= n:
            return
        e = min(n, s + len(w))
        bus[s:e] += w[:e - s] * g
    mf = lambda m: 440.0 * 2 ** ((m - 69) / 12)

    def pad(f, t):
        env = np.minimum(1, t / 0.6) * np.minimum(1, (t[-1] - t) / 0.5 + 0.02)
        return _saw(f, t, 1800, 3, 0.006) * env
    if style == "menu":
        def pluck(f, t):
            return _saw(f, t, 3200, 1) * np.exp(-t * 9) * np.minimum(1, t / 0.003)
    else:
        def pluck(f, t):
            return np.tanh(_saw(f, t, 2600, 2, 0.01) * 2.5) * np.exp(-t * 6) * np.minimum(1, t / 0.003)

    def bass(f, t):
        return np.tanh((_saw(f, t, 700, 1) + 0.5 * np.sin(TAU * f / 2 * t)) * 1.8) * \
            np.minimum(1, t / 0.004) * np.exp(-t * (3 if style == "menu" else 5))

    def lead(f, t):
        vib = np.sin(TAU * 5.5 * t) * 0.004 * np.minimum(1, t / 0.3)
        w = _saw(f, t * (1 + vib), 3600, 2, 0.008)
        return np.tanh(w * 1.6) * np.minimum(1, t / 0.01) * np.exp(-t * 0.8)
    tk = np.arange(int(0.28 * sr)) / sr
    kick = np.sin(TAU * (48 * tk + 22 * (1 - np.exp(-tk * 30)))) * np.exp(-tk * 9) * 1.1
    ts = np.arange(int(0.25 * sr)) / sr
    snare = (rng.normal(0, 1, len(ts)) * np.exp(-ts * 18) * 0.8 + np.sin(TAU * 185 * ts) * np.exp(-ts * 25) * 0.6)
    th = np.arange(int(0.05 * sr)) / sr
    hat = np.diff(rng.normal(0, 1, len(th) + 1)) * np.exp(-th * 90) * 0.35
    to = np.arange(int(0.25 * sr)) / sr
    ohat = np.diff(rng.normal(0, 1, len(to) + 1)) * np.exp(-to * 14) * 0.25
    duck = np.ones(n)
    melody_race = [76, 79, 83, 81, 79, 76, 74, 76, 72, 76, 79, 76, 74, 71, 74, 79]
    melody_menu = [69, 72, 76, 74, 72, 69, 67, 69]
    for b, (root, minor) in enumerate(prog):
        t0 = b * bar
        tones = [0, 3 if minor else 4, 7]
        for k in tones:
            put(wetbus, t0, note(("pad", root + k), mf(root + k), bar * 1.02, pad), 0.11)
        steps = 16
        for k in range(steps):
            m = root + 12 + [0, 7, 12, tones[1] + 12][k % 4]
            put(wetbus if style == "menu" else dry, t0 + k * bar / steps, note(("pl", m), mf(m), 0.35, pluck),
                0.12 if style == "menu" else 0.09)
        bsteps = 8 if style == "menu" else 16
        for k in range(bsteps):
            m = root - 12 + (12 if (style != "menu" and k % 2) else 0)
            put(dry, t0 + k * bar / bsteps, note(("bs", m, bsteps), mf(m), bar / bsteps * 0.95, bass), 0.32)
        for q in range(4):
            if style == "menu":
                if q in (0, 2):
                    put(dry, t0 + q * beat, kick, 0.75)
                if q in (1, 3):
                    put(wetbus, t0 + q * beat, snare, 0.5)
            else:
                put(dry, t0 + q * beat, kick, 0.85)
                s = int((t0 + q * beat) * sr)
                e = min(n, s + int(0.22 * sr))
                duck[s:e] = np.minimum(duck[s:e], 0.45 + 0.55 * np.linspace(0, 1, e - s))
                if q in (1, 3):
                    put(wetbus, t0 + q * beat, snare, 0.6)
                put(dry, t0 + q * beat + beat / 2, ohat, 0.8)
            for e8 in range(2):
                put(dry, t0 + q * beat + e8 * beat / 2, hat, 0.5)
        if style != "menu" and b >= 8:
            mel = melody_race
            for k in range(4):
                m = mel[(b * 4 + k) % len(mel)]
                put(wetbus, t0 + k * beat, note(("ld", m), mf(m), beat * 0.95, lead), 0.12)
        if style == "menu" and b >= 4 and b % 2 == 0:
            for k in range(4):
                m = melody_menu[(b * 2 + k) % len(melody_menu)]
                put(wetbus, t0 + k * beat * 2, note(("ldm", m), mf(m), beat * 1.9, lead), 0.09)
    wetbus = _reverb(wetbus, sr, 2.2 if style == "menu" else 1.4, 0.35)
    out = (dry + wetbus) * duck
    out = np.tanh(out * 1.3)
    out /= np.max(np.abs(out)) + 1e-9
    f = int(0.01 * sr)
    out[:f] *= np.linspace(0, 1, f)
    out[-f:] *= np.linspace(1, 0, f)
    return out.astype(np.float32) * 0.92


def cached_song(sound, style):
    folder = os.path.join(HERE, "cache_hd")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{style}_{sound.freq}.wav")
    if not os.path.exists(path):
        data = synth_hd(sound.freq, style) if style != "ambience" else crowd_ambience(sound.freq, 10.0) * 0.8
        a = (np.clip(data, -1, 1) * 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sound.freq)
            wf.writeframes(np.column_stack([a, a]).tobytes())
    return pygame.mixer.Sound(path)


class MusicHD:
    """Menu y carreras de dia/tarde: canciones propias. Noche: soundtrack de A7X (8-bit) aleatorio."""

    def __init__(self, sound):
        self.sound = sound
        self.ok = getattr(sound, "ok", False)
        self.mode = None
        self.now = None
        self.np_t = 99.0
        self.vol = 0.6
        self.cur = None
        folder = os.path.join(HERE, "music")
        try:
            self.playlist = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                             if f.lower().endswith((".mp3", ".ogg", ".wav", ".flac"))]
        except OSError:
            self.playlist = []
        if not self.ok:
            return
        self.ch = pygame.mixer.Channel(1)
        self.songs = {k: cached_song(sound, k) for k in ("menu", "race", "ambience")}

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
            self.ch.play(self.songs[mode], loops=-1, fade_ms=900)
        elif mode == "playlist":
            self.next_song()
        self.apply_volume()

    def next_song(self):
        if not self.ok:
            return
        tried = set()
        for _ in range(len(self.playlist)):
            pool = [p for p in self.playlist if p != self.cur and p not in tried] or \
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


# ---------------------------------------------------------------- multijugador en red local (LAN)
GAME_PORT = 47777        # partida (UDP)
DISC_PORT = 47778        # anuncios del anfitrion para que el invitado lo encuentre (UDP broadcast)
NET_RATE = 1 / 30        # envios por segundo


def local_ip():
    """IP de esta computadora en la red local (no envia nada a internet)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Net:
    """Conexion UDP entre anfitrion e invitado (mensajes JSON pequenos)."""

    def __init__(self, role, host_ip=None):
        self.role = role
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        if role == "host":
            self.sock.bind(("", GAME_PORT))
            self.peer = None
        else:
            self.sock.bind(("", 0))
            self.peer = (host_ip, GAME_PORT)
        self.sock.setblocking(False)
        self.ip = local_ip()
        self.last_rx = 0.0
        self.send_t = 0.0
        self.beacon_t = 0.0
        self.name = socket.gethostname()[:20]

    def send(self, obj, addr=None):
        addr = addr or self.peer
        if addr is None:
            return
        try:
            self.sock.sendto(json.dumps(obj, separators=(",", ":")).encode(), addr)
        except OSError:
            pass

    def beacon(self):
        msg = json.dumps({"t": "b", "name": self.name, "port": GAME_PORT}).encode()
        for dst in ("<broadcast>", "255.255.255.255"):
            try:
                self.sock.sendto(msg, (dst, DISC_PORT))
            except OSError:
                pass

    def recv_all(self):
        out = []
        while True:
            try:
                data, addr = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            try:
                out.append((json.loads(data.decode()), addr))
            except ValueError:
                continue
        if out:
            self.last_rx = time.time()
        return out

    def connected(self, timeout=3.0):
        return time.time() - self.last_rx < timeout

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class Discovery:
    """El invitado escucha los anuncios de las partidas en su red."""

    def __init__(self):
        self.hosts = {}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(("", DISC_PORT))
            self.ok = True
        except OSError:
            self.ok = False
        self.sock.setblocking(False)

    def poll(self):
        while self.ok:
            try:
                data, addr = self.sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError, OSError):
                break
            try:
                msg = json.loads(data.decode())
            except ValueError:
                continue
            if msg.get("t") == "b":
                self.hosts[addr[0]] = (msg.get("name", "?"), time.time())
        now = time.time()
        self.hosts = {ip: v for ip, v in self.hosts.items() if now - v[1] < 4}
        return sorted(self.hosts.items())

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def pack_car(c):
    flags = (1 if c.braking else 0) | (2 if c.boosting else 0) | (4 if c.drifting else 0)
    return [c.slot, round(c.x, 3), round(c.z, 3), round(c.ang, 4), round(c.vx, 3), round(c.vz, 3), round(c.steer, 3),
            flags, round(c.nitro, 3), round(c.u, 2), c.lap_idx, 1 if c.finished else 0,
            round(c.finish_time, 3) if c.finished else -1, round(c.roll, 4), round(c.pitch, 4),
            round(c.best_lap, 3) if c.best_lap else -1]


def unpack_car(c, a):
    c.net = dict(x=a[1], z=a[2], ang=a[3], vx=a[4], vz=a[5], t=time.perf_counter())
    c.steer, c.nitro, c.u, c.lap_idx = a[6], a[8], a[9], a[10]
    c.braking, c.boosting, c.drifting = bool(a[7] & 1), bool(a[7] & 2), bool(a[7] & 4)
    if a[11] and not c.finished:
        c.finished, c.finish_time = True, a[12]
    c.roll, c.pitch = a[13], a[14]
    c.best_lap = a[15] if a[15] >= 0 else None


# ---------------------------------------------------------------- juego HD
_FONTS = {}


def font(size, bold=True, italic=True, name="bahnschrift,segoeui,arial"):
    key = (size, bold, italic, name)
    if key not in _FONTS:
        _FONTS[key] = pygame.font.SysFont(name, size, bold=bold, italic=italic)
    return _FONTS[key]


def mat_T(x, y, z):
    m = np.identity(4, F4)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def mat_Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], F4)


def mat_Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], F4)


MIRROR_Z = np.diag([1, 1, -1, 1]).astype(F4)
MIRROR_Y = np.diag([1, -1, 1, 1]).astype(F4)
ACCENT = (0, 210, 255)
ORANGE = (255, 140, 20)


class GameHD:
    def __init__(self, headless=False):
        self.headless = headless
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.init()
        if headless:
            self.ctx = moderngl.create_standalone_context()
            self.screen_tex = self.ctx.texture((WIN_W, WIN_H), 4)
            self.screen_fbo = self.ctx.framebuffer([self.screen_tex])
        else:
            pygame.display.set_mode((WIN_W, WIN_H), pygame.OPENGL | pygame.DOUBLEBUF, vsync=VSYNC)
            pygame.display.set_caption("GTC Racing HD - Gran Premio CDMX")
            self.ctx = moderngl.create_context()
            self.screen_fbo = self.ctx.screen
        self.hud = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        self.R = Renderer(self.ctx, WIN_W, WIN_H, make_atlas_hd())
        self.qi = 2
        self.R.set_quality(QUALITY[self.qi][1])
        self.loading("Construyendo el autódromo...", 0.05)
        self.track = Track()
        g, gc, crowd, lights, self.info = build_world(self.track)
        self.loading("Subiendo geometría a la GPU...", 0.55)
        self.static = StaticChunks(self.R, g.array(), self.track.center)
        self.concert = self.R.vao_scene(gc.array())
        self.static_lights = np.array(lights, F4).reshape(-1, 8)
        self.crowd = CrowdGPU(self.ctx, self.R.p_crowd, crowd.array())
        self.loading("Preparando los vehículos...", 0.7)
        self.vehicles = {}
        for p in PILOTS:
            body, glass, wheel, vinfo = build_vehicle_hd(p)
            standing = Geo()
            character(standing, p["driver"], 0, 0, 0, standing=True)
            self.vehicles[p["name"]] = dict(body=self.R.vao_scene(body.array()), glass=self.R.vao_scene(glass.array())[0],
                                            wheel=self.R.vao_scene(wheel.array()), info=vinfo,
                                            figure=self.R.vao_scene(standing.array())[0], pilot=p)
        self.studio = self.R.vao_scene(build_studio().array())[0]
        self.fx = FXHD(self.ctx, self.R.p_fx, self.info)
        self.loading("Afinando el sonido...", 0.85)
        if headless:
            self.sound = Sound.__new__(Sound)
            self.sound.ok, self.sound.muted = False, True
        else:
            self.sound = SoundHD()
        self.music = MusicHD(self.sound)
        self.clock = pygame.time.Clock()
        self.t = 0.0
        self.tod = 0
        self.sel = len(PILOTS) - 1
        self.diff = 1
        self.cam_mode = 0
        self.show_fps = False
        self.fps = 60.0
        self.toast, self.toast_t = "", 0.0
        self.shake = 0.0
        self.slowmo = 0.0
        self.flash = 0.0
        self.fade = 0.0
        self.tv_t = 0.0
        self.tv_target = 0
        self.mm_cache = None
        self.net = None          # conexion LAN activa
        self.disc = None         # busqueda de partidas (invitado)
        self.mp = None           # None, "host" o "guest"
        self.rid = 0             # numero de carrera multijugador
        self.order = None
        self.peer_pilot = None
        self.browse_sel = 0
        self.ip_typed = ""
        self.lost_t = 0.0
        self.new_race(attract=True)
        self.loading("¡Listo!", 1.0)

    # ------------------------------------------------------------ util
    def loading(self, msg, frac):
        h = self.hud
        h.fill((5, 6, 10, 255))
        self.txt("GTC RACING", 96, (255, 255, 255), (WIN_W // 2, WIN_H // 2 - 80), "center", glow=ACCENT)
        self.txt("HD", 40, ACCENT, (WIN_W // 2 + 270, WIN_H // 2 - 115), "center")
        pygame.draw.rect(h, (40, 44, 55), (WIN_W // 2 - 300, WIN_H // 2 + 20, 600, 10), border_radius=5)
        pygame.draw.rect(h, ACCENT, (WIN_W // 2 - 300, WIN_H // 2 + 20, int(600 * frac), 10), border_radius=5)
        self.txt(msg, 22, (200, 205, 215), (WIN_W // 2, WIN_H // 2 + 60), "center", italic=False)
        self.ctx.clear(0, 0, 0)
        self.R.present(self.screen_fbo, h)
        if not self.headless:
            pygame.display.flip()
            pygame.event.pump()

    def txt(self, s, size, col, pos, anchor="topleft", glow=None, italic=True, shadow=True, alpha=255, bold=True,
            fname=None):
        f = font(size, bold, italic) if fname is None else font(size, bold, italic, fname)
        tc = self.__dict__.setdefault("_tcache", {})
        tk = (s, size, col, bold, italic, fname)
        img = tc.get(tk)
        if img is None:
            if len(tc) > 600:
                tc.clear()
            img = tc[tk] = f.render(s, True, col)
        r = img.get_rect(**{anchor: pos})
        if glow:
            key = (s, size, glow, italic)
            gl = self._glow.get(key) if hasattr(self, "_glow") else None
            if gl is None:
                if not hasattr(self, "_glow"):
                    self._glow = {}
                base_ = f.render(s, True, glow)
                pad = size // 2
                surf = pygame.Surface((base_.get_width() + pad * 2, base_.get_height() + pad * 2), pygame.SRCALPHA)
                surf.blit(base_, (pad, pad))
                small = pygame.transform.smoothscale(surf, (max(1, surf.get_width() // 6), max(1, surf.get_height() // 6)))
                gl = pygame.transform.smoothscale(small, surf.get_size())
                if len(self._glow) > 300:
                    self._glow.clear()
                self._glow[key] = gl
            gl.set_alpha(int(alpha * 0.9))
            self.hud.blit(gl, gl.get_rect(center=r.center))
        if shadow:
            sk = (s, size, (0, 0, 0), bold, italic, fname)
            sh = tc.get(sk)
            if sh is None:
                sh = tc[sk] = f.render(s, True, (0, 0, 0))
            sh.set_alpha(int(alpha * 0.55))
            self.hud.blit(sh, r.move(2, 3))
        img.set_alpha(alpha)
        self.hud.blit(img, r)
        return r

    def panel(self, rect, alpha=170, col=(8, 10, 16), accent=None, slant=14):
        x, y, w, h = rect
        pts = [(x + slant, y), (x + w, y), (x + w - slant, y + h), (x, y + h)]
        pygame.draw.polygon(self.hud, col + (alpha,), pts)
        if accent:
            pygame.draw.polygon(self.hud, accent + (255,), [(x + slant, y), (x + slant + 6, y), (x + 6, y + h), (x, y + h)])

    def tod_params(self):
        d = dict(TOD_HD[self.tod][1])
        d["sun_n"] = tuple(map(float, norm(d["sun"])))
        if self.state in ("select", "lobby"):
            d.update(sun_n=(0.2, 0.95, 0.25), sun_col=(0.6, 0.6, 0.7), sky_top=(0.02, 0.022, 0.035),
                     sky_hor=(0.03, 0.035, 0.05), ground=(0.01, 0.01, 0.014), fog_den=0.0, wet=0.0, lamps=1.0,
                     shadows=0.0, exposure=0.8, night=0.0)
        return d

    # ------------------------------------------------------------ carrera
    def new_race(self, attract=False, intro=True):
        self.state = "menu" if attract else ("intro" if intro else "countdown")
        self.countdown, self.last_count = 3.99, 4
        self.race_time = 0.0
        self.message, self.msg_t = "", 0.0
        self.results_t = 0.0
        self.go_t = -10.0
        self.takedowns = 0
        me = PILOTS[self.sel] if not attract else PILOTS[-1]
        others = [p for p in PILOTS if p is not me]
        random.shuffle(others)
        order = others[:3] + [me] + others[3:]
        _, skill, corner, _, _, brake_f = DIFFS[self.diff if not attract else 2]
        jitter = [0.015, 0.01, 0.005, 0.0, -0.005, -0.01]
        tr = self.track
        self.cars = []
        for slot, pilot in enumerate(order):
            s = tr.L - 10 - (slot // 2) * 12
            x, z, dx, dz = tr.point(s)
            off = -3.3 if slot % 2 == 0 else 3.3
            x += -dz * off
            z += dx * off
            car = HDCar(pilot, x, z, math.atan2(dz, dx), tr, skill + jitter[slot], corner, off * 0.5, pilot is me,
                        slot, brake_f)
            car.vis = self.vehicles[pilot["name"]]
            car.wheel_r = car.vis["info"]["radius"]
            self.cars.append(car)
            if car.is_player:
                self.player = car
        self.cam_yaw = self.player.ang
        self.cam_pos = None
        self.fade = 0.0
        if self.state == "intro":
            self.build_intro()
        self.update_camera(0.0)
        if not attract and self.tod == 2 and self.music.mode == "playlist":
            self.music.next_song()

    # ------------------------------------------------------------ multijugador LAN
    def start_host(self):
        self.leave_mp(silent=True)
        try:
            self.net = Net("host")
        except OSError:
            self.toast, self.toast_t = "NO SE PUDO ABRIR EL PUERTO 47777", 3.0
            self.net = None
            return
        self.mp = "host"
        self.peer_pilot = None
        self.state = "lobby"
        self.fade = 0.0

    def start_browse(self):
        self.leave_mp(silent=True)
        self.disc = Discovery()
        self.browse_sel = 0
        self.ip_typed = ""
        self.state = "browse"

    def join(self, ip):
        try:
            self.net = Net("guest", ip)
        except OSError:
            self.toast, self.toast_t = "NO SE PUDO CONECTAR", 3.0
            return
        self.mp = "guest"
        self.peer_pilot = None
        self.join_t = time.time()
        if self.disc:
            self.disc.close()
            self.disc = None
        self.toast, self.toast_t = f"CONECTANDO CON {ip}...", 3.0

    def leave_mp(self, silent=False, msg=None):
        if self.net:
            self.net.close()
        if self.disc:
            self.disc.close()
        self.net = self.disc = None
        was = self.mp
        self.mp = None
        self.order = None
        if was and not silent:
            self.new_race(attract=True)
            if msg:
                self.toast, self.toast_t = msg, 3.5

    def start_mp_race(self):
        """El anfitrion arma la parrilla: 4 bots + los dos jugadores, y avisa al invitado."""
        g_pi = self.peer_pilot if self.peer_pilot is not None else 0
        free = [i for i in range(len(PILOTS)) if i not in (self.sel, g_pi)]
        random.shuffle(free)
        bots = free[:4]
        while len(bots) < 4:
            bots.append(random.randrange(len(PILOTS)))
        order = [[bots[0], "b"], [bots[1], "b"], [self.sel, "h"], [g_pi, "g"], [bots[2], "b"], [bots[3], "b"]]
        self.rid += 1
        self.new_race_mp(order)

    def new_race_mp(self, order):
        self.order = order
        self.state = "countdown"
        self.countdown, self.last_count = 3.99, 4
        self.race_time = 0.0
        self.message, self.msg_t = "", 0.0
        self.results_t = 0.0
        self.go_t = -10.0
        self.takedowns = 0
        self.lost_t = 0.0
        _, skill, corner, _, _, brake_f = DIFFS[self.diff]
        tr = self.track
        self.cars = []
        for slot, (pi, owner) in enumerate(order):
            pilot = PILOTS[pi]
            s = tr.L - 10 - (slot // 2) * 12
            x, z, dx, dz = tr.point(s)
            off = -3.3 if slot % 2 == 0 else 3.3
            x += -dz * off
            z += dx * off
            mine = (owner == "h" and self.mp == "host") or (owner == "g" and self.mp == "guest")
            car = HDCar(pilot, x, z, math.atan2(dz, dx), tr, skill, corner, off * 0.5, mine, slot, brake_f)
            car.vis = self.vehicles[pilot["name"]]
            car.wheel_r = car.vis["info"]["radius"]
            car.owner = owner
            car.remote = (self.mp == "host" and owner == "g") or (self.mp == "guest" and owner in ("h", "b"))
            car.net = None
            if owner == "h":
                car.name = pilot["name"] + " · J1"
            elif owner == "g":
                car.name = pilot["name"] + " · J2"
            self.cars.append(car)
            if mine:
                self.player = car
        self.cam_yaw = self.player.ang
        self.cam_pos = None
        self.fade = 0.0
        self.update_camera(0.0)
        if self.tod == 2 and self.music.mode == "playlist":
            self.music.next_song()

    def car_by_slot(self, slot):
        for c in self.cars:
            if c.slot == slot:
                return c
        return None

    def lead_human_u(self):
        humans = [c for c in self.cars if getattr(c, "owner", "") in ("h", "g")] or [self.player]
        return max(c.u for c in humans)

    def remote_follow(self, car, dt):
        """Mueve un coche remoto hacia su ultima posicion recibida (con prediccion por su velocidad)."""
        st = car.net
        if st is None:
            return
        age = min(0.25, time.perf_counter() - st["t"])
        tx, tz = st["x"] + st["vx"] * age, st["z"] + st["vz"] * age
        if math.hypot(tx - car.x, tz - car.z) > 15:
            car.x, car.z, car.ang = tx, tz, st["ang"]
        k = min(1.0, dt * 14)
        car.x += (tx - car.x) * k
        car.z += (tz - car.z) * k
        car.ang = (car.ang + wrap_angle(st["ang"] - car.ang) * k) % TAU
        car.vx, car.vz = st["vx"], st["vz"]
        car.wheel_rot = (car.wheel_rot + car.vf * dt / car.wheel_r) % TAU

    def apply_hit(self, msg):
        c = self.car_by_slot(msg.get("slot", -1))
        if c is None or getattr(c, "remote", False):
            return
        c.spin = float(msg.get("spin", 4.0))
        c.vx *= 0.55
        c.vz *= 0.55
        c.crash_t = 3.0
        if c.is_player:
            self.show("¡TE HICIERON TAKEDOWN!", 2.0)
            self.shake = 1.0
            self.flash = 0.25

    def net_update(self, rdt):
        net = self.net
        racing = self.state in ("countdown", "race", "finished")
        for msg, addr in net.recv_all():
            t = msg.get("t")
            if t == "hit":
                self.apply_hit(msg)
            elif net.role == "host" and t == "g":
                if net.peer is None:
                    self.toast, self.toast_t = "¡SE UNIÓ UN JUGADOR!", 2.5
                net.peer = addr
                self.peer_pilot = msg.get("pilot", 0)
                if racing and msg.get("rid") == self.rid and msg.get("car"):
                    c = self.car_by_slot(msg["car"][0])
                    if c is not None and c.remote:
                        unpack_car(c, msg["car"])
            elif net.role == "guest" and t == "h":
                self.peer_pilot = msg.get("pilot", 0)
                self.diff = msg.get("diff", self.diff)
                self.tod = msg.get("tod", self.tod)
                hs = msg.get("st")
                if self.state == "browse" or (hs == "lobby" and self.state != "lobby"):
                    self.state = "lobby"
                    self.toast, self.toast_t = "¡CONECTADO!", 2.0
                if hs in ("countdown", "race") and msg.get("order") and msg.get("rid") != self.rid:
                    self.rid = msg["rid"]
                    self.new_race_mp(msg["order"])
                racing = self.state in ("countdown", "race", "finished")
                if racing and msg.get("rid") == self.rid:
                    if hs == "countdown":
                        self.countdown = msg.get("cd", self.countdown)
                        n = int(self.countdown)
                        if n < self.last_count and n >= 1:
                            self.sound.play("beep")
                        self.last_count = n
                    elif hs == "race" and self.state == "countdown":
                        self.state = "race"
                        self.go_t = self.t
                        self.sound.play("go")
                        self.show("¡ARRANCA!", 1.2)
                    rt = msg.get("rt", self.race_time)
                    if abs(self.race_time - rt) > 0.15:
                        self.race_time = rt
                    for a in msg.get("cars", []):
                        c = self.car_by_slot(a[0])
                        if c is not None and c.remote:
                            unpack_car(c, a)
        # conexion perdida
        if self.mp == "guest" and self.state in ("lobby", "browse") and not net.connected(6.0) and \
                time.time() - getattr(self, "join_t", 0) > 6.0:
            self.leave_mp(msg="NO SE ENCONTRÓ AL ANFITRIÓN")
            return
        if racing and not net.connected(4.0):
            self.lost_t += rdt
            if self.lost_t > 3.0:
                self.leave_mp(msg="SE PERDIÓ LA CONEXIÓN")
                return
        else:
            self.lost_t = 0.0
        # envio
        net.send_t -= rdt
        if net.send_t > 0:
            return
        net.send_t = NET_RATE
        if net.role == "host":
            if self.state == "lobby":
                net.beacon_t -= NET_RATE
                if net.beacon_t <= 0:
                    net.beacon_t = 1.0
                    net.beacon()
            st = "lobby" if self.state == "lobby" else ("countdown" if self.state == "countdown" else "race")
            msg = {"t": "h", "st": st, "pilot": self.sel, "diff": self.diff, "tod": self.tod, "rid": self.rid}
            if racing:
                msg.update(cd=round(self.countdown, 3), rt=round(self.race_time, 3), order=self.order,
                           cars=[pack_car(c) for c in self.cars if not c.remote])
            net.send(msg)
        else:
            msg = {"t": "g", "pilot": self.sel, "rid": self.rid}
            if racing:
                msg["car"] = pack_car(self.player)
            net.send(msg)

    def build_intro(self):
        tr = self.track
        self.gc = np.array([np.mean([c.x for c in self.cars]), np.mean([c.z for c in self.cars])])
        _, _, fx, fz = tr.point(tr.L - 20)
        self.gfwd, self.gright = np.array([fx, fz]), np.array([-fz, fx])
        order = sorted(self.cars, key=lambda c: -c.slot)
        self.shots = [("aerial", 5.0, None)] + ([("concert", 5.0, None)] if self.tod == 2 else []) + \
            [("car", 2.1, c) for c in order] + [("final", 2.4, self.player)]
        self.intro_total = sum(s[1] for s in self.shots)
        self.intro_t = 0.0

    def end_intro(self):
        self.state = "countdown"
        self.countdown, self.last_count = 3.99, 4
        self.cam_yaw = self.player.ang
        self.cam_pos = None
        self.update_camera(0.0)

    def player_inputs(self):
        k = pygame.key.get_pressed()
        thr = k[pygame.K_UP] or k[pygame.K_w]
        brk = k[pygame.K_DOWN] or k[pygame.K_s]
        st = (1 if (k[pygame.K_RIGHT] or k[pygame.K_d]) else 0) - (1 if (k[pygame.K_LEFT] or k[pygame.K_a]) else 0)
        nitro = k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT] or k[pygame.K_x]
        return thr, brk, st, k[pygame.K_SPACE], nitro

    def show(self, msg, dur=2.0):
        self.message, self.msg_t = msg, dur

    def music_mode(self):
        if self.state in ("menu", "select", "lobby", "browse"):
            return "menu"
        if self.tod == 2:
            return "playlist"
        if self.state in ("intro", "countdown"):
            return "ambience"
        return "race"

    def update(self, rdt):
        self.toast_t = max(0.0, self.toast_t - rdt)
        self.flash = max(0.0, self.flash - rdt * 2.5)
        self.slowmo = max(0.0, self.slowmo - rdt)
        dt = rdt * (0.3 if self.slowmo > 0 else 1.0)
        self.t += dt
        self.msg_t = max(0.0, self.msg_t - rdt)
        self.fade = min(1.0, self.fade + rdt * 2.0)
        self.fx.concert = self.tod == 2
        self.fx.update(dt, self.t)
        self.music.set_mode(self.music_mode())
        self.music.update(rdt)
        if self.net:
            self.net_update(rdt)
        if self.state == "browse" and self.disc:
            self.found = self.disc.poll()
        if self.state in ("select", "lobby"):
            self.sound.stop_engine()
            self.update_camera(rdt)
            return
        if self.state == "intro":
            self.intro_t += rdt
            if self.intro_t >= self.intro_total:
                self.end_intro()
        if self.state == "countdown" and self.mp != "guest":
            self.countdown -= rdt
            n = int(self.countdown)
            if n < self.last_count and n >= 1:
                self.sound.play("beep")
            self.last_count = n
            if self.countdown <= 0:
                self.state = "race"
                self.go_t = self.t
                self.sound.play("go")
                self.show("¡ARRANCA!", 1.2)
        if self.state in ("race", "finished"):
            self.race_time += dt
        if self.state == "finished":
            self.results_t += rdt
        tr = self.track
        boost, slow = DIFFS[self.diff][3] if self.state != "menu" else (0.0, 0.0)
        for car in self.cars:
            car.rb = 1.0
            if not car.is_player and self.state in ("race", "finished") and getattr(car, "owner", "b") == "b":
                gap = (self.lead_human_u() if self.mp else self.player.u) - car.u
                car.rb = 1.0 + boost if gap > 120 else (1.0 - slow if gap < -120 else 1.0)
        for car in self.cars:
            if getattr(car, "remote", False):
                self.remote_follow(car, rdt)
                if car.boosting:
                    self.fx.nitro_flame(car, self.exhausts(car))
                continue
            if self.state in ("countdown", "intro"):
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
                if car.is_player:
                    self.shake = max(self.shake, min(1.0, strength * 0.06))
            if impact > 4 and car.is_player and self.state != "menu":
                self.sound.play("crash")
            if car.boosting:
                self.fx.nitro_flame(car, self.exhausts(car))
                if car.is_player and car.boost_start and self.state == "race":
                    self.sound.play("whoosh")
        self.collide_cars()
        L = tr.L
        for car in self.cars:
            if getattr(car, "remote", False):
                continue
            ds = car.s_new - car.s
            ds -= L if ds > L / 2 else 0
            ds += L if ds < -L / 2 else 0
            car.s = car.s_new
            car.u += ds
            car.wrong = car.wrong + dt if (ds < -1e-3 and abs(car.vf) > 1) else max(0.0, car.wrong - 2 * dt)
            if self.state not in ("race", "finished"):
                continue
            li = math.floor(car.u / L)
            if li > car.lap_idx:
                if li >= 1 and not car.finished:
                    car.last_lap = self.race_time - car.lap_start
                    car.best_lap = car.last_lap if car.best_lap is None else min(car.best_lap, car.last_lap)
                    car.lap_start = self.race_time
                    if car.is_player and li < LAPS:
                        self.show("VUELTA FINAL" if li == LAPS - 1 else f"VUELTA {li + 1}", 2.0)
                        self.sound.play("beep")
                car.lap_idx = li
            if not car.finished and car.u >= LAPS * L:
                car.finished, car.finish_time = True, self.race_time
                if car.is_player:
                    self.state = "finished"
                    self.results_t = 0.0
                    self.sound.play("go")
                    self.show("¡META!", 2.5)
                    self.fx.confetti(car.x, car.z)
        self.update_camera(rdt)
        if self.state in ("menu", "browse"):
            self.sound.stop_engine()
        elif self.state == "intro":
            self.sound.update_engine(25 + 18 * math.sin(self.t * 2.3), True)
        else:
            self.sound.update_engine(self.player.vf, self.player.throttle)

    def exhausts(self, car):
        return [(e[0], e[1], e[2]) for e in car.vis["info"]["exhaust"]]

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
                ra, rb_ = getattr(a, "remote", False), getattr(b, "remote", False)
                if ra and rb_:
                    continue
                ia, ib = (0.0 if ra else 1 / a.mass), (0.0 if rb_ else 1 / b.mass)
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
                    if (a.is_player or b.is_player) and self.state == "race":
                        p, o = (a, b) if a.is_player else (b, a)
                        if -rel > 3:
                            self.sound.play("crash")
                            self.shake = max(self.shake, min(1.0, -rel * 0.08))
                        # TAKEDOWN: golpe fuerte del jugador
                        pv = p.vx * (o.x - p.x) + p.vz * (o.z - p.z)
                        if -rel > 9 and pv > 0 and o.crash_t <= 0:
                            spin = random.choice((-1, 1)) * (3.5 + min(6.0, -rel * 0.25))
                            o.crash_t = 3.0
                            if getattr(o, "remote", False):
                                if self.net:          # el coche es de la otra computadora: se lo avisamos
                                    for _ in range(3):
                                        self.net.send({"t": "hit", "slot": o.slot, "spin": spin})
                            else:
                                o.spin = spin
                                o.vx *= 0.55
                                o.vz *= 0.55
                            p.nitro = min(1.0, p.nitro + 0.35)
                            self.takedowns += 1
                            self.show("¡TAKEDOWN!", 2.2)
                            if not self.mp:
                                self.slowmo = 0.7
                            self.flash = 0.35

    def standings(self):
        return sorted(self.cars, key=lambda c: (0, c.finish_time) if c.finished else (1, -c.u))

    # ------------------------------------------------------------ camaras
    def update_camera(self, dt):
        self.cam = self.camera(dt)

    def chase_pose(self, p, yaw, far=1.0):
        dx, dz = math.cos(yaw), math.sin(yaw)
        d = p.cam_d * 0.82 * far
        eye = (p.x - dx * d, p.cam_h * 0.72 * (0.85 + 0.15 * far), p.z - dz * d)
        target = (p.x + dx * 5, 1.15 * p.size, p.z + dz * 5)
        return eye, target

    def camera(self, dt):
        p = self.player
        self.shake = max(0.0, self.shake - dt * 1.8)
        if self.state in ("select", "lobby"):
            sz = VEH[PILOTS[self.sel]["vehicle"]]["length"] / 4.5
            a = 0.55 + 0.25 * math.sin(self.t * 0.25)
            r = 5.5 + 3.2 * sz
            self.focus = r
            return (r * math.cos(a), 1.6 + 0.6 * sz, r * math.sin(a)), (0.0, 0.85 * sz, 0.4), 40
        if self.state == "intro":
            cam = self.intro_cam()
            if cam:
                return cam
        if self.state in ("menu", "browse"):
            return self.tv_camera(dt)
        self.cam_yaw += wrap_angle(p.ang - self.cam_yaw) * min(1.0, dt * 5.0)
        if self.cam_mode in (0, 1):
            eye, target = self.chase_pose(p, self.cam_yaw, 1.0 if self.cam_mode == 0 else 1.4)
            eye = np.array(eye)
            for c in self.cars:            # no meter la camara dentro de otro vehiculo (p. ej. un camion detras)
                if c is not p:
                    d = math.hypot(eye[0] - c.x, eye[2] - c.z)
                    lim = c.radius * 2.2 * max(1.0, c.size)
                    if d < lim:
                        eye[1] = max(eye[1], (3.3 * c.size + 1.2) * (1 - d / lim) + eye[1] * d / lim)
            if self.cam_pos is not None:
                eye = self.cam_pos + (eye - self.cam_pos) * min(1.0, dt * 9)
            self.cam_pos = eye
        else:
            dx, dz = math.cos(p.ang), math.sin(p.ang)
            if self.cam_mode == 2:
                eye = np.array([p.x + dx * 0.15 * p.size, p.hood, p.z + dz * 0.15 * p.size])
            else:
                eye = np.array([p.x + dx * 2.45 * p.size, 0.62, p.z + dz * 2.45 * p.size])
            self.cam_pos = None
            target = (eye[0] + dx * 20, eye[1] - 0.4, eye[2] + dz * 20)
        sh = self.shake + (0.25 if p.boosting else 0.0) + (0.08 if p.surface else 0.0)
        if sh > 0:
            eye = eye + np.array([math.sin(self.t * 47), math.sin(self.t * 61) * 0.6, math.cos(self.t * 53)]) * 0.12 * sh
        self.fov_b = getattr(self, "fov_b", 0.0)
        self.fov_b += ((1.0 if p.boosting else 0.0) - self.fov_b) * min(1.0, dt * 4)
        fov = 58 + min(18.0, abs(p.vf) * 0.2) + 12 * self.fov_b
        self.focus = 10.0
        return tuple(eye), target, fov

    def tv_camera(self, dt):
        self.tv_t -= dt
        if self.tv_t <= 0:
            self.tv_t = 6.0
            self.tv_target = random.randrange(len(self.cars))
            self.tv_mode = random.choice(("tv", "tv", "chase", "heli"))
        c = self.cars[self.tv_target]
        cams = self.info["tvcams"]
        if getattr(self, "tv_mode", "tv") == "tv" and cams:
            pos = min(cams, key=lambda q: (q[0] - c.x) ** 2 + (q[2] - c.z) ** 2)
            d = math.hypot(pos[0] - c.x, pos[2] - c.z)
            if d < 260:
                self.focus = d
                fov = clamp(math.degrees(2 * math.atan(7.0 / max(d, 1))), 8, 55)
                return pos, (c.x, 1.0, c.z), fov
        if getattr(self, "tv_mode", "") == "heli":
            a = self.t * 0.15
            eye = (c.x + math.cos(a) * 40, 22, c.z + math.sin(a) * 40)
            self.focus = 45
            return eye, (c.x, 0.5, c.z), 45
        eye, tgt = self.chase_pose(c, c.ang, 1.2)
        self.focus = 10
        return eye, tgt, 62

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
            e0, e1 = g + fw * 200 + rt * 120, g + fw * 32 + rt * 24
            ex, ez = e0 + (e1 - e0) * e
            tg = (g + fw * 70) + ((g - fw * 4) - (g + fw * 70)) * e
            self.focus = 80
            return (ex, 105 + (6 - 105) * e, ez), (tg[0], 1.0, tg[1]), 52
        if kind == "concert":
            S, ff, rr = self.info["stage"]
            a = -0.7 + 1.4 * e
            dx, dz = ff[0] * math.cos(a) + rr[0] * math.sin(a), ff[1] * math.cos(a) + rr[1] * math.sin(a)
            dd = 60 - 16 * e
            self.focus = dd
            return (S[0] + dx * dd, 18 - 8 * e, S[1] + dz * dd), (S[0], 15.0 + 6 * e, S[1]), 62
        if kind == "car":
            th = math.radians(28 + 55 * e)
            d = (6.4 - 1.4 * e) * max(1.0, car.size)
            fx, fz = math.cos(car.ang), math.sin(car.ang)
            rx, rz = -fz, fx
            side = car.grid_side
            ex = car.x + (fx * math.cos(th) + rx * side * math.sin(th)) * d
            ez = car.z + (fz * math.cos(th) + rz * side * math.sin(th)) * d
            ey = (0.8 + 0.5 * e) * max(1.0, car.size * 0.9)
            self.focus = d
            return (ex, ey, ez), (car.x + fx * 0.4, 0.9 * max(1.0, car.size * 0.9), car.z + fz * 0.4), 42
        eye1, tgt1 = self.chase_pose(car, car.ang)
        fx, fz = math.cos(car.ang), math.sin(car.ang)
        eye0 = (car.x - fx * 45 + fz * 14, 24, car.z - fz * 45 - fx * 14)
        tgt0 = (car.x + fx * 20, 0.5, car.z + fz * 20)
        self.focus = 12
        return lerp(eye0, eye1, e), lerp(tgt0, tgt1, e), 60

    # ------------------------------------------------------------ luces dinamicas
    def gather_lights(self, eye, tod):
        L = []
        ex, ey, ez = eye
        if self.state in ("select", "lobby"):
            L = [(4.0, 6.0, 4.0, 30.0, 1.0, 0.85, 0.7, 1.3), (-4.0, 5.0, -3.0, 30.0, 0.5, 0.7, 1.0, 0.9),
                 (0.0, 9.0, -6.0, 30.0, 1.0, 1.0, 1.0, 0.9), (-6.0, 2.0, 5.0, 25.0, 0.3, 0.6, 1.0, 0.8)]
        else:
            lamps = tod["lamps"]
            if lamps > 0 and len(self.static_lights):
                sl = self.static_lights
                d = np.sqrt((sl[:, 0] - ex) ** 2 + (sl[:, 2] - ez) ** 2) - sl[:, 3]
                idx = np.argsort(d)[:14]
                for i in idx:
                    if d[i] < 220:
                        x, y, z, r, cr, cg, cb, it = sl[i]
                        L.append((x, y, z, r, cr, cg, cb, it * lamps))
            for c in sorted(self.cars, key=lambda c: (c.x - ex) ** 2 + (c.z - ez) ** 2)[:4]:
                fx, fz = math.cos(c.ang), math.sin(c.ang)
                if lamps > 0:
                    L.append((c.x + fx * 9, 1.3, c.z + fz * 9, 24.0, 1.0, 0.93, 0.8, 1.5 * lamps))
                if c.boosting:
                    L.append((c.x - fx * 3, 0.8, c.z - fz * 3, 11.0, 0.3, 0.55, 1.0, 4.0))
            for x, y, z, col, life, inten in self.fx.flashes[-5:]:
                L.append((x, y, z, 30 + inten * 5, col[0], col[1], col[2], inten * life * 2.0))
            if self.tod == 2:
                S, ff, rr = self.info["stage"]
                if math.hypot(S[0] - ex, S[1] - ez) < 520:
                    for k in range(3):
                        a = (k - 1) * 18
                        hue = self.t * 0.6 + k * 2.1
                        col = (0.5 + 0.5 * math.cos(hue), 0.5 + 0.5 * math.cos(hue + 2.1), 0.5 + 0.5 * math.cos(hue + 4.2))
                        L.append((S[0] + rr[0] * a + ff[0] * 25, 14.0, S[1] + rr[1] * a + ff[1] * 25, 55.0, *col, 5.0))
                    if self.t - self.fx.fireball_t < 0.8:
                        L.append((S[0] + ff[0] * 5, 18.0, S[1] + ff[1] * 5, 120.0, 1.0, 0.5, 0.15,
                                  12.0 * (1 - (self.t - self.fx.fireball_t) / 0.8)))
        L.sort(key=lambda l: (l[0] - ex) ** 2 + (l[1] - ey) ** 2 + (l[2] - ez) ** 2)
        L = L[:48]
        self.L_pos = np.array([l[0:4] for l in L], F4).reshape(-1, 4)
        self.L_col = np.array([l[4:8] for l in L], F4).reshape(-1, 4)
        self._lkey = None
        return self.light_bytes(None, 0)

    def light_sets(self, centers, radii):
        """Para cada objeto (centro, radio) los indices de las luces que lo alcanzan."""
        P = self.L_pos
        if len(P) <= 10 or len(centers) == 0:
            return None
        D = np.sqrt(((centers[:, None, :] - P[None, :, 0:3]) ** 2).sum(-1))
        M = D < (P[None, :, 3] + radii[:, None])
        return [np.nonzero(row)[0][:MAXL] for row in M]

    def apply_set(self, prog, idx):
        if idx is None:
            idx = np.arange(min(len(self.L_pos), MAXL))
        key = (id(prog), idx.tobytes())
        if key == getattr(self, "_lkey", None):
            return
        pos = np.zeros((MAXL, 4), F4)
        col = np.zeros((MAXL, 4), F4)
        pos[:len(idx)] = self.L_pos[idx]
        col[:len(idx)] = self.L_col[idx]
        setu(prog, u_lpos=pos.tobytes(), u_lcol=col.tobytes(), u_nl=len(idx))
        self._lkey = key

    def light_bytes(self, c, r, prog=None):
        """Solo las luces que alcanzan este objeto (c, r); None = las mas cercanas a la camara."""
        P, C = self.L_pos, self.L_col
        if c is None or len(P) <= 10:
            idx = np.arange(min(len(P), MAXL))
        else:
            d = np.linalg.norm(P[:, 0:3] - np.asarray(c, F4), axis=1)
            idx = np.nonzero(d < P[:, 3] + r)[0][:MAXL]
        pos = np.zeros((MAXL, 4), F4)
        col = np.zeros((MAXL, 4), F4)
        pos[:len(idx)] = P[idx]
        col[:len(idx)] = C[idx]
        if prog is not None:
            key = (id(prog), tuple(idx.tolist()))
            if key != getattr(self, "_lkey", None):
                setu(prog, u_lpos=pos.tobytes(), u_lcol=col.tobytes(), u_nl=len(idx))
                self._lkey = key
            return None
        return (pos.tobytes(), col.tobytes()), len(idx)

    # ------------------------------------------------------------ dibujo 3D
    def car_models(self, cars):
        out = []
        for c in cars:
            vis = c.vis
            M = model_full(c.x, 0.0, c.z, c.ang, c.roll, c.pitch)
            r = vis["info"]["radius"]
            wheels = []
            for wx, wz in vis["info"]["wheels"]:
                for side in (-1, 1):
                    W_ = M @ mat_T(wx, r, side * wz)
                    if wx > 0:
                        W_ = W_ @ mat_Ry(-c.steer * 0.45)
                    if side < 0:
                        W_ = W_ @ MIRROR_Z
                    wheels.append(W_ @ mat_Rz(-c.wheel_rot))
            out.append((c, vis, M, wheels))
        return out

    def draw_cars(self, prog, items, shadow=False, headl=0.0):
        for c, vis, M, wheels in items:
            p = vis["pilot"]
            if not shadow and hasattr(self, "L_pos"):
                self.light_bytes((c.x, 1.0, c.z), 6.0, prog)
            if shadow:
                setu(prog, u_model=gl_bytes(M))
                vis["body"][1].render()
                for W_ in wheels:
                    setu(prog, u_model=gl_bytes(W_))
                    vis["wheel"][1].render()
                continue
            setu(prog, u_model=gl_bytes(M), u_paint2=rgb(*p["accent"]),
                 u_stripe=1.0 if p["vehicle"] in ("sport", "hatch") else 0.0,
                 u_brake=1.0 if getattr(c, "braking", False) else 0.0, u_headl=headl)
            vis["body"][0].render()
            setu(prog, u_brake=0.0)
            for W_ in wheels:
                setu(prog, u_model=gl_bytes(W_))
                vis["wheel"][0].render()
        setu(prog, u_model=gl_bytes(np.identity(4, F4)), u_brake=0.0, u_stripe=0.0)

    def draw_glass(self, prog, items):
        setu(prog, u_glass_alpha=0.55)
        for c, vis, M, wheels in items:
            if hasattr(self, "L_pos"):
                self.light_bytes((c.x, 1.0, c.z), 6.0, prog)
            setu(prog, u_model=gl_bytes(M))
            vis["glass"].render()
        setu(prog, u_model=gl_bytes(np.identity(4, F4)))

    def showroom_items(self):
        p = PILOTS[self.sel]
        vis = self.vehicles[p["name"]]

        class _C:
            pass
        c = _C()
        c.x, c.z, c.ang = 0.0, 0.0, self.t * 0.35
        c.roll = c.pitch = 0.0
        c.steer = 0.35 * math.sin(self.t * 0.4)
        c.wheel_rot = 0.0
        c.braking = False
        c.vis = vis
        return self.car_models([c]), vis

    def render(self):
        R, ctx = self.R, self.ctx
        tod = self.tod_params()
        eye, target, fov = self.cam
        proj = perspective(fov, WIN_W / WIN_H, 0.5, 9000.0)
        view = look_at(eye, target)
        vp = proj @ view
        studio = self.state in ("select", "lobby")
        lights_b, nl = self.gather_lights(eye, tod)
        night = self.tod == 2 and not studio
        startl = 0.0
        if self.state == "countdown":
            startl = clamp(4.0 - self.countdown, 0, 5) * 5 / 3
        go = 1.0 if (self.state == "race" and self.t - self.go_t < 3.0) else 0.0
        headl = 1.0 if tod["lamps"] > 0 else 0.0
        extra = dict(concert=1.0 if night else 0.0, studio=1.0 if studio else 0.0,
                     stage=tuple(map(float, self.info["stage"][0])), startl=startl, go=go,
                     pit=self.info["pit"], headl=headl)
        if studio:
            items, vis = self.showroom_items()
        else:
            items = self.car_models(self.cars)
            vis = None
        fwd = norm(np.asarray(target, F4) - np.asarray(eye, F4))
        tanh_ = math.tan(math.radians(fov) / 2)
        cosl = math.cos(min(1.5, math.atan(tanh_ * WIN_W / WIN_H * 1.05) + 0.12))
        # ---- sombras
        if R.shs and tod["shadows"] > 0.5 and not studio:
            sun = np.array(tod["sun_n"], F4)
            ahead = np.asarray(eye, F4) + fwd * 70
            snap = np.round(ahead / 40.0) * 40.0
            key = (tuple(snap.tolist()), self.tod)
            if key != R.static_key:
                R.lvp_s = R.light_matrix((snap[0], 0.0, snap[2]), sun, 190.0, 700.0)
                R.shadow_pass(R.shs, R.lvp_s, lambda prog: self.static.draw_shadow((snap[0], 0.0, snap[2]), 190.0))
                R.static_key = key
            c0 = self.player if self.state != "menu" else self.cars[getattr(self, "tv_target", 0)]
            R.lvp_d = R.light_matrix((c0.x, 0.0, c0.z), sun, 48.0, 260.0)
            R.shadow_pass(R.shd, R.lvp_d, lambda prog: self.draw_cars(prog, items, shadow=True))
        # ---- efectos (se construyen una vez y se usan en reflejo y escena)
        beams = concert_beams_hd(self.t, self.info) if night else []
        self.fx.build(view, eye, beams)
        # ---- reflejo planar del piso mojado
        if R.refl and (tod["wet"] > 0.05 or studio):
            rt, rd, rf = R.refl
            rf.use()
            rf.clear(0, 0, 0, 1, depth=1.0)
            eye_r = (eye[0], -eye[1], eye[2])
            vp_r = proj @ view @ MIRROR_Y
            ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
            R.sky(vp_r, eye_r, tod, self.t, 1.0 if studio else 0.0)
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.enable_direct(0x3000)
            R.bind_textures(True)
            R.common(R.p_scene, vp_r, eye_r, tod, self.t, True, lights_b, nl, extra)
            if studio:
                self.studio.render()
            else:
                self._lkey = None
                sets = self.light_sets(self.static.C, self.static.Rr)
                self.static.draw(eye_r, fwd * np.array([1, -1, 1], F4), cosl, 1400.0,
                                 pre=lambda i, r: self.apply_set(R.p_scene, None if (sets is None or i is None) else sets[i]))
                if night:
                    self.light_bytes(self.info["stage"][0].tolist() and (self.info["stage"][0][0], 10.0,
                                     self.info["stage"][0][1]), 140.0, R.p_scene)
                    self.concert[0].render()
            self.draw_cars(R.p_scene, items, headl=headl)
            if studio:
                setu(R.p_scene, u_model=gl_bytes(mat_T(2.6, 0, -2.3) @ mat_Ry(-0.55)))
                vis["figure"].render()
                setu(R.p_scene, u_model=gl_bytes(np.identity(4, F4)))
            ctx.disable_direct(0x3000)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            rf.depth_mask = False
            setu(R.p_fx, u_vp=gl_bytes(vp_r), u_cam=eye_r, u_fog_den=tod["fog_den"], u_time=float(self.t))
            self.fx.render()
            rf.depth_mask = True
            ctx.disable(moderngl.BLEND)
        # ---- escena principal HDR
        R.fbo_hdr.use()
        R.fbo_hdr.clear(0, 0, 0, 1, depth=1.0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        R.sky(vp, eye, tod, self.t, 1.0 if studio else 0.0)
        ctx.enable(moderngl.DEPTH_TEST)
        R.bind_textures(False)
        R.common(R.p_scene, vp, eye, tod, self.t, False, lights_b, nl, extra)
        if studio:
            self.light_bytes(None, 0, R.p_scene)
            self.studio.render()
            setu(R.p_scene, u_model=gl_bytes(mat_T(2.6, 0, -2.3) @ mat_Ry(-0.55)))
            vis["figure"].render()
            setu(R.p_scene, u_model=gl_bytes(np.identity(4, F4)))
        else:
            # pre-pasada de profundidad: el shader caro calcula cada pixel una sola vez
            vis_ = self.static.visible(eye, fwd, cosl, 2600.0 if self.tod == 0 else 1800.0)
            R.fbo_hdr.color_mask = (False, False, False, False)
            setu(R.p_depth, u_vp=gl_bytes(vp), u_model=gl_bytes(np.identity(4, F4)), u_time=float(self.t), u_clip=0.0,
                 u_cam=tuple(map(float, eye)))
            self.static.draw(eye, fwd, cosl, 0, depth=True, vis=vis_)
            if night:
                self.concert[2].render()
            R.fbo_hdr.color_mask = (True, True, True, True)
            ctx.depth_func = "<="
            self._lkey = None
            sets = self.light_sets(self.static.C, self.static.Rr)
            self.static.draw(eye, fwd, cosl, 0, vis=vis_,
                             pre=lambda i, r: self.apply_set(R.p_scene, None if (sets is None or i is None) else sets[i]))
            if night:
                self.light_bytes((self.info["stage"][0][0], 10.0, self.info["stage"][0][1]), 140.0, R.p_scene)
                self.concert[0].render()
            ctx.depth_func = "<"
            R.common(R.p_crowd, vp, eye, tod, self.t, False, lights_b, nl, extra)
            self._lkey = None
            if not hasattr(self.crowd, "C"):
                self.crowd.draw(eye, fwd, R.quality, night, 0.0, pre=None) if False else None
                self.crowd.C = np.array([ch["c"] for ch in self.crowd.chunks], F4).reshape(-1, 3)
                self.crowd.Rr = np.array([ch["r"] for ch in self.crowd.chunks], F4)
                self.crowd.flag = np.array([ch["concert"] for ch in self.crowd.chunks], bool)
            csets = self.light_sets(self.crowd.C, self.crowd.Rr)
            self.crowd_drawn = self.crowd.draw(eye, fwd, R.quality, night, math.tan(math.radians(fov) / 2),
                                               pre=lambda i, r: self.apply_set(R.p_crowd, None if csets is None else csets[i]))
            self._lkey = None
        self.draw_cars(R.p_scene, items, headl=headl)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        R.fbo_hdr.depth_mask = False
        self.draw_glass(R.p_scene, items)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        setu(R.p_fx, u_vp=gl_bytes(vp), u_cam=tuple(map(float, eye)), u_fog_den=tod["fog_den"], u_time=float(self.t))
        self.fx.render()
        R.fbo_hdr.depth_mask = True
        ctx.disable(moderngl.BLEND)
        # ---- post
        p = self.player
        spd = abs(p.vf) * 3.6 if self.state in ("race", "finished", "countdown") else 0.0
        boost = 1.0 if (p.boosting and self.state == "race") else 0.0
        P = dict(exposure=tod["exposure"], bloom=0.065 if not studio else 0.035, streak=0.045, sat=tod["sat"],
                 grade=tod["grade"], time=self.t, vig=0.42 if self.state != "race" else 0.32,
                 radial=clamp((spd - 210) / 2500, 0, 0.022) + 0.035 * boost, ca=0.0007 + 0.004 * boost,
                 dof=1.0 if self.state in ("intro", "select", "menu", "lobby", "browse") or (self.state == "finished" and self.results_t > 2) else 0.0,
                 focus=getattr(self, "focus", 10.0), aperture=0.55 if self.state in ("select", "lobby") else 0.35,
                 fade=smooth(self.fade), flash=self.flash, thresh=1.4 if self.tod else 1.2)
        R.post(P)
        self.draw_hud()
        R.present(self.screen_fbo, self.hud)

    # ------------------------------------------------------------ HUD
    def draw_hud(self):
        self.hud.fill((0, 0, 0, 0))
        st = self.state
        if st == "menu":
            self.hud_menu()
        elif st == "select":
            self.hud_select()
        elif st == "lobby":
            self.hud_lobby()
        elif st == "browse":
            self.hud_browse()
        elif st == "intro":
            self.hud_intro()
        else:
            self.hud_race()
        if self.toast_t > 0:
            a = int(255 * min(1.0, self.toast_t * 2))
            w = font(24).size(self.toast)[0] + 90
            self.panel((WIN_W // 2 - w // 2, 18, w, 42), int(a * 0.7), accent=ACCENT)
            self.txt(self.toast, 24, (255, 255, 255), (WIN_W // 2, 39), "center", alpha=a)

    def hud_lobby(self):
        host = self.mp == "host"
        net = self.net
        p = PILOTS[self.sel]
        self.txt("PARTIDA EN RED LOCAL", 46, (255, 255, 255), (60, 36), glow=ACCENT)
        self.panel((40, 110, 560, 470), 185, accent=ACCENT, slant=24)
        self.txt("ANFITRIÓN" if host else "INVITADO", 30, ORANGE, (80, 122))
        if host:
            self.txt(f"Tu IP:  {net.ip if net else '-'}", 22, (220, 230, 240), (82, 164), italic=False)
            self.txt("El otro jugador debe elegir UNIRSE (tecla J) en la misma red Wi-Fi.", 15,
                     (170, 185, 200), (82, 194), italic=False)
        else:
            self.txt(f"Conectado a {net.peer[0] if net and net.peer else '-'}", 22, (220, 230, 240), (82, 164),
                     italic=False)
        connected = bool(net and net.connected() and (net.peer is not None))
        if connected and self.peer_pilot is not None:
            other = PILOTS[self.peer_pilot]["name"]
            self.txt(f"{'J2' if host else 'J1'}:  {other}", 28, (120, 255, 150), (82, 232))
        else:
            dots = "." * (1 + int(self.t * 2) % 3)
            self.txt(("Esperando invitado" if host else "Esperando al anfitrión") + dots, 26, (255, 215, 60), (82, 232))
        self.txt("TU PILOTO", 16, (170, 185, 200), (82, 290), italic=False)
        self.txt(f"‹  {p['name']}  ›", 40, (255, 255, 255), (82, 310))
        self.txt(p["vname"], 18, (190, 200, 215), (86, 360), italic=False)
        dname = DIFFS[self.diff][0]
        self.txt("DIFICULTAD", 16, (170, 185, 200), (82, 404), italic=False)
        self.txt(dname, 28, (255, 215, 60), (82, 424))
        self.txt("HORARIO", 16, (170, 185, 200), (330, 404), italic=False)
        self.txt(TOD_HD[self.tod][0], 28, (255, 255, 255), (330, 424))
        if host:
            msg = "ENTER  iniciar carrera" if connected else "(ENTER se activa cuando se una el invitado)"
            col = (255, 255, 255) if connected else (150, 160, 175)
            self.txt(msg, 22 if connected else 16, col, (82, 480), italic=connected)
            hint = "← →  piloto     ↑ ↓  dificultad     N  horario     ESC  salir"
        else:
            self.txt("El anfitrión elige dificultad y horario e inicia la carrera.", 16, (170, 185, 200), (82, 480),
                     italic=False)
            hint = "← →  piloto     ESC  salir"
        self.txt(hint, 18, (200, 210, 225), (WIN_W // 2, WIN_H - 26), "center", italic=False, fname="segoeui,arial")

    def hud_browse(self):
        grad = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        grad.fill((0, 0, 0, 120))
        self.hud.blit(grad, (0, 0))
        self.txt("UNIRSE A UNA PARTIDA", 50, (255, 255, 255), (WIN_W // 2, 90), "center", glow=ACCENT)
        self.panel((WIN_W // 2 - 360, 150, 720, 400), 200, accent=ACCENT, slant=24)
        found = getattr(self, "found", [])
        if self.disc and not self.disc.ok:
            self.txt("No se pudo escuchar la red (puerto 47778 ocupado). Escribe la IP del anfitrión.", 16,
                     (255, 150, 120), (WIN_W // 2, 184), "center", italic=False)
        elif not found:
            dots = "." * (1 + int(self.t * 2) % 3)
            self.txt("Buscando partidas en tu red" + dots, 26, (255, 215, 60), (WIN_W // 2, 200), "center")
        self.browse_sel = min(self.browse_sel, max(0, len(found) - 1))
        for i, (ip, (name, _)) in enumerate(found[:5]):
            y = 240 + i * 50
            sel = i == self.browse_sel and not self.ip_typed
            self.panel((WIN_W // 2 - 300, y, 600, 42), 210 if sel else 110, accent=ORANGE if sel else ACCENT, slant=12)
            self.txt(f"{name}", 24, (255, 255, 255), (WIN_W // 2 - 260, y + 7))
            self.txt(ip, 20, (190, 200, 215), (WIN_W // 2 + 270, y + 10), "topright", italic=False)
        self.txt("¿No aparece? Escribe la IP del anfitrión:", 17, (190, 200, 215), (WIN_W // 2 - 300, 492),
                 italic=False)
        cursor = "_" if int(self.t * 2) % 2 == 0 else " "
        self.txt((self.ip_typed or "") + cursor, 26, (255, 255, 255), (WIN_W // 2 + 110, 486))
        self.txt("↑ ↓  elegir     ENTER  unirse     números y punto  IP manual     ESC  volver", 18,
                 (200, 210, 225), (WIN_W // 2, WIN_H - 26), "center", italic=False, fname="segoeui,arial")
        if self.show_fps:
            self.txt(f"{self.fps:4.0f} FPS", 18, (180, 255, 180), (WIN_W // 2, WIN_H - 22), "center", italic=False)

    def song_banner(self, y):
        mu = self.music
        if mu.mode == "playlist" and mu.now and mu.np_t < 7 and not self.sound.muted:
            a = int(255 * clamp(min(mu.np_t / 0.5, (7 - mu.np_t) / 1.0), 0, 1))
            s = f"♪  AVENGED SEVENFOLD  —  {mu.now}  (8-bit)"
            w = font(22).size(s)[0] + 60
            self.panel((WIN_W // 2 - w // 2, y, w, 40), int(a * 0.75), accent=(240, 40, 40))
            self.txt(s, 22, (255, 225, 90), (WIN_W // 2 + 6, y + 20), "center", alpha=a)

    def hud_race(self):
        h = self.hud
        p = self.player
        order = self.standings()
        pos = order.index(p) + 1
        # posicion y vueltas
        self.panel((18, 18, 300, 150), 150, accent=ACCENT)
        self.txt("POS", 18, (170, 200, 220), (44, 28))
        self.txt(f"{pos}", 92, (255, 255, 255), (40, 34), glow=ACCENT if pos == 1 else None)
        self.txt(f"/{len(self.cars)}", 34, (190, 200, 210), (40 + font(92).size(str(pos))[0] + 6, 84))
        lap = min(LAPS, max(0, p.lap_idx) + 1)
        self.txt("VUELTA", 18, (170, 200, 220), (176, 28))
        self.txt(f"{lap}/{LAPS}", 40, (255, 255, 255), (176, 46))
        tm = self.race_time if not p.finished else p.finish_time
        self.txt(fmt_time(tm), 28, (255, 255, 255), (176, 96))
        self.txt(f"MEJOR {fmt_time(p.best_lap)}", 16, (170, 200, 220), (178, 130), italic=False)
        # tabla de posiciones
        y0 = 184
        for i, c in enumerate(order):
            y = y0 + i * 30
            col = (255, 215, 60) if c.is_player else (235, 240, 245)
            self.panel((18, y, 230, 26), 120 if not c.is_player else 190, accent=tuple(c.map_color), slant=8)
            self.txt(f"{i + 1}", 18, col, (40, y + 3))
            self.txt(c.name, 17, col, (66, y + 4))
        self.minimap()
        self.speedometer(p)
        # mensajes
        if self.state == "countdown":
            n = int(self.countdown)
            if n >= 1:
                fr = self.countdown - int(self.countdown)
                size = int(150 + 80 * fr)
                self.txt(str(n), size, (255, 255, 255), (WIN_W // 2, WIN_H // 2 - 40), "center", glow=(255, 60, 40))
        if self.msg_t > 0 and self.message:
            a = int(255 * min(1.0, self.msg_t * 3))
            if self.message == "¡TAKEDOWN!":
                self.panel((WIN_W // 2 - 330, WIN_H // 2 - 110, 660, 120), int(a * 0.55), col=(30, 8, 0),
                           accent=ORANGE, slant=30)
                self.txt("¡TAKEDOWN!", 96, (255, 180, 40), (WIN_W // 2, WIN_H // 2 - 52), "center", glow=ORANGE, alpha=a)
                self.txt("+ NITRO", 26, ACCENT, (WIN_W // 2, WIN_H // 2 + 20), "center", alpha=a)
            else:
                self.txt(self.message, 84, (255, 255, 255), (WIN_W // 2, WIN_H // 2 - 70), "center", glow=ACCENT,
                         alpha=a)
        if self.state == "race" and p.wrong > 1.0 and int(self.t * 3) % 2 == 0:
            self.txt("¡SENTIDO CONTRARIO!", 56, (255, 70, 50), (WIN_W // 2, 210), "center", glow=(255, 0, 0))
        if p.drifting and self.state == "race":
            self.txt("DERRAPE  +NITRO", 22, ACCENT, (WIN_W - 190, WIN_H - 290), "center")
        self.song_banner(WIN_H - 62)
        if self.state == "finished" and self.results_t > 2.0:
            self.results(order)

    def speedometer(self, p):
        h = self.hud
        cx, cy, R = WIN_W - 175, WIN_H - 150, 118
        kmh = abs(p.vf) * 3.6
        a0, a1 = math.radians(215), math.radians(-35)
        pygame.draw.circle(h, (6, 8, 14, 150), (cx, cy), R + 14)
        N = 44
        frac = clamp(kmh / 320, 0, 1)
        for i in range(N):
            f0, f1 = i / N, (i + 0.78) / N
            aa, bb = a0 + (a1 - a0) * f0, a0 + (a1 - a0) * f1
            on = f0 < frac
            if on:
                col = (0, 200, 255) if f0 < 0.55 else ((255, 255, 255) if f0 < 0.72 else ((255, 150, 30) if f0 < 0.86 else (255, 50, 40)))
            else:
                col = (45, 52, 66)
            r0, r1 = R - 16, R
            pts = [(cx + r0 * math.cos(aa), cy - r0 * math.sin(aa)), (cx + r1 * math.cos(aa), cy - r1 * math.sin(aa)),
                   (cx + r1 * math.cos(bb), cy - r1 * math.sin(bb)), (cx + r0 * math.cos(bb), cy - r0 * math.sin(bb))]
            pygame.draw.polygon(h, col + (255,), pts)
        for v in range(0, 321, 40):
            a = a0 + (a1 - a0) * v / 320
            self.txt(str(v), 15, (190, 200, 215), (cx + (R - 32) * math.cos(a), cy - (R - 32) * math.sin(a)), "center",
                     shadow=False, italic=False)
        gear = "R" if p.vf < -0.5 else ("N" if kmh < 2 else str(1 + sum(kmh > g for g in (48, 85, 125, 165, 205))))
        self.txt(f"{int(kmh)}", 66, (255, 255, 255), (cx, cy - 8), "center", glow=ACCENT if p.boosting else None)
        self.txt("KM/H", 16, (160, 190, 210), (cx, cy + 30), "center", italic=False)
        pygame.draw.rect(h, (255, 255, 255, 230), (cx - 20, cy + 50, 40, 40), 2, border_radius=6)
        self.txt(gear, 30, (255, 210, 60), (cx, cy + 70), "center")
        # nitro
        nx, ny, nw = cx - 140, cy + R + 2, 280
        segs = 12
        for i in range(segs):
            f0 = i / segs
            on = f0 < p.nitro
            col = ((150, 230, 255) if p.boosting else (0, 170, 255)) if on else (35, 45, 60)
            x = nx + i * nw / segs
            pygame.draw.polygon(h, col + (255,), [(x + 8, ny), (x + nw / segs - 2, ny), (x + nw / segs - 10, ny + 14),
                                                  (x, ny + 14)])
        self.txt("NITRO", 14, (150, 210, 255), (nx - 52, ny - 1), italic=True)

    def minimap(self):
        h = self.hud
        p = self.player
        size = 230
        cx, cy = WIN_W - size // 2 - 24, size // 2 + 24
        if self.mm_cache is None:
            self.mm_cache = pygame.Surface((size, size), pygame.SRCALPHA)
            mask = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2 - 2)
            self.mm_mask = mask
        s = self.mm_cache
        s.fill((6, 8, 14, 170))
        sc = (size / 2) / 650.0
        ang = -p.ang - math.pi / 2
        ca, sa = math.cos(ang), math.sin(ang)
        pts = self.track.pts[::4]
        rx, rz = pts[:, 0] - p.x, pts[:, 1] - p.z
        X = (rx * ca - rz * sa) * sc + size / 2
        Y = (rx * sa + rz * ca) * sc + size / 2
        poly = list(zip(X.tolist(), Y.tolist()))
        pygame.draw.lines(s, (20, 24, 34, 255), True, poly, 10)
        pygame.draw.lines(s, (210, 220, 235, 255), True, poly, 4)
        sx0 = self.track.pts[0]
        fx_, fz_ = sx0[0] - p.x, sx0[1] - p.z
        pygame.draw.circle(s, (255, 255, 255), ((fx_ * ca - fz_ * sa) * sc + size / 2, (fx_ * sa + fz_ * ca) * sc + size / 2), 5)
        for c in self.cars:
            if c.is_player:
                continue
            ox, oz = c.x - p.x, c.z - p.z
            pygame.draw.circle(s, (0, 0, 0), ((ox * ca - oz * sa) * sc + size / 2, (ox * sa + oz * ca) * sc + size / 2), 7)
            pygame.draw.circle(s, c.map_color, ((ox * ca - oz * sa) * sc + size / 2, (ox * sa + oz * ca) * sc + size / 2), 5)
        m = size / 2
        pygame.draw.polygon(s, (255, 215, 60), [(m, m - 11), (m + 7, m + 8), (m, m + 4), (m - 7, m + 8)])
        s.blit(self.mm_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        h.blit(s, (cx - size // 2, cy - size // 2))
        pygame.draw.circle(h, ACCENT + (255,), (cx, cy), size // 2 - 1, 2)

    def results(self, order):
        panel = pygame.Rect(WIN_W // 2 - 380, 90, 760, 520)
        self.panel(panel, 215, accent=ACCENT, slant=30)
        self.txt("RESULTADOS", 54, (255, 255, 255), (WIN_W // 2, 140), "center", glow=ACCENT)
        for i, c in enumerate(order):
            y = 200 + i * 58
            col = (255, 215, 60) if c.is_player else (235, 240, 245)
            self.panel((panel.x + 50, y, panel.w - 100, 50), 130, accent=tuple(c.map_color), slant=16)
            self.txt(f"{i + 1}", 36, col, (panel.x + 80, y + 4))
            self.txt(c.name, 28, col, (panel.x + 130, y + 2))
            self.txt(c.pilot["brand"], 15, (170, 185, 200), (panel.x + 134, y + 32), italic=False)
            self.txt(fmt_time(c.finish_time) if c.finished else "EN PISTA", 28, col,
                     (panel.right - 80, y + 6), "topright")
        self.txt(f"TAKEDOWNS: {self.takedowns}", 20, ORANGE, (panel.x + 70, panel.bottom - 44))
        self.txt("R  OTRA CARRERA     ESC  MENÚ", 20, (210, 220, 230), (panel.right - 70, panel.bottom - 44), "topright")

    def hud_menu(self):
        h = self.hud
        grad = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for y in range(0, WIN_H, 4):
            a = int(160 * (1 - y / WIN_H) ** 2)
            grad.fill((0, 0, 0, a), (0, y, WIN_W, 4))
        h.blit(grad, (0, 0))
        self.txt("GTC RACING", 120, (255, 255, 255), (WIN_W // 2 - 40, 150), "center", glow=ACCENT)
        self.panel((WIN_W // 2 + 290, 108, 110, 70), 255, col=(0, 180, 235), slant=16)
        self.txt("HD", 54, (255, 255, 255), (WIN_W // 2 + 348, 142), "center", shadow=False)
        self.txt("GRAN PREMIO CDMX", 36, ACCENT, (WIN_W // 2, 240), "center")
        if int(self.t * 1.6) % 2 == 0:
            self.txt("PRESIONA ENTER", 34, (255, 255, 255), (WIN_W // 2, WIN_H - 250), "center", glow=(255, 255, 255))
        self.panel((WIN_W // 2 - 330, WIN_H - 196, 660, 54), 170, accent=ORANGE, slant=18)
        self.txt("MULTIJUGADOR LAN:   H  crear partida      J  unirse", 22, (255, 255, 255),
                 (WIN_W // 2 + 6, WIN_H - 169), "center")
        self.panel((0, WIN_H - 120, WIN_W, 120), 150, slant=0)
        rows = ["FLECHAS/WASD  conducir      ESPACIO  derrapar      SHIFT  nitro      C  cámara",
                f"N  horario: {TOD_HD[self.tod][0]}      F1  calidad: {QUALITY[self.qi][0]}      CTRL  canción (noche)"
                "      M  sonido      ESC  salir"]
        for i, r in enumerate(rows):
            self.txt(r, 19, (210, 220, 232), (WIN_W // 2, WIN_H - 88 + i * 34), "center", italic=False)

    def hud_select(self):
        p = PILOTS[self.sel]
        self.txt("ELIGE TU PILOTO", 52, (255, 255, 255), (60, 40), glow=ACCENT)
        self.panel((40, 120, 520, 330), 175, accent=tuple(p["map"]), slant=24)
        self.txt(p["name"], 60, (255, 255, 255), (80, 130))
        self.txt(p["brand"].upper(), 26, tuple(p["map"]) if p["map"] != (255, 255, 255) else (210, 220, 235),
                 (84, 205))
        self.txt(p["vname"], 20, (190, 200, 215), (86, 242), italic=False)
        stats = [("VELOCIDAD", p["vmax"], 84, 93), ("ACELERACIÓN", p["accel"], 10.0, 12.5),
                 ("AGARRE", p["grip"], 24.5, 27.5), ("PESO", p["mass"], 0.8, 1.7)]
        for i, (lab, v, lo, hi) in enumerate(stats):
            y = 290 + i * 38
            self.txt(lab, 17, (200, 210, 225), (84, y), italic=False)
            fill = clamp((v - lo) / (hi - lo), 0.1, 1.0)
            for k in range(10):
                on = k / 10 < fill
                x = 250 + k * 28
                pygame.draw.polygon(self.hud, (ACCENT if on else (45, 52, 66)) + (255,),
                                    [(x + 6, y), (x + 26, y), (x + 20, y + 18), (x, y + 18)])
        dname, _, _, _, ddesc, _ = DIFFS[self.diff]
        dcol = [(90, 230, 120), (255, 215, 60), (255, 140, 30), (255, 60, 60)][self.diff]
        self.panel((40, 470, 520, 120), 175, accent=dcol, slant=24)
        self.txt("DIFICULTAD", 17, (190, 200, 215), (84, 482), italic=False)
        self.txt(f"‹ {dname} ›", 36, dcol, (84, 502))
        self.txt(ddesc, 17, (190, 200, 215), (84, 548), italic=False)
        self.txt("HORARIO", 17, (190, 200, 215), (340, 482), italic=False)
        self.txt(TOD_HD[self.tod][0], 30, (255, 255, 255), (340, 506))
        n = len(PILOTS)
        cw = 170
        x0 = WIN_W // 2 - n * cw // 2 + 250
        for i, q in enumerate(PILOTS):
            r = (x0 + i * (cw - 30) - 200, WIN_H - 92, cw - 40, 54)
            sel = i == self.sel
            self.panel(r, 210 if sel else 120, accent=tuple(q["map"]), slant=14)
            self.txt(q["name"], 15, (255, 255, 255) if sel else (160, 170, 185), (r[0] + r[2] // 2 + 4, r[1] + 27),
                     "center")
        self.txt("← →  piloto     ↑ ↓  dificultad     N  horario     ENTER  correr     ESC  volver", 18,
                 (200, 210, 225), (WIN_W // 2, WIN_H - 22), "center", italic=False, fname="segoeui,arial")

    def hud_intro(self):
        h = self.hud
        shot, lt = self.current_shot()
        if shot is None:
            return
        kind, dur, car = shot
        bars = int(86 * smooth(self.intro_t / 0.7))
        pygame.draw.rect(h, (0, 0, 0, 255), (0, 0, WIN_W, bars))
        pygame.draw.rect(h, (0, 0, 0, 255), (0, WIN_H - bars, WIN_W, bars))
        if kind == "aerial":
            a = int(255 * smooth((lt - 0.6) / 0.8))
            self.txt("GRAN PREMIO CDMX", 92, (255, 255, 255), (WIN_W // 2, WIN_H // 2 + 120), "center", glow=ACCENT,
                     alpha=a)
            self.txt(f"AUTÓDROMO  ·  {self.track.L / 1000:.1f} KM  ·  {LAPS} VUELTAS  ·  {DIFFS[self.diff][0]}  ·  "
                     f"{TOD_HD[self.tod][0]}", 26, ACCENT, (WIN_W // 2, WIN_H // 2 + 190), "center", alpha=a)
        elif kind == "concert":
            a = int(255 * smooth((lt - 0.3) / 0.6))
            self.txt("ESTADIO GNP SEGUROS", 30, (255, 255, 255), (WIN_W // 2, WIN_H // 2 + 150), "center", alpha=a)
            self.txt("AVENGED SEVENFOLD  ·  EN VIVO", 70, (255, 70, 50), (WIN_W // 2, WIN_H // 2 + 205), "center",
                     glow=(255, 30, 0), alpha=a)
        elif kind == "car":
            slide = smooth(lt / 0.35)
            x = int(-560 + 620 * slide)
            self.panel((x, WIN_H - 270, 560, 150), 200, accent=tuple(car.map_color), slant=30)
            self.txt(car.name, 58, (255, 255, 255), (x + 50, WIN_H - 262))
            self.txt(f"{car.pilot['brand']}  ·  {car.pilot['vname']}", 20, (200, 215, 230), (x + 54, WIN_H - 188),
                     italic=False)
            self.txt(f"SALIDA  P{car.slot + 1}", 18, ACCENT, (x + 54, WIN_H - 158))
            if car.is_player:
                self.txt("TÚ", 40, (255, 215, 60), (x + 500, WIN_H - 256), "topright", glow=(255, 180, 0))
        else:
            a = int(255 * smooth(lt / 0.5))
            self.txt("¡PREPÁRATE!", 100, (255, 255, 255), (WIN_W // 2, WIN_H // 2 - 60), "center", glow=ACCENT, alpha=a)
        self.txt("ENTER  saltar", 17, (190, 200, 215), (WIN_W - 24, WIN_H - 34), "topright", italic=False)
        self.song_banner(24)

    # ------------------------------------------------------------ entrada
    IP_KEYS = {**{getattr(pygame, f"K_{i}"): str(i) for i in range(10)},
               **{getattr(pygame, f"K_KP{i}"): str(i) for i in range(10)},
               pygame.K_PERIOD: ".", pygame.K_KP_PERIOD: "."}

    def handle_key(self, key):
        st = self.state
        if key == pygame.K_ESCAPE:
            if st == "menu":
                return False
            if self.mp or self.disc:
                self.leave_mp(silent=True)
            self.new_race(attract=True)
            return True
        elif st == "menu":
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.state = "select"
                self.fade = 0.0
            elif key == pygame.K_h:
                self.start_host()
            elif key == pygame.K_j:
                self.start_browse()
        elif st == "browse":
            found = getattr(self, "found", [])
            if key in self.IP_KEYS and len(self.ip_typed) < 15:
                self.ip_typed += self.IP_KEYS[key]
            elif key == pygame.K_BACKSPACE:
                self.ip_typed = self.ip_typed[:-1]
            elif key in (pygame.K_UP, pygame.K_w):
                self.browse_sel = max(0, self.browse_sel - 1)
            elif key in (pygame.K_DOWN, pygame.K_s):
                self.browse_sel = min(max(0, len(found) - 1), self.browse_sel + 1)
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self.ip_typed:
                    self.join(self.ip_typed)
                elif found:
                    self.join(found[self.browse_sel][0])
            return True
        elif st == "lobby":
            if key in (pygame.K_LEFT, pygame.K_a):
                self.sel = (self.sel - 1) % len(PILOTS)
            elif key in (pygame.K_RIGHT, pygame.K_d):
                self.sel = (self.sel + 1) % len(PILOTS)
            elif self.mp == "host" and key in (pygame.K_UP, pygame.K_w):
                self.diff = min(len(DIFFS) - 1, self.diff + 1)
            elif self.mp == "host" and key in (pygame.K_DOWN, pygame.K_s):
                self.diff = max(0, self.diff - 1)
            elif self.mp == "host" and key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self.net and self.net.peer and self.net.connected():
                    self.start_mp_race()
                else:
                    self.toast, self.toast_t = "TODAVÍA NO SE UNE NADIE", 2.0
            elif key == pygame.K_n and self.mp == "host":
                self.tod = (self.tod + 1) % len(TOD_HD)
            if key == pygame.K_n:
                return True
        elif self.mp and key == pygame.K_r:
            if self.mp == "host":
                self.start_mp_race()
            else:
                self.toast, self.toast_t = "SOLO EL ANFITRIÓN PUEDE REINICIAR", 2.0
            return True
        elif self.mp and key == pygame.K_n:
            if self.mp == "guest":
                self.toast, self.toast_t = "EL HORARIO LO ELIGE EL ANFITRIÓN", 2.0
                return True
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
        elif key == pygame.K_r:
            self.new_race(intro=False)
        if key == pygame.K_c:
            self.cam_mode = (self.cam_mode + 1) % 4
            self.toast, self.toast_t = ["CÁMARA: PERSECUCIÓN", "CÁMARA: LEJANA", "CÁMARA: COFRE",
                                        "CÁMARA: DEFENSA"][self.cam_mode], 1.5
        elif key == pygame.K_n:
            self.tod = (self.tod + 1) % len(TOD_HD)
            self.toast, self.toast_t = f"HORARIO: {TOD_HD[self.tod][0]}", 1.5
        elif key == pygame.K_F1:
            self.qi = (self.qi + 1) % len(QUALITY)
            self.R.set_quality(QUALITY[self.qi][1])
            self.toast, self.toast_t = f"CALIDAD: {QUALITY[self.qi][0]}", 1.5
        elif key == pygame.K_F3:
            self.show_fps = not self.show_fps
        elif key in (pygame.K_LCTRL, pygame.K_RCTRL):
            if self.music.mode == "playlist":
                self.music.next_song()
        elif key == pygame.K_m:
            self.sound.muted = not self.sound.muted
        return True

    def screenshot(self, path):
        data = self.screen_fbo.read(components=3)
        img = pygame.image.frombuffer(data, (WIN_W, WIN_H), "RGB")
        pygame.image.save(pygame.transform.flip(img, False, True), path)

    def run(self):
        while True:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            self.fps = self.fps * 0.95 + (1.0 / max(dt, 1e-3)) * 0.05
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and not self.handle_key(e.key)):
                    pygame.quit()
                    sys.exit()
            self.update(dt)
            self.render()
            pygame.display.flip()


if __name__ == "__main__":
    GameHD().run()

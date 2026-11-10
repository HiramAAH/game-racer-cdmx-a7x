"""
GTC RACING HD - autos nuevos y modelos mejorados.

Este modulo NO modifica los otros archivos del juego: agrega pilotos/vehiculos y reemplaza en tiempo de
ejecucion el constructor de modelos 3D de gtc_racing_hd.

Uso (antes de crear el juego):
    import gtc_autos
    gtc_autos.install()

Contenido:
    VEH_EXTRA          dimensiones/camara de cada tipo de vehiculo nuevo
    ROSTER_EXTRA       pilotos nuevos (mismas claves que PILOTS + "num" opcional)
    build_vehicle_hd2  mismo contrato que gtc_racing_hd.build_vehicle_hd -> (body, glass, wheel, info)
    install()          integra todo (idempotente)

Marco local del coche: +x adelante, y arriba (y=0 es el piso), z a los lados (+z = derecha).
Las ruedas se dibujan centradas en (x, radio, +-z); la malla de la rueda tiene su eje en z y la cara exterior
hacia +z (el juego la refleja para el lado izquierdo).
"""
import math
import random

import numpy as np

import gtc_racing_3d as base
import gtc_racing_hd as hd
from gtc_racing_hd import (Geo, quad, grid, obox, box, superellipsoid, sphere, revolve, cylinder, transform_geo,
                           rgb, mat, norm, sgnpow, F4, TAU, M_DEF, M_METAL, M_GLASS, M_PAINT, M_HEAD, M_TAIL,
                           M_TIRE, M_EMIS, M_SIGN)

# ================================================================ materiales y colores comunes
PAINT = mat(M_PAINT, 0.26)              # pintura con barniz (lleva uv = (x, z) para la franja del shader)
PAINT_MATE = mat(M_DEF, 0.32)           # pintura en piezas chicas (no se pinta con la franja de sport/hatch)
TRIM = mat(M_DEF, 0.6)                  # plastico negro
RUBBER = mat(M_DEF, 0.85)
CARBON = mat(M_DEF, 0.35, 0.25)
CHROME = mat(M_METAL, 0.12, 1.0)
ALU = mat(M_METAL, 0.3, 1.0)
GLASS_M = mat(M_GLASS, 0.04)
HEAD_M = mat(M_HEAD, 0.2)
TAIL_M = mat(M_TAIL, 0.2)
TIRE_M = mat(M_TIRE, 0.85)
SIGN_M = mat(M_SIGN, 0.3)
AMBER_M = mat(M_EMIS, 0.3, 0.0, 1.6)
LED_M = mat(M_EMIS, 0.2, 0.0, 3.0)

DARK = rgb(18, 18, 20)
BLACK = rgb(10, 10, 12)
GREY = rgb(70, 72, 78)
WHITE = rgb(240, 240, 242)
CHROME_C = rgb(215, 215, 220)
GLASS_C = rgb(62, 82, 108)
TIRE_C = rgb(22, 22, 24)
HEAD_C = (1.0, 0.97, 0.9)
TAIL_C = (1.0, 0.08, 0.05)
AMBER_C = (1.0, 0.45, 0.02)


def C(t):
    """(r, g, b) 0-255 -> 0-1."""
    return rgb(*t)


# ================================================================ herramientas de geometria
def _catmull(A, sub):
    """Interpola filas de control: x lineal (monotono), el resto con Catmull-Rom."""
    n = len(A)
    P = np.vstack([A[:1], A, A[-1:]])
    out = []
    for i in range(n - 1):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        for k in range(sub):
            t = k / sub
            r = 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
            r[0] = p1[0] + (p2[0] - p1[0]) * t
            out.append(r)
    out.append(A[-1])
    return np.array(out)


class Loft:
    """Cuerpo "barrido" a lo largo de x: cada estacion es una superelipse en el plano (z, y).

    Filas de control: (x, y_abajo, y_arriba, medio_ancho_abajo, medio_ancho_arriba[, e]) con x creciente.
    e < 1 = seccion cuadrada (0.2 muy cuadrada, 1.0 eliptica). zc = centro lateral (para pontones)."""

    def __init__(self, ctrl, e=0.3, sub=3, smooth=True, zc=0.0):
        A = np.array(ctrl, np.float64)
        if A.shape[1] == 5:
            A = np.c_[A, np.full(len(A), e)]
        self.S = _catmull(A, sub) if (smooth and sub > 1 and len(A) > 2) else A
        self.zc = zc

    def rows(self, xs):
        xs = np.atleast_1d(np.asarray(xs, np.float64))
        S = self.S
        return np.stack([xs] + [np.interp(xs, S[:, 0], S[:, k]) for k in range(1, 6)], 1)

    def eval(self, R, ph):
        ph = np.asarray(ph, np.float64)
        if ph.ndim == 1:
            ph = ph[None, :]
        c, s = np.cos(ph), np.sin(ph)
        E = R[:, 5:6]
        cz, cy = sgnpow(c, E), sgnpow(s, E)
        yc, hh = (R[:, 1:2] + R[:, 2:3]) / 2, (R[:, 2:3] - R[:, 1:2]) / 2
        t = (cy + 1) / 2
        hw = R[:, 3:4] * (1 - t) + R[:, 4:5] * t
        x = np.broadcast_to(R[:, 0:1], cz.shape)
        return np.stack([x, yc + hh * cy, self.zc + hw * cz], -1)

    def mesh(self, g, col, m, n=24, colfn=None, caps=True, uv=True, split=None, col2=None, m2=None):
        """Malla cerrada. split = altura y del corte de dos tonos: abajo 'col', arriba 'col2' (borde nitido)."""
        if split is not None:
            R = self.S
            yc, hh = (R[:, 1] + R[:, 2]) / 2, np.maximum((R[:, 2] - R[:, 1]) / 2, 1e-6)
            f = np.clip((split - yc) / hh, -0.995, 0.995)
            phs = np.sign(f) * np.arcsin(np.abs(f) ** (1 / R[:, 5]))
            t = np.linspace(0, 1, n // 2 + 1)[None, :]
            for ph, cc, mm in (((phs[:, None] + (np.pi - 2 * phs)[:, None] * t), col2, m2 or m),
                               (((np.pi - phs)[:, None] + (np.pi + 2 * phs)[:, None] * t), col, m)):
                P = self.eval(R, ph).astype(F4)
                axis = np.stack([P[..., 0], np.broadcast_to(yc[:, None], P.shape[:2]),
                                 np.full(P.shape[:2], self.zc)], -1)
                grid(g, P, cc, mm, UV=P[..., [0, 2]] if uv else None, orient=P - axis)
            if colfn is None:
                c_lo, c_hi = np.asarray(col, F4), np.asarray(col2, F4)
                colfn = lambda Q: np.where((Q[..., 1:2] > split), c_hi, c_lo).astype(F4)
            ph = np.linspace(0, TAU, n, endpoint=False)
            P = self.eval(self.S, ph).astype(F4)
        else:
            ph = np.linspace(0, TAU, n, endpoint=False)
            P = self.eval(self.S, ph).astype(F4)
            Cc = colfn(P) if colfn is not None else None
            grid(g, P, col, m, UV=P[..., [0, 2]] if uv else None, wrap=True, C=Cc, autoorient=True)
        if caps:
            cen = P.reshape(-1, 3).mean(0)
            for ring in (P[0], P[-1]):
                c0 = ring.mean(0)
                nn = np.cross(ring[1] - ring[0], ring[n // 4] - ring[0])
                ln = np.linalg.norm(nn)
                if ln < 1e-7:
                    continue
                nn = nn / ln
                if np.dot(nn, c0 - cen) < 0:
                    nn = -nn
                nxt = np.roll(ring, -1, 0)
                T = np.stack([np.repeat(c0[None], n, 0), ring, nxt], 1).reshape(-1, 3)
                cc = col if colfn is None else colfn(T.reshape(-1, 1, 3)).reshape(-1, 3)
                uvv = T[:, [0, 2]] if uv else None
                g.tris(T, nn, cc, m, uvv)
        return P

    def patch(self, g, x0, x1, ph0, ph1, col, m, nx=8, nph=6, off=0.006):
        """Parche de la superficie (para franjas, calcomanias pintadas) despegado 'off' hacia afuera."""
        R = self.rows(np.linspace(x0, x1, nx))
        P = self.eval(R, np.linspace(ph0, ph1, nph))
        dR = np.gradient(P, axis=0)
        dC = np.gradient(P, axis=1)
        N = np.cross(dC, dR)
        N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
        out = P.copy()
        out[..., 1] = (R[:, 1:2] + R[:, 2:3]) / 2
        out[..., 2] = self.zc
        out[..., 0] = P[..., 0]
        sgn = np.sign(np.sum(N * (P - out), -1, keepdims=True))
        sgn[sgn == 0] = 1
        N = N * sgn
        P = P + N * off
        grid(g, P.astype(F4), col, m, UV=P[..., [0, 2]], orient=N)

    def z_at(self, x, y):
        R = self.rows([x])[0]
        yc, hh = (R[1] + R[2]) / 2, max((R[2] - R[1]) / 2, 1e-6)
        e = R[5]
        cy = min(1.0, abs((y - yc) / hh))
        cz = (1 - cy ** (2 / e)) ** (e / 2)
        t = ((y - yc) / hh + 1) / 2
        return self.zc + (R[3] * (1 - t) + R[4] * t) * cz

    def side_z(self, x0, x1, y0, y1):
        """Medio ancho maximo en un rectangulo del costado (para pegar calcomanias planas sin que se hundan)."""
        return max(self.z_at(x, y) for x in np.linspace(x0, x1, 5) for y in np.linspace(y0, y1, 5)) - self.zc

    def y_top(self, x, z=0.0):
        R = self.rows([x])[0]
        yc, hh, e = (R[1] + R[2]) / 2, (R[2] - R[1]) / 2, R[5]
        cz = min(1.0, abs((z - self.zc) / max(R[4], 1e-6)))
        return yc + hh * (1 - cz ** (2 / e)) ** (e / 2)


def ph_top(f, e):
    """Angulo de la superelipse para una fraccion f del medio ancho en la parte de ARRIBA."""
    return math.acos(min(1.0, max(f, 0.0)) ** (1 / e))


def ph_side(f, e):
    """Angulo para una fraccion f de la media altura en el costado (+z). f<0 = abajo."""
    return math.copysign(math.asin(min(1.0, abs(f)) ** (1 / e)), f)


def bar(g, p0, p1, t, col, m, up=(0, 1, 0), t2=None):
    """Barra de seccion rectangular de p0 a p1 (grosor t, alto t2)."""
    p0, p1 = np.asarray(p0, F4), np.asarray(p1, F4)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return
    dn = d / L
    u = np.asarray(up, F4)
    if abs(float(np.dot(u, dn))) > 0.95:
        u = np.array([1, 0, 0], F4) if abs(dn[0]) < 0.9 else np.array([0, 0, 1], F4)
    a = norm(np.cross(dn, u))
    b = np.cross(a, dn)
    t2 = t if t2 is None else t2
    obox(g, p0 - a * t / 2 - b * t2 / 2, d, b * t2, a * t, col, m)


def tube(g, pts, r, col, m, seg=8, joints=True):
    """Tubo por una polilinea (jaulas antivuelco, halo, escapes)."""
    for a, b in zip(pts[:-1], pts[1:]):
        cylinder(g, a, b, r, col, m, seg=seg, caps=False)
    if joints:
        for p in pts:
            sphere(g, p, r, col, m, seg=seg, rings=4)


def prism_xy(g, pts, z0, z1, col, m):
    """Poligono convexo en el plano (x, y) extruido en z (aleron, placas, aletas)."""
    pts = [np.array(p, F4) for p in pts]
    n = len(pts)
    for zz, sg in ((z0, -1), (z1, 1)):
        c0 = np.mean(pts, 0)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            g.tris([(c0[0], c0[1], zz), (a[0], a[1], zz), (b[0], b[1], zz)], (0, 0, sg), col, m)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        quad(g, (a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1), col, m)


def prism_xz(g, pts, y0, y1, col, m):
    """Poligono convexo en el plano (x, z) extruido en y (placas horizontales: piso, alerones)."""
    pts = [np.array(p, F4) for p in pts]
    n = len(pts)
    c0 = np.mean(pts, 0)
    for yy, sg in ((y0, -1), (y1, 1)):
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            g.tris([(c0[0], yy, c0[1]), (a[0], yy, a[1]), (b[0], yy, b[1])], (0, sg, 0), col, m)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        quad(g, (a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y1, a[1]), col, m)


def disc(g, c, r, nrm, col, m, seg=16, r_in=0.0):
    """Disco (o anillo si r_in > 0) plano con normal 'nrm'."""
    c, nrm = np.asarray(c, F4), norm(nrm)
    tmp = np.array([0, 1, 0], F4) if abs(nrm[1]) < 0.9 else np.array([1, 0, 0], F4)
    u = norm(np.cross(nrm, tmp))
    v = np.cross(nrm, u)
    a = np.linspace(0, TAU, seg + 1)
    ring = c + r * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
    if r_in <= 0:
        P = np.stack([np.repeat(c[None], seg, 0), ring[:-1], ring[1:]], 1).reshape(-1, 3)
        g.tris(P, nrm, col, m)
    else:
        inner = c + r_in * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
        P = np.stack([inner[:-1], ring[:-1], ring[1:], inner[:-1], ring[1:], inner[1:]], 1).reshape(-1, 3)
        g.tris(P, nrm, col, m)


def lamp(g, c, half, col, m, e=0.35, seg=12, rings=6):
    superellipsoid(g, c, half, e, col, m, seg, rings)


def both(fn):
    """Llama fn(s) para s = -1 y +1 (piezas simetricas)."""
    for s in (-1, 1):
        fn(s)


def fender(g, cx, cy, a0, a1, r_in, thick, z_in, z_out, col, m=PAINT, n=16, taper=0.45, e=0.6, k=7,
           under=BLACK, sides=(-1, 1)):
    """Salpicadera redonda sobre la rueda (vocho, baja): arco de a0 a a1 (0 = adelante, pi/2 = arriba)
    con perfil de cuarto de superelipse entre el labio exterior (r_in, z_out) y el cuerpo (r_in+thick, z_in)."""
    A = np.linspace(a0, a1, n)
    f = np.clip(np.minimum(A - a0, a1 - A) / taper, 0, 1)
    f = np.sin(f * np.pi / 2)
    th = thick * (0.3 + 0.7 * f)
    T = np.linspace(0, np.pi / 2, k)
    for s in sides:
        zo = z_in + (z_out - z_in) * (0.35 + 0.65 * f)
        rho = r_in + th[:, None] * sgnpow(np.sin(T), e)[None, :]
        zz = z_in + (zo[:, None] - z_in) * sgnpow(np.cos(T), e)[None, :]
        P = np.stack([cx + rho * np.cos(A)[:, None], cy + rho * np.sin(A)[:, None], s * zz], -1)
        O = np.stack([np.broadcast_to(np.cos(A)[:, None], rho.shape), np.broadcast_to(np.sin(A)[:, None], rho.shape),
                      np.full(rho.shape, 0.7 * s)], -1)
        grid(g, P.astype(F4), col, m, UV=P[..., [0, 2]], orient=O)
        Q = np.stack([np.stack([cx + r_in * np.cos(A), cy + r_in * np.sin(A), s * zo], -1),
                      np.stack([cx + r_in * np.cos(A), cy + r_in * np.sin(A), np.full(n, s * z_in)], -1)], 1)
        grid(g, Q.astype(F4), under, TRIM, orient=-np.stack([np.cos(A), np.sin(A), np.zeros(n)], -1)[:, None, :])


def spare_tire(g, c, r, w, rim=GREY):
    """Llanta de refaccion acostada (eje y)."""
    prof = [(0.62 * r, -w / 2), (0.92 * r, -w / 2), (r, -w / 4), (r, w / 4), (0.92 * r, w / 2), (0.62 * r, w / 2)]
    revolve(g, prof, c, (0, 1, 0), TIRE_C, TIRE_M, seg=18)
    disc(g, np.asarray(c, F4) + np.array([0, w / 2 - 0.02, 0], F4), 0.63 * r, (0, 1, 0), rim, ALU, seg=14)


# ---------------------------------------------------------------- tipografia de pixeles (numeros, TAXI...)
FONT = {
    "0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"], "3": ["111", "001", "011", "001", "111"],
    "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"], "7": ["111", "001", "001", "010", "010"],
    "8": ["111", "101", "111", "101", "111"], "9": ["111", "101", "111", "001", "111"],
    "T": ["111", "010", "010", "010", "010"], "A": ["010", "101", "111", "101", "101"],
    "X": ["101", "101", "010", "101", "101"], "I": ["111", "010", "010", "010", "111"],
    "L": ["100", "100", "100", "100", "111"], "B": ["110", "101", "110", "101", "110"],
    "R": ["110", "101", "110", "101", "101"], "E": ["111", "100", "110", "100", "111"],
    "G": ["111", "100", "101", "101", "111"], "C": ["111", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"], "S": ["111", "100", "111", "001", "111"],
    "O": ["111", "101", "101", "101", "111"], "P": ["110", "101", "110", "100", "100"],
    "J": ["001", "001", "001", "101", "111"], "U": ["101", "101", "101", "101", "111"],
    "N": ["110", "101", "101", "101", "101"], "H": ["101", "101", "111", "101", "101"],
    "-": ["000", "000", "111", "000", "000"], " ": ["000", "000", "000", "000", "000"],
}


def text(g, s, c, right, up, h, col, m, bold=1.0):
    """Texto de pixeles centrado en c, en el plano (right, up); h = alto de las letras."""
    c, right, up = np.asarray(c, F4), norm(right), norm(up)
    nrm = norm(np.cross(right, up))
    px = h / 5.0
    w = (len(s) * 4 - 1) * px
    o = c - right * w / 2 + up * h / 2
    for i, ch in enumerate(s):
        gl = FONT.get(ch.upper(), FONT[" "])
        for r, row in enumerate(gl):
            j = 0
            while j < 3:
                if row[j] == "1":
                    k = j
                    while k < 3 and row[k] == "1":
                        k += 1
                    x0, x1 = (i * 4 + j) * px, (i * 4 + k) * px
                    y0, y1 = r * px, (r + 1) * px * bold + r * px * (1 - bold)
                    a = o + right * x0 - up * y0
                    b = o + right * x1 - up * y0
                    cc = o + right * x1 - up * y1
                    d = o + right * x0 - up * y1
                    g.tris([a, d, cc, a, cc, b], nrm, col, m)
                    j = k
                else:
                    j += 1


def roundel(g, c, nrm, right, num, r, bg=WHITE, fg=BLACK):
    """Circulo de carreras con numero (puerta, cofre, techo)."""
    nrm = norm(nrm)
    disc(g, c, r, nrm, bg, PAINT_MATE, seg=20)
    disc(g, np.asarray(c, F4) + nrm * 0.002, r, nrm, fg, PAINT_MATE, seg=20, r_in=r * 0.9)
    up = np.cross(nrm, norm(right))
    text(g, str(num), np.asarray(c, F4) + nrm * 0.004, right, up, r * 1.05, fg, PAINT_MATE)


def side_text(g, s, x, y, z_abs, h, col, m=PAINT_MATE):
    """Texto en ambos costados, legible desde afuera."""
    text(g, s, (x, y, z_abs), (1, 0, 0), (0, 1, 0), h, col, m)
    text(g, s, (x, y, -z_abs), (-1, 0, 0), (0, 1, 0), h, col, m)


def decal_hd(g, xc, zh, width, y0, y1, k):
    """Calcomania del atlas de anuncios en ambos costados (k = indice de letrero)."""
    u0, u1, v0, v1 = hd.bb_uv(k)
    for sz in (-1, 1):
        z_ = sz * zh
        a, b = (xc + width / 2, xc - width / 2) if sz < 0 else (xc - width / 2, xc + width / 2)
        quad(g, (a, y0, z_), (b, y0, z_), (b, y1, z_), (a, y1, z_), (1, 1, 1), SIGN_M,
             ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))


# ---------------------------------------------------------------- detalles de carroceria
def wells(g, loft, wheels, r, ra, ybot, col=BLACK, lip=None, lip_w=0.05, lip_out=0.03):
    """Huecos de rueda oscuros pegados al costado (+ ceja/salpicadera opcional)."""
    for wx in wheels:
        for s in (-1, 1):
            pts = [(wx + ra, ybot)]
            for k in range(13):
                a = k / 12 * math.pi
                x = wx + ra * math.cos(a)
                R_ = loft.rows([x])[0]
                ymax = R_[1] + (R_[2] - R_[1]) * 0.86
                pts.append((x, min(r + ra * math.sin(a), ymax)))
            pts.append((wx - ra, ybot))
            P = [(x, y, s * (abs(loft.z_at(x, y) - loft.zc) + 0.004) + loft.zc) for x, y in pts]
            cx, cy = wx, (ybot + r) / 2
            c0 = (cx, cy, s * (abs(loft.z_at(cx, cy)) + 0.004))
            for a, b in zip(P[:-1], P[1:]):
                g.tris([c0, a, b], (0, 0, s), col, TRIM)
            if lip is not None:
                for k in range(12):
                    a0, a1 = k / 12 * math.pi, (k + 1) / 12 * math.pi
                    q = []
                    for a in (a0, a1):
                        x, y = wx + ra * math.cos(a), r + ra * math.sin(a)
                        z = abs(loft.z_at(x, y))
                        x2, y2 = wx + (ra + lip_w) * math.cos(a), r + (ra + lip_w) * math.sin(a)
                        z2 = abs(loft.z_at(x2, y2))
                        q.append(((x, y, s * (z + lip_out)), (x2, y2, s * (z2 + 0.002))))
                    quad(g, q[0][0], q[1][0], q[1][1], q[0][1], lip, PAINT_MATE)


def seam(g, loft, pts, w=0.012, col=BLACK, sides=(-1, 1), off=0.004):
    """Linea fina (junta de puerta, cofre) siguiendo el costado del cuerpo. pts = [(x, y), ...]."""
    fine = [pts[0]]
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(xb - xa, yb - ya) / 0.15))
        fine += [(xa + (xb - xa) * k / n, ya + (yb - ya) * k / n) for k in range(1, n + 1)]
    for s in sides:
        for (xa, ya), (xb, yb) in zip(fine[:-1], fine[1:]):
            d = np.array([xb - xa, yb - ya])
            ln = np.linalg.norm(d)
            if ln < 1e-6:
                continue
            nx, ny = -d[1] / ln * w / 2, d[0] / ln * w / 2
            P = []
            for x, y in ((xa - nx, ya - ny), (xb - nx, yb - ny), (xb + nx, yb + ny), (xa + nx, ya + ny)):
                P.append((x, y, loft.zc + s * (abs(loft.z_at(x, y) - loft.zc) + off)))
            quad(g, *P, col, TRIM)


def top_seam(g, loft, pts, w=0.012, col=BLACK, off=0.004):
    """Linea fina sobre la parte de arriba (cofre/cajuela). pts = [(x, z), ...]."""
    fine = [pts[0]]
    for (xa, za), (xb, zb) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(xb - xa, zb - za) / 0.15))
        fine += [(xa + (xb - xa) * k / n, za + (zb - za) * k / n) for k in range(1, n + 1)]
    for (xa, za), (xb, zb) in zip(fine[:-1], fine[1:]):
        d = np.array([xb - xa, zb - za])
        ln = np.linalg.norm(d)
        if ln < 1e-6:
            continue
        nx, nz = -d[1] / ln * w / 2, d[0] / ln * w / 2
        P = [(x, loft.y_top(x, z) + off, z) for x, z in ((xa - nx, za - nz), (xb - nx, zb - nz),
                                                        (xb + nx, zb + nz), (xa + nx, za + nz))]
        quad(g, *P, col, TRIM)


def mirrors(g, x, y, z, col, m=PAINT_MATE, size=1.0):
    for s in (-1, 1):
        bar(g, (x, y, s * (z - 0.1)), (x - 0.02, y + 0.04, s * (z + 0.02)), 0.03, DARK, TRIM)
        superellipsoid(g, (x - 0.03, y + 0.07, s * (z + 0.06)), (0.07 * size, 0.06 * size, 0.1 * size), 0.5,
                       col, m, 10, 6)
        disc(g, (x - 0.1 * size, y + 0.07, s * (z + 0.06)), 0.05 * size, (-1, 0, 0), rgb(150, 170, 190),
             mat(M_METAL, 0.05, 1.0), seg=8)


def plate(g, x, y, front=True, z=0.0, col=WHITE, band=rgb(30, 70, 160), w=0.34, h=0.12):
    """Placa (blanca con franja de color, estilo placa mexicana)."""
    sx = 1 if front else -1
    box(g, x - 0.012 * sx, y - h / 2, z - w / 2, x + 0.008 * sx, y + h / 2, z + w / 2, col, PAINT_MATE)
    box(g, x + 0.009 * sx - 0.002, y + h / 2 - 0.03, z - w / 2, x + 0.011 * sx, y + h / 2 - 0.005, z + w / 2,
        band, PAINT_MATE)
    for k in range(4):
        zz = z - w / 2 + 0.07 + k * 0.065
        box(g, x + 0.009 * sx - 0.002, y - h / 2 + 0.02, zz - 0.022, x + 0.011 * sx, y + h / 2 - 0.045, zz + 0.022,
            DARK, PAINT_MATE)


def wipers(g, x, y, z_list, ang=0.25, L=0.45):
    for zc in z_list:
        bar(g, (x, y, zc), (x - 0.02, y + 0.02, zc - L), 0.02, BLACK, TRIM)


def grille_slats(g, x, y0, y1, z0, z1, n, col=DARK, slat=CHROME_C, sm=CHROME, depth=0.03):
    box(g, x - depth, y0, z0, x, y1, z1, col, TRIM)
    for k in range(n):
        yy = y0 + (y1 - y0) * (k + 0.5) / n
        box(g, x - depth * 0.3, yy - 0.008, z0, x + 0.006, yy + 0.008, z1, slat, sm)


# ================================================================ ruedas por estilo
def make_wheel(style, r, w, rim=CHROME_C, accent=None, tread=False):
    """Rueda con eje z, cara exterior hacia +z. style: spoke5, spoke6, mesh, steel, hubcap, cover, f2,
    nascar, beadlock, dually."""
    g = Geo()
    rr = {"cover": 0.72, "f2": 0.72, "beadlock": 0.6, "nascar": 0.66, "mesh": 0.7, "spoke5": 0.7}.get(style, 0.64)
    side = 0.035 if style not in ("beadlock",) else 0.06
    prof = [(rr * r, -w / 2), (0.9 * r, -w / 2 - 0.01), (0.985 * r, -w / 2 + side), (r, -w / 4), (r, w / 4),
            (0.985 * r, w / 2 - side), (0.9 * r, w / 2 + 0.01), (rr * r, w / 2)]
    revolve(g, prof, (0, 0, 0), (0, 0, 1), TIRE_C, TIRE_M, seg=26)
    disc(g, (0, 0, -w / 2 + 0.03), rr * r, (0, 0, -1), DARK, TRIM, seg=14)        # tapa interior
    zf = w / 2 - 0.04                                                                  # plano del rin
    rim_m = mat(M_METAL, 0.18, 1.0)
    if tread:                                                                          # tacos todo terreno
        n = 16
        for k in range(n):
            a = (k + 0.5) / n * TAU
            d = np.array([math.cos(a), math.sin(a), 0], F4)
            p = np.array([-math.sin(a), math.cos(a), 0], F4)
            for zz in ((-w / 2 + 0.02, -0.02) if k % 2 else (0.02, w / 2 - 0.02),):
                obox(g, d * (r - 0.005) - p * 0.045 + np.array([0, 0, zz[0]], F4), d * 0.035, p * 0.09,
                     np.array([0, 0, zz[1] - zz[0]], F4), TIRE_C, TIRE_M)
            # hombro
            obox(g, d * (0.86 * r) - p * 0.04 + np.array([0, 0, w / 2 - 0.005], F4), d * 0.12 * r, p * 0.08,
                 np.array([0, 0, 0.025], F4), TIRE_C, TIRE_M)
    # barril del rin (hacia adentro, oscuro)
    revolve(g, [(rr * r, zf + 0.02), (rr * r * 0.97, zf - 0.02), (rr * r * 0.9, -w / 2 + 0.05)], (0, 0, 0),
            (0, 0, 1), GREY, ALU, seg=18)

    def spokes(n, width, col, z=zf, inner=0.16, outer=0.95, thick=0.035, twist=0.0):
        for k in range(n):
            a = k / n * TAU
            d = np.array([math.cos(a), math.sin(a), 0], F4)
            p = np.array([-math.sin(a), math.cos(a), 0], F4)
            obox(g, d * inner * rr * r - p * width / 2 + np.array([0, 0, z - thick], F4),
                 d * (outer - inner) * rr * r + p * twist, p * width, np.array([0, 0, thick], F4), col, rim_m)

    if style in ("spoke5", "spoke6", "mesh", "f2"):
        revolve(g, [(rr * r * 0.86, zf - 0.03), (rr * r, zf + 0.005), (rr * r * 1.02, zf + 0.012)], (0, 0, 0),
                (0, 0, 1), rim, rim_m, seg=24)                                         # labio del rin
        disc(g, (0, 0, zf - 0.05), rr * r * 0.9, (0, 0, 1), rgb(30, 30, 34), TRIM, seg=14)
        if style == "spoke5":
            spokes(5, 0.11 * r / 0.37, rim, thick=0.04)
            spokes(5, 0.03, rim, thick=0.03, twist=0.0)
        elif style == "spoke6":
            spokes(6, 0.075 * r / 0.37, rim)
        elif style == "mesh":
            spokes(10, 0.032 * r / 0.37, rim, thick=0.03)
            revolve(g, [(rr * r * 0.5, zf - 0.035), (rr * r * 0.53, zf - 0.005), (rr * r * 0.56, zf - 0.035)],
                    (0, 0, 0), (0, 0, 1), rim, rim_m, seg=20)
        else:
            spokes(8, 0.045 * r / 0.37, rim, thick=0.03)
        cylinder(g, (0, 0, zf - 0.04), (0, 0, zf + 0.01), 0.1 * r, rim, rim_m, seg=10)
        disc(g, (0, 0, zf + 0.012), 0.06 * r, (0, 0, 1), DARK if accent is None else accent, PAINT_MATE, seg=8)
    elif style == "cover":                                                            # F1 2022: tapa lisa
        revolve(g, [(0.0, zf + 0.02), (rr * r * 0.5, zf + 0.015), (rr * r, zf - 0.005), (rr * r * 1.02, zf - 0.01)],
                (0, 0, 0), (0, 0, 1), rgb(25, 25, 28), CARBON, seg=24)
        if accent is not None:
            disc(g, (0, 0, zf + 0.022), rr * r * 0.82, (0, 0, 1), accent, PAINT_MATE, seg=24, r_in=rr * r * 0.7)
        cylinder(g, (0, 0, zf + 0.0), (0, 0, zf + 0.05), 0.09 * r, rgb(230, 30, 30), PAINT_MATE, seg=8)
    elif style in ("steel", "hubcap"):
        revolve(g, [(0.0, zf - 0.02), (rr * r * 0.4, zf - 0.03), (rr * r * 0.8, zf - 0.015), (rr * r * 0.98, zf + 0.0),
                    (rr * r * 1.02, zf + 0.006)], (0, 0, 0), (0, 0, 1), rim, rim_m if style == "hubcap" else
                PAINT_MATE, seg=24)
        if style == "steel":
            for k in range(6):
                a = k / 6 * TAU
                disc(g, (math.cos(a) * rr * r * 0.62, math.sin(a) * rr * r * 0.62, zf - 0.012), 0.035 * r / 0.32,
                     (0, 0, 1), DARK, TRIM, seg=8)
            revolve(g, [(rr * r * 0.48, zf - 0.02), (rr * r * 0.4, zf + 0.03), (rr * r * 0.2, zf + 0.06),
                        (0.0, zf + 0.065)], (0, 0, 0), (0, 0, 1), CHROME_C, CHROME, seg=16)  # tapon cromado
        else:
            for k in range(8):
                a = (k + 0.5) / 8 * TAU
                d = np.array([math.cos(a), math.sin(a), 0], F4)
                p = np.array([-math.sin(a), math.cos(a), 0], F4)
                obox(g, d * rr * r * 0.45 - p * 0.02 + np.array([0, 0, zf - 0.02], F4), d * rr * r * 0.4, p * 0.04,
                     np.array([0, 0, 0.02], F4), DARK, TRIM)
            disc(g, (0, 0, zf - 0.005), rr * r * 0.25, (0, 0, 1), GREY, ALU, seg=12)
    elif style == "nascar":
        revolve(g, [(0.0, zf - 0.03), (rr * r * 0.85, zf - 0.03), (rr * r * 1.02, zf + 0.006)], (0, 0, 0), (0, 0, 1),
                rim, PAINT_MATE, seg=24)
        for k in range(10):
            a = k / 10 * TAU
            disc(g, (math.cos(a) * rr * r * 0.62, math.sin(a) * rr * r * 0.62, zf - 0.025), 0.04, (0, 0, 1), DARK,
                 TRIM, seg=8)
        cylinder(g, (0, 0, zf - 0.03), (0, 0, zf + 0.04), 0.13 * r, CHROME_C, CHROME, seg=10)
        cylinder(g, (0, 0, zf + 0.04), (0, 0, zf + 0.07), 0.07 * r, rgb(220, 180, 40), PAINT_MATE, seg=6)
    elif style in ("beadlock", "dually"):
        revolve(g, [(0.0, zf - 0.08), (rr * r * 0.5, zf - 0.07), (rr * r * 0.85, zf - 0.03), (rr * r, zf + 0.005)],
                (0, 0, 0), (0, 0, 1), rim, rim_m, seg=24)
        disc(g, (0, 0, zf + 0.008), rr * r * 1.04, (0, 0, 1), DARK if accent is None else accent, PAINT_MATE,
             seg=24, r_in=rr * r * 0.86)
        for k in range(16):
            a = k / 16 * TAU
            disc(g, (math.cos(a) * rr * r * 0.95, math.sin(a) * rr * r * 0.95, zf + 0.014), 0.014, (0, 0, 1),
                 CHROME_C, CHROME, seg=5)
        for k in range(8):
            a = k / 8 * TAU
            disc(g, (math.cos(a) * rr * r * 0.55, math.sin(a) * rr * r * 0.55, zf - 0.04), 0.045, (0, 0, 1), DARK,
                 TRIM, seg=8)
        cylinder(g, (0, 0, zf - 0.08), (0, 0, zf + 0.0), 0.14 * r, CHROME_C, CHROME, seg=10)
    return g


# ================================================================ pilotos (personajes nuevos)
DRIVERS = {
    # nombre: ropa, sombrero/casco, color1, color2, bigote, lentes, pelo
    "taxista": dict(cloth=(110, 160, 210), hat="cap", c1=(130, 25, 45), c2=(210, 170, 70), mous=True),
    "vochero": dict(cloth=(245, 245, 238), hat="sombrero", c1=(225, 195, 125), c2=(120, 70, 30), mous=True),
    "chuy": dict(cloth=(190, 45, 110), hat="bun", c1=(85, 75, 75), c2=(240, 240, 240), glasses=True),
    "texano": dict(cloth=(150, 40, 35), hat="texana", c1=(235, 220, 190), c2=(70, 40, 20), mous=True),
    "tunero": dict(cloth=(30, 30, 35), hat="capr", c1=(240, 240, 240), c2=(120, 40, 200), glasses=True),
    "casco_mx": dict(cloth=(0, 110, 60), hat="helmet", c1=(0, 120, 65), c2=(250, 250, 250), c3=(215, 30, 40)),
    "casco_f2": dict(cloth=(20, 60, 170), hat="helmet", c1=(245, 245, 250), c2=(240, 120, 20), c3=(20, 60, 170)),
    "casco_nas": dict(cloth=(240, 110, 20), hat="helmet", c1=(240, 110, 20), c2=(20, 50, 140), c3=(250, 250, 250)),
    "casco_de": dict(cloth=(240, 240, 245), hat="helmet", c1=(240, 240, 245), c2=(30, 70, 170), c3=(220, 30, 40)),
    "casco_baja": dict(cloth=(25, 25, 28), hat="helmet", c1=(240, 100, 20), c2=(25, 25, 28), c3=(250, 210, 40)),
}
_ORIG_CHARACTER = hd.character


def character2(g, kind, x, y, z, standing=False):
    """Igual que gtc_racing_hd.character pero con los personajes nuevos (DRIVERS)."""
    if kind not in DRIVERS:
        return _ORIG_CHARACTER(g, kind, x, y, z, standing)
    d = DRIVERS[kind]
    sk = hd.SKIN_HD
    yb = y + 0.95 if standing else y
    hy = yb + 0.82
    hc = (x, hy, z)
    cloth = C(d["cloth"])
    m_cl = mat(M_DEF, 0.75)
    superellipsoid(g, (x - 0.02, yb + 0.32, z), (0.2, 0.32, 0.27), 0.7, cloth, m_cl, 14, 10)
    helmet = d["hat"] == "helmet"
    if standing:
        pants = rgb(35, 40, 70) if not helmet else cloth
        for sz in (-0.11, 0.11):
            cylinder(g, (x, yb + 0.02, z + sz), (x, y + 0.08, z + sz * 1.1), 0.085, pants, m_cl, seg=10)
            superellipsoid(g, (x + 0.06, y + 0.05, z + sz * 1.1), (0.13, 0.05, 0.07), 0.6,
                           rgb(90, 50, 25) if d["hat"] in ("texana", "sombrero") else rgb(25, 25, 28),
                           mat(M_DEF, 0.5), 10, 6)
        for sz in (-1, 1):
            cylinder(g, (x, yb + 0.55, z + sz * 0.29), (x + 0.05, yb + 0.05, z + sz * 0.34), 0.065, cloth, m_cl, seg=8)
            sphere(g, (x + 0.05, yb + 0.02, z + sz * 0.34), 0.07, sk if not helmet else rgb(25, 25, 28), m_cl, 8, 6)
    else:
        for sz in (-1, 1):
            cylinder(g, (x, yb + 0.52, z + sz * 0.29), (x + 0.42, yb + 0.36, z + sz * 0.22), 0.065, cloth, m_cl,
                     seg=8)
            sphere(g, (x + 0.44, yb + 0.36, z + sz * 0.22), 0.07, sk if not helmet else rgb(25, 25, 28), m_cl, 8, 6)
        revolve(g, [(0.2, -0.02), (0.21, 0.0), (0.2, 0.02)], (x + 0.5, yb + 0.4, z), (1, -0.5, 0),
                rgb(25, 25, 28), mat(M_DEF, 0.5), seg=14)
    cylinder(g, (x, yb + 0.6, z), (x, hy - 0.15, z), 0.08, sk, m_cl, seg=8)
    c1, c2 = C(d["c1"]), C(d["c2"])

    def dirs(P):
        return norm(P - np.asarray(hc, F4))
    if helmet:
        c3 = C(d["c3"])

        def cf(P):
            dd = dirs(P)
            Cc = np.empty(P.shape, F4)
            Cc[:] = c1
            Cc[np.abs(dd[..., 2]) < 0.16] = c2
            Cc[(np.abs(dd[..., 2]) < 0.06)] = c3
            Cc[(dd[..., 1] < -0.2) & (dd[..., 0] < 0.3)] = c2
            return Cc
        sphere(g, hc, 0.3, c1, mat(M_PAINT, 0.2), 18, 12, colfn=cf)
        superellipsoid(g, (x + 0.2, hy + 0.02, z), (0.12, 0.11, 0.22), 0.6, rgb(15, 20, 30), mat(M_GLASS, 0.05),
                       12, 8)
        return
    hair = C(d["c1"]) if d["hat"] == "bun" else rgb(30, 22, 15)

    def cf(P):
        dd = dirs(P)
        Cc = np.empty(P.shape, F4)
        Cc[:] = sk
        Cc[(dd[..., 0] < -0.1) & (dd[..., 1] > -0.35)] = hair
        if d["hat"] == "bun":
            Cc[dd[..., 1] > 0.45] = hair
        return Cc
    sphere(g, hc, 0.28, sk, mat(M_DEF, 0.6), 18, 12, colfn=cf)
    for sz in (-0.1, 0.1):
        sphere(g, (x + 0.26, hy + 0.05, z + sz), 0.035, rgb(15, 15, 15), mat(M_DEF, 0.2), 8, 5)
    if d.get("mous"):
        superellipsoid(g, (x + 0.26, hy - 0.08, z), (0.04, 0.035, 0.15), 0.6, rgb(40, 25, 15), m_cl, 10, 6)
    if d.get("glasses"):
        for sz in (-0.1, 0.1):
            revolve(g, [(0.06, -0.01), (0.075, 0.0), (0.06, 0.01)], (x + 0.27, hy + 0.05, z + sz), (1, 0, 0),
                    rgb(20, 20, 22) if kind != "chuy" else rgb(180, 140, 40), mat(M_METAL, 0.3, 0.6), seg=10)
    hat = d["hat"]
    if hat == "cap":
        superellipsoid(g, (x - 0.01, hy + 0.12, z), (0.27, 0.17, 0.27), 0.8, c1, m_cl, 14, 8)
        superellipsoid(g, (x + 0.25, hy + 0.17, z), (0.16, 0.02, 0.19), 0.5, c1, m_cl, 10, 6)
        sphere(g, (x + 0.2, hy + 0.2, z), 0.05, c2, m_cl, 8, 5)
    elif hat == "capr":                                                    # gorra al reves
        superellipsoid(g, (x - 0.01, hy + 0.12, z), (0.27, 0.17, 0.27), 0.8, c2, m_cl, 14, 8)
        superellipsoid(g, (x - 0.27, hy + 0.15, z), (0.15, 0.02, 0.18), 0.5, c2, m_cl, 10, 6)
    elif hat == "sombrero":
        revolve(g, [(0.0, 0.0), (0.55, 0.0), (0.62, 0.06), (0.6, 0.075), (0.5, 0.03), (0.2, 0.04), (0.19, 0.28),
                    (0.1, 0.33), (0.0, 0.34)], (x, hy + 0.12, z), (0, 1, 0), c1, mat(M_DEF, 0.85), seg=18)
        revolve(g, [(0.205, 0.05), (0.205, 0.11)], (x, hy + 0.12, z), (0, 1, 0), c2, m_cl, seg=18)
    elif hat == "texana":
        revolve(g, [(0.0, 0.0), (0.42, 0.0), (0.46, 0.08), (0.43, 0.09), (0.36, 0.03), (0.21, 0.03), (0.2, 0.22),
                    (0.14, 0.27), (0.0, 0.24)], (x, hy + 0.13, z), (0, 1, 0), c1, mat(M_DEF, 0.8), seg=18)
        revolve(g, [(0.212, 0.04), (0.212, 0.09)], (x, hy + 0.13, z), (0, 1, 0), c2, m_cl, seg=18)
    elif hat == "bun":
        sphere(g, (x - 0.2, hy + 0.2, z), 0.13, hair, mat(M_DEF, 0.7), 10, 6)
        superellipsoid(g, (x - 0.08, yb + 0.6, z), (0.2, 0.08, 0.3), 0.6, c2, m_cl, 12, 6)   # rebozo


def driver(g, kind, x, y, z, s=1.0):
    """Personaje sentado (escalado)."""
    tmp = Geo()
    character2(tmp, kind, 0.0, 0.0, 0.0)
    g.merge(transform_geo(tmp, 0.0, (x, y, z), s))


# ================================================================ modelos nuevos
def _tsuru(p, body, glass, info):
    """Sedan de 3 volumenes de los 90 (tipo Tsuru), taxi guinda y oro con letrero en el techo."""
    lo, up_ = C(p["body"]), C(p["accent"])
    L = Loft([(-2.15, 0.34, 0.86, 0.72, 0.70), (-2.11, 0.27, 0.97, 0.78, 0.76), (-1.9, 0.25, 0.99, 0.8, 0.77),
              (-1.3, 0.24, 0.97, 0.81, 0.78), (0.75, 0.24, 0.93, 0.81, 0.78), (1.6, 0.25, 0.86, 0.8, 0.77),
              (2.07, 0.28, 0.82, 0.78, 0.75), (2.15, 0.36, 0.72, 0.72, 0.68)], e=0.22, sub=3)

    L.mesh(body, lo, PAINT, n=24, split=0.62, col2=up_)
    gl = Loft([(-1.33, 0.93, 0.97, 0.77, 0.76), (-0.85, 0.93, 1.36, 0.77, 0.62), (0.15, 0.92, 1.38, 0.78, 0.63),
               (0.82, 0.91, 0.93, 0.79, 0.78)], e=0.25, sub=4)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-0.9, 1.33, 1.41, 0.64, 0.57), (-0.75, 1.34, 1.43, 0.65, 0.58), (0.05, 1.35, 1.44, 0.66, 0.59),
          (0.2, 1.33, 1.42, 0.65, 0.58)], e=0.3, sub=2).mesh(body, up_, PAINT, n=16)

    def pillars(s):
        bar(body, (0.8, 0.93, s * 0.775), (0.17, 1.37, s * 0.625), 0.06, up_, PAINT_MATE)
        bar(body, (-0.3, 0.94, s * 0.775), (-0.3, 1.37, s * 0.64), 0.08, BLACK, TRIM)
        bar(body, (-0.86, 1.37, s * 0.62), (-1.3, 0.96, s * 0.76), 0.13, up_, PAINT_MATE)
        bar(body, (-1.3, 0.95, s * 0.79), (0.8, 0.935, s * 0.795), 0.025, CHROME_C, CHROME)   # moldura
    both(pillars)
    wells(body, L, (1.25, -1.25), 0.32, 0.38, 0.24)
    seam(body, L, [(0.75, 0.3), (0.72, 0.92)])
    seam(body, L, [(-0.33, 0.28), (-0.3, 0.92)])
    seam(body, L, [(-1.2, 0.36), (-1.28, 0.92)])
    seam(body, L, [(0.76, 0.3), (-1.2, 0.3)], w=0.01)
    for xh in (0.45, -0.6):                                              # manijas
        for s in (-1, 1):
            box(body, xh - 0.09, 0.8, s * (L.z_at(xh, 0.8) - 0.005), xh + 0.02, 0.83, s * (L.z_at(xh, 0.8) + 0.012),
                DARK, TRIM)
    top_seam(body, L, [(0.8, -0.7), (1.98, -0.68), (1.98, 0.68), (0.8, 0.7)])
    top_seam(body, L, [(-1.35, -0.7), (-2.05, -0.7), (-2.05, 0.7), (-1.35, 0.7)])
    # defensas grises, parrilla, faros rectangulares, calaveras
    superellipsoid(body, (2.12, 0.42, 0), (0.07, 0.11, 0.81), 0.3, GREY, PAINT_MATE, 16, 6)
    superellipsoid(body, (-2.12, 0.45, 0), (0.07, 0.11, 0.81), 0.3, GREY, PAINT_MATE, 16, 6)
    grille_slats(body, 2.15, 0.6, 0.72, -0.3, 0.3, 3)
    both(lambda s: lamp(body, (2.12, 0.66, s * 0.53), (0.04, 0.065, 0.2), HEAD_C, HEAD_M, 0.25))
    both(lambda s: lamp(body, (2.09, 0.66, s * 0.77), (0.05, 0.06, 0.035), AMBER_C, AMBER_M, 0.3))
    both(lambda s: lamp(body, (-2.13, 0.78, s * 0.52), (0.035, 0.08, 0.24), TAIL_C, TAIL_M, 0.25))
    box(body, -2.15, 0.72, -0.27, -2.13, 0.84, 0.27, rgb(60, 10, 15), PAINT_MATE)
    plate(body, 2.19, 0.43, True)
    plate(body, -2.2, 0.62, False)
    mirrors(body, 0.66, 0.92, 0.82, up_)
    wipers(body, 0.78, 0.94, (0.55, -0.05))
    if p.get("police"):                                    # patrulla: leyenda en las puertas (la torreta la pone el modo persecucion)
        side_text(body, "POLICIA", -0.15, 0.55, 0.835, 0.1, up_)
        side_text(body, "P-" + str(p.get("num", 12)), -0.85, 0.46, 0.836, 0.07, rgb(20, 60, 170))
    else:
        # letrero TAXI en el techo
        box(body, -0.42, 1.43, -0.32, -0.22, 1.47, 0.32, DARK, TRIM)
        superellipsoid(body, (-0.32, 1.55, 0), (0.09, 0.09, 0.34), 0.3, (1.0, 0.92, 0.6), mat(M_EMIS, 0.3, 0, 1.2),
                       12, 6)
        text(body, "TAXI", (-0.225, 1.55, 0), (0, 0, -1), (0, 1, 0), 0.1, BLACK, TRIM)
        text(body, "TAXI", (-0.415, 1.55, 0), (0, 0, 1), (0, 1, 0), 0.1, BLACK, TRIM)
        side_text(body, "TAXI", -0.15, 0.55, 0.835, 0.1, up_)
        side_text(body, "A-" + str(p.get("num", 27)), -0.85, 0.46, 0.836, 0.07, WHITE)
    driver(body, p["driver"], -0.3, 0.28, -0.33, 0.8)
    info["wheels"] = [(1.25, 0.78), (-1.25, 0.78)]
    info["exhaust"] = [(-2.2, 0.3, 0.5)]
    info["head"] = [(2.2, 0.66, s * 0.53) for s in (-1, 1)]
    return 0.32, 0.2, make_wheel("hubcap", 0.32, 0.2, rim=rgb(200, 200, 205))


def _f1(p, body, glass, info, f2=False):
    """Monoplaza: F1 (2022+: nariz baja, pontones de efecto suelo, tapas en rines) o F2 (nariz alta)."""
    col, acc = C(p["body"]), C(p["accent"])
    red = rgb(210, 30, 40)
    if not f2:
        mono = Loft([(-2.35, 0.2, 0.42, 0.1, 0.08), (-1.6, 0.13, 0.62, 0.2, 0.12), (-0.9, 0.12, 0.88, 0.3, 0.14),
                     (-0.38, 0.12, 1.02, 0.33, 0.12), (-0.18, 0.12, 1.0, 0.33, 0.13), (0.02, 0.12, 0.68, 0.34, 0.3),
                     (0.9, 0.14, 0.62, 0.27, 0.2), (1.7, 0.17, 0.5, 0.17, 0.13), (2.38, 0.17, 0.3, 0.1, 0.06)],
                    e=0.45, sub=3)
    else:
        mono = Loft([(-2.3, 0.22, 0.45, 0.11, 0.09), (-1.6, 0.14, 0.64, 0.22, 0.14), (-0.9, 0.13, 0.86, 0.3, 0.15),
                     (-0.38, 0.13, 1.0, 0.32, 0.12), (-0.18, 0.13, 0.98, 0.32, 0.13), (0.02, 0.13, 0.7, 0.34, 0.3),
                     (0.9, 0.18, 0.66, 0.26, 0.2), (1.8, 0.3, 0.6, 0.15, 0.13), (2.3, 0.33, 0.5, 0.09, 0.07)],
                    e=0.5, sub=3)
    mono.mesh(body, col, PAINT, n=24)
    # pontones
    for s in (-1, 1):
        if not f2:
            sp = Loft([(-1.55, 0.14, 0.24, 0.06, 0.04), (-0.9, 0.12, 0.42, 0.2, 0.12), (-0.2, 0.12, 0.6, 0.25, 0.2),
                       (0.3, 0.13, 0.62, 0.25, 0.21), (0.34, 0.2, 0.58, 0.2, 0.17)], e=0.35, sub=3, zc=s * 0.5)
        else:
            sp = Loft([(-1.4, 0.14, 0.3, 0.08, 0.06), (-0.8, 0.12, 0.5, 0.21, 0.18), (0.1, 0.12, 0.58, 0.24, 0.21),
                       (0.5, 0.14, 0.56, 0.23, 0.2), (0.53, 0.2, 0.5, 0.19, 0.16)], e=0.6, sub=3, zc=s * 0.48)
        sp.mesh(body, col, PAINT, n=16)
        xi = 0.31 if not f2 else 0.5
        superellipsoid(body, (xi + 0.02, 0.4, s * (0.5 if not f2 else 0.48)), (0.03, 0.15, 0.17), 0.35, BLACK, TRIM,
                       12, 6)                                                      # toma de aire
        if not f2:                                                                 # franja blanca en pontones
            sp.patch(body, -0.9, 0.25, 0.9, 1.3, acc, PAINT_MATE, nx=6, nph=4)
    # piso, difusor
    prism_xz(body, [(1.3, -0.3), (1.3, 0.3), (0.6, 0.62), (-1.9, 0.62), (-1.9, -0.62), (0.6, -0.62)], 0.07, 0.11,
             BLACK, CARBON)
    for s in (-1, 1):
        prism_xy(body, [(-1.9, 0.07), (-2.45, 0.24), (-2.45, 0.28), (-1.9, 0.11)], s * 0.6, s * 0.58, BLACK, CARBON)
    prism_xy(body, [(-1.9, 0.07), (-2.45, 0.24), (-2.45, 0.27), (-1.9, 0.1)], -0.62, 0.62, rgb(25, 25, 28), CARBON)
    # alerón delantero
    if not f2:
        for k, (x0, x1, y0, dy) in enumerate([(2.2, 2.6, 0.08, 0.0), (2.12, 2.36, 0.12, 0.05),
                                              (2.06, 2.22, 0.17, 0.07)]):
            col_w = acc if k == 1 else col
            prism_xy(body, [(x0, y0 + dy), (x1, y0), (x1, y0 + 0.018), (x0, y0 + dy + 0.02)], -0.97, 0.97, col_w,
                     PAINT_MATE)
        for s in (-1, 1):
            prism_xy(body, [(1.95, 0.07), (2.62, 0.07), (2.62, 0.2), (2.2, 0.3), (1.95, 0.3)], s * 0.97, s * 0.99,
                     col, PAINT_MATE)
    else:
        for k, (x0, x1, y0, dy) in enumerate([(2.15, 2.6, 0.09, 0.0), (2.1, 2.35, 0.14, 0.07)]):
            prism_xy(body, [(x0, y0 + dy), (x1, y0), (x1, y0 + 0.03), (x0, y0 + dy + 0.035)], -0.95, 0.95,
                     acc if k == 0 else col, PAINT_MATE)
        for s in (-1, 1):
            prism_xy(body, [(2.0, 0.08), (2.6, 0.08), (2.55, 0.4), (2.1, 0.4)], s * 0.93, s * 0.96, acc, PAINT_MATE)
            bar(body, (2.2, 0.36, s * 0.07), (2.25, 0.15, s * 0.12), 0.03, BLACK, CARBON)   # pilares de la nariz
    # alerón trasero
    zr = 0.5
    for s in (-1, 1):
        if not f2:
            prism_xy(body, [(-2.05, 0.42), (-2.62, 0.42), (-2.66, 1.02), (-2.12, 0.98)], s * zr, s * (zr + 0.02), col,
                     PAINT_MATE)
        else:
            prism_xy(body, [(-2.12, 0.55), (-2.6, 0.5), (-2.64, 1.0), (-2.12, 1.0)], s * zr, s * (zr + 0.02), acc,
                     PAINT_MATE)
    prism_xy(body, [(-2.18, 0.86), (-2.52, 0.83), (-2.52, 0.87), (-2.2, 0.9)], -zr, zr, acc if not f2 else col,
             PAINT_MATE)
    prism_xy(body, [(-2.45, 0.92), (-2.64, 0.97), (-2.64, 1.0), (-2.47, 0.96)], -zr, zr, BLACK, CARBON)
    if not f2:
        prism_xy(body, [(-2.25, 0.45), (-2.5, 0.43), (-2.5, 0.46), (-2.26, 0.48)], -zr, zr, BLACK, CARBON)  # beam wing
    bar(body, (-2.2, 0.4, 0), (-2.4, 0.84, 0), 0.04, BLACK, CARBON)
    # aleta del motor y toma de aire
    prism_xy(body, [(-0.6, 0.98), (-1.9, 0.6), (-1.9, 0.45), (-0.6, 0.85)], -0.012, 0.012, col, PAINT_MATE)
    disc(body, (-0.165, 0.92, 0), 0.085, (1, 0, 0), BLACK, TRIM, seg=10)
    # halo
    pts = [(0.55, 0.64, 0.0), (0.42, 0.86, 0.0)]
    halo = [(0.42, 0.86, 0.0), (0.3, 0.88, 0.2), (0.0, 0.86, 0.3), (-0.3, 0.84, 0.31), (-0.36, 0.7, 0.31)]
    tube(body, pts, 0.03, BLACK, CARBON, seg=8)
    tube(body, halo, 0.03, BLACK, CARBON, seg=8)
    tube(body, [(a, b, -c) for a, b, c in halo], 0.03, BLACK, CARBON, seg=8)
    # suspension (brazos al centro de la rueda)
    fx, rx, wz = (1.72, -1.55, 0.78)
    for wx in (fx, rx):
        for s in (-1, 1):
            for yy in (0.22, 0.44):
                bar(body, (wx + 0.22, yy, s * 0.22), (wx, 0.36, s * 0.62), 0.03, BLACK, CARBON, t2=0.015)
                bar(body, (wx - 0.22, yy, s * 0.22), (wx, 0.36, s * 0.62), 0.03, BLACK, CARBON, t2=0.015)
            bar(body, (wx, 0.36, s * 0.55), (wx, 0.36, s * 0.66), 0.08, GREY, ALU)
    # espejos, luz de lluvia, numero
    for s in (-1, 1):
        bar(body, (0.35, 0.6, s * 0.36), (0.32, 0.72, s * 0.52), 0.025, BLACK, CARBON)
        superellipsoid(body, (0.32, 0.74, s * 0.55), (0.04, 0.04, 0.08), 0.5, col, PAINT_MATE, 10, 6)
    lamp(body, (-2.37, 0.33, 0), (0.02, 0.04, 0.04), TAIL_C, TAIL_M, 0.3)
    num = str(p.get("num", 7))
    ny = mono.y_top(1.95) + 0.004
    text(body, num, (1.95, ny, 0), (0, 0, -1), (-1, 0.0, 0), 0.16, WHITE, PAINT_MATE)
    side_text(body, num, -1.1, 0.68, mono.side_z(-1.3, -0.9, 0.6, 0.76) + 0.005, 0.16, acc)
    if not f2:
        side_text(body, "GTC", -0.45, 0.3, 0.755, 0.1, WHITE)
        # franja roja en la nariz (bandera)
        mono.patch(body, 1.0, 2.0, 1.2, 1.94, red, PAINT_MATE, nx=8, nph=4)
    else:
        mono.patch(body, 0.2, 2.3, 1.25, 1.89, acc, PAINT_MATE, nx=10, nph=4)
    driver(body, p["driver"], 0.05, 0.14, 0.0, 0.78)
    info["wheels"] = [(fx, wz), (rx, wz)]
    info["exhaust"] = [(-2.42, 0.5, 0.0)]
    cylinder(body, (-2.2, 0.5, 0), (-2.44, 0.5, 0), 0.05, rgb(90, 80, 75), ALU, seg=10)
    info["head"] = [(2.4, 0.3, 0.0), (2.4, 0.3, 0.0)]
    if not f2:
        return 0.36, 0.36, make_wheel("cover", 0.36, 0.36, accent=acc)
    return 0.36, 0.34, make_wheel("f2", 0.36, 0.34, rim=rgb(30, 30, 34), accent=acc)


def _vocho(p, body, glass, info):
    """Vocho clasico: cuerpo redondo, salpicaderas separadas, estribos, faros redondos, defensas cromadas."""
    col, top = C(p["body"]), C(p["accent"])
    r, wz = 0.31, 0.63
    L = Loft([(-1.98, 0.42, 0.72, 0.42, 0.3, 0.7), (-1.88, 0.32, 0.9, 0.56, 0.42, 0.6),
              (-1.35, 0.3, 1.0, 0.6, 0.54, 0.5), (-0.9, 0.3, 1.01, 0.61, 0.58, 0.45), (0.65, 0.3, 0.99, 0.61, 0.58, 0.45),
              (1.35, 0.33, 0.88, 0.58, 0.47, 0.55), (1.86, 0.4, 0.66, 0.46, 0.32, 0.7), (1.98, 0.46, 0.56, 0.3, 0.2, 0.8)],
             sub=3)
    L.mesh(body, col, PAINT, n=24)
    gl = Loft([(-1.12, 0.95, 1.0, 0.6, 0.6, 0.6), (-0.65, 0.95, 1.43, 0.61, 0.5, 0.6), (0.1, 0.95, 1.49, 0.61, 0.52, 0.6),
               (0.72, 0.94, 0.99, 0.62, 0.61, 0.6)], sub=4)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-0.98, 1.17, 1.22, 0.58, 0.56, 0.6), (-0.7, 1.36, 1.46, 0.57, 0.47, 0.6), (0.05, 1.41, 1.52, 0.57, 0.48, 0.6),
          (0.42, 1.31, 1.4, 0.58, 0.52, 0.6)], sub=3).mesh(body, top, PAINT, n=20)

    def pillars(s):
        bar(body, (0.7, 0.97, s * 0.6), (0.35, 1.38, s * 0.53), 0.045, top, PAINT_MATE)
        bar(body, (-0.25, 0.98, s * 0.61), (-0.25, 1.47, s * 0.5), 0.05, top, PAINT_MATE)
        bar(body, (-0.7, 1.43, s * 0.5), (-1.05, 1.0, s * 0.6), 0.09, top, PAINT_MATE)
        bar(body, (-1.0, 0.98, s * 0.625), (0.66, 0.975, s * 0.625), 0.018, CHROME_C, CHROME)
    both(pillars)
    # salpicaderas, estribos
    fender(body, 1.2, r, -0.25, math.pi + 0.35, r + 0.04, 0.26, 0.44, 0.76, col)
    fender(body, -1.2, r, -0.3, math.pi + 0.15, r + 0.04, 0.28, 0.44, 0.77, col)
    both(lambda s: box(body, -0.9, 0.24, s * 0.56, 0.9, 0.31, s * 0.74, rgb(35, 35, 38), RUBBER))
    # faros redondos sobre las salpicaderas, cuernos de claxon, calaveras
    for s in (-1, 1):
        cylinder(body, (1.4, 0.66, s * 0.6), (1.56, 0.7, s * 0.6), 0.1, col, PAINT_MATE, seg=14)
        cylinder(body, (1.56, 0.7, s * 0.6), (1.58, 0.705, s * 0.6), 0.1, CHROME_C, CHROME, seg=14)
        disc(body, (1.585, 0.705, s * 0.6), 0.085, (1, 0.25, 0), HEAD_C, HEAD_M, seg=14)
        lamp(body, (1.5, 0.5, s * 0.44), (0.03, 0.05, 0.06), CHROME_C, CHROME, 0.5)
        lamp(body, (1.25, 0.77, s * 0.72), (0.05, 0.025, 0.03), AMBER_C, AMBER_M, 0.5)
        lamp(body, (-1.68, 0.66, s * 0.6), (0.06, 0.1, 0.06), TAIL_C, TAIL_M, 0.5)
        cylinder(body, (-1.96, 0.33, s * 0.18), (-2.12, 0.33, s * 0.18), 0.03, CHROME_C, CHROME, seg=8)
    # defensas de navaja con cuernitos
    for x, sx in ((1.98, 1), (-1.97, -1)):
        superellipsoid(body, (x, 0.42, 0), (0.035, 0.045, 0.78), 0.3, CHROME_C, CHROME, 14, 6)
        for s in (-1, 1):
            bar(body, (x - 0.12 * sx, 0.42, s * 0.5), (x, 0.42, s * 0.5), 0.04, CHROME_C, CHROME)
            bar(body, (x, 0.36, s * 0.32), (x + 0.02 * sx, 0.58, s * 0.32), 0.035, CHROME_C, CHROME)
    plate(body, 1.97, 0.53, True, w=0.3, h=0.1)
    plate(body, -1.95, 0.56, False, w=0.3, h=0.1)
    # cofre con moldura, rejillas del motor
    top_seam(body, L, [(0.75, 0.0), (1.85, 0.0)], w=0.03, col=CHROME_C)
    top_seam(body, L, [(0.7, -0.45), (1.7, -0.3), (1.85, 0.0), (1.7, 0.3), (0.7, 0.45)])
    for k in range(5):
        yy = 0.8 + k * 0.035
        xx = -1.78 - k * 0.02
        box(body, xx - 0.01, yy, -0.22, xx + 0.012, yy + 0.015, 0.22, BLACK, TRIM)
    seam(body, L, [(0.62, 0.34), (0.66, 0.95)])
    seam(body, L, [(-0.62, 0.34), (-0.6, 0.95)])
    seam(body, L, [(0.62, 0.34), (-0.62, 0.34)], w=0.01)
    both(lambda s: box(body, -0.45, 0.8, s * 0.6, -0.33, 0.83, s * 0.635, CHROME_C, CHROME))
    mirrors(body, 0.62, 0.95, 0.63, CHROME_C, CHROME, 0.8)
    wipers(body, 0.7, 0.98, (0.42, -0.08), L=0.35)
    # letrero de taxi y leyenda
    box(body, -0.2, 1.5, -0.25, -0.02, 1.53, 0.25, DARK, TRIM)
    superellipsoid(body, (-0.11, 1.6, 0), (0.07, 0.08, 0.27), 0.3, (1.0, 0.95, 0.75), mat(M_EMIS, 0.3, 0, 1.2), 12, 6)
    text(body, "TAXI", (-0.035, 1.6, 0), (0, 0, -1), (0, 1, 0), 0.08, rgb(20, 90, 40), TRIM)
    text(body, "TAXI", (-0.185, 1.6, 0), (0, 0, 1), (0, 1, 0), 0.08, rgb(20, 90, 40), TRIM)
    side_text(body, "LIBRE", 0.0, 0.62, L.side_z(-0.3, 0.3, 0.57, 0.67) + 0.004, 0.09, WHITE)
    driver(body, p["driver"], -0.3, 0.3, -0.3, 0.75)
    info["wheels"] = [(1.2, wz), (-1.2, wz)]
    info["exhaust"] = [(-2.12, 0.33, -0.18), (-2.12, 0.33, 0.18)]
    info["head"] = [(1.6, 0.7, s * 0.6) for s in (-1, 1)]
    return r, 0.19, make_wheel("steel", r, 0.19, rim=WHITE)


def _chevy(p, body, glass, info):
    """Chevy Pop: hatchback chico de 3 puertas, redondito, con olla tamalera en la parrilla del techo."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.31, 0.69
    L = Loft([(-1.88, 0.36, 0.94, 0.72, 0.64), (-1.83, 0.27, 1.01, 0.78, 0.71), (-1.3, 0.25, 0.99, 0.8, 0.74),
              (0.55, 0.25, 0.9, 0.8, 0.74), (1.4, 0.27, 0.8, 0.77, 0.7), (1.84, 0.3, 0.69, 0.72, 0.62),
              (1.93, 0.37, 0.57, 0.62, 0.5)], e=0.36, sub=3)
    L.mesh(body, col, PAINT, n=24)
    gl = Loft([(-1.8, 0.97, 1.0, 0.72, 0.71), (-1.5, 0.97, 1.3, 0.73, 0.6), (-0.95, 0.95, 1.37, 0.75, 0.6),
               (0.05, 0.93, 1.36, 0.76, 0.62), (0.62, 0.89, 0.91, 0.77, 0.76)], e=0.4, sub=3)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-1.45, 1.25, 1.31, 0.6, 0.55), (-1.25, 1.3, 1.38, 0.62, 0.56), (-0.15, 1.34, 1.42, 0.64, 0.58),
          (0.1, 1.31, 1.39, 0.63, 0.57)], e=0.4, sub=2).mesh(body, col, PAINT, n=16)

    def pillars(s):
        bar(body, (0.6, 0.91, s * 0.765), (0.08, 1.36, s * 0.62), 0.06, col, PAINT_MATE)
        bar(body, (-0.68, 0.96, s * 0.76), (-0.68, 1.38, s * 0.62), 0.07, col, PAINT_MATE)
        prism_xy(body, [(-1.05, 1.36), (-1.42, 1.3), (-1.78, 0.98), (-1.2, 0.97)], s * 0.7, s * 0.735, col,
                 PAINT_MATE)                                                   # pilar C ancho del hatch
        bar(body, (-1.65, 0.96, s * 0.785), (0.6, 0.925, s * 0.79), 0.02, DARK, TRIM)
    both(pillars)
    wells(body, L, (1.2, -1.2), r, 0.37, 0.25)
    seam(body, L, [(0.58, 0.3), (0.56, 0.9)])
    seam(body, L, [(-0.66, 0.32), (-0.66, 0.94)])
    seam(body, L, [(0.58, 0.3), (-0.66, 0.32)], w=0.01)
    zh = L.z_at(-0.45, 0.8)
    both(lambda s: box(body, -0.52, 0.8, s * (zh - 0.004), -0.38, 0.83, s * (zh + 0.012), DARK, TRIM))
    top_seam(body, L, [(0.62, -0.68), (1.8, -0.62), (1.8, 0.62), (0.62, 0.68)])
    # defensas de plastico, faros ovalados, parrilla, calaveras verticales
    superellipsoid(body, (1.88, 0.4, 0), (0.08, 0.11, 0.77), 0.4, rgb(55, 57, 62), PAINT_MATE, 16, 6)
    superellipsoid(body, (-1.86, 0.42, 0), (0.08, 0.11, 0.79), 0.4, rgb(55, 57, 62), PAINT_MATE, 16, 6)
    both(lambda s: lamp(body, (1.86, 0.63, s * 0.48), (0.05, 0.075, 0.17), HEAD_C, HEAD_M, 0.6))
    both(lambda s: lamp(body, (1.82, 0.63, s * 0.68), (0.05, 0.05, 0.04), AMBER_C, AMBER_M, 0.5))
    box(body, 1.9, 0.6, -0.26, 1.925, 0.68, 0.26, DARK, TRIM)
    box(body, 1.92, 0.632, -0.06, 1.935, 0.648, 0.06, CHROME_C, CHROME)
    both(lambda s: lamp(body, (-1.86, 0.76, s * 0.62), (0.035, 0.14, 0.09), TAIL_C, TAIL_M, 0.35))
    plate(body, 1.97, 0.42, True, w=0.3, h=0.1)
    plate(body, -1.89, 0.66, False, w=0.3, h=0.1)
    mirrors(body, 0.5, 0.9, 0.79, col)
    wipers(body, 0.62, 0.92, (0.55, -0.05), L=0.42)
    # parrilla del techo + olla tamalera
    for xx in (-1.05, -0.15):
        bar(body, (xx, 1.47, -0.6), (xx, 1.47, 0.6), 0.04, BLACK, TRIM)
        for s in (-1, 1):
            bar(body, (xx, 1.38, s * 0.58), (xx, 1.47, s * 0.58), 0.04, BLACK, TRIM)
    both(lambda s: bar(body, (-1.15, 1.43, s * 0.6), (-0.05, 1.43, s * 0.6), 0.03, BLACK, TRIM))
    revolve(body, [(0.0, 0.0), (0.21, 0.0), (0.22, 0.02), (0.21, 0.3), (0.23, 0.32), (0.2, 0.35), (0.09, 0.37),
                   (0.04, 0.42), (0.0, 0.42)], (-0.6, 1.49, 0.0), (0, 1, 0), rgb(195, 198, 205), ALU, seg=16)
    both(lambda s: bar(body, (-0.6, 1.7, s * 0.21), (-0.6, 1.7, s * 0.26), 0.03, DARK, TRIM))
    for k in range(5):                                                   # hojas de maiz asomandose
        a = k / 5 * TAU
        superellipsoid(body, (-0.6 + 0.1 * math.cos(a), 1.92, 0.1 * math.sin(a)), (0.035, 0.025, 0.06), 0.6,
                       rgb(225, 205, 140), mat(M_DEF, 0.8), 8, 4)
    side_text(body, "TAMALES", -0.05, 0.56, L.side_z(-0.45, 0.35, 0.5, 0.62) + 0.004, 0.1, acc)
    driver(body, p["driver"], -0.35, 0.3, -0.33, 0.78)
    info["wheels"] = [(1.2, wz), (-1.2, wz)]
    info["exhaust"] = [(-1.95, 0.3, 0.45)]
    info["head"] = [(1.95, 0.63, s * 0.48) for s in (-1, 1)]
    return r, 0.19, make_wheel("steel", r, 0.19, rim=rgb(170, 172, 178))


def _vette(p, body, glass, info):
    """GT americano V8 (tipo Corvette): cofre largo, fastback, cuatro calaveras redondas, 4 escapes al centro."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.36, 0.8
    L = Loft([(-2.3, 0.36, 0.8, 0.76, 0.7), (-2.24, 0.28, 0.94, 0.88, 0.8), (-1.8, 0.26, 0.92, 0.96, 0.82),
              (-1.3, 0.25, 0.86, 0.97, 0.8), (-0.4, 0.25, 0.8, 0.9, 0.78), (0.3, 0.25, 0.8, 0.92, 0.8),
              (1.38, 0.25, 0.73, 0.96, 0.8), (2.0, 0.24, 0.6, 0.91, 0.72), (2.32, 0.26, 0.44, 0.72, 0.5)],
             e=0.28, sub=3)
    L.mesh(body, col, PAINT, n=24)
    gl = Loft([(-1.98, 0.88, 0.9, 0.74, 0.72), (-0.95, 0.8, 1.17, 0.75, 0.56), (-0.25, 0.79, 1.22, 0.76, 0.57),
               (0.4, 0.77, 0.79, 0.78, 0.76)], e=0.3, sub=4)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-0.95, 1.12, 1.19, 0.58, 0.52), (-0.75, 1.15, 1.23, 0.59, 0.53), (-0.2, 1.17, 1.25, 0.59, 0.53),
          (-0.02, 1.15, 1.22, 0.58, 0.52)], e=0.3, sub=2).mesh(body, acc, PAINT, n=16)

    def pillars(s):
        bar(body, (0.38, 0.79, s * 0.77), (-0.08, 1.2, s * 0.6), 0.05, acc, PAINT_MATE)
        prism_xy(body, [(-0.7, 1.19), (-1.0, 1.13), (-1.9, 0.9), (-0.85, 0.85)], s * 0.7, s * 0.745, col,
                 PAINT_MATE)                                               # panel lateral (pilar C)
        prism_xy(body, [(-0.75, 1.1), (-0.98, 1.06), (-1.3, 0.94), (-0.88, 0.93)], s * 0.744, s * 0.75, BLACK,
                 GLASS_M)                                                  # ventanita trasera
    both(pillars)
    wells(body, L, (1.38, -1.32), r, 0.41, 0.25)
    # cofre: domo central, rejilla y branquias laterales
    top_seam(body, L, [(0.45, -0.3), (1.9, -0.18), (1.9, 0.18), (0.45, 0.3)], w=0.015)
    Loft([(0.6, 0.7, 0.81, 0.2, 0.15), (1.0, 0.7, 0.8, 0.24, 0.18), (1.6, 0.64, 0.72, 0.2, 0.14)], e=0.4,
         sub=2).mesh(body, col, PAINT, n=12)
    box(body, 1.05, L.y_top(1.1) + 0.005, -0.12, 1.3, L.y_top(1.2) + 0.012, 0.12, BLACK, TRIM)
    for k in range(3):
        xa = 0.65 + k * 0.08
        seam(body, L, [(xa + 0.025, 0.42), (xa + 0.095, 0.6)], w=0.035)
    seam(body, L, [(0.33, 0.3), (0.38, 0.78)])
    seam(body, L, [(-0.85, 0.3), (-0.8, 0.8)])
    # nariz: faros afilados, toma inferior, splitter
    both(lambda s: lamp(body, (2.08, 0.55, s * 0.62), (0.16, 0.035, 0.18), HEAD_C, HEAD_M, 0.25))
    both(lambda s: lamp(body, (2.16, 0.52, s * 0.72), (0.08, 0.015, 0.05), AMBER_C, AMBER_M, 0.3))
    superellipsoid(body, (2.27, 0.35, 0), (0.05, 0.06, 0.5), 0.3, BLACK, TRIM, 14, 6)
    prism_xz(body, [(2.0, -0.85), (2.38, -0.6), (2.38, 0.6), (2.0, 0.85)], 0.2, 0.23, BLACK, CARBON)
    # cola: cuatro calaveras redondas, difusor y 4 escapes al centro
    for s in (-1, 1):
        for zc in (0.33, 0.58):
            disc(body, (-2.305, 0.66, s * zc), 0.095, (-1, 0, 0), CHROME_C, CHROME, seg=16)
            disc(body, (-2.31, 0.66, s * zc), 0.08, (-1, 0, 0), TAIL_C, TAIL_M, seg=16)
            disc(body, (-2.315, 0.66, s * zc), 0.035, (-1, 0, 0), rgb(60, 5, 5), PAINT_MATE, seg=10)
    superellipsoid(body, (-2.22, 0.3, 0), (0.1, 0.05, 0.72), 0.3, BLACK, TRIM, 14, 6)
    for zc in (-0.21, -0.07, 0.07, 0.21):
        cylinder(body, (-2.1, 0.32, zc), (-2.36, 0.32, zc), 0.045, CHROME_C, CHROME, seg=10)
        disc(body, (-2.361, 0.32, zc), 0.035, (-1, 0, 0), BLACK, TRIM, seg=10)
    box(body, -2.26, 0.92, -0.75, -2.18, 0.97, 0.75, acc, PAINT_MATE)          # coletita (ducktail)
    plate(body, -2.32, 0.48, False, w=0.3, h=0.1)
    mirrors(body, 0.25, 0.82, 0.86, col)
    both(lambda s: box(body, -1.0, 0.21, s * 0.86, 0.95, 0.27, s * 0.95, BLACK, CARBON))     # estribos
    wipers(body, 0.38, 0.8, (0.5, -0.1), L=0.45)
    driver(body, p["driver"], -0.55, 0.18, -0.33, 0.76)
    info["wheels"] = [(1.38, wz), (-1.32, wz)]
    info["exhaust"] = [(-2.38, 0.32, -0.14), (-2.38, 0.32, 0.14)]
    info["head"] = [(2.2, 0.55, s * 0.62) for s in (-1, 1)]
    return r, 0.3, make_wheel("spoke5", r, 0.3, rim=rgb(40, 40, 44))


def _nascar(p, body, glass, info):
    """Stock car tipo NASCAR: carroceria cuadrada de dos tonos, numerotes en puertas y techo, spoiler."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.36, 0.82
    num = str(p.get("num", 52))
    L = Loft([(-2.5, 0.3, 0.92, 0.84, 0.78), (-2.45, 0.21, 1.02, 0.95, 0.88), (-1.8, 0.19, 1.03, 0.97, 0.9),
              (-1.2, 0.19, 1.0, 0.98, 0.9), (0.7, 0.19, 0.95, 0.98, 0.9), (1.6, 0.19, 0.86, 0.97, 0.88),
              (2.3, 0.18, 0.74, 0.94, 0.82), (2.5, 0.2, 0.55, 0.85, 0.7)], e=0.25, sub=3)
    L.mesh(body, acc, PAINT, n=24, split=0.52, col2=col)
    seam(body, L, [(-2.45, 0.52), (2.45, 0.52)], w=0.03, col=WHITE)
    gl = Loft([(-1.45, 0.98, 1.0, 0.86, 0.85), (-0.75, 0.98, 1.36, 0.84, 0.66), (0.0, 0.96, 1.38, 0.85, 0.67),
               (0.75, 0.93, 0.95, 0.88, 0.87)], e=0.28, sub=4)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    roof = Loft([(-0.8, 1.32, 1.4, 0.68, 0.62), (-0.65, 1.33, 1.42, 0.69, 0.63), (-0.05, 1.34, 1.43, 0.7, 0.64),
                 (0.1, 1.32, 1.4, 0.69, 0.63)], e=0.3, sub=2)
    roof.mesh(body, col, PAINT, n=16)

    def pillars(s):
        bar(body, (0.74, 0.95, s * 0.88), (0.05, 1.38, s * 0.67), 0.06, col, PAINT_MATE)
        bar(body, (-0.72, 1.37, s * 0.66), (-1.42, 0.99, s * 0.86), 0.14, col, PAINT_MATE)
        tube(body, [(0.5, 0.98, s * 0.6), (0.0, 1.3, s * 0.5), (-0.6, 1.3, s * 0.5), (-1.0, 1.0, s * 0.6)], 0.025,
             GREY, ALU, seg=6, joints=False)
    both(pillars)
    tube(body, [(0.0, 1.3, -0.5), (0.0, 1.3, 0.5)], 0.025, GREY, ALU, seg=6, joints=False)
    wells(body, L, (1.4, -1.4), r, 0.41, 0.19)
    # numerotes: puertas (con contorno negro) y techo
    zs = L.side_z(-0.6, 0.4, 0.5, 0.97) + 0.006
    for s in (-1, 1):
        rgt = (1, 0, 0) if s > 0 else (-1, 0, 0)
        text(body, num, (-0.1 + 0.03 * s * rgt[0] * s, 0.71, s * zs), rgt, (0, 1, 0), 0.45, BLACK, PAINT_MATE)
        text(body, num, (-0.1, 0.74, s * (zs + 0.004)), rgt, (0, 1, 0), 0.45, WHITE, PAINT_MATE)
    text(body, num, (-0.38, 1.433, 0.03), (1, 0, 0), (0, 0, -1), 0.6, BLACK, PAINT_MATE)
    text(body, num, (-0.35, 1.437, 0), (1, 0, 0), (0, 0, -1), 0.6, WHITE, PAINT_MATE)
    # spoiler, splitter, calcomanias de faros y parrilla
    prism_xy(body, [(-2.42, 1.0), (-2.5, 1.0), (-2.53, 1.24), (-2.47, 1.24)], -0.9, 0.9, BLACK, CARBON)
    both(lambda s: prism_xy(body, [(-2.1, 1.0), (-2.5, 1.0), (-2.53, 1.26), (-2.3, 1.06)], s * 0.9, s * 0.92, col,
                            PAINT_MATE))
    prism_xz(body, [(2.2, -0.95), (2.62, -0.8), (2.62, 0.8), (2.2, 0.95)], 0.12, 0.16, BLACK, CARBON)
    both(lambda s: lamp(body, (2.38, 0.68, s * 0.6), (0.1, 0.05, 0.2), (1.0, 0.95, 0.6), HEAD_M, 0.25))
    grille_slats(body, 2.53, 0.3, 0.5, -0.45, 0.45, 4, slat=rgb(40, 40, 44), sm=TRIM)
    both(lambda s: lamp(body, (-2.47, 0.82, s * 0.55), (0.03, 0.06, 0.24), TAIL_C, TAIL_M, 0.25))
    decal_hd(body, -1.95, L.z_at(-1.95, 0.75) + 0.006, 0.9, 0.62, 0.84, 15)
    decal_hd(body, 1.85, L.z_at(1.85, 0.66) + 0.006, 0.7, 0.55, 0.725, 3)
    zc = L.z_at(1.1, 0.35)
    for k, cc in enumerate((rgb(250, 210, 30), WHITE, rgb(30, 160, 230), rgb(230, 30, 40))):
        for s in (-1, 1):                                                 # calcomanias de patrocinadores chicos
            box(body, 1.0 + k * 0.17, 0.26, s * (zc - 0.002), 1.14 + k * 0.17, 0.44, s * (zc + 0.008), cc, PAINT_MATE)
    # red de ventana (lado del piloto)
    for k in range(5):
        bar(body, (-0.55 + k * 0.2, 1.0, -0.84), (-0.55 + k * 0.2, 1.3, -0.72), 0.012, BLACK, TRIM)
    for k in range(3):
        bar(body, (-0.6, 1.05 + k * 0.1, -0.83 + k * 0.04), (0.3, 1.05 + k * 0.1, -0.83 + k * 0.04), 0.012, BLACK,
            TRIM)
    mirrors(body, 0.5, 0.97, 0.88, col, size=0.8)
    driver(body, p["driver"], -0.3, 0.2, -0.35, 0.8)
    info["wheels"] = [(1.4, wz), (-1.4, wz)]
    info["exhaust"] = [(-0.6, 0.22, 1.0)]
    cylinder(body, (-0.4, 0.22, 0.9), (-0.62, 0.22, 1.0), 0.06, CHROME_C, CHROME, seg=10)
    info["head"] = [(2.45, 0.68, s * 0.6) for s in (-1, 1)]
    return r, 0.32, make_wheel("nascar", r, 0.32, rim=rgb(25, 25, 28))


def _gtr(p, body, glass, info):
    """Superdeportivo japones tipo GT-R: 4 calaveras redondas, pilar C con 'navaja', aleron grande."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.36, 0.8
    L = Loft([(-2.35, 0.36, 0.92, 0.76, 0.7), (-2.29, 0.28, 1.0, 0.86, 0.78), (-1.8, 0.26, 1.02, 0.95, 0.8),
              (-1.2, 0.25, 0.98, 0.95, 0.78), (0.55, 0.25, 0.92, 0.92, 0.78), (1.4, 0.25, 0.84, 0.95, 0.8),
              (2.15, 0.25, 0.72, 0.9, 0.74), (2.36, 0.3, 0.56, 0.78, 0.6)], e=0.28, sub=3)
    L.mesh(body, col, PAINT, n=24)
    gl = Loft([(-1.95, 0.98, 1.01, 0.74, 0.73), (-1.0, 0.97, 1.33, 0.75, 0.58), (-0.1, 0.95, 1.37, 0.76, 0.6),
               (0.62, 0.91, 0.93, 0.79, 0.78)], e=0.3, sub=4)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-1.02, 1.29, 1.36, 0.6, 0.55), (-0.85, 1.31, 1.39, 0.62, 0.56), (-0.15, 1.33, 1.41, 0.63, 0.57),
          (0.0, 1.31, 1.38, 0.62, 0.56)], e=0.3, sub=2).mesh(body, col, PAINT, n=16)

    def pillars(s):
        bar(body, (0.6, 0.92, s * 0.78), (-0.02, 1.36, s * 0.61), 0.05, col, PAINT_MATE)
        bar(body, (-0.98, 1.34, s * 0.58), (-1.75, 0.99, s * 0.76), 0.16, col, PAINT_MATE)
        bar(body, (-1.25, 1.2, s * 0.7), (-1.62, 1.0, s * 0.79), 0.03, acc, CHROME, t2=0.06)   # navaja
        bar(body, (-0.42, 0.96, s * 0.78), (-0.42, 1.38, s * 0.62), 0.07, BLACK, TRIM)
    both(pillars)
    wells(body, L, (1.4, -1.4), r, 0.41, 0.25)
    seam(body, L, [(0.55, 0.3), (0.58, 0.9)])
    seam(body, L, [(-0.5, 0.3), (-0.45, 0.95)])
    seam(body, L, [(-1.0, 0.3), (-1.03, 0.97)])
    # parrilla trapezoidal, faros bumeran, tomas, ductos del cofre
    prism_xy(body, [(2.3, 0.3), (2.37, 0.3), (2.35, 0.56), (2.28, 0.56)], -0.32, 0.32, BLACK, TRIM)
    for k in range(4):
        box(body, 2.345, 0.34 + k * 0.055, -0.3, 2.365, 0.355 + k * 0.055, 0.3, rgb(60, 60, 66), ALU)
    bar(body, (2.33, 0.57, -0.32), (2.33, 0.57, 0.32), 0.02, CHROME_C, CHROME)
    for s in (-1, 1):
        lamp(body, (2.16, 0.64, s * 0.58), (0.14, 0.05, 0.2), HEAD_C, HEAD_M, 0.3)
        lamp(body, (2.24, 0.6, s * 0.72), (0.06, 0.06, 0.05), HEAD_C, HEAD_M, 0.3)
        lamp(body, (2.27, 0.4, s * 0.62), (0.05, 0.06, 0.13), BLACK, TRIM, 0.3)
        lamp(body, (2.27, 0.32, s * 0.64), (0.03, 0.012, 0.12), HEAD_C, LED_M, 0.3)
        box(body, 1.1, L.y_top(1.1, s * 0.3) - 0.01, s * 0.26 - 0.06, 1.35, L.y_top(1.3, s * 0.3) + 0.008,
            s * 0.26 + 0.06, BLACK, TRIM)
    prism_xz(body, [(2.1, -0.85), (2.42, -0.7), (2.42, 0.7), (2.1, 0.85)], 0.22, 0.25, BLACK, CARBON)
    # cola: 4 anillos rojos, difusor, 4 escapes de titanio
    for s in (-1, 1):
        for zc in (0.36, 0.6):
            disc(body, (-2.355, 0.8, s * zc), 0.095, (-1, 0, 0), TAIL_C, TAIL_M, seg=18, r_in=0.05)
            disc(body, (-2.35, 0.8, s * zc), 0.05, (-1, 0, 0), rgb(40, 5, 5), PAINT_MATE, seg=12)
        for zc in (0.48, 0.64):
            cylinder(body, (-2.15, 0.33, s * zc), (-2.42, 0.33, s * zc), 0.055, rgb(120, 110, 140), ALU, seg=10)
            disc(body, (-2.421, 0.33, s * zc), 0.045, (-1, 0, 0), BLACK, TRIM, seg=10)
    for k in range(5):
        zz = -0.32 + k * 0.16
        prism_xy(body, [(-2.0, 0.24), (-2.4, 0.3), (-2.4, 0.42), (-2.0, 0.27)], zz - 0.01, zz + 0.01, BLACK, CARBON)
    box(body, -2.4, 0.26, -0.4, -2.0, 0.28, 0.4, BLACK, CARBON)
    plate(body, -2.35, 0.55, False, w=0.3, h=0.1)
    # aleron grande
    prism_xy(body, [(-2.05, 1.22), (-2.45, 1.2), (-2.47, 1.25), (-2.07, 1.29)], -0.88, 0.88, acc, PAINT_MATE)
    both(lambda s: prism_xy(body, [(-2.0, 1.16), (-2.5, 1.14), (-2.5, 1.33), (-2.05, 1.33)], s * 0.88, s * 0.9, BLACK,
                            CARBON))
    for s in (-1, 1):
        bar(body, (-2.15, L.y_top(-2.15, s * 0.5) - 0.02, s * 0.5), (-2.2, 1.22, s * 0.5), 0.04, BLACK, CARBON, t2=0.12)
    mirrors(body, 0.45, 0.93, 0.85, acc)
    wipers(body, 0.6, 0.93, (0.55, -0.1), L=0.45)
    driver(body, p["driver"], -0.45, 0.24, -0.33, 0.78)
    info["wheels"] = [(1.4, wz), (-1.4, wz)]
    info["exhaust"] = [(-2.45, 0.33, s * 0.56) for s in (-1, 1)]
    info["head"] = [(2.3, 0.64, s * 0.58) for s in (-1, 1)]
    return r, 0.3, make_wheel("mesh", r, 0.3, rim=rgb(60, 60, 66))


def _gt3(p, body, glass, info):
    """GT3 aleman (tipo M4): rinones verticales enormes, salpicaderas anchas, aleron de cuello de cisne."""
    col, acc = C(p["body"]), C(p["accent"])
    lb, red = rgb(90, 170, 230), rgb(215, 30, 40)
    r, wz = 0.37, 0.82
    e = 0.25
    L = Loft([(-2.45, 0.3, 0.98, 0.8, 0.74), (-2.38, 0.2, 1.06, 0.94, 0.82), (-1.4, 0.18, 1.02, 1.02, 0.82),
              (-0.6, 0.18, 0.97, 0.93, 0.8), (0.5, 0.18, 0.95, 0.94, 0.8), (1.45, 0.18, 0.88, 1.02, 0.82),
              (2.2, 0.17, 0.78, 0.96, 0.8), (2.46, 0.2, 0.62, 0.84, 0.66)], e=e, sub=3)
    L.mesh(body, col, PAINT, n=24)
    gl = Loft([(-1.55, 1.0, 1.04, 0.78, 0.77), (-0.85, 0.99, 1.38, 0.77, 0.6), (0.0, 0.97, 1.4, 0.78, 0.6),
               (0.62, 0.93, 0.95, 0.8, 0.79)], e=0.28, sub=4)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-0.88, 1.33, 1.4, 0.62, 0.56), (-0.7, 1.35, 1.43, 0.64, 0.58), (-0.05, 1.36, 1.44, 0.64, 0.58),
          (0.1, 1.34, 1.41, 0.63, 0.57)], e=0.3, sub=2).mesh(body, BLACK, CARBON, n=16)     # techo de carbono

    def pillars(s):
        bar(body, (0.62, 0.95, s * 0.79), (0.02, 1.38, s * 0.62), 0.05, col, PAINT_MATE)
        bar(body, (-0.85, 1.38, s * 0.6), (-1.5, 1.02, s * 0.77), 0.13, col, PAINT_MATE)
        bar(body, (-0.42, 0.98, s * 0.79), (-0.42, 1.4, s * 0.62), 0.07, BLACK, TRIM)
    both(pillars)
    wells(body, L, (1.45, -1.4), r, 0.43, 0.18, lip=BLACK, lip_w=0.05, lip_out=0.035)
    # franjas tricolor (cofre, cajuela y costados)
    for k, cc in enumerate((lb, acc, red)):
        f0, f1 = 0.06 + k * 0.1, 0.14 + k * 0.1
        L.patch(body, 0.55, 2.3, ph_top(f1, e), ph_top(f0, e), cc, PAINT_MATE, nx=10, nph=2)
        L.patch(body, -2.36, -1.45, ph_top(f1, e), ph_top(f0, e), cc, PAINT_MATE, nx=6, nph=2)
        yf0, yf1 = -0.35 + k * 0.12, -0.27 + k * 0.12
        L.patch(body, -1.0, 1.0, ph_side(yf0, e), ph_side(yf1, e), cc, PAINT_MATE, nx=10, nph=2)
        L.patch(body, -1.0, 1.0, math.pi - ph_side(yf1, e), math.pi - ph_side(yf0, e), cc, PAINT_MATE, nx=10, nph=2)
    # rinones verticales, faros, canards, rejillas
    for s in (-1, 1):
        superellipsoid(body, (2.43, 0.45, s * 0.13), (0.04, 0.18, 0.115), 0.35, CHROME_C, CHROME, 12, 8)
        superellipsoid(body, (2.45, 0.45, s * 0.13), (0.03, 0.16, 0.095), 0.35, BLACK, TRIM, 12, 8)
        for k in range(5):
            box(body, 2.455, 0.33 + k * 0.055, s * 0.13 - 0.085, 2.475, 0.345 + k * 0.055, s * 0.13 + 0.085,
                rgb(50, 50, 55), ALU)
        lamp(body, (2.33, 0.68, s * 0.58), (0.12, 0.045, 0.2), HEAD_C, HEAD_M, 0.25)
        lamp(body, (2.36, 0.3, s * 0.62), (0.08, 0.07, 0.18), BLACK, TRIM, 0.3)
        prism_xz(body, [(2.2, s * 0.85), (2.48, s * 0.72), (2.45, s * 0.98)], 0.38, 0.4, BLACK, CARBON)
        for k in range(4):
            xx = 1.25 + k * 0.1
            box(body, xx, L.y_top(xx, s * 0.75) - 0.01, s * 0.66 - 0.12, xx + 0.05, L.y_top(xx, s * 0.75) + 0.01,
                s * 0.66 + 0.12, BLACK, TRIM)
    prism_xz(body, [(2.15, -0.98), (2.62, -0.85), (2.62, 0.85), (2.15, 0.98)], 0.1, 0.14, BLACK, CARBON)
    # cola: calaveras en L, difusor, aleron de cuello de cisne
    for s in (-1, 1):
        lamp(body, (-2.42, 0.86, s * 0.58), (0.04, 0.05, 0.2), TAIL_C, TAIL_M, 0.25)
        lamp(body, (-2.42, 0.78, s * 0.72), (0.04, 0.1, 0.05), TAIL_C, TAIL_M, 0.25)
    for k in range(5):
        zz = -0.5 + k * 0.25
        prism_xy(body, [(-2.0, 0.12), (-2.62, 0.28), (-2.62, 0.45), (-2.0, 0.16)], zz - 0.012, zz + 0.012, BLACK,
                 CARBON)
    box(body, -2.62, 0.12, -0.75, -2.0, 0.15, 0.75, BLACK, CARBON)
    prism_xy(body, [(-2.2, 1.34), (-2.68, 1.3), (-2.7, 1.36), (-2.22, 1.4)], -0.92, 0.92, BLACK, CARBON)
    prism_xy(body, [(-2.55, 1.4), (-2.72, 1.42), (-2.72, 1.45), (-2.55, 1.43)], -0.92, 0.92, BLACK, CARBON)
    both(lambda s: prism_xy(body, [(-2.15, 1.25), (-2.75, 1.22), (-2.75, 1.5), (-2.25, 1.5)], s * 0.92, s * 0.94,
                            BLACK, CARBON))
    for s in (-1, 1):
        y0 = L.y_top(-2.0, s * 0.35)
        tube(body, [(-1.95, y0, s * 0.35), (-2.15, y0 + 0.25, s * 0.35), (-2.35, 1.47, s * 0.35),
                    (-2.45, 1.4, s * 0.35)], 0.022, BLACK, CARBON, seg=6, joints=False)
    # numero en puertas y cofre, escapes laterales, espejos
    num = str(p.get("num", 96))
    zs = L.side_z(-0.2, 0.2, 0.46, 0.86) + 0.006
    for s in (-1, 1):
        roundel(body, (0.0, 0.66, s * zs), (0, 0, s), (s, 0, 0), num, 0.2, WHITE, BLACK)
        cylinder(body, (-0.75, 0.22, s * 0.88), (-0.9, 0.22, s * 1.02), 0.055, rgb(120, 110, 100), ALU, seg=10)
    roundel(body, (1.75, L.y_top(1.75) + 0.008, 0), (0, 1, 0), (0, 0, -1), num, 0.2, WHITE, BLACK)
    mirrors(body, 0.45, 0.95, 0.87, col, size=0.9)
    driver(body, p["driver"], -0.45, 0.17, -0.33, 0.78)
    info["wheels"] = [(1.45, wz), (-1.4, wz)]
    info["exhaust"] = [(-0.92, 0.22, s * 1.04) for s in (-1, 1)]
    info["head"] = [(2.45, 0.68, s * 0.58) for s in (-1, 1)]
    return r, 0.34, make_wheel("mesh", r, 0.34, rim=rgb(30, 30, 34), accent=red)


def _baja(p, body, glass, info):
    """Trophy truck de la Baja 1000: suspension de largo recorrido, jaula, barra de luces, refacciones."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.52, 0.92
    fx, rx = 1.75, -1.7
    white = rgb(230, 230, 235)
    cab = Loft([(-0.78, 0.86, 1.45, 0.82, 0.78), (0.4, 0.86, 1.42, 0.84, 0.79), (1.25, 0.88, 1.32, 0.82, 0.75),
                (2.3, 0.86, 1.18, 0.76, 0.66), (2.62, 0.84, 1.02, 0.6, 0.48)], e=0.3, sub=3)
    cab.mesh(body, acc, PAINT, n=24, split=1.06, col2=col)
    gl = Loft([(-0.74, 1.4, 1.42, 0.78, 0.77), (-0.6, 1.4, 1.93, 0.78, 0.68), (0.05, 1.39, 1.96, 0.79, 0.69),
               (0.62, 1.37, 1.41, 0.8, 0.79)], e=0.25, sub=4)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-0.72, 1.89, 1.98, 0.7, 0.66), (0.08, 1.92, 2.0, 0.71, 0.67)], e=0.3, smooth=False).mesh(body, col, PAINT,
                                                                                                    16)

    def pillars(s):
        bar(body, (0.6, 1.4, s * 0.79), (0.05, 1.95, s * 0.69), 0.07, col, PAINT_MATE)
        bar(body, (-0.72, 1.42, s * 0.78), (-0.68, 1.95, s * 0.69), 0.1, col, PAINT_MATE)
    both(pillars)
    # caja corta con jaula, refacciones y tanque
    both(lambda s: box(body, -2.5, 0.92, s * 0.76 - 0.05, -0.8, 1.32, s * 0.76 + 0.05, col, PAINT_MATE))
    box(body, -2.5, 0.92, -0.81, -2.4, 1.3, 0.81, acc, PAINT_MATE)
    box(body, -2.5, 0.86, -0.78, -0.8, 0.92, 0.78, BLACK, TRIM)
    decal_hd(body, -1.6, 0.815, 1.3, 0.98, 1.3, 15)
    spare_tire(body, (-1.6, 1.15, 0.0), r * 0.95, 0.36)
    spare_tire(body, (-1.6, 1.52, 0.0), r * 0.95, 0.36)
    box(body, -2.35, 0.95, -0.7, -2.0, 1.3, -0.35, rgb(40, 40, 44), ALU)            # celda de combustible
    for s in (-1, 1):
        tube(body, [(-0.75, 1.95, s * 0.6), (-1.3, 1.9, s * 0.62), (-2.3, 1.35, s * 0.7)], 0.035, BLACK, TRIM, seg=6)
        tube(body, [(-1.3, 1.9, s * 0.62), (-1.3, 1.3, s * 0.74)], 0.03, BLACK, TRIM, seg=6, joints=False)
    tube(body, [(-1.3, 1.9, -0.62), (-1.3, 1.9, 0.62)], 0.03, BLACK, TRIM, seg=6, joints=False)
    # salpicaderas abiertas de fibra
    fender(body, fx, r, -0.15, math.pi + 0.05, r + 0.08, 0.2, 0.62, 1.15, col, n=14, taper=0.3)
    fender(body, rx, r, -0.05, math.pi + 0.05, r + 0.08, 0.14, 0.62, 1.12, acc, PAINT, n=14, taper=0.3)
    # suspension delantera: brazos A gigantes y dos amortiguadores por rueda
    for s in (-1, 1):
        for yy, yh in ((0.45, 0.42), (0.78, 0.7)):
            bar(body, (fx + 0.35, yy, s * 0.3), (fx, yh, s * 0.78), 0.05, white, PAINT_MATE, t2=0.04)
            bar(body, (fx - 0.35, yy, s * 0.3), (fx, yh, s * 0.78), 0.05, white, PAINT_MATE, t2=0.04)
        for dx in (-0.12, 0.12):
            cylinder(body, (fx + dx, 0.5, s * 0.66), (fx + dx * 0.5, 1.25, s * 0.5), 0.05, rgb(200, 30, 30), PAINT_MATE,
                     seg=8)
            cylinder(body, (fx + dx * 0.75, 0.85, s * 0.58), (fx + dx * 0.5, 1.25, s * 0.5), 0.07, rgb(250, 200, 30),
                     PAINT_MATE, seg=8)
        bar(body, (fx, r, s * 0.68), (fx, r, s * 0.82), 0.12, GREY, ALU)
        # traseras: brazos de arrastre y amortiguadores al eje rigido
        bar(body, (rx + 1.5, 0.75, s * 0.4), (rx, r - 0.05, s * 0.55), 0.07, white, PAINT_MATE)
        bar(body, (rx + 1.2, 0.95, s * 0.3), (rx, r + 0.12, s * 0.5), 0.05, white, PAINT_MATE)
        for dx in (-0.15, 0.1):
            cylinder(body, (rx + dx, r + 0.05, s * 0.55), (rx + dx * 0.5 + 0.15, 1.5, s * 0.55), 0.05, rgb(200, 30, 30),
                     PAINT_MATE, seg=8)
            cylinder(body, (rx + dx * 0.6 + 0.07, 1.05, s * 0.55), (rx + dx * 0.5 + 0.15, 1.5, s * 0.55), 0.07,
                     rgb(250, 200, 30), PAINT_MATE, seg=8)
    cylinder(body, (rx, r, -0.82), (rx, r, 0.82), 0.09, GREY, ALU, seg=10)            # eje trasero
    superellipsoid(body, (rx, r, 0.1), (0.18, 0.2, 0.18), 0.6, GREY, ALU, 12, 8)      # diferencial
    tube(body, [(-2.5, 0.8, -0.4), (2.4, 0.75, -0.4)], 0.04, BLACK, TRIM, seg=6, joints=False)
    tube(body, [(-2.5, 0.8, 0.4), (2.4, 0.75, 0.4)], 0.04, BLACK, TRIM, seg=6, joints=False)
    # defensa tubular, faros auxiliares, barra de luces
    tube(body, [(2.55, 0.62, -0.62), (2.78, 0.68, -0.4), (2.82, 0.7, 0.4), (2.55, 0.62, 0.62)], 0.04, BLACK, TRIM,
         seg=6)
    box(body, 2.66, 0.6, -0.5, 2.74, 0.74, 0.5, BLACK, TRIM)
    both(lambda s: lamp(body, (2.6, 0.98, s * 0.48), (0.03, 0.05, 0.12), HEAD_C, HEAD_M, 0.3))
    grille_slats(body, 2.63, 0.86, 1.02, -0.3, 0.3, 3, slat=rgb(40, 40, 44), sm=TRIM)
    for zc in (-0.3, 0.3):
        cylinder(body, (2.3, 1.22, zc), (2.42, 1.24, zc), 0.1, BLACK, TRIM, seg=12)
        disc(body, (2.425, 1.24, zc), 0.085, (1, 0, 0), HEAD_C, HEAD_M, seg=12)
    box(body, 0.0, 2.0, -0.72, 0.16, 2.12, 0.72, BLACK, TRIM)
    for k in range(8):
        zc = -0.63 + k * 0.18
        box(body, 0.16, 2.02, zc - 0.07, 0.175, 2.1, zc + 0.07, HEAD_C, HEAD_M)
    both(lambda s: lamp(body, (-2.51, 1.2, s * 0.7), (0.02, 0.06, 0.05), TAIL_C, TAIL_M, 0.3))
    lamp(body, (-2.51, 1.28, 0), (0.02, 0.03, 0.2), (1.0, 0.6, 0.1), AMBER_M, 0.3)
    # numero y espejos
    zs = cab.side_z(-0.27, 0.07, 1.03, 1.37) + 0.006
    for s in (-1, 1):
        roundel(body, (-0.1, 1.2, s * zs), (0, 0, s), (s, 0, 0), str(p.get("num", 13)), 0.17, WHITE, BLACK)
    mirrors(body, 0.45, 1.42, 0.86, col)
    driver(body, p["driver"], -0.25, 0.9, -0.33, 0.8)
    info["wheels"] = [(fx, wz), (rx, wz)]
    info["exhaust"] = [(-2.6, 0.95, 0.45)]
    cylinder(body, (-2.2, 0.95, 0.45), (-2.6, 0.95, 0.45), 0.06, rgb(120, 110, 100), ALU, seg=10)
    info["head"] = [(2.7, 1.0, s * 0.48) for s in (-1, 1)]
    return r, 0.4, make_wheel("beadlock", r, 0.4, rim=rgb(30, 30, 34), accent=acc, tread=True)


# ================================================================ modelos originales, rehechos con mas detalle
def _sport(p, body, glass, info):
    """Deportivo GTC (motor central): cuna baja, cabina adelantada, tomas laterales, aleron, LED."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.37, 0.8
    L = Loft([(-2.27, 0.32, 0.78, 0.8, 0.72), (-2.2, 0.26, 0.9, 0.92, 0.8), (-1.5, 0.24, 0.92, 0.98, 0.82),
              (-0.6, 0.24, 0.85, 0.92, 0.78), (0.4, 0.24, 0.75, 0.94, 0.8), (1.45, 0.24, 0.66, 0.98, 0.82),
              (2.1, 0.24, 0.52, 0.92, 0.72), (2.3, 0.27, 0.38, 0.75, 0.5)], e=0.28, sub=3)
    L.mesh(body, col, PAINT, n=24)
    gl = Loft([(-1.35, 0.86, 0.9, 0.72, 0.7), (-0.8, 0.86, 1.15, 0.74, 0.55), (-0.15, 0.8, 1.17, 0.75, 0.56),
               (0.55, 0.72, 0.75, 0.78, 0.76)], e=0.32, sub=4)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-0.75, 1.1, 1.17, 0.56, 0.5), (-0.6, 1.12, 1.19, 0.57, 0.51), (-0.25, 1.13, 1.2, 0.57, 0.51),
          (-0.1, 1.11, 1.17, 0.56, 0.5)], e=0.3, sub=2).mesh(body, col, PAINT, n=16)

    def pillars(s):
        bar(body, (0.52, 0.76, s * 0.77), (-0.15, 1.16, s * 0.58), 0.05, BLACK, TRIM)
        prism_xy(body, [(-0.7, 1.15), (-1.0, 1.05), (-1.4, 0.88), (-0.75, 0.85)], s * 0.7, s * 0.74, col,
                 PAINT_MATE)
    both(pillars)
    wells(body, L, (1.45, -1.45), r, 0.42, 0.24)
    # tomas laterales, rejillas del motor, faros LED, splitter, faldones
    for s in (-1, 1):
        superellipsoid(body, (-0.85, 0.55, s * L.side_z(-1.0, -0.7, 0.45, 0.65)), (0.3, 0.13, 0.06), 0.4, BLACK,
                       TRIM, 12, 6)
        lamp(body, (2.12, 0.5, s * 0.6), (0.14, 0.03, 0.2), HEAD_C, HEAD_M, 0.25)
        lamp(body, (2.2, 0.44, s * 0.68), (0.06, 0.012, 0.12), HEAD_C, LED_M, 0.3)
        lamp(body, (2.24, 0.33, s * 0.55), (0.04, 0.06, 0.17), BLACK, TRIM, 0.3)
        box(body, -1.0, 0.2, s * 0.86, 1.0, 0.26, s * 0.96, BLACK, CARBON)
    for k in range(6):
        xx = -1.45 - k * 0.1
        y = L.y_top(xx) + 0.004
        box(body, xx - 0.03, y - 0.01, -0.45, xx + 0.03, y + 0.008, 0.45, BLACK, TRIM)
    prism_xz(body, [(2.0, -0.9), (2.42, -0.7), (2.42, 0.7), (2.0, 0.9)], 0.2, 0.23, BLACK, CARBON)
    # cola: barra LED, difusor, escapes, aleron
    lamp(body, (-2.27, 0.66, 0), (0.025, 0.035, 0.7), TAIL_C, TAIL_M, 0.25)
    both(lambda s: lamp(body, (-2.25, 0.62, s * 0.66), (0.03, 0.08, 0.09), TAIL_C, TAIL_M, 0.3))
    for k in range(5):
        zz = -0.4 + k * 0.2
        prism_xy(body, [(-1.9, 0.22), (-2.42, 0.32), (-2.42, 0.42), (-1.9, 0.25)], zz - 0.012, zz + 0.012, BLACK,
                 CARBON)
    box(body, -2.42, 0.22, -0.6, -1.9, 0.25, 0.6, BLACK, CARBON)
    prism_xy(body, [(-1.88, 1.12), (-2.32, 1.1), (-2.33, 1.15), (-1.9, 1.18)], -0.92, 0.92, acc, PAINT_MATE)
    for s in (-1, 1):
        prism_xy(body, [(-1.85, 1.05), (-2.36, 1.03), (-2.36, 1.22), (-1.92, 1.22)], s * 0.92, s * 0.94, BLACK,
                 CARBON)
        bar(body, (-2.0, L.y_top(-2.0, s * 0.55) - 0.02, s * 0.55), (-2.1, 1.12, s * 0.55), 0.04, BLACK, CARBON,
            t2=0.1)
    info["exhaust"] = [(-2.32, 0.42, -0.42), (-2.32, 0.42, 0.42)]
    for e_ in info["exhaust"]:
        cylinder(body, (e_[0] + 0.25, e_[1], e_[2]), (e_[0], e_[1], e_[2]), 0.07, CHROME_C, CHROME, seg=10)
        disc(body, (e_[0] - 0.001, e_[1], e_[2]), 0.055, (-1, 0, 0), BLACK, TRIM, seg=10)
    zs = L.side_z(-0.5, 0.5, 0.36, 0.6) + 0.006
    decal_hd(body, 0.05, zs, 1.0, 0.36, 0.61, 0 if p["sign"] == 9 else p["sign"])     # el 9 cambia en cada mapa
    plate(body, -2.28, 0.47, False, w=0.3, h=0.1)
    mirrors(body, 0.4, 0.76, 0.86, col)
    driver(body, p["driver"], -0.55, 0.16, -0.3, 0.78)
    info["wheels"] = [(1.45, wz), (-1.45, wz)]
    info["head"] = [(2.25, 0.5, s * 0.6) for s in (-1, 1)]
    return r, 0.3, make_wheel("spoke5", r, 0.3, rim=rgb(25, 25, 28), accent=acc)


def _hatch(p, body, glass, info):
    """Hatchback de reparto: 5 puertas, caja de entregas en el techo con letrero."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.35, 0.74
    L = Loft([(-1.95, 0.38, 0.98, 0.78, 0.72), (-1.9, 0.29, 1.05, 0.86, 0.8), (-1.3, 0.27, 1.03, 0.88, 0.82),
              (0.55, 0.27, 0.95, 0.88, 0.82), (1.4, 0.29, 0.85, 0.86, 0.78), (1.88, 0.33, 0.72, 0.8, 0.68),
              (1.96, 0.41, 0.6, 0.7, 0.55)], e=0.32, sub=3)
    L.mesh(body, col, PAINT, n=24)
    gl = Loft([(-1.88, 1.0, 1.03, 0.8, 0.79), (-1.68, 1.0, 1.43, 0.8, 0.66), (-0.6, 0.98, 1.5, 0.82, 0.68),
               (0.05, 0.96, 1.48, 0.83, 0.69), (0.62, 0.93, 0.95, 0.85, 0.84)], e=0.32, sub=3)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-1.68, 1.39, 1.46, 0.67, 0.62), (-1.5, 1.42, 1.5, 0.69, 0.63), (-0.05, 1.45, 1.53, 0.7, 0.64),
          (0.12, 1.43, 1.5, 0.69, 0.63)], e=0.32, sub=2).mesh(body, col, PAINT, n=16)

    def pillars(s):
        bar(body, (0.6, 0.94, s * 0.84), (0.08, 1.47, s * 0.68), 0.06, col, PAINT_MATE)
        bar(body, (-0.5, 0.98, s * 0.85), (-0.5, 1.5, s * 0.68), 0.08, BLACK, TRIM)
        prism_xy(body, [(-1.25, 1.48), (-1.62, 1.44), (-1.86, 1.0), (-1.35, 0.99)], s * 0.77, s * 0.81, col,
                 PAINT_MATE)
        bar(body, (-1.7, 0.985, s * 0.87), (0.6, 0.955, s * 0.875), 0.02, BLACK, TRIM)
    both(pillars)
    wells(body, L, (1.25, -1.25), r, 0.41, 0.27)
    seam(body, L, [(0.58, 0.32), (0.56, 0.94)])
    seam(body, L, [(-0.5, 0.3), (-0.5, 0.97)])
    seam(body, L, [(-1.3, 0.4), (-1.35, 0.99)])
    for xh in (0.3, -0.75):
        zh = L.z_at(xh, 0.82)
        both(lambda s: box(body, xh - 0.08, 0.8, s * (zh - 0.004), xh + 0.04, 0.83, s * (zh + 0.012), DARK, TRIM))
    top_seam(body, L, [(0.65, -0.72), (1.84, -0.66), (1.84, 0.66), (0.65, 0.72)])
    superellipsoid(body, (1.9, 0.42, 0), (0.08, 0.12, 0.86), 0.35, rgb(40, 40, 44), PAINT_MATE, 16, 6)
    superellipsoid(body, (-1.92, 0.44, 0), (0.07, 0.12, 0.87), 0.35, rgb(40, 40, 44), PAINT_MATE, 16, 6)
    grille_slats(body, 1.94, 0.56, 0.68, -0.32, 0.32, 3, slat=rgb(60, 60, 66), sm=ALU)
    both(lambda s: lamp(body, (1.86, 0.72, s * 0.56), (0.07, 0.07, 0.2), HEAD_C, HEAD_M, 0.4))
    both(lambda s: lamp(body, (1.9, 0.38, s * 0.66), (0.03, 0.03, 0.06), HEAD_C, LED_M, 0.4))
    both(lambda s: lamp(body, (-1.9, 0.86, s * 0.66), (0.04, 0.12, 0.12), TAIL_C, TAIL_M, 0.35))
    plate(body, 1.99, 0.44, True)
    plate(body, -1.95, 0.68, False)
    mirrors(body, 0.48, 0.95, 0.87, col)
    wipers(body, 0.62, 0.96, (0.6, 0.0), L=0.5)
    bar(body, (-1.88, 1.35, -0.25), (-1.92, 1.1, 0.0), 0.02, BLACK, TRIM)                 # limpiador trasero
    # caja de reparto con letrero
    for xx in (-1.1, -0.1):
        both(lambda s: bar(body, (xx, 1.48, s * 0.6), (xx, 1.58, s * 0.6), 0.05, BLACK, TRIM))
    superellipsoid(body, (-0.6, 1.72, 0), (0.6, 0.16, 0.47), 0.2, acc, PAINT_MATE, 16, 8)
    u0, u1, v0, v1 = hd.bb_uv(p["sign"])
    for s in (-1, 1):
        z_ = s * 0.475
        a, b = (0.0, -1.2) if s < 0 else (-1.2, 0.0)
        quad(body, (a, 1.6, z_), (b, 1.6, z_), (b, 1.84, z_), (a, 1.84, z_), (1, 1, 1), SIGN_M,
             ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))
    zs = L.side_z(-1.1, 0.3, 0.55, 0.85) + 0.006
    decal_hd(body, -0.4, zs, 1.3, 0.55, 0.875, p["sign"])
    driver(body, p["driver"], -0.3, 0.33, -0.36, 0.8)
    info["wheels"] = [(1.25, wz), (-1.25, wz)]
    info["exhaust"] = [(-2.0, 0.35, -0.5)]
    cylinder(body, (-1.8, 0.33, -0.5), (-2.0, 0.33, -0.5), 0.045, CHROME_C, CHROME, seg=8)
    info["head"] = [(1.95, 0.72, s * 0.56) for s in (-1, 1)]
    return r, 0.24, make_wheel("spoke6", r, 0.24, rim=rgb(200, 200, 205))


def _pickup(p, body, glass, info):
    """Camioneta del mandado: pickup compacta con caja llena de compras."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.42, 0.82
    L = Loft([(-2.45, 0.46, 1.12, 0.9, 0.88), (-2.4, 0.4, 1.14, 0.96, 0.93), (0.6, 0.4, 1.14, 0.97, 0.93),
              (1.3, 0.42, 1.12, 0.96, 0.9), (2.25, 0.44, 1.02, 0.94, 0.86), (2.45, 0.5, 0.88, 0.85, 0.75)],
             e=0.22, sub=3)
    L.mesh(body, col, PAINT, n=24)
    seam(body, L, [(-2.4, 0.62), (2.4, 0.62)], w=0.05, col=acc)                       # franja
    gl = Loft([(-0.5, 1.1, 1.14, 0.9, 0.9), (-0.46, 1.1, 1.82, 0.9, 0.78), (0.2, 1.1, 1.86, 0.91, 0.79),
               (1.05, 1.1, 1.14, 0.93, 0.92)], e=0.25, sub=3)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(-0.48, 1.8, 1.88, 0.8, 0.75), (0.25, 1.83, 1.92, 0.81, 0.76)], e=0.3, smooth=False).mesh(body, col,
                                                                                                   PAINT, 16)

    def pillars(s):
        bar(body, (1.02, 1.13, s * 0.92), (0.25, 1.86, s * 0.79), 0.06, col, PAINT_MATE)
        bar(body, (0.3, 1.13, s * 0.92), (0.3, 1.86, s * 0.79), 0.07, BLACK, TRIM)
        bar(body, (-0.47, 1.13, s * 0.91), (-0.47, 1.86, s * 0.79), 0.1, col, PAINT_MATE)
    both(pillars)
    box(body, -0.52, 1.12, -0.88, -0.42, 1.84, 0.88, col, PAINT_MATE)                # pared trasera de cabina
    # caja: paredes, compuerta y el mandado
    both(lambda s: box(body, -2.42, 1.12, s * 0.93 - 0.05, -0.55, 1.45, s * 0.93 + 0.05, col, PAINT_MATE))
    box(body, -2.47, 1.12, -0.95, -2.39, 1.45, 0.95, col, PAINT_MATE)
    both(lambda s: box(body, -2.43, 1.45, s * 0.93 - 0.06, -0.54, 1.48, s * 0.93 + 0.06, BLACK, TRIM))
    box(body, -2.48, 1.45, -0.95, -2.38, 1.48, 0.95, BLACK, TRIM)
    box(body, -2.2, 1.14, -0.8, -1.6, 1.64, -0.15, rgb(250, 210, 30), mat(M_DEF, 0.8))
    box(body, -1.5, 1.14, -0.75, -0.9, 1.56, -0.1, rgb(245, 245, 245), mat(M_DEF, 0.8))
    box(body, -2.2, 1.14, 0.1, -1.55, 1.48, 0.8, rgb(200, 120, 50), mat(M_DEF, 0.85))
    sphere(body, (-1.05, 1.44, 0.45), 0.3, rgb(40, 130, 50), mat(M_DEF, 0.4), 14, 10)
    cylinder(body, (-0.8, 1.14, -0.55), (-0.8, 1.54, -0.55), 0.13, rgb(120, 170, 220), mat(M_DEF, 0.15),
             seg=10)                                                    # garrafon de agua
    for k in range(5):
        a = k / 5 * TAU
        sphere(body, (-1.3 + 0.1 * math.cos(a), 1.2, 0.5 + 0.1 * math.sin(a)), 0.07, rgb(240, 150, 30),
               mat(M_DEF, 0.6), 8, 5)
    wells(body, L, (1.6, -1.6), r, 0.5, 0.4)
    seam(body, L, [(1.05, 0.45), (1.02, 1.12)])
    seam(body, L, [(0.3, 0.45), (0.3, 1.12)])
    seam(body, L, [(-0.5, 0.45), (-0.5, 1.12)])
    # frente: parrilla cromada, faros, defensa, estribos
    grille_slats(body, 2.47, 0.62, 0.84, -0.48, 0.48, 4)
    bar(body, (2.47, 0.85, -0.5), (2.47, 0.85, 0.5), 0.03, CHROME_C, CHROME)
    both(lambda s: lamp(body, (2.4, 0.86, s * 0.68), (0.06, 0.08, 0.17), HEAD_C, HEAD_M, 0.35))
    both(lambda s: lamp(body, (2.38, 0.86, s * 0.86), (0.05, 0.06, 0.03), AMBER_C, AMBER_M, 0.35))
    superellipsoid(body, (2.47, 0.5, 0), (0.07, 0.1, 0.96), 0.3, CHROME_C, CHROME, 16, 6)
    superellipsoid(body, (-2.47, 0.52, 0), (0.06, 0.1, 0.96), 0.3, CHROME_C, CHROME, 16, 6)
    both(lambda s: lamp(body, (-2.47, 1.0, s * 0.82), (0.03, 0.12, 0.08), TAIL_C, TAIL_M, 0.35))
    both(lambda s: box(body, -0.5, 0.36, s * 0.92, 1.05, 0.42, s * 1.05, rgb(40, 40, 44), ALU))
    plate(body, 2.52, 0.6, True)
    plate(body, -2.5, 0.75, False)
    top_seam(body, L, [(1.1, -0.8), (2.38, -0.76), (2.38, 0.76), (1.1, 0.8)])
    mirrors(body, 0.95, 1.16, 0.95, BLACK, TRIM, size=1.1)
    wipers(body, 1.04, 1.15, (0.6, 0.0), L=0.55)
    zs = L.side_z(-0.4, 1.0, 0.7, 1.05) + 0.006
    decal_hd(body, 0.35, zs, 1.25, 0.7, 1.02, p["sign"])
    driver(body, p["driver"], 0.2, 0.68, -0.38, 0.85)
    info["wheels"] = [(1.6, wz), (-1.6, wz)]
    info["exhaust"] = [(-2.5, 0.45, 0.6)]
    cylinder(body, (-2.2, 0.42, 0.6), (-2.5, 0.42, 0.6), 0.05, CHROME_C, CHROME, seg=8)
    info["head"] = [(2.45, 0.86, s * 0.68) for s in (-1, 1)]
    return r, 0.3, make_wheel("spoke6", r, 0.3, rim=rgb(175, 175, 180))


def _van(p, body, glass, info):
    """Van de farmacia de techo alto con torreta y letreros."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.42, 0.84
    L = Loft([(-2.42, 0.48, 2.3, 0.93, 0.88), (-2.38, 0.42, 2.4, 1.0, 0.93), (0.85, 0.42, 2.42, 1.0, 0.93),
              (0.95, 0.42, 1.42, 1.0, 0.95), (1.85, 0.44, 1.32, 0.98, 0.92), (2.3, 0.46, 1.16, 0.95, 0.86),
              (2.42, 0.52, 0.98, 0.86, 0.76)], e=0.18, smooth=False)
    L.mesh(body, col, PAINT, n=24)
    seam(body, L, [(-2.4, 0.9), (2.4, 0.9)], w=0.16, col=acc)                       # franja
    gl = Loft([(0.84, 1.36, 2.36, 0.97, 0.9), (1.25, 1.36, 2.28, 0.97, 0.9), (1.95, 1.3, 1.36, 0.97, 0.94)],
              e=0.2, smooth=False)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(0.84, 2.26, 2.42, 0.94, 0.9), (1.32, 2.22, 2.38, 0.94, 0.9)], e=0.2, smooth=False).mesh(body, col,
                                                                                                PAINT, 16)

    def pillars(s):
        bar(body, (1.92, 1.34, s * 0.94), (1.28, 2.32, s * 0.9), 0.08, col, PAINT_MATE)
        bar(body, (0.88, 1.38, s * 0.96), (0.88, 2.36, s * 0.92), 0.1, col, PAINT_MATE)
    both(pillars)
    wells(body, L, (1.55, -1.55), r, 0.5, 0.42)
    seam(body, L, [(0.85, 0.5), (0.85, 2.35)])
    seam(body, L, [(-0.35, 0.5), (-0.35, 2.3)], sides=(1,))
    seam(body, L, [(0.75, 2.05), (-0.35, 2.05)], sides=(1,), w=0.03, col=GREY)        # riel de la puerta corrediza
    # puertas traseras con ventanitas
    box(body, -2.43, 0.5, -0.006, -2.4, 2.3, 0.006, BLACK, TRIM)
    both(lambda s: box(body, -2.44, 1.55, s * 0.12, -2.41, 2.1, s * 0.75, rgb(30, 40, 55), GLASS_M))
    both(lambda s: lamp(body, (-2.4, 1.0, s * 0.86), (0.04, 0.2, 0.08), TAIL_C, TAIL_M, 0.35))
    box(body, -2.42, 2.3, -0.2, -2.38, 2.36, 0.2, TAIL_C, TAIL_M)
    superellipsoid(body, (-2.45, 0.55, 0), (0.06, 0.1, 0.96), 0.3, rgb(50, 50, 55), PAINT_MATE, 16, 6)
    # frente
    superellipsoid(body, (2.44, 0.58, 0), (0.07, 0.12, 0.92), 0.3, rgb(50, 50, 55), PAINT_MATE, 16, 6)
    grille_slats(body, 2.44, 0.75, 0.95, -0.45, 0.45, 3, slat=rgb(60, 60, 66), sm=ALU)
    both(lambda s: lamp(body, (2.36, 1.0, s * 0.66), (0.08, 0.1, 0.2), HEAD_C, HEAD_M, 0.35))
    plate(body, 2.5, 0.6, True)
    plate(body, -2.46, 0.8, False)
    mirrors(body, 1.75, 1.4, 1.0, BLACK, TRIM, size=1.3)
    wipers(body, 1.9, 1.36, (0.65, 0.0), L=0.6)
    # torreta y cruz
    box(body, 0.9, 2.42, -0.6, 1.3, 2.46, 0.6, DARK, TRIM)
    box(body, 0.92, 2.46, -0.6, 1.28, 2.6, 0.0, (1.0, 0.1, 0.08), mat(M_EMIS, 0.3, 0, 4.0))
    box(body, 0.92, 2.46, 0.0, 1.28, 2.6, 0.6, (0.1, 0.3, 1.0), mat(M_EMIS, 0.3, 0, 4.0))
    zs = L.side_z(-2.2, 0.6, 1.2, 2.3) + 0.006
    decal_hd(body, -0.85, zs, 2.6, 1.35, 2.2, p["sign"])
    for s in (-1, 1):                                                    # cruz de farmacia en la puerta
        box(body, 1.25, 1.0, s * (L.side_z(1.1, 1.4, 0.95, 1.3) + 0.004) - 0.004, 1.55, 1.08,
            s * (L.side_z(1.1, 1.4, 0.95, 1.3) + 0.01), rgb(40, 170, 80), PAINT_MATE)
        box(body, 1.36, 0.89, s * (L.side_z(1.1, 1.4, 0.95, 1.3) + 0.004) - 0.004, 1.44, 1.19,
            s * (L.side_z(1.1, 1.4, 0.95, 1.3) + 0.01), rgb(40, 170, 80), PAINT_MATE)
    driver(body, p["driver"], 1.0, 0.85, -0.42, 0.9)
    info["wheels"] = [(1.55, wz), (-1.55, wz)]
    info["exhaust"] = [(-2.45, 0.42, 0.6)]
    info["head"] = [(2.45, 1.0, s * 0.66) for s in (-1, 1)]
    return r, 0.28, make_wheel("steel", r, 0.28, rim=rgb(190, 190, 195))


def _cabover(body, glass, col, acc, x0, x1):
    """Cabina de camion: parte baja pintada, vidrios arriba, techo; frente plano."""
    L = Loft([(x0, 0.82, 1.82, 1.1, 1.06), (x1 - 0.1, 0.82, 1.82, 1.12, 1.08), (x1, 0.88, 1.78, 1.04, 1.0)],
             e=0.18, smooth=False)
    L.mesh(body, col, PAINT, n=24)
    gl = Loft([(x0 + 0.02, 1.78, 2.55, 1.06, 1.0), (x1 - 0.12, 1.78, 2.55, 1.08, 1.02), (x1 - 0.02, 1.76, 2.45, 1.0,
                                                                                         0.96)],
              e=0.18, smooth=False)
    gl.mesh(glass, GLASS_C, GLASS_M, n=24)
    Loft([(x0, 2.5, 2.8, 1.1, 1.05), (x1 - 0.1, 2.5, 2.78, 1.12, 1.06), (x1, 2.45, 2.7, 1.04, 0.98)], e=0.2,
         smooth=False).mesh(body, acc, PAINT, n=20)
    for s in (-1, 1):
        bar(body, (x1 - 0.05, 1.8, s * 1.0), (x1 - 0.05, 2.5, s * 0.97), 0.1, col, PAINT_MATE)
        bar(body, (x0 + 0.05, 1.8, s * 1.08), (x0 + 0.05, 2.52, s * 1.04), 0.12, col, PAINT_MATE)
        bar(body, (x0 + 0.6, 1.8, s * 1.1), (x0 + 0.6, 2.52, s * 1.06), 0.06, BLACK, TRIM)
    # parrilla, faros, defensa, espejos de camion, escalones
    grille_slats(body, x1 + 0.005, 1.0, 1.55, -0.6, 0.6, 5, slat=rgb(180, 180, 185), sm=CHROME)
    for s in (-1, 1):
        lamp(body, (x1 - 0.02, 1.05, s * 0.82), (0.05, 0.1, 0.16), HEAD_C, HEAD_M, 0.35)
        lamp(body, (x1 - 0.03, 2.72, s * 0.7), (0.03, 0.03, 0.06), AMBER_C, AMBER_M, 0.4)
        bar(body, (x1 - 0.2, 2.2, s * 1.1), (x1 + 0.15, 2.25, s * 1.32), 0.035, BLACK, TRIM)
        box(body, x1 + 0.1, 1.8, s * 1.3 - 0.04, x1 + 0.16, 2.3, s * 1.3 + 0.04, BLACK, TRIM)
        box(body, x0 + 0.3, 0.55, s * 1.05, x0 + 0.75, 0.6, s * 1.15, GREY, ALU)
        box(body, x0 + 0.3, 0.8, s * 1.08, x0 + 0.75, 0.84, s * 1.15, GREY, ALU)
    superellipsoid(body, (x1 + 0.05, 0.7, 0), (0.08, 0.14, 1.12), 0.3, rgb(60, 60, 66), PAINT_MATE, 16, 6)
    plate(body, x1 + 0.14, 0.7, True)
    wipers(body, x1 + 0.01, 1.8, (0.75, 0.0), L=0.65)
    return L


def _truck(p, body, glass, info):
    """Camion panadero: cabina chata y caja seca grande con letrero; tres ejes."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.5, 0.92
    box(body, -3.0, 0.48, -0.55, 3.0, 0.78, 0.55, DARK, TRIM)                       # chasis
    _cabover(body, glass, col, acc, 1.55, 3.0)
    caja = Loft([(-3.0, 1.0, 3.3, 1.17, 1.17), (1.42, 1.0, 3.3, 1.17, 1.17)], e=0.1, smooth=False)
    caja.mesh(body, col, PAINT, n=24)
    box(body, -3.0, 0.9, -1.17, 1.42, 1.02, 1.17, acc, PAINT_MATE)
    both(lambda s: box(body, -3.0, 3.18, s * 1.175 - 0.01, 1.42, 3.26, s * 1.175 + 0.01, acc, PAINT_MATE))
    decal_hd(body, -0.8, 1.185, 4.0, 1.35, 3.05, p["sign"])
    # puertas traseras
    box(body, -3.02, 1.05, -0.008, -2.99, 3.25, 0.008, GREY, ALU)
    for s in (-1, 1):
        for yy in (1.5, 2.6):
            box(body, -3.03, yy, s * 0.5, -3.0, yy + 0.06, s * 0.9, GREY, ALU)
        bar(body, (-3.03, 1.2, s * 0.15), (-3.03, 3.1, s * 0.15), 0.03, CHROME_C, CHROME)
        lamp(body, (-3.02, 0.85, s * 0.95), (0.03, 0.08, 0.12), TAIL_C, TAIL_M, 0.35)
        box(body, -2.95, 0.4, s * 0.98, -2.9, 0.95, s * 1.15, BLACK, RUBBER)          # loderas
        box(body, -0.7, 0.55, s * 0.95, 0.9, 0.65, s * 1.12, GREY, ALU)               # defensas laterales
        cylinder(body, (0.3, 0.75, s * 0.85), (1.2, 0.75, s * 0.85), 0.22, CHROME_C, CHROME, seg=12)  # tanque
    box(body, -3.05, 0.55, -1.1, -2.98, 0.68, 1.1, rgb(240, 200, 30), PAINT_MATE)     # defensa trasera
    cylinder(body, (1.5, 1.0, 0.98), (1.5, 3.2, 0.98), 0.08, CHROME_C, CHROME, seg=10)
    driver(body, p["driver"], 2.35, 1.3, -0.45, 1.0)
    info["wheels"] = [(2.2, wz), (-1.4, wz), (-2.3, wz)]
    info["exhaust"] = [(1.5, 3.2, 0.98)]
    info["head"] = [(3.05, 1.05, s * 0.82) for s in (-1, 1)]
    return r, 0.36, make_wheel("steel", r, 0.36, rim=rgb(200, 200, 205))


def _gas(p, body, glass, info):
    """Pipa de gas: cabina chata, tanque con franjas, escalera, valvulas y carrete de manguera."""
    col, acc = C(p["body"]), C(p["accent"])
    r, wz = 0.5, 0.92
    box(body, -3.0, 0.48, -0.55, 3.0, 0.78, 0.55, DARK, TRIM)
    _cabover(body, glass, col, acc, 1.55, 3.0)
    prof = [(0.0, -2.2), (0.6, -2.2), (0.92, -2.05), (0.98, -1.8), (0.98, 1.8), (0.92, 2.05), (0.6, 2.2), (0.0, 2.2)]
    revolve(body, prof, (-0.85, 1.9, 0), (1, 0, 0), col, mat(M_PAINT, 0.18, 0.4), seg=28)
    for x0 in (-2.3, -0.85, 0.6):
        revolve(body, [(0.97, -0.1), (1.0, 0.0), (0.97, 0.1)], (x0, 1.9, 0), (1, 0, 0), acc, PAINT_MATE, seg=28)
    for x0 in (-2.6, 0.9):
        box(body, x0, 0.85, -0.6, x0 + 0.35, 1.05, 0.6, rgb(60, 60, 66), mat(M_METAL, 0.4, 0.8))
    decal_hd(body, -1.55, 1.0, 1.4, 1.55, 2.2, p["sign"])
    text(body, "GAS LP", (-3.07, 1.9, 0), (0, 0, 1), (0, 1, 0), 0.22, acc, PAINT_MATE)
    # pasarela, valvulas, escalera, carrete de manguera
    box(body, -2.4, 2.9, -0.3, 0.7, 2.93, 0.3, GREY, ALU)
    for k in range(6):
        x = -2.3 + k * 0.6
        both(lambda s: bar(body, (x, 2.93, s * 0.3), (x, 3.25, s * 0.3), 0.025, rgb(240, 200, 30), PAINT_MATE))
    both(lambda s: bar(body, (-2.35, 3.25, s * 0.3), (0.65, 3.25, s * 0.3), 0.025, rgb(240, 200, 30), PAINT_MATE))
    for x in (-1.6, -0.2):
        cylinder(body, (x, 2.85, 0.0), (x, 3.02, 0.0), 0.08, rgb(40, 120, 200), PAINT_MATE, seg=10)
    for k in range(8):
        y = 0.95 + k * 0.28
        bar(body, (-3.05, y, -0.35), (-3.05, y, -0.05), 0.03, GREY, ALU)
    both(lambda s: bar(body, (-3.05, 0.9, s * 0.2 - 0.15), (-3.05, 2.95, s * 0.2 - 0.15), 0.035, GREY, ALU))
    cylinder(body, (-0.3, 0.95, 1.0), (-0.3, 0.95, 1.18), 0.3, rgb(200, 30, 30), PAINT_MATE, seg=14)
    revolve(body, [(0.2, -0.08), (0.24, 0.0), (0.2, 0.08)], (-0.3, 0.95, 1.1), (0, 0, 1), BLACK, RUBBER, seg=14)
    for s in (-1, 1):
        lamp(body, (-3.02, 0.85, s * 0.95), (0.03, 0.08, 0.12), TAIL_C, TAIL_M, 0.35)
        box(body, -2.95, 0.4, s * 0.98, -2.9, 0.95, s * 1.15, BLACK, RUBBER)
    box(body, -3.05, 0.55, -1.1, -2.98, 0.68, 1.1, acc, PAINT_MATE)
    cylinder(body, (1.5, 1.0, 0.98), (1.5, 3.2, 0.98), 0.08, CHROME_C, CHROME, seg=10)
    driver(body, p["driver"], 2.35, 1.3, -0.45, 1.0)
    info["wheels"] = [(2.2, wz), (-1.4, wz), (-2.3, wz)]
    info["exhaust"] = [(1.5, 3.2, 0.98)]
    info["head"] = [(3.05, 1.05, s * 0.82) for s in (-1, 1)]
    return r, 0.36, make_wheel("steel", r, 0.36, rim=rgb(200, 200, 205))


BUILDERS = {"tsuru": _tsuru, "f1": _f1, "f2": lambda p, b, g, i: _f1(p, b, g, i, f2=True), "vocho": _vocho,
            "chevy": _chevy, "vette": _vette, "nascar": _nascar, "gtr": _gtr, "gt3": _gt3, "baja": _baja,
            "sport": _sport, "hatch": _hatch, "pickup": _pickup, "van": _van, "truck": _truck, "gas": _gas}


def build_vehicle_hd2(p):
    """Mismo contrato que gtc_racing_hd.build_vehicle_hd: (body, glass, wheel, info)."""
    kind = p["vehicle"]
    if kind not in BUILDERS:
        return _ORIG_BUILD(p)
    body, glass = Geo(), Geo()
    info = {}
    r, w, wheel = BUILDERS[kind](p, body, glass, info)
    if glass.count() == 0:                       # el juego no acepta mallas vacias: triangulo escondido
        glass.tris([(0, 0.3, 0), (0.01, 0.3, 0), (0, 0.31, 0)], (0, 1, 0), GLASS_C, GLASS_M)
    info["radius"], info["width"] = r, w
    info.setdefault("head", [(2.0, 0.7, -0.5), (2.0, 0.7, 0.5)])
    return body, glass, wheel, info


_ORIG_BUILD = hd.build_vehicle_hd

# ================================================================ datos de vehiculos y pilotos
VEH_EXTRA = {
    "tsuru": dict(length=4.3, width=1.68, radius=1.12, cam_d=9.2, cam_h=3.6, hood=1.7),
    "vocho": dict(length=4.05, width=1.6, radius=1.08, cam_d=9.0, cam_h=3.6, hood=1.75),
    "chevy": dict(length=3.8, width=1.62, radius=1.08, cam_d=8.8, cam_h=3.5, hood=1.7),
    "f1": dict(length=5.2, width=1.95, radius=1.2, cam_d=9.5, cam_h=3.0, hood=1.15),
    "f2": dict(length=5.2, width=1.92, radius=1.2, cam_d=9.5, cam_h=3.0, hood=1.15),
    "vette": dict(length=4.6, width=1.96, radius=1.2, cam_d=9.6, cam_h=3.5, hood=1.45),
    "nascar": dict(length=5.0, width=1.98, radius=1.25, cam_d=10.0, cam_h=3.6, hood=1.6),
    "gtr": dict(length=4.7, width=1.92, radius=1.2, cam_d=9.6, cam_h=3.5, hood=1.55),
    "gt3": dict(length=4.9, width=2.04, radius=1.25, cam_d=9.9, cam_h=3.5, hood=1.5),
    "baja": dict(length=5.6, width=2.3, radius=1.45, cam_d=12.0, cam_h=4.6, hood=2.3),
}

ROSTER_EXTRA = [
    {"name": "EL VOCHERO", "brand": "Taxi ecológico CDMX", "vehicle": "vocho", "vname": "Vocho taxi verde",
     "body": (25, 135, 65), "accent": (245, 245, 240), "map": (40, 190, 90), "sign": 11, "driver": "vochero",
     "vmax": 84.0, "accel": 12.6, "grip": 26.8, "mass": 0.8, "num": 13},
    {"name": "DOÑA CHUY", "brand": "Tamales Doña Chuy", "vehicle": "chevy", "vname": "Chevy Pop con olla tamalera",
     "body": (35, 160, 175), "accent": (250, 240, 215), "map": (60, 210, 220), "sign": 15, "driver": "chuy",
     "vmax": 85.0, "accel": 12.4, "grip": 27.2, "mass": 0.82},
    {"name": "EL TEXANO", "brand": "Garage Monterrey V8", "vehicle": "vette", "vname": "GT americano V8 (tipo Corvette)",
     "body": (200, 18, 28), "accent": (20, 20, 22), "map": (255, 60, 60), "sign": 3, "driver": "texano",
     "vmax": 96.0, "accel": 11.4, "grip": 25.6, "mass": 1.2},
    {"name": "EL TRAILERO", "brand": "Refaccionaria El Güero", "vehicle": "nascar", "vname": "Stock car tipo NASCAR",
     "body": (245, 115, 20), "accent": (20, 50, 140), "map": (255, 150, 40), "sign": 15, "driver": "casco_nas",
     "vmax": 97.0, "accel": 10.8, "grip": 25.0, "mass": 1.35, "num": 52},
    {"name": "EL TUNERO", "brand": "Drift Tijuana", "vehicle": "gtr", "vname": "Superdeportivo japonés tipo GT-R",
     "body": (75, 35, 125), "accent": (225, 228, 235), "map": (170, 90, 255), "sign": 0, "driver": "tunero",
     "vmax": 93.0, "accel": 12.6, "grip": 27.0, "mass": 1.15, "num": 32},
    {"name": "EL INGE", "brand": "Taller Alemán Puebla", "vehicle": "gt3", "vname": "GT3 alemán tipo M4",
     "body": (240, 240, 244), "accent": (25, 55, 150), "map": (200, 215, 255), "sign": 0, "driver": "casco_de",
     "vmax": 94.0, "accel": 11.8, "grip": 27.8, "mass": 1.1, "num": 96},
    {"name": "EL COYOTE", "brand": "Ensenada Off-Road", "vehicle": "baja", "vname": "Trophy truck Baja 1000",
     "body": (240, 100, 20), "accent": (25, 25, 28), "map": (255, 120, 40), "sign": 15, "driver": "casco_baja",
     "vmax": 90.0, "accel": 12.0, "grip": 24.6, "mass": 1.7, "num": 13},
    {"name": "EL TAXISTA", "brand": "Sitio Bellas Artes", "vehicle": "tsuru", "vname": "Tsuru taxi guinda y oro",
     "body": (125, 22, 45), "accent": (196, 160, 98), "map": (200, 60, 90), "sign": 11, "driver": "taxista",
     "vmax": 87.0, "accel": 11.3, "grip": 26.2, "mass": 0.95, "num": 27},
    {"name": "EL HALCÓN AZTECA", "brand": "Escudería Azteca", "vehicle": "f1", "vname": "Fórmula 1",
     "body": (0, 120, 65), "accent": (245, 245, 245), "map": (0, 200, 110), "sign": 0, "driver": "casco_mx",
     "vmax": 95.0, "accel": 12.4, "grip": 28.6, "mass": 0.72, "num": 7},
    {"name": "LA CENTELLA", "brand": "Jalisco Junior Team", "vehicle": "f2", "vname": "Fórmula 2",
     "body": (20, 60, 170), "accent": (245, 125, 20), "map": (70, 120, 255), "sign": 0, "driver": "casco_f2",
     "vmax": 93.5, "accel": 12.0, "grip": 28.0, "mass": 0.76, "num": 21},
]


GRID_SIZE = 6          # el juego base esta hecho para 6 autos en la parrilla (lista 'jitter' de 6)


def _patch_game():
    """Ajustes en tiempo de ejecucion de GameHD (sin tocar los archivos) para que funcione con mas de 6 pilotos:
    - new_race: elige al jugador + 5 rivales al azar (el original metia a TODOS los pilotos y truena con 7+).
      Si el juego tiene el atributo 'grid_pilots' (lista de dicts de PILOTS), se usan esos rivales.
    - hud_select: la tira de nombres de abajo muestra una ventana de 6 pilotos alrededor del seleccionado."""
    G = hd.GameHD
    if getattr(G, "_autos_patched", False):
        return
    orig_new_race, orig_hud = G.new_race, G.hud_select

    class _Grid(list):
        """Lista completa para indexar (PILOTS[sel]) pero que al recorrerse solo da la parrilla elegida."""

        def __init__(self, full, subset):
            list.__init__(self, full)
            self._subset = subset

        def __iter__(self):
            return iter(self._subset)

    def new_race(self, attract=False, intro=True):
        full = hd.PILOTS
        if len(full) <= GRID_SIZE:
            return orig_new_race(self, attract, intro)
        me = full[self.sel] if not attract else full[-1]
        pick = [q for q in (getattr(self, "grid_pilots", None) or []) if q is not me and q in full]
        others = [q for q in full if q is not me and q not in pick]
        random.shuffle(others)
        rivals = (pick + others)[:GRID_SIZE - 1]
        hd.PILOTS = _Grid(full, [me] + rivals)
        try:
            return orig_new_race(self, attract, intro)
        finally:
            hd.PILOTS = full

    def hud_select(self):
        full, sel = hd.PILOTS, self.sel
        n = len(full)
        if n <= GRID_SIZE + 1:
            return orig_hud(self)
        start = min(max(sel - GRID_SIZE // 2, 0), n - GRID_SIZE)
        hd.PILOTS, self.sel = list(full[start:start + GRID_SIZE]), sel - start
        try:
            orig_hud(self)
        finally:
            hd.PILOTS, self.sel = full, sel
        self.txt(("‹ " if start > 0 else "") + f"{sel + 1} / {n}" + (" ›" if start + GRID_SIZE < n else ""), 18,
                 (200, 210, 225), (1040, hd.WIN_H - 78), italic=False)

    G.new_race, G.hud_select = new_race, hud_select
    G._autos_patched = True


def install(patch_game=True):
    """Integra autos y pilotos al juego. Llamar ANTES de crear el juego (GameHD/GameMaps arman los modelos
    en __init__). Es idempotente."""
    base.VEH.update(VEH_EXTRA)
    names = {q["name"] for q in base.PILOTS}
    nuevos = [p for p in ROSTER_EXTRA if p["name"] not in names]
    if nuevos:
        k = len(base.PILOTS) - 1 if base.PILOTS and base.PILOTS[-1]["name"] == "PILOTO GTC" else len(base.PILOTS)
        base.PILOTS[k:k] = nuevos          # el ultimo sigue siendo PILOTO GTC (seleccion por defecto)
    hd.build_vehicle_hd = build_vehicle_hd2
    hd.character = character2
    if patch_game:
        _patch_game()
    return base.PILOTS

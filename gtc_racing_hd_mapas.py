"""
GTC RACING HD - MAPAS

Version HD con varias pistas. Usa todo lo de gtc_racing_hd.py (graficos, autos, musica, multijugador LAN)
sin modificarlo, y agrega:
  - un catalogo de mapas (lista MAPS, abajo)
  - un constructor de escenario automatico para cualquier trazado nuevo
  - selector de pista:  Q / E  en la seleccion de piloto (y en la sala LAN para el anfitrion)

PISTAS
  ESTADIO GNP SEGUROS  Autodromo original con el concierto de Avenged Sevenfold (noche).
  ESTADIO FRAY NANO    Circuito callejero CDMX: sale junto al Estadio Fray Nano (concierto de Limp Bizkit
                       de noche), sube por el Blvd. Puerto Aereo junto al aeropuerto (aviones aterrizando y
                       despegando), baja por el Palacio de los Deportes y el Estadio GNP. Puentes peatonales,
                       semaforos, cebras, locales, puestos y mucho publico.
                       Soundtrack de noche: pon las canciones en la carpeta "music_limp_bizkit"
                       (mientras este vacia suena la musica de carrera del juego).
  CHAPULTEPEC          Fiesta del Mundial, 11.6 km de pura velocidad: sale del Museo de Antropologia por
                       Paseo de la Reforma (Angel, torres), cruza Insurgentes, pasa bajo los arcos del Monumento
                       a la Revolucion (Fan Fest), sube a la Basilica, regresa por Insurgentes y entra al bosque
                       rodeando el Castillo. Rampas de turbo, banderines, balones gigantes y, de noche, fuegos
                       artificiales en toda la ruta. Soundtrack de noche: carpeta "music_mundial".

COMO AGREGAR UNA PISTA NUEVA
  Copia la plantilla PLANTILLA_DE_MAPA, cambia la clave, el nombre y los puntos del trazado (en metros,
  recorridos en el sentido de la carrera, la meta esta en el primer punto) y agregala a MAPS.
  builder="calles" arma un circuito callejero como el de Fray Nano.

Requiere en la misma carpeta: gtc_racing_hd.py y gtc_racing_3d.py.
"""
import math
import os
import random
import sys
import threading
import time

import numpy as np
import pygame

import gtc_racing_3d as base
import gtc_racing_hd as hd
from gtc_racing_hd import (Geo, Crowd, grid, quad, obox, box, cylinder, frame_box, rgb, mat, norm, F4, TAU, HW, LIMIT,
                           CURB_W, PILOTS, DIFFS, TOD_HD, WIN_W, WIN_H, ACCENT, ORANGE, smooth, font,
                           M_GRASS, M_GRAVEL, M_CONC, M_FACADE, M_LAMP, M_SIGN, M_METAL, M_LED, SPONSORS_HD, bb_uv,
                           M_EMIS, M_LINE, M_DEF, M_PAINT, M_WINDOWS, M_LEAF, M_BARK,
                           revolve, sphere, transform_geo, billboard_hd, model_full, setu, gl_bytes,
                           build_track_hd, grandstand_hd, bowl_hd, bridge_hd, tree_hd, angel_hd, flagpole_hd,
                           guadalupe_wall_hd, papel_picado_hd, city_hd, mountain_hd, build_concert_hd)

ORIG_SIGNS = list(hd.SIGNS)
ORIG_BB_UV = hd.bb_uv


# ================================================================ catalogo de mapas
class MapDef:
    """Definicion de una pista.

    key          clave corta unica (sin espacios), se usa en el multijugador
    name         nombre que se muestra
    subtitle     descripcion corta
    control      puntos del trazado [(x, z), ...] en metros, en el sentido de la carrera (meta = primer punto)
    scale        escala aplicada a todos los puntos
    stadium      (inicio, fin) indices de 'control' que pasan por dentro de un estadio, o None
    stadium_scale escala de la seccion del estadio (alrededor de su centro)
    concert      True = concierto de noche en el estadio (requiere stadium o venue)
    laps         vueltas de la carrera
    ground       suelo alrededor: "pasto", "tierra" o "concreto"
    trees        cantidad de arboles (0 = ninguno)
    city         edificios en el horizonte
    mountains    cordillera alrededor
    volcanoes    Popocatepetl e Iztaccihuatl al oriente
    extras       monumentos: "angel", "bandera", "guadalupe", "papel"
    stands       tribunas extra en las rectas (ademas de la principal)
    bridges      puentes peatonales sobre la pista
    pits         edificio de pits junto a la recta principal
    builder      "cdmx" = escenario original del Autodromo; "auto" = constructor automatico;
                 "calles" = circuito callejero CDMX
    venue        estadio junto a la pista (no la cruza) con el concierto de noche:
                 dict(name, band, pos=(x, z), r0=radio del campo, rows=filas de gradas, kind="beisbol")
    landmarks    lugares en el paisaje: [("gnp", (x, z)), ("palacio", (x, z)), ("aeropuerto", (x, z), angulo)]
    urban        circuito callejero CDMX: edificios, locales, postes, semaforos, cruces de cebra, puestos
    ped_bridges  puentes peatonales (circuito callejero)
    fans         probabilidad de grupos de aficionados en cada tramo de banqueta (0-1)
    signs        letreros propios del mapa {indice_del_atlas: (fondo, fondo2, texto, color_texto)}
    band_sign    indice del atlas con el nombre de la banda (se pone en el escenario)
    music_dir    carpeta con el soundtrack de noche (si esta vacia suena la cancion de carrera)
    """

    def __init__(self, key, name, subtitle, control, scale=1.0, stadium=None, stadium_scale=1.0, concert=False,
                 laps=3, ground="pasto", trees=700, city=True, mountains=True, volcanoes=True,
                 extras=("papel",), stands=4, bridges=2, pits=True, builder="auto", venue=None, landmarks=(),
                 urban=False, ped_bridges=0, fans=0.75, signs=None, band_sign=10, music_dir="music",
                 palette=None, fireworks=False, pads=0, city_margin=750.0, city_step=34.0, detail_dist=400.0):
        self.key, self.name, self.subtitle = key, name, subtitle
        self.control, self.scale = list(control), scale
        self.stadium, self.stadium_scale = stadium, stadium_scale
        self.venue = venue
        self.concert = concert and (stadium is not None or venue is not None)
        self.laps, self.ground, self.trees = laps, ground, trees
        self.city, self.mountains, self.volcanoes = city, mountains, volcanoes
        self.extras, self.stands, self.bridges, self.pits = tuple(extras), stands, bridges, pits
        self.builder = builder
        self.landmarks, self.urban, self.ped_bridges, self.fans = list(landmarks), urban, ped_bridges, fans
        self.signs, self.band_sign, self.music_dir = dict(signs or {}), band_sign, music_dir
        self.venue_name = (venue or {}).get("name", "ESTADIO GNP SEGUROS")
        self.band = (venue or {}).get("band", "AVENGED SEVENFOLD")
        self.venue_tag = (venue or {}).get("tag", "EN VIVO")
        self.palette, self.fireworks, self.pads = palette, fireworks, pads
        self.city_margin, self.city_step, self.detail_dist = city_margin, city_step, detail_dist
        self.length = None

    def w(self, p):
        """Punto del mapa (mismas unidades que 'control') a metros del mundo."""
        return np.array(p, float) * self.scale


# Ruta de las calles alrededor del Estadio Fray Nano (puntos tomados del mapa, 1 px = 4.5 m).
# Sale del punto negro (meta) hacia el noreste, toma el Blvd. Puerto Aereo al norte, pasa junto al
# aeropuerto, baja hacia el Viaducto / Palacio de los Deportes / GNP y regresa en horquilla a la meta.
FRAY_NANO_ROUTE = [(414, 380), (445, 341), (500, 272), (540, 226), (575, 221), (622, 224), (645, 206), (672, 145),
                   (700, 80), (716, 42), (745, 22), (790, 24), (830, 62), (885, 104), (942, 138), (930, 182),
                   (900, 220), (860, 300), (830, 362), (805, 403), (770, 410), (700, 405), (640, 405), (550, 418),
                   (470, 420), (422, 420), (396, 412), (390, 397)]

# Ruta del Mundial (puntos del mapa, 1 px = 4.5 m): sale del Museo de Antropologia por Paseo de la Reforma,
# pasa el Angel, cruza Insurgentes, pasa bajo el Monumento a la Revolucion, sube a la Basilica, regresa por
# Insurgentes y entra al bosque de Chapultepec rodeando el Castillo hasta la meta.
CHAPULTEPEC_ROUTE = [
    (45, 687), (110, 694), (185, 702), (215, 698), (245, 689), (275, 678), (305, 667), (335, 657), (360, 648),
    (380, 637), (398, 610), (415, 582), (425, 562), (436, 548), (452, 543), (475, 547), (505, 549), (530, 535),
    (555, 505), (580, 478), (610, 452), (640, 432), (660, 400), (680, 355), (700, 310), (712, 270), (725, 230),
    (745, 180), (768, 135), (785, 100), (795, 70), (790, 40), (775, 25), (752, 17), (735, 22), (715, 45),
    (690, 85), (660, 125), (630, 170), (600, 195), (570, 222), (545, 250), (520, 280), (495, 312), (470, 340),
    (452, 368), (440, 420), (428, 470), (410, 510), (390, 550), (372, 590), (358, 625), (350, 655), (343, 685),
    (336, 708), (320, 717), (290, 719), (255, 723), (220, 730), (190, 737), (150, 738), (100, 736), (60, 735),
    (25, 734), (-5, 740), (-35, 748), (-62, 738), (-78, 714), (-70, 690), (-45, 680), (-15, 683), (15, 686)]

MAPS = [
    MapDef("cdmx", "ESTADIO GNP SEGUROS", "Autódromo · concierto de Avenged Sevenfold",
           control=base.CONTROL, scale=base.TRACK_SCALE,
           stadium=(base.STADIUM_PTS.start, base.STADIUM_PTS.stop), stadium_scale=base.STADIUM_SCALE,
           concert=True, extras=("angel", "bandera", "guadalupe", "papel"), builder="cdmx"),
    MapDef("fraynano", "ESTADIO FRAY NANO", "Calles de la CDMX · concierto de Limp Bizkit",
           control=FRAY_NANO_ROUTE, scale=4.5, laps=2, concert=True, builder="calles", urban=True,
           ped_bridges=9, fans=0.9, extras=("papel", "guadalupe"),
           venue=dict(name="ESTADIO FRAY NANO", band="LIMP BIZKIT", pos=(345, 372), r0=62, rows=26),
           landmarks=[("gnp", (690, 472)), ("palacio", (565, 464)), ("aeropuerto", (800, -22), (0.77, -0.64))],
           signs={1: ((10, 60, 130), (30, 100, 190), "AEROPUERTO AICM", (255, 255, 255)),
                  2: ((5, 5, 5), (45, 5, 5), "AVENGED SEVENFOLD", (240, 40, 40)),
                  9: ((0, 70, 40), (20, 120, 60), "ESTADIO FRAY NANO", (255, 255, 255)),
                  10: ((5, 5, 5), (70, 30, 0), "LIMP BIZKIT", (255, 120, 20)),
                  14: ((5, 5, 5), (50, 30, 0), "HOY EN VIVO · LIMP BIZKIT", (255, 170, 40))},
           band_sign=10, music_dir="music_limp_bizkit"),
    MapDef("chapultepec", "CHAPULTEPEC", "Reforma · Insurgentes · fiesta del Mundial",
           control=CHAPULTEPEC_ROUTE, scale=4.5, laps=1, concert=True, builder="calles", urban=True,
           ped_bridges=10, fans=0.85, extras=("banderines", "balones", "guadalupe"), palette="mexico",
           fireworks=True, pads=14, city_margin=450.0, city_step=40.0, detail_dist=250.0,
           venue=dict(name="MONUMENTO A LA REVOLUCIÓN", band="FIESTA MUNDIALISTA", tag="¡VAMOS MÉXICO!",
                      pos=(482, 600), r0=55, kind="fanfest"),
           landmarks=[("museo", (95, 662)), ("parque", ((-110, 650), (125, 778)), 900), ("lago", (70, 713), (70, 24)),
                      ("castillo", (-36, 713)), ("angel", (240, 676)), ("torres", (0.025, 0.105), 10),
                      ("monumento", (420, 570)), ("basilica", (815, 35))],
           signs={1: ((0, 104, 71), (0, 140, 90), "MUNDIAL 2026", (255, 255, 255)),
                  2: ((206, 17, 38), (240, 40, 60), "¡GOOOOL!", (255, 255, 255)),
                  9: ((0, 104, 71), (20, 140, 90), "¡VAMOS MÉXICO!", (255, 255, 255)),
                  10: ((10, 10, 10), (0, 80, 50), "FAN FEST CDMX", (255, 215, 40)),
                  12: ((10, 25, 80), (25, 60, 150), "PASEO DE LA REFORMA", (255, 255, 255)),
                  14: ((5, 5, 5), (0, 60, 40), "HOY: FIESTA MUNDIALISTA", (255, 215, 40))},
           band_sign=10, music_dir="music_mundial"),
    # --- aqui van las pistas nuevas ---
]

# Plantilla (NO esta en MAPS). Copiala, cambia los datos y agregala a la lista de arriba.
PLANTILLA_DE_MAPA = MapDef(
    "clave", "NOMBRE DE LA PISTA", "Descripción corta",
    control=[(0, 0), (600, 0), (700, 80), (650, 220), (300, 260), (0, 220), (-90, 110)],
    scale=1.0, laps=3, ground="pasto", trees=500, extras=("papel",), stands=3, bridges=1)


def map_index(key):
    for i, m in enumerate(MAPS):
        if m.key == key:
            return i
    return None


# ================================================================ pista a partir de un mapa
class MapTrack(base.Track):
    """Igual que la pista original, pero con los puntos de cualquier MapDef."""

    def __init__(self, md):
        self.md = md
        raw = np.array(md.control, float)
        ctrl = raw * md.scale
        if md.stadium is not None:
            sl = slice(*md.stadium)
            st = raw[sl]
            cst = st.mean(axis=0)
            ctrl[sl] = cst * md.scale + (st - cst) * md.stadium_scale
        self.ctrl = ctrl
        self.pts = base.centripetal_cr(ctrl, base.SPACING)
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
        self.nrm = np.column_stack([-t[:, 1], t[:, 0]])
        head = np.arctan2(t[:, 1], t[:, 0])
        dh = (head - np.roll(head, 1) + np.pi) % TAU - np.pi
        k = dh / self.spacing
        self.curv = sum(np.roll(k, j) for j in range(-3, 4)) / 7.0
        self.sand = np.zeros((self.n, 2), bool)
        sharp = np.abs(self.curv) > 1 / 110
        self.sand[sharp & (self.curv > 0), 0] = True
        self.sand[sharp & (self.curv < 0), 1] = True
        if md.urban:                      # en la calle no hay grava: muros
            self.sand[:] = False
        self.center = self.pts.mean(axis=0)
        self.cross_pts = self.find_crossings() if md.urban else []

    def find_crossings(self):
        """Lugares donde la pista se cruza consigo misma: [(i, j, punto)] con i < j."""
        P, n = self.pts, self.n
        cand = []
        for i in range(0, n, 2):
            d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
            ds = np.abs(self.cum - self.cum[i])
            d[np.minimum(ds, self.L - ds) < 150] = 1e9
            j = int(np.argmin(d))
            if d[j] < 2 * LIMIT:
                cand.append((float(d[j]), i, j))
        out = []
        for d, i, j in sorted(cand):
            i, j = min(i, j), max(i, j)
            if any(min(abs(i - a), n - abs(i - a)) < 60 or min(abs(i - b), n - abs(i - b)) < 60 for a, b, _ in out):
                continue
            out.append((i, j, (P[i] + P[j]) / 2))
        return out

    def project(self, x, z):
        """Con cruces: cada coche se busca solo cerca de donde iba (para no 'saltar' a la otra calle)."""
        if not self.cross_pts:
            return base.Track.project(self, x, z)
        car = sys._getframe(1).f_locals.get("self")
        i0 = getattr(car, "idx", None)
        if not isinstance(i0, (int, np.integer)):
            return base.Track.project(self, x, z)
        w = (np.arange(-30, 90) + int(i0)) % self.n
        px, pz = x - self.pts[w, 0], z - self.pts[w, 1]
        d = self.d[w]
        t = np.clip((px * d[:, 0] + pz * d[:, 1]) * self.inv_len2[w], 0, 1)
        ex, ez = px - d[:, 0] * t, pz - d[:, 1] * t
        e2 = ex * ex + ez * ez
        k = int(np.argmin(e2))
        if e2[k] > (3 * LIMIT) ** 2:                    # el coche aparecio en otro lado
            return base.Track.project(self, x, z)
        i = int(w[k])
        nx, nz = self.nrm[i]
        lat = ex[k] * nx + ez[k] * nz
        cx, cz = self.pts[i, 0] + d[k, 0] * t[k], self.pts[i, 1] + d[k, 1] * t[k]
        return float(self.cum[i] + t[k] * self.seg_len[i]), i, float(lat), cx, cz, nx, nz


# ================================================================ constructor de escenario automatico
def straight_runs(track, max_curv=1 / 800, min_len=120.0):
    """Rectas de la pista: [(indice_inicio, num_muestras), ...] de la mas larga a la mas corta."""
    n = track.n
    ok = np.abs(track.curv) < max_curv
    runs = []
    if ok.all():
        return [(0, n)]
    start = int(np.argmin(ok))
    i, cur = 0, None
    while i < n:
        k = (start + i) % n
        if ok[k]:
            cur = k if cur is None else cur
            ln = 1
            while i + ln < n and ok[(start + i + ln) % n]:
                ln += 1
            if ln * track.spacing >= min_len:
                runs.append((k, ln))
            i += ln
            cur = None
        else:
            i += 1
    return sorted(runs, key=lambda r: -r[1])


def footprint_clear(track, o, along, out, length, depth, margin=3.0):
    q = np.array([o + along * a + out * d for a in np.linspace(0, length, 8) for d in (0, depth / 2, depth)])
    return track.nearest_dist(q).min() >= LIMIT + margin


def pit_building_oriented(g, o, along, out, length, lights):
    """Edificio de pits orientado a cualquier recta."""
    white, dark = rgb(235, 236, 238), rgb(28, 30, 34)
    ax = np.array([along[0], 0, along[1]], F4)
    oz = np.array([out[0], 0, out[1]], F4)
    org = np.array([o[0], 0, o[1]], F4)
    obox(g, org, ax * length, (0, 4.2, 0), oz * 18, white, mat(M_CONC, 0.6))
    obox(g, org + (0, 4.2, 0) - oz * 1.0, ax * length, (0, 6.8, 0), oz * 19, rgb(60, 80, 100), mat(M_FACADE, 0.12, 0.6))
    obox(g, org + (0, 11.0, 0) - oz * 3 - ax * 2, ax * (length + 4), (0, 0.6, 0), oz * 22, white, mat(M_METAL, 0.4, 0.2))
    for k in range(int(length // 21)):
        p0 = org + ax * (6 + k * 20.6) - oz * 0.06
        p1 = p0 + ax * 15
        quad(g, (p1[0], 0.2, p1[2]), (p0[0], 0.2, p0[2]), (p0[0], 3.4, p0[2]), (p1[0], 3.4, p1[2]), (1.0, 0.82, 0.6),
             mat(M_LAMP, 0.6, 0, 1.6))
        u0, u1, v0, v1 = bb_uv(SPONSORS_HD[k % len(SPONSORS_HD)])
        quad(g, (p1[0], 3.7, p1[2]), (p0[0], 3.7, p0[2]), (p0[0], 4.15, p0[2]), (p1[0], 4.15, p1[2]), (1, 1, 1),
             mat(M_SIGN, 0.35), ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))
    for k in range(3):
        c = org + ax * (length * (k + 0.5) / 3) - oz * 6
        lights.append((float(c[0]), 6.0, float(c[2]), 40.0, 1.0, 0.85, 0.7, 3.0))


def no_concert_info(track):
    far = np.array([1e6, 1e6])
    return dict(stage=(far, np.array([1.0, 0.0]), np.array([0.0, 1.0])), pit=(1e6, 1e6), launch=[], flames=[],
                fireballs=[], fountain=[], movers=[], stadium=(track.center, 0.0, 0.0))


def build_world_auto(track, md):
    """Escenario completo para cualquier trazado, con las opciones de su MapDef."""
    rnd = random.Random(sum(map(ord, md.key)) * 7919)
    g, gc, crowd, lights, info = Geo(), Geo(), Crowd(), [], {}
    pts, nrm, tan, n = track.pts, track.nrm, track.tan, track.n
    cxm, czm = track.center
    G = 6500
    xs = np.linspace(cxm - G, cxm + G, 41)
    zs = np.linspace(czm - G, czm + G, 41)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    ground = {"pasto": (rgb(70, 120, 45), M_GRASS, 0.95), "tierra": (rgb(150, 120, 85), M_GRAVEL, 0.95),
              "concreto": (rgb(120, 120, 118), M_CONC, 0.85)}.get(md.ground, (rgb(70, 120, 45), M_GRASS, 0.95))
    grid(g, np.stack([X, np.zeros_like(X), Z], -1), ground[0], mat(ground[1], ground[2]), orient=(0, 1, 0))
    build_track_hd(g, track, info, lights)
    keep = []
    # ---- estadio (opcional) y concierto
    rmax = 0.0
    if md.stadium is not None:
        sl = slice(*md.stadium)
        sc = track.ctrl[sl].mean(axis=0)
        r0 = float(np.max(np.linalg.norm(track.ctrl[sl] - sc, axis=1))) + LIMIT + 6
        ang = np.linspace(0, TAU, 65)
        for j in range(64):
            quad(g, (sc[0], 0.03, sc[1]), (sc[0], 0.03, sc[1]),
                 (sc[0] + r0 * math.cos(ang[j + 1]), 0.03, sc[1] + r0 * math.sin(ang[j + 1])),
                 (sc[0] + r0 * math.cos(ang[j]), 0.03, sc[1] + r0 * math.sin(ang[j])), rgb(150, 146, 138),
                 mat(M_CONC, 0.75))
        rmax, Hb = bowl_hd(g, crowd, rnd, track, sc, r0, lights)
        keep.append((sc[0], sc[1], rmax + 25))
        if md.concert:
            build_concert_hd(gc, crowd, rnd, track, sc, r0, rmax, Hb, info)
        info["stadium"] = (sc, rmax, r0)
    if "stage" not in info:
        info.update({k: v for k, v in no_concert_info(track).items() if k not in info})
    sc_ = info["stadium"][0]

    def far_from_stadium(c, extra=10.0):
        return rmax <= 0 or math.hypot(c[0] - sc_[0], c[1] - sc_[1]) > rmax + extra
    # ---- tribuna principal en la recta mas larga + pits enfrente
    runs = straight_runs(track)
    used = []
    if runs:
        i0, ln = runs[0]
        length = min(760.0, ln * track.spacing - 40)
        i_mid = (i0 + ln // 2) % n
        i_start = (i_mid - int(length / 2 / track.spacing)) % n
        along = tan[i_start]
        for side in (-1, 1):
            out = nrm[i_start] * side
            o = pts[i_start] + out * (LIMIT + 4)
            if footprint_clear(track, o, along, out, length, 24) and far_from_stadium(o + along * length / 2):
                grandstand_hd(g, crowd, rnd, o, along, out, length, 24, 0.9, 0.5, rgb(30, 80, 190), 14, lights)
                c = o + along * length / 2 + out * 12
                keep.append((c[0], c[1], length / 2 + 30))
                used.append(i_mid)
                if md.pits:
                    po = pts[i_start] - out * (LIMIT + 8) + along * 40
                    plen = min(420.0, length - 80)
                    if plen > 100 and footprint_clear(track, po, along, -out, plen, 20, 2.0):
                        pit_building_oriented(g, po, along, -out, plen, lights)
                        c = po + along * plen / 2 - out * 10
                        keep.append((c[0], c[1], plen / 2 + 20))
                break
    # ---- tribunas extra
    for i0, ln in runs[1:]:
        if len(used) > md.stands:
            break
        i_mid = (i0 + ln // 2) % n
        if any(min(abs(i_mid - j), n - abs(i_mid - j)) < 70 for j in used):
            continue
        length = min(90.0, ln * track.spacing - 20)
        i_start = (i_mid - int(length / 2 / track.spacing)) % n
        for side in (1, -1):
            out = nrm[i_start] * side
            o = pts[i_start] + out * (LIMIT + 4)
            if footprint_clear(track, o, tan[i_start], out, length, 16) and far_from_stadium(o):
                grandstand_hd(g, crowd, rnd, o, tan[i_start], out, length, 16, 0.9, 0.5,
                              rnd.choice([rgb(30, 80, 190), rgb(220, 120, 30), rgb(190, 40, 40), rgb(40, 150, 70)]), 9,
                              lights)
                c = o + tan[i_start] * length / 2 + out * 8
                keep.append((c[0], c[1], length / 2 + 25))
                used.append(i_mid)
                break
    # ---- puentes
    placed_b = []
    for i0, ln in runs:
        if len(placed_b) >= md.bridges:
            break
        i = (i0 + ln // 3) % n
        if any(min(abs(i - j), n - abs(i - j)) < n * 0.15 for j in placed_b):
            continue
        side_pts = np.array([pts[i] + nrm[i] * s * (LIMIT + 2.6) for s in (-1, 1)])
        if track.nearest_dist(side_pts).min() < LIMIT + 2:
            continue
        bridge_hd(g, track, i, [0, 12, 2][len(placed_b) % 3], lights)
        placed_b.append(i)

    # ---- monumentos
    def spot(clear):
        for _ in range(400):
            a = rnd.uniform(0, TAU)
            r = rnd.uniform(80, 420)
            c = np.array([cxm + r * math.cos(a), czm + r * math.sin(a)])
            if track.nearest_dist(c[None])[0] >= clear and far_from_stadium(c, 30) and \
                    not any(math.hypot(c[0] - kx, c[1] - kz) < kr + clear * 0.5 for kx, kz, kr in keep):
                return c
        return None
    if "angel" in md.extras:
        c = spot(45)
        if c is not None:
            angel_hd(g, c[0], c[1], lights)
            keep.append((c[0], c[1], 32))
            info["angel"] = c
    if "bandera" in md.extras:
        c = spot(34)
        if c is not None:
            flagpole_hd(g, c[0], c[1])
            keep.append((c[0], c[1], 30))
            info["flag"] = c
    bb = set()
    if "guadalupe" in md.extras:
        for frac in (0.12, 0.3, 0.47, 0.68, 0.86):
            i = int(frac * n)
            for side in (1, -1):
                c = pts[i] + nrm[i] * side * (LIMIT + 7)
                if track.nearest_dist(c[None])[0] < LIMIT + 6 or not far_from_stadium(c, 8):
                    continue
                guadalupe_wall_hd(g, c, nrm[i] * side, lights, rnd)
                keep.append((c[0], c[1], 12))
                bb.update(range(i - 8, i + 9))
                break
    if "papel" in md.extras:
        for frac in (0.02, 0.26, 0.55, 0.78, 0.4):
            i = int(frac * n)
            papel_picado_hd(g, pts[i] + nrm[i] * -(HW + 2.5), pts[i] + nrm[i] * (HW + 2.5))
    # ---- aficionados detras de las mallas
    for i in range(0, n, 7):
        if i in bb:
            continue
        for side in (1, -1):
            if rnd.random() > 0.75:
                continue
            base_ = pts[i] + nrm[i] * side * (LIMIT + 1.6)
            if track.nearest_dist(base_[None])[0] < LIMIT + 1.2 or not far_from_stadium(base_, -30):
                continue
            for _ in range(rnd.randint(4, 9)):
                t_, d_ = rnd.uniform(-8, 8), rnd.uniform(0.0, 1.6)
                crowd.add(base_[0] + tan[i, 0] * t_ + nrm[i, 0] * side * d_, 0.0,
                          base_[1] + tan[i, 1] * t_ + nrm[i, 1] * side * d_, -nrm[i, 0] * side, -nrm[i, 1] * side, rnd)
    # ---- arboles, ciudad y montanas
    lo, hi = pts.min(axis=0) - 240, pts.max(axis=0) + 240
    placed_t = 0
    for _ in range(md.trees * 6):
        if placed_t >= md.trees:
            break
        x, z = rnd.uniform(lo[0], hi[0]), rnd.uniform(lo[1], hi[1])
        if not far_from_stadium((x, z), 18) or any(math.hypot(x - kx, z - kz) < kr for kx, kz, kr in keep):
            continue
        if track.nearest_dist(np.array([[x, z]]))[0] < LIMIT + 12:
            continue
        tree_hd(g, x, z, rnd.uniform(0.9, 1.5), rnd)
        placed_t += 1
    span = float(np.max(hi - lo)) / 2
    if md.city:
        city_hd(g, cxm, czm, rnd, lights) if span < 1300 else None
    if md.mountains:
        for k in range(26):
            a = k / 26 * TAU + rnd.uniform(-0.1, 0.1)
            r = max(3500.0, span + 2400) + rnd.uniform(0, 900)
            mountain_hd(g, cxm + r * math.cos(a), czm + r * math.sin(a), rnd.uniform(800, 1400),
                        rnd.uniform(300, 650), rnd)
    if md.volcanoes:
        d0 = max(4300.0, span + 3000)
        mountain_hd(g, cxm + d0, czm + 1000, 1900, 1350, rnd, snow=1.0, seg=36, rings=14)
        mountain_hd(g, cxm + d0 - 300, czm - 1000, 2300, 1100, rnd, snow=1.0, seg=36, rings=14)
    # ---- camaras de TV para el modo demo
    cams = []
    for i in range(0, n, 45):
        side = 1 if (i // 45) % 2 else -1
        c = pts[i] + nrm[i] * side * (LIMIT + 9)
        if track.nearest_dist(c[None])[0] > LIMIT + 6:
            cams.append((float(c[0]), 6.5, float(c[1])))
    info["tvcams"] = cams
    if not gc.parts:                     # sin concierto: malla minima invisible (bajo tierra)
        quad(gc, (0, -50, 0), (0.1, -50, 0), (0.1, -50, 0.1), (0, -50, 0.1), (0, 0, 0), mat(0))
    return g, gc, crowd, lights, info


# ================================================================ circuito callejero CDMX
FACADES = [rgb(222, 126, 84), rgb(236, 196, 112), rgb(196, 72, 70), rgb(112, 164, 186), rgb(236, 226, 204),
           rgb(168, 200, 120), rgb(204, 112, 150), rgb(244, 172, 64), rgb(150, 152, 162), rgb(120, 96, 150),
           rgb(250, 240, 220), rgb(90, 140, 110), rgb(230, 90, 60), rgb(250, 210, 160)]
AWNINGS = [rgb(200, 30, 35), rgb(30, 120, 60), rgb(30, 70, 170), rgb(240, 150, 20), rgb(240, 240, 240),
           rgb(200, 0, 110), rgb(250, 200, 0)]
LONAS = [rgb(220, 30, 120), rgb(30, 90, 200), rgb(240, 170, 20), rgb(30, 160, 80), rgb(200, 30, 30)]
SHOP_SIGNS = [4, 5, 6, 7, 8, 15, 13, 11, 0]
WALK = LIMIT + 7.0                       # frente de los edificios (distancia al centro de la calle)
STORIES = [1] * 5 + [2] * 25 + [3] * 30 + [4] * 20 + [5] * 10 + [6] * 5 + [8] * 3 + [12] * 2


def v2(a):
    return np.asarray(a, float)


_TRI = [0, 1, 2, 0, 2, 3]


def quad_dir(g, a, b, c, d, col, m, want, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    """Quad con la normal hacia 'want' (3D)."""
    a, b, c, d = (np.asarray(v, F4) for v in (a, b, c, d))
    if np.dot(np.cross(b - a, c - a), np.asarray(want, F4)) < 0:
        b, d = d, b
        uv = (uv[0], uv[3], uv[2], uv[1])
    quad(g, a, b, c, d, col, m, uv)


def face_quad(g, a, b, y0, y1, col, m, facing, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    """Quad vertical entre a y b (2D) con la normal hacia 'facing' (2D)."""
    P = np.array([(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y1, a[1])], F4)
    g.tris(P[_TRI], (float(facing[0]), 0.0, float(facing[1])), col, m, np.asarray(uv, F4)[_TRI])


def flat_rect(g, c, f, hl, hw, y, col, m):
    """Rectangulo horizontal orientado (uv en metros)."""
    fx, fz = float(f[0]), float(f[1])
    rx, rz = -fz, fx
    cx, cz = float(c[0]), float(c[1])
    Q = np.array([(cx - fx * hl - rx * hw, cz - fz * hl - rz * hw), (cx + fx * hl - rx * hw, cz + fz * hl - rz * hw),
                  (cx + fx * hl + rx * hw, cz + fz * hl + rz * hw), (cx - fx * hl + rx * hw, cz - fz * hl + rz * hw)],
                 F4)[_TRI]
    P = np.column_stack([Q[:, 0], np.full(6, y, F4), Q[:, 1]])
    g.tris(P, (0.0, 1.0, 0.0), col, m, Q)


_SIDE_V = np.array([0, 1, 1, 0, 1, 0])          # 0 = punto A, 1 = punto B del borde
_SIDE_Y = np.array([0, 0, 1, 0, 1, 1])          # 0 = abajo, 1 = arriba


def fbox(g, c, f, hl, hw, y0, y1, col, m, bottom=False):
    """Caja apoyada en el piso orientada con f (rapida: un solo bloque de triangulos).
    uv de las paredes: (metros a lo largo, altura) para que las ventanas cuadren por piso."""
    fx, fz = float(f[0]), float(f[1])
    rx, rz = -fz, fx
    cx, cz = float(c[0]), float(c[1])
    Q = np.array([(cx - fx * hl - rx * hw, cz - fz * hl - rz * hw), (cx + fx * hl - rx * hw, cz + fz * hl - rz * hw),
                  (cx + fx * hl + rx * hw, cz + fz * hl + rz * hw), (cx - fx * hl + rx * hw, cz - fz * hl + rz * hw)])
    A, B = Q, Q[[1, 2, 3, 0]]
    Ls = np.array([2 * hl, 2 * hw, 2 * hl, 2 * hw])
    Nn = np.array([(-rx, 0, -rz), (fx, 0, fz), (rx, 0, rz), (-fx, 0, -fz)])
    XZ = np.where(_SIDE_V[None, :, None] == 0, A[:, None, :], B[:, None, :])          # (4, 6, 2)
    Y = np.where(_SIDE_Y == 0, y0, y1)
    P = np.concatenate([np.stack([XZ[..., 0], np.broadcast_to(Y, (4, 6)), XZ[..., 1]], -1).reshape(-1, 3),
                        np.column_stack([Q[_TRI, 0], np.full(6, y1), Q[_TRI, 1]])] +
                       ([np.column_stack([Q[_TRI, 0], np.full(6, y0), Q[_TRI, 1]])] if bottom else []))
    N = np.concatenate([np.repeat(Nn, 6, 0), np.tile([(0, 1, 0)], (6, 1))] + ([np.tile([(0, -1, 0)], (6, 1))]
                                                                              if bottom else []))
    U = np.concatenate([np.stack([_SIDE_V[None, :] * Ls[:, None], np.broadcast_to(Y, (4, 6))], -1).reshape(-1, 2),
                        Q[_TRI]] + ([Q[_TRI]] if bottom else []))
    g.tris(P, N, col, m, U)


def frame_box(g, center, fwd, half, col, m, y0=0.0, y1=1.0):
    """Mismo uso que hd.frame_box, pero mucho mas rapido (se usa en los constructores de este archivo)."""
    fbox(g, center, fwd, half[0], half[1], y0, y1, col, m, bottom=y0 > 0.01)


def fcyl(g, x, z, y0, y1, r, col, m, seg=8, r1=None, cap=True):
    """Cilindro vertical rapido (postes, tinacos)."""
    r1 = r if r1 is None else r1
    a = np.linspace(0, TAU, seg + 1)
    ca, sa = np.cos(a), np.sin(a)
    i0, i1 = np.arange(seg), np.arange(1, seg + 1)
    k = np.stack([i0, i1, i1, i0, i1, i0], 1).ravel()
    top = np.array([0, 0, 1, 0, 1, 1] * seg)
    rr = np.where(top == 1, r1, r)
    P = np.column_stack([x + rr * ca[k], np.where(top == 1, y1, y0), z + rr * sa[k]])
    N = np.column_stack([ca[k], np.full(len(k), (r - r1) / max(1e-6, y1 - y0)), sa[k]])
    if cap:
        cp = np.stack([np.full(seg, -1), i0, i1], 1).ravel()
        Pc = np.column_stack([np.where(cp < 0, x, x + r1 * ca[np.maximum(cp, 0)]), np.full(len(cp), y1),
                              np.where(cp < 0, z, z + r1 * sa[np.maximum(cp, 0)])])
        P = np.concatenate([P, Pc])
        N = np.concatenate([N, np.tile([(0.0, 1.0, 0.0)], (len(cp), 1))])
    g.tris(P, N, col, m)


def sign_quad(g, c, F, hw, y0, y1, k):
    """Letrero del atlas centrado en c, legible desde el lado hacia donde apunta F."""
    rx, rz = F[1], -F[0]
    l, q = (c[0] - rx * hw, c[1] - rz * hw), (c[0] + rx * hw, c[1] + rz * hw)
    u0, u1, v0, v1 = bb_uv(k)
    quad(g, (l[0], y0, l[1]), (q[0], y0, q[1]), (q[0], y1, q[1]), (l[0], y1, l[1]), (1, 1, 1), mat(M_SIGN, 0.35),
         ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))


class Occupancy:
    """Celdas ocupadas del piso (para que los edificios no se encimen)."""

    def __init__(self, cell=4.0):
        self.cell, self.s = cell, set()

    def cells(self, c, f, hl, hw):
        f = v2(f)
        r = np.array([-f[1], f[0]])
        st = self.cell * 0.5
        us = np.linspace(-hl, hl, max(2, int(2 * hl / st) + 2))
        vs = np.linspace(-hw, hw, max(2, int(2 * hw / st) + 2))
        P = v2(c)[None, None] + us[:, None, None] * f + vs[None, :, None] * r
        return set(map(tuple, np.floor(P.reshape(-1, 2) / self.cell).astype(int).tolist()))

    def free(self, cells):
        return self.s.isdisjoint(cells)

    def add(self, cells):
        self.s |= cells

    def rect(self, c, f, hl, hw):
        self.add(self.cells(c, f, hl, hw))


class Zones:
    """Zonas reservadas (estadios, aeropuerto...) donde no se construye ciudad."""

    def __init__(self):
        self.circles, self.rects = [], []

    def circle(self, c, r):
        self.circles.append((float(c[0]), float(c[1]), float(r)))

    def rect(self, c, f, hl, hw):
        self.rects.append((v2(c), v2(f), hl, hw))

    def hit(self, P):
        P = np.asarray(P, float).reshape(-1, 2)
        out = np.zeros(len(P), bool)
        for x, z, r in self.circles:
            out |= np.hypot(P[:, 0] - x, P[:, 1] - z) < r
        for c, f, hl, hw in self.rects:
            d = P - c
            out |= (np.abs(d @ f) < hl) & (np.abs(d @ np.array([-f[1], f[0]])) < hw)
        return out


def beside_track(track, hint, clear):
    """Mueve 'hint' para que quede justo a 'clear' metros de la pista (al lado, sin tocarla)."""
    c = v2(hint)
    for _ in range(16):
        dd = np.hypot(track.pts[:, 0] - c[0], track.pts[:, 1] - c[1])
        i = int(np.argmin(dd))
        v = c - track.pts[i]
        ln = float(np.linalg.norm(v))
        if ln < 1e-6:
            v, ln = track.nrm[i], 1.0
        if abs(ln - clear) < 0.5:
            break
        c = track.pts[i] + v / ln * clear
    return c


def pick_spots(track, count, avoid, gap, max_curv, clear_d, offset=0.5):
    """Indices de la pista repartidos en la vuelta, en tramos rectos y con espacio a los lados."""
    n, out = track.n, []
    for k in range(count):
        target = int((k + offset) / count * n)
        half = max(1, n // (2 * count))
        for delta in sorted(range(-half, half), key=abs):
            i = (target + delta) % n
            if any(min(abs(i - j), n - abs(i - j)) < gp for j, gp in [(j, gap) for j in out] + avoid):
                continue
            win = [(i + q) % n for q in range(-9, 10)]
            if np.max(np.abs(track.curv[win])) > max_curv:
                continue
            side = np.array([track.pts[i] + track.nrm[i] * s * clear_d for s in (-1, 1)])
            if track.nearest_dist(side).min() < clear_d - 1.0:
                continue
            out.append(i)
            break
    return out


def street_lamp_hd(g, c, toward, lights):
    """Poste de luz de la calle (reemplaza a las torres del autodromo)."""
    steel = rgb(85, 92, 98)
    fx, fz = toward
    fcyl(g, c[0], c[1], 0, 10.2, 0.15, steel, mat(M_METAL, 0.45, 0.8), seg=8, r1=0.1)
    tip = (c[0] + fx * 3.6, c[1] + fz * 3.6)
    cylinder(g, (c[0], 10.0, c[1]), (tip[0], 10.5, tip[1]), 0.08, steel, mat(M_METAL, 0.45, 0.8), seg=6, caps=False)
    frame_box(g, tip, (fx, fz), (0.7, 0.28), rgb(60, 64, 70), mat(M_METAL, 0.5, 0.6), 10.25, 10.6)
    frame_box(g, tip, (fx, fz), (0.6, 0.22), (1.0, 0.78, 0.45), mat(M_LAMP, 0.5, 0, 7.0), 10.13, 10.26)
    lights.append((tip[0] + fx * 1.5, 9.0, tip[1] + fz * 1.5, 40.0, 1.0, 0.72, 0.42, 3.4))


def building_hd(g, crowd, rnd, c, f, o, hl, hdp, h, col, shop=True, roof_fans=False):
    """Edificio de colonia: fachada con ventanas, local con toldo y letrero, tinacos en la azotea."""
    frame_box(g, c, f, (hl, hdp), col, mat(M_WINDOWS, 0.6, 0.0), 0, h)
    F = c - o * (hdp + 0.06)
    if rnd.random() < 0.35:                                                    # pretil de la azotea
        frame_box(g, c, f, (hl, hdp), tuple(v * 0.85 for v in col), mat(M_CONC, 0.8), h, h + 0.8)
    if shop and hl > 3.2:
        w = hl - 0.8
        face_quad(g, F - f * w, F + f * w, 0.25, 2.8, (1.0, 0.84, 0.6), mat(M_LAMP, 0.5, 0, 2.2), -o)
        aw = rnd.choice(AWNINGS)
        a0, a1 = F - f * w, F + f * w
        b0, b1 = a0 - o * 1.6, a1 - o * 1.6
        quad_dir(g, (a0[0], 3.15, a0[1]), (a1[0], 3.15, a1[1]), (b1[0], 2.6, b1[1]), (b0[0], 2.6, b0[1]), aw,
                 mat(M_DEF, 0.75), (-o[0], 1.0, -o[1]))
        sign_quad(g, F - o * 0.05, -o, min(w, 4.5), 3.3, 4.2, rnd.choice(SHOP_SIGNS))
    for _ in range(rnd.choice([0, 1, 1, 2, 3])):                               # tinacos
        p = c + f * rnd.uniform(-hl + 1, hl - 1) + o * rnd.uniform(-hdp + 1, hdp - 1)
        fcyl(g, p[0], p[1], h, h + 1.4, 0.62, rgb(22, 22, 24), mat(M_DEF, 0.6), seg=10)
    if rnd.random() < 0.18:                                                    # cuarto de azotea
        p = c + o * rnd.uniform(-hdp * 0.4, hdp * 0.4)
        frame_box(g, p, f, (min(2.5, hl * 0.4), 2.0), (0.9, 0.88, 0.84), mat(M_CONC, 0.8), h, h + 2.6)
    if roof_fans:                                                              # el desmadre: gente en la azotea
        for _ in range(rnd.randint(3, 8)):
            p = F + f * rnd.uniform(-hl + 0.8, hl - 0.8) + o * rnd.uniform(0.6, 2.2)
            crowd.add(p[0], h, p[1], -o[0], -o[1], rnd)


def jacaranda_hd(g, x, z, s, rnd):
    fcyl(g, x, z, 0, 3.0 * s, 0.22 * s, rgb(80, 60, 45), mat(M_BARK, 0.9), seg=6, r1=0.16 * s, cap=False)
    for _ in range(3):
        a, r = rnd.uniform(0, TAU), rnd.uniform(0.5, 1.4) * s
        col = rgb(*[int(v * rnd.uniform(0.88, 1.1)) for v in (150, 100, 205)])
        sphere(g, (x + r * math.cos(a), (3.6 + rnd.uniform(0, 1.2)) * s, z + r * math.sin(a)),
               rnd.uniform(1.6, 2.3) * s, col, mat(M_LEAF, 0.8), seg=8, rings=5, scale=(1, 0.7, 1))


def puesto_hd(g, c, f, rnd):
    """Puesto callejero con lona de color y foco."""
    f = v2(f)
    r = np.array([-f[1], f[0]])
    frame_box(g, c, f, (1.1, 0.55), rgb(150, 150, 155), mat(M_METAL, 0.5, 0.6), 0, 0.95)
    for k in range(3):
        frame_box(g, c + f * (-0.7 + k * 0.7), f, (0.3, 0.3), rnd.choice(LONAS), mat(M_DEF, 0.7), 0.95, 1.25)
    for a in (-1.15, 1.15):
        for b in (-0.8, 0.8):
            p = c + f * a + r * b
            fcyl(g, p[0], p[1], 0, 2.3, 0.03, rgb(120, 120, 125), mat(M_METAL, 0.4, 0.8), seg=4, cap=False)
    P = [c - f * 1.35 - r * 1.0, c + f * 1.35 - r * 1.0, c + f * 1.35 + r * 1.0, c - f * 1.35 + r * 1.0]
    quad_dir(g, *[(p[0], 2.38 if k < 2 else 2.2, p[1]) for k, p in enumerate(P)], rnd.choice(LONAS), mat(M_DEF, 0.8),
             (0, 1, 0))
    frame_box(g, c, f, (0.12, 0.12), (1.0, 0.85, 0.6), mat(M_LAMP, 0.4, 0, 6.0), 2.0, 2.18)


def pesero_hd(g, c, f, rnd):
    """Microbus verde y blanco estacionado."""
    body = rnd.choice([rgb(40, 150, 80), rgb(40, 150, 80), rgb(230, 230, 230), rgb(200, 40, 40)])
    frame_box(g, c, f, (4.2, 1.15), body, mat(M_PAINT, 0.35), 0.45, 1.7)
    frame_box(g, c, f, (4.22, 1.17), rgb(20, 24, 30), mat(M_DEF, 0.15), 1.7, 2.45)
    frame_box(g, c, f, (4.2, 1.15), (0.92, 0.92, 0.92), mat(M_PAINT, 0.35), 2.45, 2.75)
    r = np.array([-f[1], f[0]])
    for a in (-2.8, 2.8):
        for b in (-1.1, 1.1):
            p = c + f * a + r * b
            cylinder(g, (p[0] - r[0] * 0.15, 0.45, p[1] - r[1] * 0.15), (p[0] + r[0] * 0.15, 0.45, p[1] + r[1] * 0.15),
                     0.45, rgb(15, 15, 15), mat(M_DEF, 0.9), seg=8)


def semaforo_hd(g, base_, arm_dir, arm_len, face, k=2):
    """Semaforo de brazo: el foco rojo se enciende con la cuenta regresiva y pasa a verde al arrancar."""
    green_gray = rgb(70, 82, 74)
    m_ = mat(M_METAL, 0.5, 0.6)
    x, z = base_
    fcyl(g, x, z, 0, 6.6, 0.12, green_gray, m_, seg=8)
    end = (x + arm_dir[0] * arm_len, z + arm_dir[1] * arm_len)
    cylinder(g, (x, 6.3, z), (end[0], 6.3, end[1]), 0.08, green_gray, m_, seg=6, caps=False)
    for e in (end, (x + arm_dir[0] * arm_len * 0.55, z + arm_dir[1] * arm_len * 0.55)):
        frame_box(g, e, face, (0.22, 0.26), rgb(18, 18, 20), mat(M_DEF, 0.5), 4.95, 6.25)
        c = (e[0] + face[0] * 0.24, e[1] + face[1] * 0.24)
        r = (-face[1] * 0.18, face[0] * 0.18)
        a, b = (c[0] - r[0], c[1] - r[1]), (c[0] + r[0], c[1] + r[1])
        face_quad(g, a, b, 5.78, 6.12, (1, 1, 1), mat(hd.M_STARTL, 0.3), face, ((k, 0), (k, 0), (k, 1), (k, 1)))
        face_quad(g, a, b, 5.39, 5.73, rgb(60, 40, 8), mat(M_DEF, 0.3), face)
        face_quad(g, a, b, 5.0, 5.34, (0.1, 1.0, 0.35), mat(M_EMIS, 0.4, 0, 5.0), face)
    # boton y semaforo peatonal en el poste
    frame_box(g, (x, z), face, (0.15, 0.18), rgb(18, 18, 20), mat(M_DEF, 0.5), 2.6, 3.3)
    c = (x + face[0] * 0.17, z + face[1] * 0.17)
    r = (-face[1] * 0.12, face[0] * 0.12)
    face_quad(g, (c[0] - r[0], c[1] - r[1]), (c[0] + r[0], c[1] + r[1]), 2.7, 3.2, (1.0, 0.95, 0.85),
              mat(M_EMIS, 0.4, 0, 2.5), face)


def ped_bridge_hd(g, crowd, rnd, track, i, k_ad, lights, occ):
    """Puente peatonal CDMX: escaleras a los lados, publicidad, techito y gente viendo la carrera."""
    p, nr, t = track.pts[i], track.nrm[i], track.tan[i]
    half = LIMIT + 6.0
    deck_y = 6.4
    steel, green = rgb(70, 74, 80), rgb(60, 110, 90)
    sm = mat(M_METAL, 0.45, 0.7)
    a = p - nr * half
    obox(g, (a[0] - t[0] * 1.5, deck_y, a[1] - t[1] * 1.5), (nr[0] * 2 * half, 0, nr[1] * 2 * half), (0, 0.45, 0),
         (t[0] * 3, 0, t[1] * 3), green, mat(M_METAL, 0.5, 0.4))
    u0, u1, v0, v1 = hd.ad_uv(k_ad)
    for sgn in (-1, 1):                                                        # barandal con publicidad
        cc = p + t * 1.52 * sgn
        l, r_ = cc - nr * half, cc + nr * half
        face_quad(g, l, r_, deck_y + 0.45, deck_y + 1.5, (1, 1, 1), mat(M_SIGN, 0.35), t * sgn,
                  ((u0, v1), (u1, v1), (u1, v0), (u0, v0)))
        for d in np.arange(-half, half + 0.1, 5.0):                           # columnas del techito
            q = p + nr * d + t * 1.45 * sgn
            fcyl(g, q[0], q[1], deck_y + 0.45, deck_y + 3.0, 0.06, steel, sm, seg=5, cap=False)
    obox(g, (a[0] - t[0] * 1.8, deck_y + 3.0, a[1] - t[1] * 1.8), (nr[0] * 2 * half, 0, nr[1] * 2 * half),
         (0, 0.12, 0), (t[0] * 3.6, 0, t[1] * 3.6), rgb(200, 205, 210), mat(M_METAL, 0.4, 0.3))
    for d in (-half + 0.6, -(LIMIT + 1.4), LIMIT + 1.4, half - 0.6):            # pilares
        q = p + nr * d
        fcyl(g, q[0], q[1], 0, deck_y, 0.38, rgb(150, 150, 148), mat(M_CONC, 0.8), seg=10, cap=False)
    for side in (-1, 1):                                                       # escaleras hacia afuera
        o = nr * side
        steps = 16
        for k in range(steps):
            d0 = half + k * 0.8
            top = deck_y * (1 - k / steps) + 0.45
            frame_box(g, p + o * (d0 + 0.4), o, (0.4, 1.2), rgb(150, 150, 148), mat(M_CONC, 0.8), top - 0.45, top)
        for sgn in (-1, 1):
            q0, q1 = p + o * half + t * 1.25 * sgn, p + o * (half + steps * 0.8) + t * 1.25 * sgn
            cylinder(g, (q0[0], deck_y + 1.4, q0[1]), (q1[0], 1.4, q1[1]), 0.04, steel, sm, seg=5, caps=False)
        occ.rect(p + o * (half + steps * 0.4), o, steps * 0.4 + 1.0, 2.5)
    for d in (-half * 0.5, half * 0.5):                                        # lamparas bajo el techito
        q = p + nr * d
        frame_box(g, q, t, (0.8, 0.2), (1.0, 0.9, 0.75), mat(M_LAMP, 0.5, 0, 7.0), deck_y + 2.8, deck_y + 2.95)
    lights.append((float(p[0]), deck_y + 2.0, float(p[1]), 30.0, 1.0, 0.88, 0.7, 3.0))
    for _ in range(rnd.randint(14, 26)):                                       # aficionados en el puente
        d = rnd.uniform(-half + 1.0, half - 1.0)
        sgn = rnd.choice((-1, 1))
        q = p + nr * d + t * sgn * rnd.uniform(0.4, 1.1)
        crowd.add(q[0], deck_y + 0.45, q[1], t[0] * sgn, t[1] * sgn, rnd)


def baseball_field_hd(g, sc, r0, home_dir):
    """Diamante de beisbol en el campo del estadio."""
    hd_ = v2(home_dir)
    flat_rect(g, sc, hd_, r0 - 1, r0 - 1, 0.035, rgb(60, 130, 50), mat(M_GRASS, 0.9))
    home = sc + hd_ * (r0 * 0.62)
    diag = -hd_
    dia = norm(np.array([diag[0] + diag[1], diag[1] - diag[0]]))[:2].astype(float)
    flat_rect(g, home + diag * 19.4, dia, 19.0, 19.0, 0.045, rgb(165, 110, 70), mat(M_GRAVEL, 0.95))
    flat_rect(g, home + diag * 19.4, dia, 13.0, 13.0, 0.055, rgb(70, 140, 55), mat(M_GRASS, 0.9))
    for k, q in enumerate([home, home + diag * 27.4]):
        flat_rect(g, q, dia, 2.6 if k else 1.8, 2.6 if k else 1.8, 0.06, rgb(165, 110, 70), mat(M_GRAVEL, 0.95))
    r_ = np.array([-diag[1], diag[0]])
    for q in (home + diag * 19.4 + r_ * 19.4, home + diag * 38.8, home + diag * 19.4 - r_ * 19.4, home):
        flat_rect(g, q, dia, 0.38, 0.38, 0.09, (0.95, 0.95, 0.95), mat(M_LINE, 0.6))
    for sgn in (-1, 1):                                                        # lineas de foul
        dd = norm(np.array([diag[0] + sgn * r_[0], diag[1] + sgn * r_[1]]))[:2].astype(float)
        flat_rect(g, home + dd * (r0 * 0.7), dd, r0 * 0.7, 0.07, 0.07, (0.95, 0.95, 0.95), mat(M_LINE, 0.6))
    return home


def remap_sign(geo, k_from, k_to):
    """Cambia en una geometria el letrero k_from del atlas por k_to."""
    a, b = bb_uv(k_from), bb_uv(k_to)
    for p in geo.parts:
        m = (np.abs(p[:, 9] - M_SIGN) < 0.5) & (p[:, 13] > a[0] - 1e-4) & (p[:, 13] < a[1] + 1e-4) & \
            (p[:, 14] > a[2] - 1e-4) & (p[:, 14] < a[3] + 1e-4)
        p[m, 13] += b[0] - a[0]
        p[m, 14] += b[2] - a[2]


def palacio_hd(g, c, r, lights):
    """Domo de cobre facetado (Palacio de los Deportes)."""
    x, z = c
    frame = [(r, 0.0), (r, 7.0)]
    revolve(g, frame, (x, 0, z), (0, 1, 0), rgb(160, 158, 152), mat(M_CONC, 0.8), seg=24)
    dome = [(r, 7.0), (r * 0.95, 15.0), (r * 0.83, 24.0), (r * 0.64, 32.0), (r * 0.4, 38.0), (r * 0.12, 41.0),
            (0.01, 41.5)]
    revolve(g, dome, (x, 0, z), (0, 1, 0), rgb(176, 98, 58), mat(M_METAL, 0.42, 0.85), seg=24)
    for k in range(8):
        a = k / 8 * TAU
        lights.append((x + (r + 6) * math.cos(a), 8.0, z + (r + 6) * math.sin(a), 35.0, 1.0, 0.8, 0.55, 2.5))


def airliner_geo(tail):
    """Avion comercial (nariz hacia +x, eje del fuselaje en y=0)."""
    g = Geo()
    white, grey = rgb(238, 240, 244), rgb(175, 180, 190)
    m = mat(M_METAL, 0.3, 0.25)
    prof = [(0.25, -22.0), (1.0, -19.0), (1.8, -15.0), (2.1, -11.0), (2.1, 14.0), (1.9, 17.5), (1.3, 20.3),
            (0.5, 21.7), (0.02, 22.1)]
    revolve(g, prof, (0, 0, 0), (1, 0, 0), white, m, seg=16)
    for sz in (-1, 1):
        obox(g, (-14, 0.45, sz * 2.1 - 0.03), (29, 0, 0), (0, 0.32, 0), (0, 0, 0.06), rgb(25, 30, 40),
             mat(M_DEF, 0.2))
        z0 = sz * 2.0
        top = [(4.5, -0.55, z0), (-6.0, -0.55, z0), (-13.0, -0.1, sz * 17.0), (-10.2, -0.1, sz * 17.0)]
        bot = [(x, y - 0.4, zz) for x, y, zz in top]
        quad_dir(g, *top, white, m, (0, 1, 0))
        quad_dir(g, *bot, grey, m, (0, -1, 0))
        quad_dir(g, top[0], top[3], bot[3], bot[0], white, m, (1, 0, 0))
        quad_dir(g, top[1], top[2], bot[2], bot[1], white, m, (-1, 0, 0))
        hs = [(-16.0, 0.7, sz * 1.0), (-21.0, 0.7, sz * 1.0), (-23.6, 0.9, sz * 7.0), (-21.8, 0.9, sz * 7.0)]
        quad_dir(g, *hs, white, m, (0, 1, 0))
        quad_dir(g, *[(x, y - 0.2, zz) for x, y, zz in hs], grey, m, (0, -1, 0))
        cylinder(g, (-3.5, -1.9, sz * 6.4), (2.5, -1.9, sz * 6.4), 1.15, grey, mat(M_METAL, 0.3, 0.7), seg=12)
        cylinder(g, (2.4, -1.9, sz * 6.4), (2.55, -1.9, sz * 6.4), 0.95, rgb(20, 20, 24), mat(M_DEF, 0.4), seg=12)
        frame_box(g, (-11.6, sz * 17.0), (1, 0), (0.25, 0.15), (0.1, 1.0, 0.3) if sz > 0 else (1.0, 0.08, 0.05),
                  mat(M_EMIS, 0.3, 0, 8.0), -0.2, 0.15)
        frame_box(g, (3.8, sz * 3.0), (1, 0), (0.25, 0.25), (1, 1, 1), mat(hd.M_HEAD, 0.2), -0.9, -0.55)
        for wx in (-1.5,):
            cylinder(g, (wx, -2.2, sz * 2.4), (wx, -3.3, sz * 2.4), 0.12, grey, mat(M_METAL, 0.4, 0.8), seg=6)
            cylinder(g, (wx - 0.6, -3.0, sz * 2.4), (wx + 0.6, -3.0, sz * 2.4), 0.45, rgb(15, 15, 15),
                     mat(M_DEF, 0.9), seg=10)
    cylinder(g, (16.5, -1.6, 0), (16.5, -3.0, 0), 0.1, grey, mat(M_METAL, 0.4, 0.8), seg=6)
    cylinder(g, (16.5, -3.0, -0.4), (16.5, -3.0, 0.4), 0.38, rgb(15, 15, 15), mat(M_DEF, 0.9), seg=10)
    fin = [(-15.0, 1.6, 0.0), (-21.5, 1.6, 0.0), (-24.2, 10.5, 0.0), (-21.2, 10.5, 0.0)]
    for sz in (-1, 1):
        quad_dir(g, *[(x, y, sz * 0.18) for x, y, _ in fin], tail, mat(M_METAL, 0.35, 0.2), (0, 0, sz))
    quad_dir(g, (-15.0, 1.6, -0.18), (-15.0, 1.6, 0.18), (-21.2, 10.5, 0.18), (-21.2, 10.5, -0.18), tail,
             mat(M_METAL, 0.35, 0.2), (1, 0.5, 0))
    frame_box(g, (-24.0, 0.0), (1, 0), (0.2, 0.2), (1, 1, 1), mat(M_EMIS, 0.3, 0, 10.0), 10.4, 10.8)
    frame_box(g, (18.9, 0.0), (1, 0), (0.7, 0.85), rgb(20, 26, 36), mat(M_DEF, 0.15), 0.55, 1.15)
    return g


PLANE_Y = 3.35                     # altura del eje del fuselaje con el tren en el piso


def flight_pose(fl, t):
    """Posicion de un avion animado: (x, y, z, angulo, cabeceo) o None si esta 'escondido'."""
    u = (t + fl["ph"]) % fl["T"]
    d = fl["d"]
    ang = math.atan2(d[1], d[0])
    if fl["kind"] == "land":
        app, v = 2300.0, 82.0
        ta = app / v
        if u < ta:
            s = -app + v * u
            return (fl["o"][0] + d[0] * s, PLANE_Y - s * 0.0524, fl["o"][1] + d[1] * s, ang, 0.05)
        w = u - ta
        if w < 22.0:
            s = v * w - 0.5 * 3.4 * w * w
            return (fl["o"][0] + d[0] * s, PLANE_Y, fl["o"][1] + d[1] * s, ang, 0.05 * max(0.0, 1 - w / 3))
        return None
    if u < 4.0:
        return (fl["o"][0], PLANE_Y, fl["o"][1], ang, 0.0)
    w = u - 4.0
    if w < 30.0:
        s = 0.5 * 2.7 * w * w
        return (fl["o"][0] + d[0] * s, PLANE_Y, fl["o"][1] + d[1] * s, ang, 0.16 * max(0.0, (w - 28.0) / 2.0))
    c = w - 30.0
    if c < 24.0:
        s = 1215.0 + 81.0 * c
        return (fl["o"][0] + d[0] * s, PLANE_Y + 12.8 * c, fl["o"][1] + d[1] * s, ang, 0.16)
    return None


def airport_hd(g, rnd, A1, d, lights, zones, track, openz=None):
    """Aeropuerto (AICM): dos pistas con luces, rodaje, terminal, torre, hangares y aviones."""
    d = v2(d) / np.linalg.norm(d)
    p = np.array([d[1], -d[0]])                     # perpendicular (hacia el noroeste si d va al noreste)
    if p[1] > 0:
        p = -p

    def P(s, q):
        return A1 + d * s + p * q
    zones.rect(P(1370, 350), d, 1440, 460)
    if openz is not None:                           # sin edificios entre la pista y las pistas del aeropuerto
        openz.rect(P(-230, 150), d, 290, 330)
    flat_rect(g, P(1370, 350), d, 1430, 450, 0.02, rgb(95, 130, 60), mat(M_GRASS, 0.95))
    asph = rgb(46, 46, 50)
    for q in (0.0, 330.0):
        flat_rect(g, P(1300, q), d, 1300, 23, 0.05, asph, mat(M_CONC, 0.75))
        for s in np.arange(60, 2560, 50):                                     # linea central
            flat_rect(g, P(s, q), d, 15, 0.45, 0.06, (0.92, 0.92, 0.92), mat(M_LINE, 0.55))
        for end in (8.0, 2592.0):                                             # umbrales (piano)
            for k in range(-7, 8):
                if k == 0:
                    continue
                flat_rect(g, P(end + (12 if end < 100 else -12), q + k * 2.8), d, 12, 0.7, 0.06, (0.92, 0.92, 0.92),
                          mat(M_LINE, 0.55))
        for s in np.arange(0, 2601, 60):                                      # luces de orilla
            for sgn in (-1, 1):
                c = P(s, q + sgn * 24.5)
                frame_box(g, c, d, (0.3, 0.3), (1.0, 0.86, 0.55), mat(M_EMIS, 0.4, 0, 4.0), 0.0, 0.45)
        for s in (-20, -45, -70, -90):                                        # luces de aproximacion
            for k in range(-3, 4):
                c = P(s, q + k * 3.0)
                if track.nearest_dist(c[None])[0] > LIMIT + 8:
                    frame_box(g, c, d, (0.35, 0.35), (1.0, 1.0, 0.95), mat(M_EMIS, 0.4, 0, 6.0), 0.0, 0.6)
    flat_rect(g, P(1300, 165), d, 1250, 11, 0.045, asph, mat(M_CONC, 0.75))      # rodaje
    for s in (300, 1300, 2300):
        flat_rect(g, P(s, 165), p, 165, 11, 0.046, asph, mat(M_CONC, 0.75))
        flat_rect(g, P(s, 450), p, 120, 14, 0.046, asph, mat(M_CONC, 0.75))
    flat_rect(g, P(1300, 560), d, 520, 70, 0.047, rgb(120, 120, 122), mat(M_CONC, 0.8))  # plataforma
    tc = P(1300, 680)                                                          # terminal
    frame_box(g, tc, d, (480, 38), rgb(90, 110, 130), mat(hd.M_FACADE, 0.12, 0.6), 0, 16)
    frame_box(g, tc, d, (490, 44), rgb(225, 228, 234), mat(M_METAL, 0.35, 0.3), 16, 18)
    for k in range(9):                                                         # salas y aviones en las puertas
        s = 1300 - 400 + k * 100
        gate = P(s, 625)
        frame_box(g, P(s, 610), p, (16, 2.0), rgb(200, 202, 208), mat(M_METAL, 0.4, 0.4), 4.0, 7.0)
        tail = rnd.choice([rgb(10, 40, 110), rgb(200, 20, 40), rgb(0, 140, 90), rgb(240, 150, 20)])
        pl = airliner_geo(tail)
        nose = p
        g.merge(transform_geo(pl, -math.atan2(nose[1], nose[0]), (gate[0] - nose[0] * 24, PLANE_Y,
                                                                    gate[1] - nose[1] * 24)))
    tw = P(700, 520)                                                           # torre de control
    cylinder(g, (tw[0], 0, tw[1]), (tw[0], 62, tw[1]), 4.2, rgb(215, 212, 205), mat(M_CONC, 0.7), seg=16, r1=3.2)
    cylinder(g, (tw[0], 62, tw[1]), (tw[0], 66, tw[1]), 6.5, rgb(225, 228, 234), mat(M_METAL, 0.4, 0.3), seg=16,
             r1=8.5)
    cylinder(g, (tw[0], 66, tw[1]), (tw[0], 71, tw[1]), 8.5, rgb(40, 70, 90), mat(hd.M_FACADE, 0.1, 0.6), seg=16)
    cylinder(g, (tw[0], 71, tw[1]), (tw[0], 73, tw[1]), 9.0, rgb(225, 228, 234), mat(M_METAL, 0.4, 0.3), seg=16,
             r1=3.0)
    frame_box(g, tw, d, (0.6, 0.6), (1.0, 0.1, 0.05), mat(M_EMIS, 0.3, 0, 10.0), 73, 76)
    lights.append((float(tw[0]), 60.0, float(tw[1]), 80.0, 1.0, 0.9, 0.8, 3.0))
    for k in range(3):                                                         # hangares
        hc = P(2150 + k * 120, 600)
        frame_box(g, hc, d, (50, 40), rgb(205, 208, 212), mat(M_METAL, 0.45, 0.5), 0, 24)
        frame_box(g, hc, d, (51, 41), rgb(170, 30, 30) if k == 1 else rgb(150, 155, 160), mat(M_METAL, 0.4, 0.5), 24, 26)
    for k in range(6):                                                         # reflectores de la plataforma
        c = P(950 + k * 140, 520)
        fcyl(g, c[0], c[1], 0, 26, 0.35, rgb(140, 145, 150), mat(M_METAL, 0.4, 0.9), seg=8)
        frame_box(g, c, d, (2.0, 0.5), (1.0, 0.95, 0.85), mat(M_LAMP, 0.4, 0, 7.0), 26, 27)
        lights.append((float(c[0]), 22.0, float(c[1]), 70.0, 1.0, 0.92, 0.8, 3.0))
    flights = [dict(kind="land", o=P(330, 0.0), d=d, T=52.0, ph=0.0),
               dict(kind="takeoff", o=P(2520, 330.0), d=-d, T=60.0, ph=21.0)]
    return flights, airliner_geo(rgb(10, 40, 110))


# ---------------------------------------------------------------- fiesta mundialista (Chapultepec / Reforma)
MEXICO_SHIRTS = [rgb(0, 104, 71), rgb(0, 104, 71), rgb(0, 120, 80), rgb(245, 245, 245), rgb(206, 17, 38),
                 rgb(20, 20, 24), rgb(0, 90, 60)]
FLAG_COLS = [(rgb(0, 104, 71), rgb(245, 245, 245), rgb(206, 17, 38)),      # Mexico
             (rgb(0, 150, 60), rgb(250, 220, 0), rgb(0, 60, 160)),         # Brasil
             (rgb(120, 180, 230), rgb(245, 245, 245), rgb(120, 180, 230)),  # Argentina
             (rgb(200, 20, 40), rgb(245, 245, 245), rgb(200, 20, 40)),      # Canada
             (rgb(20, 40, 120), rgb(245, 245, 245), rgb(200, 20, 40)),      # Estados Unidos / Francia
             (rgb(200, 20, 30), rgb(250, 200, 0), rgb(200, 20, 30)),        # Espana
             (rgb(20, 20, 20), rgb(220, 20, 20), rgb(250, 200, 0)),         # Alemania
             (rgb(245, 245, 245), rgb(220, 20, 40), rgb(245, 245, 245))]    # Japon
_PHI = (1 + 5 ** 0.5) / 2
ICO = np.array([(-1, _PHI, 0), (1, _PHI, 0), (-1, -_PHI, 0), (1, -_PHI, 0), (0, -1, _PHI), (0, 1, _PHI),
                (0, -1, -_PHI), (0, 1, -_PHI), (_PHI, 0, -1), (_PHI, 0, 1), (-_PHI, 0, -1), (-_PHI, 0, 1)], float)
ICO /= np.linalg.norm(ICO, axis=1, keepdims=True)


class PaletteCrowd(Crowd):
    """Publico con playeras del tema del mapa (por ejemplo, la verde de Mexico)."""

    def __init__(self, palette):
        super().__init__()
        self.pal = palette

    def add(self, x, y, z, face_x, face_z, rnd, shirt=None, typ=None, concert=False, palette=None):
        if self.pal and shirt is None and (palette is None or concert) and rnd.random() < 0.75:
            palette = self.pal
        super().add(x, y, z, face_x, face_z, rnd, shirt, typ, concert, palette)


def balon_hd(g, x, y, z, r):
    """Balon de futbol gigante (parches negros en los 12 vertices de un icosaedro)."""
    c = np.array([x, y, z], float)

    def colfn(P):
        d = P - c
        d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
        k = (d @ ICO.T).max(-1)
        return np.where(k[..., None] > 0.9, np.array([0.03, 0.03, 0.035]), np.array([0.95, 0.95, 0.95])).astype(F4)
    sphere(g, (x, y, z), r, (1, 1, 1), mat(M_DEF, 0.35), seg=28, rings=16, colfn=colfn)


def trofeo_hd(g, x, z, s, lights=None):
    """Copa dorada gigante."""
    fbox(g, (x, z), (1, 0), 2.2 * s, 2.2 * s, 0, 1.8 * s, rgb(20, 110, 70), mat(M_DEF, 0.3, 0.1))
    prof = [(1.6, 1.8), (1.2, 2.6), (0.75, 4.2), (0.9, 5.4), (1.6, 7.2), (2.2, 8.6), (2.4, 10.2), (1.9, 11.6),
            (0.9, 12.3), (0.05, 12.6)]
    revolve(g, [(a * s, b * s) for a, b in prof], (x, 0, z), (0, 1, 0), rgb(235, 185, 60), mat(hd.M_GOLD, 0.22, 1.0),
            seg=20)
    if lights is not None:
        lights.append((float(x), 6.0 * s, float(z), 30.0 * s, 1.0, 0.85, 0.5, 3.0))


def banderines_hd(g, pa, pb, h=8.0, rows=3, k0=0):
    """Banderines de los paises sobre la calle (se mueven con el viento como el papel picado)."""
    dx, dz = pb[0] - pa[0], pb[1] - pa[1]
    ln = math.hypot(dx, dz)
    ux, uz = dx / ln, dz / ln
    nn = int(ln / 0.95)
    for r in range(rows):
        yb = h - r * 1.0
        for i in range(nn):
            t = (i + 0.5) / nn
            px, pz = pa[0] + dx * t, pa[1] + dz * t
            y = yb - 1.4 * 4 * t * (1 - t)
            col = FLAG_COLS[(i // 3 + r + k0) % len(FLAG_COLS)][i % 3]
            m_ = (float(hd.M_PAPEL), 0.7, 0.0, i * 0.7 + r)
            quad(g, (px - ux * 0.36, y, pz - uz * 0.36), (px + ux * 0.36, y, pz + uz * 0.36), (px, y - 0.85, pz),
                 (px, y - 0.85, pz), col, m_)


def turbo_pad_hd(g, track, i):
    """Rampa de turbo: flechas que brillan sobre el asfalto."""
    p, t, nr = track.pts[i], track.tan[i], track.nrm[i]
    for k in range(4):
        a = k * 3.2
        for sgn in (-1, 1):
            tip = p + t * (a + 2.4)
            end = p + t * a + nr * sgn * (HW - 1.2)
            w = t * 0.9
            P = [(end[0], 0.085, end[1]), (tip[0], 0.085, tip[1]), (tip[0] + w[0], 0.085, tip[1] + w[1]),
                 (end[0] + w[0], 0.085, end[1] + w[1])]
            quad_dir(g, *P, (0.15, 1.0, 0.55), mat(M_EMIS, 0.4, 0, 3.5), (0, 1, 0))


def lift_crossings(geo, track):
    """En un cruce las dos calles estan a la misma altura: la segunda se sube 3 cm para que no parpadee."""
    keep = np.array([1, 4, 5, 6, M_LINE], float)          # asfalto, pianos, grava, orillas, lineas
    for i, j, c in track.cross_pts:
        sB = track.cum[j]
        for p in geo.parts:
            near = np.hypot(p[:, 0] - c[0], p[:, 2] - c[1]) < 80
            if not near.any():
                continue
            ds = np.abs((p[:, 13] % track.L) - sB)
            ds = np.minimum(ds, track.L - ds)
            m = near & (p[:, 1] < 0.3) & np.isin(np.round(p[:, 9]), keep) & (ds < 110)
            p[m, 1] += 0.03


def monumento_rev_hd(g, track, i, lights):
    """Monumento a la Revolucion: la pista pasa por debajo de sus cuatro arcos."""
    p, t, nr = track.pts[i], track.tan[i], track.nrm[i]
    H = LIMIT + 10.0
    stone, dark = rgb(132, 124, 112), rgb(96, 90, 82)
    ms = mat(M_CONC, 0.8)
    for a in (-1, 1):
        for b in (-1, 1):
            c = p + t * a * (H - 4) + nr * b * (H - 4)
            fbox(g, c, t, 5.0, 5.0, 0, 3.0, dark, ms)
            fbox(g, c, t, 4.0, 4.0, 3.0, 26.0, stone, ms)
            fbox(g, c, t, 1.2, 1.2, 32.0, 37.0, rgb(60, 60, 58), mat(M_METAL, 0.4, 0.6))  # esculturas
    for f, o in ((t, nr), (nr, t)):
        for sgn in (-1, 1):
            c = p + o * sgn * (H - 4)
            fbox(g, c, f, H, 4.0, 24.0, 32.0, stone, ms, bottom=True)
            for k in range(14):                                                 # intrados del arco
                u = -1 + (k + 0.5) / 14 * 2
                yb = 13.0 + 11.0 * math.sqrt(max(0.0, 1 - u * u))
                fbox(g, c + f * u * (H - 8), f, (H - 8) / 14 + 0.05, 3.6, yb, 24.0, stone, ms, bottom=True)
    fbox(g, p, t, H, H, 32.0, 36.0, dark, ms, bottom=True)
    fcyl(g, p[0], p[1], 36.0, 44.0, 15.0, stone, ms, seg=24, cap=False)
    revolve(g, [(15.0, 44.0), (14.0, 50.0), (11.5, 55.0), (7.5, 59.0), (2.5, 61.0), (0.3, 61.3)], (p[0], 0, p[1]),
            (0, 1, 0), rgb(70, 112, 96), mat(M_METAL, 0.45, 0.6), seg=24)
    fcyl(g, p[0], p[1], 61.0, 66.0, 1.8, rgb(235, 185, 60), mat(hd.M_GOLD, 0.25, 1.0), seg=10)
    sphere(g, (p[0], 67.0, p[1]), 1.3, rgb(235, 185, 60), mat(hd.M_GOLD, 0.25, 1.0), seg=10, rings=6)
    for a in (-1, 1):
        for b in (-1, 1):
            c = p + t * a * (H + 3) + nr * b * (H + 3)
            lights.append((float(c[0]), 10.0, float(c[1]), 60.0, 1.0, 0.8, 0.5, 4.0))
    lights.append((float(p[0]), 20.0, float(p[1]), 45.0, 1.0, 0.85, 0.6, 3.0))


def basilica_hd(g, crowd, rnd, c, f, lights):
    """Basilica de Guadalupe: atrio, la basilica nueva (techo de cobre) y la antigua (amarilla)."""
    f = v2(f)
    r = np.array([-f[1], f[0]])
    flat_rect(g, c, f, 95, 70, 0.04, rgb(196, 186, 168), mat(M_CONC, 0.8))
    nb = c - f * 30 + r * 25
    fcyl(g, nb[0], nb[1], 0, 10, 48, rgb(215, 205, 185), mat(M_CONC, 0.7), seg=32, cap=False)
    revolve(g, [(50, 10), (42, 17), (30, 26), (16, 35), (4, 43), (0.5, 44.5)], (nb[0], 0, nb[1]), (0, 1, 0),
            rgb(70, 135, 105), mat(M_METAL, 0.45, 0.6), seg=32)
    fbox(g, nb + f * 52, f, 2.5, 2.5, 0, 46, rgb(225, 220, 210), mat(M_CONC, 0.7))           # campanario
    fbox(g, nb + f * 52, f, 0.4, 3.0, 40, 41, rgb(235, 185, 60), mat(hd.M_GOLD, 0.3, 1.0))
    ob = c - f * 30 - r * 45
    fbox(g, ob, f, 24, 16, 0, 22, rgb(205, 170, 120), mat(M_CONC, 0.75))
    for a in (-1, 1):
        for b in (-1, 1):
            q = ob + f * a * 22 + r * b * 14
            fbox(g, q, f, 3.2, 3.2, 0, 34, rgb(210, 175, 125), mat(M_CONC, 0.75))
            sphere(g, (q[0], 36.0, q[1]), 3.0, rgb(225, 185, 70), mat(M_METAL, 0.4, 0.5), seg=12, rings=6)
    revolve(g, [(10, 22), (9.5, 28), (7.5, 33), (4, 36), (0.4, 37)], (ob[0], 0, ob[1]), (0, 1, 0), rgb(225, 185, 70),
            mat(M_METAL, 0.4, 0.5), seg=20)
    for _ in range(450):                                                       # peregrinos y aficionados
        q = c + f * rnd.uniform(-10, 90) + r * rnd.uniform(-90, 90)
        crowd.add(q[0], 0.04, q[1], rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd)
    for k in range(4):
        q = c + f * 40 + r * (k - 1.5) * 40
        lights.append((float(q[0]), 12.0, float(q[1]), 60.0, 1.0, 0.85, 0.6, 3.0))


def castillo_hd(g, rnd, c, r, lights):
    """Cerro de Chapultepec con el castillo y una bandera monumental."""
    h = 40.0
    mountain_hd(g, c[0], c[1], r, h, rnd, seg=32, rings=10)
    wall = rgb(228, 218, 198)
    fbox(g, c, (1, 0), 40, 20, h * 0.55, h + 12, wall, mat(M_WINDOWS, 0.5, 0.05), bottom=True)
    fbox(g, c, (1, 0), 41, 21, h + 12, h + 13.2, rgb(200, 190, 170), mat(M_CONC, 0.8))
    fbox(g, c + (34, 0), (1, 0), 10, 24, h * 0.6, h + 8, wall, mat(M_WINDOWS, 0.5, 0.05), bottom=True)
    t0 = c + (-18, -8)
    fbox(g, t0, (1, 0), 5.5, 5.5, h, h + 26, wall, mat(M_WINDOWS, 0.5, 0.05))        # caballero alto
    fbox(g, t0, (1, 0), 6.5, 6.5, h + 26, h + 28, rgb(200, 190, 170), mat(M_CONC, 0.8))
    for k in range(10):                                                        # arboles del cerro
        a = k / 10 * TAU
        q = c + np.array([math.cos(a), math.sin(a)]) * r * 0.62
        tree_hd(g, q[0], q[1], rnd.uniform(1.0, 1.5), rnd)
    flagpole_hd(g, c[0] + 18, c[1] + 26, height=h + 45)
    for a in (0.5, 2.6, 4.5):
        q = c + np.array([math.cos(a), math.sin(a)]) * 60
        lights.append((float(q[0]), h + 4, float(q[1]), 90.0, 1.0, 0.85, 0.6, 4.0))


def museo_hd(g, c, f, lights):
    """Museo Nacional de Antropologia con su 'paraguas'."""
    f = v2(f)
    stone, dark = rgb(205, 196, 178), rgb(92, 84, 74)
    fbox(g, c, f, 80, 48, 0, 13, stone, mat(M_CONC, 0.75))
    fbox(g, c, f, 80.5, 48.5, 9, 13, dark, mat(M_CONC, 0.7))
    pc = c - f * 10
    fcyl(g, pc[0], pc[1], 13, 23, 2.4, rgb(120, 110, 95), mat(M_CONC, 0.7), seg=14, cap=False)
    fbox(g, pc, f, 42, 28, 23, 25, rgb(170, 160, 145), mat(M_CONC, 0.7), bottom=True)
    tl = c - f * 56
    fbox(g, tl, f, 1.6, 1.6, 0, 7.5, rgb(110, 108, 104), mat(M_CONC, 0.9))               # Tlaloc
    lights.append((float(tl[0]), 6.0, float(tl[1]), 40.0, 1.0, 0.85, 0.6, 3.0))


def fanfest_hd(g, gc, crowd, rnd, track, sc, r0, md, info, lights):
    """Fan Fest mundialista: plaza, pantallas, escenario (de noche), copa y balon gigantes."""
    ang = np.linspace(0, TAU, 65)
    R = r0 + 8
    for j in range(64):
        quad(g, (sc[0], 0.03, sc[1]), (sc[0], 0.03, sc[1]),
             (sc[0] + R * math.cos(ang[j + 1]), 0.03, sc[1] + R * math.sin(ang[j + 1])),
             (sc[0] + R * math.cos(ang[j]), 0.03, sc[1] + R * math.sin(ang[j])), rgb(160, 155, 146), mat(M_CONC, 0.8))
    i_n = int(np.argmin(np.hypot(track.pts[:, 0] - sc[0], track.pts[:, 1] - sc[1])))
    tt = track.pts[i_n] - sc
    tt = tt / np.linalg.norm(tt)
    rr = np.array([-tt[1], tt[0]])
    for k in range(48):                                                        # vallas
        a = k / 48 * TAU
        q = sc + np.array([math.cos(a), math.sin(a)]) * (r0 + 6)
        fbox(g, q, (-math.sin(a), math.cos(a)), 3.4, 0.08, 0, 1.2, rgb(150, 155, 160), mat(M_METAL, 0.4, 0.7))
    fe = sc + tt * (r0 + 7)                                                    # portico de entrada
    for sgn in (-1, 1):
        q = fe + rr * sgn * 16
        fbox(g, q, tt, 1.0, 1.0, 0, 18, rgb(30, 30, 34), mat(M_METAL, 0.45, 0.7))
    fbox(g, fe, rr, 17, 1.0, 16, 22, rgb(30, 30, 34), mat(M_METAL, 0.45, 0.7))
    sign_quad(g, fe + tt * 1.05, tt, 15, 16.3, 21.7, md.band_sign)
    sign_quad(g, fe - tt * 1.05, -tt, 15, 16.3, 21.7, 1)
    for sgn in (-1, 1):                                                        # pantallas gigantes
        q = sc + rr * sgn * (r0 - 6) + tt * 10
        fbox(g, q, rr, 7, 0.6, 0, 16, rgb(25, 25, 28), mat(M_DEF, 0.5))
        face_quad(g, q + tt * 0.65 - rr * 6.4, q + tt * 0.65 + rr * 6.4, 5.0, 15.4, (1, 1, 1),
                  mat(M_LED, 0.3, 0, 2.5), tt, ((0, 0), (1, 0), (1, 1), (0, 1)))
    tq, bq = sc + tt * (r0 - 14) - rr * 18, sc + tt * (r0 - 14) + rr * 18
    trofeo_hd(g, tq[0], tq[1], 1.3, lights)
    balon_hd(g, bq[0], 6.5, bq[1], 6.5)
    for k in range(5):                                                         # torres de luz
        a = math.atan2(tt[1], tt[0]) + (k + 0.5) / 5 * TAU
        q = sc + np.array([math.cos(a), math.sin(a)]) * (r0 + 4)
        fcyl(g, q[0], q[1], 0, 24, 0.45, rgb(140, 145, 150), mat(M_METAL, 0.4, 0.9), seg=8)
        fbox(g, q, (-math.sin(a), math.cos(a)), 2.5, 0.5, 24, 26, (1.0, 0.95, 0.85), mat(M_LAMP, 0.4, 0, 7.0))
        lights.append((float(q[0]), 20.0, float(q[1]), 60.0, 1.0, 0.92, 0.8, 3.5))
    for _ in range(700):                                                       # aficionados (dia y noche)
        a, rr_ = rnd.uniform(0, TAU), rnd.uniform(4, r0 + 4)
        x, z = sc[0] + rr_ * math.cos(a), sc[1] + rr_ * math.sin(a)
        crowd.add(x, 0.0, z, rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd)
    if md.concert:
        build_concert_hd(gc, crowd, rnd, track, sc, r0, r0 + 4, 10.0, info)
    info["stadium"] = (sc, r0 + 4, r0)


def torres_reforma_hd(g, rnd, track, i0, i1, count, zones, occ, lights):
    """Rascacielos de Paseo de la Reforma a los lados del tramo [i0, i1]."""
    placed = 0
    for _ in range(count * 25):
        if placed >= count:
            break
        i = rnd.randrange(i0, i1)
        side = rnd.choice((-1, 1))
        half = rnd.uniform(14, 22)
        c = track.pts[i] + track.nrm[i] * side * (WALK + half + rnd.uniform(8, 60))
        f = track.tan[i]
        cells = occ.cells(c, f, half, half)
        if track.nearest_dist(c[None])[0] < WALK + half * 1.45 or zones.hit(c).any() or not occ.free(cells):
            continue
        h = rnd.uniform(120, 245)
        fbox(g, c, f, half, half, 0, h, rnd.choice([rgb(90, 115, 140), rgb(70, 95, 120), rgb(120, 140, 160)]),
             mat(M_WINDOWS, 0.15, 0.5))
        fbox(g, c, f, half * 0.6, half * 0.6, h, h + 8, rgb(60, 70, 80), mat(M_METAL, 0.4, 0.6))
        box(g, c[0] - 0.8, h + 8, c[1] - 0.8, c[0] + 0.8, h + 18, c[1] + 0.8, (1.0, 0.1, 0.05), mat(M_EMIS, 0.5, 0, 8.0))
        occ.add(cells)
        placed += 1


def build_world_streets(track, md):
    """Circuito callejero CDMX: calles, banquetas, edificios, puentes peatonales, semaforos y mucho publico."""
    rnd = random.Random(sum(map(ord, md.key)) * 104729)
    g, gc, lights, info = Geo(), Geo(), [], {}
    crowd = PaletteCrowd(MEXICO_SHIRTS if md.palette == "mexico" else md.palette) if md.palette else Crowd()
    pts, nrm, tan, n = track.pts, track.nrm, track.tan, track.n
    cxm, czm = track.center
    occ, zones = Occupancy(), Zones()
    openz = Zones()                          # explanadas abiertas (sin edificios) frente a los lugares
    noveg = Zones()                          # donde no van arboles del parque
    # monumentos que pasan sobre la pista: buscar el tramo mas recto cerca del punto indicado
    over = {}
    for lm in md.landmarks:
        if lm[0] == "monumento":
            pos = md.w(lm[1])
            i_h = int(np.argmin(np.hypot(pts[:, 0] - pos[0], pts[:, 1] - pos[1])))
            cands = [(i_h + q) % n for q in range(-40, 41)]
            over["monumento"] = min(cands, key=lambda i: (np.max(np.abs(track.curv[[(i + k) % n for k in range(-12, 13)]])),
                                                          abs(i - i_h)))
    avoid = [(0, 30)] + [(i, 28) for i in over.values()] + [(a, 30) for a, b, _ in track.cross_pts] + \
        [(b, 30) for a, b, _ in track.cross_pts]
    G = 6500
    xs = np.linspace(cxm - G, cxm + G, 41)
    zs = np.linspace(czm - G, czm + G, 41)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    grid(g, np.stack([X, np.zeros_like(X), Z], -1), rgb(112, 112, 108), mat(M_CONC, 0.85), orient=(0, 1, 0))
    # ---- lugares reservados: salida, puentes peatonales e intersecciones
    bridges = pick_spots(track, md.ped_bridges, avoid, 40, 1 / 400, LIMIT + 19, offset=0.5) if md.ped_bridges else []
    ncross = max(16, int(track.L / 430))
    cross = pick_spots(track, ncross, avoid + [(b, 14) for b in bridges], 30, 1 / 300, LIMIT + 2, offset=0.0)
    reserved = [(pts[i][0], pts[i][1], LIMIT + 22) for i in bridges] + [(pts[i][0], pts[i][1], LIMIT + 12)
                                                                         for i in cross]
    reserved += [(pts[i][0], pts[i][1], LIMIT + 34) for i in over.values()]
    reserved += [(c[0], c[1], LIMIT + 30) for _, _, c in track.cross_pts]
    res = np.array(reserved, float).reshape(-1, 3)

    def is_reserved(x, z, extra=0.0):
        return len(res) and bool(np.any(np.hypot(res[:, 0] - x, res[:, 1] - z) < res[:, 2] + extra))
    o_tower, o_bb = hd.light_tower, hd.billboard_hd

    def lamp(g_, c, toward, lights_):
        if not is_reserved(c[0], c[1]):
            street_lamp_hd(g_, c, toward, lights_)

    def bboard(g_, cx, cz, fx, fz, width, y0, y1, k, lights_=None):
        if not is_reserved(cx, cz, 6.0):
            o_bb(g_, cx, cz, fx, fz, width, y0, y1, k, lights_)
    hd.light_tower, hd.billboard_hd = lamp, bboard
    gt = Geo()
    try:
        build_track_hd(gt, track, info, lights)
    finally:
        hd.light_tower, hd.billboard_hd = o_tower, o_bb
    lift_crossings(gt, track)
    g.merge(gt)
    if "monumento" in over:
        monumento_rev_hd(g, track, over["monumento"], lights)
        i = over["monumento"]
        occ.rect(pts[i], tan[i], LIMIT + 14, LIMIT + 14)
    for a, b, c in track.cross_pts:                                            # gran cruce con semaforos
        for k in (a, b):
            for side in (-1, 1):
                sb = pts[k] - tan[k] * (LIMIT + 10) * side + nrm[k] * side * (LIMIT + 1.3)
                semaforo_hd(g, sb, -nrm[k] * side, LIMIT + 1.3 - 3.0, -tan[k] * side)
        occ.rect(c, (1, 0), LIMIT + 24, LIMIT + 24)
    near_cross = np.zeros(n, bool)
    for i in cross:
        near_cross[[(i + q) % n for q in range(-3, 4)]] = True
    # ---- banquetas
    for side in (-1, 1):
        q = pts + nrm * side * (LIMIT + 3)
        ok = (track.nearest_dist(q) >= LIMIT + 2.5) & ~near_cross
        for r in hd.runs(ok):
            P = hd.offset_grid(track, r, [LIMIT + 0.4, WALK + 1.0], 0.14, side)
            grid(g, P, rgb(172, 168, 160), mat(M_CONC, 0.85), orient=(0, 1, 0))
            P = hd.offset_grid(track, r, [LIMIT + 2.2, LIMIT + 2.35], 0.15, side)
            grid(g, P, rgb(200, 190, 60), mat(M_CONC, 0.8), orient=(0, 1, 0))
        occ_pts = pts + nrm * side * (LIMIT + 3.5)
        for k in range(0, n, 2):
            occ.rect(occ_pts[k], tan[k], 3.5, 3.6)
    # ---- estadio del concierto (al lado de la pista)
    keep_info = None
    if md.venue and md.venue.get("kind") == "fanfest":
        v = md.venue
        r0 = float(v.get("r0", 55))
        sc = beside_track(track, md.w(v["pos"]), r0 + 8 + LIMIT + 10)
        fanfest_hd(g, gc, crowd, rnd, track, sc, r0, md, info, lights)
        zones.circle(sc, r0 + 9)
        openz.circle(sc, float(track.nearest_dist(sc[None])[0]) - LIMIT - 1)
        keep_info = (sc, r0 + 8)
    elif md.venue:
        v = md.venue
        r0, rows = float(v.get("r0", 62)), int(v.get("rows", 26))
        rmax_est = r0 + rows * 1.0
        sc = beside_track(track, md.w(v["pos"]), rmax_est + LIMIT + 14)
        ang = np.linspace(0, TAU, 65)
        for j in range(64):
            quad(g, (sc[0], 0.03, sc[1]), (sc[0], 0.03, sc[1]),
                 (sc[0] + (r0 + 0.5) * math.cos(ang[j + 1]), 0.03, sc[1] + (r0 + 0.5) * math.sin(ang[j + 1])),
                 (sc[0] + (r0 + 0.5) * math.cos(ang[j]), 0.03, sc[1] + (r0 + 0.5) * math.sin(ang[j])),
                 rgb(150, 146, 138), mat(M_CONC, 0.75))
        i_n = int(np.argmin(np.hypot(pts[:, 0] - sc[0], pts[:, 1] - sc[1])))
        to_track = norm(np.array([pts[i_n][0] - sc[0], pts[i_n][1] - sc[1], 0.0]))[:2].astype(float)
        baseball_field_hd(g, sc, r0, to_track)
        rmax, Hb = bowl_hd(g, crowd, rnd, track, sc, r0, lights, rows=rows, dd=1.0, dh=0.55, inst_rows=12)
        for k in range(6):                                                     # torres de luz del estadio
            a = math.atan2(to_track[1], to_track[0]) + (k + 0.5) / 6 * TAU
            c = sc + np.array([math.cos(a), math.sin(a)]) * (rmax + 2.5)
            fcyl(g, c[0], c[1], 0, Hb + 30, 0.7, rgb(140, 145, 150), mat(M_METAL, 0.4, 0.9), seg=10, r1=0.45)
            f = -np.array([math.cos(a), math.sin(a)])
            frame_box(g, c, (-f[1], f[0]), (5.0, 0.6), rgb(60, 60, 66), mat(M_METAL, 0.5, 0.7), Hb + 29, Hb + 34)
            cc = c + f * 0.7
            frame_box(g, cc, (-f[1], f[0]), (4.6, 0.1), (1.0, 0.95, 0.85), mat(M_LAMP, 0.4, 0, 8.0), Hb + 29.5,
                      Hb + 33.5)
            lights.append((float(c[0] + f[0] * 20), Hb + 20, float(c[1] + f[1] * 20), 90.0, 1.0, 0.95, 0.85, 4.0))
        fc = sc + to_track * (rmax + 0.6)                                      # letreros hacia la pista
        sign_quad(g, fc, to_track, 22, Hb - 6, Hb + 2, 9)
        for sgn in (-1, 1):
            a = math.atan2(to_track[1], to_track[0]) + sgn * 0.42
            dd = np.array([math.cos(a), math.sin(a)])
            sign_quad(g, sc + dd * (rmax + 0.6), dd, 14, Hb - 8, Hb - 1, md.band_sign if sgn < 0 else 14)
        if md.concert:
            build_concert_hd(gc, crowd, rnd, track, sc, r0, rmax, Hb, info)
        info["stadium"] = (sc, rmax, r0)
        zones.circle(sc, rmax + 6)
        keep_info = (sc, rmax)
        # fans que llegan al concierto + puestos alrededor del estadio
        FA = np.array([(rnd.uniform(0, TAU), rnd.uniform(rmax + 3, rmax + 30)) for _ in range(1400)])
        FP = sc + np.column_stack([np.cos(FA[:, 0]), np.sin(FA[:, 0])]) * FA[:, 1:2]
        for (x, z), dd_ in zip(FP, track.nearest_dist(FP)):
            if dd_ >= LIMIT + 1.5:
                crowd.add(x, 0.0, z, sc[0] - x + rnd.uniform(-20, 20), sc[1] - z + rnd.uniform(-20, 20), rnd)
        for k in range(26):
            a = k / 26 * TAU + rnd.uniform(-0.05, 0.05)
            c = sc + np.array([math.cos(a), math.sin(a)]) * (rmax + 36)
            if track.nearest_dist(c[None])[0] > LIMIT + 4:
                puesto_hd(g, c, (-math.sin(a), math.cos(a)), rnd)
        zones.circle(sc, rmax + 40)
    if "stage" not in info:
        info.update({k: v for k, v in no_concert_info(track).items() if k not in info})
    # ---- lugares del paisaje
    planes = None
    for lm in md.landmarks:
        kind = lm[0]
        pos = md.w(lm[1]) if kind not in ("torres", "parque") else None
        if kind == "monumento":
            continue                                                           # ya se construyo sobre la pista
        if kind == "angel":                                                    # glorieta con festejo
            c = beside_track(track, pos, LIMIT + 22)
            for j in range(48):
                a0, a1 = j / 48 * TAU, (j + 1) / 48 * TAU
                quad(g, (c[0], 0.05, c[1]), (c[0], 0.05, c[1]), (c[0] + 20 * math.cos(a1), 0.05, c[1] + 20 * math.sin(a1)),
                     (c[0] + 20 * math.cos(a0), 0.05, c[1] + 20 * math.sin(a0)), rgb(175, 170, 160), mat(M_CONC, 0.8))
            angel_hd(g, c[0], c[1], lights)
            info["angel"] = c
            for _ in range(450):
                a, rr = rnd.uniform(0, TAU), rnd.uniform(9, 40)
                x, z = c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)
                if track.nearest_dist(np.array([[x, z]]))[0] > LIMIT + 1.5:
                    crowd.add(x, 0.1, z, c[0] - x, c[1] - z, rnd)
            zones.circle(c, 22)
            openz.circle(c, 45)
            noveg.circle(c, 45)
        elif kind == "basilica":
            c = beside_track(track, pos, LIMIT + 110)
            i_n = int(np.argmin(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])))
            f = c - pts[i_n]
            basilica_hd(g, crowd, rnd, c, f / np.linalg.norm(f), lights)
            zones.circle(c, 100)
            openz.circle(c, float(track.nearest_dist(c[None])[0]) - LIMIT - 1)
        elif kind == "castillo":
            c = v2(pos)
            r = float(track.nearest_dist(c[None])[0]) - LIMIT - 14
            castillo_hd(g, rnd, c, r, lights)
            zones.circle(c, r)
            noveg.circle(c, r * 0.75)
        elif kind == "museo":
            hl, hw = 80.0, 48.0
            c = beside_track(track, pos, hw + LIMIT + 22)
            i_n = int(np.argmin(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])))
            museo_hd(g, c, tan[i_n], lights)
            zones.rect(c, tan[i_n], hl + 3, hw + 3)
            noveg.rect(c, tan[i_n], hl + 14, hw + 14)
            bq = pts[i_n] + nrm[i_n] * np.sign(np.dot(c - pts[i_n], nrm[i_n])) * (WALK + 10)
            balon_hd(g, bq[0], 9.0, bq[1], 9.0)                               # balon gigante en la salida
            occ.rect(bq, (1, 0), 10, 10)
        elif kind == "lago":
            c, half = v2(pos), lm[2]
            flat_rect(g, c, (1, 0), half[0], half[1], 0.06, rgb(32, 58, 78), mat(M_DEF, 0.05, 0.3))
            for k in range(8):                                                 # lanchas de pedales
                q = c + np.array([rnd.uniform(-half[0] * 0.8, half[0] * 0.8), rnd.uniform(-half[1] * 0.7, half[1] * 0.7)])
                fbox(g, q, (1, 0), 1.4, 0.8, 0.0, 0.7, rnd.choice(LONAS), mat(M_PAINT, 0.4))
            zones.rect(c, (1, 0), half[0], half[1])
            noveg.rect(c, (1, 0), half[0] + 4, half[1] + 4)
        elif kind == "parque":                                                 # bosque de Chapultepec
            (x0, z0), (x1, z1) = md.w(lm[1][0]), md.w(lm[1][1])
            cc, hh = np.array([(x0 + x1) / 2, (z0 + z1) / 2]), (abs(x1 - x0) / 2, abs(z1 - z0) / 2)
            flat_rect(g, cc, (1, 0), hh[0], hh[1], 0.02, rgb(70, 115, 45), mat(M_GRASS, 0.95))
            openz.rect(cc, (1, 0), hh[0], hh[1])
            info["park"] = (cc, hh, lm[2] if len(lm) > 2 else 700)
        elif kind == "torres":
            f0, f1 = lm[1]
            info["torres"] = (int(f0 * n), int(f1 * n), lm[2] if len(lm) > 2 else 8)
        elif kind == "gnp":
            r0, rows = 95.0, 34
            sc = beside_track(track, pos, r0 + rows + LIMIT + 60)
            ang = np.linspace(0, TAU, 65)
            for j in range(64):
                quad(g, (sc[0], 0.03, sc[1]), (sc[0], 0.03, sc[1]),
                     (sc[0] + r0 * math.cos(ang[j + 1]), 0.03, sc[1] + r0 * math.sin(ang[j + 1])),
                     (sc[0] + r0 * math.cos(ang[j]), 0.03, sc[1] + r0 * math.sin(ang[j])), rgb(60, 125, 50),
                     mat(M_GRASS, 0.9))
            rmax, Hb = bowl_hd(g, crowd, rnd, track, sc, r0, lights, rows=rows, dd=1.0, dh=0.5, inst_rows=5)
            i_n = int(np.argmin(np.hypot(pts[:, 0] - sc[0], pts[:, 1] - sc[1])))
            tt = norm(np.array([pts[i_n][0] - sc[0], pts[i_n][1] - sc[1], 0.0]))[:2].astype(float)
            sign_quad(g, sc + tt * (rmax + 3.6), tt, 30, Hb + 1, Hb + 9, 12)
            if md.concert:                                                     # A7X tambien toca en el GNP
                g2, dummy = Geo(), {}
                build_concert_hd(g2, crowd, rnd, track, sc, r0, rmax, Hb, dummy)
                remap_sign(g2, 10, 2)
                gc.merge(g2)
            zones.circle(sc, rmax + 20)
            clear = float(track.nearest_dist(sc[None])[0])
            openz.circle(sc, clear - LIMIT - 1)
            PA = np.array([(rnd.uniform(0, TAU), rnd.uniform(rmax + 4, clear - LIMIT - 9)) for _ in range(500)])
            for a, rr in PA:                                                   # explanada: fans y puestos
                x, z = sc[0] + rr * math.cos(a), sc[1] + rr * math.sin(a)
                crowd.add(x, 0.0, z, sc[0] - x + rnd.uniform(-30, 30), sc[1] - z + rnd.uniform(-30, 30), rnd)
            for k in range(18):
                a = k / 18 * TAU
                c = sc + np.array([math.cos(a), math.sin(a)]) * (rmax + 12)
                puesto_hd(g, c, (-math.sin(a), math.cos(a)), rnd)
        elif kind == "palacio":
            r = 78.0
            c = beside_track(track, pos, r + LIMIT + 40)
            palacio_hd(g, c, r, lights)
            zones.circle(c, r + 14)
            openz.circle(c, float(track.nearest_dist(c[None])[0]) - LIMIT - 1)
        elif kind == "aeropuerto":
            d = v2(lm[2]) if len(lm) > 2 else np.array([0.77, -0.64])
            flights, plane = airport_hd(g, rnd, pos, d, lights, zones, track, openz)
            info["flights"], planes = flights, plane
            i_n = int(np.argmin(np.hypot(pts[:, 0] - pos[0], pts[:, 1] - pos[1])))
            for off in (-60, 60):                                              # anuncios del aeropuerto
                i = (i_n + off) % n
                for side in (-1, 1):
                    c = pts[i] + nrm[i] * side * (WALK + 3)
                    if np.dot(nrm[i] * side, pos - pts[i]) > 0 and track.nearest_dist(c[None])[0] > WALK:
                        billboard_hd(g, c[0], c[1], nrm[i, 0] * side, nrm[i, 1] * side, 18, 6.0, 11.0, 1, lights)
                        occ.rect(c, tan[i], 10, 3)
    info["plane_geo"] = planes
    # ---- puentes peatonales
    for k, i in enumerate(bridges):
        ped_bridge_hd(g, crowd, rnd, track, i, [11, 6, 8, 15, 4][k % 5], lights, occ)
    # ---- cruces: calle transversal, cebras y semaforos
    for i in cross:
        p, t = pts[i], tan[i]
        for side in (-1, 1):
            o = nrm[i] * side
            length = 170.0
            for ln in (170.0, 90.0, 40.0):
                tip = p + o * (LIMIT + ln)
                mid = p + o * (LIMIT + ln * 0.5)
                if track.nearest_dist(np.array([tip, mid]))[0] > LIMIT + 8 and not zones.hit([tip, mid]).any():
                    length = ln
                    break
            else:
                continue
            c = p + o * (LIMIT + 0.4 + length / 2)
            flat_rect(g, c, o, length / 2, 6.5, 0.05, rgb(56, 56, 60), mat(M_CONC, 0.7))
            for s in np.arange(LIMIT + 14, LIMIT + length - 4, 9.0):
                flat_rect(g, p + o * s, o, 1.6, 0.08, 0.06, (0.95, 0.8, 0.15), mat(M_LINE, 0.6))
            for w in np.arange(-5.4, 5.5, 1.2):
                flat_rect(g, p + o * (WALK + 3.0) + t * w, o, 2.0, 0.3, 0.06, (0.92, 0.92, 0.92), mat(M_LINE, 0.6))
            occ.rect(c, o, length / 2, 7.5)
            if length > 60 and rnd.random() < 0.8:
                pc = p + o * (LIMIT + rnd.uniform(35, length - 12)) + t * 4.0 * rnd.choice((-1, 1))
                pesero_hd(g, pc, o, rnd)
            sb = p - t * 12.0 + o * (LIMIT + 1.3)
            semaforo_hd(g, sb, -o, LIMIT + 1.3 - 3.0, -t)
            for _ in range(rnd.randint(4, 10)):                               # gente esperando en la esquina
                q = p + o * rnd.uniform(LIMIT + 1.5, WALK) + t * rnd.uniform(-9, 9)
                crowd.add(q[0], 0.14, q[1], -o[0], -o[1], rnd)
        for a_off in (-8.5, 8.5):                                              # cebra sobre la calle principal
            for w in np.arange(-HW + 0.7, HW - 0.5, 1.25):
                flat_rect(g, p + t * a_off + nrm[i] * w, t, 2.0, 0.3, 0.075, (0.93, 0.93, 0.93), mat(M_LINE, 0.55))
        flat_rect(g, p - t * 11.5, t, 0.25, HW, 0.075, (0.93, 0.93, 0.93), mat(M_LINE, 0.55))
    # ---- tribunas en las rectas largas
    runs_ = straight_runs(track, max_curv=1 / 600, min_len=200)
    placed = 0
    for i0, ln in runs_:
        if placed >= 3:
            break
        length = min(260.0, ln * track.spacing - 60)
        i_mid = (i0 + ln // 2) % n
        i_s = (i_mid - int(length / 2 / track.spacing)) % n
        if any(min(abs(i_s + k - j) % n, n - abs(i_s + k - j) % n) < 8 for j in bridges + cross
               for k in range(0, int(length / track.spacing), 4)):
            i_s = (i_s + 20) % n
        for side in (1, -1):
            out = nrm[i_s] * side
            o = pts[i_s] + out * (LIMIT + 4)
            if not footprint_clear(track, o, tan[i_s], out, length, 22) or zones.hit(
                    [o + tan[i_s] * a + out * b for a in (0, length / 2, length) for b in (0, 20)]).any():
                continue
            if keep_info is not None and math.hypot(*(o + tan[i_s] * length / 2 - keep_info[0])) < keep_info[1] + 60:
                continue
            grandstand_hd(g, crowd, rnd, o, tan[i_s], out, length, 18, 0.9, 0.5,
                          rnd.choice([rgb(30, 80, 190), rgb(220, 120, 30), rgb(190, 40, 40), rgb(40, 150, 70)]), 11,
                          lights)
            occ.rect(o + tan[i_s] * length / 2 + out * 13, tan[i_s], length / 2 + 4, 15)
            zones.rect(o + tan[i_s] * length / 2 + out * 13, tan[i_s], length / 2 + 2, 14)
            placed += 1
            break
    # ---- bardas de la Virgen y papel picado sobre la calle
    if "guadalupe" in md.extras:
        for frac in (0.18, 0.43, 0.63, 0.9):
            i = int(frac * n)
            for side in (1, -1):
                c = pts[i] + nrm[i] * side * (WALK + 1.5)
                cells = occ.cells(c, tan[i], 7, 1.2)
                if track.nearest_dist(c[None])[0] < WALK or zones.hit(c).any() or openz.hit(c).any() or \
                        not occ.free(cells):
                    continue
                guadalupe_wall_hd(g, c, nrm[i] * side, lights, rnd)
                occ.add(occ.cells(c, tan[i], 8, 4))
                break
    if "papel" in md.extras:
        for frac in (0.04, 0.15, 0.31, 0.5, 0.58, 0.74, 0.86, 0.95):
            i = int(frac * n)
            papel_picado_hd(g, pts[i] + nrm[i] * -(HW + 2.5), pts[i] + nrm[i] * (HW + 2.5))
    if "banderines" in md.extras:                                              # banderines del mundial
        for k, i in enumerate(range(40, n, 55)):
            if any(min(abs(i - a), n - abs(i - a)) < 15 for a, _ in avoid[1:]):
                continue
            banderines_hd(g, pts[i] + nrm[i] * -(HW + 2.5), pts[i] + nrm[i] * (HW + 2.5), k0=k)
    if "torres" in info:
        i0, i1, cnt = info.pop("torres")
        torres_reforma_hd(g, rnd, track, i0, i1, cnt, zones, occ, lights)
    if md.pads:                                                                # rampas de turbo en las rectas
        pads = []
        for i0, ln in straight_runs(track, max_curv=1 / 1200, min_len=260):
            for k in range(int(25 / track.spacing), ln - int(60 / track.spacing), int(380 / track.spacing)):
                i = (i0 + k) % n
                if len(pads) < md.pads and not any(min(abs(i - a), n - abs(i - a)) < 25 for a, _ in avoid):
                    pads.append(i)
        for i in pads:
            turbo_pad_hd(g, track, i)
        info["pads"] = np.array(sorted(pads), int)
    # ---- edificios con locales frente a la calle
    for side in (-1, 1):
        s = 0.0
        while s < track.L:
            w = rnd.uniform(9.0, 24.0)
            dp = rnd.uniform(10.0, 22.0)
            x, z, tx, tz = track.point(s + w / 2)
            i = int(((s + w / 2) % track.L) / track.spacing) % n
            f, o = np.array([tx, tz]), nrm[i] * side
            c = np.array([x, z]) + o * (WALK + 0.4 + dp / 2)
            corners = np.array([c + f * a + o * b for a in (-w / 2, w / 2) for b in (-dp / 2, dp / 2)] + [c - o * dp / 2])
            cells = occ.cells(c, f, w / 2, dp / 2)
            if (track.nearest_dist(corners) >= WALK - 0.8).all() and occ.free(cells) and \
                    not (zones.hit(corners) | openz.hit(corners)).any():
                h = rnd.choice(STORIES) * 3.6 + 0.4
                building_hd(g, crowd, rnd, c, f, o, w / 2, dp / 2, h, rnd.choice(FACADES),
                            shop=rnd.random() < 0.85, roof_fans=h < 20 and rnd.random() < 0.12)
                if "balones" in md.extras and h < 25 and rnd.random() < 0.05:
                    balon_hd(g, c[0], h + 3.0, c[1], 3.0)
                occ.add(cells)
                s += w + rnd.uniform(0.0, 1.0)
            else:
                s += 4.0
    # ---- jacarandas y puestos en la banqueta
    for side in (-1, 1):
        idx = np.arange(5, n, 9)
        CP = pts[idx] + nrm[idx] * side * (WALK - 1.0)
        okd = (track.nearest_dist(CP) >= WALK - 1.5) & ~zones.hit(CP)
        for i, c, ok_ in zip(idx, CP, okd):
            r_ = rnd.random()
            if not ok_ or near_cross[i] or is_reserved(c[0], c[1], 4.0):
                continue
            if r_ < 0.22:
                jacaranda_hd(g, c[0], c[1], rnd.uniform(0.9, 1.25), rnd)
            elif r_ < 0.3:
                puesto_hd(g, pts[i] + nrm[i] * side * (WALK - 1.6), tan[i], rnd)
    # ---- resto de la ciudad (manzanas)
    lo, hi = pts.min(axis=0) - md.city_margin, pts.max(axis=0) + md.city_margin
    st_ = md.city_step
    gx, gz = np.meshgrid(np.arange(lo[0], hi[0], st_), np.arange(lo[1], hi[1], st_), indexing="ij")
    C = np.column_stack([gx.ravel(), gz.ravel()]) + np.array([[rnd.uniform(-6, 6), rnd.uniform(-6, 6)]])
    D = track.nearest_dist(C)
    blocked = zones.hit(C) | openz.hit(C)
    yaw0 = rnd.uniform(0, TAU)
    for c, dist, bl in zip(C, D, blocked):
        if bl or dist < WALK + 26:
            continue
        w, dp = rnd.uniform(9, 16) * st_ / 34, rnd.uniform(9, 16) * st_ / 34
        f = np.array([math.cos(yaw0), math.sin(yaw0)])
        cells = occ.cells(c, f, w, dp)
        if not occ.free(cells):
            continue
        occ.add(cells)
        tall = rnd.random() < (0.05 if dist > 300 else 0.015)
        h = rnd.uniform(40, 110) if tall else rnd.choice(STORIES) * 3.6 + 0.4
        building_hd(g, crowd, rnd, c, f, np.array([-f[1], f[0]]), w, dp, h, rnd.choice(FACADES),
                    shop=False, roof_fans=False) if dist < md.detail_dist else \
            frame_box(g, c, f, (w, dp), rnd.choice(FACADES), mat(M_WINDOWS, 0.5, 0.05), 0, h)
        if tall:
            box(g, c[0] - 0.8, h, c[1] - 0.8, c[0] + 0.8, h + 5, c[1] + 0.8, (1.0, 0.1, 0.05), mat(M_EMIS, 0.5, 0, 6.0))
    if "park" in info:                                                         # bosque de Chapultepec
        cc, hh, cnt = info.pop("park")
        TP = cc + np.column_stack([np.array([rnd.uniform(-hh[0], hh[0]) for _ in range(cnt * 3)]),
                                   np.array([rnd.uniform(-hh[1], hh[1]) for _ in range(cnt * 3)])])
        okd = (track.nearest_dist(TP) > LIMIT + 9) & ~noveg.hit(TP) & ~zones.hit(TP)
        placed_t = 0
        for (x, z), ok_ in zip(TP, okd):
            if placed_t >= cnt:
                break
            if ok_ and occ.free(occ.cells((x, z), (1, 0), 2, 2)):
                if rnd.random() < 0.2:
                    jacaranda_hd(g, x, z, rnd.uniform(1.0, 1.4), rnd)
                else:
                    tree_hd(g, x, z, rnd.uniform(1.0, 1.7), rnd)
                placed_t += 1
    span = float(np.max(pts.max(axis=0) - pts.min(axis=0))) / 2 + md.city_margin
    for _ in range(260):                                                       # horizonte
        a = rnd.uniform(0, TAU)
        r = rnd.uniform(span + 300, span + 1800)
        x, z = cxm + r * math.cos(a), czm + r * math.sin(a)
        if zones.hit([(x, z)]).any():
            continue
        w, dp = rnd.uniform(25, 60), rnd.uniform(25, 60)
        h = rnd.choice([12, 18, 25, 35, 50, 70, 90, 130])
        fbox(g, (x, z), (1, 0), w / 2, dp / 2, 0, h, rnd.choice(FACADES), mat(M_WINDOWS, 0.4, 0.1))
    for k in range(8):                                                         # torres de Reforma al poniente
        x, z = cxm - span - 900 + k * 80 + rnd.uniform(-20, 20), czm - 900 + rnd.uniform(-150, 150)
        h = rnd.uniform(150, 245)
        w = rnd.uniform(28, 40)
        fbox(g, (x, z), (1, 0), w / 2, w / 2, 0, h, rgb(90, 115, 140), mat(M_WINDOWS, 0.15, 0.5))
        box(g, x - 1, h, z - 1, x + 1, h + 14, z + 1, (1.0, 0.1, 0.05), mat(M_EMIS, 0.5, 0, 8.0))
    if md.mountains:
        for k in range(26):
            a = k / 26 * TAU + rnd.uniform(-0.1, 0.1)
            r = span + 2600 + rnd.uniform(0, 900)
            mountain_hd(g, cxm + r * math.cos(a), czm + r * math.sin(a), rnd.uniform(800, 1400),
                        rnd.uniform(300, 650), rnd)
    if md.volcanoes:
        d0 = span + 3300
        mountain_hd(g, cxm + d0, czm + 1000, 1900, 1350, rnd, snow=1.0, seg=36, rings=14)
        mountain_hd(g, cxm + d0 - 300, czm - 1000, 2300, 1100, rnd, snow=1.0, seg=36, rings=14)
    # ---- mucho publico en las banquetas
    for side in (1, -1):
        idx = np.arange(0, n, 3)
        BP = pts[idx] + nrm[idx] * side * (LIMIT + 1.3)
        okd = (track.nearest_dist(BP) >= LIMIT + 1.0) & ~zones.hit(BP)
        for i, base_, ok_ in zip(idx, BP, okd):
            if not ok_ or rnd.random() > md.fans:
                continue
            for _ in range(rnd.randint(3, 8)):
                t_, d_ = rnd.uniform(-4.5, 4.5), rnd.uniform(0.0, 4.4)
                crowd.add(base_[0] + tan[i, 0] * t_ + nrm[i, 0] * side * d_, 0.14,
                          base_[1] + tan[i, 1] * t_ + nrm[i, 1] * side * d_, -nrm[i, 0] * side, -nrm[i, 1] * side, rnd)
    # ---- camaras de TV para el modo demo
    cams = []
    for i in range(0, n, 45):
        side = 1 if (i // 45) % 2 else -1
        c = pts[i] + nrm[i] * side * (LIMIT + 4)
        if track.nearest_dist(c[None])[0] > LIMIT + 3:
            cams.append((float(c[0]), 7.5, float(c[1])))
    info["tvcams"] = cams
    if not gc.parts:
        quad(gc, (0, -50, 0), (0.1, -50, 0), (0.1, -50, 0.1), (0, -50, 0.1), (0, 0, 0), mat(0))
    return g, gc, crowd, lights, info


BUILD_WORLD_CDMX = hd.build_world        # escenario original del Autodromo (se guarda antes de reemplazarlo)


def build_world_for(track):
    md = track.md
    if md.builder == "cdmx":
        return BUILD_WORLD_CDMX(track)
    if md.builder == "calles":
        return build_world_streets(track, md)
    return build_world_auto(track, md)


# ================================================================ juego con mapas
class MapFX(hd.FXHD):
    """Efectos: el concierto solo se activa si la pista tiene concierto."""
    allow_concert = True

    @property
    def concert(self):
        return getattr(self, "_c", False) and self.allow_concert

    @concert.setter
    def concert(self, v):
        self._c = v


ORIG_CHUNKS = hd.StaticChunks
SONG_EXT = (".mp3", ".ogg", ".wav", ".flac")


class GameMaps(hd.GameHD):
    def __init__(self, headless=False, map_key=None):
        for m in MAPS:                                   # largo de cada pista (para el selector)
            m.length = MapTrack(m).L
        self.map_i = map_index(map_key) if map_key and map_index(map_key) is not None else 0
        self.map_sel = self.map_i
        self.pending_map = None
        self._gpu = []
        self.plane_vao = None
        hd.Track = lambda: MapTrack(MAPS[self.map_i])   # el juego base construye la pista y el mundo del mapa
        hd.build_world = build_world_for
        hd.StaticChunks = self._chunks
        hd.LAPS = MAPS[self.map_i].laps
        self.apply_signs(MAPS[self.map_i], upload=False)
        super().__init__(headless)
        self._wrap_fx()
        self.setup_planes()
        self.apply_music()
        if not headless:
            pygame.display.set_caption(f"GTC Racing HD - {MAPS[self.map_i].name}")

    @property
    def md(self):
        return MAPS[self.map_i]

    def _chunks(self, R, arr, center, keep_radius=None):
        """Bloques del escenario; en pistas grandes se recorta por bloques hasta mas lejos."""
        tr = getattr(self, "track", None)
        if keep_radius is None:
            keep_radius = 1300.0
            if tr is not None and self.md.builder != "cdmx":
                keep_radius = max(1300.0, float(np.max(np.hypot(*(tr.pts - tr.center).T))) + 900.0)
        return ORIG_CHUNKS(R, arr, center, keep_radius)

    def apply_signs(self, md, upload=True):
        """Letreros propios del mapa (nombre del estadio, banda...) en el atlas de texturas."""
        signs = list(ORIG_SIGNS)
        for k, v in md.signs.items():
            signs[k] = v
        if signs == list(hd.SIGNS):
            return
        hd.SIGNS = signs
        if upload:
            self.R.atlas.write(hd.make_atlas_hd())
            self.R.atlas.build_mipmaps()

    def apply_music(self):
        """Soundtrack de noche de este mapa (carpeta music_dir)."""
        folder = os.path.join(hd.HERE, self.md.music_dir)
        try:
            pl = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(SONG_EXT)]
        except OSError:
            pl = []
        self.music.playlist = pl
        self.music.cur = None

    def music_mode(self):
        m = super().music_mode()
        if m == "playlist" and not self.music.playlist:          # aun sin canciones de la banda
            return "ambience" if self.state in ("intro", "countdown") else "race"
        return m

    # ------------------------------------------------------------ aviones del aeropuerto
    def setup_planes(self):
        geo = self.info.get("plane_geo")
        self.plane_vao = self.R.vao_scene(geo.array()) if geo is not None else None

    def plane_models(self):
        out = []
        for fl in self.info.get("flights", []):
            p = flight_pose(fl, self.t)
            if p is not None:
                x, y, z, ang, pitch = p
                out.append((model_full(x, y, z, ang, 0.0, pitch), (x, y, z)))
        return out

    def draw_cars(self, prog, items, shadow=False, headl=0.0):
        super().draw_cars(prog, items, shadow, headl)
        if self.plane_vao is None or self.state in ("select", "lobby"):
            return
        for M_, pos in self.plane_models():
            if not shadow and hasattr(self, "L_pos"):
                self.light_bytes(pos, 40.0, prog)
            setu(prog, u_model=gl_bytes(M_))
            self.plane_vao[1 if shadow else 0].render()
        setu(prog, u_model=gl_bytes(np.identity(4, F4)))

    def _wrap_fx(self):
        self.fx.__class__ = MapFX
        self.fx.__dict__.pop("concert", None)
        self.fx.allow_concert = self.md.concert

    # ------------------------------------------------------------ cambio de mapa
    def _keepalive(self):
        """Durante una carga larga: mandar y recibir paquetes para que la partida LAN no se corte."""
        if self.net:
            try:
                self.net_send(0.05)
                self.net.recv_all()
            except Exception:
                pass

    def load_map(self, i, keep_state=False):
        """Construye el escenario de otra pista (se libera la memoria de video de la anterior)."""
        if i == self.map_i:
            return
        state = self.state
        self.map_i = i
        md = self.md
        self.loading(f"Cargando {md.name}...", 0.1)
        self.apply_signs(md)
        for obj in self._gpu:
            try:
                obj.release()
            except Exception:
                pass
        self._gpu = []
        created = []
        orig_buffer = self.ctx.buffer

        def tracked_buffer(*a, **k):
            b = orig_buffer(*a, **k)
            created.append(b)
            return b
        # el escenario se arma en otro hilo: la ventana sigue respondiendo y la partida LAN no se corta
        res = {}

        def work():
            try:
                tr = MapTrack(md)
                res["world"] = (tr, build_world_for(tr))
            except BaseException as e:            # se vuelve a lanzar en el hilo principal
                res["err"] = e
        th = threading.Thread(target=work, daemon=True)
        th.start()
        t0 = time.time()
        while th.is_alive():
            th.join(0.2)
            self.loading(f"Cargando {md.name}...", 0.1 + 0.48 * min(1.0, (time.time() - t0) / 15.0))
            self._keepalive()
        if "err" in res:
            raise res["err"]
        self.ctx.buffer = tracked_buffer
        try:
            self.track, (g, gc, crowd, lights, self.info) = res["world"]
            self.loading("Subiendo geometría a la GPU...", 0.6)
            self.static = hd.StaticChunks(self.R, g.array(), self.track.center)
            self._keepalive()
            self.concert = self.R.vao_scene(gc.array())
            self.static_lights = np.array(lights, F4).reshape(-1, 8)
            self.crowd = hd.CrowdGPU(self.ctx, self.R.p_crowd, crowd.array())
            self._keepalive()
            self.setup_planes()
        finally:
            self.ctx.buffer = orig_buffer
        self._gpu = created
        if self.net and self.net.connected(6.0):
            self.net.last_rx = time.time()        # el otro jugador siguio conectado durante la carga
        hd.LAPS = md.laps
        self.apply_music()
        self.fx.info = self.info
        self.fx.p = self.fx.p[:0]
        for name in ("v", "c", "life", "maxl", "size", "kind"):
            setattr(self.fx, name, getattr(self.fx, name)[:0])
        self.fx.allow_concert = md.concert
        self.mm_cache = None
        self.R.static_key = None
        self.loading("¡Listo!", 1.0)
        if not self.headless:
            pygame.display.set_caption(f"GTC Racing HD - {md.name}")
        if keep_state:
            if self.mp == "guest":
                self.rid = 0            # el invitado vuelve a esperar la orden de salida del anfitrion
            self.new_race(attract=True)
            self.state = "lobby"
        else:
            self.new_race(attract=True)

    # ------------------------------------------------------------ multijugador: el anfitrion elige la pista
    def _wrap_net(self):
        net = self.net
        if not net:
            return
        osend, orecv = net.send, net.recv_all

        def send(obj, addr=None):
            if net.role == "host" and obj.get("t") == "h":
                obj["map"] = MAPS[self.map_sel if self.state == "lobby" else self.map_i].key
            osend(obj, addr)

        def recv():
            msgs = orecv()
            if net.role != "guest":
                return msgs
            keep = []
            for m, a in msgs:
                if m.get("t") == "h" and m.get("map"):
                    k = map_index(m["map"])
                    if k is not None and k != self.map_i:
                        self.pending_map = k
                        continue                  # no arrancar la carrera hasta tener la misma pista
                keep.append((m, a))
            return keep
        net.send, net.recv_all = send, recv

    def start_host(self):
        super().start_host()
        self._wrap_net()

    def join(self, ip):
        super().join(ip)
        self._wrap_net()

    def update(self, rdt):
        if self.pending_map is not None and self.mp == "guest":
            k, self.pending_map = self.pending_map, None
            if k != self.map_i:
                self.map_sel = k
                self.load_map(k, keep_state=True)
                if self.net:
                    self.net.last_rx = time.time()
                return
        super().update(rdt)
        self.map_extras(rdt)

    def map_extras(self, rdt):
        """Fuegos artificiales por toda la ruta (noche) y rampas de turbo."""
        md = self.md
        if md.fireworks and self.tod == 2 and self.fx.concert and \
                self.state in ("menu", "intro", "countdown", "race", "finished"):
            self._fw = getattr(self, "_fw", 0.0) - rdt
            if self._fw <= 0:
                self._fw = random.uniform(0.1, 0.3)
                ref = self.player if self.state != "menu" else self.cars[getattr(self, "tv_target", 0) % len(self.cars)]
                x, z, dx, dz = self.track.point(ref.s + random.uniform(-80, 500))
                side, off = random.choice((-1, 1)), LIMIT + random.uniform(12, 120)
                self.fx.spawn((x - dz * side * off, 1.0, z + dx * side * off),
                              [random.gauss(0, 4), random.uniform(30, 46), random.gauss(0, 4)], [3.0, 2.2, 1.2], 3.5,
                              1.4, 1)
        pads = self.info.get("pads")
        if pads is not None and len(pads) and self.state == "race":
            n = self.track.n
            for c in self.cars:
                if getattr(c, "remote", False):
                    continue
                hit = pads[(c.idx - pads) % n < 4]
                if len(hit) and abs(c.lat) < HW and getattr(c, "_pad", -1) != int(hit[0]):
                    c._pad = int(hit[0])
                    c.nitro = min(1.0, c.nitro + 0.5)
                    c.vx += math.cos(c.ang) * 7.0           # empujon hacia adelante
                    c.vz += math.sin(c.ang) * 7.0
                    if c.is_player:
                        self.show("¡TURBO!", 0.9)

    def build_intro(self):
        super().build_intro()
        if not self.md.concert:
            self.shots = [s for s in self.shots if s[0] != "concert"]
            self.intro_total = sum(s[1] for s in self.shots)

    # ------------------------------------------------------------ entrada
    def handle_key(self, key):
        st = self.state
        if st == "select" or (st == "lobby" and self.mp == "host"):
            if key in (pygame.K_q, pygame.K_e):
                self.map_sel = (self.map_sel + (1 if key == pygame.K_e else -1)) % len(MAPS)
                return True
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.map_sel != self.map_i:
                if st == "lobby" and not (self.net and self.net.peer and self.net.connected()):
                    pass
                else:
                    self.load_map(self.map_sel, keep_state=(st == "lobby"))
                    if st == "select":
                        self.state = "select"
        return super().handle_key(key)

    # ------------------------------------------------------------ interfaz
    def map_panel(self, x, y, can_change=True):
        m = MAPS[self.map_sel]
        self.panel((x, y, 520, 112), 185, accent=ORANGE, slant=20)
        self.txt("PISTA", 16, (170, 185, 200), (x + 40, y + 10), italic=False)
        arrows = len(MAPS) > 1 and can_change
        self.txt(f"‹ {m.name} ›" if arrows else m.name, 30, (255, 255, 255), (x + 40, y + 28))
        km = f"{m.length / 1000:.1f} KM  ·  {m.laps} VUELTA{'S' if m.laps != 1 else ''}" if m.length else f"{m.laps} VUELTAS"
        self.txt(f"{km}  ·  {m.subtitle}", 15, (200, 210, 225), (x + 42, y + 72), italic=False)
        if can_change:
            self.txt(f"Q / E  cambiar pista  ({self.map_sel + 1}/{len(MAPS)})", 14, ORANGE, (x + 500, y + 10), "topright",
                     italic=False)

    def hud_select(self):
        super().hud_select()
        self.map_panel(WIN_W - 580, 36)

    def hud_lobby(self):
        super().hud_lobby()
        self.map_panel(WIN_W - 580, 36, can_change=self.mp == "host")

    def hud_intro(self):
        md = self.md
        shot, lt = self.current_shot()
        if shot is not None and shot[0] == "aerial":
            # mismo intro del juego base, con el nombre de esta pista
            h = self.hud
            bars = int(86 * smooth(self.intro_t / 0.7))
            pygame.draw.rect(h, (0, 0, 0, 255), (0, 0, WIN_W, bars))
            pygame.draw.rect(h, (0, 0, 0, 255), (0, WIN_H - bars, WIN_W, bars))
            a = int(255 * smooth((lt - 0.6) / 0.8))
            self.txt(md.name, 92, (255, 255, 255), (WIN_W // 2, WIN_H // 2 + 120), "center", glow=ACCENT, alpha=a)
            self.txt(f"{self.track.L / 1000:.1f} KM  ·  {md.laps} VUELTA{'S' if md.laps != 1 else ''}  ·  {DIFFS[self.diff][0]}  ·  "
                     f"{TOD_HD[self.tod][0]}", 26, ACCENT, (WIN_W // 2, WIN_H // 2 + 190), "center", alpha=a)
            self.txt("ENTER  saltar", 17, (190, 200, 215), (WIN_W - 24, WIN_H - 34), "topright", italic=False)
            self.song_banner(24)
            return
        if shot is not None and shot[0] == "concert":
            # toma del concierto con el estadio y la banda de esta pista
            h = self.hud
            bars = int(86 * smooth(self.intro_t / 0.7))
            pygame.draw.rect(h, (0, 0, 0, 255), (0, 0, WIN_W, bars))
            pygame.draw.rect(h, (0, 0, 0, 255), (0, WIN_H - bars, WIN_W, bars))
            a = int(255 * smooth((lt - 0.3) / 0.6))
            self.txt(md.venue_name, 30, (255, 255, 255), (WIN_W // 2, WIN_H // 2 + 150), "center", alpha=a)
            s = f"{md.band}  ·  {md.venue_tag}"
            size = 70
            while size > 36 and font(size).size(s)[0] > WIN_W - 80:
                size -= 4
            self.txt(s, size, (255, 70, 50), (WIN_W // 2, WIN_H // 2 + 205), "center", glow=(255, 30, 0), alpha=a)
            self.txt("ENTER  saltar", 17, (190, 200, 215), (WIN_W - 24, WIN_H - 34), "topright", italic=False)
            self.song_banner(24)
            return
        super().hud_intro()

    def song_banner(self, y):
        mu = self.music
        if mu.mode == "playlist" and mu.now and mu.np_t < 7 and not self.sound.muted:
            a = int(255 * hd.clamp(min(mu.np_t / 0.5, (7 - mu.np_t) / 1.0), 0, 1))
            s = f"♪  {self.md.band}  —  {mu.now}" + ("  (8-bit)" if self.md.music_dir == "music" else "")
            w = font(22).size(s)[0] + 60
            self.panel((WIN_W // 2 - w // 2, y, w, 40), int(a * 0.75), accent=(240, 40, 40))
            self.txt(s, 22, (255, 225, 90), (WIN_W // 2 + 6, y + 20), "center", alpha=a)


if __name__ == "__main__":
    GameMaps().run()

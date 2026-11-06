"""
GTC RACING HD - MAPAS

Version HD con varias pistas. Usa todo lo de gtc_racing_hd.py (graficos, autos, musica, multijugador LAN)
sin modificarlo, y agrega:
  - un catalogo de mapas (lista MAPS, abajo)
  - un constructor de escenario automatico para cualquier trazado nuevo
  - selector de pista:  Q / E  en la seleccion de piloto (y en la sala LAN para el anfitrion)

PISTAS
  ESTADIO GNP SEGUROS  Autodromo original con el concierto de Avenged Sevenfold (noche).
  Los otros tres son CALLES REALES (OpenStreetMap): el trazado, el ancho de cada calle, los puentes
  vehiculares por los que se pasa POR ENCIMA, los deprimidos, las subidas y bajadas (altura real SRTM),
  los cruces y los edificios reales donde OSM los tiene (donde no, manzanas a lo largo de la calle real).
  Se choca con fachadas, vallas, parapetos, camellones y postes. Arriba se ve el nombre de la calle.
  ESTADIO FRAY NANO    Fernando Iglesias Calderon, Fray Servando, Circuito Interior/Blvd. Puerto Aereo
                       (junto al aeropuerto), Col. Federal, Calle 47, Viaducto y Av. Morelos. Concierto de
                       Limp Bizkit en el estadio (noche). Musica: carpeta "music_limp_bizkit".
  CHAPULTEPEC          Reforma desde el Museo de Antropologia, el Angel, bajo el Monumento a la Revolucion,
                       Reforma Norte, Calzada de Guadalupe a la Basilica, Insurgentes, Av. Chapultepec y el
                       bosque alrededor del Castillo. Fiesta del Mundial (noche). Musica: "music_mundial".
  SAN LUIS POTOSÍ      Jardin Hidalgo de Soledad, Zaragoza, Autopista 57 (puentes), Salvador Nava, Coronel
                       Romero (Politecnica, Hospital del Nino y la Mujer), Periferico, Tangamanga, Mariano
                       Jimenez, Damian Carmona, Rio Santiago, Acceso Norte, Av. Soledad, Citadina y
                       Av. Hidalgo. Feria de la Enchilada (noche). Musica: "music_feria".
  PARQUE TANGAMANGA II 6.4 km (2 vueltas): sale del Autodromo Tangamanga (pianos en las curvas), recorre los
                       caminos del parque y regresa. Bosque con miles de arboles y publico en vallas en todo
                       el recorrido. Ruta = trazo del usuario calibrado con las glorietas reales.
  Los datos de las calles estan en la carpeta mapas_osm/ (se generan con gtc_mapas_osm.py).
  Datos (c) colaboradores de OpenStreetMap, licencia ODbL.

COMO AGREGAR UNA PISTA NUEVA
  Pista inventada: copia PLANTILLA_DE_MAPA, cambia la clave, el nombre y los puntos del trazado (en metros,
  en el sentido de la carrera, la meta es el primer punto) y agregala a MAPS.
  Pista con calles reales: agrega el circuito en gtc_mapas_osm.py (CIRCUITS), corre
  "python gtc_mapas_osm.py <clave>" y usa builder="osm", osm="<clave>" en su MapDef.

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
                 palette=None, fireworks=False, pads=0, city_margin=750.0, city_step=34.0, detail_dist=400.0,
                 osm=None, datum=None, fan_rate=0.45, fan_density=1.0, forest=0, tropical=None):
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
        self.osm, self.datum = osm, datum
        self.fan_rate, self.fan_density, self.forest = fan_rate, fan_density, forest
        self.tropical = tropical          # dict(town=(x, z, radio) en unidades de 'control')
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

# San Luis Potosi (1 px = 5.5 m): sale del Jardin Hidalgo de Soledad por Zaragoza a la carretera 57 (puentes
# vehiculares), Coronel Romero, periferico al Tangamanga, Salvador Nava, Mariano Jimenez, Damian Carmona,
# Rio Santiago, Acceso Norte y regresa por Av. Soledad / Av. Hidalgo.
SLP_ROUTE = [
    (746, 258), (744, 278), (730, 287), (712, 286), (686, 289), (660, 297), (640, 312), (624, 335), (616, 365),
    (614, 400), (614, 450), (615, 500), (616, 550), (617, 600), (611, 628), (588, 642), (545, 660), (505, 672),
    (478, 667), (460, 659), (444, 657), (434, 666), (430, 684), (429, 705), (431, 735), (425, 756), (398, 761),
    (360, 761), (333, 755), (307, 730), (318, 710), (340, 690), (357, 662), (372, 636), (382, 631), (400, 628),
    (414, 618), (425, 598), (437, 580), (450, 550), (461, 527), (470, 500), (472, 460), (469, 422), (470, 402),
    (488, 393), (530, 390), (575, 388), (640, 382), (660, 352), (680, 320), (700, 287), (716, 255), (722, 235),
    (731, 216), (746, 206), (760, 214), (758, 238)]

CARNAVAL_SHIRTS = [rgb(230, 40, 140), rgb(250, 200, 20), rgb(30, 190, 200), rgb(130, 40, 190), rgb(250, 120, 30),
                   rgb(40, 200, 90), rgb(255, 255, 255)]

# Cozumel (trazo del usuario, escala compacta): centro de San Miguel, costa oriente, Punta Sur y costa poniente
COZUMEL_ROUTE = [(348, 96), (365, 103), (380, 92), (400, 100), (425, 128), (460, 140), (500, 175), (540, 215),
                 (580, 250), (620, 290), (655, 318), (658, 345), (630, 370), (600, 405), (575, 440), (555, 460),
                 (520, 540), (490, 570), (465, 585), (440, 620), (410, 660), (370, 690), (330, 710), (310, 730),
                 (290, 725), (265, 690), (230, 640), (185, 595), (160, 540), (152, 500), (150, 450), (158, 400),
                 (170, 350), (200, 305), (240, 265), (265, 225), (285, 190), (282, 160), (300, 110), (330, 82)]

MAPS = [
    MapDef("cdmx", "ESTADIO GNP SEGUROS", "Autódromo · concierto de Avenged Sevenfold",
           control=base.CONTROL, scale=base.TRACK_SCALE,
           stadium=(base.STADIUM_PTS.start, base.STADIUM_PTS.stop), stadium_scale=base.STADIUM_SCALE,
           concert=True, extras=("angel", "bandera", "guadalupe", "papel"), builder="cdmx"),
    MapDef("fraynano", "ESTADIO FRAY NANO", "Calles reales: Jardín Balbuena · Puerto Aéreo · Viaducto",
           control=FRAY_NANO_ROUTE, scale=4.5, laps=1, concert=True, builder="osm", osm="fraynano",
           datum="fraynano", urban=True, fans=0.9, extras=("papel",),
           venue=dict(name="ESTADIO FRAY NANO", band="LIMP BIZKIT", pos=(345, 372), r0=62, rows=26),
           landmarks=["estadio_concierto", "gnp", "palacio", "aicm"],
           signs={1: ((10, 60, 130), (30, 100, 190), "AEROPUERTO AICM", (255, 255, 255)),
                  2: ((5, 5, 5), (45, 5, 5), "AVENGED SEVENFOLD", (240, 40, 40)),
                  9: ((0, 70, 40), (20, 120, 60), "ESTADIO FRAY NANO", (255, 255, 255)),
                  10: ((5, 5, 5), (70, 30, 0), "LIMP BIZKIT", (255, 120, 20)),
                  14: ((5, 5, 5), (50, 30, 0), "HOY EN VIVO · LIMP BIZKIT", (255, 170, 40))},
           band_sign=10, music_dir="music_limp_bizkit"),
    MapDef("chapultepec", "CHAPULTEPEC", "Reforma · Insurgentes · fiesta del Mundial",
           control=CHAPULTEPEC_ROUTE, scale=4.5, laps=1, concert=True, builder="osm", osm="chapultepec",
           datum="monumento", urban=True, fans=0.85, extras=("banderines",), palette="mexico",
           fireworks=True,
           venue=dict(name="MONUMENTO A LA REVOLUCIÓN", band="FIESTA MUNDIALISTA", tag="¡VAMOS MÉXICO!",
                      pos=(482, 600), r0=55, kind="fanfest"),
           landmarks=["museo", "castillo", "angel", "monumento", "basilica", "fanfest"],
           signs={1: ((0, 104, 71), (0, 140, 90), "MUNDIAL 2026", (255, 255, 255)),
                  2: ((206, 17, 38), (240, 40, 60), "¡GOOOOL!", (255, 255, 255)),
                  9: ((0, 104, 71), (20, 140, 90), "¡VAMOS MÉXICO!", (255, 255, 255)),
                  10: ((10, 10, 10), (0, 80, 50), "FAN FEST CDMX", (255, 215, 40)),
                  12: ((10, 25, 80), (25, 60, 150), "PASEO DE LA REFORMA", (255, 255, 255)),
                  14: ((5, 5, 5), (0, 60, 40), "HOY: FIESTA MUNDIALISTA", (255, 215, 40))},
           band_sign=10, music_dir="music_mundial"),
    MapDef("slp", "SAN LUIS POTOSÍ", "Carretera 57 · Tangamanga · Feria de la Enchilada",
           control=SLP_ROUTE, scale=5.5, laps=1, concert=True, builder="osm", osm="slp", datum="jardin",
           urban=True, fans=0.85, extras=("papel",), fireworks=True,
           venue=dict(name="JARDÍN HIDALGO · SOLEDAD", band="FERIA DE LA ENCHILADA", tag="¡QUE SIGA LA FIESTA!",
                      pos=(815, 300), r0=38, kind="feria"),
           landmarks=["jardin", "feria", ("edificio", "cruzroja", 13, rgb(235, 235, 235)),
                      ("edificio", "politecnica", 14, rgb(120, 150, 200)),
                      ("edificio", "hospital", 12, rgb(225, 185, 205)),
                      ("edificio", "citadina", 9, rgb(230, 225, 215)), "catedral", "cajadelagua"],
           signs={1: ((20, 60, 140), (40, 100, 200), "SAN LUIS POTOSÍ", (255, 255, 255)),
                  2: ((200, 30, 40), (240, 60, 60), "FERIA DE LA ENCHILADA", (255, 220, 60)),
                  7: ((230, 120, 20), (255, 160, 40), "ENCHILADAS POTOSINAS", (255, 255, 255)),
                  9: ((0, 100, 60), (20, 140, 90), "SOLEDAD DE G. S.", (255, 255, 255)),
                  10: ((10, 10, 10), (90, 20, 20), "BAILE DE LA FERIA", (255, 215, 40)),
                  12: ((230, 230, 240), (255, 255, 255), "HOSPITAL DEL NIÑO Y LA MUJER", (180, 40, 120)),
                  13: ((235, 235, 235), (255, 255, 255), "UNIVERSIDAD CRUZ ROJA", (210, 20, 30)),
                  14: ((20, 50, 110), (40, 80, 160), "UNIVERSIDAD POLITÉCNICA", (255, 255, 255))},
           band_sign=10, music_dir="music_feria"),
    MapDef("tangamanga", "PARQUE TANGAMANGA II", "San Luis Potosí · autódromo y bosque",
           control=SLP_ROUTE, scale=5.5, laps=2, concert=False, builder="osm", osm="tangamanga", urban=True,
           ground="bosque", volcanoes=False, fans=1.0, fan_rate=0.95, fan_density=1.6, forest=3500,
           extras=("tribunas",),
           signs={1: ((0, 100, 60), (20, 140, 80), "PARQUE TANGAMANGA II", (255, 255, 255)),
                  9: ((20, 20, 24), (60, 60, 70), "AUTÓDROMO TANGAMANGA", (255, 215, 40)),
                  2: ((200, 30, 40), (240, 60, 60), "¡VAMOS SAN LUIS!", (255, 255, 255))},
           music_dir="music_feria"),
    MapDef("segundopiso", "PERIFÉRICO · SEGUNDO PISO", "CDMX · Periférico Sur abajo y el Segundo Piso arriba",
           control=SLP_ROUTE, scale=5.5, laps=1, concert=False, builder="osm", osm="segundopiso", urban=True,
           fans=0.8, extras=("papel",), fireworks=True,
           signs={1: ((20, 40, 90), (40, 80, 160), "SEGUNDO PISO", (255, 255, 255)),
                  2: ((10, 90, 50), (30, 140, 80), "PERIFÉRICO SUR", (255, 255, 255)),
                  9: ((200, 0, 110), (240, 40, 150), "CDMX", (255, 255, 255))},
           music_dir="music"),
    MapDef("cozumel", "ISLA DE COZUMEL", "Caribe mexicano · vuelta a la isla real · selva, mar y carnaval",
           control=COZUMEL_ROUTE, scale=6.5, laps=1, concert=True, builder="osm", osm="cozumel", urban=True,
           fans=0.9, extras=("papel",), fireworks=True, volcanoes=False, mountains=False,
           tropical=dict(lm="plaza", r=2600.0), palette=CARNAVAL_SHIRTS,
           venue=dict(name="SAN MIGUEL DE COZUMEL", band="CARNAVAL DE COZUMEL", tag="¡A BAILAR!",
                      lm="start", at=90, side=1, r0=45, kind="fanfest", carnaval=True),
           landmarks=["fanfest"],
           signs={1: ((0, 150, 170), (40, 200, 210), "COZUMEL", (255, 255, 255)),
                  9: ((230, 60, 140), (250, 120, 40), "¡VIVA EL CARNAVAL!", (255, 255, 255)),
                  10: ((120, 20, 160), (230, 60, 140), "CARNAVAL DE COZUMEL", (255, 230, 60)),
                  14: ((10, 10, 10), (120, 20, 160), "HOY: GRAN DESFILE", (255, 230, 60))},
           band_sign=10, music_dir="music_feria"),
    # --- aqui van las pistas nuevas ---
]
# los mapas reales cuyo archivo todavia no se genera (python gtc_mapas_osm.py <mapa>) no aparecen en el selector
_faltan = [m for m in MAPS if m.osm and not os.path.exists(os.path.join(hd.HERE, "mapas_osm", m.osm + ".json.gz"))]
for _m in _faltan:
    print(f"(aviso) falta mapas_osm/{_m.osm}.json.gz: el mapa '{_m.name}' se omite "
          f"(generalo con: python gtc_mapas_osm.py {_m.osm})")
MAPS = [m for m in MAPS if m not in _faltan]

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
        self._setup(base.centripetal_cr(ctrl, base.SPACING))

    def _setup(self, pts):
        md = self.md
        self.pts = pts
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
        fcyl(g, p[0], p[1], h, h + 1.4, 0.62, rgb(22, 22, 24), mat(M_DEF, 0.6), seg=6)
    if rnd.random() < 0.18:                                                    # cuarto de azotea
        p = c + o * rnd.uniform(-hdp * 0.4, hdp * 0.4)
        frame_box(g, p, f, (min(2.5, hl * 0.4), 2.0), (0.9, 0.88, 0.84), mat(M_CONC, 0.8), h, h + 2.6)
    if roof_fans:                                                              # el desmadre: gente en la azotea
        for _ in range(rnd.randint(3, 8)):
            p = F + f * rnd.uniform(-hl + 0.8, hl - 0.8) + o * rnd.uniform(0.6, 2.2)
            crowd.add(p[0], h, p[1], -o[0], -o[1], rnd)


def palm_hd(g, x, z, s, rnd, lean=None):
    """Palmera: tronco curvo e inclinado y penacho de hojas."""
    a = rnd.uniform(0, TAU) if lean is None else lean
    dx, dz = math.cos(a), math.sin(a)
    pts_ = [(x + dx * (t * t) * 2.2 * s, t * 8.5 * s, z + dz * (t * t) * 2.2 * s) for t in np.linspace(0, 1, 4)]
    for (x0, y0, z0), (x1, y1, z1) in zip(pts_[:-1], pts_[1:]):
        cylinder(g, (x0, y0, z0), (x1, y1, z1), 0.22 * s, rgb(130, 105, 75), mat(M_BARK, 0.9), seg=4, caps=False)
    tx, ty, tz = pts_[-1]
    leaf = rgb(*[int(v * rnd.uniform(0.85, 1.1)) for v in (40, 120, 45)])
    for k in range(6):
        b = k / 6 * TAU + rnd.uniform(-0.2, 0.2)
        ux, uz = math.cos(b), math.sin(b)
        px, pz = -uz, ux
        L = 3.6 * s
        mx, my, mz = tx + ux * L * 0.55, ty + 0.5 * s, tz + uz * L * 0.55
        ex, ey, ez = tx + ux * L, ty - 1.6 * s, tz + uz * L
        w = 0.55 * s
        quad(g, (tx, ty, tz), (mx + px * w, my, mz + pz * w), (ex, ey, ez), (mx - px * w, my, mz - pz * w), leaf,
             mat(M_LEAF, 0.8))
    fbox(g, (tx, tz), (1, 0), 0.35 * s, 0.35 * s, ty - 0.7 * s, ty, rgb(110, 80, 40), mat(M_DEF, 0.7))


def jungle_tree(g, x, z, s, rnd):
    """Arbol de selva baja (chacah, chicozapote, ceiba): copa ancha y verde intenso."""
    r_ = rnd.random()
    leaf = rgb(*[int(v * rnd.uniform(0.8, 1.15)) for v in ((28, 92, 34) if r_ < 0.55 else (52, 118, 38))])
    if r_ < 0.15:                                                         # ceiba alta
        fcyl(g, x, z, 0, 9.5 * s, 0.4 * s, rgb(150, 140, 120), mat(M_BARK, 0.9), seg=4, r1=0.28 * s, cap=False)
        sphere(g, (x, 10.5 * s, z), 4.2 * s, leaf, mat(M_LEAF, 0.8), seg=6, rings=3, scale=(1, 0.42, 1))
    else:
        fcyl(g, x, z, 0, 2.8 * s, 0.24 * s, rgb(96, 76, 56), mat(M_BARK, 0.9), seg=4, r1=0.16 * s, cap=False)
        sphere(g, (x, 3.9 * s, z), 2.7 * s, leaf, mat(M_LEAF, 0.8), seg=6, rings=3, scale=(1, 0.78, 1))


def trop_tree(g, x, z, s, rnd):
    """Arbol de camellon o parque en el Caribe: palmera o arbol de selva."""
    if rnd.random() < 0.6:
        palm_hd(g, x, z, s * 0.9, rnd)
    else:
        jungle_tree(g, x, z, s, rnd)


def palapa_hd(g, x, z, rnd):
    """Sombrilla de palma en la playa."""
    fcyl(g, x, z, 0, 2.6, 0.08, rgb(120, 95, 65), mat(M_BARK, 0.9), seg=5)
    fcyl(g, x, z, 2.3, 3.4, 2.1, rgb(196, 160, 96), mat(M_GRAVEL, 0.95), seg=9, r1=0.1)


def cruise_ship_hd(g, c, f, L, y0, lights):
    """Crucero atracado frente a San Miguel."""
    f = np.asarray(f, float) / np.linalg.norm(f)
    o = np.array([-f[1], f[0]])
    W = L * 0.13
    white, blue = rgb(245, 246, 248), rgb(20, 50, 120)
    fbox(g, c, f, L / 2, W / 2, y0 - 4, y0 + 10, white, mat(M_CONC, 0.5), bottom=True)
    fbox(g, c + f * (L / 2 + 9), f, 9, W * 0.3, y0 - 4, y0 + 10, white, mat(M_CONC, 0.5), bottom=True)
    fbox(g, c, f, L / 2 + 0.1, W / 2 + 0.1, y0 + 1, y0 + 3, blue, mat(M_CONC, 0.5))
    for k in range(7):
        fbox(g, c - f * (L * 0.03 + k * 2), f, L / 2 * (0.9 - 0.04 * k), W / 2 * 0.94, y0 + 10 + k * 3.1,
             y0 + 13.1 + k * 3.1, white if k % 2 == 0 else rgb(150, 190, 225), mat(M_WINDOWS, 0.4, 0.2))
    fbox(g, c - f * L * 0.28, f, 9, W * 0.22, y0 + 32, y0 + 42, rgb(200, 30, 40), mat(M_CONC, 0.5))
    for k in range(-3, 4):
        q = c + f * (k * L / 8)
        lights.append((float(q[0]), y0 + 24, float(q[1]), 60.0, 1.0, 0.9, 0.7, 3.0))


def faro_hd(g, c, y0, lights):
    """Faro Celarain (Punta Sur)."""
    fbox(g, c, (1, 0), 6, 4, y0, y0 + 4, rgb(240, 236, 228), mat(M_CONC, 0.7), bottom=True)
    fcyl(g, c[0], c[1], y0, y0 + 22, 2.2, rgb(245, 245, 245), mat(M_CONC, 0.6), seg=12, r1=1.6)
    fcyl(g, c[0], c[1], y0 + 13, y0 + 16, 1.95, rgb(200, 40, 40), mat(M_CONC, 0.6), seg=12, r1=1.85)
    fcyl(g, c[0], c[1], y0 + 22, y0 + 25, 1.3, (1.0, 0.9, 0.6), mat(M_LAMP, 0.4, 0, 9.0), seg=10)
    fcyl(g, c[0], c[1], y0 + 25, y0 + 27, 1.6, rgb(200, 40, 40), mat(M_CONC, 0.6), seg=10, r1=0.2)
    lights.append((float(c[0]), y0 + 24, float(c[1]), 90.0, 1.0, 0.9, 0.6, 5.0))


def _steps_to(M, K):
    """Pasos (de celda) hasta la celda True mas cercana, con tope K (distancia octagonal aprox.)."""
    D = np.full(M.shape, K, np.int16)
    cur = M.copy()
    D[cur] = 0
    for k in range(1, K):
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        if k % 2 == 0:
            nxt[1:, 1:] |= cur[:-1, :-1]
            nxt[:-1, :-1] |= cur[1:, 1:]
            nxt[1:, :-1] |= cur[:-1, 1:]
            nxt[:-1, 1:] |= cur[1:, :-1]
        D[nxt & ~cur] = k
        cur = nxt
        if cur.all():
            break
    return D


class Coast:
    """Tierra y mar a partir de la linea de costa de OSM: distancia con signo a la orilla (+ tierra, - mar)."""

    def __init__(self, polys, lo, hi, res=25.0, K=40):
        self.res, self.c0 = res, np.asarray(lo, float)
        nx, nz = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
        L = np.zeros((nx, nz), bool)
        for P in polys:
            A, B = P, np.roll(P, -1, 0)
            for j in range(nz):                                           # relleno por renglones
                z = lo[1] + (j + 0.5) * res
                c = (A[:, 1] > z) != (B[:, 1] > z)
                if not c.any():
                    continue
                xi = np.sort(A[c, 0] + (z - A[c, 1]) * (B[c, 0] - A[c, 0]) / (B[c, 1] - A[c, 1]))
                for a, b in zip(xi[0::2], xi[1::2]):
                    i0 = max(0, int(math.ceil((a - lo[0]) / res - 0.5)))
                    i1 = min(nx - 1, int(math.floor((b - lo[0]) / res - 0.5)))
                    if i1 >= i0:
                        L[i0:i1 + 1, j] ^= True
        self.land = L
        self.D = np.where(L, _steps_to(~L, K), -_steps_to(L, K)).astype(np.float32) * res
        self.lo_c = np.argwhere(L).min(0) * res + self.c0 if L.any() else self.c0
        self.hi_c = np.argwhere(L).max(0) * res + self.c0 if L.any() else self.c0

    def ds(self, Q):
        Q = np.asarray(Q, float).reshape(-1, 2)
        fx = np.clip((Q[:, 0] - self.c0[0]) / self.res - 0.5, 0, self.D.shape[0] - 1.001)
        fz = np.clip((Q[:, 1] - self.c0[1]) / self.res - 0.5, 0, self.D.shape[1] - 1.001)
        ix, iz = fx.astype(int), fz.astype(int)
        tx, tz = fx - ix, fz - iz
        G = self.D
        return ((G[ix, iz] * (1 - tx) + G[ix + 1, iz] * tx) * (1 - tz) +
                (G[ix, iz + 1] * (1 - tx) + G[ix + 1, iz + 1] * tx) * tz)


def masked_grid(g, P, C, cells, m):
    """Como grid() pero solo dibuja las celdas marcadas (P: (R, Cn, 3), cells: (R-1, Cn-1) bool)."""
    P = np.asarray(P, F4)
    dR, dC = np.gradient(P, axis=0), np.gradient(P, axis=1)
    N = np.cross(dC, dR)
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    N = np.where((N[..., 1] < 0)[..., None], -N, N)
    i, j = np.nonzero(cells)
    ir = np.stack([i, i + 1, i + 1, i, i + 1, i], 1).ravel()
    jc = np.stack([j, j, j + 1, j, j + 1, j + 1], 1).ravel()
    V = P[ir, jc]
    g.tris(V, N[ir, jc], np.asarray(C, F4)[ir, jc], m, V[:, [0, 2]] / 8.0)


def forest_tree(g, x, z, s, rnd):
    """Arbol de bosque ligero (pocos triangulos: se siembran miles)."""
    fcyl(g, x, z, 0, 2.4 * s, 0.22 * s, rgb(84, 60, 40), mat(M_BARK, 0.9), seg=5, r1=0.15 * s, cap=False)
    r_ = rnd.random()
    if r_ < 0.45:                                                         # pino
        leaf = rgb(*[int(v * rnd.uniform(0.8, 1.15)) for v in (34, 88, 42)])
        fcyl(g, x, z, 1.6 * s, 6.5 * s, 2.3 * s, leaf, mat(M_LEAF, 0.8), seg=7, r1=0.15 * s)
        fcyl(g, x, z, 4.6 * s, 8.6 * s, 1.6 * s, leaf, mat(M_LEAF, 0.8), seg=7, r1=0.05 * s)
    elif r_ < 0.6:                                                        # jacaranda
        col = rgb(*[int(v * rnd.uniform(0.88, 1.1)) for v in (150, 100, 205)])
        sphere(g, (x, 3.8 * s, z), 2.3 * s, col, mat(M_LEAF, 0.8), seg=7, rings=4, scale=(1, 0.75, 1))
    else:                                                                 # mezquite / fresno
        leaf = rgb(*[int(v * rnd.uniform(0.8, 1.15)) for v in (62, 110, 46)])
        sphere(g, (x, 3.6 * s, z), 2.6 * s, leaf, mat(M_LEAF, 0.8), seg=7, rings=4, scale=(1, 0.8, 1))


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


def ped_bridge_hd(g, crowd, rnd, track, i, k_ad, lights, occ, half=None):
    """Puente peatonal CDMX: escaleras a los lados, publicidad, techito y gente viendo la carrera."""
    p, nr, t = track.pts[i], track.nrm[i], track.tan[i]
    half = LIMIT + 6.0 if half is None else half
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
    for d in (-half + 0.6, -(half - 4.6), half - 4.6, half - 0.6):              # pilares
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
    p = _flight_pose(fl, t)
    if p is None:
        return None
    return (p[0], p[1] + fl.get("y0", 0.0), p[2], p[3], p[4])


def _flight_pose(fl, t):
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
    rw_ = getattr(track, "rw", None)
    for _ in range(12):                                       # pilares (7 m) fuera de la calle
        ok = True
        for a in (-1, 1):
            for b in (-1, 1):
                c = p + t * a * (H - 4) + nr * b * (H - 4)
                d = np.hypot(track.pts[:, 0] - c[0], track.pts[:, 1] - c[1])
                lim = (rw_ if rw_ is not None else np.full(track.n, HW)) + 1.6 + 7.2
                if (d < lim).any():
                    ok = False
        if ok:
            break
        H += 2.0
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


def carroza_hd(g, c, f, k, lights):
    """Carro alegorico del Carnaval de Cozumel: plataforma de colores, mascara gigante y abanicos de plumas."""
    f = v2(f)
    o = np.array([-f[1], f[0]])
    cols = [(rgb(230, 40, 140), rgb(250, 200, 20), rgb(30, 190, 200)),
            (rgb(130, 40, 190), rgb(250, 120, 30), rgb(40, 200, 90))][k % 2]
    for a in (-1, 1):                                                          # llantas
        for b in (-1, 1):
            q = c + f * a * 5.0 + o * b * 2.6
            fcyl(g, q[0], q[1], 0, 1.0, 0.5, rgb(20, 20, 22), mat(M_DEF, 0.6), seg=8)
    fbox(g, c, f, 7.0, 3.0, 0.6, 2.0, cols[0], mat(M_DEF, 0.5, 0.1), bottom=True)
    fbox(g, c - f * 1.0, f, 5.5, 2.6, 2.0, 3.4, cols[1], mat(M_DEF, 0.5, 0.1))
    fbox(g, c - f * 2.0, f, 3.5, 2.2, 3.4, 4.6, cols[2], mat(M_DEF, 0.5, 0.1))
    m = c + f * 3.0                                                            # mascara
    sphere(g, (m[0], 6.8, m[1]), 2.6, cols[1], mat(M_DEF, 0.35, 0.3), seg=12, rings=8, scale=(1, 1.15, 0.8))
    for sg in (-1, 1):
        q = m + o * sg * 0.95 + f * 1.9
        sphere(g, (q[0], 7.4, q[1]), 0.55, (1, 1, 1), mat(M_EMIS, 0.4, 0, 3.0), seg=8, rings=5)
    for j in range(9):                                                         # plumas
        a = (j - 4) / 4 * 1.2
        tip = m + o * math.sin(a) * 6.5 - f * 0.8
        quad(g, (m[0] - o[0] * 0.5, 8.0, m[1] - o[1] * 0.5), (m[0] + o[0] * 0.5, 8.0, m[1] + o[1] * 0.5),
             (tip[0] + o[0] * 0.9, 8.0 + math.cos(a) * 7.5, tip[1] + o[1] * 0.9),
             (tip[0] - o[0] * 0.9, 8.0 + math.cos(a) * 7.5, tip[1] - o[1] * 0.9), cols[j % 3], mat(M_EMIS, 0.5, 0, 1.6))
    lights.append((float(m[0]), 10.0, float(m[1]), 30.0, 1.0, 0.4, 0.8, 3.0))


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
    if (md.venue or {}).get("carnaval"):                                      # carros alegoricos y palmeras
        for k_, q in enumerate((tq, bq)):
            carroza_hd(g, q, rr if k_ else -rr, k_, lights)
        for k in range(10):
            a = k / 10 * TAU + 0.3
            q = sc + np.array([math.cos(a), math.sin(a)]) * (r0 + 1)
            palm_hd(g, q[0], q[1], rnd.uniform(0.9, 1.15), rnd)
    else:
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


# ---------------------------------------------------------------- San Luis Potosi: Feria de la Enchilada
CANTERA = rgb(196, 140, 122)                 # cantera rosa potosina


def fit_spot(track, hint, clear, zones=None, search=320.0, step=12.0):
    """El punto mas cercano a 'hint' que quede a por lo menos 'clear' metros de la pista (y fuera de zonas)."""
    h = v2(hint)
    xs = np.arange(-search, search + 0.1, step)
    G = np.array([(h[0] + a, h[1] + b) for a in xs for b in xs])
    ok = track.nearest_dist(G) >= clear
    if zones is not None:
        ok &= ~zones.hit(G)
    if not ok.any():
        return beside_track(track, hint, clear)
    cand = G[ok]
    return cand[int(np.argmin(np.hypot(cand[:, 0] - h[0], cand[:, 1] - h[1])))]


def rueda_geo(R):
    """Rueda de la fortuna (gira alrededor del eje z local; el centro esta en el origen)."""
    g = Geo()
    steel, white = rgb(230, 232, 238), rgb(250, 250, 250)
    sm = mat(M_METAL, 0.35, 0.6)
    a = np.linspace(0, TAU, 33)
    for z in (-1.6, 1.6):
        for k in range(32):
            cylinder(g, (R * math.cos(a[k]), R * math.sin(a[k]), z), (R * math.cos(a[k + 1]), R * math.sin(a[k + 1]), z),
                     0.22, steel, sm, seg=6, caps=False)
        for k in range(16):
            cylinder(g, (0, 0, z), (R * math.cos(a[2 * k]), R * math.sin(a[2 * k]), z), 0.1, white, sm, seg=5,
                     caps=False)
    cylinder(g, (0, 0, -2.2), (0, 0, 2.2), 0.8, rgb(90, 95, 105), sm, seg=12)
    cols = [(1.0, 0.25, 0.2), (1.0, 0.85, 0.2), (0.3, 0.6, 1.0), (0.3, 1.0, 0.4), (1.0, 0.3, 0.9)]
    for k in range(64):                                                         # foquitos del aro
        q = (R + 0.3) * math.cos(k / 64 * TAU), (R + 0.3) * math.sin(k / 64 * TAU)
        for z in (-1.75, 1.75):
            box(g, q[0] - 0.18, q[1] - 0.18, z - 0.12, q[0] + 0.18, q[1] + 0.18, z + 0.12, cols[k % 5],
                mat(M_LAMP, 0.4, 0, 9.0))
    return g


def gondola_geo():
    """Canastilla de la rueda (cuelga del punto de giro, siempre derecha)."""
    g = Geo()
    fbox(g, (0, 0), (1, 0), 1.3, 1.1, -3.0, -1.0, rgb(220, 40, 50), mat(M_PAINT, 0.35), bottom=True)
    fbox(g, (0, 0), (1, 0), 1.45, 1.25, -1.0, -0.8, rgb(250, 210, 40), mat(M_PAINT, 0.35), bottom=True)
    cylinder(g, (0, -0.8, 0), (0, 0, 0), 0.08, rgb(200, 200, 205), mat(M_METAL, 0.4, 0.8), seg=5, caps=False)
    box(g, -0.2, -1.15, -0.2, 0.2, -0.95, 0.2, (1.0, 0.9, 0.6), mat(M_LAMP, 0.4, 0, 6.0))
    return g


def carrusel_geo():
    """Carrusel (gira alrededor de y)."""
    g = Geo()
    fcyl(g, 0, 0, 0, 0.8, 9.5, rgb(200, 40, 60), mat(M_PAINT, 0.4), seg=24)
    fcyl(g, 0, 0, 0.8, 7.5, 1.2, rgb(240, 200, 60), mat(M_METAL, 0.3, 0.8), seg=12, cap=False)
    revolve(g, [(10.0, 6.8), (9.6, 7.4), (6.0, 8.6), (1.0, 10.2), (0.1, 10.5)], (0, 0, 0), (0, 1, 0),
            rgb(250, 250, 250), mat(M_DEF, 0.6), seg=16)
    cols = [rgb(250, 250, 250), rgb(40, 40, 45), rgb(150, 90, 50), rgb(240, 200, 60)]
    for k in range(16):
        a = k / 16 * TAU
        for rr in (5.0, 7.8):
            x, z = rr * math.cos(a + rr), rr * math.sin(a + rr)
            fcyl(g, x, z, 0.8, 7.0, 0.06, rgb(240, 200, 60), mat(M_METAL, 0.3, 1.0), seg=5, cap=False)
            y = 2.2 + 0.4 * math.sin(k * 1.7 + rr)
            fbox(g, (x, z), (-math.sin(a), math.cos(a)), 0.9, 0.22, y, y + 0.7, cols[k % 4], mat(M_PAINT, 0.4),
                 bottom=True)
            fbox(g, (x + 0.7 * -math.sin(a), z + 0.7 * math.cos(a)), (-math.sin(a), math.cos(a)), 0.2, 0.15, y + 0.5,
                 y + 1.3, cols[k % 4], mat(M_PAINT, 0.4), bottom=True)
        q = (9.8 * math.cos(a), 9.8 * math.sin(a))
        box(g, q[0] - 0.2, 6.9, q[1] - 0.2, q[0] + 0.2, 7.3, q[1] + 0.2, (1.0, 0.85, 0.5), mat(M_LAMP, 0.4, 0, 8.0))
    return g


def foquitos_hd(g, pa, pb, h, rnd, sag=1.2):
    """Serie de foquitos de colores colgando entre dos postes."""
    cols = [(1.0, 0.85, 0.5), (1.0, 0.3, 0.25), (0.3, 1.0, 0.4), (0.35, 0.55, 1.0), (1.0, 0.9, 0.2)]
    ln = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
    for k in range(int(ln / 1.4)):
        t = (k + 0.5) / int(ln / 1.4)
        x, z = pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t
        y = h - sag * 4 * t * (1 - t)
        fbox(g, (x, z), (1, 0), 0.13, 0.13, y - 0.2, y, cols[k % 5], mat(M_LAMP, 0.4, 0, 8.0), bottom=True)


def feria_hd(g, gc, crowd, rnd, track, sc, r0, md, info, lights):
    """Feria de la Enchilada: escenario (noche), rueda de la fortuna, carrusel, puestos, foquitos y castillo."""
    R = r0 + 62
    ang = np.linspace(0, TAU, 65)
    for j in range(64):
        quad(g, (sc[0], 0.03, sc[1]), (sc[0], 0.03, sc[1]),
             (sc[0] + R * math.cos(ang[j + 1]), 0.03, sc[1] + R * math.sin(ang[j + 1])),
             (sc[0] + R * math.cos(ang[j]), 0.03, sc[1] + R * math.sin(ang[j])), rgb(168, 150, 124), mat(M_CONC, 0.9))
    i_n = int(np.argmin(np.hypot(track.pts[:, 0] - sc[0], track.pts[:, 1] - sc[1])))
    tt = track.pts[i_n] - sc
    tt = tt / np.linalg.norm(tt)
    base_a = math.atan2(tt[1], tt[0])
    fe = sc + tt * (R - 2)                                                     # arco de entrada
    rr = np.array([-tt[1], tt[0]])
    for sgn in (-1, 1):
        q = fe + rr * sgn * 15
        fbox(g, q, tt, 1.2, 1.2, 0, 14, rgb(200, 40, 60), mat(M_PAINT, 0.4))
    fbox(g, fe, rr, 16.5, 1.0, 12, 18, rgb(250, 200, 40), mat(M_PAINT, 0.4), bottom=True)
    sign_quad(g, fe + tt * 1.05, tt, 14, 12.4, 17.6, 2)
    sign_quad(g, fe - tt * 1.05, -tt, 14, 12.4, 17.6, 2)
    spinners = []
    # rueda de la fortuna y carrusel
    a1 = base_a + 2.2
    wc = sc + np.array([math.cos(a1), math.sin(a1)]) * (r0 + 30)
    Rw, Hw = 20.0, 23.0
    yaw = a1 + math.pi / 2
    fx, fz = math.cos(yaw), math.sin(yaw)
    for zs in (-3.2, 3.2):                                                     # patas en A
        px, pz = wc[0] - fz * zs, wc[1] + fx * zs
        for sgn in (-1, 1):
            cylinder(g, (px + fx * sgn * 9, 0, pz + fz * sgn * 9), (px, Hw, pz), 0.45, rgb(230, 232, 238),
                     mat(M_METAL, 0.35, 0.6), seg=8)
    spinners.append(dict(kind="rueda", c=(float(wc[0]), Hw, float(wc[1])), yaw=yaw, R=Rw, w=0.12))
    lights.append((float(wc[0]), Hw, float(wc[1]), 55.0, 1.0, 0.6, 0.8, 4.0))
    a2 = base_a - 2.0
    cc = sc + np.array([math.cos(a2), math.sin(a2)]) * (r0 + 26)
    spinners.append(dict(kind="carrusel", c=(float(cc[0]), 0.0, float(cc[1])), yaw=0.0, R=10.0, w=0.5))
    lights.append((float(cc[0]), 7.0, float(cc[1]), 35.0, 1.0, 0.85, 0.5, 3.5))
    # castillo pirotecnico
    a3 = base_a + math.pi
    kc = sc + np.array([math.cos(a3), math.sin(a3)]) * (r0 + 40)
    fcyl(g, kc[0], kc[1], 0, 24, 0.35, rgb(110, 80, 50), mat(M_DEF, 0.8), seg=6)
    for k in range(5):
        y = 5 + k * 4.2
        fbox(g, kc, (1, 0), 3.5 - k * 0.5, 0.15, y, y + 0.25, rgb(110, 80, 50), mat(M_DEF, 0.8), bottom=True)
        fbox(g, kc, (0, 1), 3.5 - k * 0.5, 0.15, y, y + 0.25, rgb(110, 80, 50), mat(M_DEF, 0.8), bottom=True)
    castillo_pts = [(float(kc[0] + dx), 6 + k * 4.2, float(kc[1] + dz)) for k in range(5)
                    for dx, dz in ((3.2 - k * 0.5, 0), (-3.2 + k * 0.5, 0))]
    # puestos de enchiladas en anillos + foquitos
    for ring, cnt in ((r0 + 10, 30), (r0 + 50, 44)):
        for k in range(cnt):
            a = k / cnt * TAU + rnd.uniform(-0.02, 0.02)
            if abs(((a - base_a + math.pi) % TAU) - math.pi) < 0.16:             # pasillo de la entrada
                continue
            q = sc + np.array([math.cos(a), math.sin(a)]) * ring
            if np.hypot(*(q - wc)) < 26 or np.hypot(*(q - cc)) < 14 or np.hypot(*(q - kc)) < 8:
                continue
            puesto_hd(g, q, (-math.sin(a), math.cos(a)), rnd)
    for k in range(18):                                                        # postes y foquitos radiales
        a = k / 18 * TAU
        p0 = sc + np.array([math.cos(a), math.sin(a)]) * (r0 + 4)
        p1 = sc + np.array([math.cos(a), math.sin(a)]) * (R - 6)
        for q in (p0, p1):
            fcyl(g, q[0], q[1], 0, 8, 0.12, rgb(90, 90, 95), mat(M_METAL, 0.4, 0.7), seg=6)
        foquitos_hd(g, p0, p1, 7.8, rnd)
        q2 = sc + np.array([math.cos(a + TAU / 18), math.sin(a + TAU / 18)]) * (R - 6)
        foquitos_hd(g, p1, q2, 7.8, rnd, sag=0.8)
        if k % 3 == 0:
            lights.append((float(p1[0]), 6.0, float(p1[1]), 30.0, 1.0, 0.7, 0.4, 2.5))
    # gente de la feria (dia y noche) + baile (noche)
    for _ in range(1600):
        a, rr_ = rnd.uniform(0, TAU), rnd.uniform(r0 + 3, R - 3)
        x, z = sc[0] + rr_ * math.cos(a), sc[1] + rr_ * math.sin(a)
        if np.hypot(x - wc[0], z - wc[1]) < 12 or np.hypot(x - cc[0], z - cc[1]) < 10.5:
            continue
        crowd.add(x, 0.0, z, rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd)
    if md.concert:
        build_concert_hd(gc, crowd, rnd, track, sc, r0, r0 + 4, 9.0, info)
        info["fountain"] = list(info.get("fountain", [])) + castillo_pts
        info["launch"] = list(info.get("launch", [])) + [(float(kc[0]), 25.0, float(kc[1]))] * 4
    info["stadium"] = (sc, R, r0)
    info["spinners"] = spinners


def puente_vehicular_hd(g, track, i, lights, occ, rnd):
    """Puente vehicular potosino sobre la pista: losa, VIGAS a la vista, cabezales, columnas y rampas."""
    p, t, nr = track.pts[i], track.tan[i], track.nrm[i]
    conc, conc2 = rgb(178, 176, 170), rgb(150, 148, 142)
    mc = mat(M_CONC, 0.8)
    W = 7.5                                                                    # medio ancho del puente (a lo largo de la pista)
    span = LIMIT + 3.0
    ramp = {}
    for sgn in (-1, 1):                                                        # largo de las rampas segun el espacio
        ln = 70.0
        while ln > 20 and track.nearest_dist(np.array([p + nr * sgn * (span + ln)]))[0] < LIMIT + 4:
            ln -= 10
        ramp[sgn] = ln
    fbox(g, p, nr, span, W, 7.0, 7.7, conc, mc, bottom=True)                   # losa
    for sgn in (-1, 1):                                                        # parapetos y letreros
        q = p + t * sgn * (W - 0.2)
        fbox(g, q, nr, span, 0.25, 7.7, 8.7, conc2, mc)
        face_quad(g, p + t * sgn * (W + 0.02) - nr * span, p + t * sgn * (W + 0.02) + nr * span, 6.95, 7.75,
                  rgb(40, 110, 60), mat(M_CONC, 0.7), t * sgn)
        sign_quad(g, p + t * sgn * (W + 0.05), t * sgn, 9.5, 6.95, 7.75, [1, 9, 2, 13][(i // 7 + sgn) % 4])
    for k in range(6):                                                         # vigas (trabes AASHTO)
        tz = -W + 1.0 + k * (2 * W - 2.0) / 5
        c = p + t * tz
        fbox(g, c, nr, span + 1.5, 0.18, 5.75, 7.0, conc2, mc, bottom=True)
        fbox(g, c, nr, span + 1.5, 0.45, 5.55, 5.8, conc2, mc, bottom=True)
        fbox(g, c, nr, span + 1.5, 0.35, 6.85, 7.0, conc2, mc, bottom=True)
    for sgn in (-1, 1):                                                        # cabezal + columnas
        q = p + nr * sgn * (LIMIT + 1.6)
        fbox(g, q, t, W + 0.5, 1.0, 4.4, 5.55, conc, mc, bottom=True)
        for tz in (-W + 1.5, 0.0, W - 1.5):
            c = q + t * tz
            fcyl(g, c[0], c[1], 0, 4.4, 0.65, conc, mc, seg=12, cap=False)
        # rampa (terraplen con muros)
        ln = ramp[sgn]
        a0, a1 = p + nr * sgn * span, p + nr * sgn * (span + ln)
        P = [(a0 - t * W), (a0 + t * W), (a1 + t * W), (a1 - t * W)]
        quad_dir(g, (P[0][0], 7.7, P[0][1]), (P[1][0], 7.7, P[1][1]), (P[2][0], 0.1, P[2][1]),
                 (P[3][0], 0.1, P[3][1]), rgb(70, 70, 74), mat(M_CONC, 0.75), (0, 1, 0))
        for side in (-1, 1):
            b0, b1 = a0 + t * side * W, a1 + t * side * W
            quad_dir(g, (b0[0], 0, b0[1]), (b1[0], 0, b1[1]), (b1[0], 0.1, b1[1]), (b0[0], 7.7, b0[1]), conc2,
                     mat(M_CONC, 0.85), (t[0] * side, 0, t[1] * side))
        q = p + nr * sgn * (span + ln * 0.35)
        lights.append((float(q[0]), 9.0, float(q[1]), 30.0, 1.0, 0.75, 0.45, 2.5))
        occ.rect(p + nr * sgn * (span + ln / 2), nr, ln / 2 + 2, W + 2)
    for sgn in (-0.5, 0.5):                                                    # postes sobre el puente
        q = p + nr * sgn * span
        for s2 in (-1, 1):
            c = q + t * s2 * (W - 0.4)
            fcyl(g, c[0], c[1], 7.7, 15.0, 0.12, rgb(85, 92, 98), mat(M_METAL, 0.45, 0.8), seg=6)
            box(g, c[0] - 0.5, 14.8, c[1] - 0.5, c[0] + 0.5, 15.0, c[1] + 0.5, (1.0, 0.78, 0.45), mat(M_LAMP, 0.5, 0, 7.0))
    lights.append((float(p[0]), 13.0, float(p[1]), 35.0, 1.0, 0.78, 0.45, 3.0))


def jardin_hd(g, crowd, rnd, c, r, f, lights):
    """Jardin Hidalgo de Soledad: plaza, kiosco, jacarandas y la parroquia."""
    ang = np.linspace(0, TAU, 49)
    for j in range(48):
        quad(g, (c[0], 0.05, c[1]), (c[0], 0.05, c[1]), (c[0] + r * math.cos(ang[j + 1]), 0.05, c[1] + r * math.sin(ang[j + 1])),
             (c[0] + r * math.cos(ang[j]), 0.05, c[1] + r * math.sin(ang[j])), rgb(190, 170, 150), mat(M_CONC, 0.8))
    fcyl(g, c[0], c[1], 0, 1.3, 7.0, CANTERA, mat(M_CONC, 0.7), seg=8)          # kiosco
    for k in range(8):
        a = k / 8 * TAU
        fcyl(g, c[0] + 6.3 * math.cos(a), c[1] + 6.3 * math.sin(a), 1.3, 6.5, 0.22, rgb(240, 240, 235),
             mat(M_METAL, 0.4, 0.4), seg=6, cap=False)
    revolve(g, [(7.6, 6.5), (6.0, 7.6), (3.0, 9.0), (0.6, 10.4), (0.05, 10.7)], (c[0], 0, c[1]), (0, 1, 0),
            rgb(60, 120, 90), mat(M_METAL, 0.45, 0.6), seg=8)
    for k in range(16):
        a = k / 16 * TAU
        box(g, c[0] + 7.2 * math.cos(a) - 0.15, 6.2, c[1] + 7.2 * math.sin(a) - 0.15, c[0] + 7.2 * math.cos(a) + 0.15,
            6.5, c[1] + 7.2 * math.sin(a) + 0.15, (1.0, 0.85, 0.55), mat(M_LAMP, 0.4, 0, 8.0))
    lights.append((float(c[0]), 6.0, float(c[1]), 30.0, 1.0, 0.85, 0.55, 3.0))
    for k in range(12):
        a = k / 12 * TAU + 0.2
        jacaranda_hd(g, c[0] + r * 0.62 * math.cos(a), c[1] + r * 0.62 * math.sin(a), rnd.uniform(0.9, 1.2), rnd)
    pc = c - v2(f) * (r - 14)                                                  # parroquia
    fv = v2(f)
    fbox(g, pc, fv, 9, 16, 0, 16, CANTERA, mat(M_CONC, 0.75))
    for sgn in (-1, 1):
        q = pc + fv * 8 + np.array([-fv[1], fv[0]]) * sgn * 13
        fbox(g, q, fv, 3.5, 3.5, 0, 30, CANTERA, mat(M_CONC, 0.75))
        sphere(g, (q[0], 31.5, q[1]), 2.4, rgb(230, 190, 80), mat(M_METAL, 0.4, 0.5), seg=10, rings=6)
    revolve(g, [(6.5, 16), (6.0, 20), (4.0, 23), (0.4, 24.5)], (pc[0], 0, pc[1]), (0, 1, 0), rgb(230, 190, 80),
            mat(M_METAL, 0.4, 0.5), seg=16)
    for _ in range(350):
        a, rr = rnd.uniform(0, TAU), rnd.uniform(9, r - 3)
        x, z = c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)
        crowd.add(x, 0.05, z, rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd)


def catedral_hd(g, c, f, lights):
    """Catedral potosina de cantera rosa con su plaza de armas."""
    fv = v2(f)
    rv = np.array([-fv[1], fv[0]])
    flat_rect(g, c + fv * 30, fv, 30, 40, 0.05, rgb(185, 165, 145), mat(M_CONC, 0.8))
    fbox(g, c - fv * 12, fv, 32, 15, 0, 22, CANTERA, mat(M_CONC, 0.75))
    for sgn in (-1, 1):
        q = c + fv * 16 + rv * sgn * 13
        fbox(g, q, fv, 5, 5, 0, 34, CANTERA, mat(M_CONC, 0.75))
        fbox(g, q, fv, 3.6, 3.6, 34, 42, CANTERA, mat(M_CONC, 0.75))
        revolve(g, [(3.6, 42), (3.0, 45), (1.2, 47.5), (0.2, 48.2)], (q[0], 0, q[1]), (0, 1, 0), rgb(230, 190, 80),
                mat(M_METAL, 0.4, 0.5), seg=12)
    q = c - fv * 18
    revolve(g, [(8, 22), (7.5, 27), (5, 31), (0.5, 33.5)], (q[0], 0, q[1]), (0, 1, 0), rgb(230, 190, 80),
            mat(M_METAL, 0.4, 0.5), seg=16)
    for k in range(3):
        q = c + fv * (35 + k * 10)
        lights.append((float(q[0]), 10.0, float(q[1]), 45.0, 1.0, 0.8, 0.55, 3.0))


def caja_agua_hd(g, c, lights):
    """Caja del Agua (fuente neoclasica de cantera)."""
    fcyl(g, c[0], c[1], 0, 1.5, 6.0, CANTERA, mat(M_CONC, 0.75), seg=8)
    fcyl(g, c[0], c[1], 1.5, 8.5, 3.6, CANTERA, mat(M_CONC, 0.75), seg=8, cap=False)
    fcyl(g, c[0], c[1], 8.5, 9.4, 4.3, rgb(180, 128, 110), mat(M_CONC, 0.75), seg=8)
    revolve(g, [(3.6, 9.4), (3.0, 11.5), (1.5, 13), (0.2, 13.6)], (c[0], 0, c[1]), (0, 1, 0), CANTERA,
            mat(M_CONC, 0.75), seg=8)
    lights.append((float(c[0]), 5.0, float(c[1]), 25.0, 1.0, 0.8, 0.55, 2.5))


def edificio_hd(g, c, f, hl, hw, h, col, k_sign, lights):
    """Edificio institucional con su letrero hacia la pista y estacionamiento."""
    fv = v2(f)
    fbox(g, c, fv, hl, hw, 0, h, col, mat(M_WINDOWS, 0.4, 0.1))
    fbox(g, c, fv, hl + 0.3, hw + 0.3, h, h + 1.2, rgb(235, 235, 238), mat(M_CONC, 0.7))
    rv = np.array([-fv[1], fv[0]])
    sign_quad(g, c + rv * (hw + 0.1), rv, min(hl - 2, 16), h - 5, h - 0.5, k_sign)
    sign_quad(g, c - rv * (hw + 0.1), -rv, min(hl - 2, 16), h - 5, h - 0.5, k_sign)
    lights.append((float(c[0] + rv[0] * (hw + 6)), 6.0, float(c[1] + rv[1] * (hw + 6)), 35.0, 0.9, 0.95, 1.0, 2.5))


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
    vehic = []
    for lm in md.landmarks:
        if lm[0] == "vehiculares":                     # puentes vehiculares sobre un tramo (p. ej. la carretera 57)
            ia, ib = (int(np.argmin(np.hypot(*(pts - md.w(q)).T))) for q in (lm[1], lm[2]))
            spanl = (ib - ia) % n
            for k in range(lm[3]):
                target = (ia + int((k + 0.5) / lm[3] * spanl)) % n
                for dl in sorted(range(-40, 41), key=abs):
                    i = (target + dl) % n
                    if any(min(abs(i - a), n - abs(i - a)) < gp for a, gp in avoid + [(j, 60) for j in vehic]):
                        continue
                    if np.max(np.abs(track.curv[[(i + q) % n for q in range(-6, 7)]])) > 1 / 300:
                        continue
                    sides = np.array([pts[i] + nrm[i] * sg * (LIMIT + 8) for sg in (-1, 1)])
                    if track.nearest_dist(sides).min() < LIMIT + 6:
                        continue
                    vehic.append(i)
                    break
    avoid += [(i, 25) for i in vehic]
    G = 6500
    xs = np.linspace(cxm - G, cxm + G, 41)
    zs = np.linspace(czm - G, czm + G, 41)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    if md.tropical:
        grid(g, np.stack([X, np.zeros_like(X), Z], -1), rgb(60, 105, 40), mat(M_GRASS, 0.95), orient=(0, 1, 0))
        tw = md.tropical["town"]
        town_c, town_r = md.w(tw[:2]), tw[2] * md.scale

        def in_town(q):
            q = np.asarray(q, float).reshape(-1, 2)
            return np.hypot(q[:, 0] - town_c[0], q[:, 1] - town_c[1]) < town_r
    else:
        grid(g, np.stack([X, np.zeros_like(X), Z], -1), rgb(112, 112, 108), mat(M_CONC, 0.85), orient=(0, 1, 0))
    # ---- lugares reservados: salida, puentes peatonales e intersecciones
    bridges = pick_spots(track, md.ped_bridges, avoid, 40, 1 / 400, LIMIT + 19, offset=0.5) if md.ped_bridges else []
    ncross = max(16, int(track.L / 430))
    cross = pick_spots(track, ncross, avoid + [(b, 14) for b in bridges], 30, 1 / 300, LIMIT + 2, offset=0.0)
    reserved = [(pts[i][0], pts[i][1], LIMIT + 22) for i in bridges] + [(pts[i][0], pts[i][1], LIMIT + 12)
                                                                         for i in cross]
    reserved += [(pts[i][0], pts[i][1], LIMIT + 34) for i in over.values()]
    reserved += [(pts[i][0], pts[i][1], LIMIT + 14) for i in vehic]
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
    for i in vehic:
        puente_vehicular_hd(g, track, i, lights, occ, rnd)
    info["vehic"] = list(vehic)
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
    if md.venue and md.venue.get("kind") == "feria":
        v = md.venue
        r0 = float(v.get("r0", 38))
        sc = fit_spot(track, md.w(v["pos"]), r0 + 62 + LIMIT + 6)
        feria_hd(g, gc, crowd, rnd, track, sc, r0, md, info, lights)
        zones.circle(sc, r0 + 64)
        keep_info = (sc, r0 + 62)
    elif md.venue and md.venue.get("kind") == "fanfest":
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
        if kind in ("monumento", "vehiculares"):
            continue                                                           # ya se construyeron sobre la pista
        if kind == "jardin":
            r = lm[2] if len(lm) > 2 else 45.0
            c = fit_spot(track, pos, r + LIMIT + 4, zones)
            i_n = int(np.argmin(np.hypot(*(pts - c).T)))
            f = pts[i_n] - c
            jardin_hd(g, crowd, rnd, c, r, f / np.linalg.norm(f), lights)
            zones.circle(c, r + 2)
            continue
        if kind == "catedral":
            c = fit_spot(track, pos, 55 + LIMIT, zones)
            i_n = int(np.argmin(np.hypot(*(pts - c).T)))
            f = pts[i_n] - c
            catedral_hd(g, c, f / np.linalg.norm(f), lights)
            info["catedral"] = c
            zones.circle(c, 55)
            continue
        if kind == "caja_agua":
            c = fit_spot(track, pos, LIMIT + 14, zones)
            caja_agua_hd(g, c, lights)
            zones.circle(c, 10)
            noveg.circle(c, 12)
            continue
        if kind == "edificio":                                                 # ("edificio", pos, letrero, color)
            hl, hw, h = 45.0, 18.0, 24.0
            c = fit_spot(track, pos, hl + LIMIT + 8, zones)
            i_n = int(np.argmin(np.hypot(*(pts - c).T)))
            edificio_hd(g, c, tan[i_n], hl, hw, h, lm[3], lm[2], lights)
            flat_rect(g, c, tan[i_n], hl + 14, hw + 22, 0.04, rgb(80, 80, 84), mat(M_CONC, 0.8))
            zones.rect(c, tan[i_n], hl + 14, hw + 22)
            noveg.rect(c, tan[i_n], hl + 16, hw + 24)
            continue
        if kind == "rio":                                                      # canal junto a la pista
            ia, ib = (int(np.argmin(np.hypot(*(pts - md.w(q)).T))) for q in (lm[1], lm[2]))
            side = lm[3]
            idx = np.array([(ia + k) % n for k in range((ib - ia) % n + 1)])
            ok = track.nearest_dist(pts[idx] + nrm[idx] * side * (LIMIT + 28)) > LIMIT + 22
            for r_ in hd.runs(ok[:-1]) if not ok.all() else [np.arange(len(idx))]:
                r_ = idx[r_[r_ < len(idx)]]
                if len(r_) < 3:
                    continue
                P = hd.offset_grid(track, r_, [LIMIT + 15, LIMIT + 19, LIMIT + 37, LIMIT + 41], [0.6, 0.02, 0.02, 0.6],
                                   side)
                grid(g, P[[0, 1]], rgb(160, 156, 148), mat(M_CONC, 0.85), orient=(0, 1, 0))
                grid(g, P[[2, 3]], rgb(160, 156, 148), mat(M_CONC, 0.85), orient=(0, 1, 0))
                grid(g, P[[1, 2]], rgb(40, 100, 125), mat(M_DEF, 0.1, 0.0), orient=(0, 1, 0))
                for k in r_[::6]:
                    q = pts[k] + nrm[k] * side * (LIMIT + 28)
                    openz.circle(q, 22)
                    noveg.circle(q, 20)
            continue
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
            half = lm[2]
            c = fit_spot(track, pos, math.hypot(*half) + LIMIT + 4)
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
            if md.tropical and not in_town(c)[0]:
                s += 4.0
                continue
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
    if md.tropical:
        blocked |= ~in_town(C)
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
    if md.tropical:
        cen = pts.mean(0)
        outer = np.sign(((pts - cen) * nrm).sum(1))                       # lado que da al mar
        coast = ~in_town(pts)
        coast = np.convolve(coast.astype(float), np.ones(15), "same") > 7
        for r in hd.runs(coast):
            for side in (-1, 1):
                rr = r[outer[r] == side]
                if len(rr) < 3:
                    continue
                for rr2 in np.split(rr, np.nonzero(np.diff(rr) % n > 1)[0] + 1):
                    if len(rr2) < 3:
                        continue
                    P = hd.offset_grid(track, rr2, [WALK, WALK + 28], 0.045, side)
                    grid(g, P, rgb(232, 214, 168), mat(M_GRAVEL, 0.95), orient=(0, 1, 0))      # arena
                    P = hd.offset_grid(track, rr2, [WALK + 28, WALK + 60, WALK + 2500], [0.06, 0.07, 0.07], side)
                    grid(g, P[[0, 1]], rgb(70, 200, 205), mat(M_DEF, 0.05, 0.3), orient=(0, 1, 0))  # turquesa
                    grid(g, P[[1, 2]], rgb(25, 95, 150), mat(M_DEF, 0.05, 0.3), orient=(0, 1, 0))   # mar abierto
        for i in range(0, n, 3):
            if not coast[i]:
                continue
            for side in (-1, 1):
                if side == outer[i]:                                       # palmeras en la playa
                    if rnd.random() < 0.75:
                        q = pts[i] + nrm[i] * side * rnd.uniform(WALK + 2, WALK + 24) + tan[i] * rnd.uniform(-3, 3)
                        if track.nearest_dist(q[None])[0] > WALK:
                            palm_hd(g, q[0], q[1], rnd.uniform(0.85, 1.25), rnd,
                                    lean=math.atan2(nrm[i, 1] * side, nrm[i, 0] * side) + rnd.uniform(-0.6, 0.6))
                else:                                                      # selva
                    for _ in range(2):
                        q = pts[i] + nrm[i] * side * rnd.uniform(WALK, WALK + 70) + tan[i] * rnd.uniform(-4, 4)
                        if track.nearest_dist(q[None])[0] > WALK - 1:
                            if rnd.random() < 0.3:
                                palm_hd(g, q[0], q[1], rnd.uniform(0.8, 1.2), rnd)
                            else:
                                forest_tree(g, q[0], q[1], rnd.uniform(1.0, 1.6), rnd)
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


# ================================================================ circuitos REALES (OpenStreetMap)
# Calles, anchos, puentes, deprimidos, alturas y edificios reales (datos de mapas_osm/, ver gtc_mapas_osm.py).
OSM_DIR = os.path.join(hd.HERE, "mapas_osm")
_OSM_DATA = {}
CORNER_SIGNS = [6, 6, 5, 15, 7, 4, 8, 13]       # OXXO (2x), SIMI, tacos, Bimbo, Bodega, gas, CDMX
POI_SIGN = {"convenience": 6, "pharmacy": 5, "chemist": 5, "restaurant": 15, "fast_food": 15, "bakery": 7,
            "supermarket": 4, "fuel": 8, "cafe": 15, "butcher": 15, "greengrocer": 4}
M_FENCE_ = hd.M_FENCE
SIDEWALK = {0: 0.0, 1: 4.5, 2: 4.0, 3: 3.6, 4: 3.0, 5: 2.6, 6: 2.3, 7: 2.0, 8: 1.8, 9: 2.3, 10: 1.5, 11: 1.5,
            12: 1.5, 13: 1.5, 14: 1.5, 15: 1.5}


def load_osm(key):
    if key not in _OSM_DATA:
        import gzip
        import json
        with gzip.open(os.path.join(OSM_DIR, key + ".json.gz"), "rt", encoding="utf-8") as f:
            _OSM_DATA[key] = json.load(f)
    return _OSM_DATA[key]


class OSMTrack(MapTrack):
    """Pista hecha con las calles reales: ancho, altura y paredes (fachadas, vallas, parapetos) por tramo."""

    def __init__(self, md):
        self.md = md
        d = self.osm = load_osm(md.osm)
        self.ctrl = np.array(d["pts"], float)
        self._setup(self.ctrl.copy())
        self.sand[:] = False
        n = self.n
        self.rw = np.maximum(np.array(d["rw"], float), 4.5)     # minimo 9 m de asfalto para 6 coches
        offa = np.abs(np.array(d["off"], float))
        wide = np.convolve((offa > 0.4).astype(float), np.ones(21), "same") > 0
        self.rw[wide] = np.maximum(self.rw[wide], 5.5)          # puentes y deprimidos: 11 m
        self.h = np.array(d["h"], float)
        self.hg = np.array(d["hg"], float)
        self.off = np.array(d["off"], float)
        self.br = np.array(d["br"], bool)
        self.tu = np.array(d["tu"], bool)
        self.ow = np.array(d.get("ow", [0] * n), bool)
        self.race = np.array([str(d["names"][k]).startswith("Autódromo") for k in d["name"]], bool)
        self.ow |= self.race
        self.ln = np.array(d.get("ln", [2] * n), int)
        self.cls = np.array(d["cls"], int)
        self.name_idx = np.array(d["name"], int)
        self.names = d["names"]
        self.wpos = self.rw + 3.0                  # paredes de cada lado (las calcula el constructor del mundo)
        self.wneg = self.rw + 3.0
        # cruces de la ruta consigo misma: a nivel (se pintan) o a desnivel (puente sobre la otra calle)
        cp = self.find_crossings()
        self.cross_pts = cp
        self.cross_flat = [(a, b, c) for a, b, c in cp if abs(self.h[a] - self.h[b]) < 3.0]
        self.fix_overlaps()
        # alturas sin escalones: suavizado ligero (la subida de los puentes queda continua)
        ks = np.exp(-0.5 * (np.arange(-4, 5) / 1.6) ** 2)
        ks /= ks.sum()
        self.h = np.convolve(np.concatenate([self.h[-4:], self.h, self.h[:4]]), ks, "same")[4:-4]
        self.dh = (np.roll(self.h, -1) - np.roll(self.h, 1)) / (2 * self.spacing)   # pendiente por muestra

    def fix_overlaps(self):
        """Donde la ruta pasa dos veces por el mismo lugar: misma altura (cruce a nivel o la misma avenida de ida y
        vuelta) o, si una va por arriba, al menos 7.5 m de altura libre (puente)."""
        n, P = self.n, self.pts
        G, cell = {}, 12.0
        for i, (x, z) in enumerate(P):
            G.setdefault((int(x // cell), int(z // cell)), []).append(i)
        I, J, D, S = [], [], [], []
        for i, (x, z) in enumerate(P):
            cx, cz = int(x // cell), int(z // cell)
            best = None
            for dx in (-2, -1, 0, 1, 2):
                for dz in (-2, -1, 0, 1, 2):
                    for j in G.get((cx + dx, cz + dz), ()):
                        ds = abs(i - j)
                        if min(ds, n - ds) < 40:
                            continue
                        d = math.hypot(P[j, 0] - x, P[j, 1] - z)
                        if d < self.rw[i] + self.rw[j] + 4.0 and (best is None or d < best[1]):
                            best = (j, d)
            if best is not None:
                j, d = best
                I.append(i)
                J.append(j)
                D.append(d)
                e = P[j] - P[i]
                S.append(1 if e[0] * self.nrm[i, 0] + e[1] * self.nrm[i, 1] >= 0 else -1)
        self.partner = np.full(n, -1)
        self.partner_d = np.zeros(n)
        self.partner_side = np.zeros(n, int)
        if not I:
            return
        I, J, D, S = map(np.array, (I, J, D, S))
        self.partner[I], self.partner_d[I], self.partner_side[I] = J, D, S
        h0, off0 = self.h.copy(), self.off.copy()
        delta = np.zeros(n)
        fixed = np.zeros(n, bool)
        dh = h0[J] - h0[I]
        grade = np.abs(off0[I] - off0[J]) > 2.0                 # una es puente y la otra no: a desnivel
        flat = (np.abs(dh) < 5.0) & ~grade
        delta[I[flat]] = 0.5 * (h0[I[flat]] + h0[J[flat]]) - h0[I[flat]] + np.where(I[flat] > J[flat], 0.03, 0.0)
        fixed[I[flat]] = True
        upper_i = np.where(grade, off0[I] > off0[J], dh < 0)
        up = ~flat & upper_i & (np.abs(dh) < 7.5)            # i va arriba de j: subir i
        dh = np.where(grade & upper_i, -np.maximum(h0[I] - h0[J], 0.0), dh)
        delta[I[up]] = 7.5 - np.abs(dh[up])
        fixed[I[up]] = True
        for _ in range(6):                                  # la zona igualada se extiende 18 m mas
            for sh in (1, -1):
                nb = np.roll(fixed, sh) & ~fixed
                delta[nb] = np.roll(delta, sh)[nb]
                fixed = fixed | nb
        if fixed.any():
            d2 = delta.copy()
            for _ in range(220):                            # transicion suave (~25 muestras) hacia cero
                avg = 0.5 * (np.roll(d2, 1) + np.roll(d2, -1)) * 0.97
                d2 = np.where(fixed, delta, avg)
            self.h = h0 + d2
            rise = np.where(d2 > 0.4, d2, 0.0)
            br = rise > 0.4
            self.off = np.where(br, np.maximum(off0, rise), off0)   # se dibuja como puente
            self.br = self.br | (br & (self.off > 3.0))

    def nearest_dist(self, q):
        """Distancia a la ruta (rapida: indice por celdas; lejos de la ruta basta una aproximacion)."""
        q = np.asarray(q, float).reshape(-1, 2)
        if not hasattr(self, "_rg"):
            self._rg = route_grid(self, 16.0)
            self._coarse = self.pts[::6].astype(np.float32)
        _, _, d = nearest_route(self, self._rg, q, 48.0)
        far = d > 1e8
        if far.any():
            Q = q[far].astype(np.float32)
            out = np.empty(len(Q), np.float32)
            for a in range(0, len(Q), 1024):
                c = Q[a:a + 1024]
                out[a:a + 1024] = np.sqrt(((c[:, None, :] - self._coarse[None]) ** 2).sum(-1).min(1))
            d[far] = out
        return d

    def h_at(self, u):
        f = (u % self.L) / self.spacing
        i = int(f) % self.n
        t = f - int(f)
        return float(self.h[i] * (1 - t) + self.h[(i + 1) % self.n] * t)

    def slope_at(self, u):
        f = (u % self.L) / self.spacing
        i = int(f) % self.n
        t = f - int(f)
        return float(self.dh[i] * (1 - t) + self.dh[(i + 1) % self.n] * t)

    def h_xz(self, x, z, near=None):
        """Altura del camino en (x, z), interpolada sobre el tramo (sin saltos)."""
        if near is None:
            i = int(np.argmin((self.pts[:, 0] - x) ** 2 + (self.pts[:, 1] - z) ** 2))
        else:
            w = (np.arange(-12, 13) + int(near)) % self.n
            i = int(w[np.argmin((self.pts[w, 0] - x) ** 2 + (self.pts[w, 1] - z) ** 2)])
        j = (i - 1) % self.n
        best = None
        for a in (j, i):
            d = self.d[a]
            t = float(np.clip(((x - self.pts[a, 0]) * d[0] + (z - self.pts[a, 1]) * d[1]) * self.inv_len2[a], 0, 1))
            e = (self.pts[a, 0] + d[0] * t - x) ** 2 + (self.pts[a, 1] + d[1] * t - z) ** 2
            if best is None or e < best[0]:
                best = (e, a, t)
        _, a, t = best
        return float(self.h[a] * (1 - t) + self.h[(a + 1) % self.n] * t)

    def project(self, x, z):
        car = sys._getframe(1).f_locals.get("self")
        i0 = getattr(car, "idx", None)
        res = None
        if isinstance(i0, (int, np.integer)):
            w = (np.arange(-30, 90) + int(i0)) % self.n
            px, pz = x - self.pts[w, 0], z - self.pts[w, 1]
            d = self.d[w]
            t = np.clip((px * d[:, 0] + pz * d[:, 1]) * self.inv_len2[w], 0, 1)
            ex, ez = px - d[:, 0] * t, pz - d[:, 1] * t
            e2 = ex * ex + ez * ez
            # se prefiere seguir en el mismo tramo: un puente encima de la propia ruta no 'atrapa' al coche
            k = int(np.argmin(e2 + ((np.arange(-30, 90) * self.spacing) * 0.15) ** 2))
            if e2[k] < (3 * LIMIT) ** 2:
                i = int(w[k])
                nx, nz = self.nrm[i]
                res = (float(self.cum[i] + t[k] * self.seg_len[i]), i, float(ex[k] * nx + ez[k] * nz),
                       self.pts[i, 0] + d[k, 0] * t[k], self.pts[i, 1] + d[k, 1] * t[k], nx, nz)
        if res is None:
            res = base.Track.project(self, x, z)
        s, i, lat, cx, cz, nx, nz = res
        # paredes reales de este tramo -> la fisica del juego base usa LIMIT fijo: se traduce la distancia
        sg = 1.0 if lat >= 0 else -1.0
        w = float(self.wpos[i] if lat >= 0 else self.wneg[i])
        u = abs(lat)
        if car is not None and hasattr(car, "idx"):
            car.true_lat = lat                     # el piso (asfalto / banqueta) se decide con la distancia real
        b1 = w - 3.2
        L0 = LIMIT
        v = u * (L0 - 3.2) / b1 if u <= b1 else u - (w - L0)
        return s, i, sg * v, cx + nx * sg * (w - L0), cz + nz * sg * (w - L0), nx, nz


class Terrain:
    """Altura del suelo en cualquier punto (interpolada de la altura real a lo largo de la ruta)."""

    def __init__(self, track):
        k = max(1, track.n // 700)
        self.P = track.pts[::k].astype(np.float32)
        src = (track.h - track.off) if hasattr(track, "off") else track.hg   # el suelo sigue al camino (sin puentes)
        self.H = src[::k].astype(np.float32)

        lo, hi = track.pts.min(0) - 700, track.pts.max(0) + 700
        self.c0, self.cs = lo, 40.0
        nx, nz = int((hi[0] - lo[0]) / 40) + 2, int((hi[1] - lo[1]) / 40) + 2
        gx, gz = np.meshgrid(lo[0] + np.arange(nx) * 40.0, lo[1] + np.arange(nz) * 40.0, indexing="ij")
        self.grid = self._idw(np.column_stack([gx.ravel(), gz.ravel()])).reshape(nx, nz)

    def __call__(self, Q):
        """Altura (interpolacion bilineal en una malla de 25 m)."""
        Q = np.asarray(Q, float).reshape(-1, 2)
        fx = np.clip((Q[:, 0] - self.c0[0]) / self.cs, 0, self.grid.shape[0] - 1.001)
        fz = np.clip((Q[:, 1] - self.c0[1]) / self.cs, 0, self.grid.shape[1] - 1.001)
        ix, iz = fx.astype(int), fz.astype(int)
        tx, tz = fx - ix, fz - iz
        G = self.grid
        return ((G[ix, iz] * (1 - tx) + G[ix + 1, iz] * tx) * (1 - tz) +
                (G[ix, iz + 1] * (1 - tx) + G[ix + 1, iz + 1] * tx) * tz).astype(np.float32)

    def _idw(self, Q):
        Q = np.asarray(Q, np.float32).reshape(-1, 2)
        out = np.empty(len(Q), np.float32)
        for a in range(0, len(Q), 1024):
            c = Q[a:a + 1024]
            d2 = ((c[:, None, :] - self.P[None]) ** 2).sum(-1)
            w = 1.0 / (d2 + 1600.0) ** 1.5
            out[a:a + 1024] = (w * self.H).sum(1) / w.sum(1)
        return out


class OffsetCrowd:
    """Publico de un monumento construido en y=0 que se sube a la altura del suelo."""

    def __init__(self, crowd, dy):
        self.crowd, self.dy = crowd, dy

    def add(self, x, y, z, *a, **k):
        self.crowd.add(x, y + self.dy, z, *a, **k)


def lift_geo(geo, dy):
    for p in geo.parts:
        p[:, 1] += dy
    return geo


def placed(g, crowd, lights, dy, fn):
    """Construye algo pensado para y=0 y lo coloca a la altura del suelo dy."""
    tg, tl = Geo(), []
    r = fn(tg, OffsetCrowd(crowd, dy), tl)
    g.merge(lift_geo(tg, dy))
    lights += [(l[0], l[1] + dy) + tuple(l[2:]) for l in tl]
    return r


def ear_clip(P):
    """Triangulacion de un poligono simple (lista de indices de triangulos)."""
    n = len(P)
    if n < 3:
        return []
    area = 0.5 * sum(P[i][0] * P[(i + 1) % n][1] - P[(i + 1) % n][0] * P[i][1] for i in range(n))
    idx = list(range(n)) if area > 0 else list(range(n))[::-1]
    tris, guard = [], 0
    while len(idx) > 3 and guard < 4 * n:
        guard += 1
        m = len(idx)
        for k in range(m):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % m]
            ax, az = P[a]
            bx, bz = P[b]
            cx, cz = P[c]
            if (bx - ax) * (cz - az) - (bz - az) * (cx - ax) <= 1e-9:
                continue
            ok = True
            for o in idx:
                if o in (a, b, c):
                    continue
                px, pz = P[o]
                d1 = (bx - ax) * (pz - az) - (bz - az) * (px - ax)
                d2 = (cx - bx) * (pz - bz) - (cz - bz) * (px - bx)
                d3 = (ax - cx) * (pz - cz) - (az - cz) * (px - cx)
                if d1 > 0 and d2 > 0 and d3 > 0:
                    ok = False
                    break
            if ok:
                tris.append((a, b, c))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def inside_poly(P, Q):
    """Que puntos Q (m, 2) caen dentro del poligono P (k, 2)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    x, z = Q[:, 0], Q[:, 1]
    ins = np.zeros(len(Q), bool)
    A, B = P, np.roll(P, -1, 0)
    for (ax, az), (bx, bz) in zip(A, B):
        c = (az > z) != (bz > z)
        xi = ax + (z - az) * (bx - ax) / np.where(np.abs(bz - az) < 1e-12, 1e-12, bz - az)
        ins ^= c & (x < xi)
    return ins


def open_runs(mask):
    """Tramos contiguos True de una linea ABIERTA (sin unir el final con el principio)."""
    out, cur = [], []
    for k, v in enumerate(mask):
        if v:
            cur.append(k)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


class StreetIndex:
    """Calles dibujadas (segmentos con su medio ancho) para no poner edificios encima."""

    def __init__(self, cell=20.0):
        self.cell, self.G, self.seg = cell, {}, []

    def add(self, a, b, half):
        k = len(self.seg)
        self.seg.append((np.asarray(a, float), np.asarray(b, float), float(half)))
        lo, hi = np.minimum(a, b) - half, np.maximum(a, b) + half
        for cx in range(int(lo[0] // self.cell), int(hi[0] // self.cell) + 1):
            for cz in range(int(lo[1] // self.cell), int(hi[1] // self.cell) + 1):
                self.G.setdefault((cx, cz), []).append(k)

    def hit(self, Q, margin=0.5):
        Q = np.asarray(Q, float).reshape(-1, 2)
        for q in Q:
            for k in self.G.get((int(q[0] // self.cell), int(q[1] // self.cell)), ()):
                a, b, half = self.seg[k]
                d = b - a
                t = min(1.0, max(0.0, float(((q - a) @ d) / max(d @ d, 1e-9))))
                if math.hypot(*(a + d * t - q)) < half + margin:
                    return True
        return False


def footprint_probe(c, f, hl, hw, step=2.0):
    """Puntos (cada ~2 m) que cubren un rectangulo orientado."""
    f = v2(f)
    r = np.array([-f[1], f[0]])
    us = np.linspace(-hl, hl, max(2, int(2 * hl / step) + 2))
    vs = np.linspace(-hw, hw, max(2, int(2 * hw / step) + 2))
    return (v2(c)[None, None] + us[:, None, None] * f + vs[None, :, None] * r).reshape(-1, 2)


class BuildingIndex:
    """Plantas de edificios reales para preguntar si un punto cae dentro de alguno."""

    def __init__(self, polys, cell=25.0):
        self.polys = [np.asarray(p, float) for p in polys]
        self.cell = cell
        self.G = {}
        for k, P in enumerate(self.polys):
            if len(P) < 4:
                continue
            lo, hi = P.min(0), P.max(0)
            for cx in range(int(lo[0] // cell), int(hi[0] // cell) + 1):
                for cz in range(int(lo[1] // cell), int(hi[1] // cell) + 1):
                    self.G.setdefault((cx, cz), []).append(k)

    def inside(self, Q, grow=0.0):
        Q = np.asarray(Q, float).reshape(-1, 2)
        out = np.zeros(len(Q), bool)
        for j, q in enumerate(Q):
            for k in self.G.get((int(q[0] // self.cell), int(q[1] // self.cell)), ()):
                P = self.polys[k]
                lo, hi = P.min(0) - grow, P.max(0) + grow
                if lo[0] <= q[0] <= hi[0] and lo[1] <= q[1] <= hi[1] and inside_poly(P, q[None])[0]:
                    out[j] = True
                    break
        return out


def route_grid(track, cell=12.0):
    """Indice espacial de las muestras de la ruta."""
    G = {}
    for i, (x, z) in enumerate(track.pts):
        G.setdefault((int(x // cell), int(z // cell)), []).append(i)
    return G, cell


_NR_BLOCKS = {}


def nearest_route(track, rg, Q, maxd=40.0):
    """Para cada punto: (indice de la muestra mas cercana, distancia lateral con signo, distancia) o -1."""
    G, cell = rg
    Q = np.asarray(Q, float).reshape(-1, 2)
    ii = np.full(len(Q), -1)
    lat = np.zeros(len(Q))
    dist = np.full(len(Q), 1e9)
    r = int(math.ceil(maxd / cell))
    B = 2 * r + 1
    occB = _NR_BLOCKS.get((id(G), B))
    if occB is None:                                                      # bloques con ruta (para saltar lo lejano)
        occB = _NR_BLOCKS[(id(G), B)] = {(cx // B, cz // B) for cx, cz in G}
    cells = {}
    for k, (x, z) in enumerate(Q):
        cells.setdefault((int(x // cell), int(z // cell)), []).append(k)
    for (cx, cz), ks in cells.items():
        bx, bz = cx // B, cz // B
        if not any((bx + a, bz + b) in occB for a in (-1, 0, 1) for b in (-1, 0, 1)):
            continue
        cand = []
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                cand += G.get((cx + dx, cz + dz), [])
        if not cand:
            continue
        cand = np.array(cand)
        q = Q[ks]
        D = np.hypot(q[:, None, 0] - track.pts[cand][None, :, 0], q[:, None, 1] - track.pts[cand][None, :, 1])
        j = np.argmin(D, axis=1)
        best = cand[j]
        ii[ks] = best
        dist[ks] = D[np.arange(len(ks)), j]
        e = q - track.pts[best]
        lat[ks] = e[:, 0] * track.nrm[best, 0] + e[:, 1] * track.nrm[best, 1]
    return ii, lat, dist


def road_ribbon(g, track, idx, lats, ys, col, m, uv_scale=None, orient=(0, 1, 0)):
    """Malla a lo largo de la ruta: lats/ys (C, len(idx)) en metros laterales y altura absoluta."""
    lats = np.asarray(lats, float)
    ys = np.broadcast_to(np.asarray(ys, float), lats.shape)
    pts, nrm = track.pts[idx], track.nrm[idx]
    k = track.curv[idx]
    lim = np.where(np.abs(k) > 1e-4, 0.92 / np.maximum(np.abs(k), 1e-4), 1e9)
    inner = np.sign(lats) == np.sign(k)[None, :]
    lats = np.where(inner, np.clip(lats, -lim[None, :], lim[None, :]), lats)
    P = np.zeros(lats.shape + (3,), F4)
    P[..., 0] = pts[None, :, 0] + nrm[None, :, 0] * lats
    P[..., 2] = pts[None, :, 1] + nrm[None, :, 1] * lats
    P[..., 1] = ys
    s = hd.s_unwrap(track, idx)
    UV = np.stack(np.broadcast_arrays(s[None, :], lats if uv_scale is None else lats * uv_scale), -1).astype(F4)
    grid(g, P, col, m, UV=UV, orient=orient)


def ped_bridge_osm(g, crowd, rnd, track, i, ang, lights, y0, occ):
    """Puente peatonal real sobre la ruta (a la altura del camino)."""
    tg = Geo()
    half = max(track.rw[i] + 5.0, 9.0)
    ped_bridge_hd(tg, OffsetCrowd(crowd, y0), rnd, track, i, rnd.choice([11, 6, 8, 15, 4]), [], occ, half=half)
    g.merge(lift_geo(tg, y0))
    lights.append((float(track.pts[i][0]), y0 + 8.4, float(track.pts[i][1]), 30.0, 1.0, 0.88, 0.7, 3.0))


def start_arch(g, track, info, lights):
    """Arco de salida/meta del ancho de la calle, con el semaforo de arranque."""
    p0, n0, t0 = track.pts[0], track.nrm[0], track.tan[0]
    y = track.h[0]
    half = track.rw[0] + 0.8
    steel = rgb(40, 42, 48)
    for side in (-1, 1):
        c = p0 + n0 * side * half
        fbox(g, c, t0, 0.5, 0.5, y, y + 8.4, steel, mat(M_METAL, 0.45, 0.8))
    a = p0 - n0 * half
    obox(g, (a[0] - t0[0] * 0.6, y + 6.6, a[1] - t0[1] * 0.6), (n0[0] * 2 * half, 0, n0[1] * 2 * half), (0, 1.8, 0),
         (t0[0] * 1.2, 0, t0[1] * 1.2), steel, mat(M_METAL, 0.45, 0.8))
    for sgn in (-1, 1):
        sign_quad(g, p0 + t0 * 0.62 * sgn, -t0 * sgn, half - 0.4, y + 6.8, y + 8.2, 3)
    for k in range(5):
        off = (k - 2) * min(2.0, half * 0.35)
        cx, cz = p0[0] - t0[0] * 0.63 + n0[0] * off, p0[1] - t0[1] * 0.63 + n0[1] * off
        for row in range(2):
            yy = y + 6.0 + row * 0.5
            quad(g, (cx - n0[0] * 0.4, yy, cz - n0[1] * 0.4), (cx + n0[0] * 0.4, yy, cz + n0[1] * 0.4),
                 (cx + n0[0] * 0.4, yy + 0.4, cz + n0[1] * 0.4), (cx - n0[0] * 0.4, yy + 0.4, cz - n0[1] * 0.4),
                 (1, 1, 1), mat(hd.M_STARTL, 0.3), ((k, 0), (k, 0), (k, 1), (k, 1)))
    a, b = p0 - n0 * track.rw[0], p0 + n0 * track.rw[0]
    quad(g, (a[0], y + 0.08, a[1]), (b[0], y + 0.08, b[1]), (b[0] + t0[0] * 2.5, y + 0.08, b[1] + t0[1] * 2.5),
         (a[0] + t0[0] * 2.5, y + 0.08, a[1] + t0[1] * 2.5), (1, 1, 1), mat(hd.M_CHECK, 0.5),
         ((0, -track.rw[0]), (0, track.rw[0]), (2.5, track.rw[0]), (2.5, -track.rw[0])))
    lights.append((float(p0[0]), y + 7.5, float(p0[1]), 30.0, 1.0, 0.95, 0.85, 3.0))
    info["gantry"] = (float(p0[0]), float(p0[1]))


def osm_building(g, P, base, top, col, rnd, route_facing=None, sign_k=None, crowd=None, floor=None):
    """Edificio real a partir de su planta: muros con ventanas, azotea, tinacos y local hacia la ruta."""
    P = np.asarray(P, float)
    if len(P) > 3 and np.hypot(*(P[0] - P[-1])) < 0.05:
        P = P[:-1]
    n = len(P)
    if n < 3:
        return
    cen = P.mean(0)
    A, B = P, np.roll(P, -1, 0)
    E = B - A
    Ln = np.hypot(E[:, 0], E[:, 1])
    ok = Ln > 0.05
    A, B, E, Ln = A[ok], B[ok], E[ok], Ln[ok]
    if len(A) < 3:
        return
    Nn = np.column_stack([E[:, 1], -E[:, 0]]) / Ln[:, None]
    mid = (A + B) / 2
    flip = ((mid - cen) * Nn).sum(1) < 0
    Nn[flip] *= -1
    k = len(A)
    u0 = np.concatenate([[0], np.cumsum(Ln)[:-1]])
    V = np.empty((k, 6, 3), F4)
    UVs = np.empty((k, 6, 2), F4)
    for j, (sa, sy) in enumerate(((0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1))):
        q = B if sa else A
        V[:, j, 0], V[:, j, 2] = q[:, 0], q[:, 1]
        V[:, j, 1] = top if sy else base
        UVs[:, j, 0] = u0 + Ln * sa
        UVs[:, j, 1] = (top - base) if sy else 0.0
    NN = np.repeat(np.column_stack([Nn[:, 0], np.zeros(k), Nn[:, 1]])[:, None, :], 6, 1)
    g.tris(V.reshape(-1, 3), NN.reshape(-1, 3), col, mat(M_WINDOWS, 0.55, 0.05), UVs.reshape(-1, 2))
    tri = ear_clip([tuple(p) for p in A])
    if tri:
        T = np.array([[A[a], A[b], A[c]] for a, b, c in tri]).reshape(-1, 2)
        g.tris(np.column_stack([T[:, 0], np.full(len(T), top), T[:, 1]]), (0.0, 1.0, 0.0),
               tuple(v * 0.8 for v in col), mat(M_CONC, 0.85), T)
    if top - base < 30 and rnd.random() < 0.35:                                  # tinacos
        for _ in range(rnd.randint(1, 2)):
            q = cen + (A[rnd.randrange(k)] - cen) * rnd.uniform(0.1, 0.6)
            fcyl(g, q[0], q[1], top, top + 1.4, 0.62, rgb(22, 22, 24), mat(M_DEF, 0.6), seg=6)
    if route_facing is not None:                                                # local hacia la ruta
        j = route_facing
        a, b, nv = A[j], B[j], Nn[j]
        f = (b - a) / Ln[j]
        w = Ln[j] / 2 - 0.6
        F = (a + b) / 2 + nv * 0.06
        gy = floor if floor is not None else base + 2.5
        face_quad(g, F - f * w, F + f * w, gy + 0.25, gy + 2.8, (1.0, 0.84, 0.6), mat(M_LAMP, 0.5, 0, 2.2), nv)
        a0, a1 = F - f * w, F + f * w
        b0, b1 = a0 + nv * 1.5, a1 + nv * 1.5
        quad_dir(g, (a0[0], gy + 3.15, a0[1]), (a1[0], gy + 3.15, a1[1]), (b1[0], gy + 2.6, b1[1]),
                 (b0[0], gy + 2.6, b0[1]), rnd.choice(AWNINGS), mat(M_DEF, 0.75), (nv[0], 1.0, nv[1]))
        if sign_k is not None:
            sign_quad(g, F + nv * 0.05, nv, min(w, 4.5), gy + 3.3, gy + 4.2, sign_k)


def build_world_osm(track, md):
    """Ciudad real: la ruta con sus anchos y alturas, calles, edificios de OSM + manzanas, puentes y lugares."""
    d = track.osm
    rnd = random.Random(sum(map(ord, md.key)) * 7907)
    g, gc, lights, info = Geo(), Geo(), [], {}
    crowd = PaletteCrowd(MEXICO_SHIRTS if md.palette == "mexico" else md.palette) if md.palette else Crowd()
    pts, nrm, tan, n = track.pts, track.nrm, track.tan, track.n
    rw, h, off = track.rw, track.h, track.off
    shared = np.zeros((n, 2), bool)                                        # lado donde la ruta vuelve a pasar
    for i in np.nonzero(track.partner >= 0)[0]:
        j = track.partner[i]
        if abs(h[i] - h[j]) > 2.5:                                         # una va encima de la otra: no se comparte
            continue
        if abs(tan[i] @ tan[j]) < 0.85:                                    # cruce en X: los dos lados
            shared[i, :] = True
        else:
            shared[i, 1 if track.partner_side[i] > 0 else 0] = True
    shared = np.stack([np.convolve(shared[:, k].astype(float), np.ones(9), "same") > 0 for k in (0, 1)], 1)
    lm0 = {k: np.array(v[:2], float) for k, v in d.get("lm", {}).items()}
    if md.datum in lm0 and not getattr(track, "_datum_done", False):     # el concierto queda en y = 0
        dy = float(Terrain(track)([lm0[md.datum]])[0])
        track.h -= dy
        track.hg -= dy
        track._datum_done = True
    T = Terrain(track)
    rg = route_grid(track)
    occ, zones, openz = Occupancy(4.0), Zones(), Zones()
    cxm, czm = track.center
    lo, hi = pts.min(0) - 650, pts.max(0) + 650
    trop = md.tropical is not None
    town_m = np.ones(n, bool)
    glo, ghi = lo, hi
    if trop:                                                               # isla: tierra, mar y el pueblo
        land_polys = []
        for a in d.get("areas", []):
            if a["k"] == "land" and len(a["p"]) > 20:
                P_ = np.array(a["p"], float)
                if inside_poly(P_, pts[::40]).mean() > 0.3:
                    land_polys.append(P_)
        allp = np.concatenate(land_polys) if land_polys else pts
        coast = Coast(land_polys, np.minimum(allp.min(0), lo) - 3000, np.maximum(allp.max(0), hi) + 3000)
        if not land_polys:
            coast.D[:] = 999.0
        ds_at = coast.ds
        town_c = lm0.get(md.tropical.get("lm", "plaza"), pts[0])
        town_r = float(md.tropical.get("r", 2300.0))

        def in_town(Q):
            Q = np.asarray(Q, float).reshape(-1, 2)
            return np.hypot(Q[:, 0] - town_c[0], Q[:, 1] - town_c[1]) < town_r
        town_m = in_town(pts)
        hb0 = h - np.maximum(off, 0.0)
        sea_y = float(np.percentile(hb0, 1)) - 1.6                         # nivel del mar
        if land_polys:
            glo = np.minimum(lo, allp.min(0) - 300)
            ghi = np.maximum(hi, allp.max(0) + 300)
        print(f"   isla: {len(land_polys)} costa(s), mar a y={sea_y:.1f}, pueblo {town_m.mean() * 100:.0f}% de la ruta")
    # ---------------------------------------------------------------- suelo con alturas reales
    cell = 75.0 if trop else 40.0
    gx = np.arange(glo[0], ghi[0] + cell, cell)
    gz = np.arange(glo[1], ghi[1] + cell, cell)
    GX, GZ = np.meshgrid(gx, gz, indexing="ij")
    GQ = np.column_stack([GX.ravel(), GZ.ravel()])
    GY = T(GQ) - 0.35
    hb = h - np.maximum(off, 0.0)                                        # altura del camino sin contar puentes
    hbmin = np.minimum.reduce([np.roll(hb, q) for q in range(-16, 17, 2)])    # lo mas bajo en +-48 m
    ii, lat_, dist_ = nearest_route(track, rg, GQ, 110)
    m = ii >= 0
    lim = hbmin[ii[m]] - 0.45 + 0.03 * np.maximum(dist_[m] - rw[ii[m]] - 30.0, 0.0)
    GY[m] = np.minimum(GY[m], lim)                                        # el piso siempre abajo del asfalto
    GY = GY.reshape(GX.shape)
    nxg, nzg = GY.shape
    probe = np.concatenate([pts + nrm * (rw * f)[:, None] for f in (-1.0, -0.5, 0.0, 0.5, 1.0)])
    ptop = np.tile(hb, 5) - 0.3
    fx = np.clip((probe[:, 0] - gx[0]) / cell, 0, nxg - 1.001)
    fz = np.clip((probe[:, 1] - gz[0]) / cell, 0, nzg - 1.001)
    ix, iz = fx.astype(int), fz.astype(int)
    tx, tz = fx - ix, fz - iz
    for _ in range(6):                                                    # bajar las esquinas que quedan arriba
        y = ((GY[ix, iz] * (1 - tx) + GY[ix + 1, iz] * tx) * (1 - tz) +
             (GY[ix, iz + 1] * (1 - tx) + GY[ix + 1, iz + 1] * tx) * tz)
        ex = y - ptop
        bad = ex > 0
        if not bad.any():
            break
        for ax, az in ((0, 0), (1, 0), (0, 1), (1, 1)):
            np.minimum.at(GY, (ix[bad] + ax, iz[bad] + az), GY[ix[bad] + ax, iz[bad] + az] - ex[bad] - 0.05)

    def gh(Q):
        """Altura del piso (la misma malla que se dibuja)."""
        Q = np.asarray(Q, float).reshape(-1, 2)
        fx_ = np.clip((Q[:, 0] - gx[0]) / cell, 0, nxg - 1.001)
        fz_ = np.clip((Q[:, 1] - gz[0]) / cell, 0, nzg - 1.001)
        ix_, iz_ = fx_.astype(int), fz_.astype(int)
        tx_, tz_ = fx_ - ix_, fz_ - iz_
        return ((GY[ix_, iz_] * (1 - tx_) + GY[ix_ + 1, iz_] * tx_) * (1 - tz_) +
                (GY[ix_, iz_ + 1] * (1 - tx_) + GY[ix_ + 1, iz_ + 1] * tx_) * tz_)
    if trop:                                                               # playa que baja al mar
        DS = ds_at(GQ).reshape(GX.shape)
        GY[:] = np.where(DS < 45, np.minimum(GY, sea_y + 0.25 + np.maximum(DS, 0) * 0.045), GY)
        GY[:] = np.where(DS < 0, np.minimum(GY, sea_y - 0.6 + np.maximum(DS, -400) * 0.02), GY)
    gmin = float(GY.min())
    if trop:
        nz_ = np.array([rnd.uniform(0.85, 1.12) for _ in range(997)])[(np.arange(GX.size) * 7919) % 997]
        nz_ = nz_.reshape(GX.shape)[..., None]
        sand, jung = np.array(rgb(228, 212, 168)), np.array(rgb(36, 74, 28))
        tb = np.clip((DS - 18.0) / 50.0, 0, 1)[..., None]
        CC = (sand * (1 - tb) + jung * tb * nz_).astype(np.float32)
        cells_ = np.maximum.reduce([DS[:-1, :-1], DS[1:, :-1], DS[:-1, 1:], DS[1:, 1:]]) > -90
        masked_grid(g, np.stack([GX, GY, GZ], -1), CC, cells_, mat(M_GRASS, 0.95))
        # mar: turquesa en la orilla, azul profundo mar adentro
        wx = np.arange(coast.c0[0], coast.c0[0] + coast.D.shape[0] * coast.res, 150.0)
        wz = np.arange(coast.c0[1], coast.c0[1] + coast.D.shape[1] * coast.res, 150.0)
        WX, WZ = np.meshgrid(wx, wz, indexing="ij")
        WD = ds_at(np.column_stack([WX.ravel(), WZ.ravel()])).reshape(WX.shape)
        tq, mid, deep = np.array(rgb(60, 205, 205)), np.array(rgb(25, 140, 180)), np.array(rgb(14, 62, 125))
        a1 = np.clip(-WD / 260.0, 0, 1)[..., None]
        a2 = np.clip((-WD - 260.0) / 600.0, 0, 1)[..., None]
        WC = (tq * (1 - a1) + mid * a1) * (1 - a2) + deep * a2
        wcell = np.maximum.reduce([WD[:-1, :-1], WD[1:, :-1], WD[:-1, 1:], WD[1:, 1:]]) > -1000
        masked_grid(g, np.stack([WX, np.full_like(WX, sea_y), WZ], -1), WC.astype(np.float32), wcell,
                    mat(M_DEF, 0.04, 0.35))
        w0, w1 = np.array([wx[0], wz[0]]), np.array([wx[-1], wz[-1]])
        quad_dir(g, (w0[0], sea_y - 0.04, w0[1]), (w1[0], sea_y - 0.04, w0[1]), (w1[0], sea_y - 0.04, w1[1]),
                 (w0[0], sea_y - 0.04, w1[1]), tuple(deep), mat(M_DEF, 0.04, 0.35), (0, 1, 0))
        G3 = 40000
        for (x0, x1, z0, z1) in ((w0[0] - G3, w0[0], w0[1] - G3, w1[1] + G3), (w1[0], w1[0] + G3, w0[1] - G3, w1[1] + G3),
                                 (w0[0], w1[0], w0[1] - G3, w0[1]), (w0[0], w1[0], w1[1], w1[1] + G3)):
            quad_dir(g, (x0, sea_y, z0), (x1, sea_y, z0), (x1, sea_y, z1), (x0, sea_y, z1), tuple(deep), mat(M_DEF, 0.04, 0.35), (0, 1, 0))
    elif md.ground == "bosque":
        grid(g, np.stack([GX, GY, GZ], -1), rgb(72, 110, 44), mat(M_GRASS, 0.95), orient=(0, 1, 0))
    else:
        grid(g, np.stack([GX, GY, GZ], -1), rgb(122, 118, 110), mat(M_CONC, 0.9), orient=(0, 1, 0))
    G2 = 7000
    for (x0, x1, z0, z1) in (() if trop else
                             ((cxm - G2, lo[0], czm - G2, czm + G2), (hi[0], cxm + G2, czm - G2, czm + G2),
                              (lo[0], hi[0], czm - G2, lo[1]), (lo[0], hi[0], hi[1], czm + G2))):
        quad_dir(g, (x0, gmin - 0.5, z0), (x1, gmin - 0.5, z0), (x1, gmin - 0.5, z1), (x0, gmin - 0.5, z1),
                 rgb(110, 108, 100), mat(M_CONC, 0.9), (0, 1, 0))
    yG = lambda x, z: float(T([(x, z)])[0])
    # ---------------------------------------------------------------- lugares reales (zonas reservadas)
    lm = {k: np.array(v[:2], float) for k, v in d.get("lm", {}).items()}
    build_osm_landmarks(g, gc, crowd, rnd, track, md, info, lights, lm, zones, openz, occ, T)
    if "stage" not in info:
        info.update({k: v for k, v in no_concert_info(track).items() if k not in info})
    # ---------------------------------------------------------------- la ruta: asfalto, lineas y orillas
    # donde la ruta recorre otra vez la MISMA avenida (paralela y encimada) se dibuja una sola capa de asfalto
    dup = np.zeros(n, bool)
    for i in np.nonzero(track.partner >= 0)[0]:
        j = track.partner[i]
        if i > j and track.partner_d[i] < 3.0 and abs(tan[i] @ tan[j]) > 0.85 and rw[i] <= rw[j] + 0.6:
            dup[i] = True
    dup = np.convolve(dup.astype(float), np.ones(3), "same") > 1.5          # sin huecos sueltos
    for r in (hd.runs(~dup) if dup.any() else [np.append(np.arange(n), 0)]):
        lat5 = np.outer([-1, -0.5, 0, 0.5, 1], rw[r])
        road_ribbon(g, track, r, lat5, (h[r] + 0.06)[None, :], rgb(58, 58, 62), mat(hd.M_ASPH, 0.7),
                    uv_scale=(7.0 / rw[r])[None, :])
    two_way = ~track.ow & ~dup
    if track.race.any():                                                  # pianos rojo/blanco en las curvas
        curvy = (np.abs(track.curv) > 1 / 60) & track.race
        curvy = np.convolve(curvy.astype(float), np.ones(9), "same") > 0
        for side in (-1, 1):
            for r in hd.runs(curvy):
                road_ribbon(g, track, r, np.array([rw[r] - 1.1, rw[r] - 0.6, rw[r]]) * side,
                            np.array([h[r] + 0.075, h[r] + 0.13, h[r] + 0.09]), (1, 1, 1), mat(hd.M_CURB, 0.45),
                            uv_scale=np.ones((1, len(r))))
    s_ = np.arange(n) * track.spacing
    for sgn in (-1, 1):                                                   # doble linea amarilla al centro
        m = two_way.copy()
        for r in hd.runs(m):
            road_ribbon(g, track, r, np.array([[sgn * 0.1] * len(r), [sgn * 0.24] * len(r)]),
                        (h[r] + 0.078)[None, :], (0.95, 0.75, 0.1), mat(M_LINE, 0.55))
    dash = (s_ % 9.0) < 4.5
    for k in range(1, 6):                                                 # lineas de carril punteadas
        lanes = track.ln
        m = (lanes > k) & dash & ~((two_way) & (lanes == 2 * k)) & ~dup
        for r in hd.runs(m):
            lat = -rw[r] + 2 * rw[r] * k / lanes[r]
            road_ribbon(g, track, r, np.array([lat - 0.07, lat + 0.07]), (h[r] + 0.078)[None, :], (0.92, 0.92, 0.92),
                        mat(M_LINE, 0.55))
    # ---------------------------------------------------------------- puentes, terraplenes y deprimidos de la ruta
    elev = off > 0.4
    sunkm = off < -0.4
    conc, conc2 = rgb(176, 174, 168), rgb(150, 148, 142)
    for side in (-1, 1):
        sh_ = shared[:, 1 if side > 0 else 0]
        for r in hd.runs(elev & ~sh_):
            e = rw[r] + 1.6
            road_ribbon(g, track, r, np.array([rw[r], rw[r] + 1.0]) * side, (h[r] + 0.06)[None, :],
                        rgb(64, 64, 68), mat(M_CONC, 0.8))
            road_ribbon(g, track, r, np.array([rw[r] + 1.0, rw[r] + 1.0, e, e]) * side,
                        np.array([h[r] + 0.06, h[r] + 0.95, h[r] + 0.95, h[r] - 1.25]), conc, mat(M_CONC, 0.8))
        for r in hd.runs(elev & ~track.br & ~sh_):                        # terraplen: muro hasta el suelo
            e = rw[r] + 1.6
            road_ribbon(g, track, r, np.array([e, e]) * side, np.array([h[r] - 1.2, T(pts[r]) - 0.6]), conc2,
                        mat(M_CONC, 0.85))
        for r in hd.runs(sunkm & ~sh_):                                   # deprimido: muros de contencion
            e = rw[r] + 1.6
            top = np.maximum(T(pts[r]) + 1.0, h[r] + 1.0)
            road_ribbon(g, track, r, np.array([rw[r], e, e, e + 0.6]) * side,
                        np.array([h[r] + 0.06, h[r] + 0.06, top, top]), conc2, mat(M_CONC, 0.85))
    for r in hd.runs(elev & track.br):                                    # puente: losa y VIGAS a la vista
        e = rw[r] + 1.6
        road_ribbon(g, track, r, np.array([-e, e]), (h[r] - 1.25)[None, :], conc2, mat(M_CONC, 0.8),
                    orient=(0, -1, 0))
        nb = 5
        for b in range(nb):
            lc = -rw[r] * 0.85 + rw[r] * 1.7 * b / (nb - 1)
            for side in (-1, 1):
                road_ribbon(g, track, r, np.array([lc + side * 0.22, lc + side * 0.22]),
                            np.array([h[r] - 1.25, h[r] - 2.45]), conc2, mat(M_CONC, 0.8))
            road_ribbon(g, track, r, np.array([lc - 0.45, lc + 0.45]), (h[r] - 2.45)[None, :], conc2,
                        mat(M_CONC, 0.8), orient=(0, -1, 0))
        for k in r[::9]:                                                  # cabezales y columnas
            y_top = h[k] - 2.45
            yg = yG(*pts[k])
            if y_top - yg < 1.5:
                continue
            fbox(g, pts[k], nrm[k], rw[k] + 0.8, 0.9, y_top - 1.1, y_top, conc, mat(M_CONC, 0.8), bottom=True)
            for lc in (-rw[k] * 0.55, rw[k] * 0.55):
                q = pts[k] + nrm[k] * lc
                fcyl(g, q[0], q[1], yg - 1.0, y_top - 1.1, 0.75, conc, mat(M_CONC, 0.8), seg=12, cap=False)
        for k in r[::12]:
            for side in (-1, 1):
                q = pts[k] + nrm[k] * side * (rw[k] + 1.45)
                fcyl(g, q[0], q[1], h[k] + 0.95, h[k] + 9.0, 0.11, rgb(85, 92, 98), mat(M_METAL, 0.45, 0.8), seg=6)
                q2 = q - nrm[k] * side * 2.2
                fbox(g, q2, tan[k], 0.5, 0.25, h[k] + 8.75, h[k] + 9.0, (1.0, 0.78, 0.45), mat(M_LAMP, 0.5, 0, 7.0),
                     bottom=True)
            lights.append((float(pts[k][0]), h[k] + 8.0, float(pts[k][1]), 34.0, 1.0, 0.75, 0.45, 3.0))
    for r in hd.runs(track.tu):                                           # tunel: techo y lamparas
        e = rw[r] + 1.2
        road_ribbon(g, track, r, np.array([-e, e]), (h[r] + 5.6)[None, :], conc2, mat(M_CONC, 0.85),
                    orient=(0, -1, 0))
        for k in r[::5]:
            fbox(g, pts[k], nrm[k], 0.6, 0.3, h[k] + 5.3, h[k] + 5.55, (1.0, 0.85, 0.6), mat(M_LAMP, 0.4, 0, 8.0),
                 bottom=True)
            lights.append((float(pts[k][0]), h[k] + 4.8, float(pts[k][1]), 18.0, 1.0, 0.8, 0.55, 3.0))
    # ---------------------------------------------------------------- otras calles reales
    bidx = BuildingIndex([b["p"] for b in d.get("bld", [])])
    park_polys = [np.array(a["p"], float) for a in d.get("areas", []) if a["k"] == "bosque"]

    def in_park(Q):
        Q = np.asarray(Q, float).reshape(-1, 2)
        out = np.zeros(len(Q), bool)
        for P_ in park_polys:
            out |= inside_poly(P_, Q)
        return out
    sidx_ = StreetIndex()

    def clear_of_roads(Q, margin):
        """Ningun punto sobre la ruta (con su ancho real) ni sobre una calle dibujada."""
        Q = np.asarray(Q, float).reshape(-1, 2)
        ii_, _, dd_ = nearest_route(track, rg, Q, 40)
        okr = (ii_ < 0) | (dd_ > rw[np.maximum(ii_, 0)] + margin)
        return bool(okr.all()) and not sidx_.hit(Q, 0.4)
    ways = []
    for w in d.get("ways", []):                                           # puntos cada 4 m (OSM los tiene muy separados)
        P = np.array(w["p"], float)
        if len(P) < 2 or w["c"] in ("service",):                          # entradas a estacionamientos: no
            continue
        if w["c"] in ("residential", "unclassified", "living_street") and not w.get("r"):
            w = dict(w, w=min(float(w["w"]), 3.0))
        elif w["c"] == "pedestrian":
            w = dict(w, w=min(float(w["w"]), 2.5))
        elif w["c"] == "raceway":
            w = dict(w, w=4.5)
        elif w["c"] == "track":
            w = dict(w, w=2.2)
        elif w["c"] in ("footway", "path"):
            w = dict(w, w=1.1)
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            k = max(1, int(np.hypot(*(b - a)) / 4.0))
            Q += [a + (b - a) * (j / k) for j in range(1, k + 1)]
        ways.append(dict(w, p=np.array(Q)))
    allv = np.array([q for w in ways for q in w["p"]], float).reshape(-1, 2)
    vi, vlat, vd = nearest_route(track, rg, allv, 40)
    vy = T(allv)
    k0 = 0
    rank = {"motorway": 7, "trunk": 6, "primary": 5, "secondary": 4, "tertiary": 3, "unclassified": 2,
            "residential": 1}
    for w in ways:
        P = np.array(w["p"], float)
        m_ = len(P)
        ii, dd, yy = vi[k0:k0 + m_], vd[k0:k0 + m_], vy[k0:k0 + m_].copy()
        k0 += m_
        if w.get("t"):
            continue
        wh = float(w["w"])
        # fuera del corredor de la ruta (ahi manda la ruta)
        if w.get("r"):
            inside = (ii >= 0) & (dd < 45)                                # es la misma calle de la ruta: ya esta dibujada
        else:
            inside = (ii >= 0) & (dd < rw[np.maximum(ii, 0)] + wh * 0.8 - 0.4)
        dq0 = np.gradient(P, axis=0)
        dq0 /= np.maximum(np.hypot(dq0[:, 0], dq0[:, 1]), 1e-6)[:, None]
        nq0 = np.column_stack([-dq0[:, 1], dq0[:, 0]]) * wh * 0.85
        inside = inside | bidx.inside(P, 1.0) | bidx.inside(P + nq0, 1.0) | bidx.inside(P - nq0, 1.0)
        if w.get("b"):                                                    # paso superior / puente de otra calle
            dq_ = np.gradient(P, axis=0)
            dq_ /= np.maximum(np.hypot(dq_[:, 0], dq_[:, 1]), 1e-6)[:, None]
            ic = np.maximum(ii, 0)
            near_c = (ii >= 0) & (dd < rw[ic] + wh + 3.0)
            cross_c = near_c & (np.abs((dq_ * tan[ic]).sum(1)) < 0.7)      # atraviesa la ruta: libre de 6.8 m
            par_c = near_c & ~cross_c                                      # va junto a la ruta: abajo
            yy[cross_c] = np.maximum(yy[cross_c], (h - np.minimum(off, 0.0))[ic[cross_c]] + 6.8)
            yy[par_c] = np.minimum(yy[par_c], (h - np.maximum(off, 0.0))[ic[par_c]] - 0.15)
            if cross_c.any():
                for rng_ in (range(1, m_), range(m_ - 2, -1, -1)):         # rampas de 12 %
                    for k_ in rng_:
                        prv = k_ - 1 if rng_.step == 1 else k_ + 1
                        yy[k_] = max(yy[k_], yy[prv] - 0.12 * float(np.hypot(*(P[k_] - P[prv]))))
            inside = par_c                                                 # puente paralelo a la ruta: no se dibuja
        occ_pts = []
        nr_ = (ii >= 0) & (dd < 40)
        if nr_.any() and not w.get("b"):
            hbw = h - np.maximum(off, 0.0)
            hbw = np.minimum.reduce([hbw, np.roll(hbw, 1), np.roll(hbw, -1), np.roll(hbw, 2), np.roll(hbw, -2)])
            hbn = hbw[ii[nr_]] - 0.12 + 0.01 * np.maximum(dd[nr_] - rw[ii[nr_]], 0.0)
            yy[nr_] = np.minimum(yy[nr_], hbn)
        yy = yy + 0.03 + 0.012 * rank.get(w["c"], 0)
        keep = ~inside
        dense = (ii >= 0) & (dd < rw[np.maximum(ii, 0)] + wh + 12)          # puntos densos solo junto a la ruta
        for r in open_runs(keep):
            r = r[r < m_]
            if len(r) < 2:
                continue
            sel = dense[r] | (np.arange(len(r)) % 6 == 0)
            sel[0] = sel[-1] = True
            r = r[sel]
            Q = P[r]
            dq = np.gradient(Q, axis=0)
            dq /= np.maximum(np.hypot(dq[:, 0], dq[:, 1]), 1e-6)[:, None]
            nq = np.column_stack([-dq[:, 1], dq[:, 0]])
            PP = np.zeros((2, len(r), 3), F4)
            for c_, sg in enumerate((-1, 1)):
                PP[c_, :, 0] = Q[:, 0] + nq[:, 0] * sg * wh
                PP[c_, :, 2] = Q[:, 1] + nq[:, 1] * sg * wh
                PP[c_, :, 1] = yy[r]
            sl = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
            UV = np.stack(np.broadcast_arrays(sl[None, :], np.array([-7.0, 7.0])[:, None]), -1).astype(F4)
            if w["c"] in ("track", "footway", "path"):                    # caminos del parque: tierra / grava
                grid(g, PP, rgb(170, 152, 118), mat(M_GRAVEL, 0.95), UV=UV, orient=(0, 1, 0))
            else:
                grid(g, PP, rgb(56, 56, 60), mat(hd.M_ASPH, 0.75), UV=UV, orient=(0, 1, 0))
            if w.get("b"):                                                # puente de otra calle: losa, vigas, columnas
                for c_, sg in enumerate((-1, 1)):
                    PW = PP.copy()
                    PW[0] = PP[c_]
                    PW[1] = PP[c_]
                    PW[1, :, 1] = yy[r] - 1.4
                    grid(g, PW, conc2, mat(M_CONC, 0.8))
                for j in range(0, len(r), 3):
                    if yy[r[j]] - vy[k0 - m_ + r[j]] > 3:
                        q = Q[j]
                        fcyl(g, q[0], q[1], vy[k0 - m_ + r[j]] - 1, yy[r[j]] - 1.4, 0.7, conc, mat(M_CONC, 0.8),
                             seg=10, cap=False)
            for q in Q:
                occ_pts.append(q)
            for j in range(len(Q) - 1):                                   # la calle no se llena de edificios
                a, b = Q[j], Q[j + 1]
                L = np.hypot(*(b - a))
                if not w.get("b"):
                    sidx_.add(a, b, wh)
                if L > 0.5:
                    occ.rect((a + b) / 2, (b - a) / L, L / 2 + 1.0, wh + 2.4)
    # ---------------------------------------------------------------- edificios reales (OSM)
    bl = d.get("bld", [])
    PAL = FACADES
    allb = np.array([q for b in bl for q in b["p"]], float).reshape(-1, 2)
    bi, blat, bdist = nearest_route(track, rg, allb, 40)
    facade_pts = []                                                       # (i, lado, distancia lateral)
    k0 = 0
    shops = [(np.array(p[:2]), p[2]) for p in d.get("pois", [])]
    nb = 0
    for b in bl:
        P = np.array(b["p"], float)
        m_ = len(P)
        ii, lt, dd = bi[k0:k0 + m_], blat[k0:k0 + m_], bdist[k0:k0 + m_]
        k0 += m_
        if m_ < 4:
            continue
        cen = P.mean(0)
        near = ii >= 0
        if near.any() and np.min(np.abs(lt[near]) - rw[ii[near]]) < 1.2:
            continue                                                      # se mete a la calle: se omite
        if near.any() and np.min(dd[near]) < 30:
            E_ = np.concatenate([P[a_] + (P[a_ + 1] - P[a_]) * t_ for a_ in range(m_ - 1)
                                 for t_ in np.linspace(0, 1, max(2, int(np.hypot(*(P[a_ + 1] - P[a_])) / 2) + 1))[:, None]
                                 ]).reshape(-1, 2)
            ei, el, _ = nearest_route(track, rg, E_, 30)
            okE = ei >= 0
            if okE.any() and np.min(np.abs(el[okE]) - rw[ei[okE]]) < 1.2:
                continue

        if zones.hit(cen).any() or float(b.get("ly", 0) or 0) < 0:
            continue
        mn_b, mx_b = P.min(0) - 2, P.max(0) + 2
        rin = np.nonzero((pts[:, 0] > mn_b[0]) & (pts[:, 0] < mx_b[0]) & (pts[:, 1] > mn_b[1]) & (pts[:, 1] < mx_b[1]))[0]
        if len(rin) and inside_poly(P, pts[rin]).any():
            continue                                                      # la ruta pasa por dentro
        area = 0.5 * abs(np.dot(P[:-1, 0], P[1:, 1]) - np.dot(P[1:, 0], P[:-1, 1]))
        if area < 12:
            continue
        hh = float(b.get("h") or 0) or float(b.get("lv") or 0) * 3.2
        if hh <= 0:
            lv = (1 + rnd.random() * 1.5 if area < 100 else 2 + rnd.random() * 2.5 if area < 400 else
                  3 + rnd.random() * 4 if area < 1500 else 4 + rnd.random() * 6)
            if b.get("k") in ("house", "residential", "detached"):
                lv = min(lv, 3)
            hh = round(lv) * 3.2 + 0.6
        hh = min(hh, 260.0)
        base = float(T([cen])[0]) - 2.5
        top = base + 2.5 + hh
        rf, sk = None, None
        if near.any() and np.min(dd[near]) < 30:
            E = np.roll(P, -1, 0) - P
            mids = P + E / 2
            ri, rl, rd = nearest_route(track, rg, mids, 30)
            best = None
            for j in range(m_ - 1):
                L = np.hypot(*E[j])
                if L < 4 or ri[j] < 0 or rd[j] > 26:
                    continue
                if best is None or rd[j] < best[1]:
                    best = (j, rd[j])
            if best is not None and rnd.random() < 0.75:
                rf = best[0]
                q = mids[rf]
                base = min(base, h[ri[rf]] - 2.5)
                sk = rnd.choice(SHOP_SIGNS)
                for sp, kind in shops:
                    if np.hypot(*(sp - q)) < 18:
                        sk = POI_SIGN.get(kind, sk)
                        break
        osm_building(g, P, base, top, PAL[hash(tuple(P[0])) % len(PAL)], rnd, rf, sk,
                     floor=(h[ri[rf]] + 0.19) if rf is not None else None)
        nb += 1
        for (x, z) in P:
            occ.rect((x, z), (1, 0), 2, 2)
        for j in range(m_ - 1):                                           # fachada -> pared de la fisica
            a, b2 = P[j], P[j + 1]
            L = np.hypot(*(b2 - a))
            for t_ in np.linspace(0, 1, max(2, int(L / 1.5) + 1)):
                facade_pts.append(a + (b2 - a) * t_)
    print(f"   edificios reales: {nb}")
    wpos = np.full(n, np.inf)
    wneg = np.full(n, np.inf)
    if facade_pts:
        FP = np.array(facade_pts)
        fi, fl, fd = nearest_route(track, rg, FP, 18)
        ok = (fi >= 0) & (np.abs(fl) < rw[np.maximum(fi, 0)] + 14)
        for i, l in zip(fi[ok], fl[ok]):
            if l > 0:
                wpos[i] = min(wpos[i], l)
            else:
                wneg[i] = min(wneg[i], -l)
    # ---------------------------------------------------------------- bocacalles: cruces a nivel
    mouth = np.zeros((n, 2), bool)                                        # [:,0] lado negativo, [:,1] positivo
    inters = []
    for j, sd, ci, nm in d.get("inter", []):
        if abs(off[j]) > 0.4:
            continue
        half_m = int(math.ceil((SIDEWALK.get(ci, 2.3) * 2 + 7.0) / track.spacing / 2)) + 1
        idxs = [(j + q) % n for q in range(-half_m, half_m + 1)]
        if sd in (0, 1):
            mouth[idxs, 1] = True
        if sd in (0, -1):
            mouth[idxs, 0] = True
        inters.append((j, sd, ci))
    # ---------------------------------------------------------------- puentes peatonales reales
    pb_done = []
    for i, ang in d.get("pedb", [])[:30]:
        if abs(off[i]) > 0.4 or any(abs(i - j) < 30 for j in pb_done):
            continue
        ped_bridge_osm(g, crowd, rnd, track, i, ang, lights, h[i], occ)
        pb_done.append(i)
    # ---------------------------------------------------------------- manzanas a lo largo de la ruta (donde OSM no tiene edificios)
    sw_des = np.array([SIDEWALK.get(int(c), 2.3) for c in track.cls])
    stories_by = {0: [6, 8, 10], 1: [4, 6, 8], 2: [3, 4, 5, 6, 8], 3: [2, 3, 4, 5, 6], 4: [2, 3, 3, 4],
                  5: [1, 2, 2, 3], 6: [1, 2, 2, 3], 7: [1, 2, 2], 8: [1, 2]}
    placed_fronts = []
    for side in (-1, 1):
        wall = wpos if side > 0 else wneg
        s = 0.0
        while s < track.L - 4:
            wdt = rnd.uniform(8.0, 20.0)
            dp = rnd.uniform(10.0, 20.0)
            i = int(((s + wdt / 2) % track.L) / track.spacing) % n
            span = range(i - int(wdt / 2 / track.spacing) - 1, i + int(wdt / 2 / track.spacing) + 2)
            spi = [q % n for q in span]
            fd = rw[i] + 0.3 + sw_des[i]
            bad = (np.abs(off[spi]).max() > 0.4 or mouth[spi, 1 if side > 0 else 0].any() or
                   np.isfinite(wall[spi]).any() or track.cls[i] <= 1 or track.cls[i] >= 10 or
                   (trop and not town_m[i]) or
                   (park_polys and in_park(pts[i] + nrm[i] * side * (rw[i] + 8))[0]))
            if bad:
                s += 3.0
                continue
            x, z, tx, tz = track.point(s + wdt / 2)
            f, o = np.array([tx, tz]), nrm[i] * side
            c = np.array([x, z]) + o * (fd + dp / 2)
            corners = np.array([c + f * a + o * b for a in (-wdt / 2, 0.0, wdt / 2) for b in (-dp / 2, 0.0, dp / 2)])
            cells = occ.cells(c, f, wdt / 2, dp / 2)
            if (track.nearest_dist(corners) >= fd - 0.6).all() and occ.free(cells) and \
                    clear_of_roads(footprint_probe(c, f, wdt / 2, dp / 2), 0.8) and \
                    not (zones.hit(corners) | openz.hit(corners)).any():
                hh = rnd.choice(stories_by.get(int(track.cls[i]), [2, 3])) * 3.6 + 0.4
                corner = any(abs(((i - j) + n // 2) % n - n // 2) * track.spacing < 14 for j, _, _ in inters)
                yb = float(h[i]) + 0.19
                col_ = rnd.choice(FACADES)
                tg = Geo()
                building_hd(tg, OffsetCrowd(crowd, yb), rnd, c, f, o, wdt / 2, dp / 2, hh, col_,
                            shop=corner or rnd.random() < 0.8, roof_fans=hh < 14 and rnd.random() < 0.06)
                g.merge(lift_geo(tg, yb))
                yg_ = float(T([c])[0])
                fbox(g, c, f, wdt / 2, dp / 2, min(yg_, yb) - 3.0, yb + 0.02, tuple(v * 0.8 for v in col_),
                     mat(M_CONC, 0.8))
                if corner:                                                # la tiendita de la esquina
                    sign_quad(g, c - o * (dp / 2 + 0.07), -o, min(wdt / 2 - 0.8, 4.5), yb + 3.3, yb + 4.2,
                              rnd.choice(CORNER_SIGNS))
                occ.add(cells)
                for q in spi:
                    wall[q] = min(wall[q], fd)
                placed_fronts.append((c - o * dp / 2, -o, hh + yb))
                s += wdt + rnd.uniform(0.0, 1.2)
            else:
                s += 3.0
    # ---------------------------------------------------------------- paredes finales, camellones, vallas y banquetas
    for wall, sidx in ((wpos, 1), (wneg, 0)):
        wall[:] = np.where(np.isfinite(wall), wall, rw + 0.3 + sw_des + 0.8)
        wall[mouth[:, sidx]] = np.maximum(wall[mouth[:, sidx]], rw[mouth[:, sidx]] + 4.5)
        wall[:] = np.minimum.reduce([np.roll(wall, q) for q in (-1, 0, 1)])
        wall[:] = np.clip(wall, rw + 1.2, rw + 14.0)
        wall[off > 0.4] = rw[off > 0.4] + 1.4                       # acotamiento + parapeto
        wall[off < -0.4] = rw[off < -0.4] + 1.4
    # camellones: otra calzada paralela muy cerca de un lado
    for side, wall in ((1, wpos), (-1, wneg)):
        par = np.zeros(n)
        for w in ways:
            if w["c"] not in ("motorway", "trunk", "primary", "secondary", "tertiary"):
                continue
            P = np.array(w["p"], float)
            ii, lt, dd = nearest_route(track, rg, P, 40)
            for k_, (i, l) in enumerate(zip(ii, lt)):
                if i < 0 or l * side <= rw[i] + 1.0 or l * side > rw[i] + 30:
                    continue
                par[i] = max(par[i], l * side) if par[i] == 0 else min(par[i], l * side)
        cam = (par > 0) & (np.abs(off) < 0.4)
        cam = np.convolve(cam.astype(float), np.ones(9), "same") > 2
        idx = np.nonzero(cam)[0]
        if len(idx):
            wall[idx] = np.minimum(wall[idx], rw[idx] + 0.6)
        for r in hd.runs(cam):
            far = np.maximum(par[r] - 4.0, rw[r] + 1.5)
            far = np.where(par[r] > 0, far, rw[r] + 2.5)
            road_ribbon(g, track, r, np.array([rw[r], rw[r], rw[r] + 0.3, far]) * side,
                        np.array([h[r] + 0.06, h[r] + 0.25, h[r] + 0.25, h[r] + 0.25]), rgb(185, 180, 165),
                        mat(M_CONC, 0.8))
            road_ribbon(g, track, r, np.array([rw[r] + 0.5, far - 0.2]) * side, (h[r] + 0.27)[None, :],
                        rgb(70, 115, 45), mat(M_GRASS, 0.95))
            for k in r[::6]:
                if rnd.random() < 0.7:
                    q = pts[k] + nrm[k] * side * (rw[k] + 0.5 + (far[0] - rw[k] - 0.5) * 0.5)
                    tg = Geo()
                    (trop_tree if trop else jacaranda_hd if rnd.random() < 0.5 else tree_hd)(tg, q[0], q[1], rnd.uniform(0.9, 1.3), rnd)
                    g.merge(lift_geo(tg, h[k] + 0.25))
    for i in np.nonzero(track.partner >= 0)[0]:                           # la calzada vecina tambien es calle
        j = track.partner[i]
        if abs(h[i] - h[j]) > 2.5:
            continue
        wall_ = wpos if track.partner_side[i] > 0 else wneg
        wall_[i] = max(wall_[i], min(track.partner_d[i] + rw[j] + 1.0, rw[i] + 14.0))
    track.wpos, track.wneg = wpos, wneg
    fac = {1: wpos.copy(), 0: wneg.copy()}
    # vallas con publico en tramos de banqueta (y junto a la salida y los lugares)
    valla = np.zeros((n, 2), bool)
    near_lm = np.zeros(n, bool)
    for v in lm.values():
        near_lm |= np.hypot(pts[:, 0] - v[0], pts[:, 1] - v[1]) < 260
    near_lm[np.arange(-50, 60) % n] = True
    for sidx in (0, 1):
        k = 0
        while k < n:
            ln_ = rnd.randint(15, 60)
            if near_lm[k] or rnd.random() < md.fans * md.fan_rate * (1.0 if town_m[k] else 0.1):
                valla[[(k + q) % n for q in range(ln_)], sidx] = True
            k += ln_ + (rnd.randint(2, 8) if md.fan_rate >= 0.9 else rnd.randint(5, 40))
    for sidx, wall in ((1, wpos), (0, wneg)):
        m = valla[:, sidx] & (wall > rw + (2.4 if md.fan_rate >= 0.9 else 3.0)) & (np.abs(off) < 0.4) &             ~mouth[:, sidx] & ~shared[:, sidx]
        valla[:, sidx] = m
        wall[m] = rw[m] + 1.3
    for wall in (wpos, wneg):                                             # la pared se angosta poco a poco
        for _ in range(8):
            wall[:] = np.maximum.reduce([wall, np.roll(wall, 1) - 0.5, np.roll(wall, -1) - 0.5])
    for sidx, wall in ((1, wpos), (0, wneg)):                             # banquetas con su guarnicion
        side = 1 if sidx else -1
        ok = (np.abs(off) < 0.4) & ~mouth[:, sidx] & ~shared[:, sidx]
        for r in hd.runs(ok & town_m):
            sw_out = np.clip(fac[sidx][r], rw[r] + 0.6, rw[r] + 0.3 + np.maximum(sw_des[r], 1.0) + 1.5)
            road_ribbon(g, track, r, np.array([rw[r], rw[r], rw[r] + 0.3, sw_out, sw_out]) * side,
                        np.array([h[r] + 0.06, h[r] + 0.19, h[r] + 0.19, h[r] + 0.19, T(pts[r]) - 0.6]),
                        rgb(178, 174, 166), mat(M_CONC, 0.85))
        for r in (hd.runs(ok & ~town_m) if trop else ()):                 # carretera: acotamiento de arena
            sw_out = rw[r] + 2.6
            road_ribbon(g, track, r, np.array([rw[r], sw_out, sw_out + 1.5]) * side,
                        np.array([h[r] + 0.06, h[r] + 0.02, np.minimum(gh(pts[r]) + 0.05, h[r] - 0.1)]),
                        rgb(204, 190, 150), mat(M_GRAVEL, 0.95))
            road_ribbon(g, track, r, np.array([rw[r] - 0.45, rw[r] - 0.3]) * side, (h[r] + 0.078)[None, :],
                        (0.95, 0.95, 0.95), mat(M_LINE, 0.55))
        for r in hd.runs(mouth[:, sidx] & (np.abs(off) < 0.4)):          # bocacalle: vallas que la cierran
            road_ribbon(g, track, r, np.array([rw[r] + 4.5, rw[r] + 4.5]) * side, np.array([h[r], h[r] + 1.2]),
                        rgb(150, 155, 160), mat(M_FENCE_, 0.4, 0.8))
            for k in r[::3]:                                              # gente que mira desde la bocacalle
                for _ in range(rnd.randint(2, 6)):
                    q = pts[k] + nrm[k] * side * rnd.uniform(rw[k] + 5.0, rw[k] + 9.0) + tan[k] * rnd.uniform(-2, 2)
                    crowd.add(q[0], h[k] + 0.05, q[1], -nrm[k, 0] * side, -nrm[k, 1] * side, rnd)
        for r in hd.runs(valla[:, sidx]):                                 # vallas con aficionados
            road_ribbon(g, track, r, np.array([rw[r] + 1.3, rw[r] + 1.3]) * side,
                        np.array([h[r] + 0.19, h[r] + 1.3]), rgb(150, 155, 160), mat(M_FENCE_, 0.4, 0.8),
                        uv_scale=None)
            road_ribbon(g, track, r, np.array([rw[r] + 1.27, rw[r] + 1.33]) * side, (h[r] + 1.3)[None, :],
                        rgb(120, 125, 130), mat(M_METAL, 0.4, 0.7))
            for k in r[::2]:
                far = min(fac[sidx][k], rw[k] + 0.3 + max(sw_des[k], 1.0) + 1.5) - 0.4
                for _ in range(int(rnd.randint(2, 5) * md.fan_density)):
                    lt = rnd.uniform(rw[k] + 1.7, max(rw[k] + 1.8, far))
                    q = pts[k] + nrm[k] * side * lt + tan[k] * rnd.uniform(-1.5, 1.5)
                    crowd.add(q[0], h[k] + 0.19, q[1], -nrm[k, 0] * side, -nrm[k, 1] * side, rnd)
        for k in range(4, n, 13):                                         # postes de luz en la orilla
            if abs(off[k]) > 0.4 or mouth[k, sidx] or shared[k, sidx] or (not town_m[k] and k % 39 != 4):
                continue
            q = pts[k] + nrm[k] * side * (rw[k] + 0.45 if not valla[k, sidx] else rw[k] + 1.6)
            fcyl(g, q[0], q[1], h[k] + 0.19, h[k] + 9.5, 0.13, rgb(85, 92, 98), mat(M_METAL, 0.45, 0.8), seg=6, r1=0.09)
            tip = q - nrm[k] * side * 2.6
            cylinder(g, (q[0], h[k] + 9.3, q[1]), (tip[0], h[k] + 9.7, tip[1]), 0.07, rgb(85, 92, 98),
                     mat(M_METAL, 0.45, 0.8), seg=5, caps=False)
            fbox(g, tip, tan[k], 0.5, 0.22, h[k] + 9.45, h[k] + 9.7, (1.0, 0.78, 0.45), mat(M_LAMP, 0.5, 0, 7.0),
                 bottom=True)
            if k % 26 == 4:
                lights.append((float(tip[0]), h[k] + 8.6, float(tip[1]), 36.0, 1.0, 0.72, 0.42, 3.2))
    # ---------------------------------------------------------------- tribunas de la salida (si hay espacio)
    if "tribunas" in md.extras:
        put = 0
        for i0 in (8, 40, n - 40):
            for side in (1, -1):
                if put >= 3:
                    break
                o = nrm[i0] * side
                wall_ = wpos if side > 0 else wneg
                base_ = pts[i0] + o * (rw[i0] + 3.2)
                L_ = 60.0
                probe = footprint_probe(base_ + tan[i0] * L_ / 2 + o * 7, tan[i0], L_ / 2, 7.5, 3.0)
                if not clear_of_roads(probe, 2.0) or bidx.inside(probe).any() or \
                        not occ.free(occ.cells(base_ + tan[i0] * L_ / 2 + o * 7, tan[i0], L_ / 2, 7.5)):
                    continue
                y0 = float(h[i0])
                placed(g, crowd, lights, y0, lambda tg, tc, tl: grandstand_hd(
                    tg, tc, rnd, base_, tan[i0], o, L_, 12, 0.9, 0.5, rnd.choice(
                        [rgb(30, 80, 190), rgb(220, 120, 30), rgb(190, 40, 40), rgb(40, 150, 70)]), 8, tl))
                occ.rect(base_ + tan[i0] * L_ / 2 + o * 7, tan[i0], L_ / 2 + 1, 8)
                put += 1
                break
    # ---------------------------------------------------------------- semaforos y cebras en los cruces
    for j, sd, ci in inters:
        if ci > 7:
            continue
        p, t_ = pts[j], tan[j]
        for a_off in (-(rw[j] + 3.5), rw[j] + 3.5):
            for wq in np.arange(-rw[j] + 0.6, rw[j] - 0.4, 1.2):
                flat_rect(g, p + t_ * a_off + nrm[j] * wq, t_, 1.6, 0.28, h[j] + 0.08, (0.93, 0.93, 0.93),
                          mat(M_LINE, 0.55))
        for side in (-1, 1):
            k = (j - int((rw[j] + 6) / track.spacing)) % n
            sb = pts[k] + nrm[k] * side * (rw[k] + 0.55)
            tg = Geo()
            semaforo_hd(tg, sb, -nrm[k] * side, max(2.0, rw[k] * 0.9), -tan[k])
            g.merge(lift_geo(tg, h[k] + 0.19))
    # ---------------------------------------------------------------- manzanas de la ciudad (relleno de bloques)
    fill = 240.0
    st_ = 26.0
    flo, fhi = (lo + 300, hi - 300) if not trop else (np.maximum(lo + 300, town_c - town_r), np.minimum(hi - 300, town_c + town_r))
    ax_ = np.arange(flo[0], fhi[0], st_)
    az_ = np.arange(flo[1], fhi[1], st_)
    FX, FZ = np.meshgrid(ax_, az_, indexing="ij")
    C = np.column_stack([FX.ravel(), FZ.ravel()]) + np.array([[rnd.uniform(-4, 4), rnd.uniform(-4, 4)]])
    Dr = track.nearest_dist(C)
    C = C[(Dr > 20) & (Dr < fill)]
    Dr = track.nearest_dist(C)
    blocked = zones.hit(C) | openz.hit(C)
    if park_polys:
        blocked |= in_park(C)
    if trop:
        blocked |= ~in_town(C) | (ds_at(C) < 20)
    CY = T(C)
    for c, dist, bl_, yb in zip(C, Dr, blocked, CY):
        if bl_:
            continue
        w_, dp_ = rnd.uniform(6, 11), rnd.uniform(6, 11)
        cells = occ.cells(c, (1, 0), w_, dp_)
        if not occ.free(cells) or not clear_of_roads(footprint_probe(c, (1, 0), w_, dp_, 3.0), 1.5):
            w_, dp_ = w_ * 0.6, dp_ * 0.6
            cells = occ.cells(c, (1, 0), w_, dp_)
            if not occ.free(cells) or not clear_of_roads(footprint_probe(c, (1, 0), w_, dp_, 3.0), 1.5):
                continue
        occ.add(cells)
        hh = rnd.choice(STORIES) * 3.4 + 0.4
        fbox(g, c, (1, 0), w_, dp_, yb - 2.0, yb + hh, rnd.choice(FACADES), mat(M_WINDOWS, 0.5, 0.05))
        if dist < 120 and rnd.random() < 0.5:
            fcyl(g, c[0] + w_ * 0.4, c[1], yb + hh, yb + hh + 1.4, 0.62, rgb(22, 22, 24), mat(M_DEF, 0.6), seg=6)
    # ---------------------------------------------------------------- parques, agua, rios y arboles
    park_budget = [500]
    for a in d.get("areas", []):
        k = a["k"]
        P = np.array(a["p"], float)
        if len(P) < 4 or k in ("runway", "taxiway", "apron", "terminal", "stadium"):
            continue
        if k == "bosque" and md.ground == "bosque":
            col = rgb(72, 110, 44)
        else:
            col = None
        col = col or {"water": rgb(36, 70, 92), "park": rgb(72, 118, 46), "garden": rgb(80, 126, 52),
               "grass": rgb(86, 130, 56), "pitch": rgb(60, 130, 50), "forest": rgb(46, 90, 36),
               "wood": rgb(46, 90, 36), "recreation_ground": rgb(78, 122, 50), "cemetery": rgb(96, 120, 70),
               "playground": rgb(170, 140, 100), "meadow": rgb(96, 130, 60), "scrub": rgb(90, 110, 60)}.get(k)
        if col is None or (trop and k in ("forest", "wood", "scrub", "meadow", "grass", "wetland")):
            continue
        if trop and not in_town(P).any() and k not in ("pitch",):
            continue
        mn_, mx_ = P.min(0), P.max(0)
        area_ = 0.5 * abs(np.dot(P[:-1, 0], P[1:, 1]) - np.dot(P[1:, 0], P[:-1, 1]))
        stp = float(np.clip(math.sqrt(max(area_, 1.0) / 250.0), 6.0, 30.0))    # mosaico pegado al piso
        ax_ = np.arange(mn_[0], mx_[0] + stp, stp)
        az_ = np.arange(mn_[1], mx_[1] + stp, stp)
        if len(ax_) < 2 or len(az_) < 2 or len(ax_) * len(az_) > 40000:
            continue
        AX, AZ = np.meshgrid(ax_, az_, indexing="ij")
        cen_ = np.column_stack([(AX[:-1, :-1] + stp / 2).ravel(), (AZ[:-1, :-1] + stp / 2).ravel()])
        insd = inside_poly(P, cen_)
        if not insd.any():
            continue
        AY = (gh(np.column_stack([AX.ravel(), AZ.ravel()])) + (0.07 if k == "water" else 0.05)).reshape(AX.shape)
        ci, cj = np.unravel_index(np.nonzero(insd)[0], (len(ax_) - 1, len(az_) - 1))
        C4 = [(ci, cj), (ci + 1, cj), (ci + 1, cj + 1), (ci, cj + 1)]
        V = np.stack([np.stack([AX[a, b], AY[a, b], AZ[a, b]], -1) for a, b in C4], 1)    # (m, 4, 3)
        V = V[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
        g.tris(V, (0.0, 1.0, 0.0), col, mat(M_DEF, 0.06, 0.25) if k == "water" else mat(M_GRASS, 0.95),
               V[:, [0, 2]])
        if k == "bosque" and md.forest:                                   # bosque del parque
            mn, mx = P.min(0), P.max(0)
            area = 0.5 * abs(np.dot(P[:-1, 0], P[1:, 1]) - np.dot(P[1:, 0], P[:-1, 1]))
            want = int(min(md.forest, area / 60))
            TP = np.column_stack([np.array([rnd.uniform(mn[0], mx[0]) for _ in range(want * 3)]),
                                  np.array([rnd.uniform(mn[1], mx[1]) for _ in range(want * 3)])])
            ri_ = np.array([rnd.randrange(n) for _ in range(want * 2)])  # la mitad, cerca del camino (se ven)
            off_ = np.array([rnd.choice((-1, 1)) * rnd.uniform(4.5, 60.0) for _ in range(want * 2)])
            TB = pts[ri_] + nrm[ri_] * (np.sign(off_) * rw[ri_] + off_)[:, None]
            TP = np.concatenate([TB, TP])
            TP = TP[inside_poly(P, TP)]
            ti_, _, td_ = nearest_route(track, rg, TP, 40)
            ok_r = (ti_ < 0) | (td_ > rw[np.maximum(ti_, 0)] + 3.5)
            TP, ty_ = TP[ok_r], gh(TP[ok_r])
            put_ = 0
            for (x, z), yt in zip(TP, ty_):
                if put_ >= want:
                    break
                if sidx_.hit([(x, z)], 1.2) or bidx.inside([(x, z)], 1.0).any():
                    continue
                if not occ.free(occ.cells((x, z), (1, 0), 1.2, 1.2)):
                    continue
                tg = Geo()
                forest_tree(tg, x, z, rnd.uniform(0.9, 1.5), rnd)
                g.merge(lift_geo(tg, yt + 0.02))
                put_ += 1
            print(f"   bosque: {put_} arboles")
            continue
        if k in ("park", "garden", "forest", "wood", "recreation_ground"):
            mn, mx = P.min(0), P.max(0)
            area = 0.5 * abs(np.dot(P[:-1, 0], P[1:, 1]) - np.dot(P[1:, 0], P[:-1, 1]))
            cnt = int(min(60, area / 400, max(0, park_budget[0])))
            park_budget[0] -= cnt
            if cnt <= 0:
                continue
            TP = np.column_stack([np.array([rnd.uniform(mn[0], mx[0]) for _ in range(cnt * 3)]),
                                  np.array([rnd.uniform(mn[1], mx[1]) for _ in range(cnt * 3)])])
            inside = inside_poly(P, TP)
            TP = TP[inside]
            if not len(TP):
                continue
            okd = track.nearest_dist(TP) > rw.max() + 4
            ty = gh(TP) + 0.3
            placed_t = 0
            for (x, z), ok_, yt in zip(TP, okd, ty):
                if placed_t >= cnt:
                    break
                if ok_ and occ.free(occ.cells((x, z), (1, 0), 1.5, 1.5)):
                    tg = Geo()
                    (trop_tree if trop else jacaranda_hd if rnd.random() < 0.2 else tree_hd)(tg, x, z, rnd.uniform(1.0, 1.6), rnd)
                    g.merge(lift_geo(tg, yt - 0.3))
                    placed_t += 1
    for wv in d.get("water", []):
        if wv.get("t", "no") != "no" or wv["k"] in ("drain",):
            continue
        P = np.array(wv["p"], float)
        if len(P) < 2:
            continue
        wh = {"river": 10.0, "canal": 6.0, "stream": 2.5}.get(wv["k"], 3.0)
        dq = np.gradient(P, axis=0)
        dq /= np.maximum(np.hypot(dq[:, 0], dq[:, 1]), 1e-6)[:, None]
        nq = np.column_stack([-dq[:, 1], dq[:, 0]])
        yy = gh(P) + 0.3
        ii, _, dd = nearest_route(track, rg, P, 40)
        clear = ~((ii >= 0) & (dd < rw[np.maximum(ii, 0)] + wh + 5))
        for r in open_runs(clear):
            r = r[r < len(P)]
            if len(r) < 2:
                continue
            PP = np.zeros((4, len(r), 3), F4)
            for c_, (lt, dy) in enumerate(((-wh - 3, 0.0), (-wh, -1.6), (wh, -1.6), (wh + 3, 0.0))):
                PP[c_, :, 0] = P[r, 0] + nq[r, 0] * lt
                PP[c_, :, 2] = P[r, 1] + nq[r, 1] * lt
                PP[c_, :, 1] = yy[r] - 0.3 + dy
            grid(g, PP[[0, 1]], rgb(160, 156, 148), mat(M_CONC, 0.85))
            grid(g, PP[[2, 3]], rgb(160, 156, 148), mat(M_CONC, 0.85))
            grid(g, PP[[1, 2]], rgb(40, 100, 125), mat(M_DEF, 0.1, 0.0), orient=(0, 1, 0))
    trees = np.array(d.get("trees", []), float).reshape(-1, 2)[:350]
    if len(trees):
        ti, tl_, td = nearest_route(track, rg, trees, 30)
        ty = T(trees)
        for (x, z), i, l, dd, yt in zip(trees, ti, tl_, td, ty):
            if i >= 0:
                wall = wpos[i] if l > 0 else wneg[i]
                if abs(l) < wall + 0.8:
                    continue
            if not occ.free(occ.cells((x, z), (1, 0), 1.0, 1.0)):
                continue
            tg = Geo()
            (trop_tree if trop else jacaranda_hd if rnd.random() < 0.45 else tree_hd)(tg, x, z, rnd.uniform(0.9, 1.3), rnd)
            g.merge(lift_geo(tg, yt - 0.3))
    if trop:
        tropical_osm(g, track, md, rnd, lights, info, town_m, ds_at, gh, sea_y, wpos, wneg, mouth, shared,
                     sidx_, bidx, occ, zones, rg, lm, town_c)
    # ---------------------------------------------------------------- salida, adornos, horizonte, montanas
    start_arch(g, track, info, lights)
    if "papel" in md.extras or "banderines" in md.extras:
        for k, i in enumerate(range(30, n, 70)):
            if abs(off[i]) > 0.4 or not town_m[i]:
                continue
            tg = Geo()
            pa, pb = pts[i] - nrm[i] * (rw[i] + 0.4), pts[i] + nrm[i] * (rw[i] + 0.4)
            if "banderines" in md.extras:
                banderines_hd(tg, pa, pb, h=7.2, rows=2, k0=k)
            else:
                papel_picado_hd(tg, pa, pb, h=7.2, rows=3)
            g.merge(lift_geo(tg, h[i]))
    span = float(np.max(hi - lo)) / 2
    for _ in range(0 if trop else 300):
        a = rnd.uniform(0, TAU)
        r = rnd.uniform(span + 200, span + 1800)
        x, z = cxm + r * math.cos(a), czm + r * math.sin(a)
        if track.nearest_dist(np.array([[x, z]]))[0] < 350:
            continue
        w_, dp_ = rnd.uniform(25, 60), rnd.uniform(25, 60)
        hh = rnd.choice([12, 18, 25, 35, 50, 70, 90])
        fbox(g, (x, z), (1, 0), w_ / 2, dp_ / 2, gmin - 1, gmin + hh, rnd.choice(FACADES), mat(M_WINDOWS, 0.4, 0.1))
    if md.mountains:
        for k in range(26):
            a = k / 26 * TAU + rnd.uniform(-0.1, 0.1)
            r = span + 2600 + rnd.uniform(0, 900)
            tg = Geo()
            mountain_hd(tg, cxm + r * math.cos(a), czm + r * math.sin(a), rnd.uniform(800, 1400),
                        rnd.uniform(300, 650), rnd)
            g.merge(lift_geo(tg, gmin))
    if md.volcanoes:
        tg = Geo()
        d0 = span + 3300
        mountain_hd(tg, cxm + d0, czm + 1000, 1900, 1350, rnd, snow=1.0, seg=36, rings=14)
        mountain_hd(tg, cxm + d0 - 300, czm - 1000, 2300, 1100, rnd, snow=1.0, seg=36, rings=14)
        g.merge(lift_geo(tg, gmin))
    cams = []
    for i in range(0, n, 40):
        side = 1 if (i // 40) % 2 else -1
        wall = wpos if side > 0 else wneg
        c = pts[i] + nrm[i] * side * max(rw[i] + 0.6, wall[i] - 0.8)
        cams.append((float(c[0]), 5.5 + max(0.0, off[i]) * 0.0, float(c[1])))
    info["tvcams"] = cams
    if not gc.parts:
        quad(gc, (0, -500, 0), (0.1, -500, 0), (0.1, -500, 0.1), (0, -500, 0.1), (0, 0, 0), mat(0))
    return g, gc, crowd, lights, info


def tropical_osm(g, track, md, rnd, lights, info, town_m, ds_at, gh, sea_y, wpos, wneg, mouth, shared,
                 sidx_, bidx, occ, zones, rg, lm, town_c):
    """Cozumel real: selva del lado de tierra, palmeras y palapas del lado del mar, cruceros y faro."""
    pts, nrm, tan, n = track.pts, track.nrm, track.tan, track.n
    rw, h, off = track.rw, track.h, track.off
    step = max(1, int(round(15.0 / track.spacing)))
    n_p = n_t = n_pa = 0

    def free(q, need):
        ii_, _, dd_ = nearest_route(track, rg, q[None], 40)
        if ii_[0] >= 0 and dd_[0] < rw[ii_[0]] + need:
            return False
        return (not sidx_.hit(q[None], 1.5) and not bidx.inside(q[None], 1.0).any() and
                occ.free(occ.cells(q, (1, 0), 1.2, 1.2)) and not zones.hit(q[None]).any())
    noise = np.convolve(np.array([rnd.uniform(-1, 1) for _ in range(n)]), np.ones(5) / 5, "same") * 7.0
    for side, wall in ((1, wpos), (-1, wneg)):
        sx = 1 if side > 0 else 0
        o = nrm * side
        base = wall + 1.0
        far = ds_at(pts + o * (base + 70)[:, None])
        near = ds_at(pts + o * (base + 8)[:, None])
        okk = ~town_m & (np.abs(off) < 0.4) & ~mouth[:, sx] & ~shared[:, sx]
        beach = okk & (far < 30)
        jung = okk & (far >= 30) & (near > 5)
        # muro de selva (barato y tupido) detras de los arboles
        L0 = base + 13.0
        Qc = pts + o * (L0 + 25)[:, None]
        clr = track.nearest_dist(Qc) > L0 + 10
        canopy = jung & clr & (ds_at(pts + o * L0[:, None]) > 12)
        canopy = np.convolve(canopy.astype(float), np.ones(7), "same") > 6
        for r in hd.runs(canopy):
            if len(r) < 10:
                continue
            r = np.append(r[:-1:4], r[-1])
            Qa, Qb = pts[r] + o[r] * L0[r][:, None], pts[r] + o[r] * (L0[r] + 40)[:, None]
            ya, yb = gh(Qa), gh(Qb)
            top = np.maximum(ya, yb) + 9.0 + noise[r]
            road_ribbon(g, track, r, np.stack([L0[r], L0[r] - 1.5, L0[r] + 6, L0[r] + 40]) * side,
                        np.stack([ya - 0.8, top - 3.0, top, top - 1.0]), rgb(28, 74, 26), mat(M_LEAF, 0.85),
                        orient=np.column_stack([-o[r, 0], np.full(len(r), 0.6), -o[r, 1]])[None])
        for i in range(0, n, step):
            if beach[i]:
                for _ in range(2):
                    if rnd.random() < 0.5:
                        q = pts[i] + o[i] * rnd.uniform(base[i] + 2, base[i] + 34) + tan[i] * rnd.uniform(-4, 4)
                        if ds_at(q)[0] < 4 or not free(q, 3.0):
                            continue
                        tg = Geo()
                        if rnd.random() < 0.12:
                            palapa_hd(tg, q[0], q[1], rnd)
                            n_pa += 1
                        else:
                            palm_hd(tg, q[0], q[1], rnd.uniform(0.85, 1.3), rnd,
                                    lean=math.atan2(o[i, 1], o[i, 0]) + rnd.uniform(-0.6, 0.6))
                            n_p += 1
                        g.merge(lift_geo(tg, float(gh(q)[0]) - 0.1))
            elif jung[i]:
                for _ in range(1 if rnd.random() < 0.6 else 2):
                    lt = rnd.uniform(base[i] + 3, base[i] + 12)
                    q = pts[i] + o[i] * lt + tan[i] * rnd.uniform(-4, 4)
                    if ds_at(q)[0] < 6 or not free(q, 3.0):
                        continue
                    tg = Geo()
                    if rnd.random() < 0.22:
                        palm_hd(tg, q[0], q[1], rnd.uniform(0.8, 1.15), rnd)
                        n_p += 1
                    else:
                        jungle_tree(tg, q[0], q[1], rnd.uniform(0.85, 1.35), rnd)
                        n_t += 1
                    g.merge(lift_geo(tg, float(gh(q)[0]) - 0.1))
    # palmeras en el pueblo (malecon y camellones)
    for i in range(0, n, step * 2):
        if not town_m[i] or abs(off[i]) > 0.4:
            continue
        for side, wall in ((1, wpos), (-1, wneg)):
            if rnd.random() < 0.35:
                q = pts[i] + nrm[i] * side * (wall[i] - 0.6)
                if free(q, 0.8):
                    tg = Geo()
                    palm_hd(tg, q[0], q[1], rnd.uniform(0.8, 1.05), rnd)
                    g.merge(lift_geo(tg, float(h[i]) + 0.19))
                    n_p += 1
    # cruceros frente al pueblo
    put = []
    for _ in range(600):
        a, rr = rnd.uniform(0, TAU), rnd.uniform(400, 3200)
        q = town_c + np.array([math.cos(a), math.sin(a)]) * rr
        d0 = float(ds_at(q)[0])
        if not (-420 < d0 < -220) or any(np.hypot(*(q - p_)) < 500 for p_ in put):
            continue
        gdx = float(ds_at(q + [5, 0])[0] - ds_at(q - [5, 0])[0])
        gdz = float(ds_at(q + [0, 5])[0] - ds_at(q - [0, 5])[0])
        cruise_ship_hd(g, q, (-gdz, gdx) if abs(gdx) + abs(gdz) > 1e-6 else (1, 0), rnd.uniform(260, 320), sea_y,
                       lights)
        put.append(q)
        if len(put) >= 3:
            break
    # faro de Punta Sur
    if "puntasur" in lm:
        c = fit_spot(track, lm["puntasur"], rw.max() + 12, zones, search=300)
        faro_hd(g, c, float(gh(c)[0]) - 0.2, lights)
        zones.circle(c, 8)
    print(f"   selva: {n_t} arboles, {n_p} palmeras, {n_pa} palapas, {len(put)} cruceros")


def build_osm_landmarks(g, gc, crowd, rnd, track, md, info, lights, lm, zones, openz, occ, T):
    """Los lugares reales de cada mapa, en su posicion real y a la altura del suelo."""
    pts = track.pts

    def gy(c):
        return float(T([c])[0])

    def clear_of_route(c, need):
        return float(track.nearest_dist(np.asarray(c, float)[None])[0]) >= need
    v = md.venue or {}
    for key in md.landmarks:
        kind = key[0] if isinstance(key, tuple) else key
        if kind == "estadio_concierto" and "fraynano" in lm:                 # Fray Nano: beisbol + Limp Bizkit
            c0 = lm["fraynano"]
            dmax = float(track.nearest_dist(c0[None])[0])
            rmax = max(45.0, min(105.0, dmax - track.rw.max() - 8))
            rows = 24
            r0 = rmax - rows
            sc = c0 if clear_of_route(c0, rmax + 12) else fit_spot(track, c0, rmax + 14, zones)
            y0 = gy(sc)
            i_n = int(np.argmin(np.hypot(*(pts - sc).T)))
            tt = (pts[i_n] - sc) / max(np.hypot(*(pts[i_n] - sc)), 1e-6)

            def fn(tg, tc, tl):
                ang = np.linspace(0, TAU, 65)
                for j in range(64):
                    quad(tg, (sc[0], 0.03, sc[1]), (sc[0], 0.03, sc[1]),
                         (sc[0] + (r0 + 0.5) * math.cos(ang[j + 1]), 0.03, sc[1] + (r0 + 0.5) * math.sin(ang[j + 1])),
                         (sc[0] + (r0 + 0.5) * math.cos(ang[j]), 0.03, sc[1] + (r0 + 0.5) * math.sin(ang[j])),
                         rgb(150, 146, 138), mat(M_CONC, 0.75))
                baseball_field_hd(tg, sc, r0, tt)
                rm, Hb = bowl_hd(tg, tc, rnd, track, sc, r0, tl, rows=rows, dd=1.0, dh=0.55, inst_rows=10)
                for k in range(6):
                    a = math.atan2(tt[1], tt[0]) + (k + 0.5) / 6 * TAU
                    q = sc + np.array([math.cos(a), math.sin(a)]) * (rm + 2.5)
                    fcyl(tg, q[0], q[1], 0, Hb + 30, 0.7, rgb(140, 145, 150), mat(M_METAL, 0.4, 0.9), seg=10, r1=0.45)
                    fq = -np.array([math.cos(a), math.sin(a)])
                    frame_box(tg, q + fq * 0.7, (-fq[1], fq[0]), (4.6, 0.1), (1.0, 0.95, 0.85),
                              mat(M_LAMP, 0.4, 0, 8.0), Hb + 29.5, Hb + 33.5)
                    tl.append((float(q[0] + fq[0] * 20), Hb + 20, float(q[1] + fq[1] * 20), 90.0, 1.0, 0.95, 0.85, 4.0))
                sign_quad(tg, sc + tt * (rm + 0.6), tt, 20, Hb - 6, Hb + 2, 9)
                sign_quad(tg, sc + tt * (rm + 0.6) + np.array([-tt[1], tt[0]]) * 26, tt, 12, Hb - 7, Hb - 1, md.band_sign)
                for _ in range(900):
                    a, rr = rnd.uniform(0, TAU), rnd.uniform(rm + 3, rm + 22)
                    x, z = sc[0] + rr * math.cos(a), sc[1] + rr * math.sin(a)
                    if clear_of_route((x, z), track.rw.max() + 4):
                        tc.add(x, 0.0, z, sc[0] - x, sc[1] - z, rnd)
                return rm, Hb
            rm, Hb = placed(g, crowd, lights, y0, fn)
            if md.concert:
                tg, inf = Geo(), {}
                build_concert_hd(tg, OffsetCrowd(crowd, y0), rnd, track, sc, r0, rm, Hb, inf)
                gc.merge(lift_geo(tg, y0))
                shift_info(inf, y0)
                info.update(inf)
            info["stadium"] = (sc, rm, r0)
            zones.circle(sc, rm + 8)
            openz.circle(sc, rm + 24)
        elif kind in ("gnp", "palacio") and kind in lm:
            c0 = lm[kind]
            if kind == "gnp":
                dmax = float(track.nearest_dist(c0[None])[0])
                r0 = max(50.0, min(95.0, dmax - 60))
                sc = c0
                y0 = gy(sc)

                def fn(tg, tc, tl):
                    ang = np.linspace(0, TAU, 65)
                    for j in range(64):
                        quad(tg, (sc[0], 0.03, sc[1]), (sc[0], 0.03, sc[1]),
                             (sc[0] + r0 * math.cos(ang[j + 1]), 0.03, sc[1] + r0 * math.sin(ang[j + 1])),
                             (sc[0] + r0 * math.cos(ang[j]), 0.03, sc[1] + r0 * math.sin(ang[j])), rgb(60, 125, 50),
                             mat(M_GRASS, 0.9))
                    rm, Hb = bowl_hd(tg, tc, rnd, track, sc, r0, tl, rows=30, dd=1.0, dh=0.5, inst_rows=4)
                    i_n = int(np.argmin(np.hypot(*(pts - sc).T)))
                    tt = (pts[i_n] - sc) / max(np.hypot(*(pts[i_n] - sc)), 1e-6)
                    sign_quad(tg, sc + tt * (rm + 3.6), tt, 30, Hb + 1, Hb + 9, 12)
                    return rm, Hb
                rm, Hb = placed(g, crowd, lights, y0, fn)
                if md.concert:
                    tg = Geo()
                    build_concert_hd(tg, OffsetCrowd(crowd, y0), rnd, track, sc, r0, rm, Hb, {})
                    remap_sign(tg, 10, 2)
                    gc.merge(lift_geo(tg, y0))
                zones.circle(sc, rm + 10)
            else:
                y0 = gy(c0)
                placed(g, crowd, lights, y0, lambda tg, tc, tl: palacio_hd(tg, c0, 70.0, tl))
                zones.circle(c0, 80)
        elif kind == "aicm":
            rw_ = [a for a in track.osm.get("areas", []) if a["k"] == "runway"]
            if rw_:
                best = max(rw_, key=lambda a: np.hypot(*(np.array(a["p"][-1]) - np.array(a["p"][0]))))
                P = np.array(best["p"], float)
                a0, a1 = P[0], P[-1]
                if a0[0] > a1[0]:
                    a0, a1 = a1, a0
                dv = (a1 - a0) / np.linalg.norm(a1 - a0)
                y0 = gy(a0)
                res = {}

                def fn(tg, tc, tl):
                    res["f"] = airport_hd(tg, rnd, a0, dv, tl, zones, track, openz)
                placed(g, crowd, lights, y0, fn)
                flights, plane = res["f"]
                for fl in flights:
                    fl["y0"] = y0
                info["flights"], info["plane_geo"] = flights, plane
        elif kind == "museo" and "museo" in lm:
            c0 = lm["museo"]
            c = c0 if clear_of_route(c0, 70) else fit_spot(track, c0, 70, zones)
            i_n = int(np.argmin(np.hypot(*(pts - c).T)))
            placed(g, crowd, lights, gy(c), lambda tg, tc, tl: museo_hd(tg, c, track.tan[i_n], tl))
            zones.rect(c, track.tan[i_n], 83, 51)
        elif kind == "castillo" and "castillo" in lm:
            c = lm["castillo"]
            r = max(40.0, float(track.nearest_dist(c[None])[0]) - track.rw.max() - 10)
            placed(g, crowd, lights, gy(c) - 2, lambda tg, tc, tl: castillo_hd(tg, rnd, c, r, tl))
            zones.circle(c, r)
        elif kind == "angel" and "angel" in lm:
            c = lm["angel"]
            if not clear_of_route(c, 12):
                c = fit_spot(track, c, 14, zones)
            y0 = gy(c)

            def fn(tg, tc, tl):
                angel_hd(tg, c[0], c[1], tl)
                for _ in range(350):
                    a, rr = rnd.uniform(0, TAU), rnd.uniform(9, 30)
                    x, z = c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)
                    if clear_of_route((x, z), track.rw.max() + 3):
                        tc.add(x, 0.1, z, c[0] - x, c[1] - z, rnd)
            placed(g, crowd, lights, y0, fn)
            info["angel"] = c
            zones.circle(c, 12)
        elif kind == "monumento":
            c = lm.get("monumento")
            if c is None:
                continue
            i = int(np.argmin(np.hypot(*(pts - c).T)))
            if np.hypot(*(pts[i] - c)) < 45:
                placed(g, crowd, lights, track.h[i], lambda tg, tc, tl: monumento_rev_hd(tg, track, i, tl))
                zones.circle(pts[i], LIMIT + 12)
        elif kind == "basilica" and "basilica" in lm:
            c0 = lm["basilica"]
            c = c0 if clear_of_route(c0, 100) else fit_spot(track, c0, 100, zones)
            i_n = int(np.argmin(np.hypot(*(pts - c).T)))
            f = (pts[i_n] - c) / max(np.hypot(*(pts[i_n] - c)), 1e-6)
            placed(g, crowd, lights, gy(c), lambda tg, tc, tl: basilica_hd(tg, tc, rnd, c, f, tl))
            zones.circle(c, 100)
        elif kind == "fanfest" and (v.get("lm", "monumento") in lm or v.get("lm") == "start"):
            r0 = float(v.get("r0", 45))
            if v.get("lm") == "start":                                     # junto a la salida
                k_ = int(v.get("at", 60)) % track.n
                hint = pts[k_] + track.nrm[k_] * v.get("side", 1) * (r0 + 30)
            else:
                hint = lm[v.get("lm", "monumento")] + np.array(v.get("off", (120.0, 60.0)))
            sc = fit_spot(track, hint, r0 + 8 + track.rw.max() + 4, zones, search=500)
            y0 = gy(sc)
            tg, gct, inf, tl = Geo(), Geo(), {}, []
            fanfest_hd(tg, gct, OffsetCrowd(crowd, y0), rnd, track, sc, r0, md, inf, tl)
            g.merge(lift_geo(tg, y0))
            gc.merge(lift_geo(gct, y0))
            lights += [(l[0], l[1] + y0) + tuple(l[2:]) for l in tl]
            shift_info(inf, y0)
            info.update(inf)
            zones.circle(sc, r0 + 10)
            openz.circle(sc, r0 + 14)
        elif kind == "feria" and "jardin" in lm:
            r0 = float(v.get("r0", 34))
            sc = fit_spot(track, lm["jardin"] + np.array([-160.0, 120.0]), r0 + 62 + track.rw.max() + 3, zones,
                          search=600)
            y0 = gy(sc)
            tg, gct, inf, tl = Geo(), Geo(), {}, []
            feria_hd(tg, gct, OffsetCrowd(crowd, y0), rnd, track, sc, r0, md, inf, tl)
            g.merge(lift_geo(tg, y0))
            gc.merge(lift_geo(gct, y0))
            lights += [(l[0], l[1] + y0) + tuple(l[2:]) for l in tl]
            shift_info(inf, y0)
            for sp in inf.get("spinners", []):
                x, y, z = sp["c"]
                sp["c"] = (x, y + y0, z)
            info.update(inf)
            zones.circle(sc, r0 + 64)
        elif kind == "jardin" and "jardin" in lm:
            c0 = lm["jardin"]
            r = 40.0
            c = c0 if clear_of_route(c0, r + track.rw.max() + 2) else fit_spot(track, c0, r + 10, zones)
            i_n = int(np.argmin(np.hypot(*(pts - c).T)))
            f = (pts[i_n] - c) / max(np.hypot(*(pts[i_n] - c)), 1e-6)
            placed(g, crowd, lights, gy(c), lambda tg, tc, tl: jardin_hd(tg, tc, rnd, c, r, f, tl))
            zones.circle(c, r + 2)
        elif kind == "catedral" and "catedral" in lm:
            c0 = lm["catedral"]
            c = c0 if clear_of_route(c0, 50) else fit_spot(track, c0, 52, zones)
            i_n = int(np.argmin(np.hypot(*(pts - c).T)))
            f = (pts[i_n] - c) / max(np.hypot(*(pts[i_n] - c)), 1e-6)
            placed(g, crowd, lights, gy(c), lambda tg, tc, tl: catedral_hd(tg, c, f, tl))
            zones.circle(c, 50)
        elif kind == "cajadelagua" and "cajadelagua" in lm:
            c0 = lm["cajadelagua"]
            c = c0 if clear_of_route(c0, 10) else fit_spot(track, c0, 12, zones)
            placed(g, crowd, lights, gy(c), lambda tg, tc, tl: caja_agua_hd(tg, c, tl))
            zones.circle(c, 9)
        elif kind == "edificio" and key[1] in lm:                          # ("edificio", nombre, letrero, color)
            c0 = lm[key[1]]
            hl, hw_, hh = 40.0, 16.0, 22.0
            c = c0 if clear_of_route(c0, hw_ + 14) else fit_spot(track, c0, hw_ + 16, zones)
            i_n = int(np.argmin(np.hypot(*(pts - c).T)))
            placed(g, crowd, lights, gy(c), lambda tg, tc, tl: edificio_hd(tg, c, track.tan[i_n], hl, hw_, hh,
                                                                           key[3], key[2], tl))
            zones.rect(c, track.tan[i_n], hl + 2, hw_ + 2)


def shift_info(inf, dy):
    for k in ("launch", "flames", "fireballs", "fountain", "movers"):
        if k in inf:
            inf[k] = [(p[0], p[1] + dy, p[2]) for p in inf[k]]


def make_track(md):
    return OSMTrack(md) if md.osm else MapTrack(md)


BUILD_WORLD_CDMX = hd.build_world        # escenario original del Autodromo (se guarda antes de reemplazarlo)


def build_world_for(track):
    md = track.md
    if md.builder == "cdmx":
        return BUILD_WORLD_CDMX(track)
    if md.builder == "osm":
        return build_world_osm(track, md)
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
            m.length = make_track(m).L
        self.map_i = map_index(map_key) if map_key and map_index(map_key) is not None else 0
        self.map_sel = self.map_i
        self.pending_map = None
        self._gpu = []
        self.plane_vao = None
        self.spin_vao = {}
        hd.Track = lambda: make_track(MAPS[self.map_i])  # el juego base construye la pista y el mundo del mapa
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
        ch = ORIG_CHUNKS(R, arr, center, keep_radius)
        if self.md.osm:                                   # ciudad real: se dibuja hasta 1.2 km (la bruma tapa el resto)
            orig_vis = ch.visible
            ch.visible = lambda eye, fwd, cosl, maxd, _o=orig_vis: _o(eye, fwd, cosl, min(maxd, 1200.0))
        return ch

    def tod_params(self):
        p = super().tod_params()
        if self.md.osm:
            p = dict(p)
            p["fog_den"] = p["fog_den"] * 2.4
        return p

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
        """Objetos animados del mapa: aviones del aeropuerto y juegos mecanicos de la feria."""
        geo = self.info.get("plane_geo")
        self.plane_vao = self.R.vao_scene(geo.array()) if geo is not None else None
        self.spin_vao = {}
        for sp in self.info.get("spinners", []):
            if sp["kind"] == "rueda" and "rueda" not in self.spin_vao:
                self.spin_vao["rueda"] = self.R.vao_scene(rueda_geo(sp["R"]).array())
                self.spin_vao["gondola"] = self.R.vao_scene(gondola_geo().array())
            elif sp["kind"] == "carrusel" and "carrusel" not in self.spin_vao:
                self.spin_vao["carrusel"] = self.R.vao_scene(carrusel_geo().array())

    def anim_models(self):
        """[(matriz, posicion, vao, radio)] de todo lo que se mueve en el escenario."""
        out = []
        if self.plane_vao is not None:
            for fl in self.info.get("flights", []):
                p = flight_pose(fl, self.t)
                if p is not None:
                    x, y, z, ang, pitch = p
                    out.append((model_full(x, y, z, ang, 0.0, pitch), (x, y, z), self.plane_vao, 40.0))
        for sp in self.info.get("spinners", []):
            x, y, z = sp["c"]
            th = self.t * sp["w"]
            if sp["kind"] == "rueda" and "rueda" in self.spin_vao:
                yaw, R = sp["yaw"], sp["R"]
                out.append((model_full(x, y, z, yaw, 0.0, th), (x, y, z), self.spin_vao["rueda"], R + 3))
                for k in range(16):
                    a = th + k / 16 * TAU
                    lx, ly = R * math.cos(a), R * math.sin(a)
                    q = (x + math.cos(yaw) * lx, y + ly, z + math.sin(yaw) * lx)
                    out.append((model_full(*q, yaw), q, self.spin_vao["gondola"], 3.0))
            elif sp["kind"] == "carrusel" and "carrusel" in self.spin_vao:
                out.append((model_full(x, 0.0, z, th), (x, 4.0, z), self.spin_vao["carrusel"], sp["R"] + 2))
        return out

    def draw_cars(self, prog, items, shadow=False, headl=0.0):
        super().draw_cars(prog, items, shadow, headl)
        if self.state in ("select", "lobby"):
            return
        anims = self.anim_models()
        for M_, pos, vao, r in anims:
            if not shadow and hasattr(self, "L_pos"):
                self.light_bytes(pos, r, prog)
            setu(prog, u_model=gl_bytes(M_))
            vao[1 if shadow else 0].render()
        if anims:
            setu(prog, u_model=gl_bytes(np.identity(4, F4)))

    def _wrap_fx(self):
        self.fx.__class__ = MapFX
        self.fx.__dict__.pop("concert", None)
        self.fx.allow_concert = self.md.concert
        fx = self.fx
        if getattr(fx, "_lifted", False):
            return
        fx._lifted = True
        for name, where in (("sparks", lambda a: (a[0], a[1])), ("confetti", lambda a: (a[0], a[1])),
                            ("nitro_flame", lambda a: (a[0].x, a[0].z))):
            orig = getattr(fx, name)

            def wrapped(*a, _orig=orig, _where=where, **k):
                n0, f0 = len(fx.p), len(fx.flashes)
                r = _orig(*a, **k)
                tr = self.track
                if isinstance(tr, OSMTrack):
                    y = self.ground_y(*_where(a))
                    if len(fx.p) > n0:
                        fx.p[n0:, 1] += y
                    fx.flashes = fx.flashes[:f0] + [(fl[0], fl[1] + y) + tuple(fl[2:]) for fl in fx.flashes[f0:]]
                return r
            setattr(fx, name, wrapped)

    # ------------------------------------------------------------ alturas (calles reales con subidas y puentes)
    def ground_y(self, x, z):
        tr = self.track
        if not isinstance(tr, OSMTrack):
            return 0.0
        return tr.h_xz(x, z)

    def car_y(self, c):
        tr = self.track
        if not isinstance(tr, OSMTrack) or not hasattr(c, "u"):
            return 0.0, 0.0
        i = getattr(c, "idx", None)
        y = tr.h_xz(c.x, c.z, near=i if isinstance(i, (int, np.integer)) else None)
        return y, tr.slope_at(c.u)

    # ------------------------------------------------------------ direccion progresiva a alta velocidad
    STEER_V0 = 25.0                     # hasta ~90 km/h la direccion queda igual que siempre

    @staticmethod
    def _steer_sat(v, grip):
        """Fraccion de direccion que ya usa todo el agarre a la velocidad v (modelo del juego base)."""
        v = max(v, 1.0)
        return math.atan(grip * base.WB / (v * v)) * (1 + v / 16) / 0.6

    def player_inputs(self):
        thr, brk, st, hb, nitro = super().player_inputs()
        p = self.player
        v = abs(p.vf)
        dt = getattr(self, "_frame_dt", 1 / 60)
        # 1) la tecla no salta a tope: rampa mas lenta a mayor velocidad (los toques cortos corrigen poquito)
        hi = hd.clamp((v - 20.0) / 50.0, 0.0, 1.0)
        cur = getattr(self, "_st_in", 0.0)
        if st != 0 and st * cur >= 0 and abs(st) > abs(cur):
            rate = 1.0 / (0.06 + 0.24 * hi)                 # 0.06 s a baja velocidad, 0.30 s a tope
        else:
            rate = 9.0                                      # soltar o cambiar de lado: rapido
        cur += hd.clamp(st - cur, -rate * dt, rate * dt)
        self._st_in = cur
        cur = st + (cur - st) * hi                          # a baja velocidad: exactamente como siempre
        # 2) la misma fraccion de volante = la misma fraccion del agarre, a cualquier velocidad
        if v > self.STEER_V0 and not hb:
            k = min(1.0, self._steer_sat(v, p.grip) / self._steer_sat(self.STEER_V0, p.grip))
            cur *= k
        return thr, brk, cur, hb, nitro

    def car_pitch(self, c):
        """Inclinacion de la carroceria en la pendiente, filtrada (sin temblor)."""
        tr = self.track
        _, sl = self.car_y(c)
        i = int((c.u % tr.L) / tr.spacing) % tr.n
        along = math.cos(c.ang) * tr.tan[i, 0] + math.sin(c.ang) * tr.tan[i, 1]
        tgt = math.atan(sl) * along
        dt = getattr(self, "_frame_dt", 1 / 60)
        p = getattr(c, "_slope_p", tgt)
        p += (tgt - p) * min(1.0, dt * 10.0)
        c._slope_p = p
        return p

    def car_models(self, cars):
        if not isinstance(self.track, OSMTrack) or self.state in ("select", "lobby"):
            return super().car_models(cars)
        out = []
        tr = self.track
        for c in cars:
            vis = c.vis
            y, _ = self.car_y(c)
            M = model_full(c.x, y, c.z, c.ang, c.roll, c.pitch + self.car_pitch(c))
            r = vis["info"]["radius"]
            wheels = []
            for wx, wz in vis["info"]["wheels"]:
                for side in (-1, 1):
                    W_ = M @ hd.mat_T(wx, r, side * wz)
                    if wx > 0:
                        W_ = W_ @ hd.mat_Ry(-c.steer * 0.45)
                    if side < 0:
                        W_ = W_ @ hd.MIRROR_Z
                    wheels.append(W_ @ hd.mat_Rz(-c.wheel_rot))
            out.append((c, vis, M, wheels))
        return out

    def camera(self, dt):
        self._frame_dt = dt if dt > 0 else 1 / 60
        eye, target, fov = super().camera(dt)
        tr = self.track
        if isinstance(tr, OSMTrack) and self.state not in ("select", "lobby"):
            if self.state in ("countdown", "race", "finished") and self.cam_mode in (0, 1, 2, 3):
                ref = self.player                                  # camara pegada a la altura del coche
                y_t, _ = self.car_y(ref)
                y_e = tr.h_xz(eye[0], eye[2], near=ref.idx)       # el ojo, a la altura de la calle donde esta
                y_e = max(y_e, y_t - 2.0) if self.cam_mode in (0, 1) else y_t
            else:
                y_t = tr.h_xz(target[0], target[2])
                y_e = tr.h_xz(eye[0], eye[2])
            k = min(1.0, dt * 12.0) if dt > 0 else 1.0               # suavizado: sin temblor en las rampas
            cy = getattr(self, "_cam_y", None)
            if cy is None or abs(cy[0] - y_e) > 6 or abs(cy[1] - y_t) > 6:
                cy = [y_e, y_t]
            cy = [cy[0] + (y_e - cy[0]) * k, cy[1] + (y_t - cy[1]) * k]
            self._cam_y = cy
            eye = (eye[0], eye[1] + cy[0], eye[2])
            target = (target[0], target[1] + cy[1], target[2])
        return eye, target, fov

    def gather_lights(self, eye, tod):
        r = super().gather_lights(eye, tod)
        if isinstance(self.track, OSMTrack) and len(self.L_pos) and self.state not in ("select", "lobby"):
            P = self.L_pos
            for c in self.cars:                                   # faros y nitro a la altura del coche
                y, _ = self.car_y(c)
                m = (P[:, 1] < 1.6) & ((P[:, 0] - c.x) ** 2 + (P[:, 2] - c.z) ** 2 < 150)
                P[m, 1] += y
            self._lkey = None
            return self.light_bytes(None, 0)
        return r

    def collide_cars(self):
        if not isinstance(self.track, OSMTrack):
            return super().collide_cars()
        # coches en distinto nivel (uno en el puente, otro abajo) no chocan
        cs = self.cars
        ys = [self.car_y(c)[0] for c in cs]
        saved = []
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if abs(ys[i] - ys[j]) > 2.5 and math.hypot(cs[i].x - cs[j].x, cs[i].z - cs[j].z) < 8:
                    saved.append((cs[j], cs[j].x))
                    cs[j].x += 1000.0
        super().collide_cars()
        for c, _ in saved:
            c.x -= 1000.0

    def hud_race(self):
        super().hud_race()
        tr = self.track
        if not isinstance(tr, OSMTrack):
            return
        p = self.player
        nm = tr.names[tr.name_idx[p.idx % tr.n]] if tr.names else ""
        if nm != getattr(self, "_street", None):
            self._street, self._street_t = nm, 0.0
        self._street_t = getattr(self, "_street_t", 0.0) + 1 / 60
        if nm and self._street_t < 5.0:
            a = int(255 * hd.clamp(min(self._street_t / 0.4, (5.0 - self._street_t) / 0.8), 0, 1))
            w = font(24).size(nm.upper())[0] + 60
            self.panel((WIN_W // 2 - w // 2, 64, w, 38), int(a * 0.7), accent=(40, 200, 120))
            self.txt(nm.upper(), 24, (255, 255, 255), (WIN_W // 2 + 6, 83), "center", alpha=a)

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
                tr = make_track(md)
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
        tr = self.track
        if isinstance(tr, OSMTrack):                       # asfalto o banqueta segun la distancia real
            for c in self.cars:
                if not getattr(c, "remote", False) and hasattr(c, "true_lat"):
                    c.surface = 0 if abs(c.true_lat) < tr.rw[c.idx % tr.n] + 0.4 else 1
                    # bot atorado contra una pared: se echa en reversa hacia el centro de la calle
                    if self.state == "race" and not c.is_player and not c.finished:
                        c._slow_t = getattr(c, "_slow_t", 0.0) + rdt if abs(c.vf) < 3.0 else 0.0
                        if c._slow_t > 2.5 and c.reverse_t <= 0:
                            c.reverse_t, c._slow_t = 1.1, 0.0
                            c.rec_steer = float(getattr(c, "steer", 0.0))
        self.map_extras(rdt)

    def new_race(self, *a, **k):
        super().new_race(*a, **k)
        self._fix_grid()

    def new_race_mp(self, order):
        super().new_race_mp(order)
        self._fix_grid()

    def _fix_grid(self):
        tr = self.track
        if not isinstance(tr, OSMTrack):
            return
        for c in self.cars:
            sg = tr.L - 10 - (c.slot // 2) * 12
            c.idx = int(sg / tr.spacing) % tr.n
            c.s = c.s_new = sg
            c.u = sg - tr.L
            c.lap_idx = math.floor(c.u / tr.L)
            c.true_lat = 0.0

    def map_extras(self, rdt):
        """Fuegos artificiales por toda la ruta (noche) y rampas de turbo."""
        md = self.md
        if md.fireworks and self.tod == 2 and self.fx.concert and \
                self.state in ("menu", "intro", "countdown", "race", "finished"):
            self._fw = getattr(self, "_fw", 0.0) - rdt
            if self._fw <= 0:
                self._fw = random.uniform(0.1, 0.3)
                ref = self.player if self.state != "menu" else self.cars[getattr(self, "tv_target", 0) % len(self.cars)]
                su = ref.s + random.uniform(-80, 500)
                x, z, dx, dz = self.track.point(su)
                side, off = random.choice((-1, 1)), LIMIT + random.uniform(12, 120)
                y0 = self.track.h_at(su) if hasattr(self.track, "h_at") else 0.0
                self.fx.spawn((x - dz * side * off, 1.0 + y0, z + dx * side * off),
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
        if m.osm:
            self.txt("Calles reales · datos © colaboradores de OpenStreetMap", 12, (150, 165, 180), (x + 42, y + 94),
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

"""
GTC RACING - descarga de mapas reales (OpenStreetMap)

Herramienta para preparar los circuitos callejeros con calles, edificios, puentes y alturas reales.
No hace falta para jugar: los datos ya quedan guardados en la carpeta mapas_osm/.
Uso:   python gtc_mapas_osm.py fraynano      (o chapultepec, slp, todos)

Fuentes (requieren internet solo al correr esta herramienta):
  - ruta por las calles: OSRM (router.project-osrm.org y routing.openstreetmap.de para tramos peatonales)
  - calles, edificios, parques, agua, comercios: Overpass API (overpass-api.de)
  - alturas del terreno: OpenTopoData (SRTM 30 m)
Datos (c) colaboradores de OpenStreetMap, licencia ODbL (www.openstreetmap.org/copyright).
"""
import gzip
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "mapas_osm")
UA = {"User-Agent": "gtc-racing-map-builder/1.0 (juego de carreras local)"}
ATTRIBUTION = "Datos (c) colaboradores de OpenStreetMap (ODbL)"

# ------------------------------------------------------------------ definicion de cada circuito
# ref: puntos de la imagen del usuario (px) con su coordenada real (lat, lon) para calibrar la imagen.
# route: puntos dibujados (px) en el orden de la carrera; la meta es el primero.
# legs: tramos que se calculan como peatonales (plazas, parques): {indice_del_punto: "foot"}
# fixed: puntos que se fuerzan a una coordenada real exacta {indice: (lat, lon)}
CIRCUITS = {
    "fraynano": dict(
        ref=[((78, 100), (19.4220125, -99.1247034)), ((113, 408), (19.4086025, -99.1227039)),
             ((406, 352), (19.4107401, -99.1087764)), ((912, 507), (19.4036757, -99.0852437))],
        route=[((414, 380), "Calle Fernando Iglesias Calderón"), ((500, 272), "Calle Fernando Iglesias Calderón"),
               ((540, 226), "Calle Fernando Iglesias Calderón"), ((575, 221), "Avenida Fray Servando Teresa de Mier"),
               ((622, 224), "Avenida Fray Servando Teresa de Mier"),
               ((672, 145), "Circuito Interior Boulevard Puerto Aéreo"),
               ((700, 80), "Circuito Interior Boulevard Puerto Aéreo"),
               ((830, 62), "Calle Asistencia Pública"),
               ((885, 104), "Calle Asistencia Pública"), ((930, 182), "Avenida Economía"),
               ((900, 220), "Avenida Economía"), ((860, 300), "Calle 47"), ((830, 362), "Calle 47"),
               ((805, 403), "Calle 47"), ((770, 410), "Viaducto Río de la Piedad"),
               ((700, 405), "Viaducto Río de la Piedad"), ((550, 418), "Viaducto Río de la Piedad"),
               ((470, 420), "Viaducto Río de la Piedad"), ((396, 412), "Avenida Morelos")],
        legs={}, fixed={}),
    "chapultepec": dict(
        ref=[((93, 663), (19.4261524, -99.1866516)), ((421, 543), (19.4361745, -99.1546214)),
             ((800, 15), (19.4848583, -99.1178571)), ((445, 410), (19.4480388, -99.1519785)),
             ((628, 517), (19.4393592, -99.1346955)), ((530, 226), (19.4658688, -99.1445808)),
             ((618, 52), (19.4822357, -99.1365275))],
        route=[((45, 687), "Avenida Paseo de la Reforma"), ((110, 694), "Avenida Paseo de la Reforma"),
               ((185, 702), "Avenida Paseo de la Reforma"), ((245, 689), "Avenida Paseo de la Reforma"),
               ((305, 667), "Avenida Paseo de la Reforma"), ((360, 648), "Avenida Paseo de la Reforma"),
               ((398, 610), "Calle Ignacio Ramírez"), ((421, 543), None), ((480, 546), "Avenida de la República"),
               ((555, 505), "Avenida Paseo de la Reforma Norte"), ((610, 452), "Avenida Paseo de la Reforma Norte"),
               ((660, 400), "Calzada de Guadalupe"), ((700, 310), "Calzada de Guadalupe"),
               ((725, 230), "Calzada de Guadalupe"), ((768, 135), "Calzada de Guadalupe"),
               ((795, 70), "Calzada de Guadalupe"), ((775, 25), None), ((735, 22), "Avenida Insurgentes Norte"),
               ((690, 85), "Avenida Insurgentes Norte"), ((630, 170), "Avenida Insurgentes Norte"),
               ((570, 222), "Avenida Insurgentes Norte"), ((520, 280), "Avenida Insurgentes Norte"),
               ((470, 340), "Avenida Insurgentes Norte"), ((440, 420), "Avenida Insurgentes Norte"),
               ((410, 510), "Avenida Insurgentes Centro"), ((372, 590), "Avenida Insurgentes Centro"),
               ((350, 655), "Avenida Insurgentes Sur"), ((336, 708), "Glorieta de los Insurgentes"),
               ((290, 719), "Avenida Chapultepec"), ((220, 730), "Avenida Chapultepec"),
               ((150, 738), "Avenida Chapultepec"), ((0, 0), None), ((0, 0), None), ((0, 0), None),
               ((0, 0), None)],
        # Monumento a la Revolucion (bajo los arcos) y el bosque alrededor del Castillo
        fixed={7: (19.4361745, -99.1546214), 31: (19.4204, -99.1766), 32: (19.4181, -99.1812),
               33: (19.4188, -99.1858), 34: (19.4236, -99.1866)},
        legs={6: "foot", 7: "foot", 30: "foot", 31: "foot", 32: "foot", 33: "foot", 34: "foot"},
        through={7: "Plaza de la República"}),
    "slp": dict(
        ref=[((528, 590), (22.1383403, -100.9706035)), ((851, 367), (22.1711248, -100.9190721)),
             ((620, 801), (22.109398, -100.9556467)), ((244, 505), (22.1510405, -101.0170912))],
        route=[((746, 258), None), ((0, 0), "Calle Zaragoza"), ((0, 0), "Prolongación Zaragoza"),
               ((624, 335), "Autopista Matehuala-San Luis Potosí"), ((614, 400), "Autopista Matehuala-San Luis Potosí"),
               ((615, 500), "Autopista Matehuala-San Luis Potosí"), ((617, 600), "Avenida Doctor Salvador Nava Martínez"),
               ((588, 642), "Avenida Doctor Salvador Nava Martínez"), ((545, 660), "Avenida Doctor Salvador Nava Martínez"),
               ((478, 667), "Avenida Doctor Salvador Nava Martínez"), ((434, 666), "Calle Coronel Romero"),
               ((429, 705), "Calle Coronel Romero"), ((425, 756), "Boulevard Periférico Antonio Rocha Cordero"),
               ((360, 761), "Boulevard Periférico Antonio Rocha Cordero"), ((307, 730), None), ((340, 690), None),
               ((372, 636), None), ((400, 628), "Avenida Doctor Salvador Nava Martínez"),
               ((425, 598), "Avenida General Mariano Jiménez"), ((450, 550), "Avenida General Mariano Jiménez"),
               ((470, 500), "Avenida Damián Carmona"), ((469, 422), "Avenida Damián Carmona"),
               ((472, 410), "Boulevard Río Santiago"), ((530, 390), "Avenida Acceso Norte"),
               ((575, 388), "Avenida Acceso Norte"), ((680, 320), "Avenida Soledad"), ((0, 0), None),
               ((716, 255), "Avenida Miguel Hidalgo")],
        # Jardin Hidalgo de Soledad (meta), calle Zaragoza y Plaza Citadina
        fixed={0: (22.1879, -100.9366), 1: (22.1908, -100.9373), 2: (22.1990, -100.9386), 26: (22.1824, -100.9501)},
        legs={14: "foot", 15: "foot", 16: "foot"}, soft_start=True),
    "segundopiso": dict(
        ll=True, strict=True, margin=0.02, ref=[((0, 0), (0.0, 0.0)), ((1, 0), (0.0, 1.0)), ((0, 1), (1.0, 0.0))],
        # Periferico Sur: se sube al Segundo Piso en San Antonio, se baja antes de Las Aguilas y se regresa
        # por abajo (pasando debajo del puente por el que se fue)
        route=[(19.3828, -99.1914, "Anillo Periférico"), (19.3760, -99.1921, "Anillo Periférico"),
               (19.3740, -99.1921, "Anillo Periférico"), (19.3700, -99.1914, "Anillo Periférico"),
               (19.3650, -99.1903, "Anillo Periférico"), (19.3605, -99.1920, "Anillo Periférico"),
               (19.3570, -99.1975, "Anillo Periférico"),
               (19.3580, -99.1960, "@arriba"), (19.3620, -99.1912, "@arriba"), (19.3680, -99.1906, "@arriba"),
               (19.3730, -99.1918, "@arriba")],
        free_last=True, free_legs=(6,),
        legs={}, fixed={}),
    "tangamanga": dict(
        # glorietas del parque (imagen 13 del usuario) con su coordenada real (OpenStreetMap)
        ref=[((268, 108), (22.18638, -100.99249)), ((233, 375), (22.18152, -100.99318)),
             ((905, 222), (22.18425, -100.97987)), ((873, 607), (22.17725, -100.98054))],
        trace=True,                 # la ruta es el trazo tal cual (autodromo + caminos del parque)
        route=[(352, 321), (370, 320), (400, 321), (430, 322), (452, 323), (448, 305), (440, 290), (428, 272),
               (415, 258), (400, 245), (385, 232), (368, 218), (352, 205), (337, 194), (325, 180), (315, 163),
               (308, 150), (302, 138), (295, 132), (285, 130), (270, 129), (255, 126), (248, 118), (248, 103),
               (244, 94), (234, 91), (224, 96), (217, 105), (208, 110), (201, 105), (200, 95), (208, 87),
               (222, 79), (238, 72), (255, 66), (275, 65), (300, 65), (320, 65), (335, 68), (345, 76), (355, 90),
               (368, 108), (380, 122), (392, 140), (405, 160), (420, 177), (440, 197), (455, 215), (468, 235),
               (480, 255), (492, 275), (503, 293), (512, 308), (522, 319), (540, 319), (560, 318), (600, 300),
               (640, 285), (680, 272), (720, 262), (760, 255), (800, 245), (840, 238), (868, 233), (880, 226),
               (888, 215), (898, 210), (910, 210), (920, 218), (923, 230), (915, 240), (908, 252), (907, 270),
               (905, 300), (900, 350), (893, 400), (887, 450), (880, 500), (877, 550), (877, 584), (887, 591),
               (893, 603), (892, 618), (880, 625), (862, 624), (856, 612), (855, 592), (840, 588), (800, 585),
               (760, 583), (720, 580), (690, 577), (670, 560), (645, 530), (620, 500), (595, 470), (575, 445),
               (560, 428), (545, 410), (530, 390), (520, 375), (505, 368), (490, 368), (475, 367), (465, 358),
               (460, 345), (456, 335), (445, 332), (420, 330), (390, 330), (360, 329), (330, 329), (300, 326),
               (270, 323), (240, 321), (200, 316), (170, 312), (145, 305), (125, 295), (117, 283), (125, 272),
               (140, 266), (155, 265), (170, 271), (185, 282), (200, 296), (220, 308), (240, 313), (262, 313),
               (280, 306), (293, 290), (296, 272), (298, 258), (308, 251), (320, 250), (337, 257), (355, 265),
               (375, 270), (392, 272), (400, 276), (403, 286), (401, 296), (380, 297), (350, 297), (332, 298),
               (324, 306), (322, 316), (334, 320)],
        # contorno del parque (zona verde de la imagen 12) para el bosque
        ref_park=[((260, 98), (22.18638, -100.99249)), ((222, 365), (22.18152, -100.99318)),
                  ((896, 214), (22.18425, -100.97987)), ((866, 600), (22.17725, -100.98054))],
        park=[(62, 278), (120, 180), (195, 68), (330, 42), (470, 210), (590, 285), (630, 195), (950, 160),
              (1000, 330), (1052, 530), (880, 600), (700, 700), (605, 772), (500, 660), (400, 610), (290, 560),
              (180, 470)],
        legs={}, fixed={}),
}


# ------------------------------------------------------------------ utilidades
def http_json(url, data=None, tries=4, timeout=180):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:                       # servidores publicos: reintentar con calma
            print(f"   (reintento {k + 1}: {e})", flush=True)
            time.sleep(5 + 10 * k)
    raise RuntimeError("no se pudo descargar: " + url[:120])


def fit_affine(ref):
    """px -> (lat, lon) por minimos cuadrados."""
    A = np.array([[x, y, 1.0] for (x, y), _ in ref])
    lat = np.array([ll[0] for _, ll in ref])
    lon = np.array([ll[1] for _, ll in ref])
    ca, *_ = np.linalg.lstsq(A, lat, rcond=None)
    co, *_ = np.linalg.lstsq(A, lon, rcond=None)
    return ca, co


def px_to_ll(p, cal):
    ca, co = cal
    v = np.array([p[0], p[1], 1.0])
    return float(v @ ca), float(v @ co)


OVERPASS = "https://overpass-api.de/api/interpreter"
CACHE_DIR = os.path.join(OUT_DIR, "_cache")
DRIVE = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
         "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service",
         "road")
WALK = ("pedestrian", "footway", "path", "cycleway", "track", "steps")
# costo por tipo de calle (las avenidas grandes 'pesan' menos: la ruta las prefiere como en el dibujo)
COST = {"motorway": 0.8, "trunk": 0.8, "primary": 0.85, "secondary": 0.9, "tertiary": 1.0, "unclassified": 1.2,
        "residential": 1.25, "living_street": 1.6, "service": 2.2, "road": 1.5, "pedestrian": 1.3, "footway": 1.5,
        "path": 1.6, "cycleway": 1.6, "track": 2.0, "steps": 50.0}


def overpass(query, cache_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_name + ".json.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    print(f"   descargando {cache_name} de Overpass...", flush=True)
    data = urllib.parse.urlencode({"data": query}).encode()
    r = http_json(OVERPASS, data=data, timeout=400)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(r, f)
    time.sleep(2)
    return r


class Graph:
    """Calles de OSM como grafo (sin respetar sentidos: es una carrera)."""

    def __init__(self, osm):
        self.node = {e["id"]: (e["lat"], e["lon"]) for e in osm["elements"] if e["type"] == "node"}
        self.ways = [e for e in osm["elements"] if e["type"] == "way" and "nodes" in e]
        self.adj = {}
        self.wrongway = set()
        self.strict = False                              # True = respetar siempre el sentido de las vias
        for wi, w in enumerate(self.ways):
            t = w.get("tags", {})
            hw = t.get("highway", "road").replace("_link", "")
            k = COST.get(hw, 2.0)
            if hw in WALK:
                k *= 1.0
            ow = t.get("oneway") in ("yes", "1", "-1")
            ns = w["nodes"]
            for a, b in zip(ns[:-1], ns[1:]):
                if a not in self.node or b not in self.node:
                    continue
                d = self.dist(a, b)
                fwd = (a, b) if t.get("oneway") != "-1" else (b, a)
                for u, v in ((a, b), (b, a)):
                    pen = 1.35 if (ow and (u, v) != fwd) else 1.0    # contra el sentido: se permite, pero cuesta
                    self.adj.setdefault(u, []).append((v, d * k * pen, wi))
                    if pen > 1.0:
                        self.wrongway.add((u, v))

    def dist(self, a, b):
        (la, oa), (lb, ob) = self.node[a], self.node[b]
        return math.hypot((ob - oa) * 111320 * math.cos(math.radians(la)), (lb - la) * 110574)

    def nearest(self, ll, allow, min_class=None, radius=70.0):
        """Nodo mas cercano a ll en calles permitidas (preferencia por avenidas)."""
        lat, lon = ll
        best, bd = None, 1e18
        for wi, w in enumerate(self.ways):
            hw = w.get("tags", {}).get("highway", "")
            if hw not in allow:
                continue
            for n in w["nodes"]:
                if n not in self.node:
                    continue
                la, lo = self.node[n]
                d = math.hypot((lo - lon) * 111320 * math.cos(math.radians(lat)), (la - lat) * 110574)
                d *= COST.get(hw.replace("_link", ""), 2.0)          # mejor una avenida un poco mas lejos
                if d < bd:
                    best, bd = n, d
        return best

    def path(self, a, b, allow, prev=None, only=None):
        """A* de a a b usando solo calles 'allow': [(nodo, indice_de_calle)].
        Cuenta las vueltas: una vuelta cerrada cuesta como varios cientos de metros (una vuelta en U, casi prohibida)."""
        import heapq
        goal = self.node[b]
        kx = 111320 * math.cos(math.radians(goal[0]))
        h = lambda n: math.hypot((self.node[n][1] - goal[1]) * kx, (self.node[n][0] - goal[0]) * 110574) * 0.8

        def vec(u, v):
            (la, lo), (lb, ob) = self.node[u], self.node[v]
            return ((ob - lo) * kx, (lb - la) * 110574)
        start = (a, prev)
        openq = [(h(a), 0.0, start)]
        came, gs = {start: (None, None)}, {start: 0.0}
        end = None
        while openq:
            _, g, st = heapq.heappop(openq)
            u, p = st
            if u == b:
                end = st
                break
            if g > gs.get(st, 1e18):
                continue
            d0 = vec(p, u) if p is not None else None
            for v, c, wi in self.adj.get(u, ()):
                if self.strict and (u, v) in self.wrongway:
                    continue
                if v == p or self.ways[wi].get("tags", {}).get("highway", "") not in allow:
                    continue
                if only is not None:
                    tg_ = self.ways[wi].get("tags", {})
                    if callable(only):
                        if not only(tg_):
                            continue
                    elif norm_name(tg_.get("name", "")) != only:
                        continue
                ng = g + c
                if d0 is not None:
                    d1 = vec(u, v)
                    l0, l1 = math.hypot(*d0), math.hypot(*d1)
                    if l0 > 0.5 and l1 > 0.5:
                        cosang = (d0[0] * d1[0] + d0[1] * d1[1]) / (l0 * l1)
                        if cosang < -0.94:
                            ng += 3000.0                         # vuelta en U
                        elif cosang < -0.6:
                            ng += 200.0                          # esquina muy aguda
                        elif cosang < 0.25:
                            ng += 30.0                           # vuelta normal (evita zigzag por la colonia)
                ns = (v, u)
                if ng < gs.get(ns, 1e18):
                    gs[ns] = ng
                    came[ns] = (st, wi)
                    heapq.heappush(openq, (ng + h(v), ng, ns))
        if end is None:
            raise RuntimeError("no hay camino entre dos puntos de la ruta")
        out, st = [], end
        while st is not None:
            pst, wi = came[st]
            out.append((st[0], wi))
            st = pst
        return out[::-1]


def layer_of(t):
    try:
        return int(str(t.get("layer", "0")).split(";")[0])
    except ValueError:
        return 0


def norm_name(s):
    import unicodedata
    s = unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def remove_loops(seq, G, max_len=1600.0):
    """Quita idas y vueltas cortas (un nodo que se repite a menos de max_len metros). Los cruces largos
    de la ruta consigo misma (p. ej. Reforma con Insurgentes) se respetan."""
    out, pos, cum = [], {}, [0.0]
    for n, wi in seq:
        if n in pos and cum[-1] - cum[pos[n]] < max_len:
            k = pos[n]
            for m, _ in out[k + 1:]:
                if pos.get(m, -1) > k:
                    pos.pop(m, None)
            out, cum = out[:k + 1], cum[:k + 1]
            out[-1] = (n, wi)
        else:
            if out:
                cum.append(cum[-1] + G.dist(out[-1][0], n))
            pos[n] = len(out)
            out.append((n, wi))
    return out


def build_route_graph(key, c):
    """Ruta por las calles reales de OSM pasando por los puntos dibujados."""
    cal = fit_affine(c["ref"])
    if c.get("ll"):                                   # ruta dada en coordenadas reales (lat, lon, calle)
        wps = [(r[0], r[1]) for r in c["route"]]
        wnames = [r[2] if len(r) > 2 else None for r in c["route"]]
    else:
        wps = [px_to_ll(r[0] if isinstance(r[0], tuple) else r, cal) for r in c["route"]]
        wnames = [r[1] if isinstance(r[0], tuple) and len(r) > 1 else None for r in c["route"]]
    for i, ll in c.get("fixed", {}).items():
        wps[i] = ll
    la = np.array([w[0] for w in wps])
    lo = np.array([w[1] for w in wps])
    m = c.get("margin", 0.008)
    bbox = (la.min() - m, lo.min() - m, la.max() + m, lo.max() + m)
    hw = "|".join(DRIVE + WALK + ("raceway",))
    q = (f'[out:json][timeout:300];way["highway"~"^({hw})$"]({bbox[0]:.5f},{bbox[1]:.5f},{bbox[2]:.5f},'
         f'{bbox[3]:.5f});out body;>;out skel qt;')
    G = Graph(overpass(q, key + "_calles" + ("" if m == 0.008 else f"_{m}")))
    G.strict = bool(c.get("strict"))
    # ---- afinar la calibracion de la imagen: cada punto con nombre se 'pega' a su calle real
    pxs = [r[0] if isinstance(r[0], tuple) else r for r in c["route"]]
    segs_by_name = {}
    for w in G.ways:
        nm = norm_name(w.get("tags", {}).get("name", ""))
        ns = [n for n in w["nodes"] if n in G.node]
        for a_, b_ in zip(ns[:-1], ns[1:]):
            segs_by_name.setdefault(nm, []).append((G.node[a_], G.node[b_]))
    kx0 = 111320 * math.cos(math.radians(float(la.mean())))

    def closest_on(nm, ll):
        S = np.array(segs_by_name[nm])                              # (k, 2, 2) lat, lon
        A_ = np.column_stack([(S[:, 0, 1] - ll[1]) * kx0, (S[:, 0, 0] - ll[0]) * 110574])
        D_ = np.column_stack([(S[:, 1, 1] - S[:, 0, 1]) * kx0, (S[:, 1, 0] - S[:, 0, 0]) * 110574])
        t_ = np.clip(-(A_ * D_).sum(1) / np.maximum((D_ * D_).sum(1), 1e-9), 0, 1)
        P_ = A_ + D_ * t_[:, None]
        k_ = int(np.argmin(np.hypot(P_[:, 0], P_[:, 1])))
        return (ll[0] + P_[k_, 1] / 110574, ll[1] + P_[k_, 0] / kx0), float(np.hypot(*P_[k_]))
    for it in range(0 if c.get("ll") else 8):
        pairs, err = [], []
        for k_, (px, nm) in enumerate(zip(pxs, wnames)):
            if nm and norm_name(nm) in segs_by_name and k_ not in c.get("fixed", {}):
                ll, e = closest_on(norm_name(nm), px_to_ll(px, cal))
                pairs.append((px, ll))
                err.append(e)
        if len(pairs) < 3:
            break
        cal = fit_affine(pairs + list(c["ref"]) * 0)
        wps = [px_to_ll(p, cal) for p in pxs]
    if wnames and any(wnames) and not c.get("ll"):
        print(f"   calibracion: error medio {np.mean(err):.0f} m, maximo {np.max(err):.0f} m")
    for i, ll in c.get("fixed", {}).items():
        wps[i] = ll
    # nodos de cada tipo de calle para buscar el mas cercano rapido
    ids, xy, cls, nms, dirs, ows, lys = [], [], [], [], [], [], []
    kx = 111320 * math.cos(math.radians(float(la.mean())))
    for w in G.ways:
        t_ = w.get("tags", {})
        h_ = t_.get("highway", "")
        nm = norm_name(t_.get("name", ""))
        ow = t_.get("oneway") in ("yes", "1", "-1") or h_ == "motorway"
        sg = -1.0 if t_.get("oneway") == "-1" else 1.0
        ns = [n for n in w["nodes"] if n in G.node]
        for k_, n in enumerate(ns):
            a_, b_ = ns[max(0, k_ - 1)], ns[min(len(ns) - 1, k_ + 1)]
            dv = np.array([(G.node[b_][1] - G.node[a_][1]) * kx, (G.node[b_][0] - G.node[a_][0]) * 110574]) * sg
            dv /= max(np.hypot(*dv), 1e-6)
            ids.append(n)
            xy.append(G.node[n])
            cls.append(h_)
            nms.append(nm)
            dirs.append(dv)
            ows.append(ow)
            lys.append(layer_of(t_))
    ids, xy, cls, nms = np.array(ids), np.array(xy), np.array(cls), np.array(nms)
    dirs, ows, lys = np.array(dirs), np.array(ows), np.array(lys)

    def snap(ll, allow, name=None, travel=None):
        ok0 = np.isin(cls, allow)
        ok = ok0.copy()
        if travel is not None:                          # calles de un sentido: solo la calzada que va hacia alla
            ok &= ~(ows & ((dirs @ travel) < 0.2))
        if name == "@arriba":                          # cualquier calzada elevada (segundo piso)
            okn = ok & (lys >= 2)
            if okn.any():
                ok = okn
            name = None
        if name:
            okn = ok & (nms == norm_name(name))
            if not okn.any():
                okn = ok0 & (nms == norm_name(name))    # la calle solo existe en el otro sentido
            if okn.any():
                ok = okn
            else:
                print("   (aviso) no encontre la calle:", name)
        d = np.hypot((xy[ok, 1] - ll[1]) * kx, (xy[ok, 0] - ll[0]) * 110574)
        d *= np.array([COST.get(h_.replace("_link", ""), 2.0) for h_ in cls[ok]])
        return int(ids[ok][int(np.argmin(d))])
    seq = []
    c["_start_ll"] = wps[0]
    if c.get("soft_start"):                               # no forzar la vuelta en el punto de salida
        wps, wnames = wps[1:], wnames[1:]
        c["legs"] = {k - 1: v for k, v in c.get("legs", {}).items() if k > 0}
        c["through"] = {k - 1: v for k, v in c.get("through", {}).items() if k > 0}
    wps.append(wps[0])
    wnames.append(wnames[0])
    for i in range(len(wps) - 1):
        allow = DRIVE + (WALK if c.get("legs", {}).get(i) == "foot" else ())
        G.strict = bool(c.get("strict")) and i not in c.get("free_legs", ()) and             not (c.get("free_last") and i == len(wps) - 2)
        def travel(k_):
            a_, b_ = wps[(k_ - 1) % (len(wps) - 1)], wps[(k_ + 1) % (len(wps) - 1)]
            if k_ % (len(wps) - 1) == 0:                  # la salida: hacia el siguiente punto
                a_ = wps[0]
            v = np.array([(b_[1] - a_[1]) * kx, (b_[0] - a_[0]) * 110574])
            return v / max(np.hypot(*v), 1e-6)
        a = snap(wps[i], allow, wnames[i], travel(i))
        b = snap(wps[i + 1], allow, wnames[i + 1], travel(i + 1))
        through = c.get("through", {})
        if i in through and seq:                          # se sale del paso obligado por el nodo de salida
            a = seq[-1][0]
        if (i + 1) in through:                            # paso obligado en linea recta (p. ej. bajo un monumento)
            ctr = wps[i + 1]

            def toward(q, dist=70.0):
                v = np.array([(q[1] - ctr[1]) * kx, (q[0] - ctr[0]) * 110574])
                v = v / max(np.hypot(*v), 1e-6) * dist
                return (ctr[0] + v[1] / 110574, ctr[1] + v[0] / kx)
            ent = snap(toward(wps[i]), DRIVE + WALK)
            ext = snap(toward(wps[(i + 2) % (len(wps) - 1)]), DRIVE + WALK)
            G.node[-7 - i] = ctr
            G.ways.append({"tags": {"highway": "pedestrian", "name": through[i + 1]}, "nodes": []})
            fw = len(G.ways) - 1
            pth = G.path(a, ent, allow, prev=seq[-2][0] if len(seq) >= 2 else None) if a != ent else [(a, None)]
            seq += pth if not seq else pth[1:]
            seq += [(-7 - i, fw), (ext, fw)]
            continue
        if a == b:
            continue
        prv = seq[-2][0] if len(seq) >= 2 else None
        p = None
        if wnames[i] == "@arriba" and wnames[i + 1] == "@arriba":
            try:                                            # el tramo se queda arriba
                p = G.path(a, b, allow, prev=prv, only=lambda t: layer_of(t) >= 2)
            except RuntimeError:
                p = None
        elif wnames[i] and wnames[i + 1] and norm_name(wnames[i]) == norm_name(wnames[i + 1]):
            try:                                            # mismo nombre: el tramo va solo por esa calle
                p = G.path(a, b, allow, prev=prv, only=norm_name(wnames[i]))
            except RuntimeError:
                p = None
        if p is None:
            try:
                p = G.path(a, b, allow, prev=prv)
            except RuntimeError:                            # punto en calle cerrada: se permite regresar
                p = G.path(a, b, allow, prev=None)
        seq += p if not seq else p[1:]
    closing = seq[-1][0] == seq[0][0]
    body = remove_loops(seq[:-1] if closing else seq, G)
    return G, body, cal, wps


def osrm_route(points, profile="car"):
    """Ruta por las calles reales a traves de 'points' [(lat, lon)]: [(lat, lon), ...] y nombres de calles."""
    base = ("https://router.project-osrm.org/route/v1/driving/" if profile == "car" else
            "https://routing.openstreetmap.de/routed-foot/route/v1/driving/")
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    url = base + coords + "?overview=full&geometries=geojson&steps=true&continue_straight=true"
    r = http_json(url)
    if r.get("code") != "Ok":
        raise RuntimeError(f"OSRM: {r.get('code')} {r.get('message')}")
    geo = [(lat, lon) for lon, lat in r["routes"][0]["geometry"]["coordinates"]]
    names = []
    for leg in r["routes"][0]["legs"]:
        for st in leg["steps"]:
            if st.get("name") and (not names or names[-1] != st["name"]):
                names.append(st["name"])
    time.sleep(1.0)
    return geo, names, r["routes"][0]["distance"]


def build_route(c):
    cal = fit_affine(c["ref"])
    pts = [px_to_ll(p, cal) for p in c["route"]]
    for i, ll in c.get("fixed", {}).items():
        pts[i] = ll
    pts.append(pts[0])                                # circuito cerrado
    out, names, dist = [], [], 0.0
    i = 0
    while i < len(pts) - 1:                           # tramos con el mismo perfil, de hasta 25 puntos
        prof = c.get("legs", {}).get(i, "car")
        j = i + 1
        while j < len(pts) - 1 and j - i < 24 and c.get("legs", {}).get(j, "car") == prof:
            j += 1
        geo, nm, d = osrm_route(pts[i:j + 1], prof)
        out += geo if not out else geo[1:]
        names += nm
        dist += d
        print(f"   tramo {i}-{j} ({prof}): {d / 1000:.2f} km  {' / '.join(nm[:8])}", flush=True)
        i = j
    return out, names, dist, cal, pts


# ------------------------------------------------------------------ procesamiento del circuito
LANDMARKS = {
    "fraynano": {"fraynano": (19.4107401, -99.1087764), "gnp": (19.4050224, -99.0955714),
                 "palacio": (19.4053099, -99.0996762), "aicm": (19.4342349, -99.0733122),
                 "autodromo": (19.4051619, -99.0960636)},
    "chapultepec": {"museo": (19.4261524, -99.1866516), "castillo": (19.4204632, -99.1820866),
                    "angel": (19.4270032, -99.1676682), "monumento": (19.4361745, -99.1546214),
                    "basilica": (19.4848583, -99.1178571)},
    "tangamanga": {"autodromo": (22.1827632, -100.990942)},
    "slp": {"jardin": (22.1879, -100.9366), "hospital": (22.1159575, -100.9859511),
            "politecnica": (22.1210642, -100.9843394), "cruzroja": (22.1400232, -100.9729303),
            "catedral": (22.1516379, -100.9755156), "cajadelagua": (22.1454392, -100.9742005),
            "citadina": (22.182424, -100.9501459), "tangamanga": (22.1263, -101.0010)},
}
CLASS_IDX = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
             "living_street", "service", "road", "pedestrian", "footway", "path", "cycleway", "track", "steps"]


def lanes_halfwidth(t, foot=False):
    """Medio ancho del asfalto (m) segun las etiquetas de la calle."""
    hw = t.get("highway", "road").replace("_link", "")
    try:
        w = float((str(t.get("width", "")).split() or ["x"])[0].replace(",", "."))
        if 4 < w < 60:
            return float(np.clip(w / 2, 3.6, 10.5))
    except ValueError:
        pass
    try:
        lanes = int(str(t.get("lanes", "")).split(";")[0])
    except ValueError:
        lanes = None
    oneway = t.get("oneway") in ("yes", "1", "-1")
    if lanes is None:
        lanes = {"motorway": 3, "trunk": 3, "primary": 3 if oneway else 4, "secondary": 3 if oneway else 4,
                 "tertiary": 2 if oneway else 3, "unclassified": 2, "residential": 2, "living_street": 2,
                 "service": 2}.get(hw, 2)
    half = lanes * 3.15 / 2
    if hw in WALK:
        half = 4.2                                # caminos del parque / plaza: se abren para la carrera
    return float(np.clip(half, 3.6 if not foot else 4.0, 10.5))


def smooth_closed(Q, need, iters=400):
    """Redondea las esquinas de una linea cerrada (1 m entre puntos) hasta que el radio de giro sea >= need."""
    Q = Q.copy()
    k = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
    k /= k.sum()
    for ax in (0, 1):                             # suavizado base (sigma 3 m)
        Q[:, ax] = sum(np.roll(Q[:, ax], -o) * k[o + 9] for o in range(-9, 10))
    for _ in range(iters):
        d = np.roll(Q, -1, 0) - Q
        hd_ = np.arctan2(d[:, 1], d[:, 0])
        dh = (np.roll(hd_, -2) - np.roll(hd_, 2) + np.pi) % (2 * np.pi) - np.pi
        seg = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-3)
        curv = np.abs(dh) / (4 * seg)
        bad = curv * need > 1.0
        if not bad.any():
            break
        m = np.convolve(np.concatenate([bad[-15:], bad, bad[:15]]).astype(float), np.ones(31), "same")[15:-15] > 0
        for _ in range(4):
            L = 0.25 * (np.roll(Q, 1, 0) + np.roll(Q, -1, 0)) + 0.5 * Q
            Q[m] = L[m]
    return Q


def resample_closed(Q, step):
    d = np.hypot(*(np.roll(Q, -1, 0) - Q).T)
    cum = np.concatenate([[0], np.cumsum(d)])
    total = cum[-1]
    n = int(round(total / step))
    s = np.arange(n) * total / n
    QQ = np.vstack([Q, Q[:1]])
    x = np.interp(s, cum, QQ[:, 0])
    z = np.interp(s, cum, QQ[:, 1])
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(Q) - 1)
    return np.column_stack([x, z]), idx


def seg_intersect(p, p2, q, q2):
    """Interseccion de los segmentos p-p2 y q-q2: (t, u) o None."""
    r, s = p2 - p, q2 - q
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    qp = q - p
    t = (qp[0] * s[1] - qp[1] * s[0]) / den
    u = (qp[0] * r[1] - qp[1] * r[0]) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t, u
    return None


def elevations(ll):
    """Altura SRTM (m) de cada punto [(lat, lon)] (OpenTopoData, 100 por consulta)."""
    out = []
    for a in range(0, len(ll), 100):
        locs = "|".join(f"{la:.6f},{lo:.6f}" for la, lo in ll[a:a + 100])
        r = http_json("https://api.opentopodata.org/v1/srtm30m?locations=" + urllib.parse.quote(locs, safe="|,."))
        out += [e["elevation"] if e["elevation"] is not None else np.nan for e in r["results"]]
        time.sleep(1.2)
    a = np.array(out, float)
    if np.isnan(a).any():
        a[np.isnan(a)] = np.nanmean(a)
    return a


def build_route_trace(key, c):
    """La ruta es el trazo del usuario tal cual (calibrado). Se agregan nodos y calles sinteticas al grafo."""
    cal = fit_affine(c["ref"])
    wps = [px_to_ll(p, cal) for p in c["route"]]
    la = np.array([w[0] for w in wps])
    lo = np.array([w[1] for w in wps])
    m = 0.008
    bbox = (la.min() - m, lo.min() - m, la.max() + m, lo.max() + m)
    hw = "|".join(DRIVE + WALK + ("raceway",))
    q = (f'[out:json][timeout:300];way["highway"~"^({hw})$"]({bbox[0]:.5f},{bbox[1]:.5f},{bbox[2]:.5f},'
         f'{bbox[3]:.5f});out body;>;out skel qt;')
    G = Graph(overpass(q, key + "_calles"))
    kx = 111320 * math.cos(math.radians(float(la.mean())))
    race = np.array([G.node[n] for w in G.ways if w.get("tags", {}).get("highway") == "raceway"
                     for n in w["nodes"] if n in G.node]).reshape(-1, 2)
    G.ways.append({"tags": {"highway": "raceway", "name": "Autódromo Tangamanga", "width": "11"}, "nodes": []})
    wa = len(G.ways) - 1
    G.ways.append({"tags": {"highway": "unclassified", "name": c.get("park_name", "Parque Tangamanga II"),
                            "width": "8"}, "nodes": []})
    wb = len(G.ways) - 1
    body = []
    for k, ll in enumerate(wps):                    # un punto cada ~8 m a lo largo del trazo
        nxt = wps[(k + 1) % len(wps)]
        dist = math.hypot((nxt[1] - ll[1]) * kx, (nxt[0] - ll[0]) * 110574)
        steps = max(1, int(dist / 8))
        for j in range(steps):
            t = j / steps
            p = (ll[0] + (nxt[0] - ll[0]) * t, ll[1] + (nxt[1] - ll[1]) * t)
            nid = -1000 - len(body)
            G.node[nid] = p
            on_race = len(race) and np.min(np.hypot((race[:, 1] - p[1]) * kx, (race[:, 0] - p[0]) * 110574)) < 18
            body.append((nid, wa if on_race else wb))
    return G, body, cal, wps


def process(key):
    c = CIRCUITS[key]
    G, body, cal, wps = (build_route_trace if c.get("trace") else build_route_graph)(key, c)
    nodes = [n for n, _ in body]
    wis = [wi for _, wi in body]
    wis[0] = wis[1] if wis[0] is None else wis[0]
    lat0 = float(np.mean([G.node[n][0] for n in nodes]))
    lon0 = float(np.mean([G.node[n][1] for n in nodes]))
    KX, KZ = 111320 * math.cos(math.radians(lat0)), 110574.0

    def proj(lat, lon):
        return ((lon - lon0) * KX, -(lat - lat0) * KZ)

    def unproj(x, z):
        return (lat0 - z / KZ, lon0 + x / KX)
    P = np.array([proj(*G.node[n]) for n in nodes])
    foot_ways = set()
    for wi in wis:
        if G.ways[wi].get("tags", {}).get("highway", "") in WALK:
            foot_ways.add(wi)
    # ---- linea a cada metro con la calle de cada tramo (el tramo i va del nodo i al i+1; usa la calle del nodo i+1)
    Q, Wq = [], []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        L = max(1, int(np.hypot(*(b - a))))
        for k in range(L):
            Q.append(a + (b - a) * k / L)
            Wq.append(wis[(i + 1) % len(P)])
    Q, Wq = np.array(Q), np.array(Wq)
    rw_q = np.array([lanes_halfwidth(G.ways[w].get("tags", {}), w in foot_ways) for w in Wq])
    rw_q = np.minimum(rw_q, np.minimum.reduce([np.roll(rw_q, o) for o in range(-25, 26, 5)]) + 2.5)
    Qs = smooth_closed(Q, np.maximum(rw_q + 2.5, 13.0))         # esquinas bruscas, pero que se puedan tomar
    pts, qi = resample_closed(Qs, 3.0)
    # la salida/meta en el tramo recto mas cercano al punto marcado (la parrilla necesita ~40 m rectos)
    dd_ = np.roll(pts, -1, 0) - pts
    hh_ = np.arctan2(dd_[:, 1], dd_[:, 0])
    bend = np.abs((np.roll(hh_, -15) - np.roll(hh_, 20) + np.pi) % (2 * np.pi) - np.pi)
    cand = np.nonzero(bend < math.radians(12))[0]
    sll = c.get("_start_ll", G.node[nodes[0]])
    s0 = np.array(proj(*sll))
    i0 = int(np.argmin(np.hypot(pts[:, 0] - s0[0], pts[:, 1] - s0[1])))
    if len(cand):
        dd0 = np.abs((cand - i0 + len(pts) // 2) % len(pts) - len(pts) // 2)
        k0 = int(cand[np.argmin(dd0)])
        pts, qi = np.roll(pts, -k0, 0), np.roll(qi, -k0)
        print(f"   salida en un tramo recto a {np.hypot(*(pts[0] - s0)):.0f} m del punto marcado")
    dd_ = np.roll(pts, -1, 0) - pts
    hh_ = np.arctan2(dd_[:, 1], dd_[:, 0])
    turn = np.abs((np.roll(hh_, -8) - hh_ + np.pi) % (2 * np.pi) - np.pi)
    for i in np.nonzero(turn > math.radians(120))[0][::8]:
        la_, lo_ = lat0 - pts[i][1] / 110574.0, lon0 + pts[i][0] / (111320 * math.cos(math.radians(lat0)))
        print(f"   (aviso) vuelta en U en el km {i * 3 / 1000:.2f} ({la_:.5f}, {lo_:.5f})")
    W = Wq[qi]
    n = len(pts)
    tags = [G.ways[w].get("tags", {}) for w in W]
    rw = np.array([lanes_halfwidth(t, w in foot_ways) for t, w in zip(tags, W)])
    for _ in range(6):                                                   # transiciones suaves de ancho
        rw = np.minimum(rw, 0.5 * (np.roll(rw, 1) + np.roll(rw, -1)) + 0.6)
    k5 = np.ones(11) / 11
    rw = np.convolve(np.concatenate([rw[-5:], rw, rw[:5]]), k5, "same")[5:-5]
    cls = np.array([CLASS_IDX.index(t.get("highway", "road").replace("_link", ""))
                    if t.get("highway", "road").replace("_link", "") in CLASS_IDX else 9 for t in tags])
    names = []
    name_idx = []
    for t in tags:
        nm = t.get("name", "")
        if nm not in names:
            names.append(nm)
        name_idx.append(names.index(nm))
    oneway = np.array([1 if t.get("oneway") in ("yes", "1", "-1") or t.get("highway") in ("motorway",) else 0
                       for t in tags])
    nlanes = np.clip(np.round(rw * 2 / 3.15), 1, 6).astype(int)
    bridge = np.array([t.get("bridge", "no") not in ("no", "") for t in tags])
    tunnel = np.array([t.get("tunnel", "no") in ("yes", "building_passage", "covered") for t in tags])
    layer = np.array([int(t.get("layer", "0")) if str(t.get("layer", "0")).lstrip("-").isdigit() else 0
                      for t in tags])
    print(f"   puntos {n}, largo {n * 3 / 1000:.2f} km, puentes {bridge.sum() * 3} m, deprimidos {tunnel.sum() * 3} m")
    # ---- alturas reales del terreno
    s = np.arange(n) * 3.0
    samp = np.arange(0, n, 13)
    ell = [unproj(*pts[i]) for i in samp]
    cache_e = os.path.join(CACHE_DIR, key + "_elev.json")
    if os.path.exists(cache_e) and len(json.load(open(cache_e))) == len(samp):
        ev = np.array(json.load(open(cache_e)))
    else:
        print("   alturas del terreno (SRTM)...", flush=True)
        ev = elevations(ell)
        json.dump(ev.tolist(), open(cache_e, "w"))
    hg = np.interp(s, s[samp], ev, period=n * 3.0)
    ks = np.exp(-0.5 * (np.arange(-40, 41) / 18.0) ** 2)                 # suavizado ~ 55 m
    ks /= ks.sum()
    hg = np.convolve(np.concatenate([hg[-40:], hg, hg[:40]]), ks, "same")[40:-40]
    base_h = float(hg.min())
    hg -= base_h
    # ---- cruces a nivel (nodos compartidos con otras calles), puentes peatonales y calles que pasan encima
    node_ways = {}
    for wi, w in enumerate(G.ways):
        for k_, nd in enumerate(w["nodes"]):
            node_ways.setdefault(nd, []).append((wi, k_))
    route_set = set(nodes)
    inter = []
    tree_pts = pts
    cum_nodes = []
    for i, nd in enumerate(nodes):
        cum_nodes.append(int(np.argmin(np.hypot(*(pts - P[i]).T))))
    for i, nd in enumerate(nodes):
        prv, nxt = nodes[i - 1], nodes[(i + 1) % len(nodes)]
        branches = []
        for wi, k_ in node_ways.get(nd, ()):
            t = G.ways[wi].get("tags", {})
            hwc = t.get("highway", "")
            if hwc not in DRIVE or hwc == "service":
                continue
            for kk in (k_ - 1, k_ + 1):
                if 0 <= kk < len(G.ways[wi]["nodes"]):
                    other = G.ways[wi]["nodes"][kk]
                    if other in (prv, nxt) or other not in G.node:
                        continue
                    branches.append((other, t))
        if not branches:
            continue
        j = cum_nodes[i]
        tj = pts[(j + 1) % n] - pts[j - 1]
        tj /= max(np.hypot(*tj), 1e-6)
        nj = np.array([-tj[1], tj[0]])
        sides = set()
        best = 9
        nm = ""
        for other, t in branches:
            v = np.array(proj(*G.node[other])) - P[i]
            v /= max(np.hypot(*v), 1e-6)
            side = 1 if v @ nj > 0.35 else (-1 if v @ nj < -0.35 else 0)
            sides.add(side)
            ci = CLASS_IDX.index(t.get("highway").replace("_link", "")) if t.get("highway").replace("_link", "") in CLASS_IDX else 9
            if ci < best:
                best, nm = ci, t.get("name", "")
        sd = 0 if (1 in sides and -1 in sides) else (1 if 1 in sides else (-1 if -1 in sides else 0))
        inter.append([j, sd, best, nm])
    # quitar cruces repetidos muy juntos
    inter.sort()
    merged = []
    for it in inter:
        if merged and (it[0] - merged[-1][0]) % n < 5:
            if it[2] < merged[-1][2]:
                merged[-1] = it
            continue
        merged.append(it)
    inter = merged
    # segmentos de la ruta en celdas para cruces geometricos
    cell = 60.0
    grid_ = {}
    for i in range(n):
        a = pts[i]
        grid_.setdefault((int(a[0] // cell), int(a[1] // cell)), []).append(i)

    def route_hits(p1, p2):
        out = []
        cs = set()
        for t_ in np.linspace(0, 1, max(2, int(np.hypot(*(p2 - p1)) / 30) + 2)):
            q = p1 + (p2 - p1) * t_
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cs.add((int(q[0] // cell) + dx, int(q[1] // cell) + dz))
        cand = set()
        for c_ in cs:
            cand.update(grid_.get(c_, ()))
        for i in cand:
            r = seg_intersect(pts[i], pts[(i + 1) % n], p1, p2)
            if r:
                out.append((i, r))
        return out
    pedb, over = [], []
    for wi, w in enumerate(G.ways):
        t = w.get("tags", {})
        hwc = t.get("highway", "")
        is_br = t.get("bridge", "no") not in ("no", "")
        is_tu = t.get("tunnel", "no") in ("yes", "building_passage")
        if not (is_br or is_tu):
            continue
        ns = [x for x in w["nodes"] if x in G.node]
        for a_, b_ in zip(ns[:-1], ns[1:]):
            if a_ in route_set and b_ in route_set:
                continue
            p1, p2 = np.array(proj(*G.node[a_])), np.array(proj(*G.node[b_]))
            for i, _ in route_hits(p1, p2):
                if bridge[i] and is_br:
                    continue
                d = (p2 - p1) / max(np.hypot(*(p2 - p1)), 1e-6)
                if hwc in WALK and is_br:
                    pedb.append([i, float(math.atan2(d[1], d[0]))])
                elif hwc in DRIVE and is_br and not bridge[i]:
                    over.append([i, float(math.atan2(d[1], d[0])), lanes_halfwidth(t)])
    # que pasa por debajo de cada puente de la ruta (para la altura libre)
    under = np.zeros(n)
    for wi, w in enumerate(G.ways):
        t = w.get("tags", {})
        if t.get("bridge", "no") not in ("no", "") or t.get("tunnel", "no") != "no":
            continue
        ns = [x for x in w["nodes"] if x in G.node and x not in route_set]
        for a_, b_ in zip(ns[:-1], ns[1:]):
            p1, p2 = np.array(proj(*G.node[a_])), np.array(proj(*G.node[b_]))
            for i, _ in route_hits(p1, p2):
                if bridge[i]:
                    under[i] = max(under[i], 6.2 if t.get("highway") in DRIVE else 3.0)
    # ---- perfil de altura del camino: terreno + puentes + deprimidos + ondulaciones
    off = np.zeros(n)
    runs = []
    for flag, sign in ((bridge, 1), (tunnel, -1)):
        f = flag.astype(int)
        if f.all() or not f.any():
            continue
        st = int(np.argmin(f))
        i = 0
        while i < n:
            k_ = (st + i) % n
            if f[k_]:
                ln = 1
                while i + ln < n and f[(st + i + ln) % n]:
                    ln += 1
                idx = [(k_ + q) % n for q in range(ln)]
                if ln * 3 >= 18:
                    clear = max([under[q] for q in idx] + [0.0])
                    hgt = (clear + 1.3 if clear > 0 else (5.5 if ln * 3 > 60 else 3.2)) * sign
                    if sign < 0:
                        hgt = -6.8
                    runs.append((k_, ln, hgt))
                i += ln
            else:
                i += 1
    for k_, ln, hgt in runs:
        ramp = int(abs(hgt) * 13 / 3)
        for q in range(-ramp, ln + ramp):
            j = (k_ + q) % n
            if q < 0:
                f_ = (q + ramp) / ramp
            elif q >= ln:
                f_ = (ln + ramp - q) / ramp
            else:
                f_ = 1.0
            f_ = f_ * f_ * (3 - 2 * f_)
            v = hgt * f_
            off[j] = max(off[j], v) if hgt > 0 else min(off[j], v)
    rng = np.random.default_rng(sum(map(ord, key)))
    und = np.zeros(n)
    for wl, amp in ((260.0, 0.9), (120.0, 0.55), (55.0, 0.3)):
        und += amp * np.sin(2 * np.pi * (s / wl + rng.uniform(0, 1)))
    for _ in range(int(n * 3 / 600)):                                    # hundimientos (vados) y lomas
        c0 = rng.uniform(0, n * 3)
        dd = (s - c0 + n * 1.5) % (n * 3) - n * 1.5
        und += rng.choice([-1.1, -0.8, 0.6]) * np.exp(-0.5 * (dd / 14.0) ** 2)
    calm = np.ones(n)
    for it in inter:
        dd = np.abs((np.arange(n) - it[0] + n // 2) % n - n // 2) * 3.0
        calm = np.minimum(calm, np.clip((dd - 12) / 25, 0, 1))
    calm = np.minimum(calm, np.where(np.abs(off) > 0.05, 0.0, 1.0))
    calm = np.convolve(np.concatenate([calm[-8:], calm, calm[:8]]), np.ones(17) / 17, "same")[8:-8]
    h = hg + off + und * calm
    # ---- datos del entorno alrededor de la ruta
    simp = [unproj(*pts[i]) for i in range(0, n, 30)] + [unproj(*pts[0])]
    feats = {"bld": [], "areas": [], "water": [], "trees": [], "pois": [], "rail": []}
    # (la busqueda 'around' con una linea no funciona bien en el servidor: se piden recuadros por tramo
    #  y se filtra aqui lo que quede cerca de la ruta)
    queries = []
    for ci, a in enumerate(range(0, len(simp) - 1, 20)):
        chunk = np.array(simp[a:a + 21])
        m_ = 0.0042
        bb = (chunk[:, 0].min() - m_, chunk[:, 1].min() - m_, chunk[:, 0].max() + m_, chunk[:, 1].max() + m_)
        b = f"{bb[0]:.5f},{bb[1]:.5f},{bb[2]:.5f},{bb[3]:.5f}"
        queries.append((f'[out:json][timeout:300];('
                        f'way["building"]({b});'
                        f'way["leisure"~"park|garden|pitch|stadium|playground"]({b});'
                        f'way["landuse"~"grass|recreation_ground|cemetery|forest|meadow"]({b});'
                        f'way["natural"~"water|wood|scrub"]({b});'
                        f'way["waterway"~"river|canal|stream|drain"]({b});'
                        f'way["railway"~"rail|light_rail|subway|tram"]({b});'
                        f'node["natural"="tree"]({b});'
                        f'node["shop"]({b});node["amenity"~"pharmacy|restaurant|fast_food|bank|fuel|cafe"]({b});'
                        f');out geom qt;', f"{key}_zona_{ci}"))
    la_ = np.array([q[0] for q in simp])
    lo_ = np.array([q[1] for q in simp])
    m_ = 0.03
    queries.append((f'[out:json][timeout:300];way["aeroway"~"runway|taxiway|apron|terminal"]'
                    f'({la_.min() - m_:.5f},{lo_.min() - m_:.5f},{la_.max() + m_:.5f},{lo_.max() + m_:.5f});'
                    f'out geom qt;', f"{key}_aeropuerto"))
    for q, name in queries:
        r = overpass(q, name)
        for e in r["elements"]:
            t = e.get("tags", {})
            if e["type"] == "node":
                xy = [round(v, 1) for v in proj(e["lat"], e["lon"])]
                if t.get("natural") == "tree":
                    feats["trees"].append(xy)
                else:
                    feats["pois"].append(xy + [t.get("shop") or t.get("amenity", ""), t.get("name", "")[:30]])
                continue
            geom = e.get("geometry")
            if not geom:
                continue
            poly = [[round(v, 1) for v in proj(g["lat"], g["lon"])] for g in geom]
            if "building" in t:
                lv = t.get("building:levels", "")
                ht = t.get("height", "")
                try:
                    lv = float((str(lv).split(";") or ["x"])[0].replace(",", "."))
                except ValueError:
                    lv = 0
                try:
                    ht = float((str(ht).split() or ["x"])[0].replace(",", "."))
                except ValueError:
                    ht = 0
                feats["bld"].append({"id": e["id"], "p": poly, "lv": lv, "h": ht, "k": t.get("building", "yes"),
                                     "n": t.get("name", "")[:40], "ly": t.get("layer", "0")})
            elif "waterway" in t:
                feats["water"].append({"id": e["id"], "k": t["waterway"], "p": poly,
                                       "t": t.get("tunnel", "no"), "n": t.get("name", "")})
            elif "railway" in t:
                feats["rail"].append({"id": e["id"], "k": t["railway"], "p": poly, "b": t.get("bridge", "no"),
                                      "t": t.get("tunnel", "no")})
            else:
                k_ = t.get("leisure") or t.get("landuse") or t.get("natural") or t.get("aeroway")
                feats["areas"].append({"id": e["id"], "k": k_, "p": poly, "n": t.get("name", "")})
    for kk in ("bld", "areas", "water", "rail"):                         # quitar repetidos entre bloques
        seen, out = set(), []
        for f_ in feats[kk]:
            if f_["id"] in seen:
                continue
            seen.add(f_["id"])
            f_.pop("id")
            out.append(f_)
        feats[kk] = out
    feats["trees"] = [list(t_) for t_ in {tuple(t_) for t_ in feats["trees"]}]
    # ---- quedarse solo con lo que esta cerca de la ruta
    RP = pts[::4].astype(np.float32)

    def dmin(Q):
        Q = np.asarray(Q, np.float32).reshape(-1, 2)
        out = np.empty(len(Q), np.float32)
        for a in range(0, len(Q), 2048):
            c = Q[a:a + 2048]
            out[a:a + 2048] = np.sqrt(((c[:, None, :] - RP[None]) ** 2).sum(-1).min(1))
        return out
    if feats["bld"]:
        cen = np.array([np.mean(b["p"], axis=0) for b in feats["bld"]])
        dd = dmin(cen)
        feats["bld"] = [b for b, x in zip(feats["bld"], dd) if x < 140]
    for kk, lim in (("areas", 450), ("water", 450), ("rail", 250)):
        keep = []
        for f_ in feats[kk]:
            if f_["k"] in ("runway", "taxiway", "apron", "terminal") or dmin(f_["p"][::3]).min() < lim:
                keep.append(f_)
        feats[kk] = keep
    for kk, lim in (("trees", 60), ("pois", 80)):
        if feats[kk]:
            dd = dmin([q[:2] for q in feats[kk]])
            feats[kk] = [q for q, x in zip(feats[kk], dd) if x < lim]
    # ---- otras calles cerca de la ruta (para la ciudad)
    ways = []
    rx = np.array(pts)
    for wi, w in enumerate(G.ways):
        t = w.get("tags", {})
        hwc = t.get("highway", "")
        if hwc not in DRIVE and hwc not in ("pedestrian", "raceway", "track", "footway", "path"):
            continue
        ns = [x for x in w["nodes"] if x in G.node]
        if len(ns) < 2:
            continue
        poly = np.array([proj(*G.node[x]) for x in ns])
        mid = poly[len(poly) // 2]
        dmin = np.min(np.hypot(rx[::10, 0] - mid[0], rx[::10, 1] - mid[1]))
        if dmin > 450:
            continue
        ways.append({"c": hwc, "n": t.get("name", "")[:40], "w": round(lanes_halfwidth(t), 2),
                     "b": 1 if t.get("bridge", "no") not in ("no", "") else 0,
                     "t": 1 if t.get("tunnel", "no") in ("yes", "building_passage") else 0,
                     "p": [[round(a, 1), round(b, 1)] for a, b in poly], "r": 1 if wi in set(W.tolist()) else 0})
    if c.get("park"):
        calp = fit_affine(c["ref_park"])
        poly = [[round(v, 1) for v in proj(*px_to_ll(q, calp))] for q in c["park"]]
        feats["areas"].append({"k": "bosque", "p": poly + [poly[0]], "n": c.get("park_name", "Parque")})
    lm = {}
    for k_, (la, lo) in LANDMARKS.get(key, {}).items():
        lm[k_] = [round(v, 1) for v in proj(la, lo)]
    data = dict(key=key, attribution=ATTRIBUTION, lat0=lat0, lon0=lon0, base_h=base_h,
                pts=np.round(pts, 2).tolist(), rw=np.round(rw, 2).tolist(), h=np.round(h, 2).tolist(),
                hg=np.round(hg, 2).tolist(), br=bridge.astype(int).tolist(), tu=tunnel.astype(int).tolist(),
                off=np.round(off, 2).tolist(), cls=cls.tolist(), name=name_idx, names=names, inter=inter,
                ow=oneway.tolist(), ln=nlanes.tolist(),
                pedb=pedb, over=over, ways=ways, lm=lm, **feats)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, key + ".json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"   guardado {path}: edificios {len(feats['bld'])}, areas {len(feats['areas'])}, "
          f"agua {len(feats['water'])}, arboles {len(feats['trees'])}, comercios {len(feats['pois'])}, "
          f"calles {len(ways)}, cruces {len(inter)}, puentes peatonales {len(pedb)}, pasos superiores {len(over)}, "
          f"{os.path.getsize(path) / 1e6:.1f} MB")
    return data


def debug_overlay(key, c, route_ll, cal, image, out_png):
    """Dibuja la ruta calculada sobre la imagen del usuario para compararla."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ca, co = cal
    M = np.array([[ca[0], ca[1]], [co[0], co[1]]])
    Mi = np.linalg.inv(M)
    ll = np.array(route_ll)
    pxy = (Mi @ (ll - np.array([ca[2], co[2]])).T).T
    img = plt.imread(image)
    plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    plt.imshow(img)
    plt.plot(pxy[:, 0], pxy[:, 1], "-", color=(0, 0.85, 0.2), lw=2.0)
    d = np.array([r[0] if isinstance(r[0], tuple) else r for r in c["route"]])
    plt.plot(d[:, 0], d[:, 1], "o", color="magenta", ms=3)
    plt.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(out_png)
    plt.close()


if __name__ == "__main__" and len(sys.argv) == 2:
    keys = list(CIRCUITS) if sys.argv[1] == "todos" else [sys.argv[1]]
    for key in keys:
        print("==", key, flush=True)
        process(key)
elif __name__ == "__main__":
    key = sys.argv[1]
    c = CIRCUITS[key]
    G, body, cal, wp = build_route_graph(key, c)
    route = [G.node[n] for n, _ in body]
    dist = sum(G.dist(a, b) for (a, _), (b, _) in zip(body[:-1], body[1:]))
    names = []
    for _, wi in body:
        nm = G.ways[wi].get("tags", {}).get("name", "?") if wi is not None else "?"
        if not names or names[-1] != nm:
            names.append(nm)
    print(f"{key}: {dist / 1000:.2f} km, {len(route)} nodos")
    print("   calles: " + " / ".join(names))
    if len(sys.argv) > 3:
        debug_overlay(key, c, route + [route[0]], cal, sys.argv[2], sys.argv[3])

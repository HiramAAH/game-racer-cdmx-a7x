"""
GTC Racing HD - menus del juego (se mezcla con GameMaps en gtc_racing_hd_mapas.py).

  - Menu principal: carrera rapida, torneo (rapido / corto / completo), multijugador LAN, configuracion, salir.
  - Pausa con ESC: continuar, reiniciar, configuracion, salir al menu, salir del juego.
  - Configuracion (se guarda en gtc_config.json): controles (teclas), sensibilidad, graficos y audio.
  - Torneo: varias pistas seguidas con los mismos rivales y tabla de puntos.
"""
import json
import math
import os
import random

import pygame

import gtc_racing_hd as hd
from gtc_racing_hd import (PILOTS, DIFFS, QUALITY, TOD_HD, ACCENT, ORANGE, WIN_W, WIN_H, font, clamp, fmt_time)

VERSION = "1.0.0"
CFG_PATH = os.path.join(hd.HERE, "gtc_config.json")

# acciones de manejo -> dos teclas cada una
ACTIONS = [("acelerar", "ACELERAR"), ("frenar", "FRENAR / REVERSA"), ("izquierda", "IZQUIERDA"),
           ("derecha", "DERECHA"), ("nitro", "NITRO"), ("derrape", "FRENO DE MANO / DERRAPE"),
           ("camara", "CAMBIAR CÁMARA"), ("cancion", "SIGUIENTE CANCIÓN")]
DEFAULT_KEYS = {"acelerar": [pygame.K_UP, pygame.K_w], "frenar": [pygame.K_DOWN, pygame.K_s],
                "izquierda": [pygame.K_LEFT, pygame.K_a], "derecha": [pygame.K_RIGHT, pygame.K_d],
                "nitro": [pygame.K_LSHIFT, pygame.K_x], "derrape": [pygame.K_SPACE, None],
                "camara": [pygame.K_c, None], "cancion": [pygame.K_LCTRL, pygame.K_RCTRL]}
DEFAULT_CFG = dict(keys=DEFAULT_KEYS, steer_sens=1.0, steer_speed=1.0, steer_assist=True,
                   quality=2, show_fps=False, cam=0, shake=True, fps=60,
                   music_vol=0.6, fx_vol=1.0, muted=False)
FPS_OPTS = [30, 60, 120, 144, 0]
CAMS = ["PERSECUCIÓN", "LEJANA", "COFRE", "DEFENSA"]

# torneos: numero de carreras (None = todas las pistas)
TOURS = [("rapido", "TORNEO RÁPIDO", 3, "3 carreras en las pistas más cortas"),
         ("corto", "TORNEO CORTO", 5, "5 carreras, de la más corta a la más larga"),
         ("completo", "TORNEO COMPLETO", None, "Todas las pistas · gran final en la más larga")]
POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

MAIN_ITEMS = [("rapida", "CARRERA RÁPIDA", "Elige piloto, pista, dificultad y horario y a correr."),
              ("torneo", "TORNEO", "Varias carreras seguidas contra los mismos rivales: gana el campeonato."),
              ("chase", "MODO PERSECUCIÓN", "Escapa de la policía el tiempo que pide la pista... o sé la policía."),
              ("local", "PANTALLA DIVIDIDA", "De 1 a 4 jugadores en la misma pantalla, con controles o teclado."),
              ("multi", "MULTIJUGADOR LAN", "Juega contra otra computadora en tu red local."),
              ("config", "CONFIGURACIÓN", "Controles, sensibilidad, gráficos y audio."),
              ("salir", "SALIR DEL JUEGO", "Cerrar GTC Racing.")]
MULTI_ITEMS = [("host", "CREAR PARTIDA", "Tú eres el anfitrión: eliges la pista y el horario."),
               ("join", "UNIRSE A UNA PARTIDA", "Busca partidas en tu red local o escribe la IP del anfitrión."),
               ("back", "VOLVER", "")]
CHASE_ITEMS = [("fugitivo", "ESCAPAR DE LA POLICÍA",
                "Aguanta el tiempo de la pista. Si la patrulla te derriba o te cierra el paso, perdiste."),
               ("policia", "SER EL POLICÍA", "Persigue al fugitivo y deténlo antes de que se acabe el tiempo."),
               ("back", "VOLVER", "")]
PAUSE_ITEMS = [("cont", "CONTINUAR"), ("restart", "REINICIAR CARRERA"), ("config", "CONFIGURACIÓN"),
               ("menu", "SALIR AL MENÚ"), ("quit", "SALIR DEL JUEGO")]
CFG_TABS = ["CONTROLES", "SENSIBILIDAD", "GRÁFICOS", "AUDIO"]


def key_name(k):
    if k is None:
        return "—"
    n = pygame.key.name(k).upper()
    return {"LEFT SHIFT": "SHIFT IZQ", "RIGHT SHIFT": "SHIFT DER", "LEFT CTRL": "CTRL IZQ", "RIGHT CTRL": "CTRL DER",
            "SPACE": "ESPACIO", "UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→", "RETURN": "ENTER"}.get(n, n)


def load_cfg():
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    try:
        with open(CFG_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        for k, v in saved.items():
            if k == "keys":
                for a, ks in v.items():
                    if a in cfg["keys"] and isinstance(ks, list) and len(ks) == 2:
                        cfg["keys"][a] = ks
            elif k in cfg:
                cfg[k] = v
    except (OSError, ValueError):
        pass
    return cfg


class MenuMixin:
    """Menus, pausa, torneo y configuracion. Va antes de GameHD en la herencia de GameMaps."""

    # ------------------------------------------------------------ estado
    def menu_init(self):
        self.cfg = load_cfg()
        self.page = "main"                 # main / torneo / multi
        self.msel = 0
        self.paused = False
        self.psel = 0
        self.cfg_open = False              # pantalla de configuracion encima de lo que sea
        self.cfg_tab, self.cfg_row = 0, 0
        self.wait_key = None               # (accion, columna) esperando una tecla
        self.tour = None
        self.quick_mode = True
        self.fps_cap = 60
        self.apply_cfg()

    def save_cfg(self):
        try:
            with open(CFG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=1)
        except OSError:
            pass

    def apply_cfg(self):
        c = self.cfg
        qi = int(clamp(c["quality"], 0, len(QUALITY) - 1))
        if qi != getattr(self, "qi", None):
            self.qi = qi
            self.R.set_quality(QUALITY[qi][1])
        self.show_fps = bool(c["show_fps"])
        self.cam_mode = int(c["cam"]) % 4
        self.fps_cap = int(c["fps"])
        self.music.vol = float(c["music_vol"])
        self.sound.muted = bool(c["muted"])
        self.apply_fx_volume()

    def apply_fx_volume(self):
        snd = self.sound
        if not getattr(snd, "ok", False):
            return
        if not hasattr(self, "_fx_base"):
            self._fx_base = {k: v.get_volume() for k, v in vars(snd).items() if isinstance(v, pygame.mixer.Sound)}
        fv = float(self.cfg["fx_vol"])
        for k, v0 in self._fx_base.items():
            getattr(snd, k).set_volume(v0 * fv)

    def keys_of(self, action):
        return [k for k in self.cfg["keys"].get(action, []) if k is not None]

    @property
    def frozen(self):
        """Pausa real (en LAN la carrera sigue: solo se muestra el menu)."""
        return getattr(self, "paused", False) and not self.mp

    # ------------------------------------------------------------ manejo con las teclas configuradas
    def player_inputs(self):
        k = pygame.key.get_pressed()
        on = lambda a: any(k[q] for q in self.keys_of(a) if 0 <= q < len(k))
        st = (1 if on("derecha") else 0) - (1 if on("izquierda") else 0)
        return on("acelerar"), on("frenar"), st, on("derrape"), on("nitro")

    # ------------------------------------------------------------ carrera: rivales al azar (6 coches) o del torneo
    def new_race(self, attract=False, intro=True):
        full = list(PILOTS)
        if attract:
            me = random.choice(full)
        else:
            me = full[self.sel % len(full)]
        others = [p for p in full if p is not me]
        if getattr(self, "tour", None) and not attract:
            rivals = [p for p in full if p["name"] in self.tour["rivals"]]
        else:
            rivals = random.sample(others, min(5, len(others)))
        sel0 = self.sel
        PILOTS[:] = rivals + [me]
        self.sel = len(PILOTS) - 1
        try:
            super().new_race(attract=attract, intro=intro)
        finally:
            PILOTS[:] = full
            self.sel = sel0
        self.paused = False

    # ------------------------------------------------------------ actualizar
    def update(self, rdt):
        if self.cfg_open or self.frozen or self.state == "menu":
            self.update_music_choice()
        if self.frozen:
            self.toast_t = max(0.0, self.toast_t - rdt)
            self.fade = min(1.0, self.fade + rdt * 2.0)
            self.music.set_mode(self.music_mode())
            self.music.update(rdt)
            self.sound.stop_engine()
            return
        if not self.cfg["shake"]:
            self.shake = 0.0
        super().update(rdt)
        if not self.cfg["shake"]:
            self.shake = 0.0
        if getattr(self.sound, "ok", False) and getattr(self.sound, "level", -1) >= 0:
            self.sound.engine_ch.set_volume(0.5 * float(self.cfg["fx_vol"]))

    def update_music_choice(self):
        pass

    # ------------------------------------------------------------ teclado
    def handle_key(self, key):
        st = self.state
        if self.wait_key is not None:                                      # reasignar una tecla
            a, col = self.wait_key
            self.wait_key = None
            if key != pygame.K_ESCAPE:
                for b, ks in self.cfg["keys"].items():                    # la tecla deja su uso anterior
                    for j in range(2):
                        if ks[j] == key and not (b == a and j == col):
                            ks[j] = None
                self.cfg["keys"][a][col] = key
            return True
        if self.cfg_open:
            return self.key_config(key)
        if self.paused:
            return self.key_pause(key)
        racing = st in ("intro", "countdown", "race", "finished")
        if key == pygame.K_ESCAPE:
            if racing:
                self.paused, self.psel = True, 0
                return True
            if st == "menu":
                if self.page != "main":
                    self.page, self.msel = "main", 0
                else:
                    self.msel = len(MAIN_ITEMS) - 1
                return True
            if st == "select":
                self.tour = None
                self.new_race(attract=True)
                self.chase = None
                self.page, self.msel = "main", 0
                return True
            return super().handle_key(key)                                 # LAN: salir de la partida
        if st == "menu":
            return self.key_menu(key)
        if st == "intro" and key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self.end_intro()
            return True
        if racing:
            if key in self.keys_of("camara"):
                self.cam_mode = (self.cam_mode + 1) % 4
                self.cfg["cam"] = self.cam_mode
                self.toast, self.toast_t = "CÁMARA: " + CAMS[self.cam_mode], 1.5
                return True
            if key in self.keys_of("cancion"):
                if self.music.mode == "playlist":
                    self.music.next_song()
                return True
            driving = [q for a in ("acelerar", "frenar", "izquierda", "derecha", "nitro", "derrape")
                       for q in self.keys_of(a)]
            if key in driving or key in (pygame.K_c, pygame.K_LCTRL, pygame.K_RCTRL):
                return True                                                # no cambia horario, camara, etc.
            if st == "finished" and key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return self.after_race()
            if st == "finished" and key == pygame.K_r and self.tour:
                return True                                                # en torneo no se repite
        if st == "select" and key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.tour:
            self.start_tour()
            return True
        if st == "select" and self.tour and key in (pygame.K_q, pygame.K_e):
            return True
        if key == pygame.K_F1:
            r = super().handle_key(key)
            self.cfg["quality"] = self.qi
            self.save_cfg()
            return r
        return super().handle_key(key)

    def key_menu(self, key):
        items = self.menu_items()
        if key in (pygame.K_UP, pygame.K_w):
            self.msel = (self.msel - 1) % len(items)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.msel = (self.msel + 1) % len(items)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            return self.menu_pick(items[self.msel][0])
        elif key == pygame.K_h:                                            # atajos de antes
            self.start_host()
        elif key == pygame.K_j:
            self.start_browse()
        elif key in (pygame.K_m, pygame.K_n, pygame.K_F1, pygame.K_F3):
            return super().handle_key(key)
        return True

    def menu_items(self):
        if self.page == "torneo":
            return [(k, n, d) for k, n, _, d in TOURS] + [("back", "VOLVER", "")]
        if self.page == "multi":
            return MULTI_ITEMS
        if self.page == "chase":
            return CHASE_ITEMS
        return MAIN_ITEMS

    def menu_pick(self, k):
        if k == "rapida":
            self.tour = None
            self.chase = None
            self.go_select()
        elif k == "chase":
            self.page, self.msel = "chase", 0
        elif k == "local":
            self.open_local()
        elif k in ("fugitivo", "policia"):
            self.start_chase_menu(k)
        elif k == "torneo":
            self.page, self.msel = "torneo", 0
        elif k == "multi":
            self.page, self.msel = "multi", 0
        elif k == "config":
            self.open_config()
        elif k == "salir":
            return False
        elif k == "back":
            self.page, self.msel = "main", 0
        elif k == "host":
            self.page = "main"
            self.start_host()
        elif k == "join":
            self.page = "main"
            self.start_browse()
        elif k in [t[0] for t in TOURS]:
            self.chase = None
            self.prepare_tour(k)
            self.go_select()
        return True

    def go_select(self):
        self.state = "select"
        self.fade = 0.0

    def key_pause(self, key):
        if key == pygame.K_ESCAPE:
            self.paused = False
        elif key in (pygame.K_UP, pygame.K_w):
            self.psel = (self.psel - 1) % len(PAUSE_ITEMS)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.psel = (self.psel + 1) % len(PAUSE_ITEMS)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            k = PAUSE_ITEMS[self.psel][0]
            if k == "cont":
                self.paused = False
            elif k == "restart":
                self.paused = False
                if self.mp:
                    if self.mp == "host":
                        self.start_mp_race()
                    else:
                        self.toast, self.toast_t = "SOLO EL ANFITRIÓN PUEDE REINICIAR", 2.0
                else:
                    self.new_race(intro=False)
            elif k == "config":
                self.open_config()
            elif k == "menu":
                self.paused = False
                self.tour = None
                if getattr(self, "chase", None):
                    self.chase_end_cleanup()
                if self.mp or self.disc:
                    self.leave_mp(silent=True)
                self.new_race(attract=True)
                self.page, self.msel = "main", 0
            elif k == "quit":
                return False
        return True

    # ------------------------------------------------------------ configuracion
    def open_config(self):
        self.cfg_open, self.cfg_row = True, 0

    def cfg_rows(self):
        c = self.cfg
        tab = CFG_TABS[self.cfg_tab]
        if tab == "CONTROLES":
            return [("key", a, lab) for a, lab in ACTIONS] + [("reset_keys", None, "RESTABLECER CONTROLES")]
        if tab == "SENSIBILIDAD":
            return [("num", "steer_sens", "SENSIBILIDAD DEL VOLANTE", 0.5, 1.5, 0.05,
                     f"{c['steer_sens'] * 100:.0f} %", "Cuánto gira el coche con la misma tecla (sobre todo a alta velocidad)."),
                    ("num", "steer_speed", "RAPIDEZ DEL VOLANTE", 0.5, 2.0, 0.1,
                     f"{c['steer_speed'] * 100:.0f} %", "Qué tan rápido llega el volante a tope al mantener la tecla."),
                    ("bool", "steer_assist", "AYUDA A ALTA VELOCIDAD", None, None, None,
                     "SÍ" if c["steer_assist"] else "NO", "Suaviza el volante a más de 90 km/h para no derrapar.")]
        if tab == "GRÁFICOS":
            return [("opt", "quality", "CALIDAD GRÁFICA", [q[0] for q in QUALITY], None, None,
                     QUALITY[c["quality"]][0], "Sombras, reflejos, distancia de dibujo y resolución interna."),
                    ("opt", "fps", "LÍMITE DE FPS", FPS_OPTS, None, None,
                     "SIN LÍMITE" if c["fps"] == 0 else str(c["fps"]), "Cuadros por segundo máximos."),
                    ("opt", "cam", "CÁMARA", CAMS, None, None, CAMS[c["cam"]], "Cámara al empezar la carrera."),
                    ("bool", "shake", "SACUDIDA DE CÁMARA", None, None, None, "SÍ" if c["shake"] else "NO",
                     "La cámara tiembla en los choques."),
                    ("bool", "show_fps", "MOSTRAR FPS", None, None, None, "SÍ" if c["show_fps"] else "NO", "")]
        return [("num", "music_vol", "VOLUMEN DE LA MÚSICA", 0.0, 1.0, 0.05, f"{c['music_vol'] * 100:.0f} %", ""),
                ("num", "fx_vol", "VOLUMEN DE EFECTOS", 0.0, 1.0, 0.05, f"{c['fx_vol'] * 100:.0f} %",
                 "Motor, choques, nitro y avisos."),
                ("bool", "muted", "SILENCIAR TODO", None, None, None, "SÍ" if c["muted"] else "NO", "")]

    def key_config(self, key):
        rows = self.cfg_rows()
        self.cfg_row = min(self.cfg_row, len(rows) - 1)
        if key == pygame.K_ESCAPE:
            self.cfg_open = False
            self.save_cfg()
            self.apply_cfg()
            return True
        if key in (pygame.K_TAB, pygame.K_q, pygame.K_e, pygame.K_PAGEUP, pygame.K_PAGEDOWN):
            d = -1 if key in (pygame.K_q, pygame.K_PAGEUP) else 1
            self.cfg_tab, self.cfg_row = (self.cfg_tab + d) % len(CFG_TABS), 0
            return True
        if key in (pygame.K_UP, pygame.K_w):
            self.cfg_row = (self.cfg_row - 1) % len(rows)
            return True
        if key in (pygame.K_DOWN, pygame.K_s):
            self.cfg_row = (self.cfg_row + 1) % len(rows)
            return True
        r = rows[self.cfg_row]
        kind = r[0]
        left, right = key in (pygame.K_LEFT, pygame.K_a), key in (pygame.K_RIGHT, pygame.K_d)
        enter = key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)
        if kind == "key":
            col = getattr(self, "cfg_col", 0)
            if left or right:
                self.cfg_col = 1 - col
            elif enter:
                self.wait_key = (r[1], col)
            elif key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                self.cfg["keys"][r[1]][col] = None
        elif kind == "reset_keys" and enter:
            self.cfg["keys"] = json.loads(json.dumps(DEFAULT_KEYS))
            self.toast, self.toast_t = "CONTROLES RESTABLECIDOS", 1.5
        elif kind == "num" and (left or right):
            _, k, _, lo, hi, step = r[:6]
            self.cfg[k] = round(clamp(self.cfg[k] + (step if right else -step), lo, hi), 2)
        elif kind == "bool" and (left or right or enter):
            self.cfg[r[1]] = not self.cfg[r[1]]
        elif kind == "opt" and (left or right or enter):
            k, opts = r[1], r[3]
            if k == "fps":
                i = FPS_OPTS.index(self.cfg[k]) if self.cfg[k] in FPS_OPTS else 1
                self.cfg[k] = FPS_OPTS[(i + (-1 if left else 1)) % len(FPS_OPTS)]
            else:
                self.cfg[k] = (self.cfg[k] + (-1 if left else 1)) % len(opts)
        self.apply_cfg()
        return True

    # ------------------------------------------------------------ torneo
    def prepare_tour(self, kind):
        import gtc_racing_hd_mapas as M
        name, n = next((t[1], t[2]) for t in TOURS if t[0] == kind)
        dist = lambda i: (M.MAPS[i].length or 0) * M.MAPS[i].laps
        order = sorted(range(len(M.MAPS)), key=dist)
        maps = order[:n] if n else order
        maps = sorted(maps, key=dist)
        self.tour = dict(kind=kind, name=name, maps=maps, idx=0, points={}, history=[], rivals=[], done=False)

    def start_tour(self):
        """Primera carrera del torneo: se fijan los rivales y se carga la primera pista."""
        t = self.tour
        me = PILOTS[self.sel]
        others = [p for p in PILOTS if p is not me]
        t["rivals"] = [p["name"] for p in random.sample(others, min(5, len(others)))]
        t["me"] = me["name"]
        t["points"] = {n: 0 for n in t["rivals"] + [me["name"]]}
        t["idx"], t["history"], t["done"] = 0, [], False
        self.tour_load_and_race()

    def tour_load_and_race(self):
        t = self.tour
        i = t["maps"][t["idx"]]
        self.map_sel = i
        if i != self.map_i:
            self.load_map(i)
        self.new_race()

    def tour_score(self):
        """Puntos de esta carrera (una sola vez)."""
        t = self.tour
        if len(t["history"]) > t["idx"]:
            return t["history"][t["idx"]]
        order = self.standings()
        res = []
        for pos, c in enumerate(order):
            pts = POINTS[pos] if pos < len(POINTS) else 0
            t["points"][c.name] = t["points"].get(c.name, 0) + pts
            res.append((c.name, pts))
        t["history"].append(res)
        return res

    def after_race(self):
        if self.results_t < 2.0:
            return True
        if not self.tour:
            self.new_race(intro=False)
            return True
        t = self.tour
        self.tour_score()
        if t["done"]:
            self.tour = None
            self.new_race(attract=True)
            self.page, self.msel = "main", 0
            return True
        if t["idx"] + 1 >= len(t["maps"]):
            t["done"] = True
            return True
        t["idx"] += 1
        self.tour_load_and_race()
        return True

    # ------------------------------------------------------------ dibujo
    def draw_hud(self):
        st = self.state
        if st == "menu":
            self.hud.fill((0, 0, 0, 0))
            if not self.cfg_open:
                self.hud_main_menu()
        elif st == "finished" and self.tour and self.tour.get("done") and not self.paused and not self.cfg_open:
            self.hud.fill((0, 0, 0, 0))
            self.hud_champion()
        else:
            super().draw_hud()
        if self.paused and not self.cfg_open:
            self.hud_pause()
        if self.cfg_open:
            self.hud_config()

    def dim(self, a=150):
        s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        s.fill((0, 0, 0, a))
        self.hud.blit(s, (0, 0))

    def hud_main_menu(self):
        h = self.hud
        grad = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for x in range(0, WIN_W, 8):
            a = int(200 * clamp(1 - x / (WIN_W * 0.62), 0, 1) ** 1.4)
            grad.fill((0, 0, 0, a), (x, 0, 8, WIN_H))
        h.blit(grad, (0, 0))
        r_ = self.txt("GTC RACING", 96, (255, 255, 255), (60, 40), glow=ACCENT)
        self.panel((r_.right + 18, 62, 96, 62), 255, col=(0, 180, 235), slant=14)
        self.txt("HD", 46, (255, 255, 255), (r_.right + 66, 92), "center", shadow=False)
        sub = {"main": "MENÚ PRINCIPAL", "torneo": "TORNEO", "multi": "MULTIJUGADOR LAN", "local": "",
               "chase": "MODO PERSECUCIÓN"}[self.page]
        self.txt(sub, 26, ACCENT, (66, 150))
        items = self.menu_items()
        self.msel %= len(items)
        for i, it in enumerate(items):
            k, name = it[0], it[1]
            y = 210 + i * 66
            sel = i == self.msel
            w = 470 if sel else 430
            self.panel((50, y, w, 54), 215 if sel else 120, accent=ORANGE if sel else (90, 100, 120), slant=18)
            col = (255, 255, 255) if sel else (175, 185, 200)
            self.txt(name, 30 if sel else 27, col, (82, y + 10), glow=ACCENT if sel else None)
            if sel and int(self.t * 2.5) % 2 == 0:
                self.txt("›", 34, ORANGE, (w + 18, y + 6))
        desc = items[self.msel][2] if len(items[self.msel]) > 2 else ""
        if self.page == "torneo" and items[self.msel][0] != "back":
            import gtc_racing_hd_mapas as M
            kind = items[self.msel][0]
            n = next(t[2] for t in TOURS if t[0] == kind)
            dist = lambda i: (M.MAPS[i].length or 0) * M.MAPS[i].laps
            order = sorted(range(len(M.MAPS)), key=dist)
            maps = sorted(order[:n] if n else order, key=dist)
            self.panel((WIN_W - 520, 210, 470, 70 + 40 * len(maps)), 180, accent=ACCENT, slant=18)
            self.txt("CARRERAS", 18, (170, 190, 210), (WIN_W - 486, 222), italic=False)
            for j, i in enumerate(maps):
                m = M.MAPS[i]
                self.txt(f"{j + 1}.  {m.name}", 22, (255, 255, 255), (WIN_W - 486, 252 + j * 40))
                self.txt(f"{dist(i) / 1000:.1f} km", 18, (190, 200, 215), (WIN_W - 70, 256 + j * 40), "topright",
                         italic=False)
        if desc:
            self.panel((50, WIN_H - 140, 760, 50), 150, slant=14)
            self.txt(desc, 20, (215, 225, 238), (76, WIN_H - 126), italic=False)
        self.panel((0, WIN_H - 64, WIN_W, 64), 150, slant=0)
        self.txt(f"v{VERSION}", 16, (150, 165, 185), (WIN_W - 16, WIN_H - 32), "midright", italic=False)
        self.txt("↑ ↓  elegir      ENTER  aceptar      ESC  volver      M  sonido", 19, (200, 210, 225),
                 (WIN_W // 2, WIN_H - 32), "center", italic=False, fname="segoeui,arial")
        self.song_banner(18)

    def hud_pause(self):
        self.dim(160)
        self.txt("PAUSA", 90, (255, 255, 255), (WIN_W // 2, 140), "center", glow=ACCENT)
        if self.mp:
            self.txt("En LAN la carrera sigue corriendo", 18, ORANGE, (WIN_W // 2, 200), "center", italic=False)
        for i, (k, name) in enumerate(PAUSE_ITEMS):
            y = 240 + i * 66
            sel = i == self.psel
            self.panel((WIN_W // 2 - 230, y, 460, 54), 215 if sel else 120, accent=ORANGE if sel else (90, 100, 120),
                       slant=18)
            self.txt(name, 30 if sel else 26, (255, 255, 255) if sel else (175, 185, 200), (WIN_W // 2, y + 27),
                     "center", glow=ACCENT if sel else None)
        self.txt("↑ ↓  elegir      ENTER  aceptar      ESC  continuar", 19, (200, 210, 225),
                 (WIN_W // 2, WIN_H - 40), "center", italic=False, fname="segoeui,arial")

    def hud_config(self):
        self.dim(185)
        self.txt("CONFIGURACIÓN", 64, (255, 255, 255), (WIN_W // 2, 60), "center", glow=ACCENT)
        tw = 230
        x0 = WIN_W // 2 - tw * len(CFG_TABS) // 2
        for i, t in enumerate(CFG_TABS):
            sel = i == self.cfg_tab
            self.panel((x0 + i * tw + 6, 110, tw - 12, 46), 220 if sel else 110,
                       accent=ORANGE if sel else (80, 90, 110), slant=14)
            self.txt(t, 22, (255, 255, 255) if sel else (160, 170, 185), (x0 + i * tw + tw // 2, 133), "center")
        rows = self.cfg_rows()
        self.cfg_row = min(self.cfg_row, len(rows) - 1)
        tab = CFG_TABS[self.cfg_tab]
        y0 = 180
        rh = 46 if tab == "CONTROLES" else 64
        if tab == "CONTROLES":
            self.txt("TECLA 1", 16, (170, 185, 200), (WIN_W // 2 + 170, y0 - 4), "center", italic=False)
            self.txt("TECLA 2", 16, (170, 185, 200), (WIN_W // 2 + 360, y0 - 4), "center", italic=False)
            y0 += 16
        for i, r in enumerate(rows):
            y = y0 + i * rh
            sel = i == self.cfg_row
            self.panel((WIN_W // 2 - 520, y, 1040, rh - 10), 200 if sel else 110,
                       accent=ORANGE if sel else (70, 80, 100), slant=12)
            lab = r[2]
            self.txt(lab, 21, (255, 255, 255) if sel else (185, 195, 210), (WIN_W // 2 - 490, y + (rh - 10) // 2),
                     "midleft")
            if r[0] == "key":
                col_sel = getattr(self, "cfg_col", 0)
                for j in range(2):
                    cx = WIN_W // 2 + 170 + j * 190
                    waiting = self.wait_key == (r[1], j)
                    on = sel and col_sel == j
                    self.panel((cx - 80, y + 5, 160, rh - 20), 230 if on else 120,
                               accent=ACCENT if on else (60, 70, 90), slant=10)
                    s = "PRESIONA..." if waiting else key_name(self.cfg["keys"][r[1]][j])
                    self.txt(s, 19, (255, 230, 90) if waiting else ((255, 255, 255) if on else (190, 200, 215)),
                             (cx, y + (rh - 10) // 2), "center", italic=False, fname="segoeui,arial")
            elif r[0] != "reset_keys":
                val = r[6]
                self.txt(f"‹  {val}  ›" if sel else val, 24, ACCENT if sel else (220, 228, 240),
                         (WIN_W // 2 + 470, y + (rh - 10) // 2), "midright")
                if r[0] == "num":
                    lo, hi = r[3], r[4]
                    f = (self.cfg[r[1]] - lo) / (hi - lo)
                    bx, bw = WIN_W // 2 + 40, 250
                    pygame.draw.rect(self.hud, (50, 58, 74, 255), (bx, y + rh // 2 - 8, bw, 6))
                    pygame.draw.rect(self.hud, ACCENT + (255,), (bx, y + rh // 2 - 8, int(bw * f), 6))
        r = rows[self.cfg_row]
        tip = r[7] if len(r) > 7 else ("ENTER cambiar tecla · ← → columna · SUPR borrar   ·   CONTROL: RT acelerar, "
                                        "LT frenar, stick dirección, A/RB nitro, B derrape, Y cámara, START pausa"
                                        if r[0] == "key" else "")
        if tip:
            self.txt(tip, 18 if font(18).size(tip)[0] < WIN_W - 40 else 14, (190, 205, 220), (WIN_W // 2, WIN_H - 78),
                     "center", italic=False, fname="segoeui,arial")
        self.panel((0, WIN_H - 56, WIN_W, 56), 150, slant=0)
        self.txt("Q / E  pestaña      ↑ ↓  opción      ← →  cambiar      ESC  guardar y volver", 19, (200, 210, 225),
                 (WIN_W // 2, WIN_H - 28), "center", italic=False, fname="segoeui,arial")

    def hud_champion(self):
        t = self.tour
        self.tour_score()
        self.dim(120)
        tab = sorted(t["points"].items(), key=lambda kv: -kv[1])
        champ = tab[0][0]
        me_won = champ == t["me"]
        self.txt("¡CAMPEÓN!" if me_won else "FIN DEL TORNEO", 80, (255, 215, 60) if me_won else (255, 255, 255),
                 (WIN_W // 2, 70), "center", glow=ORANGE if me_won else ACCENT)
        self.txt(t["name"], 26, ACCENT, (WIN_W // 2, 130), "center")
        self.standings_panel(WIN_W // 2 - 380, 170, 760, tab, big=True)
        self.txt("ENTER  volver al menú", 22, (210, 220, 230), (WIN_W // 2, WIN_H - 50), "center", italic=False)

    def standings_panel(self, x, y, w, tab, big=False):
        rh = 58 if big else 40
        self.panel((x, y, w, 30 + rh * len(tab)), 200, accent=ORANGE, slant=24)
        cols = {p["name"]: tuple(p["map"]) for p in PILOTS}
        for i, (name, pts) in enumerate(tab):
            yy = y + 16 + i * rh
            me = self.tour and name == self.tour.get("me")
            col = (255, 215, 60) if me else (235, 240, 245)
            self.panel((x + 24, yy, w - 48, rh - 8), 130, accent=cols.get(name, ACCENT), slant=12)
            self.txt(f"{i + 1}", 30 if big else 22, col, (x + 48, yy + (rh - 8) // 2), "midleft")
            self.txt(name, 28 if big else 20, col, (x + 96, yy + (rh - 8) // 2), "midleft")
            self.txt(f"{pts} PTS", 28 if big else 20, col, (x + w - 48, yy + (rh - 8) // 2), "midright")

    def results(self, order):
        if not self.tour:
            super().results(order)
            self.panel((WIN_W // 2 - 380, WIN_H - 92, 760, 40), 170, slant=14)
            self.txt("ENTER  otra carrera      ESC  pausa / menú", 20, (210, 220, 230), (WIN_W // 2, WIN_H - 72),
                     "center", italic=False)
            return
        t = self.tour
        res = self.tour_score()
        x = 40
        self.panel((x, 90, 580, 80 + 52 * len(res)), 215, accent=ACCENT, slant=26)
        self.txt(f"CARRERA {t['idx'] + 1} DE {len(t['maps'])}", 34, (255, 255, 255), (x + 40, 104), glow=ACCENT)
        for i, (name, pts) in enumerate(res):
            y = 160 + i * 52
            car = next((c for c in order if c.name == name), None)
            col = (255, 215, 60) if name == t["me"] else (235, 240, 245)
            self.txt(f"{i + 1}", 28, col, (x + 44, y))
            self.txt(name, 24, col, (x + 90, y + 2))
            tm = fmt_time(car.finish_time) if car is not None and car.finished else "EN PISTA"
            self.txt(tm, 20, (190, 200, 215), (x + 400, y + 6), "topright", italic=False)
            self.txt(f"+{pts}", 24, ORANGE, (x + 540, y + 2), "topright")
        tab = sorted(t["points"].items(), key=lambda kv: -kv[1])
        self.txt("CAMPEONATO", 30, (255, 255, 255), (WIN_W - 520, 96), glow=ORANGE)
        self.standings_panel(WIN_W - 560, 140, 520, tab)
        last = t["idx"] + 1 >= len(t["maps"])
        nxt = "ENTER  ver al campeón" if last else "ENTER  siguiente carrera"
        self.panel((WIN_W // 2 - 300, WIN_H - 92, 600, 40), 170, slant=14)
        self.txt(nxt + "      ESC  pausa", 20, (210, 220, 230), (WIN_W // 2, WIN_H - 72), "center", italic=False)

    # ------------------------------------------------------------ pantalla de seleccion de piloto
    def hud_select_menu(self, map_panel):
        p = PILOTS[self.sel]
        ch = getattr(self, "chase", None)
        title = self.tour["name"] if self.tour else ("MODO PERSECUCIÓN" if ch else "CARRERA RÁPIDA")
        self.txt(title, 22, ORANGE, (62, 24))
        head = ("ELIGE AL FUGITIVO" if ch["role"] == "policia" else "ELIGE TU AUTO") if ch else "ELIGE TU PILOTO"
        self.txt(head, 50, (255, 255, 255), (60, 46), glow=ACCENT)
        self.panel((40, 118, 520, 330), 175, accent=tuple(p["map"]), slant=24)
        self.txt(p["name"], 52 if font(52).size(p["name"])[0] < 460 else 40, (255, 255, 255), (80, 128))
        self.txt(p["brand"].upper(), 24, tuple(p["map"]) if p["map"] != (255, 255, 255) else (210, 220, 235), (84, 196))
        self.txt(p["vname"], 20, (190, 200, 215), (86, 232), italic=False)
        stats = [("VELOCIDAD", p["vmax"], 84, 97), ("ACELERACIÓN", p["accel"], 10.0, 13.0),
                 ("AGARRE", p["grip"], 24.5, 29.0), ("PESO", p["mass"], 0.7, 1.7)]
        for i, (lab, v, lo, hi) in enumerate(stats):
            y = 282 + i * 38
            self.txt(lab, 17, (200, 210, 225), (84, y), italic=False)
            fill = clamp((v - lo) / (hi - lo), 0.1, 1.0)
            for k in range(10):
                on = k / 10 < fill
                x = 250 + k * 28
                pygame.draw.polygon(self.hud, (ACCENT if on else (45, 52, 66)) + (255,),
                                    [(x + 6, y), (x + 26, y), (x + 20, y + 18), (x, y + 18)])
        dname, _, _, _, ddesc, _ = DIFFS[self.diff]
        dcol = [(90, 230, 120), (255, 215, 60), (255, 140, 30), (255, 60, 60)][self.diff]
        self.panel((40, 466, 520, 112), 175, accent=dcol, slant=24)
        self.txt("DIFICULTAD  ↑ ↓", 16, (190, 200, 215), (84, 476), italic=False, fname="segoeui,arial")
        self.txt(f"‹ {dname} ›", 34, dcol, (84, 496))
        self.txt(ddesc, 16, (190, 200, 215), (84, 540), italic=False)
        self.txt("HORARIO  N", 16, (190, 200, 215), (350, 476), italic=False)
        self.txt(TOD_HD[self.tod][0], 28, (255, 255, 255), (350, 498))
        # tira de pilotos (desplazable)
        n = len(PILOTS)
        show = min(n, 9)
        cw = 132
        x0 = WIN_W // 2 - show * cw // 2
        for j in range(show):
            i = (self.sel - show // 2 + j) % n
            q = PILOTS[i]
            r = (x0 + j * cw + 4, WIN_H - 96, cw - 8, 50)
            sel = i == self.sel
            self.panel(r, 215 if sel else 110, accent=tuple(q["map"]), slant=12)
            nm = q["name"] if font(14).size(q["name"])[0] < cw - 24 else q["name"][:12] + "…"
            self.txt(nm, 14, (255, 255, 255) if sel else (160, 170, 185), (r[0] + r[2] // 2 + 3, r[1] + 25), "center")
        self.txt(f"{self.sel + 1}/{n}", 16, (170, 185, 200), (WIN_W // 2, WIN_H - 112), "center", italic=False)
        if self.tour:
            import gtc_racing_hd_mapas as M
            t = self.tour
            self.panel((WIN_W - 580, 36, 520, 60 + 34 * len(t["maps"])), 185, accent=ORANGE, slant=20)
            self.txt("PISTAS DEL TORNEO", 16, (170, 185, 200), (WIN_W - 540, 46), italic=False)
            for j, i in enumerate(t["maps"]):
                m = M.MAPS[i]
                self.txt(f"{j + 1}. {m.name}", 20, (255, 255, 255), (WIN_W - 540, 72 + j * 34))
                self.txt(f"{(m.length or 0) * m.laps / 1000:.1f} km", 16, (190, 200, 215), (WIN_W - 80, 76 + j * 34),
                         "topright", italic=False)
            hint = "← →  piloto     ↑ ↓  dificultad     N  horario     ENTER  empezar torneo     ESC  volver"
        else:
            map_panel(WIN_W - 580, 36)
            hint = "← →  piloto     ↑ ↓  dificultad     Q / E  pista     N  horario     ENTER  correr     ESC  volver"
        if ch:
            import gtc_racing_hd_mapas as M
            from gtc_chase import chase_time, CHASE_DIFF
            t_ = chase_time(M.MAPS[self.map_sel])
            n_ = CHASE_DIFF[self.diff][0]
            fug = ch["role"] == "fugitivo"
            self.panel((WIN_W - 580, 160, 520, 96), 185, accent=(255, 60, 60) if fug else (60, 120, 255), slant=20)
            self.txt("ERES EL FUGITIVO" if fug else "ERES LA POLICÍA", 26, (255, 255, 255), (WIN_W - 540, 172))
            self.txt((f"Escapa {fmt_time(t_)} de {n_} patrullas" if fug else
                      f"Detén al fugitivo en {fmt_time(t_)} (con 1 patrulla de apoyo)"), 18, (200, 210, 225),
                     (WIN_W - 538, 212), italic=False)
            hint = hint.replace("ENTER  correr", "ENTER  empezar")
        self.txt(hint, 17, (200, 210, 225), (WIN_W // 2, WIN_H - 24), "center", italic=False, fname="segoeui,arial")

    # ------------------------------------------------------------ ciclo principal (con limite de FPS configurable)
    def run(self):
        import sys
        while True:
            dt = min(self.clock.tick(self.fps_cap) / 1000.0, 0.05)
            self.fps = self.fps * 0.95 + (1.0 / max(dt, 1e-3)) * 0.05
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and not self.handle_key(e.key)):
                    self.save_cfg()
                    pygame.quit()
                    sys.exit()
            self.update(dt)
            self.render()
            pygame.display.flip()

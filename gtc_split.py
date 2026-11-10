"""
GTC Racing HD - CONTROLES (gamepad) y PANTALLA DIVIDIDA (1 a 4 jugadores en la misma pantalla).

Controles (Xbox One, GameSir en modo XInput y cualquier control que SDL reconozca):
    RT acelerar · LT frenar / reversa · stick izquierdo o cruceta: direccion · A o RB: nitro · B: freno de mano
    Y: camara · START: pausa · BACK/VIEW: siguiente cancion.  En los menus: cruceta/stick, A aceptar, B volver,
    LB/RB cambiar pista o pestana, Y horario.
Pantalla dividida: menu principal -> PANTALLA DIVIDIDA. Cada jugador se une con A (control) o ENTER (teclado).
"""
import math
import random

import moderngl
import numpy as np
import pygame

import gtc_racing_hd as hd
from gtc_racing_hd import (PILOTS, DIFFS, TOD_HD, QUALITY, ACCENT, ORANGE, WIN_W, WIN_H, font, clamp, fmt_time, setu)

try:
    import pygame._sdl2.controller as sdlc
except Exception:                                  # pragma: no cover
    sdlc = None

DEAD = 0.14
PLAYER_COLS = [(255, 215, 60), (40, 200, 255), (255, 80, 200), (90, 230, 110)]
SPLIT_Q = {2: dict(QUALITY[1][1], shadow=0, refl=0), 3: dict(QUALITY[0][1], scale=0.85),
           4: dict(QUALITY[0][1], scale=0.85)}


# ---------------------------------------------------------------- controles
class Pads:
    """Controles conectados (con conexion y desconexion en caliente)."""

    def __init__(self):
        self.ctrls = {}                       # instance_id -> objeto (Controller o Joystick)
        self.kind = {}                        # instance_id -> "ctrl" / "joy"
        self.acc = {}                         # acumuladores para gatillo analogico -> pulsos
        self.stick_dir = {}                   # repeticion del stick en menus
        self.ok = False
        try:
            if sdlc is not None:
                sdlc.init()
            pygame.joystick.init()
            self.ok = True
            for i in range(pygame.joystick.get_count()):
                self.add(i)
        except Exception as e:
            print("(aviso) controles no disponibles:", e)

    def add(self, index):
        try:
            if sdlc is not None and sdlc.is_controller(index):
                c = sdlc.Controller(index)
                iid = c.as_joystick().get_instance_id()
                self.ctrls[iid], self.kind[iid] = c, "ctrl"
            else:
                j = pygame.joystick.Joystick(index)
                j.init()
                iid = j.get_instance_id()
                if iid in self.ctrls:
                    return
                self.ctrls[iid], self.kind[iid] = j, "joy"
        except Exception as e:
            print("(aviso) no se pudo abrir el control:", e)

    def remove(self, iid):
        self.ctrls.pop(iid, None)
        self.kind.pop(iid, None)

    def ids(self):
        return sorted(self.ctrls)

    def name(self, iid):
        c = self.ctrls.get(iid)
        try:
            n = c.name if self.kind.get(iid) == "ctrl" else c.get_name()
        except Exception:
            n = "Control"
        return (n or "Control")[:22]

    # lectura cruda normalizada
    def state(self, iid):
        c = self.ctrls.get(iid)
        if c is None:
            return None
        try:
            if self.kind[iid] == "ctrl":
                ax = lambda a: c.get_axis(a) / 32767.0
                b = c.get_button
                s = dict(lx=ax(pygame.CONTROLLER_AXIS_LEFTX), ly=ax(pygame.CONTROLLER_AXIS_LEFTY),
                         lt=max(0.0, ax(pygame.CONTROLLER_AXIS_TRIGGERLEFT)),
                         rt=max(0.0, ax(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)),
                         a=b(pygame.CONTROLLER_BUTTON_A), b=b(pygame.CONTROLLER_BUTTON_B),
                         x=b(pygame.CONTROLLER_BUTTON_X), y=b(pygame.CONTROLLER_BUTTON_Y),
                         lb=b(pygame.CONTROLLER_BUTTON_LEFTSHOULDER), rb=b(pygame.CONTROLLER_BUTTON_RIGHTSHOULDER),
                         left=b(pygame.CONTROLLER_BUTTON_DPAD_LEFT), right=b(pygame.CONTROLLER_BUTTON_DPAD_RIGHT),
                         up=b(pygame.CONTROLLER_BUTTON_DPAD_UP), down=b(pygame.CONTROLLER_BUTTON_DPAD_DOWN))
            else:                                                     # joystick generico (distribucion XInput)
                na, nb = c.get_numaxes(), c.get_numbuttons()
                ax = lambda a: c.get_axis(a) if a < na else 0.0
                bt = lambda k: bool(c.get_button(k)) if k < nb else False
                hat = c.get_hat(0) if c.get_numhats() else (0, 0)
                trig = lambda a: (ax(a) + 1) / 2 if a < na else 0.0
                s = dict(lx=ax(0), ly=ax(1), lt=max(trig(4), 1.0 if bt(6) and nb > 10 else 0.0),
                         rt=max(trig(5), 1.0 if bt(7) and nb > 10 else 0.0),
                         a=bt(0), b=bt(1), x=bt(2), y=bt(3), lb=bt(4), rb=bt(5),
                         left=hat[0] < 0, right=hat[0] > 0, up=hat[1] > 0, down=hat[1] < 0)
            return s
        except Exception:
            return None

    def drive(self, iid, dt=1 / 60):
        """(acelerar, frenar, direccion -1..1, freno de mano, nitro) de un control."""
        s = self.state(iid)
        if s is None:
            return False, False, 0.0, False, False
        x = s["lx"]
        st = 0.0 if abs(x) < DEAD else math.copysign(((abs(x) - DEAD) / (1 - DEAD)) ** 1.4, x)
        if s["left"]:
            st = -1.0
        elif s["right"]:
            st = 1.0
        acc = self.acc.setdefault(iid, [0.0, 0.0])

        def pulse(k, v):                     # gatillo analogico -> acelerador a pulsos (como medio pedal)
            if v > 0.92:
                return True
            if v < 0.08:
                acc[k] = 0.0
                return False
            acc[k] += v
            if acc[k] >= 1.0:
                acc[k] -= 1.0
                return True
            return False
        thr = pulse(0, s["rt"])
        brk = pulse(1, s["lt"])
        return thr, brk, st, bool(s["b"]), bool(s["a"] or s["rb"])

    def rumble(self, iid, low, high, ms):
        c = self.ctrls.get(iid)
        if c is None:
            return
        try:
            c.rumble(clamp(low, 0, 1), clamp(high, 0, 1), int(ms))
        except Exception:
            pass

    def event_action(self, e):
        """Evento de control -> (instance_id, accion) o None."""
        if e.type == pygame.CONTROLLERBUTTONDOWN:
            m = {pygame.CONTROLLER_BUTTON_A: "a", pygame.CONTROLLER_BUTTON_B: "b", pygame.CONTROLLER_BUTTON_X: "x",
                 pygame.CONTROLLER_BUTTON_Y: "y", pygame.CONTROLLER_BUTTON_START: "start",
                 pygame.CONTROLLER_BUTTON_BACK: "back", pygame.CONTROLLER_BUTTON_LEFTSHOULDER: "lb",
                 pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: "rb", pygame.CONTROLLER_BUTTON_DPAD_UP: "up",
                 pygame.CONTROLLER_BUTTON_DPAD_DOWN: "down", pygame.CONTROLLER_BUTTON_DPAD_LEFT: "left",
                 pygame.CONTROLLER_BUTTON_DPAD_RIGHT: "right"}
            a = m.get(e.button)
            return (e.instance_id, a) if a else None
        if e.type == pygame.JOYBUTTONDOWN and self.kind.get(e.instance_id) == "joy":
            m = {0: "a", 1: "b", 2: "x", 3: "y", 4: "lb", 5: "rb", 6: "back", 7: "start"}
            a = m.get(e.button)
            return (e.instance_id, a) if a else None
        if e.type == pygame.JOYHATMOTION and self.kind.get(e.instance_id) == "joy":
            x, y = e.value
            a = "left" if x < 0 else "right" if x > 0 else "up" if y > 0 else "down" if y < 0 else None
            return (e.instance_id, a) if a else None
        if e.type == pygame.CONTROLLERAXISMOTION and e.axis in (pygame.CONTROLLER_AXIS_LEFTX,
                                                                 pygame.CONTROLLER_AXIS_LEFTY):
            v = e.value / 32767.0
            key = (e.instance_id, e.axis)
            prev = self.stick_dir.get(key, 0)
            d = 0 if abs(v) < 0.6 else (1 if v > 0 else -1)
            self.stick_dir[key] = d
            if d != 0 and d != prev:                                  # el stick tambien navega los menus
                if e.axis == pygame.CONTROLLER_AXIS_LEFTX:
                    return e.instance_id, "right" if d > 0 else "left"
                return e.instance_id, "down" if d > 0 else "up"
        return None


# ---------------------------------------------------------------- pantalla dividida
def layout(n):
    """Rectangulos (x, y, w, h) en pixeles (origen arriba a la izquierda)."""
    W, H = WIN_W, WIN_H
    if n <= 1:
        return [(0, 0, W, H)]
    if n == 2:
        return [(0, 0, W, H // 2), (0, H // 2, W, H // 2)]
    return [(0, 0, W // 2, H // 2), (W // 2, 0, W // 2, H // 2), (0, H // 2, W // 2, H // 2),
            (W // 2, H // 2, W // 2, H // 2)][:n]


class ViewTargets:
    """Buferes de render del tamano de una vista (se intercambian con los del Renderer)."""
    KEYS = ("hdr", "depth", "fbo_hdr", "bloom", "streak", "blurtex", "RW", "RH", "ldr", "fbo_ldr", "W", "H", "scale")

    def __init__(self, R, w, h, scale):
        ctx = R.ctx
        self.W, self.H = w, h
        W, H = max(16, int(w * scale)), max(16, int(h * scale))
        self.RW, self.RH, self.scale = W, H, scale
        self.hdr = ctx.texture((W, H), 4, dtype="f2")
        self.hdr.repeat_x = self.hdr.repeat_y = False
        self.depth = ctx.depth_texture((W, H))
        self.depth.compare_func = ""
        self.fbo_hdr = ctx.framebuffer([self.hdr], self.depth)
        self.bloom = []
        bw, bh = W, H
        for _ in range(6):
            bw, bh = max(1, bw // 2), max(1, bh // 2)
            t = ctx.texture((bw, bh), 4, dtype="f2")
            t.repeat_x = t.repeat_y = False
            self.bloom.append((t, ctx.framebuffer([t])))
        sw, sh = self.bloom[1][0].size
        self.streak, self.blurtex = [], []
        for lst in (self.streak, self.blurtex):
            for _ in range(2):
                t = ctx.texture((sw, sh), 4, dtype="f2")
                t.repeat_x = t.repeat_y = False
                lst.append((t, ctx.framebuffer([t])))
        self.ldr = ctx.texture((w, h), 4)
        self.ldr.repeat_x = self.ldr.repeat_y = False
        self.fbo_ldr = ctx.framebuffer([self.ldr])

    def swap_in(self, R):
        saved = {k: getattr(R, k) for k in self.KEYS}
        for k in self.KEYS:
            setattr(R, k, getattr(self, k))
        return saved

    @staticmethod
    def restore(R, saved):
        for k, v in saved.items():
            setattr(R, k, v)

    def release(self):
        for t, f in self.bloom + self.streak + self.blurtex:
            f.release()
            t.release()
        for o in (self.fbo_hdr, self.hdr, self.depth, self.fbo_ldr, self.ldr):
            o.release()


CAM_STATE = ("cam_pos", "cam_yaw", "fov_b", "_cam_y", "shake", "focus")


class SplitMixin:
    """Controles y pantalla dividida. Va primero en la herencia de GameMaps."""
    local_active = False                  # (valores por omision: el juego base arranca una carrera antes del init)
    lcars = ()
    lslots = ()

    # ------------------------------------------------------------ estado
    def split_init(self):
        self.pads = Pads()
        self.local = None                 # sala de pantalla dividida (en el menu)
        self.local_active = False         # carrera en pantalla dividida
        self.lslots = []                  # [{dev, pilot, ready}]
        self.lcars = []                   # coches de los jugadores locales (orden J1..J4)
        self._targets = {}
        self._input_slot = 0
        _install_ai_hook(self)

    # ------------------------------------------------------------ entrada: teclado + control
    def slot_device(self, k):
        if self.local_active and k < len(self.lslots):
            return self.lslots[k]["dev"]
        return None                                                    # un jugador: teclado y cualquier control

    def player_inputs(self):
        dev = self.slot_device(getattr(self, "_input_slot", 0))
        kb = super().player_inputs() if dev in (None, "kb") else (False, False, 0, False, False)
        pads = self.pads.ids() if dev is None else ([dev] if dev != "kb" else [])
        thr, brk, st, hb, nitro = kb
        st = float(st)
        for iid in pads:
            t2, b2, s2, h2, n2 = self.pads.drive(iid)
            thr, brk, hb, nitro = thr or t2, brk or b2, hb or h2, nitro or n2
            if st == 0 and s2 != 0:
                st = s2
        return thr, brk, st, hb, nitro

    def local_inputs(self, car):
        """Entradas del jugador local k (J2..J4): mismo filtro de volante que J1."""
        k = car.local
        saved = (self.player, getattr(self, "_st_in", 0.0))
        self.player, self._st_in, self._input_slot = car, getattr(car, "_st_in", 0.0) or 0.0, k
        try:
            out = self.player_inputs()
        finally:
            car._st_in = self._st_in
            self.player, self._st_in = saved
            self._input_slot = 0
        return out

    def pad_event(self, e):
        """Eventos de control: conexion y botones (en menus se traducen a teclas). Devuelve False para salir."""
        if not self.pads.ok:
            return True
        if e.type in (pygame.CONTROLLERDEVICEADDED, pygame.JOYDEVICEADDED):
            if e.type == pygame.CONTROLLERDEVICEADDED or not (sdlc and sdlc.is_controller(e.device_index)):
                before = set(self.pads.ids())
                self.pads.add(e.device_index)
                new = set(self.pads.ids()) - before
                for iid in new:
                    self.toast, self.toast_t = f"CONTROL CONECTADO: {self.pads.name(iid).upper()}", 2.5
            return True
        if e.type in (pygame.CONTROLLERDEVICEREMOVED, pygame.JOYDEVICEREMOVED):
            iid = getattr(e, "instance_id", None)
            if iid in self.pads.ctrls:
                self.pads.remove(iid)
                self.toast, self.toast_t = "CONTROL DESCONECTADO", 2.5
                if self.local_active and any(s["dev"] == iid for s in self.lslots) and self.state == "race":
                    self.paused, self.psel = True, 0
            return True
        act = self.pads.event_action(e)
        if act is None:
            return True
        iid, a = act
        if self.state == "menu" and self.page == "local" and not self.cfg_open:
            return self.local_action(iid, a)
        racing = self.state in ("intro", "countdown", "race", "finished") and not self.paused and not self.cfg_open
        K = pygame
        if racing:
            if a == "start":
                return self.handle_key(K.K_ESCAPE)
            if a == "y":
                ks = self.keys_of("camara")
                return self.handle_key(ks[0]) if ks else True
            if a == "back":
                if self.music.mode == "playlist":
                    self.music.next_song()
                return True
            if a == "a" and self.state in ("intro", "finished"):
                return self.handle_key(K.K_RETURN)
            return True
        key = {"up": K.K_UP, "down": K.K_DOWN, "left": K.K_LEFT, "right": K.K_RIGHT, "a": K.K_RETURN,
               "start": K.K_RETURN, "b": K.K_ESCAPE, "back": K.K_ESCAPE, "lb": K.K_q, "rb": K.K_e,
               "y": K.K_n, "x": K.K_BACKSPACE}.get(a)
        if self.paused and a == "start":
            key = K.K_ESCAPE
        return self.handle_key(key) if key else True

    # ------------------------------------------------------------ sala de pantalla dividida
    def open_local(self):
        self.page, self.local = "local", dict(t=0.0)
        self.lslots = []
        self.tour, self.chase = None, None

    def local_free_pilot(self, start, step, skip=None):
        taken = {s["pilot"] for s in self.lslots if s is not skip}
        n = len(PILOTS)
        i = start
        for _ in range(n):
            i = (i + step) % n
            if i not in taken:
                return i
        return start

    def local_action(self, dev, a):
        """Una accion de un dispositivo en la sala ('kb' o el id de un control)."""
        slot = next((s for s in self.lslots if s["dev"] == dev), None)
        if a in ("a", "start"):
            if slot is None:
                if len(self.lslots) < 4:
                    p0 = self.local_free_pilot(len(PILOTS) - 2 - len(self.lslots) * 3, 1)
                    self.lslots.append(dict(dev=dev, pilot=p0, ready=False))
                    if dev != "kb":
                        self.pads.rumble(dev, 0.4, 0.4, 150)
            else:
                slot["ready"] = not slot["ready"]
        elif a in ("b", "back"):
            if slot is not None and slot["ready"]:
                slot["ready"] = False
            elif slot is not None:
                self.lslots.remove(slot)
            elif not self.lslots:
                self.page, self.msel = "main", 0
        elif a in ("left", "right") and slot is not None and not slot["ready"]:
            slot["pilot"] = self.local_free_pilot(slot["pilot"], -1 if a == "left" else 1, skip=slot)
        elif a in ("up", "down"):
            self.diff = int(clamp(self.diff + (1 if a == "up" else -1), 0, len(DIFFS) - 1))
        elif a in ("lb", "rb"):
            import gtc_racing_hd_mapas as M
            self.map_sel = (self.map_sel + (1 if a == "rb" else -1)) % len(M.MAPS)
        elif a == "y":
            self.tod = (self.tod + 1) % len(TOD_HD)
        if self.lslots and all(s["ready"] for s in self.lslots):
            self.start_local()
        return True

    def key_menu(self, key):
        if self.page != "local":
            return super().key_menu(key)
        K = pygame
        a = {K.K_RETURN: "a", K.K_KP_ENTER: "a", K.K_ESCAPE: "b", K.K_BACKSPACE: "b", K.K_LEFT: "left",
             K.K_a: "left", K.K_RIGHT: "right", K.K_d: "right", K.K_UP: "up", K.K_w: "up", K.K_DOWN: "down",
             K.K_s: "down", K.K_q: "lb", K.K_e: "rb", K.K_n: "y"}.get(key)
        return self.local_action("kb", a) if a else True

    def handle_key(self, key):
        if key == pygame.K_ESCAPE and self.state == "menu" and self.page == "local" and not self.cfg_open:
            return self.local_action("kb", "b")
        return super().handle_key(key)

    def start_local(self):
        """Todos listos: se carga la pista y arranca la carrera en pantalla dividida."""
        if self.map_sel != self.map_i:
            self.load_map(self.map_sel)
        self.local_active = True
        self.page, self.msel = "main", 0
        n = len(self.lslots)
        if n > 1:
            self._q_saved = self.qi
            self.R.set_quality(SPLIT_Q[n])
        self.new_race()

    def local_end(self):
        if not self.local_active:
            return
        self.local_active = False
        self.lcars = []
        if hasattr(self, "_q_saved"):
            self.R.set_quality(QUALITY[self._q_saved][1])
            del self._q_saved

    # ------------------------------------------------------------ carrera
    def new_race(self, attract=False, intro=True):
        if attract or not self.local_active:
            if attract:
                self.local_end()
            return super().new_race(attract=attract, intro=intro)
        full = list(PILOTS)
        players = [full[s["pilot"]] for s in self.lslots]
        others = [p for p in full if p not in players]
        ai = random.sample(others, max(0, 6 - len(players)))
        sel0 = self.sel
        PILOTS[:] = ai + players[1:] + players[:1]
        self.sel = len(PILOTS) - 1
        hd.LAPS = self.md.laps
        try:
            hd.GameHD.new_race(self, attract=False, intro=intro)
        finally:
            PILOTS[:] = full
            self.sel = sel0
        self.paused = False
        by_name = {c.name: c for c in self.cars}
        self.lcars = []
        for k, p in enumerate(players):
            c = by_name[p["name"]]
            c.local = k
            if k > 0:                                       # J2..J4: con todo su auto (no como la IA)
                c.owner = "p"
                c.vmax, c.accel = p["vmax"], p["accel"]
                c.nitro_ai = False
            c.cam = dict(cam_pos=None, cam_yaw=c.ang, fov_b=0.0, _cam_y=None, shake=0.0, focus=10.0)
            c._st_in = 0.0
            c._vprev = (0.0, 0.0)
            self.lcars.append(c)
        return None

    def update(self, rdt):
        super().update(rdt)
        if not self.local_active or not self.lcars or self.frozen:
            self.rumble_single(rdt)
            return
        # vibracion por choques de cada jugador
        for c in self.lcars:
            dv = math.hypot(c.vx - c._vprev[0], c.vz - c._vprev[1])
            c._vprev = (c.vx, c.vz)
            dev = self.lslots[c.local]["dev"]
            if dev != "kb" and dv > 3.0 and self.state == "race":
                self.pads.rumble(dev, min(1.0, dv / 10), min(1.0, dv / 8), 180)
        # la carrera termina cuando terminan TODOS los jugadores
        if self.state == "finished" and not all(c.finished for c in self.lcars):
            self.state = "race"
        elif self.state == "race" and all(c.finished for c in self.lcars):
            self.state = "finished"
            self.results_t = 0.0
        if len(self.lcars) > 1 and self.state in ("countdown", "race", "finished"):
            self.cams = [self.cam]
            p1 = self.player
            for c in self.lcars[1:]:
                saved = {k: getattr(self, k) for k in CAM_STATE if hasattr(self, k)}   # solo lo que ya existe
                for k, v in c.cam.items():
                    setattr(self, k, v)
                self.player = c
                try:
                    self.update_camera(rdt)
                    c.cam = {k: getattr(self, k, None) for k in CAM_STATE}
                    self.cams.append(self.cam)
                finally:
                    for k in CAM_STATE:
                        if k in saved:
                            setattr(self, k, saved[k])
                        elif hasattr(self, k):
                            delattr(self, k)
                    self.player = p1
            self.cam = self.cams[0]

    def rumble_single(self, rdt):
        """Un jugador: vibran todos los controles en los choques."""
        if self.state != "race" or not self.pads.ctrls or self.frozen:
            return
        p = self.player
        vp = getattr(p, "_vprev", (p.vx, p.vz))
        dv = math.hypot(p.vx - vp[0], p.vz - vp[1])
        p._vprev = (p.vx, p.vz)
        if dv > 3.0:
            for iid in self.pads.ids():
                self.pads.rumble(iid, min(1.0, dv / 10), min(1.0, dv / 8), 180)

    # ------------------------------------------------------------ render en varias vistas
    def split_on(self):
        return (self.local_active and len(self.lcars) > 1 and self.state in ("countdown", "race", "finished")
                and len(getattr(self, "cams", [])) == len(self.lcars))

    def render(self):
        if not self.split_on():
            return super().render()
        R, ctx = self.R, self.ctx
        rects = layout(len(self.lcars))
        q = R.quality or {}
        scale = q.get("scale", 1.0)
        p1, cam1 = self.player, self.cam
        orig_present, orig_hud = R.present, self.draw_hud
        R.present = lambda *a, **k: None
        self.draw_hud = lambda: None
        scr = self.screen_fbo
        scr.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        try:
            for k, c in enumerate(self.lcars):
                x, y, w, h = rects[k]
                key = (w, h, scale)
                if key not in self._targets:
                    for t in self._targets.values():
                        t.release()
                    self._targets = {key: ViewTargets(R, w, h, scale)}
                vt = self._targets[key]
                saved = vt.swap_in(R)
                hd.WIN_W, hd.WIN_H = w, h
                self.player, self.cam = c, self.cams[k]
                R.static_key = None
                try:
                    hd.GameHD.render(self)
                finally:
                    hd.WIN_W, hd.WIN_H = WIN_W, WIN_H
                    ViewTargets.restore(R, saved)
                scr.viewport = (x, WIN_H - y - h, w, h)              # copiar la vista a su parte de la pantalla
                scr.use()
                ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
                vt.ldr.use(0)
                setu(R.p_fxaa, u_src=0, u_texel=(1.0 / w, 1.0 / h), u_on=1.0)
                R.quads[R.p_fxaa].render(moderngl.TRIANGLE_STRIP)
        finally:
            R.present, self.draw_hud = orig_present, orig_hud
            self.player, self.cam = p1, cam1
            scr.viewport = (0, 0, WIN_W, WIN_H)
        # interfaz de toda la pantalla
        self.hud.fill((0, 0, 0, 0))
        self.hud_split(rects)
        scr.use()
        R.hud_tex.write(self.hud.get_buffer().raw)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        R.hud_tex.use(0)
        setu(R.p_hud, u_hud=0)
        R.quads[R.p_hud].render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

    # ------------------------------------------------------------ interfaz
    def hud_split(self, rects):
        order = self.standings()
        n = len(self.lcars)
        for k, c in enumerate(self.lcars):
            x, y, w, h = rects[k]
            col = PLAYER_COLS[k]
            pos = order.index(c) + 1
            pygame.draw.rect(self.hud, col + (255,), (x, y, w, h), 3)
            self.panel((x + 10, y + 10, 250, 84), 150, accent=col, slant=12)
            self.txt(f"J{k + 1}", 22, col, (x + 26, y + 14))
            self.txt(c.name, 15, (220, 228, 240), (x + 70, y + 19), italic=False)
            self.txt(f"{pos}", 48, (255, 255, 255), (x + 26, y + 36), glow=col if pos == 1 else None)
            self.txt(f"/{len(self.cars)}", 20, (190, 200, 210), (x + 26 + font(48).size(str(pos))[0] + 4, y + 62))
            lap = min(hd.LAPS, max(0, c.lap_idx) + 1)
            tm = self.race_time if not c.finished else c.finish_time
            self.txt(f"V {lap}/{hd.LAPS}", 20, (255, 255, 255), (x + 120, y + 42))
            self.txt(fmt_time(tm), 18, (210, 220, 232), (x + 120, y + 68), italic=False)
            # velocidad y nitro
            spd = abs(c.vf) * 3.6
            self.panel((x + w - 190, y + h - 70, 180, 60), 150, accent=col, slant=12)
            self.txt(f"{spd:.0f}", 36, (255, 255, 255), (x + w - 112, y + h - 66), "topright")
            self.txt("KM/H", 14, (190, 200, 215), (x + w - 106, y + h - 46), italic=False)
            bx = x + w - 172
            pygame.draw.rect(self.hud, (40, 46, 60, 255), (bx, y + h - 22, 150, 7))
            pygame.draw.rect(self.hud, ((200, 230, 255) if c.boosting else (90, 170, 255)) + (255,),
                             (bx, y + h - 22, int(150 * c.nitro), 7))
            if self.state == "race" and c.wrong > 1.0 and int(self.t * 3) % 2 == 0 and not c.finished:
                self.txt("¡SENTIDO CONTRARIO!", 26, (255, 70, 50), (x + w // 2, y + h // 2 - 40), "center")
            if c.finished:
                self.txt(f"¡META!  {pos}º", 40, col, (x + w // 2, y + h // 2), "center", glow=col)
        if n == 3:                                                     # cuarta parte: tabla de posiciones
            x, y, w, h = WIN_W // 2, WIN_H // 2, WIN_W // 2, WIN_H // 2
            pygame.draw.rect(self.hud, (8, 10, 16, 235), (x, y, w, h))
            self.txt("POSICIONES", 26, (255, 255, 255), (x + 30, y + 16), glow=ACCENT)
            for i, c in enumerate(order):
                k = getattr(c, "local", None)
                col = PLAYER_COLS[k] if k is not None else (200, 210, 225)
                tag = f"J{k + 1}  " if k is not None else ""
                self.txt(f"{i + 1}.  {tag}{c.name}", 20, col, (x + 34, y + 62 + i * 40))
        if self.state == "countdown":
            cn = int(self.countdown)
            if cn >= 1:
                fr = self.countdown - cn
                self.txt(str(cn), int(120 + 60 * fr), (255, 255, 255), (WIN_W // 2, WIN_H // 2), "center",
                         glow=(255, 60, 40))
        if self.msg_t > 0 and self.message and self.state == "countdown":
            self.txt(self.message, 60, (255, 255, 255), (WIN_W // 2, WIN_H // 2), "center", glow=ACCENT)
        if self.state == "finished" and self.results_t > 2.0:
            self.results(order)
        if self.toast_t > 0:
            a = int(255 * min(1.0, self.toast_t * 2))
            self.txt(self.toast, 22, (255, 255, 255), (WIN_W // 2, WIN_H // 2 - 20), "center", alpha=a)
        if self.paused and not self.cfg_open:
            self.hud_pause()
        if self.cfg_open:
            self.hud_config()

    def hud_main_menu(self):
        if self.page != "local":
            return super().hud_main_menu()
        self.dim(140)
        self.txt("PANTALLA DIVIDIDA", 64, (255, 255, 255), (WIN_W // 2, 50), "center", glow=ACCENT)
        self.txt(f"{len(self.pads.ctrls)} control(es) conectado(s)  ·  hasta 4 jugadores", 18, (190, 205, 220),
                 (WIN_W // 2, 98), "center", italic=False)
        sw = 290
        x0 = WIN_W // 2 - sw * 2
        for k in range(4):
            x = x0 + k * sw + 8
            col = PLAYER_COLS[k]
            s = self.lslots[k] if k < len(self.lslots) else None
            self.panel((x, 130, sw - 16, 300), 200 if s else 110, accent=col, slant=16)
            self.txt(f"J{k + 1}", 40, col, (x + 30, 140), glow=col if s else None)
            if s is None:
                blink = int(self.t * 2) % 2 == 0
                self.txt("PRESIONA A", 22, (255, 255, 255) if blink else (150, 160, 175), (x + sw // 2 - 8, 260),
                         "center")
                self.txt("(o ENTER en el teclado)", 15, (160, 170, 185), (x + sw // 2 - 8, 292), "center",
                         italic=False)
                continue
            dev = "TECLADO" if s["dev"] == "kb" else self.pads.name(s["dev"]).upper()
            self.txt(dev, 14, (170, 185, 200), (x + 30, 192), italic=False)
            p = PILOTS[s["pilot"]]
            nm = p["name"]
            self.txt(f"‹ {nm} ›" if not s["ready"] else nm, 22 if font(22).size(nm)[0] < 200 else 17,
                     (255, 255, 255), (x + 30, 226))
            self.txt(p["vname"][:28], 14, (190, 200, 215), (x + 30, 258), italic=False)
            stats = [("VEL", p["vmax"], 84, 97), ("ACEL", p["accel"], 10, 13), ("AGARRE", p["grip"], 24.5, 29)]
            for i, (lab, v, lo, hi) in enumerate(stats):
                yy = 290 + i * 26
                self.txt(lab, 13, (180, 190, 205), (x + 30, yy), italic=False)
                f = clamp((v - lo) / (hi - lo), 0.1, 1)
                pygame.draw.rect(self.hud, (45, 52, 66, 255), (x + 110, yy + 4, 130, 9))
                pygame.draw.rect(self.hud, col + (255,), (x + 110, yy + 4, int(130 * f), 9))
            if s["ready"]:
                self.panel((x + 30, 380, sw - 80, 38), 230, col=(30, 120, 50), accent=(90, 230, 110), slant=10)
                self.txt("¡LISTO!", 24, (255, 255, 255), (x + sw // 2 - 12, 399), "center")
            else:
                self.txt("A: listo   B: salir", 15, (190, 200, 215), (x + 30, 392), italic=False)
        self.map_panel(WIN_W // 2 - 560, 450)
        dname = DIFFS[self.diff][0]
        self.panel((WIN_W // 2 + 0, 450, 560, 112), 185, accent=ORANGE, slant=20)
        self.txt("DIFICULTAD DE LA IA", 16, (170, 185, 200), (WIN_W // 2 + 40, 460), italic=False)
        self.txt(f"‹ {dname} ›", 30, (255, 255, 255), (WIN_W // 2 + 40, 482))
        self.txt("HORARIO", 16, (170, 185, 200), (WIN_W // 2 + 330, 460), italic=False)
        self.txt(TOD_HD[self.tod][0], 28, (255, 255, 255), (WIN_W // 2 + 330, 484))
        rest = max(0, 6 - max(1, len(self.lslots)))
        self.txt(f"+ {rest} rivales de la computadora", 15, (190, 200, 215), (WIN_W // 2 + 42, 530), italic=False)
        self.panel((0, WIN_H - 64, WIN_W, 64), 150, slant=0)
        self.txt("A / ENTER  unirse y listo     ← →  piloto     ↑ ↓  dificultad     LB RB / Q E  pista     "
                 "Y / N  horario     B / ESC  salir", 17, (200, 210, 225), (WIN_W // 2, WIN_H - 32), "center",
                 italic=False, fname="segoeui,arial")

    def run(self):
        import sys
        while True:
            dt = min(self.clock.tick(self.fps_cap) / 1000.0, 0.05)
            self.fps = self.fps * 0.95 + (1.0 / max(dt, 1e-3)) * 0.05
            for e in pygame.event.get():
                quit_ = e.type == pygame.QUIT
                if e.type == pygame.KEYDOWN:
                    quit_ = not self.handle_key(e.key)
                elif e.type >= pygame.JOYAXISMOTION and e.type not in (pygame.KEYDOWN, pygame.KEYUP):
                    quit_ = not self.pad_event(e)
                if quit_:
                    self.save_cfg()
                    pygame.quit()
                    sys.exit()
            self.update(dt)
            self.render()
            pygame.display.flip()


def _install_ai_hook(game):
    """Los jugadores 2..4 manejan su coche: el bucle del juego base les pide entradas a 'ai_inputs'."""
    import weakref
    ref = weakref.ref(game)
    orig = getattr(hd, "_ai_inputs_orig", hd.ai_inputs)
    hd._ai_inputs_orig = orig

    def ai_or_local(car, track, t, dt):
        g = ref()
        k = getattr(car, "local", None)
        if g is not None and k and g.local_active and g.state in ("race", "finished") and not car.finished:
            return g.local_inputs(car)
        return orig(car, track, t, dt)
    hd.ai_inputs = ai_or_local

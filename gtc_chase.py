"""
GTC Racing HD - MODO PERSECUCION.

  - FUGITIVO: escapa de las patrullas durante un tiempo que depende de la pista.
      Pierdes si te DERRIBAN (cada golpe fuerte de una patrulla hace dano; al 100 % se acabo)
      o si te ATRAPAN (una patrulla pegada a ti mientras vas casi parado).
  - POLICIA: manejas la patrulla (con un companero) y tienes que detener al fugitivo antes de que se acabe el tiempo:
      derribalo a golpes o cierrale el paso hasta que se detenga.
"""
import math
import random

import numpy as np
import pygame

import gtc_racing_hd as hd
from gtc_racing_hd import (PILOTS, DIFFS, VEH, ACCENT, ORANGE, WIN_W, WIN_H, font, clamp, fmt_time, Geo, mat, box,
                           M_EMIS, M_METAL)

# patrullas por dificultad: (numero de patrullas, velocidad de la policia, velocidad del fugitivo IA)
CHASE_DIFF = [(2, 1.00, 0.96), (3, 1.04, 1.00), (3, 1.08, 1.03), (4, 1.12, 1.06)]
ARREST_D = 8.0           # distancia para arrestar (m)
ARREST_V = 7.0           # el perseguido va a menos de ~25 km/h
ARREST_T = 2.2           # segundos pegado para el arresto
POLICE_COLORS = [((245, 245, 248), (20, 60, 170)), ((20, 20, 26), (245, 245, 248)), ((245, 245, 248), (20, 60, 170)),
                 ((20, 40, 110), (245, 245, 248))]


def chase_time(md):
    """Segundos que hay que escapar en esta pista (mas larga = mas tiempo)."""
    km = (getattr(md, "length", 0) or 4000) / 1000.0
    return int(clamp(60 + km * 15, 90, 300) // 5 * 5)


def police_pilot(k, kind):
    body, acc = POLICE_COLORS[k % len(POLICE_COLORS)]
    return {"name": f"PATRULLA {k + 1}", "brand": "Policía", "vehicle": kind, "vname": "Patrulla",
            "body": body, "accent": acc, "map": (60, 120, 255), "sign": 1, "driver": "helmet",
            "vmax": 93.0, "accel": 11.8, "grip": 27.0, "mass": 1.35, "police": True}


class ChaseMixin:
    """Modo persecucion. Va antes de MenuMixin en la herencia de GameMaps."""

    # ------------------------------------------------------------ vehiculos de policia
    def police_kind(self):
        for k in ("patrulla", "charger", "tsuru", "bmw", "sport"):
            if k in VEH:
                return k
        return "sport"

    def ensure_vehicle(self, p):
        """Construye (una vez) el modelo de un piloto que no estaba en la lista original, con torreta si es patrulla."""
        if p["name"] in self.vehicles:
            return
        body, glass, wheel, vinfo = hd.build_vehicle_hd(p)
        if p.get("police"):
            A = body.array().reshape(-1, 15) if hasattr(body, "array") else None
            top = 1.4
            if A is not None and len(A):
                m = np.abs(A[:, 0]) < 0.9
                if m.any():
                    top = float(A[m, 1].max())
            # torreta: roja y azul, con base de metal
            box(body, -0.25, top, -0.62, 0.25, top + 0.08, 0.62, (0.12, 0.12, 0.14), mat(M_METAL, 0.4, 0.6))
            box(body, -0.2, top + 0.08, -0.6, 0.2, top + 0.24, -0.04, (1.0, 0.08, 0.06), mat(M_EMIS, 0.3, 0, 6.0))
            box(body, -0.2, top + 0.08, 0.04, 0.2, top + 0.24, 0.6, (0.1, 0.3, 1.0), mat(M_EMIS, 0.3, 0, 6.0))
            vinfo = dict(vinfo, lightbar=top + 0.2)
        standing = Geo()
        hd.character(standing, p["driver"], 0, 0, 0, standing=True)
        R = self.R
        self.vehicles[p["name"]] = dict(body=R.vao_scene(body.array()), glass=R.vao_scene(glass.array())[0],
                                        wheel=R.vao_scene(wheel.array()), info=vinfo,
                                        figure=R.vao_scene(standing.array())[0], pilot=p)

    # ------------------------------------------------------------ arranque
    def start_chase_menu(self, role):
        self.tour = None
        self.chase = dict(role=role)
        self.go_select()

    def new_race(self, attract=False, intro=True):
        ch = getattr(self, "chase", None)
        if attract or not ch:
            if ch is not None and attract:
                self.chase_end_cleanup()
            hd.LAPS = self.md.laps
            return super().new_race(attract=attract, intro=intro)
        n_pol, f_pol, f_fug = CHASE_DIFF[self.diff]
        kind = self.police_kind()
        me = PILOTS[self.sel % len(PILOTS)]
        if ch["role"] == "fugitivo":
            cops = [police_pilot(k, kind) for k in range(n_pol)]
            roster = cops + [me]
        else:
            cop_me = dict(police_pilot(0, kind), name="TÚ · PATRULLA", map=(255, 215, 60))
            partner = police_pilot(1, kind)
            roster = [me, partner, cop_me]                     # el elegido es el fugitivo
        for p in roster:
            self.ensure_vehicle(p)
        full, sel0 = list(PILOTS), self.sel
        PILOTS[:] = roster
        self.sel = len(PILOTS) - 1
        hd.LAPS = 99999
        try:
            hd.GameHD.new_race(self, attract=False, intro=False)
        finally:
            PILOTS[:] = full
            self.sel = sel0
        self.paused = False
        ch.update(time=chase_time(self.md), busted=0.0, damage=0.0, over=None, hit_cd=0.0,
                  vprev={}, f_pol=f_pol, f_fug=f_fug, siren_on=False)
        return None

    def chase_place(self):
        """Acomoda los coches: el fugitivo adelante y las patrullas atras."""
        ch = self.chase
        tr = self.track
        L = tr.L
        _, skill, corner, _, _, brake_f = DIFFS[self.diff]
        cops = [c for c in self.cars if c.pilot.get("police")]
        fug = next(c for c in self.cars if not c.pilot.get("police"))
        ch["fug"], ch["cops"] = fug, cops
        base_s = L - 12.0
        if ch["role"] == "fugitivo":
            self.place(fug, base_s, 0.0)
            for k, c in enumerate(cops):
                self.place(c, base_s - 55 - k * 14, (-3.0, 3.0)[k % 2])
                self.tune(c, ch["f_pol"], corner, brake_f, aggressive=True)
        else:
            self.place(fug, base_s + 75, 0.0)
            self.tune(fug, ch["f_fug"], corner, brake_f)
            for k, c in enumerate(cops):
                self.place(c, base_s - k * 14, (-2.5, 2.5)[k % 2])
                if not c.is_player:
                    self.tune(c, ch["f_pol"], corner, brake_f, aggressive=True)

    def place(self, c, s, lane):
        tr = self.track
        L = tr.L
        s %= L
        x, z, dx, dz = tr.point(s)
        c.x, c.z = x - dz * lane, z + dx * lane
        c.ang = math.atan2(dz, dx)
        c.vx = c.vz = 0.0
        c.s = c.s_new = s
        c.idx = int(s / tr.spacing) % tr.n
        c.u = s - L if s > L * 0.5 else s
        c.lap_idx = math.floor(c.u / L)
        c.lane = lane
        if hasattr(c, "true_lat"):
            c.true_lat = lane

    def tune(self, c, f, corner, brake_f, aggressive=False):
        c.vmax = c.pilot["vmax"] * f
        c.accel = c.pilot["accel"] * f
        c.prof = self.track.speed_profile(c.grip * corner, hd.BRAKE * brake_f, c.vmax)
        c.nitro_ai = True
        c.aggr = aggressive

    def chase_end_cleanup(self):
        self.chase = None
        self.siren(0.0)
        hd.LAPS = self.md.laps

    # ------------------------------------------------------------ logica de la persecucion
    def update(self, rdt):
        super().update(rdt)
        ch = getattr(self, "chase", None)
        if not ch or "fug" not in ch or self.frozen or self.state == "menu":
            if not ch:
                self.siren(0.0)
            return
        if self.message.startswith("VUELTA"):                        # en persecucion no hay vueltas
            self.message, self.msg_t = "", 0.0
        fug, cops = ch["fug"], ch["cops"]
        tr = self.track
        L = tr.L
        # las patrullas apuntan al carril del fugitivo cuando lo alcanzan
        x, z, dx, dz = tr.point(fug.s)
        flat = (fug.x - x) * (-dz) + (fug.z - z) * dx
        for c in cops:
            if c.is_player:
                continue
            gap = fug.u - c.u
            if gap < 90:
                c.lane = clamp(flat, -hd.HW + 1.8, hd.HW - 1.8)
            if gap > 140:
                c.nitro = 1.0                                          # alcanzarlo
        if ch["role"] == "policia" and fug.u - self.player.u < 40 and random.random() < 0.02:
            fug.lane = random.uniform(-4, 4)                          # el fugitivo zigzaguea
        if self.state != "race" or ch["over"]:
            self.siren(0.3 if ch["over"] else 0.0)
            return
        dt = rdt
        # golpes de patrulla al fugitivo (cambio brusco de velocidad con una patrulla pegada)
        v = (fug.vx, fug.vz)
        vp = ch["vprev"].get("fug", v)
        dv = math.hypot(v[0] - vp[0], v[1] - vp[1])
        ch["vprev"]["fug"] = v
        near = min(cops, key=lambda c: (c.x - fug.x) ** 2 + (c.z - fug.z) ** 2)
        dn = math.hypot(near.x - fug.x, near.z - fug.z)
        ch["hit_cd"] = max(0.0, ch["hit_cd"] - dt)
        if dv > 3.0 and dn < fug.radius + near.radius + 1.6 and ch["hit_cd"] <= 0:
            dmg = clamp((dv - 2.5) * 0.07, 0.06, 0.45)
            ch["damage"] = min(1.0, ch["damage"] + dmg)
            ch["hit_cd"] = 0.35
            who = "TE GOLPEARON" if ch["role"] == "fugitivo" else "¡GOLPE!"
            self.show(f"{who}  {int(ch['damage'] * 100)} %", 1.0)
            self.shake = max(self.shake, 0.5)
        # arresto: patrulla pegada y el fugitivo casi parado
        if dn < ARREST_D and abs(fug.vf) < ARREST_V:
            ch["busted"] = min(1.0, ch["busted"] + dt / ARREST_T)
        else:
            ch["busted"] = max(0.0, ch["busted"] - dt * 0.6)
        ch["near_d"] = dn
        ch["gap"] = fug.u - max(c.u for c in cops)
        self.siren(clamp(1.2 - dn / 160.0, 0.15, 1.0))
        left = ch["time"] - self.race_time
        if ch["damage"] >= 1.0:
            self.chase_over("derribo")
        elif ch["busted"] >= 1.0:
            self.chase_over("arresto")
        elif left <= 0:
            self.chase_over("tiempo")

    def chase_over(self, why):
        ch = self.chase
        ch["over"] = why
        fugitive = ch["role"] == "fugitivo"
        win = (why == "tiempo") == fugitive
        ch["win"] = win
        p = self.player
        p.finished, p.finish_time = True, self.race_time
        self.state = "finished"
        self.results_t = 0.0
        if fugitive:
            msg = {"tiempo": "¡ESCAPASTE!", "derribo": "¡DERRIBADO!", "arresto": "¡TE ATRAPARON!"}[why]
        else:
            msg = {"tiempo": "SE ESCAPÓ", "derribo": "¡DERRIBO!", "arresto": "¡ARRESTADO!"}[why]
        self.show(msg, 2.5)
        self.sound.play("go" if win else "crash")
        if win:
            self.fx.confetti(p.x, p.z)

    # ------------------------------------------------------------ sirena (sonido) y torretas (luz)
    def siren(self, vol):
        snd = self.sound
        if not getattr(snd, "ok", False):
            return
        if not hasattr(self, "_siren"):
            sr = snd.freq
            t = np.arange(int(sr * 2.4)) / sr
            f = 760 + 420 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 2.4))       # ulular
            ph = 2 * np.pi * np.cumsum(f) / sr
            w = np.sign(np.sin(ph)) * 0.25 + np.sin(ph) * 0.35 + np.sin(2 * ph) * 0.12
            self._siren = snd._make(w, 0.5)
            pygame.mixer.set_num_channels(max(8, pygame.mixer.get_num_channels()))
            self._siren_ch = pygame.mixer.Channel(6)
        ch = self._siren_ch
        v = 0.0 if snd.muted else vol * float(self.cfg.get("fx_vol", 1.0)) * 0.6
        if v <= 0.01:
            if ch.get_busy():
                ch.stop()
            return
        if not ch.get_busy():
            ch.play(self._siren, loops=-1)
        ch.set_volume(v)

    def gather_lights(self, eye, tod):
        r = super().gather_lights(eye, tod)
        ch = getattr(self, "chase", None)
        if not ch or "cops" not in ch or self.state in ("select", "lobby", "menu"):
            return r
        ex, _, ez = eye
        extra = []
        for k, c in enumerate(ch["cops"]):
            if (c.x - ex) ** 2 + (c.z - ez) ** 2 > 200 ** 2:
                continue
            y, _ = self.car_y(c) if hasattr(self, "car_y") else (0.0, None)
            red = int(self.t * 4 + k) % 2 == 0
            col = (1.0, 0.1, 0.06) if red else (0.1, 0.3, 1.0)
            extra.append((c.x, y + 2.2, c.z, 22.0, *col, 6.0))
        if extra:
            P = np.concatenate([self.L_pos, np.array([e[0:4] for e in extra], np.float32)])[:48]
            C = np.concatenate([self.L_col, np.array([e[4:8] for e in extra], np.float32)])[:48]
            self.L_pos, self.L_col = P, C
            self._lkey = None
            return self.light_bytes(None, 0)
        return r

    # ------------------------------------------------------------ interfaz
    def hud_race(self):
        ch = getattr(self, "chase", None)
        if not ch or "fug" not in ch:
            return super().hud_race()
        self.hud_chase()

    def hud_chase(self):
        ch = self.chase
        p = self.player
        fugitive = ch["role"] == "fugitivo"
        left = max(0.0, ch["time"] - self.race_time) if self.state != "countdown" else ch["time"]
        if ch.get("over"):
            left = max(0.0, ch["time"] - p.finish_time)
        # reloj
        col = (255, 255, 255) if left > 15 or not fugitive else (255, 215, 60)
        self.panel((WIN_W // 2 - 190, 14, 380, 96), 170, accent=(255, 60, 60) if fugitive else (60, 120, 255), slant=18)
        self.txt("ESCAPA DURANTE" if fugitive else "DETÉN AL FUGITIVO EN", 17, (190, 205, 220), (WIN_W // 2, 26),
                 "center", italic=False)
        self.txt(fmt_time(left), 52, col, (WIN_W // 2, 70), "center", glow=(255, 60, 40) if left < 10 else None)
        # barras
        self.panel((18, 18, 330, 168), 160, accent=(255, 60, 60) if fugitive else (60, 120, 255))
        self.txt("FUGITIVO" if fugitive else "POLICÍA", 30, (255, 255, 255), (40, 26))
        self.txt(ch["fug"].name if not fugitive else f"{len(ch['cops'])} PATRULLAS TE SIGUEN", 16,
                 (190, 205, 220), (42, 62), italic=False)

        def bar(y, lab, v, col):
            self.txt(lab, 16, (200, 210, 225), (40, y), italic=False)
            pygame.draw.rect(self.hud, (40, 46, 60, 255), (150, y + 3, 170, 14))
            pygame.draw.rect(self.hud, col + (255,), (150, y + 3, int(170 * clamp(v, 0, 1)), 14))
        bar(96, "DAÑO", ch["damage"], (255, 120, 40))
        bar(124, "ATRAPADO" if fugitive else "ARRESTO", ch["busted"], (255, 60, 60))
        d = ch.get("near_d")
        if d is not None:
            s = f"PATRULLA A {d:.0f} m" if fugitive else f"FUGITIVO A {d:.0f} m"
            self.txt(s, 18, (255, 215, 60) if d < 30 else (210, 220, 232), (40, 152), italic=False)
        self.minimap()
        self.speedometer(p)
        if self.state == "countdown":
            n = int(self.countdown)
            if n >= 1:
                fr = self.countdown - int(self.countdown)
                self.txt(str(n), int(150 + 80 * fr), (255, 255, 255), (WIN_W // 2, WIN_H // 2 - 40), "center",
                         glow=(255, 60, 40))
        if self.msg_t > 0 and self.message:
            a = int(255 * min(1.0, self.msg_t * 3))
            self.txt(self.message, 72, (255, 255, 255), (WIN_W // 2, WIN_H // 2 - 90), "center",
                     glow=(255, 60, 40) if fugitive else ACCENT, alpha=a)
        if ch["busted"] > 0.05 and self.state == "race" and int(self.t * 4) % 2 == 0:
            self.txt("¡TE ESTÁN CERRANDO EL PASO!" if fugitive else "¡CIÉRRALE EL PASO!", 30, (255, 80, 60),
                     (WIN_W // 2, 150), "center")
        self.song_banner(WIN_H - 62)
        if self.state == "finished" and self.results_t > 1.5:
            self.results(self.standings())

    def results(self, order):
        ch = getattr(self, "chase", None)
        if not ch or not ch.get("over"):
            return super().results(order)
        win = ch.get("win")
        fugitive = ch["role"] == "fugitivo"
        why = ch["over"]
        panel = pygame.Rect(WIN_W // 2 - 360, 150, 720, 330)
        self.panel(panel, 215, accent=(90, 230, 120) if win else (255, 60, 60), slant=30)
        if fugitive:
            title = {"tiempo": "¡ESCAPASTE!", "derribo": "¡DERRIBADO!", "arresto": "¡TE ATRAPARON!"}[why]
            sub = {"tiempo": "La policía te perdió la pista.",
                   "derribo": "La patrulla te sacó del camino.",
                   "arresto": "Te cerraron el paso y te detuvieron."}[why]
        else:
            title = {"tiempo": "SE ESCAPÓ", "derribo": "¡FUGITIVO DERRIBADO!", "arresto": "¡FUGITIVO ARRESTADO!"}[why]
            sub = {"tiempo": "Se acabó el tiempo y el fugitivo se perdió.",
                   "derribo": "Lo sacaste del camino a golpes.",
                   "arresto": "Le cerraste el paso hasta detenerlo."}[why]
        self.txt(title, 72, (90, 230, 120) if win else (255, 80, 60), (WIN_W // 2, 220), "center",
                 glow=(30, 160, 60) if win else (255, 0, 0))
        self.txt(sub, 24, (220, 228, 240), (WIN_W // 2, 290), "center", italic=False)
        tm = ch["fug"].finish_time if False else self.player.finish_time
        self.txt(f"TIEMPO: {fmt_time(tm)}  /  {fmt_time(ch['time'])}      DAÑO: {int(ch['damage'] * 100)} %", 22,
                 (200, 210, 225), (WIN_W // 2, 340), "center", italic=False)
        self.txt("ENTER  otra vez      ESC  pausa / menú", 22, (210, 220, 230), (WIN_W // 2, 420), "center",
                 italic=False)

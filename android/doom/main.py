"""
GTC Racing (version Doom) para Android: agrega controles tactiles al juego de gtc_racing.py.

Controles en pantalla:
  izquierda: girar a la izquierda / derecha
  derecha:   ACELERAR, FRENO (reversa) y DERRAPE (freno de mano)
  arriba:    MENU (volver al menu) y SONIDO
  tocar la pantalla en el menu o en los resultados: empezar carrera
  boton ATRAS del telefono: volver al menu / salir
"""
import sys

import pygame

import gtc_racing as g

SW, SH, VIEW_H = g.SW, g.SH, g.VIEW_H


class TouchGame(g.Game):
    def __init__(self):
        super().__init__()
        self.fingers = {}                 # id del dedo -> (x, y) en pixeles logicos
        self.used_touch = False
        y_low = VIEW_H - 175
        self.buttons = {
            "left": pygame.Rect(18, y_low, 150, 160),
            "right": pygame.Rect(182, y_low, 150, 160),
            "gas": pygame.Rect(SW - 185, y_low - 10, 170, 170),
            "brake": pygame.Rect(SW - 365, y_low + 45, 165, 115),
            "drift": pygame.Rect(SW - 365, y_low - 80, 165, 110),
            "menu": pygame.Rect(16, 14, 110, 52),
            "sound": pygame.Rect(140, 14, 110, 52),
        }
        self.labels = {"left": "<", "right": ">", "gas": "GAS", "brake": "FRENO", "drift": "DERRAPE",
                       "menu": "MENU", "sound": "SONIDO"}
        self.f_btn = pygame.font.SysFont("arial", 30, bold=True)
        self.f_btn_s = pygame.font.SysFont("arial", 20, bold=True)
        self.overlay = pygame.Surface((SW, SH), pygame.SRCALPHA)

    # ------------------------------------------------------------ toques
    def pressed(self, name):
        r = self.buttons[name]
        return any(r.collidepoint(p) for p in self.fingers.values())

    def player_inputs(self):
        k = pygame.key.get_pressed()     # el teclado sigue funcionando (util para probar en PC)
        thr = self.pressed("gas") or k[pygame.K_UP] or k[pygame.K_w]
        brk = self.pressed("brake") or k[pygame.K_DOWN] or k[pygame.K_s]
        right = self.pressed("right") or k[pygame.K_RIGHT] or k[pygame.K_d]
        left = self.pressed("left") or k[pygame.K_LEFT] or k[pygame.K_a]
        hb = self.pressed("drift") or k[pygame.K_SPACE]
        return thr, brk, (1 if right else 0) - (1 if left else 0), hb

    def tap(self, pos):
        if self.buttons["menu"].collidepoint(pos):
            if self.state != "menu":
                self.new_race(attract=True)
            return
        if self.buttons["sound"].collidepoint(pos):
            self.sound.muted = not self.sound.muted
            return
        if self.state == "menu":
            self.new_race()
        elif self.state == "finished" and self.results_t > 2.0:
            self.new_race()

    def to_logical(self, e):
        return int(e.x * SW), int(e.y * SH)

    # ------------------------------------------------------------ dibujo
    def draw(self):
        real_flip = pygame.display.flip
        pygame.display.flip = lambda: None   # el juego base hace flip al final; primero dibujamos los botones
        try:
            super().draw()
        finally:
            pygame.display.flip = real_flip
        o = self.overlay
        o.fill((0, 0, 0, 0))
        racing = self.state in ("countdown", "race")
        for name, r in self.buttons.items():
            if name in ("menu", "sound"):
                if self.state == "menu" and name == "menu":
                    continue
            elif not racing:
                continue
            on = self.pressed(name)
            col = (255, 220, 60, 150) if on else (255, 255, 255, 70)
            if name in ("gas", "left", "right"):
                pygame.draw.ellipse(o, col, r)
                pygame.draw.ellipse(o, (0, 0, 0, 120), r, 3)
            else:
                pygame.draw.rect(o, col, r, border_radius=18)
                pygame.draw.rect(o, (0, 0, 0, 120), r, 3, border_radius=18)
            label = self.labels[name]
            if name == "sound" and self.sound.muted:
                label = "MUDO"
            f = self.f_btn if name in ("gas", "left", "right") else self.f_btn_s
            t = f.render(label, True, (20, 20, 20))
            o.blit(t, t.get_rect(center=r.center))
        if self.state == "menu":
            pygame.draw.rect(o, (0, 0, 0, 160), (SW // 2 - 260, 300, 520, 60), border_radius=14)
            t = self.f_mid.render("TOCA LA PANTALLA PARA CORRER", True, (255, 220, 60))
            o.blit(t, t.get_rect(center=(SW // 2, 330)))
        elif self.state == "finished" and self.results_t > 2.0:
            t = self.f_btn_s.render("Toca para otra carrera", True, (255, 255, 255))
            o.blit(t, t.get_rect(center=(SW // 2, SH - 120)))
        self.screen.blit(o, (0, 0))
        pygame.display.flip()

    # ------------------------------------------------------------ bucle
    def run(self):
        while True:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif e.type == pygame.FINGERDOWN:
                    self.used_touch = True
                    p = self.to_logical(e)
                    self.fingers[e.finger_id] = p
                    self.tap(p)
                elif e.type == pygame.FINGERMOTION:
                    self.fingers[e.finger_id] = self.to_logical(e)
                elif e.type == pygame.FINGERUP:
                    self.fingers.pop(e.finger_id, None)
                elif e.type == pygame.MOUSEBUTTONDOWN and not self.used_touch and not getattr(e, "touch", False):
                    self.fingers["mouse"] = e.pos          # para probar con el raton en la PC
                    self.tap(e.pos)
                elif e.type == pygame.MOUSEMOTION and "mouse" in self.fingers:
                    self.fingers["mouse"] = e.pos
                elif e.type == pygame.MOUSEBUTTONUP:
                    self.fingers.pop("mouse", None)
                elif e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_AC_BACK):
                        if self.state == "menu":
                            pygame.quit()
                            sys.exit()
                        self.new_race(attract=True)
                    elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.state == "menu":
                        self.new_race()
                    elif e.key == pygame.K_r and self.state != "menu":
                        self.new_race()
            self.update(dt)
            self.draw()


if __name__ == "__main__":
    TouchGame().run()

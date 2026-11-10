# -*- coding: utf-8 -*-
"""
gtc_metal.py - Sintetizador procedural de canciones de METAL instrumental
para GTC Racing (100% numpy, sin scipy ni ffmpeg).

Todas las composiciones son ORIGINALES: los riffs, progresiones y melodias se
construyen con plantillas ritmicas propias + generadores con semilla fija
(salida reproducible). Solo se imitan los rasgos de estilo de cada genero:

  * metalcore / metal melodico: galopes con palm-mute, doble bombo,
    leads gemelos armonizados en terceras, solo veloz y breakdown.
  * nu-metal: afinacion grave (drop B / drop C), grooves sincopados,
    toms tribales, lead disonante y una rafaga de blast-beat.
  * post-grunge / rock alternativo pesado: medio tiempo, estrofas con
    arpegios limpios, coros con power-chords grandes, lead "cantado".
  * thrash / speed: skank beat rapido, riffs de semicorcheas.

API publica:
    SONGS                                   -> lista de metadatos
    generate_all(out_dir, force=False, progress=None)
    render_song(song)                       -> np.ndarray (n, 2) float
    write_wav(path, stereo)

Uso:  python gtc_metal.py      (genera en ./music_metal/ junto al modulo)
"""

import os
import sys
import time
import wave

import numpy as np

# --------------------------------------------------------------------------
# Parametros globales de audio
# --------------------------------------------------------------------------
SR = 22050          # frecuencia de muestreo (estereo 16 bits -> ~5.3 MB/min)
TAIL_S = 3.0        # cola final (acorde sonando + fade)
TARGET_S = 128.0    # duracion objetivo de cada cancion (s)
MASTER_RMS_DB = -12.5   # sonoridad objetivo (RMS de las partes activas)
CEIL = 0.89         # techo del limitador (~ -1 dBFS)

# --------------------------------------------------------------------------
# Escalas (semitonos sobre la tonica)
# --------------------------------------------------------------------------
SCALES = {
    'minor':    [0, 2, 3, 5, 7, 8, 10],
    'harm':     [0, 2, 3, 5, 7, 8, 11],
    'phryg':    [0, 1, 3, 5, 7, 8, 10],
    'phrygdom': [0, 1, 4, 5, 7, 8, 10],
    'dorian':   [0, 2, 3, 5, 7, 9, 10],
}

# --------------------------------------------------------------------------
# Lista de canciones (metadatos publicos + parametros internos)
# --------------------------------------------------------------------------
SONGS = [
    dict(file='01 - Nitro en el Periferico.wav', title='Nitro en el Periférico',
         style='Metalcore', bpm=172, key='Mi menor', root=40, scale='minor',
         tpl='metalcore', seed=1101,
         prog=dict(chorus=[0, 5, 2, 6], pre=[(5, 6), (3, 4)], verse=[0, 5, 3, 6],
                   solo=[0, 3, 5, 6])),
    dict(file='02 - Tormenta en Cozumel.wav', title='Tormenta en Cozumel',
         style='Metal melódico (leads gemelos)', bpm=150,
         key='Re menor armónica (Drop D)', root=38, scale='harm',
         tpl='melodic', seed=2207,
         prog=dict(chorus=[0, 5, 3, 4], pre=[5, 4], verse=[0, 5, 3, 4],
                   solo=[0, 5, 3, 4])),
    dict(file='03 - Segundo Piso.wav', title='Segundo Piso',
         style='Nu-metal', bpm=104, key='Si frigio (Drop B)', root=35,
         scale='phryg', tpl='nu', seed=3313,
         prog=dict(chorus=[0, 5, 1, 0], pre=[0, 1], verse=[0, 1, 0, 6],
                   solo=[0, 5, 1, 0])),
    dict(file='04 - Ruido de Motores.wav', title='Ruido de Motores',
         style='Thrash metal', bpm=206, key='Mi frigio', root=40,
         scale='phryg', tpl='thrash', seed=4409,
         prog=dict(chorus=[0, 6, 5, 1], pre=[(5, 6), (1, 0)], verse=[0, 6, 5, 1],
                   solo=[0, 5, 6, 0])),
    dict(file='05 - Asfalto Mojado.wav', title='Asfalto Mojado',
         style='Post-grunge / rock alternativo pesado', bpm=96,
         key='Do# menor (Drop C#)', root=37, scale='minor', tpl='grunge',
         seed=5519,
         prog=dict(chorus=[5, 2, 6, 0], pre=[3, 4], verse=[0, 5, 3, 6],
                   solo=[5, 6, 0, 0])),
    dict(file='06 - Calaveras al Volante.wav', title='Calaveras al Volante',
         style='Nu-metal', bpm=112, key='Do frigio dominante (Drop C)', root=36,
         scale='phrygdom', tpl='nu', seed=6637,
         prog=dict(chorus=[0, 1, 5, 0], pre=[1, 0], verse=[0, 1, 0, 5],
                   solo=[0, 1, 5, 0])),
    dict(file='07 - Reina del Tangamanga.wav', title='Reina del Tangamanga',
         style='Metalcore (breakdowns)', bpm=144, key='Do# menor', root=37,
         scale='minor', tpl='metalcore', seed=7741,
         prog=dict(chorus=[5, 6, 0, 4], pre=[(3, 4), 5], verse=[0, 5, 6, 4],
                   solo=[0, 5, 6, 4])),
    dict(file='08 - Medianoche en Reforma.wav', title='Medianoche en Reforma',
         style='Post-grunge / metal alternativo', bpm=88, key='Re dórico (Drop D)',
         root=38, scale='dorian', tpl='grunge', seed=8853,
         prog=dict(chorus=[0, 6, 3, 0], pre=[2, 6], verse=[0, 3, 6, 3],
                   solo=[0, 6, 3, 4])),
    dict(file='09 - Ultima Vuelta.wav', title='Última Vuelta',
         style='Speed metal', bpm=188, key='Fa# menor armónica', root=42,
         scale='harm', tpl='thrash', seed=9967,
         prog=dict(chorus=[0, 5, 6, 4], pre=[(5, 4), (5, 4)], verse=[0, 5, 6, 4],
                   solo=[0, 5, 3, 4])),
]

# --------------------------------------------------------------------------
# Plantillas de estructura (seccion, compases base) -> se ajustan a ~2:08
# --------------------------------------------------------------------------
TPL = {
    'metalcore': [('intro_lead', 4), ('verse', 8), ('pre', 4), ('chorus', 8),
                  ('verse2', 8), ('pre', 4), ('chorus', 8), ('breakdown', 8),
                  ('solo', 8), ('chorus', 8), ('outro', 4)],
    'melodic':   [('intro_lead', 8), ('verse', 8), ('pre', 4), ('chorus', 8),
                  ('interlude', 4), ('verse2', 8), ('pre', 4), ('chorus', 8),
                  ('solo', 16), ('chorus', 8), ('outro', 4)],
    'nu':        [('intro_perc', 4), ('verse', 8), ('pre', 4), ('chorus', 8),
                  ('verse2', 8), ('pre', 4), ('chorus', 8), ('blast', 2),
                  ('breakdown', 8), ('chorus', 8), ('outro', 2)],
    'thrash':    [('intro_riff', 4), ('verse', 8), ('pre', 4), ('chorus', 8),
                  ('verse2', 8), ('pre', 4), ('chorus', 8), ('solo', 16),
                  ('verse', 8), ('chorus', 8), ('outro', 4)],
    'grunge':    [('intro_clean', 4), ('verse_clean', 8), ('pre', 4),
                  ('chorus', 8), ('verse_clean2', 8), ('pre', 4), ('chorus', 8),
                  ('solo', 8), ('chorus', 8), ('outro', 4)],
}

# --------------------------------------------------------------------------
# Riffs originales (2 compases = 32 pasos de semicorchea)
#   x  = chug palm-mute en la tonica grave      .  = silencio   - = sostener
#   P Q R = power-chord abierto (grados del motivo del riff)
#   p q r = nota suelta palm-mute (grados del motivo)
#   A..J  = power-chord abierto en grado 0..9   a..j = power-chord palm-mute
#   0..9  = nota suelta palm-mute en grado 0..9
# --------------------------------------------------------------------------
RIFFS = {
    # metalcore (galopes)
    'M1': "x.xxx.xxP---x.xx" + "x.xxx.xxQ---R---",
    'M2': "xxx.xxx.xxx.P-Q-" + "xxx.xxx.xxx.R---",
    'M3': "x.xxP-x.xxQ-x.xx" + "x.xxP-x.xxR---p-",
    'M4': "P---x.xxQ---x.xx" + "R---x.xxx.xxp-q-",
    'M5': "xxxxP-xxxxQ-xxxx" + "xxxxR-xxxxP-q-r-",
    # thrash / speed
    'T1': "xxxxP-xxxxQ-xxxx" + "xxxxP-xxR-Q-P---",
    'T2': "x.xxx.xxx.xxP-Q-" + "x.xxx.xxx.xxR-p-",
    'T3': "xxxxxxxxP-P-xxxx" + "xxxxxxxxQ-Q-R-R-",
    'T4': "PxxxQxxxPxxxRxxx" + "PxxxQxxxp.q.r.p.",
    # nu-metal (sincopado, afinacion grave)
    'N1': "x..x..x.P---.x.." + "x..x..x.Q---R-..",
    'N2': "x.x...x.x.P-...." + "x.x...x.x.Q-R...",
    'N3': "P-..x.x.x...x..." + "Q-..x.x.x...p-q-",
    'N4': "x...x..xx...P---" + "x...x..xx...Q-R-",
    # breakdowns
    'B1': "x...x..x..x.P---" + "x...x..x..x.Q---",
    'B2': "x..x..x..x..x..." + "x..x..x..P---...",
    'B3': "x.......x..x..x." + "x.......x..xP---",
    # post-grunge
    'G1': "x.x.x.xxP---Q---" + "x.x.x.xxR-------",
    'G2': "P-----x.Q-----x." + "R-----x.x.x.p-q-",
    'G3': "x-x-x-x-P-------" + "x-x-x-x-Q---R---",
}
for _k, _v in RIFFS.items():
    assert len(_v) == 32, (_k, len(_v))

# Patrones de acordes por compas ('@' = acorde abierto, '#' = acorde palm-mute)
CPAT = {
    'drive8':  "@-@-@-@-@-@-@-@-",
    'gallopc': "@.@@@.@@@.@@@.@@",
    'pmdrive': "@---#.#.@---#.#.",
    'sustain': "@---------------",
    'half':    "@-------@-------",
    'big':     "@-----------@---",
    'syncop':  "@-----@-----@---",
    'trem':    "@@@@@@@@@@@@@@@@",
}
for _k, _v in CPAT.items():
    assert len(_v) == 16, _k

# Configuracion de cada estilo
ESTILOS = {
    'metalcore': dict(riffs=('M1', 'M2', 'M3', 'M4', 'M5'), bd=('B1', 'B2', 'B3'),
                      verse_dr=('gallop', 'hat8'), verse2_dr=('follow', 'ride4'),
                      pre_pat='half', pre_dr=('rock2', 'ride8'),
                      chorus_pat='drive8', chorus_dr=('double', 'crash4'),
                      solo_pat='pmdrive', solo_dr=('double', 'ride4'),
                      lead='lead', harm=True, gain=1.0),
    'melodic':   dict(riffs=('M1', 'M3', 'M4', 'M5', 'T2'), bd=('B1', 'B3'),
                      verse_dr=('double8', 'hat8'), verse2_dr=('gallop', 'ride4'),
                      pre_pat='half', pre_dr=('rock2', 'ride8'),
                      chorus_pat='gallopc', chorus_dr=('gallop', 'crash4'),
                      solo_pat='drive8', solo_dr=('double', 'ride8'),
                      lead='lead', harm=True, gain=0.95),
    'nu':        dict(riffs=('N1', 'N2', 'N3', 'N4'), bd=('B1', 'B2', 'B3'),
                      verse_dr=('follow', 'hat8'), verse2_dr=('nu', 'hat16'),
                      pre_pat='riff', pre_dr=('tribal', 'hat8'),
                      chorus_pat='syncop', chorus_dr=('follow', 'crash4'),
                      solo_pat='syncop', solo_dr=('follow', 'ride8'),
                      lead='eerie', harm=False, gain=1.1),
    'thrash':    dict(riffs=('T1', 'T2', 'T3', 'T4'), bd=('B1', 'B3'),
                      verse_dr=('skank', 'hat8'), verse2_dr=('skank', 'ride8'),
                      pre_pat='half', pre_dr=('double8', 'ride8'),
                      chorus_pat='drive8', chorus_dr=('skank', 'crash4'),
                      solo_pat='pmdrive', solo_dr=('double8', 'ride8'),
                      lead='lead', harm=True, gain=1.0),
    'grunge':    dict(riffs=('G1', 'G2', 'G3'), bd=('B1',),
                      verse_dr=('rock', 'hat8'), verse2_dr=('rock2', 'hat8'),
                      pre_pat='half', pre_dr=('rock', 'ride8'),
                      chorus_pat='big', chorus_dr=('rock', 'ohat8'),
                      solo_pat='half', solo_dr=('rock2', 'ride8'),
                      lead='vocal', harm=False, gain=0.7),
}

# Patrones de bateria (bombo, tarola). 'F' = el bombo sigue a la guitarra
DR = {
    'gallop':  ('x.xxx.xxx.xxx.xx', '....x.......x...'),
    'double':  ('xxxxxxxxxxxxxxxx', '....x.......x...'),
    'double8': ('x.x.x.x.x.x.x.x.', '....x.......x...'),
    'skank':   ('x.x.x.x.x.x.x.x.', '..x...x...x...x.'),
    'rock':    ('x.....x.x.......', '....x.......x...'),
    'rock2':   ('x.x.....x.x..x..', '....x.......x...'),
    'clean':   ('x.......x.x.....', '....x.......x...'),
    'nu':      ('x..x..x...x..x..', '....x.......x...'),
    'follow':  ('F', '....x.......x...'),
    'half':    ('F', '........x.......'),
    'blast':   ('xxxxxxxxxxxxxxxx', 'xxxxxxxxxxxxxxxx'),
    'tribal':  ('x.....x...x.....', '............x...'),
    'none':    ('................', '................'),
}
TOMS_TRIBAL = {'t1': '......x.......xx', 't2': '...x.....x...x..',
               't3': 'x.....x...x.....'}
CYM = {
    'hat8':   ('hh', 'x.x.x.x.x.x.x.x.'),
    'hat16':  ('hh', 'xxxxxxxxxxxxxxxx'),
    'ride4':  ('ride', 'x...x...x...x...'),
    'ride8':  ('ride', 'x.x.x.x.x.x.x.x.'),
    'crash4': ('crash', 'x...x...x...x...'),
    'china4': ('china', 'x...x...x...x...'),
    'ohat8':  ('oh', 'x.x.x.x.x.x.x.x.'),
    'none':   (None, '................'),
}
FILLS8 = [['s', 's', 't1', 't1', 't2', 't2', 't3', 't3'],
          ['t1', 't1', 's', 't2', 't2', 's', 't3', 't3'],
          ['s', 't1', 's', 't1', 't2', 't3', 't3', 't3']]
FILLS4 = [['t1', 't2', 't3', 't3'], ['s', 's', 't2', 't3'], ['s', 's', 's', 's']]

# Ritmos de melodia (n = nota, - = sostener, . = silencio)
MEL_R = ["n---n-n-n---n---", "n-n-n---n-n-n---", "n--n--n-n-------",
         "n-----n-n-n-n---", "..n-n-n-n---n-n-", "n---n---n---n-n-",
         "n-------n--n--n-", "n-n-n-n-n---n---"]
MEL_END = ["n-------n-------", "n---n---n-------", "n-n-n-----------",
           "n---------------", "n--n--n---------"]
MEL_VOC = ["n-----n-n-------", "..n-n-n---n-----", "n---n-----n-n---",
           "....n-n-n-n-----", "n-------..n-n---"]


# ==========================================================================
# Utilidades DSP (todo vectorizado)
# ==========================================================================
def _deg(sc, i):
    """Grado de escala (puede ser negativo o > 7) -> semitonos."""
    i = int(i)
    return sc[i % 7] + 12 * (i // 7)


def _nfft(n):
    return 1 << int(np.ceil(np.log2(max(2, n))))


def _lp(f, fc, order=2):
    return 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))


def _hp(f, fc, order=2):
    return 1.0 / np.sqrt(1.0 + (fc / np.maximum(f, 1e-3)) ** (2 * order))


def _bell(f, fc, db, oct_=0.7):
    g = 10.0 ** (db / 20.0)
    lf = np.log2(np.maximum(f, 1.0) / fc)
    return 1.0 + (g - 1.0) * np.exp(-0.5 * (lf / oct_) ** 2)


def _filt(x, resp, pad=None):
    """Filtro de fase cero en el dominio de la frecuencia (FFT).
    resp(f) -> ganancia por frecuencia. Acepta mono (n,) o estereo (n, 2)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if pad is None:
        pad = min(n, SR // 4) + 64
    N = _nfft(n + pad)
    f = np.fft.rfftfreq(N, 1.0 / SR)
    H = resp(f)
    X = np.fft.rfft(x, N, axis=0)
    if x.ndim == 2:
        H = H[:, None]
    return np.fft.irfft(X * H, N, axis=0)[:n]


def _ma(x, n):
    """Media movil centrada (cumsum)."""
    n = int(n)
    if n <= 1 or len(x) == 0:
        return x.copy()
    if len(x) <= n:
        return np.full(len(x), x.mean())
    c = np.cumsum(np.concatenate(([0.0], x)))
    y = (c[n:] - c[:-n]) / n
    l = (n - 1) // 2
    r = len(x) - len(y) - l
    return np.concatenate((np.full(l, y[0]), y, np.full(r, y[-1])))


def _maxfilt(a, w):
    """Maximo deslizante centrado (ancho = potencia de 2 >= w), O(n log w)."""
    p = int(np.ceil(np.log2(max(w, 2))))
    m = a.copy()
    k = 1
    for _ in range(p):
        m2 = m.copy()
        m2[:-k] = np.maximum(m[:-k], m[k:])
        m = m2
        k *= 2
    s = (1 << p) // 2
    out = np.empty_like(m)
    out[s:] = m[:-s]
    out[:s] = m[0]
    return out


def _minfilt(a, w):
    return -_maxfilt(-a, w)


def _saw(ph, dt):
    """Diente de sierra con PolyBLEP (menos aliasing). ph = fase acumulada."""
    p = ph - np.floor(ph)
    y = 2.0 * p - 1.0
    dt = np.broadcast_to(np.asarray(dt, dtype=np.float64), p.shape)
    m = p < dt
    if m.any():
        t = p[m] / dt[m]
        y[m] -= t + t - t * t - 1.0
    m = p > 1.0 - dt
    if m.any():
        t = (p[m] - 1.0) / dt[m]
        y[m] -= t * t + t + t + 1.0
    return y


def _ref(x):
    """Nivel de referencia robusto (percentil 99 de |x| de lo no silencioso)."""
    a = np.abs(x)
    a = a[a > 1e-5]
    if a.size == 0:
        return 1.0
    if a.size > 400000:
        a = a[:: a.size // 400000]
    return float(np.percentile(a, 99)) + 1e-9


def _rms_activo(x):
    """RMS de las partes activas (bloques de 0.5 s por encima de -20 dB)."""
    sq = x * x if x.ndim == 1 else (x * x).mean(axis=1)
    b = SR // 2
    nb = len(sq) // b
    if nb == 0:
        return float(np.sqrt(sq.mean()))
    e = sq[:nb * b].reshape(nb, b).mean(axis=1)
    if e.max() <= 0:
        return 0.0
    act = e[e > e.max() * 0.01]
    return float(np.sqrt(act.mean()))


def _norm(x, db):
    r = _rms_activo(x)
    return x * (10.0 ** (db / 20.0) / r) if r > 0 else x


def _comp(x, thr_db, ratio, win_ms=10.0, rel_ms=80.0):
    """Compresor sencillo (envolvente RMS + ganancia suavizada)."""
    mono = np.abs(x) if x.ndim == 1 else np.max(np.abs(x), axis=1)
    env = np.sqrt(_ma(mono * mono, SR * win_ms / 1000.0)) + 1e-9
    db = 20.0 * np.log10(env)
    gdb = np.minimum(0.0, (thr_db - db) * (1.0 - 1.0 / ratio))
    g = 10.0 ** (_ma(gdb, SR * rel_ms / 1000.0) / 20.0)
    return x * (g if x.ndim == 1 else g[:, None])


def _limiter(x, ceil=CEIL, look_ms=3.0, rel_ms=60.0):
    """Limitador de picos con 'look-ahead' (filtros max/min + media movil)."""
    a = np.max(np.abs(x), axis=1)
    pk = _maxfilt(a, int(SR * look_ms / 1000.0) * 2)
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    w = int(SR * rel_ms / 1000.0)
    g = _ma(_minfilt(g, w), w)
    return x * g[:, None]


def _pan(x, p):
    """Paneo de potencia constante, p en [-1, 1]."""
    a = (p + 1.0) * np.pi / 4.0
    return np.stack((x * np.cos(a), x * np.sin(a)), axis=1)


def _shift(x, d):
    y = np.zeros_like(x)
    if d < len(x):
        y[d:] = x[:len(x) - d]
    return y


# ==========================================================================
# Bateria sintetizada
# ==========================================================================
def _make_kit(rng):
    kit = {}

    def tt(dur):
        return np.arange(int(dur * SR)) / SR

    def nrm(x):
        return x / (np.max(np.abs(x)) + 1e-9)

    def metal(t, base, hp, ratios=(1.0, 1.4823, 1.7428, 2.5506, 2.6334, 3.9122)):
        # Ruido "metalico": ondas cuadradas inarmonicas + ruido, pasa-altos
        s = np.zeros(len(t))
        for r in ratios:
            s += np.sign(np.sin(2 * np.pi * base * r * t + rng.random() * 6.28))
        s += rng.standard_normal(len(t)) * 1.6
        return _filt(s, lambda f: _hp(f, hp, 3))

    for nombre in ('k', 's', 't1', 't2', 't3', 'hh', 'oh', 'ride', 'crash',
                   'crash2', 'china'):
        kit[nombre] = []
    for v in range(4):
        dv = 1.0 + 0.02 * (v - 1.5)
        # --- bombo: seno con caida de tono + click de batidor
        t = tt(0.45)
        f = 47.0 * dv + 115.0 * np.exp(-t / 0.026)
        body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.2)
        nz = rng.standard_normal(len(t)) * np.exp(-t / 0.003)
        click = _filt(nz, lambda f_: _hp(f_, 1800, 2) * _lp(f_, 7000, 2))
        click = click / (np.max(np.abs(click)) + 1e-9)
        click += 0.5 * np.sin(2 * np.pi * 3100 * t) * np.exp(-t / 0.0018)
        kit['k'].append(nrm(np.tanh(1.6 * (body + 0.55 * click))))
        # --- tarola: tono + ruido de bordonero
        t = tt(0.4)
        tone = (0.65 * np.sin(2 * np.pi * 182 * dv * t * (1 + 0.08 * np.exp(-t / 0.01)))
                * np.exp(-t / 0.075)
                + 0.35 * np.sin(2 * np.pi * 329 * dv * t) * np.exp(-t / 0.05))
        nz = rng.standard_normal(len(t)) * (np.exp(-t / 0.16) * 0.8 + 0.2 * np.exp(-t / 0.02))
        nz = _filt(nz, lambda f_: _hp(f_, 1100, 2) * _lp(f_, 9000, 2)
                   * _bell(f_, 4200, 4, 0.6))
        nz /= np.max(np.abs(nz)) + 1e-9
        kit['s'].append(nrm(np.tanh(1.7 * (tone + 0.85 * nz))))
        # --- toms (agudo, medio, de piso)
        for nom, f0 in (('t1', 165.0), ('t2', 118.0), ('t3', 84.0)):
            t = tt(0.6)
            f = f0 * dv * (1 + 0.45 * np.exp(-t / 0.05))
            body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.3)
            nz = rng.standard_normal(len(t)) * np.exp(-t / 0.02)
            nz = _filt(nz, lambda f_: _lp(f_, 3000, 2) * _hp(f_, 200, 1))
            nz /= np.max(np.abs(nz)) + 1e-9
            kit[nom].append(nrm(np.tanh(1.4 * (body + 0.25 * nz))))
        # --- hi-hat cerrado / abierto
        t = tt(0.12)
        kit['hh'].append(nrm(metal(t, 410 * dv, 6800) * np.exp(-t / 0.032)))
        t = tt(0.7)
        kit['oh'].append(nrm(metal(t, 410 * dv, 6000) * np.exp(-t / 0.28)))
        # --- ride: lavado + campana
        t = tt(1.6)
        wash = metal(t, 530 * dv, 4500) * np.exp(-t / 0.75)
        wash /= np.max(np.abs(wash)) + 1e-9
        bell = (np.sin(2 * np.pi * 2950 * dv * t) + 0.6 * np.sin(2 * np.pi * 4410 * dv * t)
                + 0.4 * np.sin(2 * np.pi * 1740 * dv * t)) * np.exp(-t / 0.35)
        tick = rng.standard_normal(len(t)) * np.exp(-t / 0.002)
        kit['ride'].append(nrm(0.6 * wash + 0.22 * bell + 0.3 * tick))
        # --- crash x2 y china
        for nom, base, hp_ in (('crash', 340 * dv, 3200), ('crash2', 385 * dv, 3600)):
            t = tt(2.4)
            env = np.exp(-t / 0.85) * np.minimum(1.0, t / 0.002)
            c = metal(t, base, hp_) * env
            c = _filt(c, lambda f_: _lp(f_, 9500, 2))
            kit[nom].append(nrm(c))
        t = tt(1.6)
        c = metal(t, 290 * dv, 1900, ratios=(1.0, 1.31, 1.83, 2.27, 2.9, 3.61))
        c = np.tanh(3.0 * c / (np.max(np.abs(c)) + 1e-9)) * np.exp(-t / 0.5)
        kit['china'].append(nrm(_filt(c, lambda f_: _hp(f_, 1500, 2))))
    return kit


DRUM_GAIN = {'k': 1.0, 's': 0.85, 't1': 0.72, 't2': 0.74, 't3': 0.78,
             'hh': 0.2, 'oh': 0.2, 'ride': 0.25, 'crash': 0.36, 'crash2': 0.36,
             'china': 0.3}
DRUM_PAN = {'k': 0.0, 's': 0.02, 't1': 0.3, 't2': -0.05, 't3': -0.35,
            'hh': 0.35, 'oh': 0.35, 'ride': -0.4, 'crash': 0.5, 'crash2': -0.55,
            'china': -0.3}


def _render_drums(events, n, kit, rng):
    bufs = {k: np.zeros(n) for k in kit}
    rr = {k: 0 for k in kit}
    for (t, ins, vel) in events:
        jit = 0.0015 if ins == 'k' else 0.0028
        s0 = int((t + rng.normal(0.0, jit)) * SR)
        if s0 >= n:
            continue
        s0 = max(0, s0)
        v = float(np.clip(vel * (1.0 + rng.normal(0.0, 0.06)), 0.05, 1.2))
        smp = kit[ins][(rr[ins] + int(rng.integers(0, 2))) % len(kit[ins])]
        rr[ins] += 1
        e = min(n, s0 + len(smp))
        bufs[ins][s0:e] += v * smp[:e - s0]
    out = np.zeros((n, 2))
    for k, b in bufs.items():
        if not b.any():
            continue
        out += _pan(b * DRUM_GAIN[k], DRUM_PAN[k])
    # compresion de bus + ligera saturacion para "pegada"
    out = _norm(out, -16.0)
    out = _comp(out, -19.0, 3.0, win_ms=6.0, rel_ms=60.0)
    out = np.tanh(out * 1.6) / 1.6
    return out


# ==========================================================================
# Guitarras, bajo y leads
# ==========================================================================
def _cab(f):
    """Respuesta tipo bafle 4x12 (golpe de graves, medios recortados,
    presencia y caida fuerte por arriba de ~5 kHz)."""
    return (_hp(f, 80, 2) * _lp(f, 4800, 4) * _bell(f, 115, 3.0, 0.5)
            * _bell(f, 430, -5.0, 0.7) * _bell(f, 2300, 4.0, 0.5))


def _render_ritmo(events, n, rng, off, detunes, gain):
    """Guitarra ritmica: osciladores desafinados + pua + palm-mute,
    distorsion asimetrica en dos etapas y simulacion de bafle."""
    pm = np.zeros(n)
    op = np.zeros(n)
    rel = int(0.05 * SR)
    for (t, dur, notes, kind, vel) in events:
        s0 = int((t + off + rng.normal(0.0, 0.0018)) * SR)
        if s0 >= n:
            continue
        s0 = max(0, s0)
        v = vel * (1.0 + rng.normal(0.0, 0.05))
        if kind == 'pm':
            L = int(min(dur, 0.32) * SR) + int(0.012 * SR)
        else:
            L = int(dur * SR) + rel
        L = min(L, n - s0)
        if L <= 8:
            continue
        tt = np.arange(L) / SR
        # la pua sube un poco la afinacion en el ataque
        cb = tt + 0.006 * 0.012 * (1.0 - np.exp(-tt / 0.012))
        sig = np.zeros(L)
        for m in notes:
            f = 440.0 * 2.0 ** ((m - 69) / 12.0)
            for dc in detunes:
                fd = f * 2.0 ** ((dc + rng.normal(0, 1.5)) / 1200.0)
                sig += _saw(fd * cb + rng.random(), fd / SR)
        sig /= np.sqrt(len(notes)) * len(detunes)
        if kind == 'pm':
            env = np.minimum(1.0, tt / 0.0015) * np.exp(-tt / 0.085)
            k = min(L, int(0.012 * SR))
            env[L - k:] *= np.linspace(1.0, 0.0, k)
            dst = pm
        else:
            env = np.minimum(1.0, tt / 0.003) * (0.7 + 0.3 * np.exp(-tt / 0.25))
            Ld = min(L, int(dur * SR))
            if L > Ld:
                env[Ld:] *= np.linspace(1.0, 0.0, L - Ld)
            dst = op
        pick = rng.standard_normal(L) * np.exp(-tt / 0.003) * 0.35
        dst[s0:s0 + L] += v * (sig * env + pick)
    # pre-EQ (palm-mute oscuro) en una sola FFT
    N = _nfft(n + SR // 4)
    f = np.fft.rfftfreq(N, 1.0 / SR)
    pre = _hp(f, 120, 1) * _bell(f, 750, 6.0, 0.8) * _lp(f, 5500, 1)
    X = np.fft.rfft(pm, N) * (_lp(f, 850, 2) * pre * 1.7) + np.fft.rfft(op, N) * pre
    x = np.fft.irfft(X, N)[:n]
    del X
    x /= _ref(x)
    # etapa 1: saturacion asimetrica fuerte
    b = 0.25
    x = np.tanh(12.0 * gain * x + b) - np.tanh(b)
    x = _filt(x, lambda f_: _lp(f_, 6500, 2) * _hp(f_, 60, 1))
    # etapa 2
    x = np.tanh(3.0 * x)
    return _filt(x, _cab)


def _render_bajo(events, n, rng):
    b = np.zeros(n)
    for (t, dur, m, kind, vel) in events:
        s0 = int((t + rng.normal(0.0, 0.001)) * SR)
        if s0 >= n:
            continue
        s0 = max(0, s0)
        if kind == 'pm':
            L = int((min(dur, 0.3) + 0.012) * SR)
        else:
            L = int((dur + 0.04) * SR)
        L = min(L, n - s0)
        if L <= 8:
            continue
        tt = np.arange(L) / SR
        f = 440.0 * 2.0 ** ((m - 69) / 12.0)
        ph = f * tt + rng.random()
        sig = 0.6 * _saw(ph, f / SR) + 0.7 * np.sin(2 * np.pi * ph)
        if kind == 'pm':
            env = np.minimum(1.0, tt / 0.002) * np.exp(-tt / 0.12)
        elif kind == 'open':
            env = np.minimum(1.0, tt / 0.003) * (0.75 + 0.25 * np.exp(-tt / 0.3))
        else:
            env = np.minimum(1.0, tt / 0.008) * np.exp(-tt / 1.4)
        k = min(L, int(0.02 * SR))
        env[L - k:] *= np.linspace(1.0, 0.0, k)
        b[s0:s0 + L] += vel * sig * env
    b = _filt(b, lambda f: _lp(f, 1400, 2))
    b /= _ref(b)
    b = np.tanh(2.2 * b)
    return _filt(b, lambda f: _hp(f, 35, 2) * _lp(f, 3000, 2) * _bell(f, 90, 3.0, 0.6)
                 * _bell(f, 900, 4.0, 0.6) * _bell(f, 250, -2.0, 0.6))


def _render_limpia(events, n, rng):
    """Guitarra limpia (arpegios): sintesis aditiva con armonicos que decaen
    mas rapido en lo agudo (como una cuerda pulsada) + chorus estereo."""
    out = np.zeros(n)
    if not events:
        return np.zeros((n, 2))
    for (t, dur, m, vel) in events:
        s0 = int((t + rng.normal(0.0, 0.003)) * SR)
        if s0 >= n:
            continue
        s0 = max(0, s0)
        L = min(int(min(dur + 0.5, 2.6) * SR), n - s0)
        if L <= 8:
            continue
        tt = np.arange(L) / SR
        f = 440.0 * 2.0 ** ((m - 69) / 12.0)
        sig = np.zeros(L)
        for k in range(1, 11):
            fk = f * k * (1.0 + 0.0004 * k * k)
            if fk > 9000:
                break
            amp = (1.0 / k ** 1.1) * (1.25 if k == 2 else 1.0)
            sig += amp * np.exp(-tt * (1.0 + 0.9 * k)) * np.sin(2 * np.pi * fk * tt
                                                                  + rng.random() * 6.28)
        sig *= np.minimum(1.0, tt / 0.002)
        k = min(L, int(0.05 * SR))
        sig[L - k:] *= np.linspace(1.0, 0.0, k)
        out[s0:s0 + L] += vel * sig
    out /= _ref(out)
    out = np.tanh(1.3 * out)
    out = _filt(out, lambda f: _hp(f, 90, 2) * _lp(f, 6000, 2) * _bell(f, 2500, 2.0, 0.7))
    # chorus: copia con retardo modulado en el canal derecho
    idx = np.arange(n, dtype=np.float64)
    tsec = idx / SR
    d = (0.011 + 0.0025 * np.sin(2 * np.pi * 0.7 * tsec)) * SR
    r = np.interp(idx - d, idx, out, left=0.0)
    d2 = (0.007 + 0.002 * np.sin(2 * np.pi * 0.53 * tsec + 1.3)) * SR
    l = np.interp(idx - d2, idx, out, left=0.0)
    return np.stack((0.75 * out + 0.45 * l, 0.75 * out + 0.45 * r), axis=1)


def _render_lead(events, n, tone, rng):
    """Lead monofonico continuo: glissandos/bends, vibrato, re-ataques."""
    if not events:
        return np.zeros(n)
    ev = sorted(events, key=lambda e: e[0])
    m = np.zeros(n)
    amp = np.zeros(n)
    vib = np.zeros(n)
    onsets = []
    rel = int(0.07 * SR)
    for (t, dur, mi, vel, gl, glt, vd) in ev:
        s0 = max(0, int((t + rng.normal(0.0, 0.002)) * SR))
        s1 = min(n, s0 + int(dur * SR))
        if s1 - s0 < 16:
            continue
        seg = s1 - s0
        tt = np.arange(seg) / SR
        mm = np.full(seg, float(mi))
        if gl:
            g = max(2, min(seg, int(glt * SR)))
            x = np.arange(g) / g
            mm[:g] += gl * (1.0 - np.sin(x * np.pi / 2.0))
        m[s0:s1] = mm
        a = vel * np.minimum(1.0, tt / 0.004) * (0.8 + 0.2 * np.exp(-tt / 0.15))
        amp[s0:s1] = a
        e2 = min(n, s1 + rel)
        if e2 > s1:
            amp[s1:e2] = np.maximum(amp[s1:e2], a[-1] * np.linspace(1.0, 0.0, e2 - s1))
            m[s1:e2] = mm[-1]
        dl = int(0.12 * SR)
        if vd and seg > dl + 16:
            vib[s0 + dl:s1] = vd * np.minimum(1.0, (np.arange(seg - dl) / SR) / 0.2)
        onsets.append(s0)
    # rellenar huecos de afinacion con el ultimo valor (sin saltos)
    idx = np.where(m > 0, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    m = m[idx]
    nz = np.nonzero(m)[0]
    if nz.size:
        m[m == 0] = m[nz[0]]
    m = _ma(m, int(0.005 * SR))
    tsec = np.arange(n) / SR
    lfo = np.sin(2 * np.pi * (5.6 * tsec + 0.15 * np.sin(2 * np.pi * 0.3 * tsec)))
    f = 440.0 * 2.0 ** ((m + vib * lfo - 69.0) / 12.0)
    dt = f / SR
    ph = np.cumsum(dt)
    if tone == 'eerie':
        sq = 0.5 * (_saw(ph, dt) - _saw(ph + 0.5, dt))
        sig = 0.6 * sq + 0.4 * _saw(ph * 1.006 + 0.2, dt * 1.006)
    else:
        sig = 0.55 * _saw(ph, dt) + 0.45 * _saw(ph * 1.004 + 0.3, dt * 1.004)
    sig *= amp
    # ataque de pua
    pk = np.zeros(n)
    L = int(0.006 * SR)
    for s0 in onsets:
        e = min(n, s0 + L)
        pk[s0:e] += np.exp(-np.arange(e - s0) / (0.0015 * SR))
    sig += pk * rng.standard_normal(n) * 0.25 * (amp > 0)
    # cadena de tono
    if tone == 'vocal':
        x = _filt(sig, lambda f_: _hp(f_, 200, 2) * _bell(f_, 900, 4.0, 0.8))
        x /= _ref(x)
        x = np.tanh(3.5 * x + 0.1) - np.tanh(0.1)
        x = _filt(x, lambda f_: _hp(f_, 150, 2) * _lp(f_, 4500, 3) * _bell(f_, 1400, 3.0, 0.6))
    elif tone == 'eerie':
        x = _filt(sig, lambda f_: _hp(f_, 250, 2) * _bell(f_, 1200, 5.0, 0.7))
        x /= _ref(x)
        x = np.tanh(6.0 * x + 0.1) - np.tanh(0.1)
        x = _filt(x, lambda f_: _hp(f_, 180, 2) * _lp(f_, 4200, 3) * _bell(f_, 2200, 3.0, 0.5))
    else:
        x = _filt(sig, lambda f_: _hp(f_, 250, 2) * _bell(f_, 1000, 6.0, 0.8))
        x /= _ref(x)
        x = np.tanh(8.0 * x + 0.1) - np.tanh(0.1)
        x = _filt(x, lambda f_: _lp(f_, 6500, 2))
        x = np.tanh(1.5 * x)
        x = _filt(x, lambda f_: _hp(f_, 140, 2) * _lp(f_, 5200, 3)
                  * _bell(f_, 1600, 3.0, 0.6) * _bell(f_, 400, -3.0, 0.7))
    return x


def _delay_pp(x, d, fb=0.32, mix=0.35):
    """Delay ping-pong vectorizado (3 repeticiones) -> estereo."""
    L = x.copy()
    R = x.copy()
    R += mix * _shift(x, d)
    L += mix * fb * 1.6 * _shift(x, 2 * d)
    R += mix * fb * fb * 1.6 * _shift(x, 3 * d)
    return np.stack((L, R), axis=1)


def _reverb(send, rng, dur=1.4, tau=0.36):
    """Reverb por convolucion FFT con respuesta de ruido decreciente."""
    n = len(send)
    m = int(dur * SR)
    t = np.arange(m) / SR
    ir = rng.standard_normal((m, 2)) * np.exp(-t / tau)[:, None]
    ir = _filt(ir, lambda f: _lp(f, 4500, 1) * _hp(f, 250, 1))
    ir[:int(0.014 * SR)] = 0.0
    ir /= np.sqrt((ir ** 2).sum(axis=0))[None, :]
    N = _nfft(n + m)
    S = np.fft.rfft(send, N)
    out = np.empty((n, 2))
    for c in range(2):
        out[:, c] = np.fft.irfft(S * np.fft.rfft(ir[:, c], N), N)[:n]
    return out


# ==========================================================================
# Composicion
# ==========================================================================
class _Arr:
    """Contenedor de eventos de una cancion (tiempos en segundos)."""

    def __init__(self, bpm):
        self.bpm = bpm
        self.step = 60.0 / bpm / 4.0
        self.bar = 16 * self.step
        self.gtr = []      # (t, dur, [midis], 'pm'/'open', vel)
        self.bass = []     # (t, dur, midi, kind, vel)
        self.clean = []    # (t, dur, midi, vel)
        self.lead1 = []    # (t, dur, midi, vel, glide, glide_t, vib)
        self.lead2 = []
        self.solo = []
        self.drums = []    # (t, inst, vel)
        self.secs = []     # (nombre, t0, t1)


def _fit(tpl, bpm):
    """Escala la plantilla de estructura para durar ~TARGET_S."""
    bar = 240.0 / bpm
    target = int(round((TARGET_S - TAIL_S) / bar))
    base = sum(b for _, b in tpl if _ != 'blast')
    fixed = sum(b for _, b in tpl if _ == 'blast')
    k = (target - fixed) / base
    secs = []
    for name, b in tpl:
        if name == 'blast':
            secs.append([name, b])
        else:
            secs.append([name, max(2, int(round(b * k / 2.0)) * 2)])
    flex = [i for i, (nm, _) in enumerate(secs)
            if nm.startswith(('chorus', 'verse', 'solo', 'breakdown'))]
    tot = sum(nb for _, nb in secs)
    j = 0
    guard = 0
    while tot > target + 1 and guard < 300:
        i = max(flex, key=lambda q: secs[q][1])
        if secs[i][1] <= 2:
            break
        secs[i][1] -= 2
        tot -= 2
        guard += 1
    while tot < target - 1 and guard < 300:
        i = flex[j % len(flex)]
        j += 1
        secs[i][1] += 2
        tot += 2
        guard += 1
    return [tuple(s) for s in secs]


def _token(ch, r, sc, motif):
    """Caracter de riff -> (notas MIDI, articulacion)."""
    if ch == 'x':
        return [r, r + 7], 'pm'
    if ch.isdigit():
        return [r + _deg(sc, int(ch))], 'pm'
    if ch in 'PQR':
        n = r + _deg(sc, motif['PQR'.index(ch)])
        return [n, n + 7, n + 12], 'open'
    if ch in 'pqr':
        return [r + _deg(sc, motif['pqr'.index(ch)])], 'pm'
    if 'A' <= ch <= 'J':
        n = r + _deg(sc, ord(ch) - 65)
        return [n, n + 7, n + 12], 'open'
    if 'a' <= ch <= 'j':
        n = r + _deg(sc, ord(ch) - 97)
        return [n, n + 7], 'pm'
    raise ValueError(ch)


def _gtr(A, song, t0, s, motif=(3, 5, 6), vel=1.0, bass=True, bass_vel=1.0):
    """Interpreta una cadena de pasos (16 por compas) y agrega guitarra+bajo.
    Devuelve el arreglo de ataques (para que el bombo siga a la guitarra)."""
    r = song['root']
    sc = SCALES[song['scale']]
    L = len(s)
    att = np.zeros(L, dtype=bool)
    i = 0
    while i < L:
        ch = s[i]
        if ch in '.-':
            i += 1
            continue
        j = i + 1
        while j < L and s[j] == '-':
            j += 1
        notes, kind = _token(ch, r, sc, motif)
        acc = 1.0 if i % 4 == 0 else 0.9
        if kind == 'open':
            acc *= 1.05
        t = t0 + i * A.step
        dur = (j - i) * A.step
        A.gtr.append((t, dur, notes, kind, vel * acc))
        if bass:
            bn = notes[0] - 12 if notes[0] - 12 >= 28 else notes[0]
            A.bass.append((t, dur, bn, kind, bass_vel * acc))
        att[i] = True
        i = j
    return att


def _prog_str(prog, nb, pat):
    s = []
    for b in range(nb):
        c = prog[b % len(prog)]
        c1, c2 = (c if isinstance(c, tuple) else (c, c))
        for i, ch in enumerate(pat):
            cc = c1 if i < 8 else c2
            if ch == '@':
                s.append(chr(65 + cc))
            elif ch == '#':
                s.append(chr(97 + cc))
            else:
                s.append(ch)
    return ''.join(s)


def _chords(prog, nb):
    out = []
    for b in range(nb):
        c = prog[b % len(prog)]
        out.append(c[0] if isinstance(c, tuple) else c)
    return out


def _riff_str(riff, nb):
    return (riff * (nb // 2 + 1))[:nb * 16]


def _drums(A, rng, t0, nb, cfg, att=None, fill=True, crash=True, build=False, dv=1.0):
    kp, sp = DR[cfg[0]]
    ci, cp = CYM[cfg[1]]
    ev = A.drums
    alt = 0
    for b in range(nb):
        tb = t0 + b * A.bar
        last = b == nb - 1
        fseq, ff = None, 16
        if last and fill and not build and cfg[0] != 'none':
            if rng.random() < 0.5:
                fseq, ff = FILLS8[int(rng.integers(len(FILLS8)))], 8
            else:
                fseq, ff = FILLS4[int(rng.integers(len(FILLS4)))], 12
        crash_here = (crash and b == 0) or (b > 0 and b % 8 == 0 and cfg[1] != 'none')
        for st in range(16):
            t = tb + st * A.step
            if fseq is not None and st >= ff:
                vv = 0.75 + 0.25 * (st - ff) / len(fseq)
                ev.append((t, fseq[st - ff], vv * dv))
                if st % 2 == 0:
                    ev.append((t, 'k', 0.75 * dv))
                continue
            if build and last:
                ev.append((t, 's', (0.3 + 0.7 * st / 15.0) * dv))
                if st % 4 == 0:
                    ev.append((t, 'k', 0.9 * dv))
                if st % 2 == 0 and ci:
                    ev.append((t, ci if ci != 'crash' else 'ride', 0.5 * dv))
                continue
            if kp == 'F':
                hit = att is not None and b * 16 + st < len(att) and att[b * 16 + st]
            else:
                hit = kp[st] == 'x'
            if hit or (crash_here and st == 0 and cfg[0] != 'none'):
                ev.append((t, 'k', (0.95 if st % 4 == 0 else 0.85) * dv))
            if sp[st] == 'x':
                vs = 0.72 if cfg[0] == 'blast' else (1.0 if st in (4, 12, 8) else 0.85)
                ev.append((t, 's', vs * dv))
            elif cfg[0] in ('rock', 'rock2', 'nu', 'clean', 'follow') and st in (7, 10, 15) \
                    and rng.random() < 0.22:
                ev.append((t, 's', 0.2 * dv))      # nota fantasma
            if cfg[0] == 'tribal':
                for tom, pt in TOMS_TRIBAL.items():
                    if pt[st] == 'x':
                        ev.append((t, tom, (0.9 if st % 4 == 0 else 0.75) * dv))
            if ci and cp[st] == 'x':
                ins = ci
                if ci == 'crash':
                    ins = 'crash' if (st // 4) % 2 == 0 else 'crash2'
                    vv = 0.62
                elif ci in ('hh', 'oh'):
                    vv = 0.8 if st % 4 == 0 else 0.55
                else:
                    vv = 0.75 if st % 4 == 0 else 0.6
                if not (crash_here and st == 0):
                    ev.append((t, ins, vv * dv))
            if crash_here and st == 0 and cfg[0] != 'none':
                ev.append((t, 'crash' if alt % 2 == 0 else 'crash2', 0.95 * dv))
                alt += 1


def _snap(d, c):
    """Tono del acorde (grado c) mas cercano al grado d."""
    best, bd = d, 1e9
    for o in (0, 2, 4):
        for k in (-1, 0, 1, 2):
            cand = c % 7 + o + 7 * k
            if abs(cand - d) < bd:
                best, bd = cand, abs(cand - d)
    return best


def _melody(A, rng, t0, chords, song, voice1, voice2=None, tone='lead',
            base_off=24, vel=0.9):
    """Melodia original por motivos (forma AABA de 2 compases), ajustada a los
    tonos del acorde en tiempos fuertes. voice2 = armonia en terceras."""
    sc = SCALES[song['scale']]
    r = song['root']
    n = len(chords)
    pool = MEL_VOC if tone == 'vocal' else MEL_R

    def frase():
        r1 = pool[int(rng.integers(len(pool)))]
        r2 = MEL_END[int(rng.integers(len(MEL_END)))]
        nn = r1.count('n') + r2.count('n')
        cont = rng.choice([-2, -1, -1, 1, 1, 2, 0, -3, 3], size=nn)
        return (r1, r2), cont

    fa, fb = frase(), frase()
    plan = 'AABA'
    d = 7
    notas = []
    for g in range(0, n, 2):
        (rr, cont) = fa if plan[(g // 2) % 4] == 'A' else fb
        d = _snap(d, chords[g])
        k = 0
        for b in range(2):
            if g + b >= n:
                break
            cb = chords[g + b]
            pat = rr[b]
            for st, ch in enumerate(pat):
                if ch != 'n':
                    continue
                if k > 0:
                    d = d + int(cont[k % len(cont)])
                if st in (0, 8):
                    d = _snap(d, cb)
                while d > 13:
                    d -= 2
                while d < 2:
                    d += 2
                j = st + 1
                while j < 16 and pat[j] == '-':
                    j += 1
                notas.append([t0 + (g + b) * A.bar + st * A.step, (j - st) * A.step, d])
                k += 1
    if notas:
        notas[-1][2] = min((0, 7, 14), key=lambda q: abs(q - notas[-1][2]))
        if notas[-1][2] == 0:
            notas[-1][2] = 7
    for (t, dur, dd) in notas:
        larga = dur > 0.3
        gl = 0.0
        if larga and rng.random() < 0.25:
            gl = -2.0 if tone == 'lead' else -1.0
        vd = (0.3 if tone != 'vocal' else 0.22) if larga else 0.0
        voice1.append((t, dur * 0.97, r + base_off + _deg(sc, dd), vel, gl, 0.09, vd))
        if voice2 is not None:
            voice2.append((t, dur * 0.97, r + base_off + _deg(sc, dd + 2), vel * 0.85,
                           gl, 0.09, vd))


def _eerie(A, rng, t0, nb, song, voice1, voice2, vel=0.75):
    """Lead disonante (nu-metal): notas largas, glissandos lentos, tritonos."""
    r = song['root']
    base = r + 36 if r < 38 else r + 24
    for g in range(0, nb, 2):
        t = t0 + g * A.bar
        if rng.random() < 0.5:
            partes = [(0.0, 1.5)]
        else:
            partes = [(0.0, 0.75), (1.0, 0.9)]
        for (o, ln) in partes:
            if g * 1.0 + o >= nb:
                continue
            mi = base + int(rng.choice([0, 1, 6, 7, 12, 13, -2]))
            gl = float(rng.choice([-3, -2, 0, 2]))
            dur = min(ln, nb - g - o) * A.bar
            voice1.append((t + o * A.bar, dur, mi, vel, gl, 0.3, 0.45))
            if rng.random() < 0.6:
                voice2.append((t + o * A.bar, dur, mi + int(rng.choice([6, 1, -1])),
                               vel * 0.6, gl, 0.3, 0.4))


def _solo(A, rng, t0, chords, song, voice, base_off=24):
    """Solo estilo 'shred': bends, escalas en secuencias, arpegios barridos."""
    sc = SCALES[song['scale']]
    r = song['root']
    npb = 4 if A.bpm >= 125 else 6
    beat = 4 * A.step
    nb = len(chords)
    planes = ['bend', 'down', 'lick', 'up', 'sweep', 'trill', 'down', 'final']
    if nb > 8:
        planes = ['bend', 'lick', 'down', 'up', 'trill', 'sweep', 'down', 'bend',
                  'lick', 'up', 'sweep', 'down', 'trill', 'sweep', 'up', 'final']

    def add(t, dur, d, vel=0.9, gl=0.0, glt=0.06, vib=0.0):
        while d > 15:
            d -= 7
        while d < -2:
            d += 7
        voice.append((t, dur * 0.96, r + base_off + _deg(sc, d), vel, gl, glt, vib))

    for b, c in enumerate(chords):
        tb = t0 + b * A.bar
        kind = 'final' if b == nb - 1 else planes[b % len(planes)]
        hi = _snap(11, c)
        nn = 4 * npb
        sd = beat / npb
        if kind == 'bend':
            add(tb, 2 * beat, hi, gl=-2.0, glt=0.14, vib=0.45)
            for k in range(4):
                add(tb + 2 * beat + k * beat / 2, beat / 2, hi - 1 - k, 0.85)
        elif kind == 'down':
            d0 = _snap(14, c)
            seq = []
            g = 0
            while len(seq) < nn:
                seq += [d0 - g, d0 - g - 1, d0 - g - 2, d0 - g - 3]
                g += 1
            for k in range(nn):
                add(tb + k * sd, sd, seq[k], 0.95 if k % npb == 0 else 0.8)
        elif kind == 'up':
            d0 = _snap(4, c)
            seq = []
            g = 0
            while len(seq) < nn:
                seq += [d0 + g, d0 + g + 1, d0 + g + 2]
                g += 1
            for k in range(nn):
                add(tb + k * sd, sd, seq[k], 0.95 if k % npb == 0 else 0.8)
        elif kind == 'lick':
            for k in range(3 * npb):
                add(tb + k * sd, sd, hi + [2, 1, 0][k % 3], 0.9 if k % 3 == 0 else 0.8)
            add(tb + 3 * beat, beat, hi - 1, 0.9, gl=-1.0, vib=0.35)
        elif kind == 'sweep':
            tones = [c, c + 2, c + 4, c + 7, c + 9, c + 11, c + 14]
            seq = tones + tones[-2:0:-1]
            for k in range(nn):
                add(tb + k * sd, sd, seq[k % len(seq)], 0.85)
        elif kind == 'trill':
            for k in range(2 * npb):
                add(tb + k * sd, sd, hi + (k % 2), 0.85)
            add(tb + 2 * beat, 2 * beat, hi + 2, 0.95, gl=-2.0, glt=0.12, vib=0.5)
        else:  # final
            add(tb, A.bar * 0.98, 14 if hi > 10 else 7, 1.0, gl=-1.0, glt=0.1, vib=0.55)


def _arp(A, song, t0, chords, rng, vel=0.8, bass=True):
    """Arpegio de guitarra limpia en corcheas + bajo en redondas."""
    sc = SCALES[song['scale']]
    r = song['root']
    pats = [[0, 1, 2, 3, 4, 3, 2, 1], [0, 2, 1, 3, 2, 5, 4, 2], [0, 1, 3, 1, 5, 1, 3, 1]]
    pat = pats[int(rng.integers(len(pats)))]
    for b, c in enumerate(chords):
        tones = [c, c + 4, c + 7, c + 9, c + 11, c + 8]
        for k in range(8):
            d = tones[pat[k]]
            A.clean.append((t0 + b * A.bar + k * 2 * A.step, 6 * A.step,
                            r + 12 + _deg(sc, d), vel * (1.0 if k % 2 == 0 else 0.82)))
        if bass:
            bn = r + _deg(sc, c)
            bn = bn - 12 if bn - 12 >= 28 else bn
            A.bass.append((t0 + b * A.bar, A.bar * 0.95, bn, 'soft', 0.75))


def _compose(song):
    est = ESTILOS[song['tpl']]
    rng = np.random.default_rng(song['seed'])
    A = _Arr(song['bpm'])
    secs = _fit(TPL[song['tpl']], song['bpm'])
    P = song['prog']
    tone = est['lead']
    harm = est['harm']

    def mrng(tag):
        # semilla fija por "tema" -> el coro suena igual cada vez que vuelve
        return np.random.default_rng(song['seed'] * 1000 + tag)

    def motif():
        m = rng.choice([1, 2, 3, 4, 5, 6], 3, replace=False)
        return tuple(int(v) for v in m)

    nombres = list(est['riffs'])
    rng.shuffle(nombres)
    R = {
        'verse': (RIFFS[nombres[0]], motif()),
        'verse2': (RIFFS[nombres[1 % len(nombres)]], motif()),
        'intro': (RIFFS[nombres[2 % len(nombres)]], motif()),
        'pre': (RIFFS[nombres[3 % len(nombres)]], motif()),
        'bd': (RIFFS[est['bd'][int(rng.integers(len(est['bd'])))]], motif()),
    }
    n_chorus = sum(1 for nm, _ in secs if nm == 'chorus')
    k_chorus = 0
    t = 0.0
    for si, (name, nb) in enumerate(secs):
        nxt = secs[si + 1][0] if si + 1 < len(secs) else None
        fill = nxt is not None
        t0 = t
        h = nb // 2

        if name == 'intro_lead':
            s = _prog_str(P['chorus'], nb, CPAT['half'])
            att = _gtr(A, song, t0, s, vel=0.95)
            _drums(A, rng, t0, h, ('none', 'none'), att, fill=False, crash=False)
            _drums(A, rng, t0 + h * A.bar, nb - h, est['verse_dr'], att[h * 16:], fill=fill)
            _melody(A, mrng(1), t0, _chords(P['chorus'], nb), song, A.lead1,
                    A.lead2 if harm else None, tone)

        elif name == 'intro_riff':
            riff, mot = R['intro']
            s = _riff_str(riff, nb)
            att = _gtr(A, song, t0, s, mot, bass=False)
            _gtr_b = 2 if nb > 2 else 1
            _drums(A, rng, t0, _gtr_b, ('none', 'none'), fill=False, crash=False)
            if nb - _gtr_b > 0:
                # bajo y bateria entran despues
                for ev in [e for e in A.gtr if e[0] >= t0 + _gtr_b * A.bar - 1e-6]:
                    bn = ev[2][0] - 12 if ev[2][0] - 12 >= 28 else ev[2][0]
                    A.bass.append((ev[0], ev[1], bn, ev[3], ev[4]))
                _drums(A, rng, t0 + _gtr_b * A.bar, nb - _gtr_b, est['verse_dr'],
                       att[_gtr_b * 16:], fill=fill)

        elif name == 'intro_perc':
            s = ("x.......x......." * nb)
            att = _gtr(A, song, t0, s, vel=0.9)
            _drums(A, rng, t0, nb, ('tribal', 'none'), att, fill=fill, crash=False)
            _eerie(A, rng, t0, nb, song, A.lead1, A.lead2)

        elif name == 'intro_clean':
            ch = _chords(P['verse'], nb)
            _arp(A, song, t0, ch, rng, vel=0.85, bass=False)
            _drums(A, rng, t0, h, ('none', 'none'), fill=False, crash=False)
            _drums(A, rng, t0 + h * A.bar, nb - h, ('clean', 'ride8'), fill=fill, dv=0.7)
            for b in range(h, nb):
                bn = song['root'] + _deg(SCALES[song['scale']], ch[b])
                A.bass.append((t0 + b * A.bar, A.bar * 0.95,
                               bn - 12 if bn - 12 >= 28 else bn, 'soft', 0.7))

        elif name in ('verse', 'verse2'):
            if name == 'verse':
                riff, mot = R['verse']
                s = _riff_str(riff, nb)
            else:
                r1, m1 = R['verse']
                r2, m2 = R['verse2']
                s1 = _riff_str(r1, h)
                s2 = _riff_str(r2, nb - h)
                att1 = _gtr(A, song, t0, s1, m1)
                att2 = _gtr(A, song, t0 + h * A.bar, s2, m2)
                att = np.concatenate((att1, att2))
                s = None
            if s is not None:
                att = _gtr(A, song, t0, s, mot)
            cfg = est['verse_dr'] if name == 'verse' else est['verse2_dr']
            _drums(A, rng, t0, nb, cfg, att, fill=fill)
            if tone == 'eerie':
                _eerie(A, rng, t0 + h * A.bar, nb - h, song, A.lead1, A.lead2, vel=0.6)
            elif song['tpl'] == 'melodic' and name == 'verse2':
                _melody(A, mrng(6), t0 + h * A.bar, _chords(P['verse'], nb - h), song,
                        A.lead1, A.lead2, tone, vel=0.75)

        elif name in ('verse_clean', 'verse_clean2'):
            ch = _chords(P['verse'], nb)
            if name == 'verse_clean':
                _arp(A, song, t0, ch, rng)
                _drums(A, rng, t0, nb, ('clean', 'hat8'), fill=fill, dv=0.75)
            else:
                _arp(A, song, t0, ch[:h], rng)
                _drums(A, rng, t0, h, ('clean', 'hat8'), fill=True, dv=0.75)
                riff, mot = R['verse']
                att = _gtr(A, song, t0 + h * A.bar, _riff_str(riff, nb - h), mot, vel=0.9)
                _drums(A, rng, t0 + h * A.bar, nb - h, est['verse_dr'], att, fill=fill)
            _melody(A, mrng(5), t0, ch, song, A.lead1, None, tone, base_off=24, vel=0.8)

        elif name == 'pre':
            if est['pre_pat'] == 'riff':
                riff, mot = R['pre']
                s = _riff_str(riff, nb)
                att = _gtr(A, song, t0, s, mot)
            else:
                s = _prog_str(P['pre'], nb, CPAT[est['pre_pat']])
                att = _gtr(A, song, t0, s)
            _drums(A, rng, t0, nb, est['pre_dr'], att, fill=False, build=fill)
            if tone == 'vocal':
                _melody(A, mrng(3), t0, _chords(P['pre'], nb), song, A.lead1, None, tone)
            elif tone == 'eerie':
                _eerie(A, rng, t0, nb, song, A.lead1, A.lead2, vel=0.65)

        elif name == 'chorus':
            k_chorus += 1
            final = k_chorus == n_chorus
            s = _prog_str(P['chorus'], nb, CPAT[est['chorus_pat']])
            att = _gtr(A, song, t0, s)
            _drums(A, rng, t0, nb, est['chorus_dr'], att, fill=fill)
            v2 = A.lead2 if (harm or final) else None
            _melody(A, mrng(2), t0, _chords(P['chorus'], nb), song, A.lead1, v2, tone,
                    base_off=24 if tone != 'eerie' else 24)

        elif name == 'interlude':
            riff, mot = R['intro']
            att = _gtr(A, song, t0, _riff_str(riff, nb), mot)
            _drums(A, rng, t0, nb, est['verse2_dr'], att, fill=fill)
            _melody(A, mrng(4), t0, _chords(P['verse'], nb), song, A.lead1, A.lead2, tone)

        elif name == 'breakdown':
            riff, mot = R['bd']
            att = _gtr(A, song, t0, _riff_str(riff, nb), mot)
            _drums(A, rng, t0, h, ('half', 'china4'), att, fill=False)
            _drums(A, rng, t0 + h * A.bar, nb - h, ('follow', 'china4'), att[h * 16:],
                   fill=fill, crash=True)
            if tone == 'eerie':
                _eerie(A, rng, t0, nb, song, A.lead1, A.lead2, vel=0.7)

        elif name == 'solo':
            s = _prog_str(P['solo'], nb, CPAT[est['solo_pat']])
            att = _gtr(A, song, t0, s, vel=0.92)
            _drums(A, rng, t0, nb, est['solo_dr'], att, fill=fill)
            _solo(A, mrng(7), t0, _chords(P['solo'], nb), song, A.solo)

        elif name == 'blast':
            s = _prog_str(P['pre'], nb, CPAT['trem'])
            att = _gtr(A, song, t0, s, vel=0.9)
            _drums(A, rng, t0, nb, ('blast', 'crash4'), att, fill=False)
            _eerie(A, rng, t0, nb, song, A.lead1, A.lead2, vel=0.7)

        elif name == 'outro':
            s = _prog_str(P['chorus'], nb - 1, CPAT['half']) + 'A' + '-' * 15
            att = _gtr(A, song, t0, s)
            _drums(A, rng, t0, nb - 1, (est['chorus_dr'][0], 'crash4'), att, fill=True)
            tf = t0 + (nb - 1) * A.bar
            A.drums += [(tf, 'k', 1.0), (tf, 'crash', 1.0), (tf, 'crash2', 0.9),
                        (tf, 's', 0.9)]
            # el ultimo acorde suena hasta el final
            tg = A.gtr[-1]
            A.gtr[-1] = (tg[0], tg[1] + TAIL_S, tg[2], tg[3], tg[4])
            tb = A.bass[-1]
            A.bass[-1] = (tb[0], tb[1] + TAIL_S, tb[2], tb[3], tb[4])
            sc = SCALES[song['scale']]
            A.lead1.append((tf, A.bar + TAIL_S - 0.6, song['root'] + 24 + _deg(sc, 7),
                            0.9, -2.0, 0.15, 0.5))
            if tone != 'vocal':
                A.lead2.append((tf, A.bar + TAIL_S - 0.6, song['root'] + 24 + _deg(sc, 9),
                                0.75, -2.0, 0.15, 0.5))
        else:
            raise ValueError(name)

        t = t0 + nb * A.bar
        A.secs.append((name, t0, t))
    A.total = t
    return A


# ==========================================================================
# Render y mezcla
# ==========================================================================
def render_song(song, return_arr=False):
    """Sintetiza una cancion completa. Devuelve array (n, 2) float en [-1, 1]."""
    est = ESTILOS[song['tpl']]
    A = _compose(song)
    n = int((A.total + TAIL_S) * SR)
    rng = np.random.default_rng(song['seed'] + 99)

    # --- guitarras ritmicas dobladas (L/R con distinta afinacion y tiempo)
    g = est['gain']
    gL = _render_ritmo(A.gtr, n, rng, 0.0, (-9.0, 0.0, 8.0), g)
    gR = _render_ritmo(A.gtr, n, rng, 0.004, (-7.0, 3.0, 10.0), g)
    gtr = _norm(np.stack((gL * 1.0 + gR * 0.22, gR * 1.0 + gL * 0.22), axis=1), -15.5)
    del gL, gR

    bajo = _norm(_pan(_render_bajo(A.bass, n, rng), 0.0), -20.0)

    kit = _make_kit(np.random.default_rng(song['seed'] + 7))
    dr = _norm(_render_drums(A.drums, n, kit, rng), -16.0)

    tone = est['lead']
    d = int(0.75 * 60.0 / song['bpm'] * SR)     # delay de corchea con puntillo
    l1 = _render_lead(A.lead1, n, tone, rng)
    l2 = _render_lead(A.lead2, n, tone, rng)
    so = _render_lead(A.solo, n, 'lead' if tone != 'eerie' else 'lead', rng)
    leads = np.zeros((n, 2))
    send = np.zeros(n)
    if A.lead1:
        x = _norm(l1, -21.0)
        leads += _pan(x, -0.2) * 0.75 + _delay_pp(x, d) * 0.25
        send += 0.35 * x
    if A.lead2:
        x = _norm(l2, -24.0)
        leads += _pan(x, 0.3) * 0.75 + _delay_pp(x, d)[:, ::-1] * 0.25
        send += 0.35 * x
    if A.solo:
        x = _norm(so, -19.5)
        leads += _pan(x, 0.0) * 0.8 + _delay_pp(x, d) * 0.2
        send += 0.3 * x
    del l1, l2, so

    limpia = _render_limpia(A.clean, n, rng)
    if A.clean:
        limpia = _norm(limpia, -20.0)
        send += 0.45 * limpia.mean(axis=1)

    send += 0.12 * dr.mean(axis=1) + 0.04 * gtr.mean(axis=1)
    rev = _reverb(send, np.random.default_rng(song['seed'] + 3))

    mix = gtr + bajo + dr + leads + limpia + 0.32 * rev
    del gtr, bajo, dr, leads, limpia, rev

    # --- master: DC fuera, EQ suave, compresion de "pegamento", limitador
    mix = _filt(mix, lambda f: _hp(f, 28, 2) * _bell(f, 3200, 1.5, 1.0) * _bell(f, 50, -2.0, 0.6)
                * _bell(f, 300, -1.0, 0.8))
    mix = _norm(mix, -16.0)
    mix = _comp(mix, -18.0, 2.2, win_ms=25.0, rel_ms=150.0)
    mix = _norm(mix, MASTER_RMS_DB)
    mix = _limiter(mix)
    mix = _limiter(mix)
    pk = np.max(np.abs(mix))
    if pk > 0.95:
        mix *= 0.95 / pk
    # fundidos
    fi = int(0.01 * SR)
    mix[:fi] *= np.linspace(0.0, 1.0, fi)[:, None]
    fo = int(2.5 * SR)
    mix[-fo:] *= (0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, fo)))[:, None]
    mix -= mix.mean(axis=0)[None, :]
    if return_arr:
        return mix, A
    return mix


def write_wav(path, stereo):
    """Escribe WAV PCM 16 bits estereo con el modulo `wave`."""
    x = np.clip(stereo, -1.0, 1.0)
    data = np.round(x * 32767.0).astype('<i2')
    tmp = path + '.tmp'
    with wave.open(tmp, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    os.replace(tmp, path)


def generate_all(out_dir, force=False, progress=None):
    """Genera todas las canciones de SONGS en out_dir (se crea si no existe).
    Omite archivos existentes salvo force=True. progress(i, n, titulo)."""
    os.makedirs(out_dir, exist_ok=True)
    rutas = []
    total = len(SONGS)
    for i, song in enumerate(SONGS):
        ruta = os.path.join(out_dir, song['file'])
        if progress:
            progress(i, total, song['title'])
        if force or not os.path.exists(ruta):
            write_wav(ruta, render_song(song))
        rutas.append(ruta)
    if progress:
        progress(total, total, '')
    return rutas


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'music_metal')
    forzar = '--force' in sys.argv
    t_ini = time.time()
    t_prev = [t_ini]

    def _prog(i, n, titulo):
        ahora = time.time()
        if i > 0:
            print(f'   listo ({ahora - t_prev[0]:.1f} s)')
        t_prev[0] = ahora
        if i < n:
            print(f'[{i + 1}/{n}] {titulo} ...', flush=True)

    generate_all(destino, force=forzar, progress=_prog)
    print(f'Total: {time.time() - t_ini:.1f} s  ->  {destino}')

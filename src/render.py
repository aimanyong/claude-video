"""Executive boardroom render: framing, camera moves, grade, motion graphics.

usage:
  python3 render.py SOURCE_VIDEO FACES_JSON MIX_WAV OUT_MP4
  python3 render.py SOURCE_VIDEO FACES_JSON --stills OUT_DIR t1 t2 ...   (output-timeline seconds)
"""
import sys, json, subprocess, math, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from config import *

SW, SH = 910, 512
TOTAL = INTRO + SRC_DUR + OUTRO
NFR = int(round(TOTAL * FPS))

NAVY = (11, 26, 46)
NAVY_D = (6, 13, 24)
GOLD = (201, 164, 92)
IVORY = (242, 238, 230)

FD = "/usr/share/fonts/opentype/inter/"
def F(name, size): return ImageFont.truetype(FD + name, size)

def ease(x):  # ease-out cubic, clipped
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3

def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)

# ------------------------------------------------------------------ sprites
def text_sprite(text, font, color, tracking=0.0):
    """RGBA float32 premultiplied sprite with manual letter-spacing."""
    asc, desc = font.getmetrics()
    w = 0.0
    for ch in text:
        w += font.getlength(ch) + tracking
    w = int(math.ceil(w - tracking)) + 8
    h = asc + desc + 8
    im = Image.new("RGBA", (w, h), color + (0,))
    d = ImageDraw.Draw(im)
    x = 4.0
    for ch in text:
        d.text((x, 4), ch, font=font, fill=color + (255,))
        x += font.getlength(ch) + tracking
    a = np.asarray(im).astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a

def rect_sprite(w, h, color, alpha=1.0):
    a = np.zeros((max(1, h), max(1, w), 4), np.float32)
    a[..., :3] = np.array(color, np.float32) / 255 * alpha
    a[..., 3] = alpha
    return a

def hgrad_sprite(w, h, color, a0, a1, power=1.6):
    x = np.linspace(0, 1, w, dtype=np.float32)
    al = a0 + (a1 - a0) * (x ** power)
    al = np.broadcast_to(al[None, :], (h, w)).copy()
    # soften top/bottom edges
    y = np.linspace(0, 1, h, dtype=np.float32)
    edge = np.clip(np.minimum(y, 1 - y) / 0.32, 0, 1)[:, None]
    edge = edge * edge * (3 - 2 * edge)
    al *= edge
    s = np.zeros((h, w, 4), np.float32)
    s[..., 3] = al
    s[..., :3] = np.array(color, np.float32)[None, None, :] / 255 * al[..., None]
    return s

def comp(frame, spr, x, y, alpha=1.0, clip_w=None):
    """Composite premultiplied sprite onto float frame at (x, y)."""
    if alpha <= 0.003: return
    x, y = int(round(x)), int(round(y))
    h, w = spr.shape[:2]
    if clip_w is not None: w = min(w, int(clip_w))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1]
    a = s[..., 3:4] * alpha
    roi *= (1 - a)
    roi += s[..., :3] * alpha

# ------------------------------------------------------------------ static layers
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
rr = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.45) / (H * 0.70)) ** 2)
VIGNETTE = (1 - 0.42 * np.clip(rr - 0.45, 0, 1) ** 1.6)[..., None].astype(np.float32)
TOPGRAD = (np.clip(1 - yy / 230, 0, 1) ** 1.6 * 0.42)[..., None].astype(np.float32)
VIG_TINT = (np.clip(rr - 0.55, 0, 1) ** 1.8 * 0.10)[..., None] * (np.array(NAVY, np.float32) / 255)

def card_background():
    g = np.clip(rr * 0.9, 0, 1)[..., None]
    c0 = np.array((18, 34, 58), np.float32) / 255
    c1 = np.array(NAVY_D, np.float32) / 255
    bg = c0 * (1 - g) + c1 * g
    # fine noise to avoid banding
    bg += (np.random.default_rng(7).standard_normal((H, W, 1)).astype(np.float32) * 0.004)
    return bg.astype(np.float32)
CARD_BG = card_background()

# drifting geometric lines (very low contrast) for title / end cards
def line_field():
    im = Image.new("L", (W + 400, H), 0)
    d = ImageDraw.Draw(im)
    for i in range(-30, 60):
        x = i * 90
        d.line([(x, H), (x + H * 0.58, 0)], fill=255, width=1)
    a = np.asarray(im).astype(np.float32) / 255
    return a
LINES = line_field()
LINE_MASK = np.clip((rr - 0.25) * 1.2, 0, 1)[..., None]  # lines only toward the edges

def corner_brackets(frame, p, inset=70, arm=70, alpha=1.0):
    L = int(arm * ease(p))
    if L < 2: return
    g = rect_sprite(L, 2, GOLD)
    v = rect_sprite(2, L, GOLD)
    for (x, y, sx, sy) in [(inset, inset, 1, 1), (W - inset, inset, -1, 1),
                           (inset, H - inset, 1, -1), (W - inset, H - inset, -1, -1)]:
        comp(frame, g, x if sx > 0 else x - L, y if sy > 0 else y - 2, alpha * 0.9)
        comp(frame, v, x if sx > 0 else x - 2, y if sy > 0 else y - L, alpha * 0.9)

# ------------------------------------------------------------------ text assets
T = {}
T["kicker"] = text_sprite(TITLE_KICKER, F("Inter-SemiBold.otf", 22), GOLD, 7)
T["title"] = text_sprite(TITLE_MAIN, F("InterDisplay-Light.otf", 82), IVORY, 0.5)
T["tsub"] = text_sprite(TITLE_SUB, F("Inter-Medium.otf", 20), IVORY, 6)
T["name"] = text_sprite(PRESENTER, F("InterDisplay-SemiBold.otf", 46), IVORY, 0.3)
T["role"] = text_sprite(PRESENTER_ROLE, F("Inter-SemiBold.otf", 19), GOLD, 4.5)
T["bug"] = text_sprite("EXECUTIVE BRIEFING", F("Inter-SemiBold.otf", 15), IVORY, 4.5)
T["end_main"] = text_sprite(END_MAIN, F("InterDisplay-Light.otf", 92), IVORY, 1)
T["end_sub"] = text_sprite(END_SUB, F("InterDisplay-Light.otf", 30), IVORY, 0.3)
T["end_name"] = text_sprite(PRESENTER, F("InterDisplay-Medium.otf", 26), IVORY, 0.3)
T["end_role"] = text_sprite(PRESENTER_ROLE, F("Inter-SemiBold.otf", 15), GOLD, 4)

f_label = F("Inter-SemiBold.otf", 19)
f_main = F("InterDisplay-Light.otf", 50)
f_sub = F("InterDisplay-Light.otf", 26)
CALL = []
for (s, e, lab, main, sub) in CALLOUTS + [(CLOSING[0], CLOSING[1], CLOSING[2], CLOSING[3], None)]:
    CALL.append(dict(s=s, e=e, lab=text_sprite(lab, f_label, GOLD, 4.5),
                     main=text_sprite(main, f_main, IVORY, 0.2),
                     sub=text_sprite(sub, f_sub, IVORY, 0.2) if sub else None))

AG_LABEL = text_sprite("THREE QUESTIONS", F("Inter-SemiBold.otf", 19), GOLD, 4.5)
AG_ITEMS = [(t, text_sprite(f"{i+1:02d}", F("InterDisplay-Light.otf", 40), GOLD, 1),
             text_sprite(txt, F("InterDisplay-Light.otf", 52), IVORY, 0.2))
            for i, (t, txt) in enumerate(AGENDA[2])]

CH = []
for (s, num, title) in CHAPTERS:
    CH.append(dict(s=s, num=text_sprite(num, F("InterDisplay-Thin.otf", 170), GOLD, 2),
                   title=text_sprite(title, F("Inter-SemiBold.otf", 40), IVORY, 9),
                   small=text_sprite(f"{num}   {title}", F("Inter-SemiBold.otf", 15), IVORY, 4.5)))

LEFT_GRAD = hgrad_sprite(1000, H, NAVY_D, 0.78, 0.0, power=0.9)

# ------------------------------------------------------------------ grade (on source frames)
def build_luts():
    x = np.arange(256, dtype=np.float32) / 255
    s = x * x * (3 - 2 * x)
    base = 0.74 * x + 0.26 * s                 # gentle S contrast
    base = np.clip(base * 1.03 - 0.012, 0, 1)  # slightly deeper blacks, a touch more exposure
    knee = 0.72                                 # soft highlight roll-off (bright ceiling / window)
    hi = base > knee
    base[hi] = knee + (1 - knee) * np.tanh((base[hi] - knee) / (1 - knee) * 1.1) / np.tanh(1.1) * 0.93
    r = base + 0.018 * base ** 2 - 0.012 * (1 - base) ** 3   # warm highs, cool shadows
    g = base + 0.004 * base ** 2
    b = base - 0.020 * base ** 2 + 0.028 * (1 - base) ** 3   # navy shadows
    return [np.clip(c * 255 + 0.5, 0, 255).astype(np.uint8) for c in (r, g, b)]
LUTS = build_luts()

def grade(rgb):
    out = np.dstack([cv2.LUT(rgb[..., i], LUTS[i]) for i in range(3)])
    f = out.astype(np.float32)
    lum = f @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    f = lum[..., None] + (f - lum[..., None]) * 0.93   # restrained saturation
    return np.clip(f, 0, 255).astype(np.uint8)

# ------------------------------------------------------------------ camera
def gaussian_smooth(v, sigma_samples):
    r = int(sigma_samples * 3)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2); k /= k.sum()
    p = np.pad(v, r, mode="edge")
    return np.convolve(p, k, mode="valid")

def build_camera(faces_path):
    fr = json.load(open(faces_path))
    ft = np.array([r[0] for r in fr])
    fx = np.array([np.nan if r[1] is None else r[1] for r in fr])
    fy = np.array([np.nan if r[2] is None else r[2] for r in fr])
    ok = ~np.isnan(fx)
    fx = np.interp(ft, ft[ok], fx[ok]); fy = np.interp(ft, ft[ok], fy[ok])
    # reject outliers vs. local median
    from scipy.ndimage import median_filter
    fx = median_filter(fx, 9, mode="nearest"); fy = median_filter(fy, 9, mode="nearest")
    rate = 1 / (ft[1] - ft[0])
    fx = gaussian_smooth(fx, 1.4 * rate); fy = gaussian_smooth(fy, 1.4 * rate)
    ts = np.arange(int(SRC_DUR * FPS) + 2) / FPS
    cx = np.interp(ts, ft, fx); cy = np.interp(ts, ft, fy)
    # zoom: step targets smoothed with gaussian, plus a slow living drift
    zt = np.zeros_like(ts)
    for i, (t0, z) in enumerate(ZOOM):
        zt[ts >= t0] = z
    z = gaussian_smooth(zt, ZOOM_SIGMA * FPS)
    z *= 1 + 0.006 * np.sin(2 * np.pi * ts / 23.0)
    return ts, cx, cy, z

def camera_matrix(i, cam):
    ts, cx, cy, z = cam
    i = min(i, len(ts) - 1)
    zz = ZOOM_BASE * z[i]
    Hc = SH / zz; Wc = Hc * 16 / 9
    x = SW / 2 + 0.6 * (cx[i] - SW / 2)
    x = min(max(x, Wc / 2), SW - Wc / 2)
    top = cy[i] - 0.34 * Hc
    top = min(max(top, 0), SH - Hc)
    s = W / Wc
    return np.array([[s, 0, -(x - Wc / 2) * s], [0, s, -top * s]], np.float32)

# ------------------------------------------------------------------ overlays
def draw_callout(fr, c, t):
    u = t - c["s"]; D = c["e"] - c["s"]
    if u < 0 or u > D: return
    o = ease((D - u) / 0.55)
    a = ease(u / 0.75)
    mw = max(c["main"].shape[1], c["sub"].shape[1] if c["sub"] is not None else 0, c["lab"].shape[1])
    ph = 168 if c["sub"] is not None else 132
    y0 = H - 96 - ph
    panel = hgrad_sprite(int(mw + 520), ph + 60, NAVY_D, 0.80, 0.0, power=1.2)
    comp(fr, panel, 0, y0 - 30, o, clip_w=panel.shape[1] * a)
    bar_h = int((ph - 36) * ease(u / 0.6))
    comp(fr, rect_sprite(3, max(bar_h, 1), GOLD), 112, y0 + 18, o)
    al = ease((u - 0.15) / 0.6)
    comp(fr, c["lab"], 134 - 14 * (1 - al), y0 + 14, o * al)
    am = ease((u - 0.3) / 0.7)
    comp(fr, c["main"], 132 - 22 * (1 - am), y0 + 40, o * am)
    if c["sub"] is not None:
        asb = ease((u - 0.55) / 0.7)
        comp(fr, c["sub"], 134 - 18 * (1 - asb), y0 + 108, o * asb * 0.82)

def draw_lower_third(fr, t):
    s, e = 0.7, 8.3
    u = t - s; D = e - s
    if u < 0 or u > D: return
    o = ease((D - u) / 0.55); a = ease(u / 0.8)
    ph = 128; y0 = H - 96 - ph
    mw = max(T["name"].shape[1], T["role"].shape[1])
    panel = hgrad_sprite(int(mw + 520), ph + 60, NAVY_D, 0.80, 0.0, power=1.2)
    comp(fr, panel, 0, y0 - 30, o, clip_w=panel.shape[1] * a)
    comp(fr, rect_sprite(3, max(1, int((ph - 36) * ease(u / 0.6))), GOLD), 112, y0 + 18, o)
    an = ease((u - 0.2) / 0.7)
    comp(fr, T["name"], 132 - 22 * (1 - an), y0 + 14, o * an)
    ar = ease((u - 0.45) / 0.7)
    comp(fr, T["role"], 134 - 16 * (1 - ar), y0 + 80, o * ar)

def draw_agenda(fr, t):
    s, e, items = AGENDA[0], AGENDA[1], AG_ITEMS
    u = t - s; D = e - s
    if u < 0 or u > D: return
    o = ease((D - u) / 0.5); a = ease(u / 0.8)
    comp(fr, LEFT_GRAD, -1000 * (1 - a) * 0.35, 0, o * a)
    y = 330
    comp(fr, AG_LABEL, 120, y, o * ease((u - 0.2) / 0.6))
    comp(fr, rect_sprite(int(380 * ease((u - 0.3) / 1.0)) + 1, 1, GOLD), 122, y + 40, o * 0.8)
    for k, (tt, num, txt) in enumerate(items):
        ai = ease((t - tt) / 0.6)
        yy_ = y + 76 + k * 96
        comp(fr, num, 118, yy_ + 6, o * ai)
        comp(fr, txt, 196 - 24 * (1 - ai), yy_, o * ai)

def draw_chapter(fr, t):
    for i, c in enumerate(CH):
        u = t - c["s"]
        if 0 <= u <= CHAPTER_HOLD:
            o = ease((CHAPTER_HOLD - u) / 0.6); a = ease(u / 0.8)
            comp(fr, LEFT_GRAD, -1000 * (1 - a) * 0.35, 0, o * a)
            an = ease((u - 0.1) / 0.8)
            comp(fr, c["num"], 104, 300 - 20 * (1 - an), o * an)
            comp(fr, rect_sprite(int(300 * ease((u - 0.35) / 0.9)) + 1, 2, GOLD), 120, 520, o)
            at = ease((u - 0.5) / 0.8)
            comp(fr, c["title"], 116 - 20 * (1 - at), 548, o * at)
        # small persistent indicator top-right
        nxt = CH[i + 1]["s"] if i + 1 < len(CH) else CLOSING[0]
        s2 = c["s"] + CHAPTER_HOLD - 0.4
        if s2 <= t <= nxt:
            ai = ease((t - s2) / 0.7) * ease((nxt - t) / 0.5)
            sm = c["small"]
            comp(fr, sm, W - 96 - sm.shape[1], 66, ai * 0.85)
            comp(fr, rect_sprite(26, 2, GOLD), W - 96 - sm.shape[1] - 40, 77, ai)

def draw_bug(fr, t):
    s, e = 9.0, CLOSING[0]
    if not (s <= t <= e): return
    a = ease((t - s) / 0.9) * ease((e - t) / 0.6)
    comp(fr, rect_sprite(8, 8, GOLD), 98, 74, a)
    comp(fr, T["bug"], 116, 64, a * 0.85)

def title_card(t):
    fr = CARD_BG.copy()
    shift = int(t * 14) % 90
    lines = LINES[:, 200 - shift:200 - shift + W][..., None] * LINE_MASK
    fr += (np.array(GOLD, np.float32) / 255 - fr) * lines * 0.07
    corner_brackets(fr, (t - 0.2) / 1.0, alpha=0.8)
    cy = H // 2
    ak = ease((t - 0.2) / 0.8)
    k = T["kicker"]; comp(fr, k, (W - k.shape[1]) / 2, cy - 130 - 10 * (1 - ak), ak)
    at = ease((t - 0.55) / 0.9)
    m = T["title"]; comp(fr, m, (W - m.shape[1]) / 2, cy - 82 + 18 * (1 - at), at)
    lw = int(420 * ease_io((t - 0.4) / 1.1))
    if lw > 1: comp(fr, rect_sprite(lw, 2, GOLD), (W - lw) / 2, cy + 36, 1.0)
    asb = ease((t - 1.0) / 0.8)
    s = T["tsub"]; comp(fr, s, (W - s.shape[1]) / 2, cy + 62, asb * 0.8)
    return fr

def end_card(t):
    fr = CARD_BG.copy()
    shift = int((t + 40) * 14) % 90
    lines = LINES[:, 200 - shift:200 - shift + W][..., None] * LINE_MASK
    fr += (np.array(GOLD, np.float32) / 255 - fr) * lines * 0.07
    corner_brackets(fr, (t - 0.1) / 1.0, alpha=0.8)
    cy = H // 2 - 40
    am = ease((t - 0.3) / 1.0)
    m = T["end_main"]; comp(fr, m, (W - m.shape[1]) / 2, cy - 110 + 16 * (1 - am), am)
    lw = int(300 * ease_io((t - 0.5) / 1.0))
    if lw > 1: comp(fr, rect_sprite(lw, 2, GOLD), (W - lw) / 2, cy + 20, 1.0)
    a2 = ease((t - 0.9) / 0.9)
    s = T["end_sub"]; comp(fr, s, (W - s.shape[1]) / 2, cy + 44, a2 * 0.85)
    a3 = ease((t - 1.4) / 0.9)
    n = T["end_name"]; comp(fr, n, (W - n.shape[1]) / 2, H - 230, a3)
    r = T["end_role"]; comp(fr, r, (W - r.shape[1]) / 2, H - 186, a3 * 0.9)
    return fr

# ------------------------------------------------------------------ frame assembly
SHARP_AMT = 0.35
def video_frame(src_rgb, i, cam):
    g = grade(src_rgb)
    M = camera_matrix(i, cam)
    up = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    f = up.astype(np.float32) / 255
    blur = cv2.GaussianBlur(f, (0, 0), 1.6)
    f = f + (f - blur) * SHARP_AMT
    f = f * VIGNETTE + VIG_TINT * (1 - VIGNETTE)
    f = f * (1 - TOPGRAD) + (np.array(NAVY_D, np.float32) / 255) * TOPGRAD
    return f

def compose(k, src_rgb, src_i, cam):
    t_out = k / FPS
    ts = t_out - INTRO
    fr = video_frame(src_rgb, src_i, cam)
    tt = min(max(ts, 0), SRC_DUR)
    draw_bug(fr, tt)
    draw_lower_third(fr, tt)
    draw_agenda(fr, tt)
    draw_chapter(fr, tt)
    for c in CALL: draw_callout(fr, c, tt)
    # title card crossfade
    if t_out < INTRO + 0.5:
        a = 1 - ease_io((t_out - (INTRO - 0.5)) / 1.0)
        if a > 0:
            tc = title_card(t_out)
            fr = fr * (1 - a) + tc * a
    # end card
    te = t_out - (INTRO + SRC_DUR - 0.7)
    if te > 0:
        a = ease_io(te / 1.1)
        ec = end_card(te)
        fr = fr * (1 - a) + ec * a
    # gentle fade from / to black at the very ends
    fr *= min(1.0, t_out / 0.6, (TOTAL - t_out) / 0.8)
    return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)

def source_reader(path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-vf", f"fps={FPS}", "-"], stdout=subprocess.PIPE, bufsize=SW * SH * 3 * 4)
    n = SW * SH * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        yield np.frombuffer(b, np.uint8).reshape(SH, SW, 3)

def main():
    src, faces = sys.argv[1], sys.argv[2]
    cam = build_camera(faces)
    if sys.argv[3] == "--stills":
        outdir = sys.argv[4]; times = [float(x) for x in sys.argv[5:]]
        for t in times:
            ts = min(max(t - INTRO, 0), SRC_DUR - 0.05)
            r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ts:.3f}", "-i", src, "-frames:v", "1",
                                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
            img = np.frombuffer(r.stdout[:SW * SH * 3], np.uint8).reshape(SH, SW, 3)
            k = int(round(t * FPS))
            out = compose(k, img, int(round(ts * FPS)), cam)
            Image.fromarray(out).save(f"{outdir}/still_{t:07.2f}.jpg", quality=88)
        return
    mix, out = sys.argv[3], sys.argv[4]
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", mix,
                            "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                            "-profile:v", "high", "-tune", "film", "-x264-params", "aq-mode=3",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                            "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out],
                           stdin=subprocess.PIPE)
    reader = source_reader(src)
    cur = next(reader); cur_i = 0
    for k in range(NFR):
        want = int(round((k / FPS - INTRO) * FPS))
        want = min(max(want, 0), int(SRC_DUR * FPS) - 1)
        while cur_i < want:
            try:
                cur = next(reader); cur_i += 1
            except StopIteration:
                break
        enc.stdin.write(compose(k, cur, cur_i, cam).tobytes())
        if k % 300 == 0: print(f"frame {k}/{NFR}", flush=True)
    enc.stdin.close(); enc.wait()
    print("done", out)

if __name__ == "__main__":
    main()

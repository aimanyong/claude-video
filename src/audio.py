"""Voice enhancement, synthesized executive score, subtle SFX, final mix.

usage: python3 audio.py SOURCE_VIDEO RNNOISE_MODEL OUT_DIR
"""
import sys, subprocess, json, re, numpy as np, wave
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from config import *

SR = 48000
src, rnn, out = sys.argv[1], sys.argv[2], sys.argv[3]
TOTAL = INTRO + SRC_DUR + OUTRO

def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)

# ---------------------------------------------------------------- voice
chain = (f"arnndn=m={rnn}:mix=0.8,highpass=f=85,lowpass=f=13500,"
         "equalizer=f=220:t=q:w=1.2:g=-2.5,equalizer=f=3000:t=q:w=1.0:g=2.5,"
         "equalizer=f=9000:t=q:w=1.5:g=1.0,"
         "acompressor=threshold=-24dB:ratio=3:attack=8:release=180:makeup=3,"
         "deesser=i=0.3")
r = run(["ffmpeg", "-y", "-i", src, "-vn", "-af", chain + ",loudnorm=I=-16:TP=-1.5:LRA=7:print_format=json",
         "-f", "null", "-"])
m = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr).group(0))
ln = (f"loudnorm=I=-16:TP=-1.5:LRA=7:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
      f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
run(["ffmpeg", "-y", "-i", src, "-vn", "-af", chain + "," + ln + f",aresample={SR}",
     "-ac", "1", "-c:a", "pcm_s16le", f"{out}/voice.wav"])

# ---------------------------------------------------------------- music
rng = np.random.default_rng(118)
N = int(TOTAL * SR)
t = np.arange(N) / SR

def midi(n): return 440.0 * 2 ** ((n - 69) / 12)

# D major, slow and restrained: Dmaj9 - Bm9 - Gmaj9 - Asus2  (8 s each)
CHORDS = [[50, 57, 61, 64, 66], [47, 54, 57, 61, 62], [43, 50, 54, 57, 62], [45, 52, 57, 59, 64]]
BAR = 8.0
pad = np.zeros(N)
for i in range(int(TOTAL // BAR) + 1):
    ch = CHORDS[i % 4]
    s0 = int(i * BAR * SR); s1 = min(N, int((i + 1) * BAR * SR + 2.5 * SR))
    if s0 >= N: break
    tt = np.arange(s1 - s0) / SR
    env = np.minimum(1, tt / 2.5) * np.clip((BAR + 2.5 - tt) / 2.5, 0, 1)
    seg = np.zeros(s1 - s0)
    for n in ch:
        f = midi(n + 12 if n < 48 else n)
        for det in (-0.12, 0.0, 0.12):
            ph = rng.uniform(0, 2 * np.pi)
            seg += np.sin(2 * np.pi * f * (1 + det / 100) * tt + ph) * 0.5
            seg += 0.12 * np.sin(2 * np.pi * 2 * f * (1 + det / 100) * tt + ph)
    bass = np.sin(2 * np.pi * midi(ch[0] - 12) * tt) * 1.4
    pad[s0:s1] += (seg + bass) * env

# sparse felt-piano notes (gentle pulse, ~64 bpm)
piano = np.zeros(N)
beat = 60 / 64
k = 0
while k * beat * 2 < TOTAL - 2:
    tb = k * beat * 2
    ch = CHORDS[int(tb // BAR) % 4]
    pattern = [ch[2] + 12, ch[3] + 12, ch[4] + 12, ch[3] + 12]
    n = pattern[k % 4]
    if k % 8 == 7: k += 1; continue
    f = midi(n); L = int(3.5 * SR); s0 = int(tb * SR); s1 = min(N, s0 + L)
    tt = np.arange(s1 - s0) / SR
    env = (1 - np.exp(-tt / 0.008)) * np.exp(-tt / 1.1)
    note = (np.sin(2*np.pi*f*tt) + 0.35*np.sin(2*np.pi*2*f*tt)*np.exp(-tt/0.4) + 0.1*np.sin(2*np.pi*3*f*tt)*np.exp(-tt/0.2))
    piano[s0:s1] += note * env * rng.uniform(0.75, 1.0)
    k += 1

def onepole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], x)

try:
    import scipy  # noqa
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy"], check=True)
from scipy.signal import fftconvolve, lfilter

pad = onepole_lp(onepole_lp(pad, 1400), 1400)
mus = pad / np.abs(pad).max() * 0.55 + piano / np.abs(piano).max() * 0.32

# simple stereo reverb: exponentially decaying noise IR
def reverb(x, secs=2.8, seed=0):
    r = np.random.default_rng(seed)
    L = int(secs * SR); tt = np.arange(L) / SR
    ir = r.standard_normal(L) * np.exp(-tt / (secs / 6.9) * 1.0)
    ir = onepole_lp(ir, 5000); ir /= np.sqrt((ir ** 2).sum())
    return fftconvolve(x, ir)[:len(x)]
musL = 0.65 * mus + 0.5 * reverb(mus, seed=1)
musR = 0.65 * mus + 0.5 * reverb(mus, seed=2)
music = np.stack([musL, musR], 1)
# fade in / out
fi = np.clip(t / 2.0, 0, 1); fo = np.clip((TOTAL - t) / 3.0, 0, 1)
music *= (fi * fo)[:, None]
music /= np.abs(music).max()

# ---------------------------------------------------------------- sfx
sfx = np.zeros((N, 2))
def add(sig, at, gain):
    s0 = int(at * SR); s1 = min(N, s0 + len(sig))
    if s0 < 0 or s0 >= N: return
    sfx[s0:s1] += sig[:s1 - s0] * gain

def whoosh(dur=1.3, f0=250, f1=2600, seed=0):
    r = np.random.default_rng(seed); L = int(dur * SR); tt = np.arange(L) / SR
    x = r.standard_normal(L)
    # time-varying band-pass via two one-pole stages around a swept centre
    fc = f0 * (f1 / f0) ** np.sin(np.pi / 2 * tt / dur)
    y = np.zeros(L); lp = 0.0; hp_prev = 0.0; lp2 = 0.0
    for i in range(L):
        a = np.exp(-2 * np.pi * fc[i] * 1.6 / SR)
        lp = (1 - a) * x[i] + a * lp
        b = np.exp(-2 * np.pi * fc[i] * 0.6 / SR)
        lp2 = (1 - b) * lp + b * lp2
        y[i] = lp - lp2
    env = np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 2
    y = y * env; y /= np.abs(y).max()
    pan = np.linspace(-0.5, 0.5, L)
    return np.stack([y * (1 - pan) / 1.5, y * (1 + pan) / 1.5], 1)

def shimmer(f=1174.66, dur=2.8):
    L = int(dur * SR); tt = np.arange(L) / SR
    env = (1 - np.exp(-tt / 0.02)) * np.exp(-tt / 0.9)
    y = (np.sin(2*np.pi*f*tt) + 0.5*np.sin(2*np.pi*1.5*f*tt) + 0.25*np.sin(2*np.pi*2*f*tt)) * env
    y = y / np.abs(y).max()
    return np.stack([y, np.roll(y, int(0.012 * SR))], 1)

def low_swell(dur=2.5):
    L = int(dur * SR); tt = np.arange(L) / SR
    env = np.sin(np.pi * tt / dur) ** 2
    y = np.sin(2 * np.pi * 55 * tt) * env + 0.3 * np.sin(2 * np.pi * 110 * tt) * env
    return np.stack([y, y], 1) / 1.3

add(low_swell(3.0), 0.1, 0.07)
add(shimmer(1174.66), 0.6, 0.06)
add(whoosh(1.4, seed=3), INTRO - 1.0, 0.16)
for i, c in enumerate(SFX_CHAPTER):
    add(whoosh(1.2, seed=10 + i), INTRO + c - 0.55, 0.11)
    add(shimmer(1174.66 if i % 2 == 0 else 1318.5), INTRO + c, 0.035)
add(whoosh(1.6, seed=40), INTRO + SRC_DUR - 1.2, 0.13)
add(low_swell(3.5), INTRO + SRC_DUR - 0.6, 0.06)
add(shimmer(1468.3, 3.5), INTRO + SRC_DUR - 0.1, 0.05)

def wr(name, x):
    x = np.clip(x, -1, 1)
    if x.ndim == 1: x = x[:, None]
    with wave.open(f"{out}/{name}", "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
wr("music.wav", music * 0.9)
wr("sfx.wav", sfx)

# ---------------------------------------------------------------- mix
d = int(INTRO * 1000)
fc = (f"[0:a]adelay={d}|{d},apad=whole_dur={TOTAL},aformat=channel_layouts=stereo,volume=3dB,asplit=2[v][sc];"
      f"[1:a]volume=-27dB[m];"
      f"[m][sc]sidechaincompress=threshold=0.02:ratio=4:attack=60:release=900:makeup=1[md];"
      f"[2:a]volume=1.0[s];"
      f"[v][md][s]amix=inputs=3:normalize=0:duration=longest,"
      f"atrim=0:{TOTAL},alimiter=limit=0.89:attack=5:release=60[out]")
run(["ffmpeg", "-y", "-i", f"{out}/voice.wav", "-i", f"{out}/music.wav", "-i", f"{out}/sfx.wav",
     "-filter_complex", fc, "-map", "[out]", "-ar", str(SR), "-c:a", "pcm_s16le", f"{out}/mix.wav"])
print("audio done", TOTAL)

# Executive Boardroom Cut: Office Leasing Pitch

A three-minute sales pitch recorded for the PNB Merdeka Ventures assessment
(Associate, Leasing & Marketing, Office Leasing), re-edited as an executive
briefing to a Board of Directors.

**Output:** `output/executive_briefing_1080p.mp4` (1920×1080, 75 MB) and `output/executive_briefing_720p.mp4` (1280×720, 39 MB). Both are 30 fps H.264/AAC, 2:51.

## What was done

| Area | Treatment |
|---|---|
| Authenticity | Original face, voice and words kept. Nothing was cut, re-ordered, re-voiced or generated. |
| Camera | Face-tracked reframing of the single take. Slow push-ins on key business statements and gentle pull-backs to wide framing at each section change. |
| Grade | Soft S-curve, highlight roll-off for the bright ceiling and window, navy-tinted shadows and warm highlights, restrained saturation to keep skin natural, subtle vignette and a top shade. |
| Graphics | Navy, charcoal and gold palette set in Inter. Includes a title card, name lower-third, the "three questions" agenda, four numbered chapter reveals and key-point callouts. A small "Executive Briefing" mark sits top left and a chapter indicator top right. The end card closes the video. |
| Sound | RNNoise denoise, high-pass and low-pass filters, de-mud and presence EQ, compression, de-esser, two-pass loudness normalisation (-16 LUFS). An original soft instrumental bed (synthesized, so royalty-free) is ducked under the voice. Quiet whoosh and shimmer cues mark chapter changes. |

Every on-screen word restates something said in the recording. The timed plan
is in `src/config.py`.

## Rebuild

```bash
# 1. face track (6 fps Haar cascade)  -> faces.json
python3 tools/track_face.py SCRATCH SOURCE.mov
# 2. audio: voice chain + music + sfx + mix -> OUT/mix.wav
python3 src/audio.py SOURCE.mov std.rnnn OUT
# 3. picture + mux
python3 src/render.py SOURCE.mov faces.json OUT/mix.wav OUT/video.mp4
# preview stills (output-timeline seconds)
python3 src/render.py SOURCE.mov faces.json --stills OUT 6.5 38 82
```

Requirements: ffmpeg, numpy, scipy, pillow, opencv-python-headless<5, and the Inter font.
The RNNoise model is `std.rnnn` from richardpl/arnndn-models. The Haar cascade
is `haarcascade_frontalface_default.xml` from opencv.

To change the presenter name, any caption or any timing, edit `src/config.py`
and re-run step 3. Re-run step 2 too if chapter times change, because the
transition sounds follow them.

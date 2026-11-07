import sys, wave, json, re, numpy as np, sherpa_onnx
S=sys.argv[1]; M=S+"/models/sherpa-onnx-whisper-small.en/"
w=wave.open(S+"/a16.wav"); a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
dur=len(a)/16000
st=[];en=[]
for l in open(S+"/work/sil.txt"):
    k,v=l.split(": "); (st if "start" in k else en).append(float(v))
mids=[(s+e)/2 for s,e in zip(st,en) if s>0.5]
# chunk boundaries: greedily keep chunks between 4 and 12 s
b=[0.0]
for m in mids:
    if m-b[-1]>=1.2: b.append(m)
b.append(dur)
rec=sherpa_onnx.OfflineRecognizer.from_whisper(encoder=M+"small.en-encoder.int8.onnx",decoder=M+"small.en-decoder.int8.onnx",tokens=M+"small.en-tokens.txt",num_threads=4,language="en",task="transcribe")
out=[]
for t0,t1 in zip(b[:-1],b[1:]):
    if t1-t0>28: print("LONG",t0,t1)
    s=rec.create_stream(); s.accept_waveform(16000,a[int(t0*16000):int(t1*16000)]); rec.decode_stream(s)
    out.append(dict(start=round(t0,2),end=round(t1,2),text=s.result.text.strip()))
    print(f"[{t0:6.2f}-{t1:6.2f}] {s.result.text.strip()}",flush=True)
json.dump(out,open(S+"/work/transcript_fine.json","w"),indent=1)

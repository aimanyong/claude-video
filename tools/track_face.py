import sys, subprocess, numpy as np, cv2, json
S=sys.argv[1]; V=sys.argv[2]; W,H=910,512
cc=cv2.CascadeClassifier(S+"/models/haar.xml")
p=subprocess.Popen(["ffmpeg","-v","error","-i",V,"-vf","fps=6","-f","rawvideo","-pix_fmt","gray","-"],stdout=subprocess.PIPE)
res=[];i=0
while True:
    b=p.stdout.read(W*H)
    if len(b)<W*H: break
    g=np.frombuffer(b,np.uint8).reshape(H,W)
    f=cc.detectMultiScale(g,1.1,5,minSize=(50,50))
    t=i/6
    if len(f):
        x,y,w,h=max(f,key=lambda r:r[2]*r[3]); res.append([t,float(x+w/2),float(y+h/2),float(w)])
    else: res.append([t,None,None,None])
    i+=1
json.dump(res,open(S+"/work/faces.json","w"))
ok=[r for r in res if r[1] is not None]
print(len(res),len(ok)); a=np.array([r[1:] for r in ok])
print("cx",np.percentile(a[:,0],[5,50,95]),"cy",np.percentile(a[:,1],[5,50,95]),"w",np.percentile(a[:,2],[5,50,95]))

import numpy as np, subprocess, sys
from PIL import Image, ImageFilter
S = sys.argv[1]; H, W = 1280, 720
def frames(f):
    r = subprocess.run(['ffmpeg','-v','error','-i',f,'-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True).stdout
    return np.frombuffer(r,np.uint8).reshape(-1,H,W,3)
v1 = frames(f'{S}/v1.mp4'); v2 = frames(f'{S}/v2.mp4')
CUT1, V2_END, PEAK, REF, K = 91, 83, 90, 85, 16
ref = v1[REF].astype(np.float32); pk = v1[PEAK].astype(np.float32)
Wc = np.array([241.,223.,223.])
import scipy.ndimage as ndi
diff = np.abs(pk-ref).max(-1)
lumc = pk.mean(-1); satc = pk.max(-1)-pk.min(-1)
m = (diff > 28) & (lumc > 175) & (satc < 60)
m = ndi.binary_closing(m, iterations=6); m = ndi.binary_fill_holes(m); m = ndi.binary_opening(m, iterations=3)
lab,n = ndi.label(m); sz = ndi.sum(m,lab,range(1,n+1)); m = lab==(1+np.argmax(sz))
soft = np.clip(diff/45,0,1)
a = np.maximum(ndi.binary_erosion(m,iterations=8).astype(np.float32), soft*m*(lumc>175))
A0 = ndi.gaussian_filter(a.astype(np.float32), 3)
col0 = pk.copy()
A0 = A0*np.clip((ndi.gaussian_filter(lumc,1.5)-150)/30,0,1)
Image.fromarray((A0*255).astype(np.uint8)).save(f'{S}/work/alpha.png')
ys,xs = np.nonzero(A0>0.5); cy,cx = ys.mean(), xs.mean(); print('cloud centre',cx,cy)
rng = np.random.default_rng(3)
def noise(seed, sc):
    r = np.random.default_rng(seed).random((H//sc+2, W//sc+2)).astype(np.float32)
    return np.asarray(Image.fromarray((r*255).astype(np.uint8)).resize((W+2*sc,H+2*sc),Image.BICUBIC).filter(ImageFilter.GaussianBlur(sc/3)))[:H,:W]/255.
N = 0.6*noise(1,40)+0.4*noise(2,14); N=(N-N.min())/(N.max()-N.min())
def xform(img, s, dy):
    # scale about cloud centre, move up by dy
    im = Image.fromarray(img) if img.dtype==np.uint8 else Image.fromarray(img.astype(np.float32),'F')
    a_ = 1/s; c = cx - a_*cx; f = cy - a_*(cy - dy) ; 
    return im.transform((W,H), Image.AFFINE, (a_,0,c,0,a_,f), resample=Image.BILINEAR)
out = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','30','-i','-',
    '-c:v','libx264','-crf','16','-preset','slow','-pix_fmt','yuv420p',f'{S}/work/video.mp4'], stdin=subprocess.PIPE)
for i in range(CUT1): out.stdin.write(v1[i].tobytes())
for k in range(V2_END):
    f = v2[k].astype(np.float32)
    if k < K:
        p = (k+1)/K; e = 1-(1-p)**2
        s = 1+0.30*e; dy = 40*e
        A = np.asarray(xform(A0, s, dy), np.float32)
        C = np.stack([np.asarray(xform(col0[...,c], s, dy), np.float32) for c in range(3)],-1)
        Nk = np.asarray(xform(N.astype(np.float32), 1+0.15*e, 70*e), np.float32)
        brk = np.clip((Nk - (p*1.15-0.25))/0.35, 0, 1)   # wisps break up from thin parts
        A = A*brk*(1-p**1.6)
        f = f*(1-A[...,None]) + C*A[...,None]
    out.stdin.write(np.clip(f,0,255).astype(np.uint8).tobytes())
out.stdin.close(); out.wait()

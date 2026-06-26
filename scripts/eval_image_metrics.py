#!/usr/bin/env python
"""Image-criteria eval (PSNR, SSIM, LPIPS, CLIP-I) of generated ego vs ego_GT — the paper's
Table-1 image metrics. Compares the right-HxH ego half of the width-concat generation, frame-aligned.

Usage (PYTHONNOUSERSITE=1 for the transformers CLIP import):
  PYTHONNOUSERSITE=1 python scripts/eval_image_metrics.py \
      --gen_dir <dir with {clip}.mp4>  --gtmap <clip->ego_GT.json>  [--ego_crop] [--device cuda]
"""
import argparse, json, os, glob, numpy as np, cv2, imageio.v2 as imageio, torch

def frames(p): return [np.asarray(f) for f in imageio.get_reader(p)]
def ego_crop(f):
    h, w = f.shape[:2]; return f[:, w - h:, :] if w > h else f
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 20*np.log10(255) - 10*np.log10(mse)
def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64); b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01*255)**2, (0.03*255)**2; g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    m1, m2 = g(a), g(b); m1s, m2s, m12 = m1*m1, m2*m2, m1*m2
    s1, s2, s12 = g(a*a)-m1s, g(b*b)-m2s, g(a*b)-m12
    return float((((2*m12+C1)*(2*s12+C2))/((m1s+m2s+C1)*(s1+s2+C2))).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--gtmap", required=True)
    ap.add_argument("--ego_crop", action="store_true", help="crop right-HxH ego half from a width-concat gen")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"

    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def lpips_score(g, t):  # uint8 HWC RGB -> [-1,1] NCHW
        def to_t(x): return (torch.from_numpy(x).float().permute(2,0,1).unsqueeze(0)/127.5-1).to(dev)
        return float(lpips_fn(to_t(g), to_t(t)).item())
    @torch.no_grad()
    def clip_i(g, t):
        ins = clip_proc(images=[g, t], return_tensors="pt").to(dev)
        e = clip.get_image_features(**ins); e = e/e.norm(dim=-1, keepdim=True)
        return float((e[0]*e[1]).sum().item())

    gtmap = json.load(open(a.gtmap))
    rows = []
    for clip_name, gt_path in gtmap.items():
        gp = os.path.join(a.gen_dir, f"{clip_name}.mp4")
        if not (os.path.exists(gp) and os.path.exists(gt_path)):
            print(f"  skip {clip_name} (missing gen or gt)"); continue
        gen = frames(gp); gt = frames(gt_path)
        if a.ego_crop: gen = [ego_crop(f) for f in gen]
        n = min(len(gen), len(gt)); H, W = gt[0].shape[:2]
        ps, ss, lp, ci = [], [], [], []
        for i in range(n):
            g = cv2.resize(gen[i], (W, H))
            ps.append(psnr(g, gt[i])); ss.append(ssim(g, gt[i]))
            lp.append(lpips_score(g, gt[i])); ci.append(clip_i(g, gt[i]))
        r = (clip_name, np.mean(ps), np.mean(ss), np.mean(lp), np.mean(ci), n)
        rows.append(r)
        print(f"  {clip_name:38s} PSNR {r[1]:5.2f} SSIM {r[2]:.3f} LPIPS {r[3]:.3f} CLIP-I {r[4]:.3f} (n={n})")

    if rows:
        A = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
        m = A.mean(0)
        print("\n=== AVERAGE over {} clips ===".format(len(rows)))
        print(f"PSNR {m[0]:.2f}  SSIM {m[1]:.3f}  LPIPS {m[2]:.3f}  CLIP-I {m[3]:.3f}")
        print("\n=== paper Table 1 (EgoX, Seen) ===")
        print("PSNR 16.05  SSIM 0.556  LPIPS 0.498  CLIP-I 0.896")

if __name__ == "__main__":
    main()

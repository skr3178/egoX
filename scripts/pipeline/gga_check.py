#!/usr/bin/env python
"""§5a GGA consistency check: does inference compute cam_rays the same way training cached them?

Replicates BOTH the training (wan_dataset.py) and inference (infer_nf4.py) cam_rays computations
on ONE clip at 49x176x704 (ego latent 22x22, F=13), using the SAME ego_intrinsic/ego_extrinsic
from meta. Then compares to each other AND to the cached (ground-truth training) tensor.

cosine sim ~1.0  => conventions cancel, inference is fine.
cosine sim < ~0.9 => inference feeds the LoRA out-of-distribution conditioning -> artifacts.
"""
import json, numpy as np, torch, cv2

CLIP = "fair_cooking_06_6_1000_1048"
CT = "/media/skr/SeagateHub1/egoexo4d/cooking_train"
META = f"{CT}/meta_val_20.json"
CACHE = f"{CT}/cache/attn_maps/wan-i2v/49x176x704/{CLIP}_exo_exo_ego_gt_prior.safetensors"
H = 22           # ego latent side (176/8)
F = 13           # 49 frames -> ::4 -> 13
SCALE = 1/8

d = json.load(open(META))[CLIP]
ego_intrinsic = torch.tensor(np.array(d["ego_intrinsics"]), dtype=torch.float64)   # (3,3)
ego_extrinsic = torch.tensor(np.array(d["ego_extrinsics"]), dtype=torch.float64)   # (49,3,4)

def scaled_K():
    K = ego_intrinsic.clone()
    K[0,0]*=SCALE; K[1,1]*=SCALE; K[0,2]*=SCALE; K[1,2]*=SCALE
    return K

def pixel_grid(h, w):
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ones = torch.ones_like(xs)
    return torch.stack([xs, ys, ones], dim=-1).view(-1,3).to(torch.float64)   # (h*w,3)

# ---------- TRAINING way (wan_dataset.py 326-340, 407) ----------
def cam_rays_training():
    W = H
    K = scaled_K()
    px = pixel_grid(H, W)
    inv = torch.linalg.inv(K)
    cr = (inv @ px.T).T                                   # pinhole
    cr = cr / cr.norm(dim=-1, keepdim=True)
    cr = cr.view(H, W, 3).unsqueeze(0).expand(F,-1,-1,-1).reshape(F, H*W, 3)
    cr = cr @ ego_extrinsic[::4,:3,:3].transpose(-1,-2)   # R^T
    cr = cr / cr.norm(dim=-1, keepdim=True)
    cr = cr.view(F, H, W, 3)
    M = torch.tensor([[0,1,0],[-1,0,0],[0,0,1]], dtype=cr.dtype)
    cr = cr @ M                                           # vector 90-rot
    return cr

# ---------- INFERENCE way (infer_nf4.py 140-170, 246) ----------
def cam_rays_inference():
    W = H
    K = scaled_K()
    px = pixel_grid(H, W)
    px_cv = px[...,:2].numpy().reshape(-1,1,2).astype(np.float32)
    Knp = K.numpy().astype(np.float32)
    D = np.array([[-0.02340373583137989,0.09388021379709244,-0.06088035926222801,0.0053304750472307205,
        0.003342868760228157,-0.0006356257363222539,0.0005087381578050554,-0.0004747129278257489,
        -0.0011330085108056664,-0.00025734835071489215,0.00009328465239377692,0.00009424977179151028]]).astype(np.float32)
    npts = cv2.undistortPoints(px_cv, Knp, D, R=np.eye(3), P=np.eye(3))   # fisheye
    npts = torch.from_numpy(npts).squeeze(1).to(torch.float64)
    ones = torch.ones_like(npts[...,:1])
    cr = torch.cat([npts, ones], dim=-1)
    cr = cr / cr.norm(dim=-1, keepdim=True)
    cr = cr @ ego_extrinsic[::4,:3,:3]                    # R (no transpose)
    cr = cr.view(F, H, W, 3)
    cr = torch.rot90(cr, k=-1, dims=[1,2])                # grid 90-rot
    return cr

def cossim(a, b):
    a = a.reshape(-1,3); b = b.reshape(-1,3)
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)

tr = cam_rays_training()
inf = cam_rays_inference()

# ground-truth: ego region of cached attn_maps (the conditioning the LoRA actually saw)
from safetensors.torch import safe_open
with safe_open(CACHE, framework="pt", device="cpu") as g:
    attn = g.get_tensor("attn_maps").double()            # (13,22,88,3)
    cached_cam = g.get_tensor("cam_rays").double()        # (13,22,22,3)
ego_from_attn = attn[:, :, -H:, :]                        # final ego cond in attn_maps

print("shapes: training",tuple(tr.shape),"inference",tuple(inf.shape),"cached_attn_ego",tuple(ego_from_attn.shape))
print()
def report(name, a, b):
    c = cossim(a, b)
    print(f"{name}: cos mean={c.mean():.4f} median={c.median():.4f} min={c.min():.4f} "
          f"frac>0.99={ (c>0.99).float().mean():.3f} frac>0.9={ (c>0.9).float().mean():.3f} frac<0={ (c<0).float().mean():.3f}")

# sanity: does my training replica reproduce the cache?
report("[sanity] training-replica vs cached attn ego", tr, ego_from_attn)
report("[sanity] training-replica vs cached cam_rays ", tr, cached_cam)
print()
# THE question:
report("[MAIN]   inference vs training-replica     ", inf, tr)
report("[MAIN]   inference vs cached attn ego       ", inf, ego_from_attn)

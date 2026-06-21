Here's the whole thing assembled end-to-end, with each generic "video diffusion model / depth estimator / renderer" slot replaced by the actual model the paper plugs in.

## The filled-in pipeline

**Inputs:** a single exocentric video `X = {Xᵢ}` and the target egocentric camera pose `φ = {φᵢ}` (from Ego-Exo4D ground truth, or set by hand via **Viser** for in-the-wild clips).

**Stage 1 — Build the egocentric prior (offline, once per clip):**
1. Per-frame monocular depth from **MoGe-2** [41] (sharp but temporally jittery).
2. Temporally smooth depth from **Video Depth Anything** [8] (smooth but affine-invariant).
3. Align the two using **ViPE**'s [16] momentum-based affine fit (Eq. 1), dynamic objects masked out so only static background drives alignment.
4. Lift aligned depth → 3D point cloud via camera intrinsics.
5. Render that point cloud from the target ego pose with **PyTorch3D**'s [34] point-cloud renderer → **ego-prior video `P`** (Eq. 2). Pixel-aligned with the target view but full of holes.

**Stage 2 — Encode to latents:** the **frozen Wan 2.1 VAE** encoder turns the exo video into `x₀` and the ego prior into `p₀`.

**Stage 3 — Conditioning layout in latent space:**
- `x₀` (clean, held fixed every timestep) is **width-concatenated** beside the noisy latent `zₜ` → forces implicit cross-view warping.
- `p₀` plus the binary mask `(m¹, m⁰)` is **channel-concatenated** into `zₜ` → injects pixel-aligned guidance. This is exactly the extra-channel slot the inpainting variant provides.
- Relation: `zₜ₋₁ = fθ(x₀, zₜ | x₀, p₀ | m¹, m⁰)` (Eq. 3).

**Stage 4 — Denoising backbone:** the **Wan 2.1 14B I2V inpainting model** [40], **frozen**, adapted only by **LoRA rank 256**. Its `N` DiT blocks each run **Geometry-Guided Self-Attention** (a precomputed 3D-direction cosine-similarity bias on the attention logits, Eqs. 4–7; directions downsampled over `4×16×16` patches to match the VAE) plus the standard **cross-attention** that consumes a **GPT-4o**-written text prompt describing both views.

**Stage 5 — Output:** after sampling, drop the width-concatenated exo half of the latent and decode only the ego half with the frozen Wan 2.1 VAE → the final egocentric video.

**Training:** LoRA rank 256, batch 1, 8×H200 (140 GB), ~1 day, on 3,600 Ego-Exo4D clips.

Let me also lay this out as a diagram you can keep and edit alongside your implementation.I've already given the full filled-in pipeline above, so the text answer stands on its own. For the diagram, Eraser just needs to know where to put it before I generate it.
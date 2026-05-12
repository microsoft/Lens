"""Single-shot CLI entry-point for the Lens text-to-image pipeline.

Example:

    python inference.py \\
        --transformer_ckpt /path/to/lens_600m.safetensors \\
        --prompt "A scenic landscape with a serene lake" \\
        --base_resolution 1024 --aspect_ratio 1:1 \\
        --steps 50 --cfg 4.0 --n 4 --seed 42 \\
        --out ./outputs

Batch by joining prompts with `|`:

    python inference.py --prompt "a cat|a dog|a robot" ...
"""

from __future__ import annotations

import argparse
import os

import torch

from lens import LensPipeline
from lens.resolution import SUPPORTED_ASPECT_RATIOS, SUPPORTED_BASE_RESOLUTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens text-to-image inference")
    parser.add_argument("--transformer_ckpt", type=str, required=True,
                        help="Path to the Lens DiT .safetensors checkpoint.")
    parser.add_argument("--gpt_oss", type=str, default="openai/gpt-oss-20b",
                        help="HF repo id (or local path) for the GPT-OSS text encoder.")
    parser.add_argument("--vae", type=str, default="black-forest-labs/FLUX.2-dev",
                        help="HF repo id for the Flux2 VAE.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt(s). Use '|' to separate multiple prompts.")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt (broadcast across all positives).")
    parser.add_argument("--base_resolution", type=int, default=1440,
                        choices=SUPPORTED_BASE_RESOLUTIONS)
    parser.add_argument("--aspect_ratio", type=str, default="1:1",
                        choices=SUPPORTED_ASPECT_RATIOS)
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg", type=float, default=5.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of images per prompt.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="./outputs")
    parser.add_argument("--mxfp4", action="store_true",
                        help="Keep GPT-OSS in MXFP4 (default: dequantize to bf16).")
    parser.add_argument("--reasoner", action="store_true",
                        help="Enable prompt reasoner (uses GPT-OSS unless --api_url is set).")
    parser.add_argument("--api_url", type=str, default=None,
                        help="OpenAI-compatible base URL for the API reasoner.")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--api_model", type=str, default=None,
                        help="Model name to send to the OpenAI-compatible endpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = [p.strip() for p in args.prompt.split("|") if p.strip()]
    if not prompts:
        raise SystemExit("No non-empty prompts after splitting on '|'.")

    os.makedirs(args.out, exist_ok=True)

    pipe = LensPipeline.from_pretrained_components(
        transformer_ckpt=args.transformer_ckpt,
        gpt_oss_path=args.gpt_oss,
        vae_path=args.vae,
        mxfp4=args.mxfp4,
        openai_api_key=args.api_key,
        openai_base_url=args.api_url,
        openai_model=args.api_model,
    )

    generator = (
        torch.Generator(device=pipe._execution_device).manual_seed(int(args.seed))
        if args.seed is not None
        else None
    )

    out = pipe(
        prompt=prompts,
        negative_prompt=args.negative_prompt,
        base_resolution=args.base_resolution,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        num_images_per_prompt=args.n,
        generator=generator,
        enable_reasoner=args.reasoner,
    )

    # Save one file per (prompt, image-index) pair.
    images = list(out.images)
    expected_images = len(prompts) * args.n
    if len(images) != expected_images:
        raise RuntimeError(
            f"Pipeline returned {len(images)} images; expected {expected_images}."
        )
    img_iter = iter(images)
    for p_idx, prompt in enumerate(prompts):
        for s_idx in range(args.n):
            img = next(img_iter)
            fname = f"p{p_idx:03d}_s{s_idx:02d}.png"
            img.save(os.path.join(args.out, fname))
            print(f"saved {fname} :: {prompt!r}")
    refined = getattr(pipe, "_last_refined_prompts", prompts)
    if any(r != p for r, p in zip(refined, prompts)):
        print("\nRefined prompts:")
        for orig, ref in zip(prompts, refined):
            print(f"  {orig!r}\n    -> {ref!r}")


if __name__ == "__main__":
    main()

"""
Percival inference — extract frozen visual-tower embeddings from CT volumes.

Accepts either pre-converted PyTorch tensors (.pt) or raw NIfTI files
(.nii / .nii.gz). For NIfTI input the script calls the same
`nifti_to_pt.load_and_resample(...)` used by the batch preprocessor, so
both paths produce identical embeddings.

Two invocation modes:

  1. Single-volume mode (one --pt-path OR one --nifti-path):

       python extract_embeddings.py \\
           --weights /path/to/visual_epoch_<N>_loss_<L>.pth \\
           --config configs/augreg_base_v0.yaml \\
           --pt-path /path/to/scan.pt \\
           --output /path/to/scan_embedding.pt

       python extract_embeddings.py \\
           --weights /path/to/visual_epoch_<N>_loss_<L>.pth \\
           --config configs/augreg_base_v0.yaml \\
           --nifti-path /path/to/scan.nii.gz \\
           --output /path/to/scan_embedding.pt

  2. Manifest mode (xlsx with pt_path and/or nifti_path columns):

       python extract_embeddings.py \\
           --weights /path/to/visual_epoch_<N>_loss_<L>.pth \\
           --config configs/augreg_base_v0.yaml \\
           --input-xlsx /path/to/manifest.xlsx \\
           --output-dir /path/to/embeddings/ \\
           --batch-size 8

       Per row: uses the value in `pt_path` if present (skips conversion);
       otherwise uses `nifti_path` and converts on-the-fly. Writes a
       sibling xlsx `<input>_with_embed.xlsx` with an `embed_path`
       column added.

The script does NOT auto-detect the best checkpoint — pass --weights
explicitly. Model architecture (vision_model_size, image_size, etc.) is
read from the YAML config you pass via --config and must match the
checkpoint architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

# Make sibling subpackages importable whether this script is run from
# the release root or from anywhere else.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_operations.percival import Percival
from nifti_to_pt.nifti_to_pt import load_and_resample


PT_PATH_COLUMN    = "pt_path"
NIFTI_PATH_COLUMN = "nifti_path"
EMBED_PATH_COLUMN = "embed_path"

# NIfTI on-the-fly resample target spacing. Hardcoded to match the
# default used by foundation/nifti_to_pt/nifti_to_pt.py (which is what
# produced every stored .pt file): `--spacing 3 1 1` in (z, y, x) order.
# The `image_spacing` field in the augreg_*_v0.yaml configs is unrelated
# metadata that's misleading here — at train time it's ignored
# (`use_target_spacing: false`) and the production extraction pipeline
# never reads it. Don't change this unless you've also re-run
# nifti_to_pt.py with a matching --spacing.
TARGET_SPACING_ZYX = (3.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_filename(input_path: str) -> str:
    h = hashlib.sha256(str(input_path).encode("utf-8")).hexdigest()[:16]
    return f"{h}.pt"


def crop_or_pad(volume: torch.Tensor, target_shape) -> torch.Tensor:
    """Centre-crop or zero-pad a 4D volume tensor (1, D, H, W) to the
    target shape (D, H, W). Inlined here so this script doesn't drag in
    data_operations/ as a hard import dependency."""
    assert volume.dim() == 4, f"expected (1, D, H, W); got {tuple(volume.shape)}"
    _, d, h, w = volume.shape
    td, th, tw = target_shape

    def _fit(cur, tgt):
        if cur >= tgt:
            start = (cur - tgt) // 2
            return slice(start, start + tgt), (0, 0)
        pad_total = tgt - cur
        pad_lo = pad_total // 2
        pad_hi = pad_total - pad_lo
        return slice(0, cur), (pad_lo, pad_hi)

    sl_d, pd_d = _fit(d, td)
    sl_h, pd_h = _fit(h, th)
    sl_w, pd_w = _fit(w, tw)

    cropped = volume[:, sl_d, sl_h, sl_w]
    cropped = torch.nn.functional.pad(
        cropped,
        (pd_w[0], pd_w[1], pd_h[0], pd_h[1], pd_d[0], pd_d[1]),
        mode="constant", value=0,
    )
    return cropped


# ---------------------------------------------------------------------------
# Volume loaders
# ---------------------------------------------------------------------------

def load_from_pt(pt_path: str, image_size_xyz) -> torch.Tensor:
    """Load a pre-converted .pt tensor (created by nifti_to_pt.py).
    Returns (1, D, H, W) float32 in [0, 1] (HU clamped to [-1000, 1000])."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    img  = data["volume"] if isinstance(data, dict) else data
    img  = img.unsqueeze(0).float().clamp(-1000, 1000)
    img  = (img + 1000) / 2000.0
    target_dhw = list(reversed(image_size_xyz))  # config image_size is (W, H, D)
    return crop_or_pad(img, target_dhw).contiguous()


def load_from_nifti(nifti_path: str, image_size_xyz) -> torch.Tensor:
    """Convert .nii / .nii.gz on the fly via load_and_resample, then
    apply the same crop/pad + HU clamp + normalization pipeline.

    Resamples to the module-level TARGET_SPACING_ZYX = (3.0, 1.0, 1.0)
    in (D, H, W) order — matches what nifti_to_pt.py produced for every
    stored .pt file. See TARGET_SPACING_ZYX docstring above.
    """
    arr, _meta = load_and_resample(nifti_path, TARGET_SPACING_ZYX)
    if arr is None:
        raise RuntimeError(f"load_and_resample returned None for {nifti_path} "
                           f"(unreadable, 4D, or degenerate dims)")
    img = torch.from_numpy(arr).unsqueeze(0).float().clamp(-1000, 1000)
    img = (img + 1000) / 2000.0
    target_dhw = list(reversed(image_size_xyz))
    return crop_or_pad(img, target_dhw).contiguous()


def load_volume(pt_path: str | None, nifti_path: str | None,
                image_size_xyz) -> torch.Tensor:
    """Dispatch: prefer pt_path when present (no conversion needed),
    otherwise fall back to nifti_path (on-the-fly conversion)."""
    if pt_path and os.path.exists(pt_path):
        return load_from_pt(pt_path, image_size_xyz)
    if nifti_path and os.path.exists(nifti_path):
        return load_from_nifti(nifti_path, image_size_xyz)
    raise FileNotFoundError(
        f"neither pt_path nor nifti_path found "
        f"(pt={pt_path!r}, nifti={nifti_path!r})"
    )


def embed_volume(model, img: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run the vision tower forward pass on a single (1, D, H, W) volume
    OR a (B, 1, D, H, W) batch. Returns CPU float32 embedding(s).

    Wraps the AMP-on-CUDA / no-AMP-on-CPU branching so the demo and the
    batch runner both call the same one-liner."""
    if img.dim() == 4:
        img = img.unsqueeze(0)        # (1, D, H, W) -> (1, 1, D, H, W)
    img = img.to(device, non_blocking=True)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                z = model.visual(img)
        else:
            z = model.visual(img)
    return z.cpu().float()


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------

def build_percival(config_path: str, weights_path: str, device: torch.device) -> Percival:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    mdl = cfg["model"]

    print(f"[INFO] building Percival '{mdl['name']}' (size={mdl['vision_model_size']}); "
          f"loading visual weights from {weights_path}", flush=True)
    model = Percival(
        name=mdl["name"],
        in_channels=mdl["in_channels"],
        projection_dim=mdl["projection_dim"],
        patch_size=tuple(mdl["patch_size"]),
        img_size=tuple(mdl["image_size"]),
        language_model=mdl.get("language_model",
                               "microsoft/BiomedVLP-CXR-BERT-specialized"),
        vision_model_size=mdl["vision_model_size"],
        vision_pretrain=mdl.get("vision_pretrain", "augreg"),
        freeze_language_model=mdl.get("freeze_language_model", False),
        use_distributed_loss=False,
        loss_type=mdl.get("loss_type", "clip"),
    )
    # strict=False — checkpoint stores only the visual tower; language
    # tower stays at construction-time init and is never touched during
    # embedding extraction.
    model.load_visual(weights_path, strict=False)
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Single-volume mode
# ---------------------------------------------------------------------------

def run_single(args, device):
    if args.pt_path and args.nifti_path:
        raise ValueError("pass either --pt-path or --nifti-path, not both")
    if not (args.pt_path or args.nifti_path):
        raise ValueError("single-volume mode requires --pt-path OR --nifti-path")
    if not args.output:
        raise ValueError("single-volume mode requires --output (where to write the embedding .pt)")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    mdl = cfg["model"]
    image_size_xyz = tuple(mdl["image_size"])

    model = build_percival(args.config, args.weights, device)

    t0 = time.time()
    img = load_volume(args.pt_path, args.nifti_path, image_size_xyz)
    z = embed_volume(model, img, device).squeeze(0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z, out_path)
    print(f"[OK] wrote {out_path}  (embedding shape={tuple(z.shape)}, "
          f"{time.time()-t0:.2f}s)", flush=True)


# ---------------------------------------------------------------------------
# Manifest (batch) mode
# ---------------------------------------------------------------------------

def run_manifest(args, device):
    in_xlsx = Path(args.input_xlsx)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_xlsx is None:
        out_xlsx = in_xlsx.with_name(in_xlsx.stem + "_with_embed.xlsx")
    else:
        out_xlsx = Path(args.output_xlsx)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    mdl = cfg["model"]
    image_size_xyz = tuple(mdl["image_size"])

    df = pd.read_excel(in_xlsx)
    has_pt    = PT_PATH_COLUMN    in df.columns
    has_nifti = NIFTI_PATH_COLUMN in df.columns
    if not (has_pt or has_nifti):
        raise RuntimeError(
            f"manifest must have at least one of '{PT_PATH_COLUMN}' or "
            f"'{NIFTI_PATH_COLUMN}' columns. present: {list(df.columns)}"
        )

    model = build_percival(args.config, args.weights, device)

    # Per-row plan: pick a source path, derive a deterministic output path.
    sources    = [None] * len(df)   # tuples (kind, path)
    embed_paths = [""]   * len(df)
    todo_rows  = []

    for i, row in df.iterrows():
        pt    = str(row[PT_PATH_COLUMN])    if has_pt    and pd.notna(row[PT_PATH_COLUMN])    else ""
        nifti = str(row[NIFTI_PATH_COLUMN]) if has_nifti and pd.notna(row[NIFTI_PATH_COLUMN]) else ""
        if pt and os.path.exists(pt):
            src = ("pt", pt)
            embed_key = pt
        elif nifti and os.path.exists(nifti):
            src = ("nifti", nifti)
            embed_key = nifti
        else:
            continue
        sources[i] = src
        embed_paths[i] = str(out_dir / _hash_filename(embed_key))
        if not os.path.exists(embed_paths[i]):
            todo_rows.append(i)

    n_with_src = sum(1 for s in sources if s is not None)
    print(f"[INFO] manifest rows: {len(df)}  with input: {n_with_src}  "
          f"already done: {n_with_src - len(todo_rows)}  to extract: {len(todo_rows)}",
          flush=True)

    # Batched forward passes
    t0 = time.time()
    batch_imgs   = []
    batch_out    = []
    batch_rowidx = []

    def flush():
        if not batch_imgs:
            return
        try:
            stacked = torch.stack(batch_imgs)   # (B, 1, D, H, W)
            z = embed_volume(model, stacked, device)
            for k, p in enumerate(batch_out):
                torch.save(z[k].clone(), p)
        except Exception as e:
            print(f"[FAIL] batch starting row {batch_rowidx[0]}: {e}", flush=True)
            for i in batch_rowidx:
                embed_paths[i] = ""

    for n, i in enumerate(todo_rows, 1):
        kind, path = sources[i]
        try:
            if kind == "pt":
                img = load_from_pt(path, image_size_xyz)
            else:
                img = load_from_nifti(path, image_size_xyz)
        except Exception as e:
            print(f"[FAIL] row {i} ({kind}={path}): {e}", flush=True)
            embed_paths[i] = ""
            continue
        batch_imgs.append(img)
        batch_out.append(embed_paths[i])
        batch_rowidx.append(i)

        if len(batch_imgs) >= args.batch_size or n == len(todo_rows):
            flush()
            batch_imgs.clear()
            batch_out.clear()
            batch_rowidx.clear()
            if (n % (10 * args.batch_size) == 0) or (n == len(todo_rows)):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"[{n:>6}/{len(todo_rows)}]  {rate:.2f} scans/s  "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)

    df_out = df.copy()
    df_out[EMBED_PATH_COLUMN] = [
        p if (p and os.path.exists(p)) else "" for p in embed_paths
    ]
    df_out.to_excel(out_xlsx, index=False)
    n_done = int((df_out[EMBED_PATH_COLUMN] != "").sum())
    print(f"\n[SUMMARY]")
    print(f"  manifest rows   = {len(df)}")
    print(f"  with embed_path = {n_done}")
    print(f"  missing/failed  = {len(df) - n_done}")
    print(f"[INFO] wrote manifest -> {out_xlsx}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="Path to the trained visual-tower .pth checkpoint "
                         "(e.g. visual_epoch_<N>_loss_<L>.pth).")
    ap.add_argument("--config",  required=True,
                    help="YAML config for the model architecture (must match "
                         "the checkpoint's training config — e.g. "
                         "configs/augreg_base_v0.yaml).")

    # Single-volume mode
    ap.add_argument("--pt-path",    type=str, default=None,
                    help="Single .pt tensor input (single-volume mode).")
    ap.add_argument("--nifti-path", type=str, default=None,
                    help="Single .nii / .nii.gz input (single-volume mode).")
    ap.add_argument("--output",     type=str, default=None,
                    help="Output path for the embedding .pt (single-volume mode).")

    # Manifest mode
    ap.add_argument("--input-xlsx", type=str, default=None,
                    help="Manifest xlsx with pt_path and/or nifti_path columns "
                         "(manifest mode).")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Directory to write per-scan embedding .pt files "
                         "(manifest mode).")
    ap.add_argument("--output-xlsx", type=str, default=None,
                    help="Where to write the manifest with the added "
                         "embed_path column. Default: <input>_with_embed.xlsx.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size for forward passes (manifest mode). "
                         "On CPU keep at 1.")

    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] device = {device}", flush=True)

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")

    if args.input_xlsx is not None:
        run_manifest(args, device)
    else:
        run_single(args, device)


if __name__ == "__main__":
    main()

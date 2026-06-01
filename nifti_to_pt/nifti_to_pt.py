"""
Convert NIfTI volumes to PyTorch tensors with resampling.
Stores full metadata (spacing, origin, direction, orientation) alongside the tensor.

Usage:
    # 1. Prepare chunks
    python nifti_to_pt.py --prepare --n-chunks 50

    # 2. Submit array job
    sbatch nifti_to_pt.sh

    # 3. Summarize results and create merged parquet
    python nifti_to_pt.py --summarize
"""

import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk

# Default paths. Override via CLI flags (--manifest, --src-root, --dst-root,
# --chunk-dir, --results-dir) or by editing these placeholders directly.
MANIFEST_PATH = "<path/to/scans_manifest.xlsx>"
SRC_ROOT      = "<path/to/nifti/source/root>"
DST_ROOT      = "<path/to/pt/output/root>"
CHUNK_DIR     = "<path/to/chunks>"
RESULTS_DIR   = "<path/to/results>"


def load_and_resample(nii_path, target_spacing):
    """Load NIfTI, resample to target spacing, return tensor and metadata.
    
    Returns (None, None) for 4D volumes (3D + time), failed reads, etc.
    """
    try:
        img = sitk.ReadImage(str(nii_path))

        # Drop 3D+time (4D) volumes
        if img.GetDimension() != 3:
            print(f'[SKIP] {nii_path}: {img.GetDimension()}D volume (expected 3D)')
            return None, None

        # Also check for a degenerate 4th dimension stored as size>1 in a 3D container
        img_size = img.GetSize()  # (x, y, z)
        if len(img_size) > 3:
            print(f'[SKIP] {nii_path}: size has {len(img_size)} dimensions {img_size}')
            return None, None

        # Record native orientation before reorienting
        native_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
            img.GetDirection()
        )

        # Reorient to LPS for consistent anatomy across all volumes
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation("LPS")
        img = orient_filter.Execute(img)

        # Extract metadata (now reflects LPS orientation)
        native_spacing = list(img.GetSpacing())       # (x, y, z)
        native_size = list(img.GetSize())              # (x, y, z)
        origin = list(img.GetOrigin())                 # (x, y, z)
        direction = list(img.GetDirection())           # 9 floats (3x3 matrix)

        arr = sitk.GetArrayFromImage(img)  # (D, H, W) — note: SimpleITK reverses axis order
        original_shape = list(arr.shape)   # (D, H, W) in array order

        # Compute target size: current_dim * (native_spacing / target_spacing)
        # arr shape is (D, H, W), native_spacing is (x, y, z)
        z, y, x = native_spacing[2], native_spacing[1], native_spacing[0]
        target_size = (
            int(arr.shape[0] * z / target_spacing[0]),  # D
            int(arr.shape[1] * y / target_spacing[1]),  # H
            int(arr.shape[2] * x / target_spacing[2]),  # W
        )

        # Sanity check: no zero-sized dimensions after resampling
        if any(d <= 0 for d in target_size):
            print(f'[SKIP] {nii_path}: target size has zero/negative dim {target_size} '
                  f'(native_spacing={native_spacing}, shape={arr.shape})')
            return None, None

        # Resample via trilinear interpolation
        arr_t = torch.from_numpy(arr.astype(np.float32))[None, None, ...]
        arr_t = torch.nn.functional.interpolate(arr_t, size=target_size, mode='trilinear').squeeze()

        metadata = {
            'native_orientation': native_orientation,  # original orientation (e.g. "RAS", "RAI")
            'orientation': 'LPS',                      # all volumes reoriented to LPS
            'native_spacing': native_spacing,          # (x, y, z) from header
            'target_spacing': list(target_spacing),    # (z, y, x) as provided
            'native_size': native_size,                # (x, y, z) from header
            'original_shape': original_shape,          # (D, H, W) array shape before resampling
            'origin': origin,
            'direction': direction,
        }

        return arr_t.numpy(), metadata

    except Exception as e:
        print(f'[FAIL] {nii_path}: {e}')
        return None, None


def prepare_chunks(manifest_path, chunk_dir, n_chunks, no_filter=False):
    """Split rows into chunks for array job.

    By default, keeps only CT rows with non-empty reports (the original
    pretraining selection criterion). Pass ``no_filter=True`` to process every
    row in the manifest as-is — useful for converting validation-only manifests
    that lack ``modality`` / ``report_length`` columns or contain rows without
    reports.
    """
    df = pd.read_excel(manifest_path)
    print(f"[INFO] Total manifest rows: {len(df)}")

    if not no_filter:
        df = df[df['modality'] == 'CT'].reset_index(drop=True)
        print(f"[INFO] CT rows: {len(df)}")

        df = df[df['report_length'] > 0].reset_index(drop=True)
        print(f"[INFO] CT rows with reports: {len(df)}")
    else:
        print(f"[INFO] --no-filter set: keeping all {len(df)} rows as-is")

    indices = np.arange(len(df))
    np.random.seed(42)
    np.random.shuffle(indices)

    chunks = np.array_split(indices, n_chunks)
    os.makedirs(chunk_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(chunk_dir, f"chunk_{i}.json")
        with open(chunk_path, 'w') as f:
            json.dump(chunk.tolist(), f)

    # Save the filtered CT-only parquet for consistent indexing
    filtered_path = os.path.join(chunk_dir, "ct_only.parquet")
    df.to_parquet(filtered_path, index=False)

    print(f"[INFO] Split {len(df)} CT rows into {n_chunks} chunks (~{len(df) // n_chunks} each)")
    print(f"[INFO] Chunks saved to {chunk_dir}")
    print(f"[INFO] Filtered parquet saved to {filtered_path}")


def process_chunk(chunk_path, chunk_id, src_root, dst_root, results_dir,
                  target_spacing, save_astype, chunk_dir):
    """Process all rows in a single chunk."""
    # Load the filtered CT-only parquet (saved during prepare)
    filtered_parquet = os.path.join(chunk_dir, "ct_only.parquet")
    df = pd.read_parquet(filtered_parquet)

    with open(chunk_path, 'r') as f:
        indices = json.load(f)

    chunk_df = df.iloc[indices]
    print(f"[INFO] Chunk {chunk_id}: processing {len(chunk_df)} files")
    print(f"[INFO] Target spacing (z, y, x): {target_spacing}")
    print(f"[INFO] Source root: {src_root}")
    print(f"[INFO] Dest root:   {dst_root}")

    results = []
    success = 0
    skipped = 0
    failed = 0
    dropped_4d = 0

    for _, row in chunk_df.iterrows():
        nii_path = str(row['nii_path'])
        name = os.path.basename(nii_path).replace('.nii.gz', '').replace('.nii', '')

        # Mirror directory structure: swap src_root for dst_root, swap extension
        rel = os.path.relpath(nii_path, src_root)
        rel = rel.replace('.nii.gz', '.pt').replace('.nii', '.pt')
        save_path = os.path.join(dst_root, rel)

        # Skip if already converted
        if os.path.exists(save_path):
            results.append({
                'name': name,
                'nii_path': nii_path,
                'pt_path': save_path,
                'success': 1,
            })
            skipped += 1
            continue

        arr, metadata = load_and_resample(nii_path, target_spacing)

        if arr is None:
            results.append({
                'name': name,
                'nii_path': nii_path,
                'pt_path': '',
                'success': 0,
            })
            failed += 1
            continue

        # Build tensor with metadata
        if save_astype == 'float32':
            volume = torch.from_numpy(arr).to(torch.float32)
        elif save_astype == 'float16':
            volume = torch.from_numpy(arr).to(torch.float16)

        data = {
            'volume': volume,
            'native_orientation': metadata['native_orientation'],
            'orientation': metadata['orientation'],
            'native_spacing': metadata['native_spacing'],
            'target_spacing': metadata['target_spacing'],
            'native_size': metadata['native_size'],
            'original_shape': metadata['original_shape'],
            'origin': metadata['origin'],
            'direction': metadata['direction'],
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(data, save_path)

        results.append({
            'name': name,
            'nii_path': nii_path,
            'pt_path': save_path,
            'success': 1,
        })
        success += 1

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"chunk_{chunk_id}.csv")
    pd.DataFrame(results).to_csv(results_path, index=False)

    print(f"[DONE] Chunk {chunk_id}: success={success}, skipped={skipped}, failed={failed}")
    print(f"[DONE] Results saved to {results_path}")


def summarize(results_dir, chunk_dir, manifest_path):
    """Aggregate results and merge pt_path back into the manifest."""
    all_results = []
    for f in sorted(Path(results_dir).glob("chunk_*.csv")):
        all_results.append(pd.read_csv(f))

    if not all_results:
        print("[ERROR] No result files found")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    total = len(results_df)
    n_success = int(results_df['success'].sum())
    n_failed = total - n_success

    print(f"[SUMMARY] Total files: {total}")
    print(f"[SUMMARY] Success: {n_success}")
    print(f"[SUMMARY] Failed: {n_failed}")

    if n_failed > 0:
        print(f"\n[FAILURES] ({n_failed} files)")
        failures = results_df[results_df['success'] == 0]
        for _, row in failures.head(20).iterrows():
            print(f"  {row['nii_path']}")
        if n_failed > 20:
            print(f"  ... and {n_failed - 20} more")

    # Merge with original manifest
    original_df = pd.read_excel(manifest_path)
    print(f"\n[INFO] Original manifest: {len(original_df)} rows")

    # Only keep successful conversions for merge
    success_df = results_df[results_df['success'] == 1][['nii_path', 'pt_path']].copy()

    merged_df = original_df.merge(success_df, on='nii_path', how='left')
    n_with_pt = merged_df['pt_path'].notna().sum()
    print(f"[INFO] Merged manifest: {len(merged_df)} rows ({n_with_pt} with pt_path)")

    # Save
    results_summary_path = os.path.join(results_dir, "summary.csv")
    results_df.to_csv(results_summary_path, index=False)

    merged_path = str(manifest_path).replace('.xlsx', '_with_pt.xlsx')
    merged_df.to_excel(merged_path, index=False)

    print(f"[INFO] Results summary saved to {results_summary_path}")
    print(f"[INFO] Merged manifest saved to {merged_path}")


def main():
    parser = argparse.ArgumentParser('Convert NIfTI to PT')
    parser.add_argument('--manifest-path', default=MANIFEST_PATH)
    parser.add_argument('--src-root', default=SRC_ROOT)
    parser.add_argument('--dst-root', default=DST_ROOT)
    parser.add_argument('--chunk-dir', default=CHUNK_DIR)
    parser.add_argument('--results-dir', default=RESULTS_DIR)
    parser.add_argument('--save-astype', default='float16', choices=['float32', 'float16'])
    parser.add_argument('--spacing', nargs='+', default=[3, 1, 1], type=float,
                        help='Target spacing in (z, y, x) order')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--n-chunks', type=int, default=50)
    parser.add_argument('--chunk-id', type=int, default=None)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip the modality==CT and report_length>0 filters '
                             'when preparing chunks. Use for validation manifests '
                             'that lack those columns or include report-less rows.')
    args = parser.parse_args()

    if args.prepare:
        prepare_chunks(args.manifest_path, args.chunk_dir, args.n_chunks,
                       no_filter=args.no_filter)
    elif args.summarize:
        summarize(args.results_dir, args.chunk_dir, args.manifest_path)
    else:
        chunk_id = args.chunk_id
        if chunk_id is None:
            chunk_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

        chunk_path = os.path.join(args.chunk_dir, f"chunk_{chunk_id}.json")
        if not os.path.exists(chunk_path):
            print(f"[ERROR] Chunk not found: {chunk_path}")
            return

        process_chunk(
            chunk_path, chunk_id,
            args.src_root, args.dst_root,
            args.results_dir,
            args.spacing, args.save_astype,
            args.chunk_dir,
        )


if __name__ == '__main__':
    main()

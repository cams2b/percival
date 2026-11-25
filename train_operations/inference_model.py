"""
InferenceModel — single Percival CT vision-language inference object that
serves BOTH disease classification (logistic linear-probe per ICD code)
AND prognostic Cox proportional-hazards risk scoring per ICD code.

Wraps:
  - Percival visual tower forward pass (NIfTI or .pt input)
  - Per-region linear-probe classifiers shipped at
        inference/classification_weights/{chest,abd_pel}/
  - Per-region Cox proportional-hazards models shipped at
        inference/cox_weights/{chest,abd_pel}/

Public API:

    model = InferenceModel(img_weights="/path/to/visual_epoch_*.pth")

    # tiny / small backbones: `projection_dim` follows `vision_model_size`.
    # The shipped diagnostic / prognostic tables are 768-d (fit on
    # augreg_base embeddings), so tiny / small models support `embed()` only.
    model = InferenceModel(img_weights="/path/to/visual_epoch_*.pth",
                           vision_model_size="tiny")

    # one-shot diagnostic across all regions
    df, summary = model.diagnostic_inference_all_conditions(img_path="scan.nii.gz")

    # one-shot prognostic across all regions
    df, summary = model.prognostic_inference_all_conditions(img_path="scan.nii.gz")

    # single region variants
    df, summary = model.diagnostic_inference(img_path="scan.nii.gz", region="CHEST")
    df, summary = model.prognostic_inference(img_path="scan.nii.gz", region="ABD_PEL")

    # embed once, score twice (skips the second forward pass)
    emb = model.embed(img_path="scan.nii.gz")
    df_d, _ = model.diagnostic_inference_all_conditions(embedding=emb)
    df_p, _ = model.prognostic_inference_all_conditions(embedding=emb)

Each scoring method returns:
    df       : pd.DataFrame with per-code predictions (see schemas below)
    summary  : dict with embedding norm, n codes scored, n high-risk calls,
               and top-K most-confident calls

Schemas:

    diagnostic   : code, region, prob, high_risk, youden_thresh, note
    prognostic   : code, region, lp, hazard_ratio, high_risk,
                   high_risk_threshold, enrichment
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from train_operations.percival import Percival


# ---------------------------------------------------------------------------
# Constants — file paths for shipped weights + the lex permutation
# ---------------------------------------------------------------------------

# Project root is the directory containing the `inference/` folder.
_RELEASE_ROOT = Path(__file__).resolve().parent.parent
_INFERENCE_DIR = _RELEASE_ROOT / "inference"

# Per-(task, region) shipped weight files.
CLASSIFIER_FILES = {
    "CHEST": (
        _INFERENCE_DIR / "classification_weights/chest/weights_percival_window60_grp-raw_region-CHEST_merged.csv",
        _INFERENCE_DIR / "classification_weights/chest/scaler_percival_window60_grp-raw_region-CHEST_merged.csv",
    ),
    "ABD_PEL": (
        _INFERENCE_DIR / "classification_weights/abd_pel/weights_percival_window60_grp-raw_region-ABD_PEL_merged.csv",
        _INFERENCE_DIR / "classification_weights/abd_pel/scaler_percival_window60_grp-raw_region-ABD_PEL_merged.csv",
    ),
}

COX_FILES = {
    "CHEST":   _INFERENCE_DIR / "cox_weights/chest/survival_weights_dedup-youngest_per_person_grp-raw_region-CHEST_merged.csv",
    "ABD_PEL": _INFERENCE_DIR / "cox_weights/abd_pel/survival_weights_dedup-youngest_per_person_grp-raw_region-ABD_PEL_merged.csv",
}

# Percival embedding dimension (augreg_base_v0).
PROJECTION_DIM = 768

# Embedding dimension per backbone size (configs/augreg_{size}_v0.yaml).
PROJECTION_DIMS = {"tiny": 192, "small": 384, "base": PROJECTION_DIM}


# ---------------------------------------------------------------------------
# Internal helpers (deferred import so the file is importable without torch)
# ---------------------------------------------------------------------------

def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class InferenceModel:
    """Percival inference object — diagnostic + prognostic in one place."""

    def __init__(
        self,
        img_weights: str,
        vision_model_size: str = "base",
        vision_pretrain: str = "augreg",
        in_channels: int = 1,
        projection_dim: Optional[int] = None,      # None -> by vision_model_size
        image_size: tuple = (352, 352, 128),       # (W, H, D)
        target_spacing: tuple = (3.0, 1.0, 1.0),   # (z, y, x) for NIfTI resample
        patch_size: tuple = (8, 16, 16),
        language_model: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        device: Union[str, torch.device] = "auto",
        strict: bool = True,
    ):
        # ---- Build Percival, load visual checkpoint, set eval mode ----
        self.device = _resolve_device(device)
        print(f"[InferenceModel] device = {self.device}")

        if projection_dim is None:
            projection_dim = PROJECTION_DIMS.get(vision_model_size, PROJECTION_DIM)

        self.image_size      = tuple(image_size)
        self.target_spacing  = tuple(target_spacing)
        self.projection_dim  = projection_dim

        print(f"[InferenceModel] building Percival "
              f"({vision_model_size}, pretrain={vision_pretrain}, "
              f"proj_dim={projection_dim})")
        self.model = Percival(
            name="percival",
            in_channels=in_channels,
            projection_dim=projection_dim,
            patch_size=tuple(patch_size),
            img_size=self.image_size,
            language_model=language_model,
            vision_model_size=vision_model_size,
            vision_pretrain=vision_pretrain,
            freeze_language_model=False,
            use_distributed_loss=False,
            loss_type="clip",
        )
        # load_visual only loads into the visual tower (the language tower
        # is HF-initialized and never touched during inference), so a
        # matching checkpoint loads with strict=True. With strict=False,
        # missing / unexpected keys are tolerated and any shape-mismatched
        # tensor is dropped and left at its construction-time init (random
        # for the projection head when `projection_dim` does not match the
        # checkpoint), with only a printed [WARN].
        self.model.load_visual(str(img_weights), strict=strict)
        self.model.eval().to(self.device)

        # ---- Caches for shipped per-region weight tables (lazy) ----
        self._classifier_cache: dict[str, dict] = {}
        self._cox_cache:        dict[str, dict] = {}

    # ----------------------------------------------------------------------
    # Public: embedding
    # ----------------------------------------------------------------------

    def embed(self, img_path: str) -> np.ndarray:
        """Forward-prop one CT through the visual tower. Auto-detects
        NIfTI (.nii / .nii.gz) vs pre-resampled PyTorch tensor (.pt) and
        dispatches to the right preprocessing pipeline.

        Returns
        -------
        np.ndarray  shape (projection_dim,)
        """
        # Imports kept local so importing this module doesn't drag in
        # SimpleITK / MONAI unless someone actually runs inference.
        from extract_embeddings import (
            load_from_pt, load_from_nifti, embed_volume,
        )

        p = Path(img_path)
        suffixes = "".join(p.suffixes).lower()
        if suffixes.endswith(".pt"):
            img = load_from_pt(str(p), self.image_size)
        elif suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
            img = load_from_nifti(str(p), self.image_size)
        else:
            raise ValueError(
                f"unsupported input file type: {p.name!r} "
                f"(expected .pt / .nii / .nii.gz)"
            )
        emb_t = embed_volume(self.model, img, self.device)   # (1, D) or (D,)
        emb   = emb_t.squeeze().detach().cpu().float().numpy()
        if emb.shape != (self.projection_dim,):
            raise RuntimeError(
                f"expected embedding shape ({self.projection_dim},); "
                f"got {emb.shape}"
            )
        return emb

    # ----------------------------------------------------------------------
    # Public: diagnostic (logistic linear-probe)
    # ----------------------------------------------------------------------

    def diagnostic_inference(
        self,
        img_path: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        region: str = "CHEST",
    ) -> tuple[pd.DataFrame, dict]:
        """Score one CT against all per-ICD-code classifiers in `region`.

        Returns (df, summary). df columns:
            code, region, prob, high_risk, youden_thresh, note
        """
        emb = self._resolve_embedding(img_path, embedding)
        tbl = self._load_classifier_table(region)
        df  = self._score_diagnostic(emb, tbl)
        return df, self._summarize_diagnostic(df, emb)

    def diagnostic_inference_all_conditions(
        self,
        img_path: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Run diagnostic_inference across every region and concatenate."""
        emb = self._resolve_embedding(img_path, embedding)
        frames = []
        for region in sorted(CLASSIFIER_FILES.keys()):
            tbl = self._load_classifier_table(region)
            frames.append(self._score_diagnostic(emb, tbl))
        df = pd.concat(frames, ignore_index=True)
        return df, self._summarize_diagnostic(df, emb)

    # ----------------------------------------------------------------------
    # Public: prognostic (Cox PH)
    # ----------------------------------------------------------------------

    def prognostic_inference(
        self,
        img_path: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        region: str = "CHEST",
    ) -> tuple[pd.DataFrame, dict]:
        """Score one CT against all per-ICD-code Cox models in `region`.

        Returns (df, summary). df columns:
            code, region, lp, hazard_ratio, high_risk,
            high_risk_threshold, enrichment
        """
        emb = self._resolve_embedding(img_path, embedding)
        tbl = self._load_cox_table(region)
        df  = self._score_prognostic(emb, tbl)
        return df, self._summarize_prognostic(df, emb)

    def prognostic_inference_all_conditions(
        self,
        img_path: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Run prognostic_inference across every region and concatenate."""
        emb = self._resolve_embedding(img_path, embedding)
        frames = []
        for region in sorted(COX_FILES.keys()):
            tbl = self._load_cox_table(region)
            frames.append(self._score_prognostic(emb, tbl))
        df = pd.concat(frames, ignore_index=True)
        return df, self._summarize_prognostic(df, emb)

    # ----------------------------------------------------------------------
    # Private — embedding resolution
    # ----------------------------------------------------------------------

    def _resolve_embedding(self, img_path, embedding) -> np.ndarray:
        if (img_path is None) == (embedding is None):
            raise ValueError("pass exactly one of `img_path` or `embedding`")
        if embedding is not None:
            emb = np.asarray(embedding, dtype=float).reshape(-1)
            if emb.shape != (self.projection_dim,):
                raise ValueError(
                    f"embedding must be shape ({self.projection_dim},); "
                    f"got {emb.shape}"
                )
            return emb
        return self.embed(img_path)

    # ----------------------------------------------------------------------
    # Private — shipped weight-table column selection
    # ----------------------------------------------------------------------

    def _table_columns(self, df: pd.DataFrame, prefix: str, path: Path) -> list:
        """Return the `{prefix}_0 .. {prefix}_{D-1}` column names of a weight
        table (D = self.projection_dim), after checking that those are
        exactly the `{prefix}_*` columns the table has."""
        cols = [f"{prefix}_{i}" for i in range(self.projection_dim)]
        n = sum(1 for c in df.columns if c.startswith(f"{prefix}_"))
        if n != len(cols) or not set(cols) <= set(df.columns):
            raise ValueError(
                f"{path.name} has {n} '{prefix}_*' columns, but this model "
                f"produces {self.projection_dim}-dimensional embeddings, so "
                f"the table cannot score them. (The tables shipped in "
                f"inference/ were fit on augreg_base, {PROJECTION_DIM}-d, "
                f"embeddings.)"
            )
        return cols

    # ----------------------------------------------------------------------
    # Private — classifier table loader (cached) + scoring
    # ----------------------------------------------------------------------

    def _load_classifier_table(self, region: str) -> dict:
        if region in self._classifier_cache:
            return self._classifier_cache[region]
        if region not in CLASSIFIER_FILES:
            raise ValueError(
                f"unknown region {region!r}. "
                f"Known: {sorted(CLASSIFIER_FILES)}"
            )
        w_path, s_path = CLASSIFIER_FILES[region]
        if not w_path.is_file() or not s_path.is_file():
            raise FileNotFoundError(
                f"shipped classification weights missing for {region}: "
                f"{w_path}"
            )
        print(f"[InferenceModel] loading classifier weights for {region}")
        w_df = pd.read_csv(w_path).set_index("code", drop=False)
        s_df = pd.read_csv(s_path).set_index("code", drop=False)

        common = w_df.index.intersection(s_df.index)
        w_df, s_df = w_df.loc[common], s_df.loc[common]

        w_cols  = self._table_columns(w_df, "w",     w_path)
        m_cols  = self._table_columns(s_df, "mean",  s_path)
        sc_cols = self._table_columns(s_df, "scale", s_path)

        tbl = {
            "codes":         w_df["code"].astype(str).to_numpy(),
            "W":             w_df[w_cols].to_numpy(dtype=float),
            "intercept":     w_df["intercept"].to_numpy(dtype=float),
            "youden_thresh": w_df["youden_thresh"].to_numpy(dtype=float),
            "mean":          s_df[m_cols].to_numpy(dtype=float),
            "scale":         s_df[sc_cols].to_numpy(dtype=float),
            "note":          w_df["note"].fillna("").astype(str).to_numpy(),
            "region":        region,
        }
        self._classifier_cache[region] = tbl
        return tbl

    def _score_diagnostic(self, emb: np.ndarray, tbl: dict) -> pd.DataFrame:
        """Vectorized logistic linear-probe scoring per ICD code:
            z          = (x - scaler_mean) / scaler_scale
            z          = z / ||z||_2
            logit      = z @ coef + intercept
            prob       = 1 / (1 + exp(-logit))
            high_risk  = prob >= youden_thresh
        """
        z = (emb[None, :] - tbl["mean"]) / tbl["scale"]                # (K, D)
        z_norm = np.linalg.norm(z, axis=1, keepdims=True)
        z_norm = np.where(z_norm == 0, 1.0, z_norm)
        z = z / z_norm

        logit = (z * tbl["W"]).sum(axis=1) + tbl["intercept"]          # (K,)
        prob  = 1.0 / (1.0 + np.exp(-logit))                           # (K,)

        thr = tbl["youden_thresh"]
        valid = np.isfinite(prob) & np.isfinite(thr)
        high_risk = np.full_like(prob, np.nan)
        high_risk[valid] = (prob[valid] >= thr[valid]).astype(float)

        df = pd.DataFrame({
            "code":          tbl["codes"],
            "region":        tbl["region"],
            "prob":          prob,
            "high_risk":     high_risk,
            "youden_thresh": thr,
            "note":          tbl["note"],
        })
        # Drop unfitted codes silently (kept in the table for completeness).
        return df[df["note"] == ""].reset_index(drop=True)

    @staticmethod
    def _summarize_diagnostic(df: pd.DataFrame, emb: np.ndarray) -> dict:
        return {
            "embedding_l2_norm": float(np.linalg.norm(emb)),
            "n_codes_scored":    int(len(df)),
            "n_high_risk":       int((df["high_risk"] == 1).sum()),
            "top_5_by_prob":     df.nlargest(5, "prob")[
                ["code", "region", "prob", "high_risk"]
            ].to_dict("records"),
        }

    # ----------------------------------------------------------------------
    # Private — cox table loader (cached) + scoring
    # ----------------------------------------------------------------------

    def _load_cox_table(self, region: str) -> dict:
        if region in self._cox_cache:
            return self._cox_cache[region]
        if region not in COX_FILES:
            raise ValueError(
                f"unknown region {region!r}. Known: {sorted(COX_FILES)}"
            )
        w_path = COX_FILES[region]
        if not w_path.is_file():
            raise FileNotFoundError(
                f"shipped cox weights missing for {region}: {w_path}"
            )
        print(f"[InferenceModel] loading cox weights for {region}")
        df = pd.read_csv(w_path).set_index("code", drop=False)

        w_cols = self._table_columns(df, "w", w_path)
        tbl = {
            "codes":               df["code"].astype(str).to_numpy(),
            "W":                   df[w_cols].to_numpy(dtype=float),
            "high_risk_threshold": df["high_risk_threshold"].to_numpy(dtype=float),
            "enrichment":          df["enrichment"].to_numpy(dtype=float),
            "region":              region,
        }
        self._cox_cache[region] = tbl
        return tbl

    @staticmethod
    def _score_prognostic(emb: np.ndarray, tbl: dict) -> pd.DataFrame:
        """Vectorized Cox PH scoring per ICD code:
            lp           = W @ x                   # log hazard ratio
            hazard_ratio = exp(lp)
            high_risk    = lp >= high_risk_threshold
        No scaling, no L2 norm, no intercept (Cox baseline absorbs it).
        """
        lp           = tbl["W"] @ emb                              # (K,)
        hazard_ratio = np.exp(lp)                                  # (K,)

        thr = tbl["high_risk_threshold"]
        valid = np.isfinite(lp) & np.isfinite(thr)
        high_risk = np.full_like(lp, np.nan)
        high_risk[valid] = (lp[valid] >= thr[valid]).astype(float)

        return pd.DataFrame({
            "code":                tbl["codes"],
            "region":              tbl["region"],
            "lp":                  lp,
            "hazard_ratio":        hazard_ratio,
            "high_risk":           high_risk,
            "high_risk_threshold": thr,
            "enrichment":          tbl["enrichment"],
        })

    @staticmethod
    def _summarize_prognostic(df: pd.DataFrame, emb: np.ndarray) -> dict:
        return {
            "embedding_l2_norm": float(np.linalg.norm(emb)),
            "n_codes_scored":    int(len(df)),
            "n_high_risk":       int((df["high_risk"] == 1).sum()),
            "top_5_by_lp":       df.nlargest(5, "lp")[
                ["code", "region", "lp", "hazard_ratio", "high_risk"]
            ].to_dict("records"),
        }

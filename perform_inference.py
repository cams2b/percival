"""
Front-facing Percival inference script.

Edit the paths in `classification_inference()` and `prognostic_inference()`
below, then run:

    python perform_inference.py
"""

from train_operations.inference_model import InferenceModel


# ---- Shared model config (augreg_base_v0 defaults are the InferenceModel
# defaults, so for the headline model you only need to set `img_weights`) ----
IMG_WEIGHTS = "<path to visual encoder>/visual_epoch_8_loss_1.3.pth"


def classification_inference():
    print("[INFO] performing classification")
    model = InferenceModel(img_weights=IMG_WEIGHTS)
    results, summary = model.diagnostic_inference_all_conditions(
        img_path="<path to image>.nii.gz"
    )
    print(summary)
    results.to_csv("predictions_diagnostic.csv", index=False)


def prognostic_inference():
    print("[INFO] performing prognostic risk scoring")
    model = InferenceModel(img_weights=IMG_WEIGHTS)
    results, summary = model.prognostic_inference_all_conditions(
        img_path="<path to image>.nii.gz"
    )
    print(summary)
    results.to_csv("predictions_prognostic.csv", index=False)


def both_in_one_pass():
    """Embed once, score both heads. Faster than calling the two
    functions above back-to-back (which would re-build the model and
    re-run the forward pass)."""
    model = InferenceModel(img_weights=IMG_WEIGHTS)
    emb = model.embed("<path to image>.nii.gz")

    diag_df, diag_summary = model.diagnostic_inference_all_conditions(embedding=emb)
    prog_df, prog_summary = model.prognostic_inference_all_conditions(embedding=emb)

    print("[diagnostic]", diag_summary)
    print("[prognostic]", prog_summary)
    diag_df.to_csv("predictions_diagnostic.csv", index=False)
    prog_df.to_csv("predictions_prognostic.csv", index=False)


if __name__ == "__main__":
    classification_inference()

# coding=utf-8
# Copyright 2025 The Percival Foundation model Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import numpy as np
from scipy.special import expit
import pandas as pd
import torch
import torch.nn as nn
from monai.transforms import (Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing,
                              ScaleIntensityRange, SpatialPad, CenterSpatialCrop, ToTensor)

from train_operations.percival import percival




def get_inference_transform(image_size=(256, 256, 128), target_spacing=(1.5, 1.5, 3)):
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=target_spacing, mode="bilinear"),
        ScaleIntensityRange(-1000, 1000, 0.0, 1.0, clip=True),
        SpatialPad(spatial_size=image_size),
        CenterSpatialCrop(roi_size=image_size),
        ToTensor(),
    ])



class inference_model(nn.Module):
    def __init__(self, 
                 vision_model_size:str='small',
                 in_channels:int=1,
                 projection_dim:int=512,
                 language_model:str='yikuan8/Clinical-Longformer',
                 image_size:tuple=(256, 256, 128),
                 target_spacing:tuple=(1.5, 1.5, 3),
                 img_weights:str=None, 
                 lang_weights:str=None,
                 classification_coef_path:str="train_operations/data/classification/classification_coefs.xlsx",
                 prognostication_coef_path:str="train_operations/data/prognosis/prognostic_coefs.xlsx",
                 pc_center_scale_path:str='train_operations/data/pca_center_scale.csv',
                 pc_rotation_path:str='train_operations/data/pca_rotation_matrix.csv'
                 ):
        super().__init__()
        self.vision_model_size = vision_model_size
        self.in_channels = in_channels
        self.projection_dim = projection_dim
        self.language_model = language_model
        self.image_size = image_size
        self.target_spacing = target_spacing
        self.img_weights = img_weights
        self.lang_weights = lang_weights
        self.classification_coef_path = classification_coef_path
        self.prognostication_coef_path = prognostication_coef_path
        self.pc_center_scale_path = pc_center_scale_path
        self.pc_rotation_path = pc_rotation_path

        self.center_scale = pd.read_csv(self.pc_center_scale_path)
        self.rotation = pd.read_csv(self.pc_rotation_path)
        self.classification_coef_df = pd.read_excel(self.classification_coef_path) 
        self.prognostication_coef_df = pd.read_excel(self.prognostication_coef_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_model()
        self.initialize_transform()

    def initialize_transform(self):
        print('[INFO] initializing transform matrix...')
        self.center = self.center_scale['center'].values
        self.scale = self.center_scale['scale'].values
        self.rotation_matrix = self.rotation[[f'PC{i}' for i in range(1, 11)]].values
        print('[INFO] success')


    def initialize_model(self):
        print('[INFO] initializing model....')
        self.model = percival(name='percival',
                     in_channels=self.in_channels,
                     projection_dim=self.projection_dim,
                     language_model=self.language_model,
                     img_size=self.image_size,
                     vision_model_size=self.vision_model_size)
        
        self.model.to(self.device)
        print('[INFO] success')
    

    def inference_from_path(self, img_path):
        self.model.eval()
        transforms = get_inference_transform(self.image_size, target_spacing=self.target_spacing)
        img = transforms(img_path)
        img = torch.swapaxes(img, 1, -1) 
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            z_img = self.model.encode_image(img, normalize=False).detach().cpu()
        z_img = z_img.numpy()

        return z_img

    def compute_principal_components(self, z_img):
        assert z_img.shape[1] == len(self.center), "[ERROR] z_img shape mismatch with PCA center"
        
        z_scaled = (z_img - self.center) / self.scale
        pc_scores = np.dot(z_scaled, self.rotation_matrix)

        return pc_scores


    def diagnostic_inference(self, img_path, target_condition=None):
        z_img = self.inference_from_path(img_path=img_path)
        pc = self.compute_principal_components(z_img=z_img)

        if target_condition is None:
            raise ValueError("Please specify a target_condition for inference")

        # Step 3: Get coefficients for the target condition
        row = self.classification_coef_df[self.classification_coef_df["phecode"] == target_condition]
        if row.empty:
            raise ValueError(f"Condition '{target_condition}' not found in coefficient table")

        row = row.iloc[0]
        intercept = row["(Intercept)"]
        description = row['description']
        threshold = row['threshold']

        pc_coefs = row[[f"PC{i}" for i in range(1, 11)]].values.astype(np.float32)

        # Step 4: Compute logits and probability
        logits = np.dot(pc, pc_coefs.T) + intercept
        prob = expit(logits[0])

        return {
            "principal_components": pc,
            "phecode": target_condition,
            "description": description,
            "predicted_probability": prob,
            "predicted_label": int(prob >= threshold) 
        }
    
    def extract_latent_components(self, img_path):
        z_img = self.inference_from_path(img_path=img_path)
        pc = self.compute_principal_components(z_img=z_img)

        return z_img, pc

    def diagnostic_inference_all_conditions(self, img_path):
        z_img = self.inference_from_path(img_path=img_path)
        pc = self.compute_principal_components(z_img=z_img)

        results = []
        for _, row in self.classification_coef_df.iterrows():
            condition = row["phecode"]
            description = row["description"]
            intercept = row["(Intercept)"]
            threshold = row['threshold']


            try:
                pc_coefs = row[[f"PC{i}" for i in range(1, 11)]].values.astype(np.float32)
            except KeyError:
                print(f"[WARNING] PC columns missing for condition: {condition}")
                continue

            # logistic regression: logit = β0 + βᵀx
            logit = np.dot(pc, pc_coefs.T) + intercept
            prob = float(expit(logit[0]))

            predicted_label = 1 if prob >= threshold else 0

            results.append({
                "phecode": condition,
                "description": description,
                "predicted_probability": prob,
                "predicted_label": predicted_label
            })

        df = pd.DataFrame(results)

        # --- Build summary ---
        summary = {
            "n_positive": (df["predicted_label"] == 1).sum(),
            "n_negative": (df["predicted_label"] == 0).sum(),
        }

        positive_examples = df[df["predicted_label"] == 1][["phecode", "description", "predicted_probability"]] \
                                .sort_values("predicted_probability", ascending=False) \
                                .head(10)

        negative_examples = df[df["predicted_label"] == 0][["phecode", "description", "predicted_probability"]] \
                                .sort_values("predicted_probability", ascending=False) \
                                .head(10)

        summary["positive_examples"] = positive_examples.to_dict(orient="records")
        summary["negative_examples"] = negative_examples.to_dict(orient="records")

        print(self.format_diagnostic_summary(summary))

        return df, summary


    
    def prognostic_inference(self, img_path, target_condition=None):
        z_img = self.inference_from_path(img_path=img_path)
        pc = self.compute_principal_components(z_img=z_img)

        if target_condition is None:
            raise ValueError("Please specify a target condition for inference")
        
        row = self.prognostication_coef_df[self.prognostication_coef_df["phecode"] == target_condition]

        row = row.iloc[0]
        description = row['description']
        pc_coefs = row[[f"PC{i}" for i in range(1, 11)]].values.astype(np.float32)
        low_risk_threshold = row['low_risk_threshold']
        high_risk_threshold = row['high_risk_threshold']
        # Compute linear predictor (LP) and hazard
        linear_predictor = np.dot(pc, pc_coefs.T).item()
        relative_hazard = np.exp(linear_predictor)
        if linear_predictor <= low_risk_threshold:
            risk_strata = "low-risk"
        elif linear_predictor > high_risk_threshold:
            risk_strata = "high-risk"
        else:
            risk_strata = "intermediate-risk"
        return {
            "principal_components": pc,
            "condition": target_condition,
            "description": description,
            "linear_predictor": linear_predictor,
            "relative_hazard": relative_hazard,
            "risk_strata": risk_strata
        }

    
    def prognostic_inference_all_conditions(self, img_path):
        # --- Compute PCs ---
        z_img = self.inference_from_path(img_path=img_path)
        pc = self.compute_principal_components(z_img=z_img)

        results = []
        # --- Loop over all coefficient rows ---
        for _, row in self.prognostication_coef_df.iterrows():
            condition = row["phecode"]
            description = row["description"]

            low_risk_threshold = row["low_risk_threshold"]
            high_risk_threshold = row["high_risk_threshold"]

            try:
                pc_coefs = row[[f"PC{i}" for i in range(1, 11)]].values.astype(np.float32)
            except KeyError:
                print(f"[WARNING] PC columns missing for condition: {condition}")
                continue

            # --- Cox model linear predictor ---
            linear_predictor = np.dot(pc, pc_coefs.T).item()
            relative_hazard = np.exp(linear_predictor)      
            # --- Risk category ---
            if linear_predictor <= low_risk_threshold:
                risk_strata = "low-risk"
            elif linear_predictor > high_risk_threshold:
                risk_strata = "high-risk"
            else:
                risk_strata = "intermediate-risk"

            results.append({
                "condition": condition,
                "description": description,
                "linear_predictor": linear_predictor,
                "relative_hazard": relative_hazard,
                "risk_strata": risk_strata
            })

        # --- Convert to DataFrame ---
        df = pd.DataFrame(results)

        # --- Build summary ---
        summary = {
            "n_low_risk": (df["risk_strata"] == "low-risk").sum(),
            "n_intermediate_risk": (df["risk_strata"] == "intermediate-risk").sum(),
            "n_high_risk": (df["risk_strata"] == "high-risk").sum(),
        }

        # Example rows
        intermediate_examples = (df[df["risk_strata"] == "intermediate-risk"][["condition", "description"]].sample(n=5, random_state=42))
        high_examples = (df[df["risk_strata"] == "high-risk"][["condition", "description"]].sample(n=10, random_state=42))

        summary["intermediate_risk_examples"] = intermediate_examples.to_dict(orient="records")
        summary["high_risk_examples"] = high_examples.to_dict(orient="records")

        print(self.format_prognosis_summary(summary))

        return df, summary

    

    def format_prognosis_summary(self, summary):
        lines = []
        lines.append("\n================ PROGNOSTIC SUMMARY ================")
        lines.append(f"Low-risk conditions:          {summary['n_low_risk']}")
        lines.append(f"Intermediate-risk conditions: {summary['n_intermediate_risk']}")
        lines.append(f"High-risk conditions:         {summary['n_high_risk']}")
        lines.append("-----------------------------------------------------")

        # Intermediate-risk examples
        lines.append("\nINTERMEDIATE-RISK (examples up to 5):")
        if len(summary["intermediate_risk_examples"]) == 0:
            lines.append("  None")
        else:
            for ex in summary["intermediate_risk_examples"]:
                lines.append(f"  • {ex['condition']}: {ex['description']}")

        # High-risk examples
        lines.append("\nHIGH-RISK (examples up to 10):")
        if len(summary["high_risk_examples"]) == 0:
            lines.append("  None")
        else:
            for ex in summary["high_risk_examples"]:
                lines.append(f"  • {ex['condition']}: {ex['description']}")

        lines.append("=====================================================\n")

        return "\n".join(lines)


    def format_diagnostic_summary(self, summary):
        lines = []
        lines.append("\n================ DIAGNOSTIC SUMMARY ================")
        lines.append(f"Predicted POSITIVE diagnoses: {summary['n_positive']}")
        lines.append(f"Predicted NEGATIVE diagnoses: {summary['n_negative']}")
        lines.append("-----------------------------------------------------")

        # Positive examples
        lines.append("\nPOSITIVE DIAGNOSES (examples up to 10):")
        if len(summary["positive_examples"]) == 0:
            lines.append("  None")
        else:
            for ex in summary["positive_examples"]:
                prob = f"{ex['predicted_probability']:.3f}"
                lines.append(f"  • {ex['phecode']}: {ex['description']}  (p={prob})")

        # Negative examples
        lines.append("\nNEGATIVE DIAGNOSES (examples up to 10):")
        if len(summary["negative_examples"]) == 0:
            lines.append("  None")
        else:
            for ex in summary["negative_examples"]:
                prob = f"{ex['predicted_probability']:.3f}"
                lines.append(f"  • {ex['phecode']}: {ex['description']}  (p={prob})")

        lines.append("=====================================================\n")

        return "\n".join(lines)


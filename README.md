# Percival

![PyPI - Python Version](https://img.shields.io/badge/python-3.10-blue)


Code repository for Percival: a generalizable vision language foundation model for computed tomography

![Key Graphic](images/percival.png)


## 🧪 Environment setup
To create and activate the conda (percival) environment:
```bash
conda env create -f environment.yml
conda activate percival
```

## Pretrained Models
The pretrained Percival models were trained on over 400,000 CT volumes paired with radiology reports from more than 50,000 Penn Medicine BioBank (PMBB) participants. These models cover multiple anatomical regions and imaging protocols.

| Model                   | Download Link                                      | Base Architecture            | Reference                                      |
|-------------------------|----------------------------------------------------|------------------------------|------------------------------------------------|
| Percival Image Encoder  | [Download](https://zenodo.org/your-image-encoder-link) | DeiT Small Patch16-224       | [DeiT Paper](https://arxiv.org/abs/2012.12877) / [timm](https://rwightman.github.io/pytorch-image-models/) |
| Percival Text Encoder   | [Download](https://zenodo.org/your-text-encoder-link)  | Clinical Longformer (BERT)   | [Clinical Longformer](https://huggingface.co/yikuan8/Clinical-Longformer) |



## 🔍 Diagnosis with Percival
```python

import pandas as pd
import numpy as np
import SimpleITK as sitk
import torch
from train_operations.percival import percival

img_path = '<Path to image (.nii)>'
in_channels = 1
projection_dim = 512
img_weights = '<Path to image encoder>/percival_vision_encoder.pth'
king_percival = percival(in_channels=in_channels, 
                         projection_dim=projection_dim, 
                         img_size=(128, 256, 256))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
king_percival.to(device)
king_percival.load_image_encoder(path=img_weights)
diagnostic_results = king_percival.diagnostic_inference_all_conditions(img_path=img_path, device=device)

```
## Circulatory System Disease Phenotype Models
| Diagnosis                                                          |   Phecode | 5-fold AUROC (95% CI)   |
|:-------------------------------------------------------------------|----------:|:------------------------|
| Heart failure with preserved EF [Diastolic heart failure]          |    428.4  | 0.82 (0.79, 0.86)       |
| Hypertensive heart disease                                         |    401.21 | 0.82 (0.79, 0.84)       |
| Heart failure with reduced EF [Systolic or combined heart failure] |    428.3  | 0.80 (0.77, 0.83)       |
| Heart failure NOS                                                  |    428.2  | 0.79 (0.76, 0.83)       |
| Congestive heart failure (CHF) NOS                                 |    428.1  | 0.79 (0.77, 0.80)       |
| Coronary atherosclerosis                                           |    411.4  | 0.79 (0.78, 0.80)       |
| Paroxysmal ventricular tachycardia                                 |    427.12 | 0.79 (0.76, 0.81)       |
| Atrial fibrillation                                                |    427.21 | 0.78 (0.76, 0.80)       |
| Cardiomegaly                                                       |    416    | 0.78 (0.76, 0.79)       |
| Essential hypertension                                             |    401.1  | 0.77 (0.76, 0.78)       |
| Primary/intrinsic cardiomyopathies                                 |    425.1  | 0.77 (0.74, 0.80)       |
| Angina pectoris                                                    |    411.3  | 0.74 (0.72, 0.75)       |
| Nonspecific chest pain                                             |    418    | 0.72 (0.69, 0.75)       |
| Arrhythmia (cardiac) NOS                                           |    427.5  | 0.70 (0.68, 0.72)       |
| Palpitations                                                       |    427.9  | 0.67 (0.65, 0.69)       |


## Respiratory Disease Phenotype Models
| Diagnosis                                                   |   Phecode | 5-fold AUROC (95% CI)   |
|:------------------------------------------------------------|----------:|:------------------------|
| Pleurisy; pleural effusion                                  |    507    | 0.79 (0.77, 0.80)       |
| Pulmonary collapse; interstitial and compensatory emphysema |    508    | 0.77 (0.75, 0.78)       |
| Empyema and pneumothorax                                    |    506    | 0.75 (0.71, 0.79)       |
| Pneumococcal pneumonia                                      |    480.11 | 0.75 (0.69, 0.82)       |
| Pneumonia                                                   |    480    | 0.74 (0.72, 0.75)       |
| Abnormal findings examination of lungs                      |    514    | 0.73 (0.71, 0.76)       |
| Shortness of breath                                         |    512.7  | 0.72 (0.71, 0.74)       |
| Postinflammatory pulmonary fibrosis                         |    502    | 0.68 (0.65, 0.71)       |
| Cough                                                       |    512.8  | 0.68 (0.65, 0.70)       |
| Asthma                                                      |    495    | 0.63 (0.60, 0.66)       |


## Endocrine/Metabolic Disease Phenotype Models
| Diagnosis                                        |   Phecode | 5-fold AUROC (95% CI)   |
|:-------------------------------------------------|----------:|:------------------------|
| Morbid obesity                                   |    278.11 | 0.90 (0.87, 0.94)       |
| Obesity                                          |    278.1  | 0.86 (0.86, 0.87)       |
| Gout                                             |    274.1  | 0.77 (0.74, 0.81)       |
| Type 2 diabetes                                  |    250.2  | 0.75 (0.73, 0.76)       |
| Polyneuropathy in diabetes                       |    250.6  | 0.74 (0.65, 0.83)       |
| Type 2 diabetes with neurological manifestations |    250.24 | 0.73 (0.65, 0.82)       |
| Hypopotassemia                                   |    276.14 | 0.73 (0.72, 0.74)       |
| Hypovolemia                                      |    276.5  | 0.71 (0.68, 0.73)       |
| Hypercholesterolemia                             |    272.11 | 0.66 (0.63, 0.69)       |
| Nontoxic multinodular goiter                     |    241.2  | 0.64 (0.58, 0.70)       |


## Genitourinary Disease Phenotype Models
| Diagnosis                                                                       |   Phecode | 5-fold AUROC (95% CI)   |
|:--------------------------------------------------------------------------------|----------:|:------------------------|
| Kidney replaced by transplant                                                   |     587   | 0.83 (0.79, 0.87)       |
| Hyperplasia of prostate                                                         |     600   | 0.83 (0.81, 0.85)       |
| Chronic renal failure [CKD]                                                     |     585.3 | 0.80 (0.80, 0.81)       |
| Acute renal failure                                                             |     585.1 | 0.79 (0.77, 0.81)       |
| Renal failure NOS                                                               |     585.2 | 0.75 (0.71, 0.79)       |
| Calculus of kidney                                                              |     594.1 | 0.75 (0.72, 0.77)       |
| Ovarian cyst                                                                    |     628   | 0.70 (0.67, 0.73)       |
| Chronic kidney disease, Stage I or II                                           |     585.4 | 0.70 (0.66, 0.74)       |
| Disorders of menstruation and other abnormal bleeding from female genital tract |     626   | 0.69 (0.64, 0.74)       |
| Elevated prostate specific antigen [PSA]                                        |     796   | 0.62 (0.56, 0.68)       |


## Musculoskeletal Disease Phenotype Models
| Diagnosis                                          |   Phecode | 5-fold AUROC (95% CI)   |
|:---------------------------------------------------|----------:|:------------------------|
| Senile osteoporosis                                |    743.12 | 0.81 (0.80, 0.82)       |
| Osteoporosis NOS                                   |    743.11 | 0.80 (0.78, 0.83)       |
| Osteoarthrosis NOS                                 |    740.9  | 0.75 (0.74, 0.76)       |
| Osteoarthritis; localized                          |    740.1  | 0.75 (0.75, 0.76)       |
| Symptoms and disorders of the joints               |    741    | 0.72 (0.68, 0.77)       |
| Pain in joint                                      |    745    | 0.68 (0.64, 0.73)       |
| Osteopenia or other disorder of bone and cartilage |    743.9  | 0.66 (0.65, 0.68)       |
| Rheumatoid arthritis                               |    714.1  | 0.66 (0.62, 0.69)       |
| Spondylosis without myelopathy                     |    721.1  | 0.64 (0.60, 0.68)       |
| Arthropathy NOS                                    |    716.9  | 0.60 (0.52, 0.68)       |

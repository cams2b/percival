# Percival
Code repository for Percival: a generalizable vision language foundation model for computed tomography

![Key Graphic](images/percival.png)


## Diagnosis with Percival
```python
projection_dim = 512
king_percival = percival(in_channels=config.in_channels, 
                         projection_dim=projection_dim, 
                         img_size=(128, 256, 256))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
king_percival.to(device)
king_percival.load_image_encoder(path=img_weights)
diagnostic_results = king_percival.diagnostic_inference_all_conditions(img_path=test_img_path, device=device)

```
## Available diagnostic models
| Diagnosis    | Phecode        | 5-fold AUROC (95% CI)  |
| :---:         |     :---:      |          :---: |
| Heart failure   | 428.2     | 0.79 (0.76, 0.82)      |
| HFrEF           | 428.3     | 0.85 (0.81, 0.88)      |
| HFpEF           | 428.4     | 0.81 (0.74, 0.87)      |
| Essential hypertension      | 401.1     | 0.77 (0.75, 0.80)      |
| Hypertensive heart disease      | 402.11     | 0.78 (0.73, 0.83)      |
| Myocardial Infarction     | 411.2     | 0.73 (0.71, 0.74)      |
| Coronary atherosclerosis     | 411.4     | 0.79 (0.77, 0.81)      |
| Chronic ischemic heart disease     | 411.8     | 0.83 (0.79, 0.88)      |
| Chronic pulmonary heart disease     | 415.2     | 0.81 (0.79, 0.83)      |
| Cardiomegaly    | 416     | 0.80 (0.79, 0.81)      |
| Atrial fibrillation    | 427.21     | 0.79 (0.77, 0.82)      |

Atrial fibrillation
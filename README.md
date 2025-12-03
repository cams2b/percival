# Percival

[![medRxiv](https://img.shields.io/badge/medRxiv-10.1101%2F2025.07.03.25330654-0077cc?style=flat)](https://www.medrxiv.org/content/10.1101/2025.07.03.25330654v1) ![PyPI - Python Version](https://img.shields.io/badge/python-3.10-blue)
 [![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/cbeeche/percival) 

Code repository for Percival: a generalizable vision language foundation model for computed tomography

![Key Graphic](images/percival.png)


## 🧪 Environment setup
To create and activate the conda (percival) environment:
```bash
conda env create -f environment.yml
conda activate percival
```
*A GPU is required to use this conda environment.*


## Pretrained Models
The pretrained Percival models were trained on over 400,000 CT volumes paired with radiology reports from more than 50,000 Penn Medicine BioBank (PMBB) participants. These models cover multiple anatomical regions and imaging protocols.

| Model                   | Download Link                                      | Base Architecture            | Reference                                      |
|-------------------------|----------------------------------------------------|------------------------------|------------------------------------------------|
| Percival Image Encoder  | [Download](https://huggingface.co/cbeeche/percival/tree/main/weights) | DeiT Small Patch16-224       | [DeiT Paper](https://arxiv.org/abs/2012.12877) / [timm](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#models) |
| Percival Text Encoder   | [Download](https://huggingface.co/cbeeche/percival/tree/main/weights)  | Clinical Longformer (BERT)   | [Clinical Longformer](https://huggingface.co/yikuan8/Clinical-Longformer) |



## 🔍 Disease Phenotype Classification with Percival
*Performance metrics reported below reflect predictions made using imaging data alone, without additional clinical covariates.*
```python

from train_operations.classification_model import inference_model

print('[INFO] performing classification')
img_weights = '<path to image weights>/image_encoder.pth'
lang_weights = '<path to language weights>/language_encoder.pth'
image_size = (256, 256, 128)
target_spacing = (1.5, 1.5, 3)
projection_dim = 512
in_channels = 1
vision_model_size = 'small'
model = inference_model(vision_model_size=vision_model_size,
						in_channels=in_channels,
						projection_dim=projection_dim,
						image_size=image_size,
						target_spacing=target_spacing,
						img_weights=img_weights,
						lang_weights=lang_weights)

results, summary = model.diagnostic_inference_all_conditions(img_path='<path to image>.nii')

```


## 🔮 Prognostic risk stratification with Percival
*Performance metrics reported below reflect predictions made using imaging data alone, without additional clinical covariates.*
```python

from train_operations.classification_model import inference_model

img_weights = '<path to image weights>/image_encoder.pth'
lang_weights = '<path to language weights>/language_encoder.pth'
image_size = (256, 256, 128)
target_spacing = (1.5, 1.5, 3)
projection_dim = 512
in_channels = 1
vision_model_size = 'small'
model = inference_model(vision_model_size=vision_model_size,
						in_channels=in_channels,
						projection_dim=projection_dim,
						image_size=image_size,
						target_spacing=target_spacing,
						img_weights=img_weights,
						lang_weights=lang_weights)

results, summary = model.prognostic_inference_all_conditions(img_path='<path to image>.nii')

```



## Citation
If you find any of the code useful please cite our article
```
@article {Beeche2025.07.03.25330654,
	author = {Beeche, Cameron and Kim, Joonghyun and Tavolinejad, Hamed and Zhao, Bingxin and Sharma, Rakesh and Duda, Jeffrey and Gee, James and Dako, Farouk and Verma, Anurag and Morse, Colleen and Hou, Bojian and Shen, Li and Sagreiya, Hersh and Davatzikos, Christos and Damrauer, Scott and Ritchie, Marylyn D. and Rader, Daniel and Long, Qi and Chen, Tianlong and Kahn, Charles E. and Chirinos, Julio and Witschey, Walter R. and Penn Medicine Biobank},
	title = {A Pan-Organ Vision-Language Model for Generalizable 3D CT Representations},
	elocation-id = {2025.07.03.25330654},
	year = {2025},
	doi = {10.1101/2025.07.03.25330654},
	publisher = {Cold Spring Harbor Laboratory Press},
	issn = {3067-2007},
	URL = {https://www.medrxiv.org/content/early/2025/07/03/2025.07.03.25330654},
	eprint = {https://www.medrxiv.org/content/early/2025/07/03/2025.07.03.25330654.full.pdf},
	journal = {medRxiv}
}

```
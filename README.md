# Percival

[![medRxiv](https://img.shields.io/badge/medRxiv-10.1101%2F2025.07.03.25330654-0077cc?style=flat)](https://www.medrxiv.org/content/10.1101/2025.07.03.25330654v1) ![PyPI - Python Version](https://img.shields.io/badge/python-3.10-blue)
 [![Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/cbeeche/percival) 

### Code repository for Percival: a generalizable vision language foundation model for computed tomography.

Percival is a large-scale vision–language foundation model for three-dimensional computed tomography (CT), trained on more than 400,000 CT–report pairs from the Penn Medicine BioBank (PMBB). This repository provides pretrained model weights, inference utilities, and reference scripts for probing biological and clinical information encoded in CT-derived representations. The codebase is designed to support research in multimodal representation learning, disease phenotype alignment, and downstream diagnostic and prognostic modeling.

![Key Graphic](images/percival.png)

### This repository provides:

- **Pretrained model weights**
- **Inference utilities** for generating CT embeddings
- **Reference scripts** for probing biological and clinical information encoded in the latent space
- **Example workflows** for downstream diagnostic and prognostic modeling

Percival is designed to support research in multimodal representation learning, disease phenotype alignment, and the characterization of biological signals captured by CT-based foundation models.

## 🧪 Environment setup
To create and activate the `percival` conda environment, first install 
[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).  
Then run:
```bash
conda env create -f environment.yml
conda activate percival
```
*A GPU is required to use this conda environment.*


## Pretrained Models
The pretrained Percival model was trained on over 400,000 CT volumes paired with radiology reports from more than 50,000 PMBB participants, covering multiple anatomical regions and imaging protocols.

| Model                   | Download Link                                      | Base Architecture            | Reference                                      |
|-------------------------|----------------------------------------------------|------------------------------|------------------------------------------------|
| Percival Image Encoder  | [Download](https://huggingface.co/cbeeche/percival/tree/main/weights) | DeiT Small Patch16-224       | [DeiT Paper](https://arxiv.org/abs/2012.12877) / [timm](https://github.com/huggingface/pytorch-image-models?tab=readme-ov-file#models) |
| Percival Text Encoder   | [Download](https://huggingface.co/cbeeche/percival/tree/main/weights)  | Clinical Longformer (BERT)   | [Clinical Longformer](https://huggingface.co/yikuan8/Clinical-Longformer) |



## 🧬 Extraction of principal component and latent features using Percival
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

z_img, pc = model.extract_latent_components(img_path='<path to image>.nii')

```


## 🔍 Disease Phenotype Classification with Percival
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

results, summary = model.diagnostic_inference_all_conditions(img_path='<path to image>.nii')

```


## 🔮 Prognostic risk stratification with Percival
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

## Acknowledgements
We thank the authors of [Merlin](https://github.com/StanfordMIMI/Merlin/tree/main), and [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) for their valuable open-source contributions that significantly influenced this work.


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
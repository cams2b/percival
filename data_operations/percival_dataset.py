import os
import json
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import medrs

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Spacing,
    ScaleIntensityRange, SpatialPad, CenterSpatialCrop, ToTensor,
    RandZoom, RandSpatialCrop, RandFlip, RandRotate90, RandGaussianNoise,
)

# Default max_length per language model
_DEFAULT_MAX_LENGTH = {
    'microsoft/BiomedVLP-CXR-BERT-specialized': 512,
    'yikuan8/Clinical-Longformer': 1024,
    'microsoft/BiomedVLP-BioViL-T': 512,
    # ModernBERT family supports up to 8192; capped to 2048 to bound activation
    # memory while staying well above PMBB report lengths.
    'thomas-sounack/BioClinical-ModernBERT-base': 2048,
    'thomas-sounack/BioClinical-ModernBERT-large': 2048,
}



def crop_or_pad(img, target_size):
    """Crop or pad a 3D image to the target size.
    
    Args:
        img: Input image as a PyTorch tensor of shape (C, D, H, W)
        target_size: Tuple of (D_target, H_target, W_target)
    """
    _, d, h, w = img.shape
    pad_d = max(target_size[0] - d, 0)
    pad_h = max(target_size[1] - h, 0)
    pad_w = max(target_size[2] - w, 0)
    pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2
    pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
    pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
    img = torch.nn.functional.pad(
        img[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2),
        mode='constant', 
        value=0
    ).squeeze(0)
    
    _, d, h, w = img.shape
    start_d = (d - target_size[0]) // 2
    start_h = (h - target_size[1]) // 2
    start_w = (w - target_size[2]) // 2
    img = img[
        :, 
        start_d:start_d + target_size[0],
        start_h:start_h + target_size[1],
        start_w:start_w + target_size[2]
    ]
    assert img.shape[1]==target_size[0] and img.shape[2]==target_size[1] and img.shape[3]==target_size[2], f"Final shape {img.shape} does not match target {target_size}"
    return img

def extract_radiology_report_text(report_path):
    with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
        report = f.read()
    return report




class percival_dataset(Dataset):
    """Dataset supporting JSON annotation for cross-sectional VLM training.

    JSON structure expected:
    [
        {
            "patient_id": "patient_xxxx",
            "studies": [
                {
                    "study_id": "study_aaaa",
                    "scans": [
                        {
                            "scan_id": "scan_aaaa",
                            "image_path": "/path/to/imageA.nii.gz",
                            "report_path": "/path/to/reportA.txt"
                        },
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ]
    """

    def __init__(self,
                 data_path: str,
                 image_size: tuple = (256, 256, 128),
                 target_spacing: tuple = (1.5, 1.5, 3),
                 use_target_spacing: bool = True,
                 augment: bool = False,
                 txt_format: str = 'report',
                 load_method: str = 'medrs',
                 data_format: str = 'nii',
                 tokenizer_name: str = 'microsoft/BiomedVLP-CXR-BERT-specialized',
                 max_length: int = None):

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.image_size = image_size
        self.target_spacing = target_spacing
        self.use_target_spacing = use_target_spacing
        self.txt_format = txt_format
        self.data_format = data_format
        self.load_method = load_method
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length or _DEFAULT_MAX_LENGTH.get(tokenizer_name, 512)
        self._tokenizer = None
        self.index_map = []
        self.image_col = 'image_path'
        for patient_idx, patient in enumerate(self.data):
            for study_idx, study in enumerate(patient['studies']):
                for scan_idx, scan in enumerate(study['scans']):
                    self.index_map.append((patient_idx, study_idx, scan_idx))



        # Compose transforms ONCE (saves significant overhead)

        if load_method == 'monai' and self.data_format == 'nii':
            if self.use_target_spacing:
                load_transforms = [
                    LoadImage(image_only=True),
                    EnsureChannelFirst(),
                    Spacing(pixdim=self.target_spacing, mode="nearest"),
                    ScaleIntensityRange(-1000, 1000, 0.0, 1.0, clip=True),
                ]
            else:
                load_transforms = [
                    LoadImage(image_only=True),
                    EnsureChannelFirst(),
                    ScaleIntensityRange(-1000, 1000, 0.0, 1.0, clip=True),
                ]

            if augment:
                self.transforms = Compose(load_transforms + [
                    RandGaussianNoise(prob=0.5, mean=0.0, std=0.01),
                    RandZoom(min_zoom=0.85, max_zoom=1.15, prob=0.5),
                    RandSpatialCrop(roi_size=self.image_size, random_center=True, random_size=False),
                    SpatialPad(spatial_size=self.image_size),
                    RandFlip(spatial_axis=[0, 1, 2], prob=0.3),
                    RandRotate90(prob=0.3),
                    ToTensor()
                ])
            else:
                self.transforms = Compose(load_transforms + [
                    SpatialPad(spatial_size=self.image_size),
                    CenterSpatialCrop(roi_size=self.image_size),
                    ToTensor(),
                ])
            self.load_image = self._load_image

        elif self.load_method == 'medrs' and self.data_format == 'nii':
            if self.use_target_spacing:
                self.load_image = self._load_image_medrs_target_spacing
            else:
                self.load_image = self._load_image_medrs
        elif self.data_format == 'pth':
            self.image_col = 'pt_path'
            self.load_image = self._load_image_pth

    def __len__(self):
        return len(self.index_map)

    def _load_image_medrs(self, image_path):
        """Load and transform a single image WITHOUT resampling to target spacing."""
        img = medrs.load(image_path)
        img = img.crop_or_pad(self.image_size)
        img = img.to_torch().float()
        img = img.clamp(-1000, 1000)
        img = (img + 1000) / 2000.0
        img = img.unsqueeze(0)
        return img.permute(0, 3, 1, 2).contiguous()

    def _load_image_medrs_target_spacing(self, image_path):
        """Load and transform a single image WITH resampling to target spacing."""
        img = medrs.load(image_path)
        img = img.resample(self.target_spacing, method="nearest")
        img = img.crop_or_pad(self.image_size)
        img = img.to_torch().float()
        img = img.clamp(-1000, 1000)
        img = (img + 1000) / 2000.0
        img = img.unsqueeze(0)
        return img.permute(0, 3, 1, 2).contiguous()

    def _load_image_pth(self, image_path):
        data = torch.load(image_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            img = data['volume']
        else:
            img = data
        img = img.unsqueeze(0)
        img = img.float()
        img = img.clamp(-1000, 1000)
        img = (img + 1000) / 2000.0
        img = crop_or_pad(img, list(reversed(self.image_size)))
        return img.contiguous()

    def _load_image(self, image_path):
        """Load and transform a single image via MONAI transforms."""
        img = self.transforms(image_path)
        img = img.permute(0, 3, 1, 2).contiguous()
        return img

    def _validate_image(self, img, index, img_path):
        """Validate image shape and content."""
        if img.shape != (1, *reversed(self.image_size)):
            raise ValueError(f"Shape mismatch: {img.shape} vs expected {(1, *reversed(self.image_size))}")
        if torch.all(img == 0):
            raise ValueError(f"Blank image detected at index {index} ({img_path})")

    @property
    def tokenizer(self):
        """Lazy tokenizer init — safe for DataLoader worker forks."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, trust_remote_code=True
            )
        return self._tokenizer

    def _load_text(self, scan):
        """Load raw text string from the scan."""
        if scan.get('full_report'):
            return scan['full_report']
        return extract_radiology_report_text(scan['report_path'])

    def _tokenize_text(self, text):
        """Tokenize text and pad to max_length. Returns (input_ids, attention_mask)."""
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

    def __getitem__(self, index):
        patient_idx, study_idx, scan_idx = self.index_map[index]
        patient = self.data[patient_idx]
        study = patient['studies'][study_idx]
        scan = study['scans'][scan_idx]

        img_path = scan[self.image_col]

        try:
            primary_img = self.load_image(img_path)
            self._validate_image(primary_img, index, img_path)

            # Load and tokenize text
            text = self._load_text(scan)
            if text is None or (isinstance(text, str) and text.strip() == ""):
                raise ValueError(f"Blank/empty text at index {index}")

            input_ids, attention_mask = self._tokenize_text(text)

            patient_id = patient.get('patient_id', str(patient_idx))
            study_id = study.get('study_id', str(study_idx))
            study_uid = f"{patient_id}_{study_id}"

        except Exception as e:
            print('[WARNING] failed to load at index {}; error: {}'.format(index, e))
            new_index = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(new_index)

        return {
            'primary_image': primary_img,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'study_uid': study_uid,
        }



class PatientAwareSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        
        # Build patient_id -> list of dataset indices
        self.patient_to_indices = {}
        for idx, (patient_idx, study_idx, scan_idx) in enumerate(dataset.index_map):
            patient_id = dataset.data[patient_idx]['patient_id']
            if patient_id not in self.patient_to_indices:
                self.patient_to_indices[patient_id] = []
            self.patient_to_indices[patient_id].append(idx)
    
    def __iter__(self):
        patient_queues = {}
        for pid, indices in self.patient_to_indices.items():
            q = indices.copy()
            if self.shuffle:
                random.shuffle(q)
            patient_queues[pid] = q
        
        all_indices = []
        
        while patient_queues:
            batch = []
            
            pids = list(patient_queues.keys())
            if self.shuffle:
                random.shuffle(pids)
            
            for pid in pids:
                if len(batch) >= self.batch_size:
                    break
                batch.append(patient_queues[pid].pop(0))
                if not patient_queues[pid]:
                    del patient_queues[pid]
            
            if batch:
                all_indices.extend(batch)
        
        return iter(all_indices)
    
    def __len__(self):
        return self.num_samples
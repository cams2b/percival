"""Language tower for Percival VLM.

Accepts pre-tokenized tensors (input_ids, attention_mask) from the dataset.
Tokenization is handled in the data pipeline, not here.

Supported text encoders (select via ``language_model`` in YAML):
  - ``'microsoft/BiomedVLP-CXR-BERT-specialized'`` — radiology-specific BERT
  - ``'yikuan8/Clinical-Longformer'`` — clinical text encoder
  - ``'microsoft/BiomedVLP-BioViL-T'`` — biomedical vision-language encoder
  - ``'thomas-sounack/BioClinical-ModernBERT-base'`` — 150M, 8192-ctx ModernBERT
  - ``'thomas-sounack/BioClinical-ModernBERT-large'`` — 396M, 8192-ctx ModernBERT
"""

import warnings

import torch
import torch.nn as nn
from transformers import AutoModel


def _select_attn_impl() -> str:
    """Pick the fastest attention backend available at import time.

    Preference order:
      1. ``flash_attention_2`` — requires the ``flash-attn`` package + a
         compatible CUDA build.
      2. ``sdpa``             — shipped with PyTorch >= 2.1; universally
         available in our env.

    We never fall through to ``eager`` here; SDPA is always present on the
    supported PyTorch versions.
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


class LanguageTower(nn.Module):
    """Text encoder for contrastive VLM training.

    Receives pre-tokenized (input_ids, attention_mask) tensors and
    encodes them via a pretrained text model, then projects to the
    VLM shared embedding space.

    Args:
        projection_dim: Output dimension for VLM space.
        language_model: HuggingFace model identifier.
        freeze_language_model: Freeze text encoder weights.
    """

    MODEL_CONFIGS = {
        'microsoft/BiomedVLP-CXR-BERT-specialized': {
            'hidden_size': 768,
            'max_length': 512,
            'pooling': 'cls',
            'is_cxr_bert': True,
        },
        'yikuan8/Clinical-Longformer': {
            'hidden_size': 768,
            'max_length': 768,
            'pooling': 'cls',
            'disable_global': False,
        },
        'microsoft/BiomedVLP-BioViL-T': {
            'hidden_size': 768,
            'max_length': 512,
            'pooling': 'cls',
        },
        'thomas-sounack/BioClinical-ModernBERT-base': {
            'hidden_size': 768,
            # 8192 native; capped to 2048 to bound activation memory.
            # PMBB reports rarely exceed ~1.5k tokens.
            'max_length': 2048,
            'pooling': 'cls',
            'supports_attn_impl': True,
        },
        'thomas-sounack/BioClinical-ModernBERT-large': {
            'hidden_size': 1024,
            'max_length': 2048,
            'pooling': 'cls',
            'supports_attn_impl': True,
        },
    }

    def __init__(self,
                 projection_dim: int = 512,
                 language_model: str = 'microsoft/BiomedVLP-CXR-BERT-specialized',
                 freeze_language_model: bool = False,
                 **kwargs):
        super().__init__()
        self.language_model = language_model
        self.projection_dim = projection_dim

        self.model_config = self.MODEL_CONFIGS.get(language_model, {
            'hidden_size': 768,
            'max_length': 512,
            'pooling': 'cls',
        })

        hidden_size = self.model_config['hidden_size']
        is_cxr_bert = self.model_config.get('is_cxr_bert', False)

        from_pretrained_kwargs = {"trust_remote_code": True}
        if self.model_config.get('supports_attn_impl', False):
            attn_impl = _select_attn_impl()
            try:
                self.text_encoder = AutoModel.from_pretrained(
                    language_model,
                    attn_implementation=attn_impl,
                    **from_pretrained_kwargs,
                )
                print(f"[INFO] LanguageTower: {language_model} (attn={attn_impl})")
            except (TypeError, ValueError, ImportError) as e:
                # Covers three cases:
                #   - `flash-attn` present but incompatible with the runtime CUDA;
                #   - HF raising inside its attn_impl validation;
                #   - an older model class that doesn't accept the kwarg at all.
                warnings.warn(
                    f"Falling back from attn_implementation={attn_impl!r} "
                    f"to transformers default for {language_model}: {e}"
                )
                self.text_encoder = AutoModel.from_pretrained(
                    language_model, **from_pretrained_kwargs,
                )
                print(f"[INFO] LanguageTower: {language_model} (attn=default)")
        else:
            self.text_encoder = AutoModel.from_pretrained(
                language_model, **from_pretrained_kwargs,
            )

        if is_cxr_bert:
            self.text_encoder.cls_projection_head = nn.Identity()
            self.text_encoder.cls = nn.Identity()
            self.linear_layer = nn.Linear(hidden_size, projection_dim, bias=False)
        else:
            self.linear_layer = nn.Linear(hidden_size, projection_dim)

        if freeze_language_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            print(f"[INFO] Frozen text encoder. Only projection layer is trainable.")
        elif self.model_config.get('disable_global', False):
            for n, p in self.text_encoder.named_parameters():
                if ("attention.self.query_global" in n or
                    "attention.self.key_global" in n or
                    "attention.self.value_global" in n or
                    n.startswith("pooler.")):
                    p.requires_grad = False

        print(f'[INFO] LanguageTower: {language_model} -> {projection_dim}-dim')

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (B, seq_len) token IDs.
            attention_mask: (B, seq_len) attention mask (1=real, 0=pad).

        Returns:
            (B, projection_dim) text embeddings.
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooling = self.model_config.get('pooling', 'cls')
        if pooling == 'cls':
            text_embeddings = outputs.last_hidden_state[:, 0, :]
        elif pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).expand(
                outputs.last_hidden_state.size()).float()
            text_embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            text_embeddings = outputs.last_hidden_state[:, 0, :]

        return self.linear_layer(text_embeddings)

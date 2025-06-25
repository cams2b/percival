# coding=utf-8
# Copyright 2025 The Percival Foundation model Authors.
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



import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer




class grail(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        self._tokenizer = None
        self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, projection_dim)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            print("[INFO] Initializing tokenizer safely post-fork")
            self._tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        return self._tokenizer

    def forward(self, text_labels):
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)

        return text_embeddings
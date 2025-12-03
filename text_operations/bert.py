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
    def __init__(self, projection_dim:int=512, language_model:str=None, disable_global:bool=True):
        super().__init__()
        #self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        self._tokenizer = None
        self.language_model = language_model
        if language_model:
            self.text_encoder = AutoModel.from_pretrained(language_model)
        else:
            self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, projection_dim)
        if disable_global:
            ## tragic. this has to be frozen when using accelerate
            for n, p in self.text_encoder.named_parameters():
                if "attention.self.query_global" in n \
                or "attention.self.key_global" in n \
                or "attention.self.value_global" in n \
                or n.startswith("pooler."):
                    p.requires_grad = False

    
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            print("[INFO] Initializing tokenizer safely post-fork")
            self._tokenizer = AutoTokenizer.from_pretrained(self.language_model)
        return self._tokenizer

    def forward(self, text_labels):
        #text_labels = [sanitize_report(text) for text in text_labels]
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

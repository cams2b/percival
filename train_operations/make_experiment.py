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

import os
import json

def make_experiment(output_path, experiment_name):
    experiment_path = os.path.join(output_path, experiment_name)
    weight_path = os.path.join(experiment_path, 'weights')
    out_path = os.path.join(experiment_path, 'output')

    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    return experiment_path, weight_path, out_path


def save_experiment_config(config: dict, output_dir: str, filename: str = "config.json"):
    """
    Save experiment configuration dictionary to a JSON file.

    Automatically converts tuples to lists and other non-serializable
    objects to strings if necessary.

    Args:
        config (dict): Dictionary of experiment parameters.
        output_dir (str): Directory to save the config file.
        filename (str): Name of the output JSON file (default: "config.json").
    """

    def serialize(obj):
        if isinstance(obj, (tuple, set)):
            return list(obj)
        if obj is None:
            return None
        return str(obj)

    config_path = os.path.join(output_dir, filename)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=serialize)

    print(f"[INFO] Configuration saved to {config_path}")
    return config_path

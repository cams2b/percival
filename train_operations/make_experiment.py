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


def make_experiment(output_path, experiment_name):
    experiment_path = os.path.join(output_path, experiment_name)
    weight_path = os.path.join(experiment_path, 'weights')
    output_path = os.path.join(experiment_path, 'output')
    print(experiment_name)
    if os.path.exists(experiment_path):
        print('[INFO] experiment path exists')
    else:
        os.mkdir(experiment_path)

    if os.path.exists(weight_path):
        print('[INFO] weight path exists')
    else:
        os.mkdir(weight_path)

    if os.path.exists(output_path):
        print('[INFO] the output path exists')
    else:
        os.mkdir(output_path)

    print('[INFO] {} directory has been created'.format(experiment_path))

    return experiment_path, weight_path, output_path

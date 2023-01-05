# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import json


def convert_extract(dataset):
    """ Format as:
    - List[{'text': str, 'entities': [Ett], 'relations': [Rel]}]
    - Ett: {id: int, start_offset: int, end_offset: int, label: str}
    - Rel: {from_id: int, to_id: int, type: str}
    """
    return [{
        "id": di, "text": data["text"], "entities": data["entities"], "relations": data["relations"]
    } for di, data in enumerate(dataset)]


def convert_cls(dataset):
    """ Format as:
    - List[{'label': str, 'text': str}]
    """
    results = []
    for outer_id, data in enumerate(dataset):
        results.append({
            "id": outer_id,
            "text": data["text"],
            "label": data["label"]
        })
    return results


def convert(dataset, task_type):
    if task_type == "ext":
        return convert_extract(dataset)
    else:
        return convert_cls(dataset)


def do_convert(args):

    if not os.path.exists(args.input_file):
        raise ValueError("Please input the correct path of label studio file.")

    with open(args.input_file) as infile:
        for content in infile:
            dataset = json.loads(content)
        results = convert(dataset, args.task_type)

    with open(args.doccano_file, "w") as outfile:
        for item in results:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--doccano_file', type=str, default='doccano_ext.jsonl')
    parser.add_argument('--task_type', type=str, choices=['ext', 'cls'], default='ext')
    args = parser.parse_args()
    do_convert(args)

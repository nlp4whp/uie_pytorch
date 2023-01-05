# Copyright (c) 2022 Heiheiyoyo. All Rights Reserved.
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

import json
import random
import re
from functools import partial

import numpy as np
import torch
from colorama import Fore
from torch.utils.data import Dataset
from tqdm import tqdm
from ..set_logger import Logger

logger = Logger("[model-utils]")

BAR_FORMAT = f'{{desc}}: {Fore.GREEN}{{percentage:3.0f}}%{Fore.RESET} {Fore.BLUE}{{bar}}{Fore.RESET}  {Fore.GREEN}{{n_fmt}}/{{total_fmt}} {Fore.RED}{{rate_fmt}}{{postfix}}{Fore.RESET} eta {Fore.CYAN}{{remaining}}{Fore.RESET}'
BAR_FORMAT_NO_TIME = f'{{desc}}: {Fore.GREEN}{{percentage:3.0f}}%{Fore.RESET} {Fore.BLUE}{{bar}}{Fore.RESET}  {Fore.GREEN}{{n_fmt}}/{{total_fmt}}{Fore.RESET}'
BAR_TYPE = [
    "░▝▗▖▘▚▞▛▙█",
    "░▖▘▝▗▚▞█",
    " ▖▘▝▗▚▞█",
    "░▒█",
    " >=",
    " ▏▎▍▌▋▊▉█"
    "░▏▎▍▌▋▊▉█"
]

tqdm = partial(tqdm, bar_format=BAR_FORMAT, ascii=BAR_TYPE[0], leave=False)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.

    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once .
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays, which is greater than a limitation.

    Args:
        probs (List[List[float]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[List[int]]: The index of the last dimension meet the conditions.
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_relation_type_dict(relation_data):

    # def unify_prompt_name(prompt):
    #     # The classification labels are shuffled during finetuning, so they need
    #     # to be unified during evaluation.
    #     if re.search(r'\[.*?\]$', prompt):
    #         prompt_prefix = prompt[:prompt.find("[", 1)]
    #         cls_options = re.search(r'\[.*?\]$', prompt).group()[1:-1].split(",")
    #         cls_options = sorted(list(set(cls_options)))
    #         cls_options = ",".join(cls_options)
    #         prompt = prompt_prefix + "[" + cls_options + "]"
    #         return prompt
    #     return prompt

    def compare(a, b):
        a = a[::-1]
        b = b[::-1]
        res = ''
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                res += a[i]
            else:
                break
        if res == "":
            return res
        elif res[::-1][0] == "的":
            return res[::-1][1:]
        return ""

    relation_type_dict = {}
    added_list = []
    for i in range(len(relation_data)):
        added = False
        if relation_data[i][0] not in added_list:
            for j in range(i + 1, len(relation_data)):
                match = compare(relation_data[i][0], relation_data[j][0])
                if match != "":
                    match = unify_prompt_name(match)
                    if relation_data[i][0] not in added_list:
                        added_list.append(relation_data[i][0])
                        relation_type_dict.setdefault(match, []).append(
                            relation_data[i][1])
                    added_list.append(relation_data[j][0])
                    relation_type_dict.setdefault(match, []).append(
                        relation_data[j][1])
                    added = True
            if not added:
                added_list.append(relation_data[i][0])
                suffix = relation_data[i][0].rsplit("的", 1)[1]
                suffix = unify_prompt_name(suffix)
                relation_type_dict[suffix] = relation_data[i][1]
    return relation_type_dict


class SpanEvaluator:
    """
    SpanEvaluator computes the precision, recall and F1-score for span detection.
    """

    def __init__(self):
        super(SpanEvaluator, self).__init__()
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def compute(self, start_probs, end_probs, gold_start_ids, gold_end_ids):
        """
        Computes the precision, recall and F1-score for span detection.
        """
        pred_start_ids = get_bool_ids_greater_than(start_probs)
        pred_end_ids = get_bool_ids_greater_than(end_probs)
        gold_start_ids = get_bool_ids_greater_than(gold_start_ids.tolist())
        gold_end_ids = get_bool_ids_greater_than(gold_end_ids.tolist())
        num_correct_spans = 0
        num_infer_spans = 0
        num_label_spans = 0
        for predict_start_ids, predict_end_ids, label_start_ids, label_end_ids in zip(
                pred_start_ids, pred_end_ids, gold_start_ids, gold_end_ids):
            [_correct, _infer, _label] = self.eval_span(
                predict_start_ids, predict_end_ids, label_start_ids,
                label_end_ids)
            num_correct_spans += _correct
            num_infer_spans += _infer
            num_label_spans += _label
        return num_correct_spans, num_infer_spans, num_label_spans

    def update(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_spans += num_infer_spans
        self.num_label_spans += num_label_spans
        self.num_correct_spans += num_correct_spans

    def eval_span(self, predict_start_ids, predict_end_ids, label_start_ids,
                  label_end_ids):
        """
        evaluate position extraction (start, end)
        return num_correct, num_infer, num_label
        input: [1, 2, 10] [4, 12] [2, 10] [4, 11]
        output: (1, 2, 2)
        """
        pred_set = get_span(predict_start_ids, predict_end_ids)
        label_set = get_span(label_start_ids, label_end_ids)
        num_correct = len(pred_set & label_set)
        num_infer = len(pred_set)
        num_label = len(label_set)
        return (num_correct, num_infer, num_label)

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.

        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """
        precision = float(
            self.num_correct_spans / self.num_infer_spans) if self.num_infer_spans else 0.
        recall = float(
            self.num_correct_spans / self.num_label_spans) if self.num_label_spans else 0.
        f1_score = float(
            2 * precision * recall / (precision + recall)) if self.num_correct_spans else 0.
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"


class IEDataset(Dataset):
    """
    Dataset for Information Extraction fron jsonl file. The line type is
    {
        content
        result_list
        prompt
    }
    """

    def __init__(self, file_path, tokenizer, max_seq_len) -> None:
        super().__init__()
        self.file_path = file_path
        self.dataset = list(reader(file_path))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return convert_example(self.dataset[index], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)


class IEMapDataset(Dataset):
    """
    Dataset for Information Extraction fron jsonl file. The line type is
    {
        content
        result_list
        prompt
    }
    """

    def __init__(self, data, tokenizer, max_seq_len) -> None:
        super().__init__()
        self.dataset = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return convert_example(self.dataset[index], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)


def reader(data_path, max_seq_len=512):
    """
    read json
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line['content']
            prompt = json_line['prompt']
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if max_seq_len <= len(prompt) + 3:
                raise ValueError(
                    "The value of max_seq_len is too small, please set a larger value"
                )
            max_content_len = max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line['result_list']
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []

                    for result in result_list:
                        if result['start'] + 1 <= max_content_len < result[
                                'end']:
                            max_content_len = result['start']
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]['end'] <= max_content_len:
                            if result_list[0]['end'] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [
                                    result for result in result_list
                                ]
                                break
                        else:
                            break

                    json_line = {
                        'content': cur_content,
                        'result_list': cur_result_list,
                        'prompt': prompt
                    }
                    json_lines.append(json_line)

                    for result in result_list:
                        if result['end'] <= 0:
                            break
                        result['start'] -= max_content_len
                        result['end'] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {
                            'content': res_content,
                            'result_list': result_list,
                            'prompt': prompt
                        }
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


def convert_example(example, tokenizer, max_seq_len):
    """
    example: {
        title
        prompt
        content
        result_list
    }
    """

    def map_offset(ori_offset, offset_mapping):
        """
        map ori offset to token offset
        """
        for index, span in enumerate(offset_mapping):
            if span[0] <= ori_offset < span[1]:
                return index
        return -1

    encoded_inputs = tokenizer(
        text=[example["prompt"]],
        text_pair=[example["content"]],
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
        return_offsets_mapping=True)
    # encoded_inputs = encoded_inputs[0]
    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"][0]]
    bias = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = offset_mapping[index - 1][1] + 1  # Includes [SEP] token
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias
    start_ids = [0 for x in range(max_seq_len)]
    end_ids = [0 for x in range(max_seq_len)]
    for item in example["result_list"]:
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0

    tokenized_output = [
        encoded_inputs["input_ids"][0], encoded_inputs["token_type_ids"][0],
        encoded_inputs["attention_mask"][0],
        start_ids, end_ids
    ]
    tokenized_output = [np.array(x, dtype="int64") for x in tokenized_output]
    tokenized_output = [
        np.pad(x, (0, max_seq_len - x.shape[-1]), 'constant') for x in tokenized_output]
    return tuple(tokenized_output)


def unify_prompt_name(prompt):
    # The classification labels are shuffled during finetuning, so they need
    # to be unified during evaluation.
    if re.search(r'\[.*?\]$', prompt):
        prompt_prefix = prompt[:prompt.find("[", 1)]
        cls_options = re.search(r'\[.*?\]$', prompt).group()[1:-1].split(",")
        cls_options = sorted(list(set(cls_options)))
        cls_options = ",".join(cls_options)
        prompt = prompt_prefix + "[" + cls_options + "]"
        return prompt
    return prompt


def get_id_and_prob(spans, offset_map):
    prompt_length = 0
    for i in range(1, len(offset_map)):
        if offset_map[i] != [0, 0]:
            prompt_length += 1
        else:
            break

    for i in range(1, prompt_length + 1):
        offset_map[i][0] -= (prompt_length + 1)
        offset_map[i][1] -= (prompt_length + 1)

    sentence_id = []
    prob = []
    for start, end in spans:
        prob.append(start[1] * end[1])
        sentence_id.append(
            (offset_map[start[0]][0], offset_map[end[0]][1]))
    return sentence_id, prob


def cut_chinese_sent(para):
    """
    Cut the Chinese sentences more precisely, reference to
    "https://blog.csdn.net/blmoistawinde/article/details/82379256".
    """
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code and code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_dir='checkpoint/early_stopping', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint/early_stopping'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_dir = save_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_pretrained(self.save_dir)
        self.val_loss_min = val_loss

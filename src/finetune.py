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

import shutil
import sys
import time
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .models import UIE
from .evaluate import evaluate
from .models.utils import logger, IEDataset, set_seed, SpanEvaluator, EarlyStopping


def do_train(
    pretrained_dir: str,
    output_dir: str,
    train_data_path: str,
    dev_data_path: str,
    max_seq_len: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    early_stopping: bool = True,
    logging_steps: int = 20,
    valid_steps: int = 500,
    max_model_num: int = 1,
    seed: int = 123456,
    device: str = 'cpu',
    show_bar: bool = False
):

    set_seed(seed)

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_dir)
    model = UIE.from_pretrained(pretrained_dir)
    if device == 'gpu':
        model = model.cuda()
    train_ds = IEDataset(train_data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
    dev_ds = IEDataset(dev_data_path, tokenizer=tokenizer, max_seq_len=max_seq_len)

    train_data_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)
    dev_data_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        lr=learning_rate, params=model.parameters())

    criterion = torch.nn.functional.binary_cross_entropy
    metric = SpanEvaluator()

    if early_stopping:
        early_stopping_save_dir = os.path.join(output_dir, "early_stopping")
        if not os.path.exists(early_stopping_save_dir):
            os.makedirs(early_stopping_save_dir)
        if show_bar:
            def trace_func(*args, **kwargs):
                with logging_redirect_tqdm([logger.logger]):
                    logger.info(*args, **kwargs)
        else:
            trace_func = logger.info
        early_stopping = EarlyStopping(
            patience=7, verbose=True, trace_func=trace_func,
            save_dir=early_stopping_save_dir)

    loss_list = []
    loss_sum = 0
    loss_num = 0
    global_step = 0
    # best_step = 0
    best_f1 = 0
    tic_train = time.time()
    epoch_iterator = range(1, num_epochs + 1)
    if show_bar:
        train_postfix_info = {'loss': 'unknown'}
        epoch_iterator = tqdm(
            epoch_iterator, desc='Training', unit='epoch')
    for epoch in epoch_iterator:
        train_data_iterator = train_data_loader
        if show_bar:
            train_data_iterator = tqdm(train_data_iterator,
                                       desc=f'Training Epoch {epoch}', unit='batch')
            train_data_iterator.set_postfix(train_postfix_info)
        for batch in train_data_iterator:
            if show_bar:
                epoch_iterator.refresh()
            input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
            if device == 'gpu':
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                att_mask = att_mask.cuda()
                start_ids = start_ids.cuda()
                end_ids = end_ids.cuda()
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=att_mask)
            start_prob, end_prob = outputs[0], outputs[1]

            start_ids = start_ids.type(torch.float32)
            end_ids = end_ids.type(torch.float32)
            loss_start = criterion(start_prob, start_ids)
            loss_end = criterion(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(float(loss))
            loss_sum += float(loss)
            loss_num += 1

            if show_bar:
                loss_avg = loss_sum / loss_num
                train_postfix_info.update({
                    'loss': f'{loss_avg:.5f}'
                })
                train_data_iterator.set_postfix(train_postfix_info)

            global_step += 1
            if global_step % logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = loss_sum / loss_num

                if show_bar:
                    with logging_redirect_tqdm([logger.logger]):
                        logger.info(
                            "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                            % (global_step, epoch, loss_avg,
                               logging_steps / time_diff))
                else:
                    logger.info(
                        "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg,
                           logging_steps / time_diff))
                tic_train = time.time()

            if global_step % valid_steps == 0:
                save_dir = os.path.join(output_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                model_to_save = model
                model_to_save.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                if max_model_num:
                    model_to_delete = global_step - max_model_num * valid_steps
                    model_to_delete_path = os.path.join(
                        save_dir, "model_%d" % model_to_delete)
                    if model_to_delete > 0 and os.path.exists(model_to_delete_path):
                        shutil.rmtree(model_to_delete_path)

                dev_loss_avg, precision, recall, f1 = evaluate(
                    model, metric, data_loader=dev_data_loader, device=device, loss_fn=criterion)

                if show_bar:
                    train_postfix_info.update({
                        'F1': f'{f1:.3f}',
                        'dev loss': f'{dev_loss_avg:.5f}'
                    })
                    train_data_iterator.set_postfix(train_postfix_info)
                    with logging_redirect_tqdm([logger.logger]):
                        logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                                    % (precision, recall, f1, dev_loss_avg))
                else:
                    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                                % (precision, recall, f1, dev_loss_avg))
                # Save model which has best F1
                if f1 > best_f1:
                    if show_bar:
                        with logging_redirect_tqdm([logger.logger]):
                            logger.info(
                                f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                            )
                    else:
                        logger.info(
                            f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                        )
                    best_f1 = f1
                    save_dir = os.path.join(save_dir, "model_best")
                    model_to_save = model
                    model_to_save.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                tic_train = time.time()

        if early_stopping:
            dev_loss_avg, precision, recall, f1 = evaluate(
                model, metric, data_loader=dev_data_loader, device=device, loss_fn=criterion)

            if show_bar:
                train_postfix_info.update({
                    'F1': f'{f1:.3f}',
                    'dev loss': f'{dev_loss_avg:.5f}'
                })
                train_data_iterator.set_postfix(train_postfix_info)
                with logging_redirect_tqdm([logger.logger]):
                    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                                % (precision, recall, f1, dev_loss_avg))
            else:
                logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                            % (precision, recall, f1, dev_loss_avg))

            # Early Stopping
            early_stopping(dev_loss_avg, model)
            if early_stopping.early_stop:
                if show_bar:
                    with logging_redirect_tqdm([logger.logger]):
                        logger.info("Early stopping")
                else:
                    logger.info("Early stopping")
                tokenizer.save_pretrained(early_stopping_save_dir)
                sys.exit(0)

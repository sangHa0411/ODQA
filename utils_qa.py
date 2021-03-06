# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pre-processing
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from typing import Optional, Tuple, Any

import numpy as np
from tqdm.auto import tqdm

import torch
import random
from transformers import is_torch_available, PreTrainedTokenizerFast, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from datasets import DatasetDict
from arguments import (
    DataTrainingArguments,
)

from konlpy.tag import Mecab

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def postprocess(pos_data, context, offsets) :
    span_start, span_end = offsets[0], offsets[1]

    post_start = -1
    post_end = -1

    tok_idx = 0
    for pos in pos_data :
        tok_span, tok_type = pos

        while context[tok_idx] != tok_span[0] :
            tok_idx += 1

        tok_size = len(tok_span)    
        tok_start = tok_idx
        tok_end = tok_idx + tok_size

        if context[tok_start:tok_end] != tok_span :
            raise IndexError ('Wrong token Index')

        if tok_start <= span_start and span_start < tok_end :
            post_start = tok_start
    
        if tok_start < span_end and span_end <= tok_end :
            post_end = tok_end
            if tok_type.startswith('J') :
                post_end -= len(tok_span)

        tok_idx += tok_size

    return context[span_start:span_end] if post_start == post_end else \
        context[post_start:post_end]


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):

    multi_flag = False
    if len(predictions) == 3 :
        multi_flag = True
        all_doc_logits, all_start_logits, all_end_logits = predictions
    elif len(predictions) == 2 :
        all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # example??? mapping?????? feature ??????
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i) # per each example, it has multiple features

    # prediction, nbest??? ???????????? OrderedDict ???????????????.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # ?????? example?????? ?????? main Loop
    for example_index, example in enumerate(tqdm(examples)):
        # ???????????? ?????? example index
        feature_indices = features_per_example[example_index]
        prelim_predictions = []

        # ?????? example??? ?????? ?????? feature ???????????????.
        for feature_index in feature_indices:

            if multi_flag == True :
                doc_logits = all_doc_logits[feature_index]
                if doc_logits < 0.5 :
                    continue

            # ??? featureure??? ?????? ?????? prediction??? ???????????????.
            start_logits = all_start_logits[feature_index] # (seq_size,)
            end_logits = all_end_logits[feature_index] # (seq_size,)
            # logit??? original context??? logit??? mapping?????????.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional : `token_is_max_context`, ???????????? ?????? ?????? ???????????? ????????? ??? ?????? max context??? ?????? answer??? ???????????????
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            # `n_best_size`?????? ??? start and end logits??? ???????????????.
            # argsort??? ???????????? ??????, logit ?????? ??? ???????????? index??? ????????????. size = n_best_size
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist() 
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            # check every index, total n_best_size ** 2 cases
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # out-of-scope answers??? ???????????? ????????????.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # ????????? < 0 ?????? > max_answer_length??? answer??? ???????????? ????????????.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # ?????? context??? ?????? answer??? ???????????? ????????????.
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    # start logit, and logit ?????? ????????? ????????? score??? ????????? ??????.
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index], # post-processing component 1, score??? ????????? ??????
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )


        # score ????????? ???????????? ?????? ?????? `n_best_size` predictions??? ???????????????. 
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # offset??? ???????????? original context?????? answer text??? ???????????????.
        # post-processing component 2, context?????? ????????? ???????????? ??????
        mecab = Mecab()
        context = example["context"]
        pos_data = mecab.pos(context)

        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = postprocess(pos_data, context, offsets)
            
        # rare edge case?????? null??? ?????? ????????? ????????? ????????? failure??? ????????? ?????? fake prediction??? ????????????.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # ?????? ????????? ?????????????????? ???????????????(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
        # ????????? example??? ????????? ?????? ???????????? prediction score?????? (start logit + end logit) ???????????? softmax??? ????????????.
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # ???????????? ????????? ???????????????.
        # softmax??? ????????? ?????? probability ?????? ????????????.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        all_predictions[example["id"]] = predictions[0]["text"]

        # np.float??? ?????? float??? casting -> `predictions`??? JSON-serializable ??????
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    # output_dir??? ????????? ?????? dicts??? ???????????????.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"predictions_{prefix}".json,
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"nbest_predictions_{prefix}".json,
        )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
            )

    return all_predictions


def check_no_error(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
) -> Tuple[Any, int]:

    # last checkpoint ??????.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Tokenizer check: ?????? script??? Fast tokenizer??? ??????????????????.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    return last_checkpoint, max_seq_length
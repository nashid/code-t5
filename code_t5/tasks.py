# coding=utf-8
# Copyright 2021 JetBrains.
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

from os.path import join
from typing import Dict

import seqio
import t5.data

from code_t5.constants import VOCAB_PATH
from code_t5.data import preprocessors


def create_output_features(vocab: seqio.SentencePieceVocabulary) -> Dict[str, seqio.Feature]:
    return {
        "inputs": seqio.Feature(vocabulary=vocab, add_eos=True, required=False),
        "targets": seqio.Feature(vocabulary=vocab, add_eos=True),
    }


def register_task(
    name, paths: Dict[str, str], num_input_examples: Dict[str, int], output_features: Dict[str, seqio.Feature]
):
    print(f"Registering {name} task...")
    seqio.TaskRegistry.add(
        name,
        source=seqio.TextLineDataSource(split_to_filepattern=paths, num_input_examples=num_input_examples),
        preprocessors=[
            preprocessors.text_file_per_line,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            t5.data.preprocessors.unsupervised,
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=[t5.evaluation.metrics.accuracy],
        output_features=output_features,
    )


# FullLine dataset for Prefix language modeling pretraining, as in Raffel et al., 2019.
def register_fl_py_top5k(data_dir: str):
    paths = {
        "train": join(data_dir, "fl_py_50stars_top5k_2019", "py5k-50.train.txt-*"),
        "validation": join(data_dir, "fl_py_50stars_top5k_2019", "py5k-50.test.txt"),
    }
    vocab = seqio.SentencePieceVocabulary(join(data_dir, VOCAB_PATH), t5.data.DEFAULT_EXTRA_IDS)
    register_task(
        "fl_py_50stars_top5k_2019", paths, {"train": 170000, "validation": 40815}, create_output_features(vocab)
    )


def register_fl_py_50_stars(data_dir: str):
    paths = {
        "train": join(data_dir, "fl_py_50stars_2019", "py_50stars_2019.train.txt-*"),
        "validation": join(data_dir, "fl_py_50stars_2019", "py_50stars_2019.test.txt-*"),
    }
    vocab = seqio.SentencePieceVocabulary(join(data_dir, VOCAB_PATH), t5.data.DEFAULT_EXTRA_IDS)
    register_task("fl_py_50stars_2019", paths, {"train": 700000, "validation": 352596}, create_output_features(vocab))


def register_fl_py_10_stars(data_dir: str):
    paths = {
        "train": join(data_dir, "fl_py_10stars_2019", "py_10stars_2019.train.txt-*"),
        "validation": join(data_dir, "fl_py_10stars_2019", "py_10stars_2019.test.txt-*"),
    }
    vocab = seqio.SentencePieceVocabulary(join(data_dir, VOCAB_PATH), t5.data.DEFAULT_EXTRA_IDS)
    register_task("fl_py_10stars_2019", paths, {"train": 1100000, "validation": 559698}, create_output_features(vocab))


# BigQuery Github dataset
def register_bq_py_2016_minus_ethpy150(data_dir: str):
    paths = {
        "train": join(data_dir, "bq_py_2016_minus_ethpy150", "github_py_minus_ethpy150.train.*.txt"),
        "validation": join(data_dir, "bq_py_2016_minus_ethpy150", "github_py_minus_ethpy150.validation.*.txt"),
    }
    vocab = seqio.SentencePieceVocabulary(join(data_dir, VOCAB_PATH), t5.data.DEFAULT_EXTRA_IDS)
    register_task(
        "bq_py_2016_minus_ethpy150", paths, {"train": 5884757, "validation": 1292044}, create_output_features(vocab)
    )


# BigQuery Github re-splited and de-duplicated
def register_bq_py_2016_dedup(data_dir: str):
    paths = {
        "train": join(data_dir, "bq_py_2016_dedup", "txt", "gh_py.train.*.txt.gz"),
        "validation": join(data_dir, "bq_py_2016_dedup", "txt", "gh_py.valid.txt.gz"),
    }
    vocab = seqio.SentencePieceVocabulary(join(data_dir, VOCAB_PATH), t5.data.DEFAULT_EXTRA_IDS)
    register_task("bq_py_2016_dedup", paths, {"train": 3800000, "validation": 20448}, create_output_features(vocab))


# Athena Github dataset
def register_at_py_2020(data_dir: str):
    paths = {
        "train": join(data_dir, "at_py_2020", "txt", "at_py.train.*.txt.gz"),
        "validation": join(data_dir, "at_py_2020", "txt", "at_py.valid.txt.gz"),
    }
    vocab = seqio.SentencePieceVocabulary(join(data_dir, VOCAB_PATH), t5.data.DEFAULT_EXTRA_IDS)
    register_task("at_py_2020", paths, {"train": 3788418, "validation": 10373}, create_output_features(vocab))


# Test task
def register_test_task(data_dir: str):
    paths = {
        "train": join(data_dir, "dataset-dev", "train.txt-*"),
        "validation": join(data_dir, "dataset-dev", "test.txt"),
    }
    vocab = seqio.SentencePieceVocabulary(join(data_dir, "dataset-test", "test.model"), t5.data.DEFAULT_EXTRA_IDS)
    register_task("test", paths, {"train": 8000, "validation": 412}, create_output_features(vocab))


def register_all_known_tasks(data_dir: str = "gs://t5-codex/data", with_test: bool = False):
    register_fl_py_top5k(data_dir)
    register_fl_py_50_stars(data_dir)
    register_fl_py_10_stars(data_dir)
    register_bq_py_2016_minus_ethpy150(data_dir)
    register_bq_py_2016_dedup(data_dir)
    register_at_py_2020(data_dir)
    if with_test:
        register_test_task(data_dir)

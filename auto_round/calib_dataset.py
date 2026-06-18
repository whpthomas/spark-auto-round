# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import multiprocessing
import os
import random
import sys

logging.getLogger("datasets").setLevel(logging.WARNING)

import torch
from datasets import Dataset, Features, IterableDataset, Sequence, Value, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from . import envs
from .utils import is_local_path, logger

CALIB_DATASETS = {}


def register_dataset(name):
    """Class decorator to register a DATASET subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        name: A string. Define the dataset type.

    Returns:
        cls: The class of register.
    """

    def register(dataset):
        if isinstance(name, list):
            names = name
        else:
            names = [name]
        for global_name in names:
            CALIB_DATASETS[global_name] = dataset
        return dataset

    return register


def apply_chat_template_to_samples(samples, tokenizer, seqlen, system_prompt=None):
    rendered_messages = []
    # if system_prompt is None: ## remove system prompt as models like deepseek don't recommend using it
    #     system_prompt = "You are a helpful assistant."
    for text in samples:
        message = []
        if system_prompt is not None and system_prompt != "":
            message.append({"role": "system", "content": system_prompt})

        if isinstance(text, list) and isinstance(text[0], dict):
            message += text
        else:
            message.append({"role": "user", "content": text})
        try:
            chat_templated = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            logger.warning("Failed to apply chat template. removing the system role in chat history.")
            message_modified = [msg for msg in message if msg["role"] != "system"]
            chat_templated = tokenizer.apply_chat_template(
                message_modified,
                tokenize=False,
                add_generation_prompt=True,
            )

        rendered_messages.append(chat_templated)
    example = tokenizer(rendered_messages, truncation=True, max_length=seqlen)
    return example


def _make_map_fingerprint(dataset, tokenizer, seqlen, apply_chat_template, system_prompt, text_key="text"):
    """Compute a stable fingerprint for Dataset.map() calls.

    datasets uses dill to serialize the transform function for cache fingerprinting.
    HuggingFace tokenizer objects are not reliably serializable by dill, causing
    a random hash to be used each run — which breaks caching entirely.

    This function computes a deterministic fingerprint from stable string
    identifiers (tokenizer name, seqlen, etc.) so that caching works correctly
    and subsequent runs can load from disk instead of re-tokenizing in RAM.
    """
    import hashlib

    parts = [
        getattr(dataset, "_fingerprint", "no_fingerprint"),
        getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
        str(seqlen),
        str(apply_chat_template),
        str(system_prompt),
        text_key,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def get_tokenizer_function(tokenizer, seqlen, apply_chat_template=False, system_prompt=None):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length of
    seqlen to the "text" field of examples.
    """

    def default_tokenizer_function(examples):
        if not apply_chat_template:
            example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
        else:
            example = apply_chat_template_to_samples(examples["text"], tokenizer, seqlen, system_prompt)
        return example

    return default_tokenizer_function


@register_dataset(["NeelNanda/pile-10k", "pile-10k"])
def get_pile_dataset(
    tokenizer,
    seqlen,
    dataset_name="NeelNanda/pile-10k",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    split = "train"

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )
    try:
        calib_dataset = load_dataset("NeelNanda/pile-10k", split=split)
    except Exception as e:
        import ssl

        error_message = str(e)
        # Check for proxy or SSL error
        if "proxy" in error_message.lower() or isinstance(e, ssl.SSLError) or "SSL" in error_message.upper():
            logger.error(
                f"Network error detected, please check proxy settings. "
                f"Error: {error_message}. Or consider using a backup dataset by `pip install modelscope` "
                f"and set '--dataset swift/pile-val-backup' in AutoRound API."
            )
        else:
            logger.error(f"Failed to load the dataset: {error_message}")
        raise RuntimeError(f"Failed to load the dataset: {error_message}")
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )

    return calib_dataset


@register_dataset(["swift/pile-val-backup", "pile-val-backup"])
def get_pile_val_dataset(
    tokenizer,
    seqlen,
    dataset_name="swift/pile-val-backup",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test", "validation").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """

    split = "validation"

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )
    from transformers.utils.versions import require_version

    require_version(
        "modelscope",
        "Loading 'swift/pile-val-backup' dataset requires modelscope to be installed, " "`pip install modelscope`",
    )
    from modelscope import MsDataset  # pylint: disable=E0401

    calib_dataset = MsDataset.load(
        "swift/pile-val-backup", "default", split=split
    ).to_iterable_dataset()  # , use_streaming=True
    calib_dataset = calib_dataset.shuffle(seed=seed).take(10000)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset(["BAAI/CCI3-HQ", "CCI3-HQ"])
def get_cci3_hq_dataset(
    tokenizer, seqlen, dataset_name="BAAI/CCI3-HQ", split=None, seed=42, apply_chat_template=False, system_prompt=None
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    calib_dataset = load_dataset("BAAI/CCI3-HQ", split="train", streaming=True)
    calib_dataset = calib_dataset.shuffle(seed=seed).take(10000)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset(["codeparrot/github-code-clean", "github-code-clean"])
def get_github_code_clean_dataset(
    tokenizer,
    seqlen,
    dataset_name="codeparrot/github-code-clean",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """

    def get_default_tokenizer_function():
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        apply_chat_template: Whether to apply chat template in tokenization.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
         of seqlen to the "code" field of examples.
        """

        def default_tokenizer_function(examples):
            if not apply_chat_template:
                example = tokenizer(examples["code"], truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(
                    examples["code"], tokenizer, seqlen, system_prompt=system_prompt
                )
            return example

        return default_tokenizer_function

    tokenizer_function = get_default_tokenizer_function()

    # Calculate shard count: load 10x oversample for filtering, ~100K samples per shard
    samples_per_shard = 100000
    target_samples = 10000
    num_shards = min(20, math.ceil(target_samples / samples_per_shard))

    # Use the full repository name for HuggingFace Hub API
    full_dataset_name = "codeparrot/github-code-clean"

    # List all Parquet files in the repository
    api = HfApi()
    files = api.list_repo_files(full_dataset_name, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]

    # Sample random shards for diversity
    random.seed(seed)
    selected_shards = random.sample(parquet_files, min(num_shards, len(parquet_files)))

    logger.debug(f"Loading {len(selected_shards)} shards from {full_dataset_name}")

    # Download and load Parquet files
    dfs = []
    for shard in selected_shards:
        path = hf_hub_download(full_dataset_name, shard, repo_type="dataset")
        df = pd.read_parquet(path)
        dfs.append(df)

    # Combine into a single dataset
    combined_df = pd.concat(dfs, ignore_index=True)
    calib_dataset = Dataset.from_pandas(combined_df)

    # Shuffle and take samples
    calib_dataset = calib_dataset.shuffle(seed=seed).take(target_samples)

    # Tokenize with caching support
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "code"
        ),
    )

    return calib_dataset


@register_dataset(["HuggingFaceH4/ultrachat_200k", "ultrachat_200k"])
def get_ultrachat_dataset(
    tokenizer,
    seqlen,
    dataset_name="HuggingFaceH4/ultrachat_200k",
    split=None,
    seed=42,
    apply_chat_template=True,
    system_prompt=None,
):
    if split is None:
        split = "train_sft"
    all_splits = ["train_sft", "test_sft", "train_gen", "test_gen"]
    if split not in all_splits:
        raise ValueError("split must be one of {} for ultrachat_200k ".format(all_splits))

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True, trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed).take(20000)

    def is_instruct_tokenizer(tokenizer):
        try:
            out = tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}])
            return bool(out and len(out) > 0)
        except Exception:
            return False

    is_instruct = is_instruct_tokenizer(tokenizer)

    if is_instruct and not apply_chat_template:
        logger.info("Tokenizer looks like an instruct/chat model, but apply_chat_template=False. Setting to True.")
        apply_chat_template = True
    elif not is_instruct and apply_chat_template:
        logger.info("Tokenizer is not an instruct/chat model, but apply_chat_template=True. Setting to False.")
    apply_chat_template = False

    def tokenize_example_batch(examples):
        if not apply_chat_template:
            texts = []
            for message_list in examples["messages"]:
                combined = "".join([msg["content"] for msg in message_list])
                texts.append(combined)
            return tokenizer(texts, truncation=True, max_length=seqlen)
        else:
            return apply_chat_template_to_samples(examples["messages"], tokenizer, seqlen, system_prompt=system_prompt)

    dataset = dataset.map(tokenize_example_batch, batched=True)
    return dataset


@register_dataset(["openbmb/Ultra-FineWeb", "openbmb/Ultra-FineWeb"])
def get_ultrafinweb_dataset(
    tokenizer,
    seqlen,
    dataset_name="openbmb/Ultra-FineWeb",
    split=None,
    seed=42,
    apply_chat_template=True,
    system_prompt=None,
):
    if split is not None:
        if split not in ["en", "zh"]:
            raise ValueError("split must be one of ['en', 'zh'] for Ultra-FineWeb dataset")
        calib_dataset = load_dataset("openbmb/Ultra-FineWeb", split=split, streaming=True, trust_remote_code=True)
    else:
        calib_dataset = load_dataset("openbmb/Ultra-FineWeb", split="en", streaming=True, trust_remote_code=True)
        # dataset_ch = load_dataset("openbmb/Ultra-FineWeb", split='zh',
        #                           streaming=True, trust_remote_code=True).shuffle(seed=seed).take(2000)

        # calib_dataset = concatenate_datasets([dataset_en, dataset_ch]) ##concat dasetset could not shuffle

    calib_dataset = calib_dataset.shuffle(seed=seed).take(20000)

    def get_default_tokenizer_function():
        def default_tokenizer_function(examples):
            if not apply_chat_template:
                example = tokenizer(examples["content"], truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(
                    examples["content"], tokenizer, seqlen, system_prompt=system_prompt
                )
            return example

        return default_tokenizer_function

    tokenizer_function = get_default_tokenizer_function()

    dataset = calib_dataset.map(tokenizer_function, batched=True)
    return dataset


@register_dataset(["madao33/new-title-chinese", "new-title-chinese"])
def get_new_chinese_title_dataset(
    tokenizer,
    seqlen,
    dataset_name="madao33/new-title-chinese",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """
    Returns a tokenized dataset for the specified parameters.

    Args:
        tokenizer: The tokenizer to use.
        seqlen: Maximum sequence length.
        dataset_name: Name of the dataset to load.
        split: Which split of the dataset to use.
        seed: Random seed for shuffling.
        apply_template: Whether to apply a template to the data.

    Returns:
        A tokenized and shuffled dataset.
    """

    def get_tokenizer_function():
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        apply_chat_template: Whether to apply chat template in tokenization.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
        of seqlen to the "text" field of examples.
        """

        def default_tokenizer_function(examples):
            if not apply_chat_template:
                example = tokenizer(examples["content"], truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(
                    examples["content"], tokenizer, seqlen, system_prompt=system_prompt
                )
            return example

        return default_tokenizer_function

    split = "train"
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function()

    calib_dataset = load_dataset("madao33/new-title-chinese", split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "content"
        ),
    )

    return calib_dataset


@register_dataset("mbpp")
@register_dataset("google-research-datasets/mbpp")
def get_mbpp_dataset(
    tokenizer,
    seqlen,
    dataset_name="google-research-datasets/mbpp",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    dataset_name = "google-research-datasets/mbpp"

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    samples = []
    splits = split
    if splits is None:
        splits = ["train", "validation", "test"]
    if isinstance(splits, str):
        splits = splits.split("+")

    for split in splits:
        dataset = load_dataset(dataset_name, split=split)
        for data in dataset:
            samples.append({"text": data["text"] + data["code"]})
    random.Random(seed).shuffle(samples)
    import datasets

    calib_dataset = datasets.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )

    return calib_dataset


@register_dataset(["audiocaps", "AudioCaps"])
def get_audiocaps_dataset(
    tokenizer,
    seqlen,
    dataset_name="audiocaps",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Returns a calibration dataset from AudioCaps (audio caption text).

    AudioCaps provides human-written captions describing audio events.  The captions
    are rich in descriptive language about sounds, music, and speech — making them
    well-suited as calibration text for audio-related model quantization.

    The dataset CSV is downloaded from the official GitHub repository on first use.

    Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        dataset_name: Dataset identifier (ignored, always loads AudioCaps).
        split: Unused (AudioCaps train split is always used).
        seed: The random seed for shuffling.
        apply_chat_template: Whether to apply chat template in tokenization.
        system_prompt: Optional system prompt for chat template.

    Returns:
        A HuggingFace Dataset with tokenized audio captions.
    """
    import csv

    from auto_round.utils import download_audiocaps_csv

    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    cache_file = download_audiocaps_csv()

    samples = []
    with open(cache_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            caption = row.get("caption", "").strip()
            if caption:
                samples.append({"text": caption})

    if not samples:
        raise RuntimeError("AudioCaps dataset is empty or could not be parsed.")

    random.Random(seed).shuffle(samples)
    import datasets as ds

    calib_dataset = ds.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )
    return calib_dataset


@register_dataset(["nvidia/OpenCodeInstruct", "opencode-instruct"])
def get_opencode_instruct_dataset(
    tokenizer,
    seqlen,
    dataset_name="nvidia/OpenCodeInstruct",
    split=None,
    seed=42,
    apply_chat_template=False,
    system_prompt=None,
):
    """Returns a tokenized dataset for nvidia/OpenCodeInstruct.

    Loads the NVIDIA OpenCodeInstruct dataset (5M coding instruction/response pairs),
    concatenates input + output fields, and tokenizes for calibration.

    Uses the same shard-download pattern as github-code-clean: downloads a few
    parquet shards via hf_hub_download (which caches locally), reads them with
    pandas, and creates a HuggingFace Dataset. This avoids the streaming+
    shuffle problem where .shuffle() forces random access across all shards.

    Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        dataset_name: The name of the dataset (default: "nvidia/OpenCodeInstruct").
        split: The data split to be used (default: "train").
        seed: The random seed for shuffling the dataset.
        apply_chat_template: Whether to apply chat template in tokenization.
        system_prompt: Optional system prompt for chat template.

    Returns:
        A tokenized HuggingFace Dataset for calibration.
    """
    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    full_dataset_name = "nvidia/OpenCodeInstruct"

    # List all Parquet files in the repository
    api = HfApi()
    files = api.list_repo_files(full_dataset_name, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet") and f.startswith("data/")]

    # Sample random shards for diversity (4 shards ≈ 400K samples)
    # After packing, ~25% become full-length sequences, so 2100 samples → ~540 packed sequences.
    # We oversample slightly so that after the filter in _get_dataset_impl drops
    # mostly-padding sequences, we still have >= nsamples (512) valid sequences.
    target_samples = 2100
    num_shards = min(4, len(parquet_files))
    random.seed(seed)
    selected_shards = random.sample(parquet_files, min(num_shards, len(parquet_files)))

    logger.debug(f"Loading {len(selected_shards)} shards from {full_dataset_name}")

    # Download and load Parquet files, releasing memory as we go
    dfs = []
    for shard in selected_shards:
        path = hf_hub_download(full_dataset_name, shard, repo_type="dataset")
        df = pd.read_parquet(path)
        dfs.append(df)

    # Combine into a single dataset and release DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    del dfs  # Release individual DataFrames
    calib_dataset = Dataset.from_pandas(combined_df)
    del combined_df  # Release pandas DataFrame

    # Shuffle and take samples
    calib_dataset = calib_dataset.shuffle(seed=seed).select(range(min(target_samples, len(calib_dataset))))

    # Concatenate input + output into a single text field
    def concat_function(examples):
        examples["text"] = [inp + "\n" + out for inp, out in zip(examples["input"], examples["output"])]
        return examples

    calib_dataset = calib_dataset.map(concat_function, batched=True)

    # Tokenize with caching support
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )

    # Pack short sequences: concatenate multiple samples to fill seqlen
    # This avoids the filter_func dropping samples shorter than seqlen
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    packed_input_ids = []
    packed_attention_mask = []
    current_ids = []

    def _finalize_sequence(seq):
        """Pad sequence to exactly seqlen and add to packed lists."""
        if not seq:
            return
        # Truncate if over seqlen, pad if under
        seq = seq[:seqlen]
        # Use all-ones mask: packed sequences are calibration data, not real
        # padded sequences. Padding 0s would disable SDPA's is_causal=True
        # fast path and cause shape-mismatch crashes (e.g. Qwen3.5).
        attention_mask = [1] * seqlen
        seq = seq + [pad_token_id] * (seqlen - len(seq))
        packed_input_ids.append(seq)
        packed_attention_mask.append(attention_mask)

    for example in calib_dataset:
        ids = example["input_ids"]
        if isinstance(ids, list):
            ids = torch.tensor(ids)

        # Skip empty sequences
        if len(ids) == 0:
            continue

        ids_list = ids.tolist()

        # Single sample fills or exceeds seqlen — finalize immediately
        if len(ids_list) >= seqlen:
            if current_ids:
                _finalize_sequence(current_ids)
                current_ids = []
            _finalize_sequence(ids_list)
            continue

        # Try to append to current packed sequence
        # Account for EOS separator if current_ids is non-empty
        extra = 1 if current_ids else 0  # EOS token between samples
        if len(current_ids) + extra + len(ids_list) <= seqlen:
            # Fits entirely — append with EOS separator
            if current_ids:
                current_ids.append(eos_token_id)
            current_ids.extend(ids_list)
        else:
            # Doesn't fit — finalize current, start new
            if current_ids:
                _finalize_sequence(current_ids)
            current_ids = list(ids_list)

    # Finalize any remaining packed sequence
    _finalize_sequence(current_ids)

    # Rebuild dataset from packed sequences
    # Convert to list first to release the Arrow table from the source dataset,
    # then create a new Dataset from the list. This avoids keeping two Arrow tables in memory.
    data = [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(packed_input_ids, packed_attention_mask)]
    del packed_input_ids, packed_attention_mask  # Release source data
    calib_dataset = Dataset.from_list(data)

    return calib_dataset


@register_dataset("local")
def get_local_dataset(
    tokenizer, seqlen, dataset_name="./tmp.json", split=None, seed=42, apply_chat_template=False, system_prompt=None
):
    """Returns a dataloader for a custom dataset and split.
    We allow the input of a json or text file containing a processed text sample each line.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name or path of the dataset, which is a json or jsonl file.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for a custom dataset and split, using the provided tokenizer and sequence length.
    """
    tokenizer_function = get_tokenizer_function(
        tokenizer, seqlen, apply_chat_template=apply_chat_template, system_prompt=system_prompt
    )

    def load_local_data(data_path):
        if data_path.endswith(".json"):
            with open(data_path, "r") as f:
                data = json.load(f)
            return data
        elif data_path.endswith(".jsonl"):
            data = []
            with open(data_path) as f:
                for line in f:
                    sample = json.loads(line)
                    data.append(sample)
            return data
        else:
            logger.error("invalid local file type, for now only support json/jsonl format data file.")

    samples = []
    dataset = load_local_data(dataset_name)
    if isinstance(dataset, dict):
        new_dataset = []
        for key in dataset.keys():
            new_dataset.append(dataset[key])
        dataset = new_dataset
    for data in dataset:
        text = data
        if isinstance(text, str):
            pass
        elif isinstance(data, dict) and len(data.keys()) == 1:
            for item in data.items():
                text = item[1]
        elif isinstance(data, dict) and "text" in data.keys():
            text = data["text"]
        elif isinstance(data, dict) and "input_ids" in data.keys():
            text = data["input_ids"]
        if not isinstance(text, str):
            raise TypeError("data must be a string")
        text = text.rstrip()
        text = text.rstrip("\n")
        samples.append({"text": text})
    random.Random(seed).shuffle(samples)
    import datasets

    calib_dataset = datasets.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(
        tokenizer_function,
        batched=True,
        new_fingerprint=_make_map_fingerprint(
            calib_dataset, tokenizer, seqlen, apply_chat_template, system_prompt, "text"
        ),
    )
    return calib_dataset


def get_dataset_len(dataset):
    """Calculates the length of a dataset.

    Args:
        dataset: The dataset object, which can be any iterable or collection.

    Returns:
        int: The length of the dataset.

    Raises:
        If the dataset does not support `len()`, iterates through it to count the number of elements.
    """
    try:
        dataset_len = len(dataset)
        return dataset_len
    except Exception:
        cnt = 0
        for _ in dataset:
            cnt += 1
        return cnt


def select(dataset, indices):
    """Selects specific elements from a dataset based on given indices.

    Args:
        dataset: The dataset object to iterate over.
        indices: An iterable of integers specifying the indices to select.

    Yields:
        Elements of the dataset corresponding to the specified indices.

    Notes:
        Stops iterating once the highest index in `indices` has been processed
        to optimize performance.
    """
    indices = set(indices)
    for idx, sample in enumerate(dataset):
        if idx in indices:
            yield sample
        if idx > max(indices):
            break


def select_dataset(dataset, indices):
    """Selects elements from a dataset using its native `select` method, if available.

    Args:
        dataset: The dataset object, which may have a `select` method.
        indices: An iterable of integers specifying the indices to select.

    Returns:
        A subset of the dataset, either using the dataset's `select` method or the
        `select` function defined above as a fallback.
    """
    try:
        return dataset.select(indices)
    except Exception:
        list_data = list(select(dataset, indices))
        import pandas as pd

        df = pd.DataFrame(list_data)
        dataset = Dataset.from_pandas(df)
        return dataset


def _get_dataset_impl(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", seed=42, nsamples=512):
    """Internal implementation: generate a dataset for calibration.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        seqlen (int): The exact sequence length. samples < seqlen will be dropped,
                      samples longer than seqlen will be truncated
        dataset_name (str, optional): The name of the dataset or datasets separated by commas.
                                     Defaults to "NeelNanda/pile-10k".
        split (str, optional): The data split to use. Defaults to None.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        nsamples (int, optional): The total number of samples to include. Defaults to 512.
        apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
        Dataset: The processed dataset ready for calibration.
    """
    dataset_names = dataset_name.split(",")

    def filter_func(example):
        if isinstance(example["input_ids"], list):
            example["input_ids"] = torch.tensor(example["input_ids"])
        if example["input_ids"].shape[-1] < seqlen:
            return False
        input_ids = example["input_ids"][:seqlen]
        input_ids_list = input_ids.tolist()
        if len(input_ids_list) > 1 and seqlen > 2 and input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
            return False
        return True

    def concat_dataset_element(dataset):
        input_ids, concat_input_ids = [eg["input_ids"] for eg in dataset], []
        attention_mask_list, attention_mask = [], torch.ones([1, seqlen]).to(torch.int64)
        buffer_input_id = torch.Tensor().to(torch.int64)
        bos_token_id, eos_token_id = tokenizer.bos_token_id, tokenizer.eos_token_id
        os_cnt, have_bos, have_eos = 0, False, False

        for input_id in input_ids:
            if input_id[0] == bos_token_id:
                input_id = input_id[1:]
                os_cnt, have_bos = os_cnt + 1, True
            if input_id[-1] == eos_token_id:
                input_id = input_id[:-1]
                os_cnt, have_eos = os_cnt + 1, True

            if buffer_input_id.shape[-1] + input_id.shape[-1] + os_cnt > seqlen:
                idx_keep = seqlen - buffer_input_id.shape[-1] - os_cnt
                input_id_to_append = [buffer_input_id, input_id[:idx_keep]]
                if have_bos:
                    input_id_to_append = [torch.tensor([bos_token_id])] + input_id_to_append
                if have_eos:
                    input_id_to_append.append(torch.tensor([eos_token_id]))

                concat_input_ids.append(torch.cat(input_id_to_append).to(torch.int64))
                attention_mask_list.append(attention_mask)
                buffer_input_id = input_id[idx_keep:]
            else:
                buffer_input_id = torch.cat([buffer_input_id, input_id])

            if buffer_input_id.shape[-1] + os_cnt == seqlen:
                input_id_to_append = [buffer_input_id]
                if have_bos:
                    input_id_to_append = [torch.tensor([bos_token_id])] + input_id_to_append
                if have_eos:
                    input_id_to_append.append(torch.tensor([eos_token_id]))
                concat_input_ids.append(torch.cat(input_id_to_append).to(torch.int64))
                attention_mask_list.append(attention_mask)
                buffer_input_id = torch.Tensor().to(torch.int64)
        data = [{"input_ids": a, "attention_mask": b} for a, b in zip(concat_input_ids, attention_mask_list)]
        import datasets

        dataset_new = datasets.Dataset.from_list(data)
        return dataset_new

    datasets, data_lens = [], {}
    system_prompt = "You are a helpful assistant."
    for name in dataset_names:
        split = None
        do_concat = False
        apply_chat_template = False

        if ":" in name:
            name, split_list = name.split(":")[0], name.split(":")[1:]
            for ele in split_list:
                key, values = ele.split("=")[0], ele.split("=")[1:]
                if key == "split":
                    split = values[0].split("+")
                if key == "num":
                    data_lens[name] = int(values[0])
                if key == "concat":
                    do_concat = False if (len(values) > 0 and values[0].lower() == "false") else True
                if key == "apply_chat_template":
                    apply_chat_template = False if (len(values) > 0 and values[0].lower() == "false") else True
                if key == "system_prompt":
                    system_prompt = values[0]
                    apply_chat_template = True
        if is_local_path(name):
            get_dataset = CALIB_DATASETS.get("local")
        else:
            calib_name = name
            if name not in CALIB_DATASETS.keys():
                calib_name = name.split("/")[-1]
                for key in CALIB_DATASETS.keys():
                    if calib_name in key:
                        calib_name = key
                        break
            get_dataset = CALIB_DATASETS.get(calib_name)
        if get_dataset is None:
            filtered_keys = [k for k in CALIB_DATASETS.keys() if "/" not in k]
            raise ValueError(
                f"Dataset '{name}' is not found. Please choose from the supported datasets: {filtered_keys}."
            )
        dataset = get_dataset(
            tokenizer,
            seqlen,
            seed=seed,
            split=split,
            dataset_name=name,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
        )
        if do_concat:
            dataset = concat_dataset_element(dataset)

        dataset = dataset.filter(filter_func)
        if name in data_lens:
            dataset = select_dataset(dataset, range(data_lens[name]))
        if isinstance(dataset, IterableDataset):
            dataset = Dataset.from_list(list(dataset))
        new_features = {}
        for k, v in dataset.features.items():
            if k == "input_ids":
                new_features[k] = Sequence(Value("int64"))
            elif k == "attention_mask":
                new_features[k] = Sequence(Value("bool"))
            else:
                new_features[k] = v

        dataset = dataset.cast(Features(new_features))
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        datasets.append(dataset)

    if len(datasets) == 1:
        dataset_final = datasets[0]
    else:
        indices = range(len(datasets))
        lens = []
        for i in range(len(datasets)):
            cnt = get_dataset_len(datasets[i])
            lens.append(cnt)
        res = sorted(zip(indices, lens), key=lambda x: x[1])

        # res = sorted(zip(indices, datasets), key=lambda x: len(x[1]))
        indices = [item[0] for item in res]
        datasets = [datasets[item[0]] for item in res]
        dataset_names = [dataset_names[index] for index in indices]
        cnt = 0 if not data_lens else sum(data_lens.values())
        dataset_cnt_info = {}
        if cnt > nsamples:
            cnt = 0

        for i in range(len(datasets)):
            name = dataset_names[i].split(":")[0]
            if name not in data_lens:
                target_cnt = (
                    (nsamples - cnt) // (len(datasets) - len(data_lens))
                    if data_lens
                    else (nsamples - cnt) // (len(datasets) - i)
                )
                target_cnt = min(target_cnt, lens[i])
                cnt += target_cnt
            else:
                target_cnt = data_lens[name]
            datasets[i] = select_dataset(datasets[i], range(target_cnt))
            dataset_cnt_info[name] = target_cnt
        if len(datasets) > 1:
            from datasets import concatenate_datasets

            dataset_final = concatenate_datasets(datasets)
            dataset_final = dataset_final.shuffle(seed=seed)
            logger.info(dataset_cnt_info)
        else:
            dataset_final = datasets[0]

    if len(dataset_final) > nsamples:
        dataset_final = select_dataset(dataset_final, range(nsamples))
    return dataset_final


def get_dataset(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", seed=42, nsamples=512):
    """Generate a dataset for calibration.

    Uses a subprocess for preprocessing to ensure all temporary memory is fully
    reclaimed by the OS when the subprocess exits.  The HuggingFace ``datasets``
    library automatically caches intermediate results (e.g. ``.map()``,
    ``.filter()``), so the main process can reload them cheaply after the
    subprocess finishes.

    Set environment variable ``AR_DISABLE_DATASET_SUBPROCESS=1`` to disable
    subprocess mode and run preprocessing in the main process.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        seqlen (int): The exact sequence length.
        dataset_name (str, optional): Dataset name(s) separated by commas.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        nsamples (int, optional): Total number of samples to include. Defaults to 512.

    Returns:
        Dataset: The processed dataset ready for calibration.
    """
    # Allow disabling subprocess mode via environment variable
    if envs.AR_DISABLE_DATASET_SUBPROCESS:
        return _get_dataset_impl(tokenizer, seqlen, dataset_name, seed, nsamples)

    # Run preprocessing in a subprocess so all temporary memory is freed on exit.
    # The HuggingFace datasets cache is warmed up as a side effect.
    logger.info("Preprocessing calibration dataset in a subprocess to avoid memory leaks...")

    try:
        if os.name == "nt":
            raise OSError("fork is not available on Windows")

        ctx = multiprocessing.get_context("fork")
        p = ctx.Process(
            target=_get_dataset_impl,
            args=(tokenizer, seqlen, dataset_name, seed, nsamples),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(f"Dataset preprocessing subprocess exited with code {p.exitcode}")

    except Exception as e:
        logger.warning(f"Subprocess dataset preprocessing failed ({e}), falling back to in-process mode.")

    # (Re-)load the dataset in the main process.  When the subprocess
    # succeeded the HF datasets cache makes this almost instant.
    return _get_dataset_impl(tokenizer, seqlen, dataset_name, seed, nsamples)


def get_dataloader(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", seed=42, bs=8, nsamples=512):
    """Generate a DataLoader for calibration using specified parameters.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        seqlen (int): The exact sequence length. samples < seqlen will be dropped,
                      samples longer than seqlen will be truncated
        dataset_name (str, optional): The name of the dataset or datasets separated by commas.
                                     Defaults to "NeelNanda/pile-10k".
        split (str, optional): The data split to use. Defaults to None.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        bs (int, optional): The batch size. Defaults to 4.
        nsamples (int, optional): The total number of samples to include. Defaults to 512.
        apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
        DataLoader: The DataLoader for the calibrated dataset.
    """

    @torch.no_grad()
    def collate_batch(batch):
        input_ids_new = []
        attention_mask_new = []
        for text in batch:
            input_ids, attention_mask = text["input_ids"], text["attention_mask"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            if isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask)
            input_ids = input_ids[:seqlen]
            input_ids_list = input_ids.tolist()
            if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
                continue
            attention_mask = attention_mask[:seqlen]
            attention_mask_new.append(attention_mask)
            input_ids_new.append(input_ids)
        if len(input_ids_new) == 0:
            return None
        input_ids_new = torch.vstack(input_ids_new)
        attention_mask_new = torch.vstack(attention_mask_new)
        res = {"input_ids": input_ids_new, "attention_mask": attention_mask_new}
        return res

    dataset_final = get_dataset(tokenizer, seqlen, dataset_name, seed, nsamples)
    calib_dataloader = DataLoader(dataset_final, batch_size=bs, shuffle=False, collate_fn=collate_batch)
    return calib_dataloader

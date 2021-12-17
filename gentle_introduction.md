# Gentle introduction to code

This document is intended to help newcomers with understanding pipeline of preprocessing data and training model.

## Useful links

What to read at first:
1. Understanding of [`tf.data`](https://cs230.stanford.edu/blog/datapipeline/)
2. [Seqio](https://github.com/google/seqio) library
3. [t5](https://github.com/google-research/text-to-text-transfer-transformer) implementation with mtf support
4. [gin](https://github.com/google/gin-config) library


## Dev dataset

It is always good to have at hand a small piece of data to run and debug all locally.
We will create it from existing FLCC dataset `Top 5k repos >50 stars`

Let `$DATA_DIR` be your directory with data.

1. Download `Top 5k repos >50 stars` dataset
```shell
wget https://5k-dataset.s3.amazonaws.com/v3/dataset-normalized-5000-with-imports.tar.gz -P $DATA_DIR
tar -xzf "$DATA_DIR/dataset-normalized-5000-with-imports.tar.gz" -C $DATA_DIR
```

2. Select a couple of repositories for dev set, e.g. all repositories started with `A`
```shell
mkdir "$DATA_DIR/dataset-dev"
mkdir -p "$DATA_DIR/dataset-dev/v3/languages/Python/.py/"
cp -R \
  "$DATA_DIR/dataset-normalized-5000-with-imports/v3/languages/Python/.py/A"* \
  "$DATA_DIR/dataset-dev/v3/languages/Python/.py"
mkdir -p "$DATA_DIR/dataset-dev/v3/repositories"
cp -R \
  "$DATA_DIR/dataset-normalized-5000-with-imports/v3/repositories/A"* \
  "$DATA_DIR/dataset-dev/v3/repositories"
```

## Preprocessing

The same as for any FLCC dataset, refer to [`README`](./README.md#preprocessing) to see preprocessing instructions.

#### Prepare files
1. Convert from FLCC to jsonl
```shell
python -m preprocessing.convert-fulline-to-jsonl \
  --data_dir "$DATA_DIR/dataset-dev" \
  --output_dir "$DATA_DIR/dataset-dev/jsonl"
# Output:
# 35 results
# {'files_processed': 10266,
#  'files_skipped: empty': 1824,
#  'files_skipped: generated': 4,
#  'files_skipped: lines long': 209,
#  'repos_processed': 35}
```
2. Merge into single file
```shell
pv "$DATA_DIR/dataset-dev/jsonl/"*.jsonl > "$DATA_DIR/dataset-dev/jsonl/all_repos.jsonl"
```
3. Remove duplicated files by sha
```shell
pv "$DATA_DIR/dataset-dev/jsonl/all_repos.jsonl" \
  | go run preprocessing/filter_dup_sha.go \
  > "$DATA_DIR/dataset-dev/jsonl/all_repos_unique.jsonl"
```

#### Split into holdouts
1. Extract all code sources from JSON into text line format
```shell
pv "$DATA_DIR/dataset-dev/jsonl/all_repos_unique.jsonl" \
  | jq -cr '.content' \
  > "$DATA_DIR/dataset-dev/dev-uniq.txt"
```
2. Split full data
```shell
wc -l "$DATA_DIR/dataset-dev/dev-uniq.txt"
# Output:
# 8412
head -n 8000 "$DATA_DIR/dataset-dev/dev-uniq.txt" > "$DATA_DIR/dataset-dev/dev-uniq.train.txt"
tail -n 412  "$DATA_DIR/dataset-dev/dev-uniq.txt" >  "$DATA_DIR/dataset-dev/dev-uniq.test.txt"
```
3. Create shards with train data
```shell
split -da 4 \
  -l $((`wc -l < "$DATA_DIR/dataset-dev/dev-uniq.txt"`/5)) \
  "$DATA_DIR/dataset-dev/dev-uniq.txt" \
  "$DATA_DIR/dataset-dev/dev-uniq.txt-" \
  --additional-suffix="-of-0005"
```

#### Train SentencePiece vocabulary
```shell
cd $DATA_DIR && \
    time spm_train \
        --allow_whitespace_only_pieces \
        --remove_extra_whitespaces=false \
        --pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 \
        --num_threads=96 \
        --vocab_size=32000 \
        --model_prefix=dataset-dev/dev \
        --input=dataset-dev/dev-uniq.train.txt
```

## Example notebook

To see how data preprocessors and model training work see notebook [`code_t5_dev.ipynb`](./notebooks/code_t5_dev.ipynb).
# Train T5 language model

  * [Datasets](#datasets)
      * [Pre-processing &amp; filtering](#pre-processing-filtering)
      * [FL-Dataset](#fl-dataset)
         * [Python: Top 5k repos &gt;50 stars](#python-top-5k-repos-50-stars)
         * [Python: all repos &gt;50 stars](#python-all-repos-50-stars)
         * [Python: all repos from 10 to 50 stars](#python-all-repos-from-10-to-50-stars)
         * [Java](#java)
         * [Preprocessing](#preprocessing)
      * [cuBERT github_python_minus_ethpy150open_dedup](#cubert-github_python_minus_ethpy150open_dedup)
      * [top400k Github DB 2020](#top400k-github-db-2020)
   * [Train the model](#train-the-model)
   * [Cache the dataset](#cache-the-dataset)
   * [Evaluate](#evaluate)
      * [HumanEval](#humaneval)
   * [Export the model](#export-the-model)
   * [Serve the predictions](#serve-the-predictions)
      * [Optimizing the model](#optimizing-the-model)
      * [TF Serving](#tf-serving)

## Datasets

We are using [google/seqio](https://github.com/google/seqio) library as way to feed the data into a model for training,
that is based on [tf.data](https://cs230.stanford.edu/blog/datapipeline/) pipeline.

Below are the steps one needs to take to convert existing raw dataset into the intermediate JSONL format and then to a plain-text line-based format supported by the library.
All that has already been done for all the Python code in FL-Dataset and thus cached `.tfrecords` can be used directly for the training.

### Pre-processing & filtering

When producing intermediate JSONL format we filter out files with:
 * non-unique sha1 hash of content
 * size > 1Mb
 * max line > 1000, avg line len > 100
 * auto-generated by [gRPC, protobuf, flatbuf, apache thrift, SWIG](https://jetbrains.team/p/code-t5/repositories/code-t5/files/af46e48d107b614073fa5ee9045817c68730e1ca/convert-fulline-to-jsonl.py?tab=source&line=40)

When converting program text we:
 * replace newlines '\n' == `\u0000a` with its codepoint+100, `Ċ`.

### FL-Dataset

Several flavors of the 2019 [fl-dataset](https://jetbrains.team/p/ccrm/repositories/fl-dataset/files/docs/README.md).

#### Python: Top 5k repos >50 stars

[Raw repositories](https://5k-dataset.s3.amazonaws.com/v3/dataset-normalized-5000-with-imports.tar.gz)

 * Total 482Mb compressed, 3Gb uncompressed
 * Python 2.2Gb
 * Projects 3047
 * Files 270,058
 * Lines 46,431,253
 * (sub-)Tokens 309,363,978 + 82,908,197

After filtering: 210k uniq files / 1.6Gb

Pre-processed
 * [SPE vocabulary](gs://t5-codex/data/py-50stars-top5k-2019/py5k-50.model)
 * [JSONL](gs://t5-codex/data/py-50stars-top5k-2019/py5k-50.tar.xz)
 * [cache in .tfrecord](gs://t5-codex/cache/py_50stars_top5k_2019)

#### Python: all repos >50 stars

Raw repositories: [1](https://5k-dataset.s3.amazonaws.com/v3/dataset-open-50-more-1.tar.gz), [2](https://5k-dataset.s3.amazonaws.com/v3/dataset-open-50-more-2.tar.gz)

 * Total 143Gb compressed, 840Gb uncompressed
 * Python 17Gb (.py)
 * Projects 37,847
 * Files 1,745,450
 * Lines 347,563,305
 * (sub-)Tokens 1,403,667,659 + 716,723,018

 After filtering: 1,052,596 uniq files / 7.3Gb

 <details>

 38506 results
 'files_processed': 1466495,
 'files_skipped: >1mb files': 278,
 'files_skipped: empty': 71561,
 'files_skipped: generated': 5557,
 'files_skipped: lines long': 7091,
 'files_skipped: unicode': 4271,
 'repos_processed': 37847,
 'repos_skipped_unk_branchs': 659
 </details>

Pre-processed
 * [JSONL](gs://t5-codex/data/py-50stars-2019/py_50stars_2019-uniq.jsonl.xz)
 * [cache in .tfrecord](gs://t5-codex/cache/py_50stars_2019)

#### Python: all repos from 10 to 50 stars

Raw repositories: [1](https://5k-dataset.s3.amazonaws.com/v3/dataset-open-50-less-1.tar.gz), [2](https://5k-dataset.s3.amazonaws.com/v3/dataset-open-50-less-2.tar.gz)

 * Total 200Gb compressed, 1.1Tb uncompressed
 * Python 29Gb (.py)
 * Projects 87,488
 * Files 2,630,187
 * Lines
 * (sub-)Tokens 2,282,295,686 + 1,110,669,615

 After filtering: 1,659,698 uniq files / 12Gb

Pre-processed
 * [TXT](gs://t5-codex/data/py-10stars-2019/py_10stars_2019.txt.xz)
 * [cache in .tfrecord](gs://t5-codex/cache/py_50stars_2019)


#### Java
TBD

#### Preprocessing

These steps are the same for all the Full-line datasets.
Use `pip install tqdm sentencepiece` to install all necessary for preprocessing packages.
Building `SentencePiece` from sources would increase speed performance,
refer to [original documentation](https://github.com/google/sentencepiece#build-and-install-sentencepiece-command-line-tools-from-c-source).

Below, preprocessing example for the [Python: Top 5k repos >50 stars](#python-top-5k-repos-50-stars) dataset.
1. Convert dataset to JSONL format, combine into a single one file, and remove duplicates.
```shell
python -m preprocessing.convert-fulline-to-jsonl \
  --data_dir data/dataset-normalized-5000-with-imports
  --output_dir data/py5k-50/jsonl

pv data/py5k-50/jsonl/*.jsonl > data/py5k-50/jsonl/all_repos.jsonl

# remove duplicated files by sha
pv data/py5k-50/jsonl/all_repos.jsonl \
  | go run preprocessing/filter_dup_sha.go \
  > data/py5k-50/jsonl/all_repos_unique.jsonl
```

2. Train SentencePiece vocabulary on full content.

```shell
pv data/py5k-50/jsonl/all_repos_unique.jsonl \
  | jq -cr '.content' \
  > data/py5k-50/py5k-50-uniq.txt

# there is no need to do it, but in case you want to recover newlines do
head data/py5k-50/py5k-50-uniq.txt | sed 's/Ċ/\
/g'

# takes ~10min for Top 5k repos >50 stars dataset on 96-core machine with 31Gb of RAM.
cd $OUTPUT_DIR && \
    time spm_train \
        --allow_whitespace_only_pieces \
        --remove_extra_whitespaces=false \
        --pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 \
        --num_threads=96 \
        --vocab_size=32000 \
        --model_prefix=data/py5k-50 \
        --input=py5k-50-uniq.txt
```

3. Generate training and testing splits.

Split, preferably, should be done by-project, to avoid leaking information between splits.
By default, files are ordered by project, so we are “leaking“ at most one project.

Numbers presented in this script from Top 5k repos >50 stars dataset.
```shell
wc -l data/py5k-50/py5k-50-uniq.txt
 210815
head -n 170000  data/py5k-50/py5k-50-uniq.txt > data/python_top_5k/py5k-50-uniq.train.txt
tail -n 40815  data/py5k-50/py5k-50-uniq.txt >  data/python_top_5k/py5k-50-uniq.test.txt
```

Split by-file may be generated the same way, just shuffle lines of JSONL before.
```shell
pv data/python_top_5k/jsonl/all_repo_unique.jsonl \
  | perl -MList::Util=shuffle -e 'print shuffle <>;'
```

Create 5 shards for train holdout (use `gsplit` on macOS).
```shell
split -da 4 \
  -l $((`wc -l < data/python_top_5k/py5k-50-uniq.train.txt`/5)) \
  data/python_top_5k/py5k-50-uniq.train.txt \
  data/python_top_5k/py5k-50-uniq.train.txt- \
  --additional-suffix="-of-0005"
```

### CuBERT github_python_minus_ethpy150open_dedup

From 2016 Github dump on [BigQuery public dataset](https://github.com/google-research/google-research/tree/master/cubert#collection-query)

 * Python: 32Gb(compressed 7Gb) / 22Gb
 * Files: total 7,176,801 / uniq 3,820,448
 * Functions: 29,083,262, docstring 5,525,609
 * (sub-)Tokens:  7,979,952,884 + 1,754,768,276

 1. Create a BigQuery table pointing to GCS `gs://cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_manifest/manifest.jsontxt-*`
 2. Run the query (processing 2.3 Tb)
    ```sql
     SELECT content.id as sha, content.content, content.size, meta.filepath, meta. license, meta.repository
     FROM
     (SELECT * FROM `data-analytics-experiments.github_python.cubert_python`) as meta
     INNER JOIN `bigquery-public-data.github_repos.contents` as content
       ON meta.id = content.id
     WHERE content.binary = FALSE
    ```
 3. Save results to another BigQuery table, export to GCS bucket

To preprocess, do
```shell
gsutil -m cp -r gs://t5-codex/data/bq_py_2016_minus_ethpy150/jsonl/ bq_py_2016
cd bq_py_2016

# list all SHAs
ls jsonl/py_file_content.jsonl-* | parallel "zcat {} | go run preprocess_bq_py_2016.go -h bq_py_2016"
cat gh_py_minus_ethpy150*.txt | sort | uniq > bq_py_2016_sha.txt

# de-duplicate
ls jsonl/* | xargs zcat > full.jsonl
pv full.jsonl | go run preprocessing/filter_dup_sha.go > uniq.json

# split train/test (by file)
head -n 3800000 uniq.jsonl > train.jsonl
tail -n 20448 uniq.jsonl > test.jsonl
# TODO(bzz) split by-project

# shard train
split -da 4 -l $((`wc -l < train.jsonl`/40)) train.jsonl train.jsonl- --additional-suffix="-of-0040"

mkdir txt

# Train: pre-process, filtering out all SHAs from at_py_2020
ls train.jsonl-* | parallel "cat {} | go run preprocess_bq_py_2016.go -d ../at_py_2020/at_py_2020_sha.txt txt"
rename 's/gh_py_minus_ethpy150\./gh_py\.train\./' txt/gh_py_minus_ethpy150.*.txt

# Valid
cat test.jsonl | go run preprocessing/preprocess_bq_py_2016.go -d ../at_py_2020_sha.txt txt
mv gh_py_minus_ethpy150.*.txt txt/gh_py.valid.txt

ls txt/*.txt | parallel gzip -k
gsutil -m cp -r txt/*.gz 'gs://t5-codex/data/bq_py_2016_dedup/txt/'

ls jsonl/*.jsonl* | parallel gzip -k
gsutil -m cp -r jsonl/*.gz 'gs://t5-codex/data/bq_py_2016_dedup/jsonl/'


#TODO add file renaming, according to splits
```

This does not include sha-based deduplication.
Applying it, according to BigQuery, will result in 25.5Gb instead of the default 40Gb.

<details>

<summany>SQL query</summary>

```sql
SELECT
    sum(size)
FROM
(select distinct(sha), size
 from `data-analytics-experiments.github_python.cubert_python_content`
 WHERE filepath like '%.py')
```

</details>


### top400k GH DB 2020

 * Python ? / 20GB? (6.4Gb compressed)
 * Files: total ? / uniq 3,798,791
 * Functions: ?, ? docstring (was 6,195,401)

 
To pre-process

```sh
gsutil -m cp -r 'gs://t5-codex/data/at_py_2020/jsonl/' at_py_2020
cd at_py_2020/

# list all SHAs
ls 20211130_*.gz | parallel "gzcat {} | go run preprocess_at_py_2020.go -h"
cat athena_py.*.txt > ../at_py_2020_sha.txt

# save content in line-based text format
ls jsonl/20211202_* | parallel "gzcat {} | go run preprocess_at_py_2020.go txt"

# train/validation split
tail -n 10373 txt/athena_py.48357538.txt > txt/valid.txt
head -n 300000 txt/athena_py.48357538.txt > txt/athena_py.483575381.txt
rm txt/athena_py.48357538.txt

rename 's/athena_py\./at_py\.train\./' txt/athena_py.*.txt
mv txt/valid.txt txt/athena_py.valid.txt

ls txt/*.txt | parallel gzip -k
gsutil -m cp -r txt/*.gz 'gs://t5-codex/data/at_py_2020/txt/'
```

<details>

At  `s3://codex-dataset` of LargeScaleML AWS account.

<summany>SQL query</summary>

```sql
WITH small_python AS (
	SELECT *
	FROM contents
	WHERE language = 'Python' AND size <= 1048576
),
small_python_with_stats AS (
	SELECT length(content) as cont_len,
		cardinality(regexp_split(content, '\n')) as n_rows,
		array_max(
			TRANSFORM(regexp_split(content, '\n'), x->length(x))
		) as max_row_len,
		content,
		content_sha
	FROM small_python
),
filtered_small_python as (
	SELECT content, content_sha
	FROM small_python_with_stats
	WHERE max_row_len < 1000
		AND cont_len / n_rows < 100
		AND content NOT LIKE '%Generated by the gRPC%'
    	AND content NOT LIKE '%Generated by the protocol buffer%'
    	AND content NOT LIKE '%Autogenerated by Thrift%'
    	AND content NOT LIKE '%was automatically generated by SWIG%'
    	AND content NOT LIKE '%giant blob of binary data%'
),
files_with_content as (
    SELECT 
        repo_id,
        files.commit_sha as commit_sha,
        path,
        size,
        files.content_sha as content_sha,
        content
    FROM filtered_small_python
    JOIN files
    ON filtered_small_python.content_sha = files.content_sha
),
result as (
    SELECT 
        ARBITRARY(owner) as owner,
        ARBITRARY(name) as name,
        ARBITRARY(files_with_content.commit_sha) as commit_sha,
        ARBITRARY(json_extract(meta, '$.license')) as license,
        ARBITRARY(path) as path,
        ARBITRARY(size) as size,
        content_sha,
        ARBITRARY(content) as content
    FROM files_with_content
    JOIN repos
    ON files_with_content.repo_id = repos.repo_id AND files_with_content.commit_sha = repos.commit_sha
    GROUP BY content_sha
)
```

</details>

## Train the model

To train the model one may create a separate from preprocessing virtual environment.

```shell
virtualenv -p python3 .venv-train
source .venv-train/bin/activate
pip install -r requirements.txt
```

To test LM Task definition and input pipeline
```shell
python -m scripts.print_dataset
```

Set up an environment configuration with proper values:
```shell
export PROJECT=<project-id>
export ZONE=<zone>
export TPU_NAME=t5-tpu
export TPU_SIZE=v2-8
export BUCKET=gs://t5-codex
export DATA_DIR="${BUCKET}/data"
export MODEL_DIR="${BUCKET}/models"
export EXPORT_DIR="${MODEL_DIR}/export"
export TASK_NAME='fl_py_50stars_top5k_2019'
```

Create TPU from Cloud VM
```shell
ctpu up --name=$TPU_NAME --project=$PROJECT --zone=$ZONE --tpu-size=$TPU_SIZE --tpu-only --noconf
```

Start model training:
```shell
python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --module_import="code_t5" \
  --gin_location_prefix="gin_configs/" \
  --gin_file="models/shared-prefix_lm.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="run.keep_checkpoint_max = 8" \
  --gin_param="MIXTURE_NAME = '${TASK_NAME}'" \
  --gin_param="run.train_steps = 10000" \
  --gin_param="mesh_train_dataset_fn.use_cached = True" \
  --additional_task_cache_dirs='${BUCKET}/cache'
```

<details>

Un-cached .txt \w FunctionDataSource:
 * TPU v2-8, bi_v1_prefix_lm, base, model_parallelism = 2
   global_step/sec: 0.89
   examples/sec: 110

 * TPU v2-8, bi_v1_prefix_lm, base, model_parallelism = 1
   global_step/sec: 0.97
   examples/sec: 124
   FLOPS Utilization
    * Utilization of TPU Matrix Units: 61.8%
    * Program's Optimal FLOPS: 56.6%
   Memory Bandwidth Utilization: 40%

Cached .tfrecords \w TextLineDataSource:
 * TPU v2-8, bi_v1_prefix_lm, base, model_parallelism = 2
   global_step/sec: 0.87
   examples/sec: 111
   FLOPS Utilization
     * Utilization of TPU Matrix Units: 56.2%
     * Program's Optimal FLOPS: 51.5%
   Memory Bandwidth Utilization: 43.5%

 * TPU v2-8, bi_v1_prefix_lm, base, model_parallelism = 1
   global_step/sec: 0.97
   examples/sec: 124
   FLOPS Utilization
     * Utilization of TPU Matrix Units: 61.9%
     * Program's Optimal FLOPS: 56.7%
   Memory Bandwidth Utilization: 40.6%

 * TPU v2-8, bi_v1_prefix_lm, base, model_parallelism = 1, tokens_per_microbatch_per_replica = None
   global_step/sec: 1.01
   examples/sec: 129
   FLOPS Utilization
     * Utilization of TPU Matrix Units: 66.0%
     * Program's Optimal FLOPS: 59.8%

 * TPU v2-8, bi_v1_prefix_lm, large, model_parallelism = 2
    global_step/sec: 0.30
    examples/sec: 39

</details>

List all trained models and download one of them:
```shell
python -m scripts.ls_models
./scripts/cp_model $MODEL_NAME $CHECKPOINT
```

## Cache the dataset

To cache the dataset on GCS as `.tfrecords`:
```shell
gcloud auth application-default login
pip install apache-beam[gcp] python-snappy
python -m seqio.scripts.cache_tasks_main \
 --module_import="code_t5.tasks" \
 --tasks="${TASK_NAME}" \
 --output_cache_dir="${BUCKET}/cache" \
 --alsologtostderr \
 --pipeline_options=["--runner=DirectRunner","--direct_num_workers 10"]
```

## Inference

List all trained models:
```shell
python -m scripts.ls_models
```

To try different sampling temperatures on examples from
[`mock-data.py`](code_t5/test/resources/mock_data.py)
using Mesh-Tensorflow API
```sh
./scripts/mtf_model_inference.sh arch-t5.1.1.small-prefix_lm-1k
```

For inference using a wrapper around HuggingFace PyTorch that uses `tf.data`, run:
``sh
./scripts/hf_model_inference.py arch-t5.1.1.small-prefix_lm-1k-dedup
``

For pure HuggingFace PyTorch example (\w `transformers.T5Tokenizer`, etc), use:
```sh
TBD
```


## Evaluate

Evaluate model on latest checkpoint \w full decoding
(27 Min initial padding on un-cached dataset :/)
```shell
python -m t5.models.mesh_transformer_main  \
  --tpu="$TPU_ADDRESS" \
  --model_dir="$MODEL_DIR" \
  --t5_tfds_data_dir="$DATA_DIR" \
  --module_import="code_t5" \
  --gin_location_prefix="gin_configs/" \
  --gin_file="models/shared-prefix_lm.gin" \
  --gin_file="eval.gin" \
  --gin_file="sample_decode.gin" \
  --gin_param="Bitransformer.decode.temperature=0.5" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '$TPU_TOPOLOGY'" \
  --gin_param="split = 'validation'" \
  --gin_param="eval_checkpoint_step = -1" \
  --gin_param="MIXTURE_NAME = '${TASK_NAME}'" \
  --gin_param="mesh_train_dataset_fn.use_cached = True" \
  --additional_task_cache_dirs='${BUCKET}/cache'
```

Fast eval only calculating perplexity

```shell
python -m t5.models.mesh_transformer_main  \
  --tpu="$TPU_ADDRESS" \
  --model_dir="$MODEL_DIR" \
  --t5_tfds_data_dir="$DATA_DIR" \
  --module_import="code_t5" \
  --gin_location_prefix="gin_configs/" \
  --gin_file="models/shared-prefix_lm.gin" \
  --gin_file="perplexity_eval.gin" \
  --gin_file="sample_decode.gin" \
  --gin_param="Bitransformer.decode.temperature=0.5" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '$TPU_TOPOLOGY'" \
  --gin_param="split = 'validation'" \
  --gin_param="eval_checkpoint_step = -1" \
  --gin_param="MIXTURE_NAME = '${TASK_NAME}'" \
  --additional_task_cache_dirs='$BASE_DIR/cache' \
  --gin_param="mesh_eval_dataset_fn.use_cached = True"
```

### HumanEval

Download and install HumanEval:
```shell
pip3 install -e "git+http://github.com/openai/human-eval#egg=human-eval"
wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
```

How to run HumanEval locally:
```shell
time python -m scripts.human-eval -a $ARCH_NAME

# linux
sed -i 's/^#\([ tab]*exec(check_program, exec_globals)\)/\1/' $HUMAN_EVAL_INSTALLATION_PATH/human_eval/execution.py
# macOS - apply 'fork' patch
#patch <path-to-installed-packages>/human_eval/execution.py < human_eval_python_3.8_macos_execution.patch

evaluate_functional_correctness humanEval-arch-t5.1.1-prefix_lm-1k-<checkpoint>.jsonl --problem_file='HumanEval.jsonl.gz'
```

Evaluation results are very sensitive to several hyperparameters,
like sampling temperature and `top_k` (tokens to consider at ever step)
so provide them as CLI args of the `human-eval.py`.

## Export the model

```shell
python -m t5.models.mesh_transformer_main \
  --gcp_project="$PROJECT" \
  --tpu_zone="$ZONE" \
  --model_dir="$MODEL_DIR" \
  --module_import="code_t5" \
  --use_model_api \
  --temperature=0.5 \
  --keep_top_k=-1 \
  --mode="export_predict" \
  --export_dir="$EXPORT_DIR"
```

## Serve the predictions

### Optimizing the model

[TODO: document](https://medium.com/google-cloud/optimizing-tensorflow-models-for-serving-959080e9ddbf)

### TF Serving

```shell
export MODEL_NAME="<model>"
export SAVED_MODEL_PATH="/path/to/export"

sudo systemctl start docker

gsutil cp "${BUCKET}/models/large/export/<number>" $SAVED_MODEL_PATH

# Download the TensorFlow Serving Docker image and repo:
docker pull tensorflow/serving:nightly

# First, run a serving image as a daemon:
docker run -d --name serving_base tensorflow/serving:nightly

# Next, copy the `SavedModel` to the container's model folder:
docker cp $SAVED_MODEL_PATH serving_base:/models/$MODEL_NAME

# Now, commit the container that's serving the model:
docker commit --change "ENV MODEL_NAME $MODEL_NAME" serving_base $MODEL_NAME

# Finally, save the image to a tar file:
docker save $MODEL_NAME -o $MODEL_NAME.tar

# stop `serving_base`:
docker kill serving_base


sudo docker run -d --rm -p 8501:8501 \
    --name "$MODEL_NAME-server" \
    $MODEL_NAME --rest_api_timeout_in_ms=120000
```

Inference for a large model on CPU takes ~1min

```shell
time curl -d '{"inputs": ["import tensorflow as"]}' \
  -X POST "http://localhost:8501/v1/models/$MODEL_NAME:predict"
```
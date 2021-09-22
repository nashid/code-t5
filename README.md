# Train T5 language model

  * [Datasets](#datasets)
    * [Pre-processing &amp; filtering](#pre-processing--filtering)
    * [FL-Dataset](#fl-dataset)
        * [Python: Top 5k repos &gt;50 stars](#python-top-5k-repos-50-stars)
        * [Python: all repos &gt;50 stars](#python-all-repos-50-stars)
        * [Python: all repos from 10 to 50 stars](#python-all-repos-from-10-to-50-stars)
        * [Java](#java)
        * [Preprocessing](#preprocessing)
  * [Train the model](#train-the-model)
  * [Cache the dataset](#cache-the-dataset)
  * [Evaluate](#evaluate)
  * [Export the model](#export-the-model)
  * [Serve the predictions](#serve-the-predictions)
  * [TF Serving](#tf-serving)

## Datasets

We are using [google/seqio](https://github.com/google/seqio) library as way to feed the data into a model for training.

Below are the steps one needs to take to conver existing raw datasets into the intermediate JSONL format and then to a plain-text line-based format supported by the libraty. All that has already beed done for all the Python code in FL-Dataset and thus cached .tfrecords can be used directly for the training.

### Pre-processing & filtering

When producing intermediate JSONL format we filter out files with:
 * size > 1Mb
 * max line > 1000, avg line len > 100
 * auto-generated by [gRPC, protobuf, flatbuf, apache thrift, SWIG](https://github.com/bzz/code-t5/blob/master/convert-fulline-to-jsonl.py#L41-L43)

When converting program text we:
 * replace newlines '\n' == `\u0000a` with its codepoint+100, `Ċ`.

### FL-Dataset

Sevral flavors of the 2019 [fl-dataset](https://jetbrains.team/p/ccrm/repositories/fl-dataset/files/docs/README.md).

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

These steps are the same for all the Fulline datasets. Below are examples for the [Python: Top 5k repos >50 stars](#python-top-5k-repos-50-stars)

```
virtualenv -p python3 .venv
source .venv/bin/activat
pip install -r requirements-preprocess.txt
```


1. Convert dataset to JSONL format
```
./convert-fulline-to-jsonl.py --data_dir='data/dataset-normalized-5000-with-imports'

pv data/dataset-normalized-5000-with-imports/*.jsonl \
  > data/jsonl/py5k-50.jsonl

# remove duplicated files by sha
pv data/jsonl/py5k-50.jsonl \
  | go run filter_dup_sha.go \
  > data/jsonl/py5k-50-uniq.jsonl
```

2. Train SentencePiece vocabulary on full content.
Follow https://github.com/google/sentencepiece#build-and-install-sentencepiece-command-line-tools-from-c-source

```
pv data/jsonl/py5k-50-uniq.jsonl \
  | jq -cr '.content' \
  > data/py5k-50-uniq.txt

# there is no need to do it, but in case you want to recover newlines do
head data/py5k-50-uniq-content.txt | sed 's/Ċ/\
/g'

# takes ~10min and 31Gb RAM on 96-core machine
time spm_train \
    --allow_whitespace_only_pieces \
    --noremove_extra_whitespaces \
    --pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 \
    --num_threads=96 \
    --vocab_size=32000 \
    --model_prefix=py5k-50 \
    --input=data/py5k-50-uniq.txt
```

3. Generate train/validation splits
Split should preferable be done by-project, to avoid leakig information between splits.
By default, files are order by project, so we are "leaking" at most one project.
```
wc -l data/py5k-50-uniq.txt
 210815
head -n 170000 data/py5k-50-uniq.txt > data/py5k-50.train.txt
tail -n 40815 data/py5k-50-uniq.txt > data/py5k-50.test.txt
```

Split by-file may be generated the same way, just shuffle lines of JSONL before
```
pv py5k-50-uniq.jsonl | perl -MList::Util=shuffle -e 'print shuffle <>;'
```

Create 5 shards
```
split -da 4 -l $((`wc -l < data/py5k-50.train.txt`/5)) data/py5k-50.train.txt data/py5k-50.train.txt- --additional-suffix="-of-0005"
```

Change `DATA_DIR` in appropriate Task defined in `tasks.py`.


### cuBERT github_python_minus_ethpy150open_dedup

From 2016 Github dump on [BigQuery public dataset](https://github.com/google-research/google-research/tree/master/cubert#collection-query)

 * Python: 42Gb / 26Gb
 * Files: total 7,176,801 / uniq 3,820,448

 1. create a BigQuery table pointing to GCS `gs://cubert/20200621_Python/github_python_minus_ethpy150open_deduplicated_manifest/manifest.jsontxt-*`
 2. run the query (processing 2.3 Tb)
    ```sql
     SELECT content.id as sha, content.content, content.size, meta.filepath, meta. license, meta.repository
     FROM
     (SELECT * FROM `data-analytics-experiments.github_python.cubert_python`) as meta
     INNER JOIN `bigquery-public-data.github_repos.contents` as content
       ON meta.id = content.id
     WHERE content.binary = FALSE
    ```
 3. save results to another BigQuery table, export to GCS bucket

To preprocess, do
```
gsutil -m cp gs://t5-codex/data/github_python_minus_ethpy150open_dedup data
ls data/py_file_content.jsonl-* | parallel "gzcat {} | go run preprocess_content.go"
#TODO rename according to slipts
```


 ### top400k Github DB 2020

 * Python 21.9GB
 * Files: 3,140,462 uniq

## Train the model

```
virtualenv -p python3 .venv-train
source .venv-train/bin/activat
pip install -r requirements-train.txt
```

To test LM Task definiton and input pipeline
```
python ./print_dataset.py
```


```
export PROJECT=<project-id>
export ZONE=<zone>
export TPU_NAME=t5-tpu
export TPU_SIZE=v2-8
export BUCKET=gs://t5-codex
export DATA_DIR="${BUCKET}/data"
export MODEL_DIR="${BUCKET}/models"
EXPORT_DIR="${MODEL_DIR}/export"
export TASK_NAME='fl_py_50stars_top5k_2019'
```

Create TPU from Cloud VM
```
ctpu up --name=$TPU_NAME --project=$PROJECT --zone=$ZONE --tpu-size=$TPU_SIZE \
        --tpu-only --noconf
```

Train
```
python -m t5.models.mesh_transformer_main \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --module_import="codeT5" \
  --gin_location_prefix="codeT5/gin/" \
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

## Cache the dataset
To cache the dataset on GCS as .tfrecords:
```
gcloud auth application-default login
pip install apache-beam[gcp] python-snappy
python -m seqio.scripts.cache_tasks_main \
 --module_import="codeT5.tasks" \
 --tasks="${TASK_NAME}" \
 --output_cache_dir="${BUCKET}/cache" \
 --alsologtostderr \
 --pipeline_options=["--runner=DirectRunner","--direct_num_workers 10"]
```


## Evaluate

Eval on latest checkpoint \w full decoding
(27 Min initial padding on un-cached dataset :/)
```
python -m t5.models.mesh_transformer_main  \
  --tpu="$TPU_ADDRESS" \
  --model_dir="$MODEL_DIR" \
  --t5_tfds_data_dir="$DATA_DIR" \
  --module_import="codeT5" \
  --gin_location_prefix="codeT5/gin/" \
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

```
python -m t5.models.mesh_transformer_main  \
  --tpu="$TPU_ADDRESS" \
  --model_dir="$MODEL_DIR" \
  --t5_tfds_data_dir="$DATA_DIR" \
  --module_import="codeT5" \
  --gin_location_prefix="codeT5/gin/" \
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

## Export the model

```
python -m t5.models.mesh_transformer_main \
  --gcp_project="$PROJECT" \
  --tpu_zone="$ZONE" \
  --model_dir="$MODEL_DIR" \
  --module_import="codeT5" \
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

```
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

```
time curl -d '{"inputs": ["import tensorflow as"]}' \
  -X POST "http://localhost:8501/v1/models/$MODEL_NAME:predict"
```
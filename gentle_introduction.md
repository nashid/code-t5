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

1. Download `Top 5k repos >50 stars` dataset
```shell
wget https://5k-dataset.s3.amazonaws.com/v3/dataset-normalized-5000-with-imports.tar.gz
tar -xzf dataset-normalized-5000-with-imports.tar.gz 
```

2. Select a couple of repositories for dev set, e.g. all repositories started with `A`
```shell
mkdir dataset-dev
cp -R \
  dataset-normalized-5000-with-imports/v3/languages/Python/.py/A* \
  dataset-dev/v3/languages/Python/.py 
cp -R \
  dataset-normalized-5000-with-imports/v3/repositories/A* \
  dataset-dev/v3/repositories
```

## Preprocessing

The same as for any FLCC dataset, refer to [`README`](./README.md#preprocessing) to see preprocessing instructions.

## Training

`t5` library allows to train model not only on TPU, but on GPU as well.

Specify dev parameters in `gin` configs:
1. TODO: how to set mixture/task in `dataset.gin`
2. TODO: set dev model hyperparameters in `shared-prefix_lm.gin`, is it possible to fit on 1 GPU? CPU only?

Run training with:
```shell
python -m t5.models.mesh_transformer_main \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --module_import="code_t5" \
  --gin_location_prefix="gin_configs/" \
  --gin_file="models/shared-prefix_lm.gin" \
  --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0']" \
  --gin_param="run.keep_checkpoint_max = 8" \
  --gin_param="MIXTURE_NAME = '${TASK_NAME}'" \
  --gin_param="run.train_steps = 10000"
```

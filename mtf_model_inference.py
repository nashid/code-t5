import os
import argparse
from contextlib import contextmanager

import t5.models
import tensorflow as tf
import seqio

import codeT5.models
from mock_data import inputs


@contextmanager
def tf_verbosity_level(level):
  og_level = tf.compat.v1.logging.get_verbosity()
  tf.compat.v1.logging.set_verbosity(level)
  yield
  tf.compat.v1.logging.set_verbosity(og_level)

def main(args):
  MODEL_DIR=os.path.join(args.models_dir, args.arch) #"gs://t5-codex/models/arch-lm_v1-lm" # "gs://t5-codex/models/base_shared_1k"

  model = codeT5.models.CustomMtfModel(
      model_dir=MODEL_DIR,
      tpu=None,
      model_type="lm" if "arch-lm" in args.arch else "bitransformer",
      model_parallelism=1,
      batch_size=2,
      sequence_length={"inputs": 128, "targets": 64},
  )

  # Write out the supplied questions to text files.
  predict_inputs_path = "predict_inputs.txt"
  predict_outputs_path = f"predict_outputs-{args.arch}-{args.temp}.txt"
  # Mock data is already preprocesed, but it can also be manually applied here.
  with tf.io.gfile.GFile(predict_inputs_path, "w") as f:
    for i in inputs:
      f.write(i.replace("\n", "Ċ"))
      f.write("\n")

  vocab = seqio.SentencePieceVocabulary("models/py5k-50.model", t5.data.DEFAULT_EXTRA_IDS)
  with tf_verbosity_level('ERROR'):
    model.predict(
        checkpoint_steps=-1, #410400,
        input_file=predict_inputs_path,
        output_file=predict_outputs_path,
        temperature=args.temp,
        vocabulary=vocab,
    )

  #prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"))
  #print("\nPredictions using checkpoint %s:\n" % prediction_files[-1].split("-")[-1])
  with tf.io.gfile.GFile(predict_outputs_path) as f: #prediction_files[-1]
    for i, o in zip(inputs, f):
      if o:
        print(i.replace('Ċ', '\n'))
        print(o.replace('Ċ', '\n'))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run model inference using Mesh Tensorflow API', )

    parser.add_argument(
        '-t', '--temp',
        type=float, default=0.6,
        help="Sampling temperature")
    parser.add_argument(
        '-a', '--arch',
        type=str, default="arch-lm_v1-lm",
        help="Model architecture")
    parser.add_argument(
        '-m', '--models_dir',
        type=str, default='models', # or 'gs://t5-codex/models'
        help='Path to the dir with sub-directories for individual checkpoints')


    args = parser.parse_args()
    main(args)

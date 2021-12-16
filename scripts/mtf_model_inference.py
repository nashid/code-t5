import argparse
import os
from contextlib import contextmanager
from os.path import join

import seqio
import t5.models
import tensorflow as tf

import code_t5.models
from code_t5.test.resources.mock_data import inputs


@contextmanager
def tf_verbosity_level(level):
    og_level = tf.compat.v1.logging.get_verbosity()
    tf.compat.v1.logging.set_verbosity(level)
    yield
    tf.compat.v1.logging.set_verbosity(og_level)


def main(temp, arch, models_dir, tpu):
    # "gs://t5-codex/models/arch-lm_v1-lm"
    # "gs://t5-codex/models/base_shared_1k"
    model_dir = join(models_dir, arch)
    gpu = tf.test.is_gpu_available()

    model = code_t5.models.CustomMtfModel(
        model_dir=model_dir,
        tpu=tpu if tpu else None,
        mesh_shape="model:1,batch:1" if gpu else None,
        mesh_devices=["gpu:0"] if gpu else None,
        model_type="lm" if "arch-lm" in arch else "bitransformer",
        model_parallelism=1,
        batch_size=2,
        sequence_length={"inputs": 128, "targets": 64},
    )

    # Write out the supplied questions to text files.
    predict_inputs_path = "predict_inputs.txt"
    predict_outputs_path = f"predict_outputs-{arch}-{temp}.txt"
    # Mock data is already preprocessed, but it can also be manually applied here.
    with tf.io.gfile.GFile(predict_inputs_path, "w") as f:
        for i in inputs:
            f.write(i.replace("\n", "Ċ"))
            f.write("\n")

    vocab = seqio.SentencePieceVocabulary(join(os.path.dirname(models_dir), "py5k-50.model"), t5.data.DEFAULT_EXTRA_IDS)
    with tf_verbosity_level("ERROR"):
        model.predict(
            checkpoint_steps=-1,  # 410400,
            input_file=predict_inputs_path,
            output_file=predict_outputs_path,
            temperature=temp,
            vocabulary=vocab,
        )

    prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"))
    print("\nPredictions using temperature {}, checkpoint {}:\n".format(temp, prediction_files[-1].split("-")[-1]))
    with tf.io.gfile.GFile(prediction_files[-1]) as f:
        for i, o in zip(inputs, f):
            if o:
                print(i.replace("Ċ", "\n"))
                print(o.replace("Ċ", "\n"))
                print()


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Run model inference using Mesh Tensorflow API")

    _parser.add_argument("-t", "--temp", type=float, default=0.6, help="Sampling temperature")
    _parser.add_argument("-a", "--arch", type=str, default="arch-lm_v1-lm", help="Model architecture")
    _parser.add_argument(
        "-m",
        "--models_dir",
        type=str,
        default="models",  # or 'gs://t5-codex/models'
        help="Path to the dir with sub-directories for individual checkpoints",
    )
    _parser.add_argument("--tpu", type=str, default="", help="Use TPU with the given address or name")

    _args = _parser.parse_args()
    main(_args.temp, _args.arch, _args.models_dir, _args.tpu)

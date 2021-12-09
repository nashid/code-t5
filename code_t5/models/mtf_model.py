import os
from typing import Mapping, Optional

import gin
import t5
import tensorflow as tf
from mesh_tensorflow.transformer import utils as mtf_utils
from mesh_tensorflow.transformer.vocabulary import Vocabulary
from t5.models import mesh_transformer
from t5.models import utils
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import TPUEstimator


@gin.configurable
def decode_from_file_utf(
    estimator: TPUEstimator,
    vocabulary: Vocabulary,
    model_type: str,
    batch_size: int,
    sequence_length: Mapping[str, int],
    checkpoint_path: Optional[str] = None,
    input_filename: str = "predict_inputs.txt",
    output_filename: str = "predict_outputs.txt",
    eos_id: int = 1,
    repeats: int = 1,
):
    """
    Decode from a text file and write to output_filename.

    Copy of the
    https://github.com/tensorflow/mesh/blob/cfc7a6754ca6fec88c494d51742dbefa75c81844/mesh_tensorflow/transformer/utils.py#L1450
    The difference with a library version in the line, where we decode prediction before recording it to a file:
    decodes = [decoded_output.decode('utf-8') for decoded_output in decodes[:dataset_size]]

    @param estimator: an instance of a tpu estimator for decoding.
    @param vocabulary: vocabulary to decode prediction.
    @param model_type: model type name.
    @param batch_size: batch size for decoding.
    @param sequence_length: sizes of sequences, e.g. {"inputs": 512, "targets": 128}.
    @param checkpoint_path: optional path to load checkpoint.
    @param input_filename: path to file with inputs to decode.
    @param output_filename: path to file to save decoded output.
    @param eos_id: id of EOS token in vocabulary.
    @param repeats: the number of times to repeat each input.
    """
    inputs = mtf_utils.get_inputs_from_file(input_filename)

    all_input_ids = mtf_utils.encode_inputs(
        inputs, vocabulary, model_type, batch_size, sequence_length["inputs"], eos_id=eos_id
    )

    def input_fn(params):
        del params
        dataset = tf.data.Dataset.from_tensor_slices({"inputs": all_input_ids})
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensors(x).repeat(repeats))
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    checkpoint_step = mtf_utils.get_step_from_checkpoint_path(checkpoint_path)
    decodes = list(mtf_utils.decode(estimator, input_fn, vocabulary, checkpoint_path=checkpoint_path))
    # Remove any padded examples
    dataset_size = len(inputs) * repeats

    decodes = [decoded_output.decode("utf-8") for decoded_output in decodes[:dataset_size]]

    output_filename = "{}-{}".format(output_filename, checkpoint_step)
    mtf_utils.write_lines_to_file(decodes, output_filename)


def _parse_operative_config(model_dir):
    with gin.unlock_config():
        gin.parse_config_file(
            os.path.join(model_dir, "operative_config.gin"), skip_unknown=mesh_transformer.DEPRECATED_GIN_REFERENCES
        )


class CustomMtfModel(t5.models.MtfModel):
    """
    Copy of the
    https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/models/mtf_model.py#L34

    The difference with a library version where we provide a custom decode_from_file function to mtf_utils.infer_model:
    decode_fn=decode_from_file_utf
    """

    def predict(
        self, input_file, output_file, checkpoint_steps=-1, beam_size=1, temperature=1.0, keep_top_k=-1, vocabulary=None
    ):
        if checkpoint_steps == -1:
            checkpoint_steps = utils.get_latest_checkpoint_from_dir(self._model_dir)

        _parse_operative_config(self._model_dir)
        with gin.unlock_config():
            gin.bind_parameter("Bitransformer.decode.beam_size", beam_size)
            gin.bind_parameter("Bitransformer.decode.temperature", temperature)
            gin.bind_parameter("Bitransformer.decode.sampling_keep_top_k", keep_top_k)
            gin.bind_parameter("mtf_model.decode_from_file_utf.input_filename", input_file)
            gin.bind_parameter("mtf_model.decode_from_file_utf.output_filename", output_file)
        if vocabulary is None:
            vocabulary = utils.get_vocabulary()
        mtf_utils.infer_model(
            self.estimator(vocabulary),
            vocabulary,
            self._sequence_length,
            self.batch_size,
            self._model_type,
            self._model_dir,
            checkpoint_steps,
            decode_fn=decode_from_file_utf,
        )

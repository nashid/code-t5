
import os

import gin
from mesh_tensorflow.transformer import utils as mtf_utils
import t5
from t5.models import mesh_transformer
from t5.models import utils
import tensorflow as tf

# Copy of the https://github.com/tensorflow/mesh/blob/cfc7a6754ca6fec88c494d51742dbefa75c81844/mesh_tensorflow/transformer/utils.py#L1450
#
# The difference with a library version is this line, where we decode prediction before recording it to a file:
# decodes = [decoded_output.decode('utf-8') for decoded_output in decodes[:dataset_size]]
@gin.configurable
def decode_from_file_utf(estimator,
                     vocabulary,
                     model_type,
                     batch_size,
                     sequence_length,
                     checkpoint_path=None,
                     input_filename='predict_inputs.txt',
                     output_filename='predict_outputs.txt',
                     eos_id=1,
                     repeats=1):
  """Decode from a text file and write to output_filename.

  Args:
    estimator: a TPUEstimator
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    model_type: a string
    batch_size: an integer
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    checkpoint_path: an optional string
    input_filename: a string
    output_filename: a string
    eos_id: EOS id
    repeats: an integer, the number of times to repeat each input.
  """
  inputs = mtf_utils.get_inputs_from_file(input_filename)

  all_input_ids = mtf_utils.encode_inputs(inputs, vocabulary, model_type, batch_size,
                                sequence_length["inputs"], eos_id=eos_id)
  def input_fn(params):
    del params
    dataset = tf.data.Dataset.from_tensor_slices({"inputs": all_input_ids})
    dataset = dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensors(x).repeat(repeats))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  checkpoint_step = mtf_utils.get_step_from_checkpoint_path(checkpoint_path)
  decodes = list(mtf_utils.decode(
      estimator, input_fn, vocabulary, checkpoint_path=checkpoint_path))
  # Remove any padded examples
  dataset_size = len(inputs) * repeats


  decodes = [decoded_output.decode('utf-8') for decoded_output in decodes[:dataset_size]]

  output_filename = "{}-{}".format(output_filename, checkpoint_step)
  mtf_utils.write_lines_to_file(decodes, output_filename)


# Copy of the https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/models/mtf_model.py#L34
#
# The difference with a library version is where we provide a custom decode_from_file function to mtf_utils.infer_model
# decode_fn=decode_from_file_utf
def _parse_operative_config(model_dir):
  with gin.unlock_config():
    gin.parse_config_file(
        os.path.join(model_dir, "operative_config.gin"),
        skip_unknown=mesh_transformer.DEPRECATED_GIN_REFERENCES)

class CustomMtfModel(t5.models.MtfModel):
    def __init__(
        self,
        model_dir,
        tpu,
        tpu_job_name=None,
        tpu_zone=None,
        gcp_project=None,
        tpu_topology="v2-8",
        model_parallelism=8,
        batch_size=("sequences_per_batch", 1),
        sequence_length=None,
        model_type="bitransformer",
        layout_rules="ensemble:ensemble,batch:batch,d_ff:model,heads:model,vocab:model,experts:batch",
        mesh_shape=None,
        mesh_devices=None,
        autostack=True,
        learning_rate_schedule=None,
        keep_checkpoint_max=None,
        save_checkpoints_steps=5000,
        optimizer=None,
        predict_fn=None,
        variable_filter=None,
        ensemble_inputs=None,
        iterations_per_loop=100,
        extra_gin_bindings=None):
        super().__init__(
            model_dir,
            tpu,
            tpu_job_name=tpu_job_name,
            tpu_zone=tpu_zone,
            gcp_project=gcp_project,
            tpu_topology=tpu_topology,
            model_parallelism=model_parallelism,
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_type=model_type,
            layout_rules=layout_rules,
            mesh_shape=mesh_shape,
            mesh_devices=mesh_devices,
            autostack=autostack,
            learning_rate_schedule=learning_rate_schedule,
            keep_checkpoint_max=keep_checkpoint_max,
            save_checkpoints_steps=save_checkpoints_steps,
            optimizer=optimizer,
            predict_fn=predict_fn,
            variable_filter=variable_filter,
            ensemble_inputs=ensemble_inputs,
            iterations_per_loop=iterations_per_loop,
            extra_gin_bindings=extra_gin_bindings
        )

    def predict(self, input_file, output_file, checkpoint_steps=-1,
                beam_size=1, temperature=1.0, keep_top_k=-1, vocabulary=None):
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
            self.estimator(vocabulary), vocabulary, self._sequence_length,
            self.batch_size, self._model_type, self._model_dir, checkpoint_steps,
            decode_fn=decode_from_file_utf)
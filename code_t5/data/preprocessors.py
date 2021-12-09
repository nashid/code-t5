from typing import Mapping

import tensorflow as tf

from code_t5.constants import NEWLINE


def prefix_lm(dataset: tf.data.Dataset):
    """
    Set incoming text as target and return in T5 format for LM.
    """

    def _to_inputs_and_targets(ex: tf.Tensor) -> Mapping[str, tf.Tensor]:
        return {
            "inputs": "",
            "targets": ex,
        }

    return dataset.map(_to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def split_lines(dataset: tf.data.Dataset):
    """
    Split each text document (file) by line, keeping the newlines.
    Should be applied before seqio.preprocessors.tokenize.
    """

    def _append_newline(x):
        x = tf.expand_dims(x, axis=-1)
        x = tf.concat([x, tf.broadcast_to([NEWLINE], x.shape)], axis=-1)
        return tf.strings.reduce_join(x, axis=-1)

    return (
        dataset.map(lambda x: tf.strings.split(x, NEWLINE))
        .flat_map(tf.data.Dataset.from_tensor_slices)
        .map(_append_newline)
    )

import tensorflow as tf
from seqio import utils


@utils.map_over_dataset
def print_dataset(features):
    """tf.print dataset fields for debugging purposes."""
    return {k: tf.print(v, [v], k + ": ") for k, v in features.items()}

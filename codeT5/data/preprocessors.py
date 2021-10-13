
import tensorflow as tf


def fl_preprocessor(ds):
    def _to_inputs_and_targets(ex):
        return {
            "inputs": "",
            "targets": ex,
        }
    return ds.map(_to_inputs_and_targets,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


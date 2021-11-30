
import tensorflow as tf

NEWLINE = 'Ċ'

def fl_preprocessor(ds):
    def _to_inputs_and_targets(ex):
        return {
            "inputs": "",
            "targets": ex,
        }
    return ds.map(_to_inputs_and_targets,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

def split_lines(ds):
    """Split each text documemnt (file) by line, keeping the newlines.
       Should be applied before seqio.preprocessors.tokenize.
    """
    def _append_newline(x):
        x = tf.expand_dims(x, axis=-1)
        x = tf.concat([x, tf.broadcast_to([NEWLINE], x.shape)], axis=-1)
        return tf.strings.reduce_join(x, axis=-1)

    return ds\
        .map(lambda x: tf.strings.split(x, NEWLINE))\
        .flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))\
        .map(_append_newline)

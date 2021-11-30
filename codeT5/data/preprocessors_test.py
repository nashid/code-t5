from absl.testing import absltest
from seqio import test_utils
from seqio import utils
import tensorflow.compat.v2 as tf
import t5

import codeT5.tasks
from codeT5.data import preprocessors as prep

tf.compat.v1.enable_eager_execution()

assert_dataset = test_utils.assert_dataset


class PreprocessorsTest(tf.test.TestCase):

    def test_fl(self):
      dataset = tf.data.Dataset.from_tensor_slices([['That is good.']])
      dataset = prep.fl_preprocessor(dataset)
      assert_dataset(dataset, {'inputs': '', 'targets': 'That is good.'})

    def test_split_lines(self):
      dataset = tf.data.Dataset.from_tensor_slices([
        'FirstĊĊSecond',
        'First lineĊSecond line'
      ])
      dataset = prep.split_lines(dataset)
      dataset = prep.fl_preprocessor(dataset)

      assert_dataset(dataset, [
        {'inputs': '', 'targets': 'FirstĊ'},
        {'inputs': '', 'targets': 'Ċ'},
        {'inputs': '', 'targets': 'SecondĊ'},
        {'inputs': '', 'targets': 'First lineĊ'},
        {'inputs': '', 'targets': 'Second lineĊ'},
      ])

    def test_group_by_newlines(self):
      dataset = tf.data.Dataset.from_tensor_slices([
        'FirstĊĊSecond',
        'First lineĊSecond line'
      ])
      dataset = prep.split_lines(dataset)
      # encode
      dataset = dataset.map(codeT5.tasks.vocab.encode_tf)
      dataset = prep.fl_preprocessor(dataset)
      assert_dataset(dataset, [
        {'inputs': '', 'targets': [5366, 3]},  #'FirstĊ'},
        {'inputs': '', 'targets': [5, 3]},     #'Ċ'},
        {'inputs': '', 'targets': [11965, 3]}, #'SecondĊ'},
        {'inputs': '', 'targets': [5366, 288, 3]}, #'First lineĊ'},
        {'inputs': '', 'targets': [11965, 288, 3]},  #'Second lineĊ'},
      ])
      # group lines
      dataset = t5.data.preprocessors.reduce_concat_tokens(dataset, feature_key='targets', batch_size=2)
      assert_dataset(dataset, [
        {'targets': [5366, 3, 5, 3]},  #'FirstĊĊ'},
        {'targets': [11965, 3, 5366, 288, 3]}, #'SecondĊFirst lineĊ'},
        {'targets': [11965, 288, 3]},  #'Second lineĊ'},
      ])


@utils.map_over_dataset
def print_dataset(features):
  """tf.print dataset fields for debugging purposes."""
  return {k: tf.print(v, [v], k + ': ') for k, v in features.items()}


if __name__ == '__main__':
  absltest.main()

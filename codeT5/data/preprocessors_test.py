import functools

from absl.testing import absltest
# import gin
# import seqio
from seqio import test_utils
from codeT5.data import preprocessors as prep
import tensorflow.compat.v2 as tf

tf.compat.v1.enable_eager_execution()

# mock = absltest.mock
assert_dataset = test_utils.assert_dataset


class PreprocessorsTest(tf.test.TestCase):
    
    def test_lm(self):
      dataset = tf.data.Dataset.from_tensor_slices([['That is good.']])
      dataset = prep.fl_preprocessor(dataset)
      assert_dataset(dataset, {'inputs': '', 'targets': 'That is good.'})

    def test_split_lines(self):
      dataset = tf.data.Dataset.from_tensor_slices([['That is good.']])
      dataset = prep.fl_preprocessor(dataset)

      assert_dataset(dataset, {'inputs': '', 'targets': 'That is good.'})


if __name__ == '__main__':
  absltest.main()

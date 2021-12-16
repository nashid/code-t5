import t5
from seqio.test_utils import assert_dataset

from code_t5.data.preprocessors import *

tf.compat.v1.enable_eager_execution()


class PreprocessorsTest(tf.test.TestCase):
    _hand_lookup = {"EOS": 0, NEWLINE: 1, "FirstĊ": 2, "SecondĊ": 3, "lineĊ": 4, "First": 5, "Second": 6}
    _init = tf.lookup.KeyValueTensorInitializer(
        tf.constant(list(_hand_lookup.keys())), tf.constant(list(_hand_lookup.values()), dtype=tf.int64)
    )
    _vocab = tf.lookup.StaticVocabularyTable(_init, 1)

    def test_fl(self):
        dataset = tf.data.Dataset.from_tensor_slices([["That is good."]])
        dataset = text_file_per_line(dataset)
        assert_dataset(dataset, {"inputs": "", "targets": "That is good."})

    def test_split_lines(self):
        dataset = tf.data.Dataset.from_tensor_slices(["FirstĊĊSecond", "First lineĊSecond line"])
        dataset = split_lines(dataset)
        dataset = text_file_per_line(dataset)

        assert_dataset(
            dataset,
            [
                {"inputs": "", "targets": "FirstĊ"},
                {"inputs": "", "targets": "Ċ"},
                {"inputs": "", "targets": "SecondĊ"},
                {"inputs": "", "targets": "First lineĊ"},
                {"inputs": "", "targets": "Second lineĊ"},
            ],
        )

    def test_group_by_newlines(self):
        dataset = tf.data.Dataset.from_tensor_slices(["FirstĊĊSecond", "First lineĊSecond line"])
        dataset = split_lines(dataset)
        # encode
        dataset = dataset.map(lambda x: tf.strings.join([x, "EOS"], separator=" ")).map(
            lambda x: self._vocab[tf.strings.split([x]).values]
        )
        dataset = text_file_per_line(dataset)
        assert_dataset(
            dataset,
            [
                {"inputs": "", "targets": [2, 0]},  #'FirstĊ'},
                {"inputs": "", "targets": [1, 0]},  #'Ċ'},
                {"inputs": "", "targets": [3, 0]},  #'SecondĊ'},
                {"inputs": "", "targets": [5, 4, 0]},  #'First lineĊ'},
                {"inputs": "", "targets": [6, 4, 0]},  #'Second lineĊ'},
            ],
        )
        dataset = t5.data.preprocessors.reduce_concat_tokens(dataset, feature_key="targets", batch_size=2)
        assert_dataset(
            dataset,
            [
                {"targets": [2, 1]},  #'FirstĊĊ'},
                {"targets": [3, 5, 4]},  #'SecondĊFirst lineĊ'},
                {"targets": [6, 4]},  #'Second lineĊ'},
            ],
        )

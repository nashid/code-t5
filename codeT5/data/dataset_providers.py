from typing import Iterable, Mapping, Optional, Union

import seqio
import tensorflow.compat.v2 as tf

class GzTextLineDataSource(seqio.FileDataSource):
  """A `TextLineDataSource` that reads lines of text from a compressed file."""

  def __init__(
      self,
      split_to_filepattern: Mapping[str, Union[str, Iterable[str]]],
      skip_header_lines: int = 0,
      num_input_examples: Optional[Mapping[str, int]] = None,
  ):
    """GzTextLineDataSource constructor.
    Thi is a copy of seqio.TextLineDataSource \w compression_type set.
    Ideally https://github.com/google/seqio/blob/b9ca50912c4862e70b8496f72d7f931693009e3b/seqio/dataset_providers.py#L514
    should be refactored to accept it as parameter in constructor.

    Args:
      split_to_filepattern: a mapping from split names to filepatterns to be
        expanded with glob.
      skip_header_lines: int, number of header lines to skip in each source
        file.
      num_input_examples: dict or None, an optional dictionary mapping split to
        its size in number of input examples (before preprocessing). The
        `num_input_examples` method will return None if not provided.
    """
    # Used during caching.
    self._skip_header_lines = skip_header_lines

    def read_file_fn(filepattern):
      return tf.data.TextLineDataset(filepattern, compression_type="GZIP").skip(skip_header_lines)

    super().__init__(
        read_file_fn=read_file_fn,
        split_to_filepattern=split_to_filepattern,
        num_input_examples=num_input_examples)

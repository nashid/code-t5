# coding=utf-8
# Copyright 2021 JetBrains.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import seqio
import t5.data

MixtureRegistry = seqio.MixtureRegistry

MixtureRegistry.add(
    "fl_py_2019_mix", # all python repos \w >10 stars in FullLine dataset
    ["fl_py_50stars_2019", "fl_py_10stars_2019"],
    default_rate=t5.data.rate_num_examples)

MixtureRegistry.add(
    "fl_bq_py_mix", # all Python data in FullLine and Github BigQuery datasets
    ["bq_py_2016_minus_ethpy150", "fl_py_50stars_2019", "fl_py_10stars_2019"],
    default_rate=t5.data.rate_num_examples)

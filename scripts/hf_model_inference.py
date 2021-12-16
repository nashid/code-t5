"""
Example of running an inference using t5 pyTorch HfPyTorchModel API.
Requires both, pyTorch and Tensorflow

To use it, do:

# get the data
mkdir -p models/base-t5.1.1
gsutil -m cp 'gs://t5-codex/models/base-t5.1.1/model.ckpt-187500*' models/base-t5.1.1

# run the inference for examples frmo mock_data.py
TRANSFORMERS_VERBOSITY=info TRANSFORMERS_OFFLINE=1 python3 hf_model_inference.py
"""

import t5.models
import torch
import transformers


from code_t5.test import inputs

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

t5_config = transformers.T5Config.from_dict(
    {
        "architectures": ["T5ForConditionalGeneration"],
        "vocab_size": 32128,
        "d_ff": 3072,
        "d_kv": 64,
        "d_model": 768,
        "decoder_start_token_id": 0,
        "dropout_rate": 0.0,
        "eos_token_id": 1,
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "layer_norm_epsilon": 1e-06,
        "model_type": "t5",
        "n_positions": 512,
        "num_heads": 12,
        "num_layers": 12,
        "output_past": True,
        "pad_token_id": 0,
        "relative_attention_num_buckets": 32,
        "task_specific_params": {},
    }
)
t5_config.to_json_file("models/bi_v1_shared-prefix_lm/config.json")

# this API can not load TF Checkpints (and has a hard-coded type check)
# model = t5.models.HfPyTorchModel(transformers.T5Config.from_json_file('models/base-t5.1.1/t5_config.json'), "./hft5/", device)

# Transformers lib can load TF checkpoins into pyTorch \wo manual conversion first (but it's slower)
#  https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
pt_model = transformers.T5ForConditionalGeneration.from_pretrained(
    "models/bi_v1_shared-prefix_lm/model.ckpt-276400.index", from_tf=True, config=t5_config  # shared
)

"""Alternatively, convert the TF checkpoint to pyTorch manually

$ TRANSFORMERS_VERBOSITY=info TRANSFORMERS_OFFLINE=1 transformers-cli convert --model_type t5 \
  --tf_checkpoint models/base-t5.1.1/model.ckpt-187500 \
  --config models/base-t5.1.1/t5_config.json \
  --pytorch_dump_output models/base-t5.1.1
"""
# then, comment the above and load it only with
model = t5.models.HfPyTorchModel("models/base-t5.1.1", "./hft5/", device)
model._model = pt_model

tokenizer = transformers.T5Tokenizer("models/py5k-50.model")
input_ids = tokenizer('open a file "f.txt" in write mode', return_tensors="pt").input_ids
outputs = pt_model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

predictions = model.predict(
    inputs,
    sequence_length={"inputs": 512, "targets": 512},
    batch_size=2,
)
for inp, pred in zip(inputs, predictions):
    i = inp.replace("Ċ", "\n")
    p = pred.replace("Ċ", "\n")
    print(f"{i}'{p}\n")

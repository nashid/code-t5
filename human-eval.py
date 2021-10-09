import argparse
from codeT5.models.mtf_model import CustomMtfModel
import os
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from human_eval.data import read_problems, write_jsonl
#from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL
#pip3 install -e 'git+http://github.com/openai/human-eval#egg=human-eval'

import codeT5.models


"""
Runs HumanEval over the given TF Checkpoint.
Can be run on TPUv2 \w --tpu or use auto-detected GPU (default).
"""


@contextmanager
def tf_verbosity_level(level):
  og_level = tf.compat.v1.logging.get_verbosity()
  tf.compat.v1.logging.set_verbosity(level)
  yield
  tf.compat.v1.logging.set_verbosity(og_level)

def pre_process(s):
    return s.replace("\n", "Ċ")
    
def post_process(s):
    """TODO(bzz): mimick the rest of the codex logic"""
    def drop_after(txt, s):
        if s not in txt:
            return txt
        stop = txt.index(s)
        return txt[:stop]

    txt = f" {s}".replace("Ċ", "\n")
    txt = drop_after(txt, "\nclass ")
    txt = drop_after(txt, "\ndef ")
    return txt


def main(args):
    MODEL_DIR=os.path.join(args.models_dir, args.arch)
    gpu = tf.test.is_gpu_available()

    model = codeT5.models.CustomMtfModel(
        model_dir=MODEL_DIR,
        tpu=args.tpu if args.tpu else None,
        mesh_shape="model:1,batch:1" if gpu else None,
        mesh_devices=['gpu:0'] if gpu else None,
        model_type="lm" if "arch-lm" in args.arch else "bitransformer",
        model_parallelism=1,
        batch_size=2, # 8 on TPUv2-8?
        sequence_length={"inputs": 512, "targets": 512},
    )

    # prepare the input from HumanEval problems
    predict_inputs_path = "predict_inputs.txt"
    problems = read_problems(args.problem_file)
    sorted_problems = sorted(problems.keys(), key=lambda s: int(s[s.index("/") + 1:]) if "/" in s else -1)
    sorted_problems = np.repeat(sorted_problems, args.num_samples)
    with tf.io.gfile.GFile(predict_inputs_path, "w") as f:
        for i in sorted_problems:
                f.write(pre_process(problems[i]["prompt"]))
                f.write("\n")
        
    predict_outputs_path = "predict_outputs.txt"
    with tf_verbosity_level('ERROR'):
        model.predict(
            checkpoint_steps=-1,
            input_file=predict_inputs_path,
            output_file=predict_outputs_path,
            temperature=args.temp,
        )

    # read the output (suffix '-{checkpoint}' was added)
    results = []
    prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"), key=lambda s: int(s.split("-")[-1]))
    checkpoint = prediction_files[-1].split("-")[-1]
    print("\nPredictions using checkpoint %s:\n" % checkpoint)
    with tf.io.gfile.GFile(prediction_files[-1]) as f:
        for i, o in zip(sorted_problems, f):
            if o:
                output = post_process(o)
                task_result = {
                    "task_id": i,
                    "completion": output #" " + output if with_additional_space else output
                }
                results.append(task_result)
    output_file=f"{args.output_file}-{args.arch}-{checkpoint}.jsonl"
    print(f"Writing results to {output_file}")
    write_jsonl(output_file, results)

    #TODO(bzz): launch eval
    print(f"Launching humanEval on {args.problem_file}")
    #from human_eval.evaluation import evaluate_functional_correctness
    #evaluate_functional_correctness(output_file, , ,args.problem_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the TF checkpoint using Mesh Tensorflow API on HumanEval', )

    parser.add_argument(
        '-t', '--temp',
        type=float, default=0.6,
        help="Sampling temperature")
    parser.add_argument(
        '-n', '--num_samples',
        type=int, default=1,
        help="Number of samples")
    parser.add_argument(
        '-a', '--arch',
        type=str, default="arch-lm_v1-lm",
        help="Model architecture")
    parser.add_argument(
        '-m', '--models_dir',
        type=str, default='models', # or 'gs://t5-codex/models'
        help='Path to the dir with sub-directories for individual checkpoints')
    parser.add_argument(
        '--problem_file',
        type=str, default="HumanEval.jsonl.gz",
        help="Path to the JSONL file with problems to evaluate up on from https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz"
    )
    parser.add_argument(
        '--output_file',
        type=str, default=f"humanEval",
        help="Path to the JSONL file with problems to evaluate up on"
    )
    parser.add_argument(
        '--tpu', type=str, default="",
        help="Use TPU with the given address or name")

    args = parser.parse_args()
    main(args)

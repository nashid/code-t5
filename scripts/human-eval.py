"""
Runs HumanEval over the given TF Checkpoint.
Can be run on TPUv2 with --tpu or use auto-detected GPU (default).

Before usage:
1. Install python package:
pip install -e 'git+http://github.com/openai/human-eval#egg=human-eval'
2. Download data:
wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
"""

import argparse
from contextlib import contextmanager
from os.path import join

import tensorflow as tf

import code_t5.models
from human_eval.data import write_jsonl, read_problems


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


def main(temp, top_k, num_samples, batch_size, arch, models_dir, problem_file, output_file, tpu):
    model_dir = join(models_dir, arch)
    gpu = tf.test.is_gpu_available()

    model = code_t5.models.CustomMtfModel(
        model_dir=model_dir,
        tpu=tpu if tpu else None,
        mesh_shape="model:1,batch:1" if gpu else None,
        mesh_devices=["gpu:0"] if gpu else None,
        model_type="lm" if "arch-lm" in arch else "bitransformer",
        model_parallelism=1,
        batch_size=batch_size,
        sequence_length={"inputs": 512, "targets": 512},
    )

    # prepare the input from HumanEval problems
    predict_inputs_path = "predict_inputs.txt"
    problems = read_problems(problem_file)
    sorted_problems = sorted(problems.keys(), key=lambda s: int(s[s.index("/") + 1 :]) if "/" in s else -1)
    sorted_problems = [problem for problem in sorted_problems for _ in range(num_samples)]
    with tf.io.gfile.GFile(predict_inputs_path, "w") as f:
        for i in sorted_problems:
            f.write(pre_process(problems[i]["prompt"]))
            f.write("\n")

    predict_outputs_path = f"predict_outputs-{arch}.txt"
    with tf_verbosity_level("ERROR"):
        model.predict(
            checkpoint_steps=-1,
            input_file=predict_inputs_path,
            output_file=predict_outputs_path,
            temperature=temp,
            keep_top_k=top_k,
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
                task_result = {"task_id": i, "completion": output}  # " " + output if with_additional_space else output
                results.append(task_result)
    output_file = f"{output_file}-{arch}-{checkpoint}.jsonl"
    print(f"Writing results to {output_file}")
    write_jsonl(output_file, results)

    # TODO(bzz): launch eval
    print(f"Launching humanEval on {problem_file}")
    # from human_eval.evaluation import evaluate_functional_correctness
    # evaluate_functional_correctness(output_file, , ,args.problem_file)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Evaluate the TF checkpoint using Mesh Tensorflow API on HumanEval")

    _parser.add_argument("-t", "--temp", type=float, default=0.6, help="Sampling temperature")
    _parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="A value [1, vocab_size) used to sample from only the k most likely logits",
    )
    _parser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of samples")
    _parser.add_argument("-b", "--batch_size", type=int, default=2, help="Size of the batch")  # 8 on TPUv2-8?
    _parser.add_argument("-a", "--arch", type=str, default="arch-lm_v1-lm", help="Model architecture")
    _parser.add_argument(
        "-m",
        "--models_dir",
        type=str,
        default="models",  # or 'gs://t5-codex/models'
        help="Path to the dir with sub-directories for individual checkpoints",
    )
    _parser.add_argument(
        "--problem_file",
        type=str,
        default="HumanEval.jsonl.gz",
        help="Path to the JSONL file with problems to evaluate up on from https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz",
    )
    _parser.add_argument(
        "--output_file", type=str, default=f"humanEval", help="Path to the JSONL file with problems to evaluate up on"
    )
    _parser.add_argument("--tpu", type=str, default="", help="Use TPU with the given address or name")

    _args = _parser.parse_args()
    main(
        _args.temp,
        _args.top_k,
        _args.num_samples,
        _args.batch_size,
        _args.arch,
        _args.models_dir,
        _args.problem_file,
        _args.output_file,
        _args.tpu,
    )

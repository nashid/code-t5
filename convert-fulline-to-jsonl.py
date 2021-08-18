#!/usr/bin/env python3
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
"""Converts dataset from FullLine completion format to JSONL
https://jetbrains.team/p/ccrm/repositories/fl-dataset/files/docs/README.md

Output: JSONL schema: (content, sha, filepath, repository, license)

cat data/dataset-normalized-5000-with-imports/*.jsonl > data/jsonl/20.jsonl
rm -f data/dataset-normalized-5000-with-imports/*.jsonl
"""
import argparse
from collections import Counter
from functools import reduce
import glob
import hashlib
import itertools
import json
import multiprocessing
import os
import pathlib

import tqdm

FILE_SIZE_LIMIT = 1*1024*1024 # 1Mb limit


def process_single_repo(repo_path: str, data_dir: str):
    """Para
    """
    branch = "master"
    org_path, repo = os.path.split(repo_path.rstrip(os.path.sep))
    _, org = os.path.split(org_path)
    # print(f"\tprocessing '{org}/{repo}'")

    # read metadata from 'org/repo/branch/paths.json'
    paths = os.path.join(data_dir, "v3", "repositories", "repositories", org,
                         repo, branch, "paths.json")

    with open(paths, "rt", encoding="utf-8") as m:
        metadata = json.load(m)

    def process_file(org_repo, metadata, py_filepath, out):
        filename = os.path.basename(py_filepath)
        orig_filepath = metadata[filename]
        # name, ext = os.path.splitext(filename)
        # orig_name, time = name.rsplit("_", maxsplit=1)
        # print(f"'{py_filepath}' = {orig_name}, {time}, {ext} -> {orig_filepath}")
        data = {}
        data["repository"] = org_repo
        data["size"] = os.path.getsize(py_filepath)
        data["sha"] = ""
        data["content"] = ""
        if data["size"] < FILE_SIZE_LIMIT:
            with open(py_filepath, "rb") as f:
                bytes = f.read()
                data["sha"] = hashlib.sha256(bytes).hexdigest()
            data["content"] = ""
        data["filepath"] = orig_filepath
        data["license"] = ""

        out.write(json.dumps(data, ensure_ascii=False))
        out.write('\n')

    jsonl = f"{data_dir}/{org}_{repo}.jsonl"
    #print(f"\twriting to '{jsonl}'")
    files_skipped, fs = 0, 0
    with open(jsonl, 'w', encoding='utf-8') as f:
        repo_py_files = pathlib.Path(f"{repo_path}/{branch}/").glob("*")
        for py_filepath in repo_py_files:
            try:
                process_file(f"{org}/{repo}", metadata, py_filepath, f)
                fs+=1
            except Exception:
                files_skipped+=1

    return {"files": fs, "files_skipped": files_skipped}



def main(args):
    limit = args.limit
    all_py = os.path.join(args.data_dir, "v3", "languages", "Python", ".py")
    all_py_repos = f"{all_py}/*/*/"
    print(f"Listing {limit if limit else ''} repos in '{all_py_repos}'")

    #root, repos, _ = next(os.walk(all_py_repos))
    #repos = list(map(lambda x: os.path.join(root, x), repos))[:limit if limit else -1]
    # for _, repo in zip(range(limit), repos):
    #     print(f"{repo}")
    # for _, repo in zip(range(limit if limit else len(repos)), repos):
    #     print(f"{repo}")

    repos = glob.glob(all_py_repos)[:limit if limit else -1]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        input_args = itertools.zip_longest(repos, [], fillvalue=args.data_dir)
        results = pool.starmap(
            process_single_repo,
            tqdm.tqdm(input_args, total = len(repos)),
            chunksize=10,
        )
    

    print(f"{len(results)} results")
    print(reduce(lambda x,y: dict(Counter(x)+Counter(y)), results, {'files':0}))

    # for repo in repos:
    #     process_single_repo(repo, args.data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert to JSONL', )

    parser.add_argument('--data_dir', help="path to the Fulline dataset root")
    parser.add_argument('--output_dir', help="path to save the JSON dataset")
    parser.add_argument('-l',
                        '--limit',
                        type=int,
                        help="limit nubber of repos to process")

    args = parser.parse_args()
    main(args)

// Copyright 2021 JetBrains.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// CLI tool to extract and preprocess Athena export in JSONL.
// Convets files to line-based text format.
// Computes hash of the content, if needed.
package main

import (
	"bufio"
	"crypto/sha1"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

var (
	hashOnlyFlag = flag.Bool("h", false, "output hash only")

	outputFile         = "athena_py.*.txt"
	newlineReplacement = []byte("Ċ") // \n + 100 == \u0100a  # printf '\u010a\n' | hexdump -C
)

var usageMessage = `usage: go run preprocess_at_py_2020.go [-h] dir

Reads JSONL from STDIN and saves pre-processed and de-duplicated by SHA content in a file in dir.
`

type FileContent struct {
	Org      string `json:"owner"`
	Repo     string `json:"name"`
	License  string `json:"license"`
	Filepath string `json:"path"`
	Size     int32  `json:"size"`
	Sha      string `json:"content_sha"`
	Content  string `json:"content"`
}

//Reads JSONL from STDIN, extracts it's SHA1
// ls jsonl/20211130_*.gz | parallel "zcat {} | go run preprocess_at_py_2020.go -h txt"
// cat txt/athena_py.*.txt | sort | uniq > at_py_2020_sha.txt

//Save content in line-based text format
// ls jsonl/20211130_*.gz | parallel "zcat {} | go run preprocess_bq_py_2016.go txt"

func main() {
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()
	if len(args) != 1 {
		usage()
	}
	dir := args[0]

	f, err := os.CreateTemp(dir, outputFile)
	check(err)

	h := sha1.New()
	d := json.NewDecoder(os.Stdin)
	buf := make([]byte, 0, 64*1024)
	for {
		var m FileContent
		if err := d.Decode(&m); err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		// SHA1
		if *hashOnlyFlag {
			// We can not do that, as it has SHA256 instead :(
			// f.WriteString(m.Sha)

			// compute SHA1
			io.WriteString(h, m.Content)
			f.WriteString(fmt.Sprintf("%x\n", h.Sum(nil)))
			continue
		}

		writeFileNelinesReplaced(f, buf, m.Content)
		f.WriteString("\n")

		// h.Reset()
	}

	err = f.Close()
	check(err)
}

func writeFileNelinesReplaced(f *os.File, buf []byte, content string) {
	scanner := bufio.NewScanner(strings.NewReader(content))
	scanner.Buffer(buf, 2*1024*1024)
	for scanner.Scan() {
		//TODO(bzz): skip generated files
		//TODO(bzz): skip files size > 1Mb
		//TODO(bzz): skip files \w lines too long (max > 1000, avg > 100)

		f.Write(scanner.Bytes())
		f.Write(newlineReplacement)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

func check(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, usageMessage)
	os.Exit(2)
}

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

// CLI tool to extract and preprocess BigQuery export in JSONL to text format
// Computes MD5 of the content
package main

import (
	"bufio"
	"crypto/md5"
	"encoding/json"
	"io"
	"log"
	"os"
	"strings"
)

var newlineReplacement = []byte("Ċ") // \n + 100 == \u0100a  # printf '\u010a\n' | hexdump -C

type Message struct {
	Sha        string
	Content    string
	Size       string
	Filepath   string
	Repository string
	License    string
}

// Reads JSONL from STDIN, extracts "content" and compute it's MD5
// ls data/py_file_content.jsonl-* | parallel "gzcat {} | go run preprocess_content.go"
func main() {
	f, err := os.CreateTemp("data", "github_py_minus_ethpy150.*.txt")
	check(err)

	h := md5.New()
	d := json.NewDecoder(os.Stdin)
	for {
		var m Message
		if err := d.Decode(&m); err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		// MD5
		//io.WriteString(h, m.Content)

		scanner := bufio.NewScanner(strings.NewReader(m.Content))
		buf := make([]byte, 0, 64*1024)
		scanner.Buffer(buf, 2*1024*1024)
		for scanner.Scan() {
			f.Write(scanner.Bytes())
			f.Write(newlineReplacement)
		}
		if err := scanner.Err(); err != nil {
			log.Fatal(err)
		}
		f.WriteString("\n")

		//TODO(bzz): skip generated files
		//TODO(bzz): skip files \w lines too long
		//TODO(bzz): replace \n -> newlineReplacement

		// fmt.Printf("%x\n", h.Sum(nil))

		h.Reset()
	}

	err = f.Close()
	check(err)
}

func check(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

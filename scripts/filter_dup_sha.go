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

package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
)

const fieldName = `"sha": "`

// Filters STDIN leaving only JSONL \w a uniq "sha" field
func main() {
	uniq := map[string]int{}

	scanner := bufio.NewScanner(os.Stdin)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 2*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()

		s := strings.Index(line, fieldName)
		sha := line[s : s+65]

		if _, ok := uniq[sha]; ok {
			continue
		}
		uniq[sha] = 1
		fmt.Println(line)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

}

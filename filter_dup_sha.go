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

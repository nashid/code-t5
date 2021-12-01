job("Code quality") {
    container(image="python:3.9") {
        shellScript {
            interpreter = "/bin/bash"
            content = """
                pip install black mypy
                black . -l 120
                mypy . --ignore-missing-imports
            """
        }
    }
}

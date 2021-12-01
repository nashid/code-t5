job("Code quality") {
    container(image="ubuntu") {
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

job("Code quality") {
    parallel {
        container(displayName = "Run black", image="python:3.9") {
            shellScript {
                interpreter = "/bin/bash"
                content = """
                    pip install black
                    black . -l 120 --diff --color
                """
            }
        }

        container(displayName = "Run mypy", image="python:3.9") {
            shellScript {
                interpreter = "/bin/bash"
                content = """
                    pip install mypy
                    mypy . --ignore-missing-imports
                """
            }
        }

        container(displayName = "Run unit tests", image="python:3.9") {
            shellScript {
                interpreter = "/bin/bash"
                content = """
                    pip install pytest
                    pytest test
                """
            }
        }
    }
}

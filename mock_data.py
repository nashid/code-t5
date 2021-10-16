bubble_sort = """from typing import List

def bubble_sort(numbers: List[int]):
    \"\"\" Sort given array of numbers in assending order using Bubble Sort algorithm.
    >>> bubble_sort([3.0, 2.0, 1.0])
    [1.0, 2.0, 3.0]

    >>> bubble_sort([3.0, 1.0, 2.0])
    [1.0, 2.0, 3.0]
    \"\"\"
""".replace("\n", "Ċ")

has_close = """from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
""".replace("\n", "Ċ")

# Generate some predictions
inputs = [
    "def send_tweet_with_image(",
    "def validate_date(date: str):\n    \"Validates wheather a string is correctly formatted datetime string in the RFC 3339 format\"",
    "public static void ",
    "import ten ",
    "Copyright ",
    "open a file 'f.txt' in write mode ",
    bubble_sort,
    has_close,
]

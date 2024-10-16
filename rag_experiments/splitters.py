from transformers import AutoTokenizer
from typing import List

class BaseSplitter:
    def __init__(self, model_name: str):
       pass

    def __call__(self, string: str) -> List[str | int]:
        return [string]

class ModelSplitter(BaseSplitter):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"

    def __call__(self, string: str):
        return self.tokenizer.encode(string)

class LinesSplitter(BaseSplitter):
    def __call__(self, string: str):
        stoplist = ["#", '"""', "'''", '"', "'", "", "(", ")", "{", "}", "[", "]"]
        lines = string.split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        lines = [line.strip() for line in lines if line]
        lines = [line for line in lines if (line.strip("#") not in stoplist)]
        return lines
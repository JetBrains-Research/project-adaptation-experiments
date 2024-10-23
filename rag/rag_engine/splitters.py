import re

from transformers import AutoTokenizer


class BaseSplitter:
    def __init__(self, use_n_grams: bool = False, n_grams: int = -1, **kwargs):
        self.use_n_grams = use_n_grams
        self.n_grams = n_grams

    def __call__(self, string: str) -> list[str | int]:
        string_split = self.generate_split(string)

        if self.use_n_grams and self.n_grams > 1:
            return self.generate_n_grams(string_split)
        
        return string_split

    def generate_split(self, string: str) -> list[str | int]:
        if self.use_n_grams:
            return string.split()
        return [string]

    def generate_n_grams(self, tokens: list[str | int]) -> list[str]:
        return [' '.join(map(str, tokens[i:i+self.n_grams])) for i in range(len(tokens)-self.n_grams+1)]


class ModelSplitter(BaseSplitter):
    def __init__(self, model_name: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"

    def generate_split(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)


class WordSplitter(BaseSplitter):

    def generate_split(self, string: str) -> list[str]:
        tokens = re.findall(r"[\w]+", string)
        return tokens


class LinesSplitter(BaseSplitter):
    def generate_split(self, string: str) -> list[str]:
        stoplist = ["#", '"""', "'''", '"', "'", "", "(", ")", "{", "}", "[", "]"]
        lines = string.split("\n")
        lines = [line.strip() for line in lines if line.strip()]

        lines = [line for line in lines if (line.strip("#").strip() not in stoplist)]
        return lines


def get_splitter(name: str, **kwargs) -> BaseSplitter:
    if name == "model_tokenizer":
        splitter = ModelSplitter(**kwargs)
    elif name == "line_splitter":
        splitter = LinesSplitter(**kwargs)
    elif name == "word_splitter":
        splitter = WordSplitter(**kwargs)
    else:
        raise ValueError(
            f"There is no {name} splitter. Only [model_tokenizer, line_splitter, word_splitter] are available"
        )

    return splitter

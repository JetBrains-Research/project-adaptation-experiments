import re

from transformers import AutoTokenizer


class BaseSplitter:
    def __init__(
        self,
        use_n_grams: bool = False,
        n_grams_min: int = -1,
        n_grams_max: int = -1,
        **kwargs,
    ):
        self.use_n_grams = use_n_grams
        self.n_grams_min = n_grams_min
        self.n_grams_max = n_grams_max

    def __call__(self, string: str) -> list[str | int]:
        string_split = self.generate_split(string)

        if self.use_n_grams:
            return self.generate_n_grams(string_split)

        return string_split

    def generate_split(self, string: str) -> list[str | int]:
        if self.use_n_grams:
            return string.split()
        return [string]

    # def generate_n_grams(self, tokens: list[str | int]) -> list[str]:
    #     return [' '.join(map(str, tokens[i:i+self.n_grams])) for i in range(len(tokens)-self.n_grams+1)]

    def generate_n_grams(self, tokens: list[str | int]) -> list[str]:
        n_grams = []
        for n in range(self.n_grams_min, self.n_grams_max + 1):
            n_grams.extend(
                [
                    " ".join(map(str, tokens[i : i + n]))
                    for i in range(len(tokens) - n + 1)
                ]
            )
        return n_grams


class ModelSplitter(BaseSplitter):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"

    def generate_split(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)


class WordSplitter(BaseSplitter):
    def generate_split(self, string: str) -> list[str]:
        tokens = re.findall(r"[\w]+", string)
        tokens = [token for token in tokens if not token.isdigit()]
        return tokens


class LinesSplitter(BaseSplitter):
    def __call__(self, string: str) -> list[str]:
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

from transformers import AutoTokenizer


class BaseSplitter:
    def __init__(self, model_name: str | None = None):
        pass

    def __call__(self, string: str) -> list[str | int]:
        return [string]


class ModelSplitter(BaseSplitter):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"

    def __call__(self, string: str) -> list[int]:
        return self.tokenizer.encode(string)


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
    else:
        raise ValueError(
            f"There is no {name} splitter. Only [model_tokenizer, line_splitter] are available"
        )

    return splitter

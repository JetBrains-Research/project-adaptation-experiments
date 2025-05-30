from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset


@dataclass
class FileStorage:
    filename: str
    content: str

    @property
    def prompt(self) -> str:
        return f"# {self.filename}\n{self.content}\n\n"


@dataclass
class SplittedFile:
    filename: str
    prefix: str
    ground_truth: str
    postfix: str
    line_type: str | None = None

    @classmethod
    def from_completion_file(
        cls,
        completion_file: FileStorage,
        completion_line_num: int,
        line_type: str | None = None,
    ):
        completion_lines = completion_file.content.split("\n")
        prefix = "\n".join(completion_lines[:completion_line_num])
        ground_truth = completion_lines[completion_line_num]
        postfix = "\n".join(completion_lines[completion_line_num + 1 :])
        return cls(
            filename=completion_file.filename,
            prefix=prefix,
            ground_truth=ground_truth,
            postfix=postfix,
            line_type=line_type,
        )

    @property
    def prompt(self) -> str:
        return f"# {self.filename}\n{self.prefix}"


@dataclass
class RepoStorage:
    filename: list[str]
    content: list[str]
    score: list[float] | None = None

    def __post_init__(self):
        if not len(self.filename) == len(self.content):
            raise ValueError("Filenames and Content must have the same lengths")
        if self.score is None:
            self.score = list()

    def __len__(self):
        return len(self.filename)

    def __iter__(self) -> Iterator[FileStorage]:
        for filename, content in zip(self.filename, self.content):
            yield FileStorage(filename, content)

    def add_item(self, filename: str, content: str):
        self.filename.append(filename)
        self.content.append(content)

    def get_dict(self):
        repo_dict = {
            filename: content for filename, content in zip(self.filename, self.content)
        }
        return repo_dict

    def filter_by_extensions(self, allowed_extensions: list[str] | None = None) -> None:
        if allowed_extensions is not None:
            repo_dict = self.get_dict()
            self.filename = [
                file
                for file in self.filename
                if any(file.endswith(ext) for ext in allowed_extensions)
            ]
            self.content = [repo_dict[file] for file in self.filename]

    def sort_by_score(self) -> None:
        if self.score is not None:
            combined = list(zip(self.score, self.filename, self.content))
            combined.sort(reverse=False)
            self.score, self.filename, self.content = zip(*combined)
            self.filename = list(self.filename)
            self.content = list(self.content)
            self.score = list(self.score)

    def top_k(self, k: int = 10) -> None:
        self.sort_by_score()

        self.filename = self.filename[-k:]
        self.content = self.content[-k:]
        self.score = self.score[-k:]


@dataclass
class Chunk:
    filename: str
    chunk_idx: int
    max_chunks: int
    content: str
    content_token: list[str | int] | None = None
    score: float | None = None

    def __str__(self):
        num_lines = len(self.content.split("\n"))
        return (
            f"Chunk {self.chunk_idx} out of {self.max_chunks} of file {self.filename}, "
            f"Len: {len(self.content)}C/{num_lines}L, Score: {self.score:.4f}"
        )

    @property
    def prompt(self) -> str:
        return f"# CHUNK {self.filename} SCORE {self.score:.2f}\n{self.content}\n\n"


@dataclass
class ChunkedFile:
    filename: str
    content_chunks: list[str]

    def __len__(self):
        return len(self.content_chunks)

    def flatten(self) -> list[Chunk]:
        result = list()
        for chunk_idx, chunk in enumerate(self.content_chunks):
            result.append(
                Chunk(
                    filename=self.filename,
                    chunk_idx=chunk_idx,
                    max_chunks=self.__len__(),
                    content=chunk,
                )
            )
        return result


@dataclass
class ChunkedRepo:
    chunks: list[Chunk] | None = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = list()

    def append(self, chunked_file: ChunkedFile):
        if not isinstance(chunked_file, ChunkedFile):
            raise ValueError("You can append only ChunkedFile")
        self.chunks.extend(chunked_file.flatten())

    def __iter__(self) -> Iterator[Chunk]:
        for chunk in self.chunks:
            yield chunk

    def __len__(self):
        return len(self.chunks)

    def set_scores(self, scores: list[float]):
        if len(scores) != len(self.chunks):
            raise ValueError("Scores must correspond to chunks")
        for score, chunk in zip(scores, self.chunks):
            chunk.score = score

    def top_k(self, k: int = 10) -> "ChunkedRepo":
        chunks = self.chunks
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        return ChunkedRepo(sorted_chunks[:k])


def get_file_and_repo(dp) -> tuple[FileStorage, RepoStorage]:
    hf_repo_filename = dp.pop("repo_snapshot_filename", None)
    if hf_repo_filename is None:
        repo_snapshot = RepoStorage(**dp["repo_snapshot"])
    else:
        # TODO: get filenames map file and finish loading the dataset
        repo_snapshot = None
    completion_file = FileStorage(**dp["completion_file"])
    return completion_file, repo_snapshot


def chunk_py_file_content(
    file_st: FileStorage,
    chunk_lines_size: int = 32,
    overlap_lines_size: int = 8,
    filter_striped: bool = False,
) -> ChunkedFile:
    if chunk_lines_size <= overlap_lines_size:
        raise ValueError("chunk_lines_size must be greater than overlap_lines_size")
    lines = file_st.content.split("\n")
    if filter_striped:
        lines = [line for line in lines if line.strip()]
    chunks = []
    total_lines = len(lines)
    stride = chunk_lines_size - overlap_lines_size
    for i in range(total_lines, chunk_lines_size - 1, -stride):
        start_idx = max(0, i - chunk_lines_size)
        chunks.append("\n".join(lines[start_idx:i]))
    # for i in range(0, total_lines - chunk_lines_size + 1, stride):
    #     chunks.append("\n".join(lines[i : i + chunk_lines_size]))
    if total_lines % stride != 0:
        chunks.append("\n".join(lines[:chunk_lines_size]))
    chunks = chunks[::-1]

    return ChunkedFile(file_st.filename, chunks)


def chunk_repository(repo_snapshot: RepoStorage, **chunking_kwargs) -> ChunkedRepo:
    chunked_repo = ChunkedRepo()
    for file_st in repo_snapshot:
        if len(file_st.content.strip()) > 1:
            chunked_file = chunk_py_file_content(file_st, **chunking_kwargs)
            chunked_repo.append(chunked_file)
    return chunked_repo


def score_chunks(completion_file: FileStorage, chunked_repo: ChunkedRepo) -> None:
    for chunk in chunked_repo:
        chunk.score = len(chunk.content) / len(completion_file.content) + len(
            chunk.filename
        ) / len(completion_file.filename)


def main():
    ds = load_dataset(
        "JetBrains-Research/lca-project-level-code-completion",
        "medium_context",
        split="test",
    )
    completion_file, repo_snapshot = get_file_and_repo(ds[0])
    chunk_kwargs = {"chunk_lines_size": 64, "overlap_lines_size": 8}
    chunked_repo = chunk_repository(repo_snapshot, **chunk_kwargs)
    for idx, chunk in enumerate(chunked_repo):
        if idx > 10:
            print("-" * 100)
            break
        print(chunk)
    score_chunks(completion_file, chunked_repo)
    for idx, chunk in enumerate(chunked_repo):
        if idx > 10:
            print("-" * 100)
            break
        print(chunk)
    # return ds['train'][0]


if __name__ == "__main__":
    main()
    # dp = main()
    # print(dp.keys())
    # print(dp['repo_snapshot_filename'])
    # repo_snapshot = load_dataset('jenyag/plcc-python-train',
    #                              data_files=['repo_data/*/unicef__iogt__33f09c0453f2c6f08060000217aeb814420102c9.parquet'],
    #                              split='train')
    # ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', 'medium_context', split='test')
    # file, repo = get_file_and_repo(ds[0])

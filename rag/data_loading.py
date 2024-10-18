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

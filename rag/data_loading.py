from dataclasses import dataclass
from typing import Iterator

import rank_bm25
from llama_index.core.base.base_retriever import BaseRetriever

COMMENT_SEPS = {"python": "#", "java": "//", "kotlin": "//"}
LANG_EXT_MAP = {"python": "py", "kotlin": "kt", "java": "java"}

@dataclass
class FileStorage:
    filename: str
    content: str

    @property
    def prompt(self) -> str:
        return f"# {self.filename}\n{self.content}\n\n"


@dataclass
class RepoStorage:
    # TODO I'd like to keep only a dict, not separate filename list and content
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
        if self.filename.endswith(".kt"):
            sep = COMMENT_SEPS["kotlin"]
        elif self.filename.endswith(".java"):
            sep = COMMENT_SEPS["java"]
        else:
            sep = COMMENT_SEPS["python"]
        if self.score is not None:
            return f"{sep} CHUNK {self.filename} SCORE {self.score:.2f}\n{self.content}\n\n"
        else:
            return f"{sep} CHUNK {self.filename}\n{self.content}\n\n"


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
    bm25: rank_bm25.BM25Okapi | None = None
    dense_retriever: BaseRetriever | None = None

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

    def get_scores(self) -> list[float]:
        return [chunk.score for chunk in self.chunks]

    def set_scores(self, scores: list[float]):
        if len(scores) != len(self.chunks):
            raise ValueError(f"Scores must correspond to chunks. scores len = {len(scores)}, chunks = {len(self.chunks)}")
        for score, chunk in zip(scores, self.chunks):
            chunk.score = score

    def sort(self):
        self.chunks = sorted(self.chunks, key=lambda x: x.score, reverse=True)

    def top_k(self, k: int = 10) -> None:
        sorted_chunks = sorted(self.chunks, key=lambda x: x.score, reverse=True)
        if k > 0:
            self.chunks = sorted_chunks[:k]

    def deduplicate_chunks(self):
        seen: set[str] = set()
        unique_chunks: list[Chunk] = []

        for chunk in self.chunks:
            if chunk.content not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk.content)

        self.chunks = unique_chunks


def map_dp_to_dataclass(dp: dict) -> RepoStorage:
    hf_repo_filename = dp.pop("repo_snapshot_filename", None)
    if hf_repo_filename is None:
        repo_snapshot = RepoStorage(**dp["repo_snapshot"])
    else:
        # TODO: get filenames map file and finish loading the dataset
        repo_snapshot = None
    return repo_snapshot

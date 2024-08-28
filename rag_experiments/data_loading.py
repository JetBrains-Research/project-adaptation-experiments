from dataclasses import dataclass
from typing import Iterator

from datasets import load_dataset


@dataclass
class FileStorage:
    filename: str
    content: str

@dataclass
class RepoStorage:
    filename: list[str]
    content: list[str]

    def __post_init__(self):
        if not len(self.filename) == len(self.content):
            raise ValueError('Filenames and Content must have the same lengths')

    def __len__(self):
        return len(self.filename)

    def __iter__(self) -> Iterator[FileStorage]:
        for filename, content in zip(self.filename, self.content):
            yield FileStorage(filename, content)


@dataclass
class Chunk:
    filename: str
    chunk_idx: int
    max_chunks: int
    content: str
    score: float | None = None

    def __str__(self):
        num_lines = len(self.content.split("\n"))
        return (f'Chunk {self.chunk_idx} out of {self.max_chunks} of file {self.filename}, '
                f'Len: {len(self.content)}C/{num_lines}L, Score: {self.score:.4f}')


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
                Chunk(filename=self.filename, chunk_idx=chunk_idx, max_chunks=self.__len__(), content=chunk)
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
            raise ValueError('You can append only ChunkedFile')
        self.chunks.extend(chunked_file.flatten())

    def __iter__(self) -> Iterator[Chunk]:
        for chunk in self.chunks:
            yield chunk

    def __len__(self):
        return len(self.chunks)

    def set_scores(self, scores: list[float]):
        if len(scores) != len(self.chunks):
            raise ValueError('Scores must correspond to chunks')
        for score, chunk in zip(scores, self.chunks):
            chunk.score = score

    def top_k(self, k: int = 10) -> 'ChunkedRepo':
        chunks = self.chunks
        scores = [chunk.score for chunk in self.chunks]
        combined = zip(scores, chunks)
        sorted_combined = sorted(combined, reverse=True)
        sorted_chunks = [chunk for _, chunk in sorted_combined]
        return ChunkedRepo(sorted_chunks[:k])

def get_file_and_repo(dp) -> tuple[FileStorage, RepoStorage]:
    hf_repo_filename = dp.pop('repo_snapshot_filename', None)
    if hf_repo_filename is None:
        repo_snapshot = RepoStorage(**dp['repo_snapshot'])
    else:
        # TODO: get filenames map file and finish loading the dataset
        repo_snapshot = None
    completion_file = FileStorage(**dp['completion_file'])
    return completion_file, repo_snapshot


def chunk_py_file_content(file_st: FileStorage, chunk_lines_size: int = 32, overlap_lines_size: int = 8) -> ChunkedFile:
    if chunk_lines_size <= overlap_lines_size:
        raise ValueError('chunk_lines_size must be greater than overlap_lines_size')
    lines = file_st.content.split('\n')
    # TODO: filter out lines with whitespace chars (spaces, tabs)
    chunks = list()
    while len(lines) > chunk_lines_size:
        chunks.append('\n'.join(lines[:chunk_lines_size]))
        lines = lines[(chunk_lines_size - overlap_lines_size):]
    chunks.append('\n'.join(file_st.content.split('\n')[-chunk_lines_size:]))
    return ChunkedFile(file_st.filename, chunks)


def chunk_repository(repo_snapshot: RepoStorage, **chunking_kwargs):
    chunked_repo = ChunkedRepo()
    for file_st in repo_snapshot:
        if len(file_st.content.strip()) > 1:
            if file_st.filename.endswith('.py'):
                chunked_file = chunk_py_file_content(file_st, **chunking_kwargs)
                chunked_repo.append(chunked_file)
                # chunked_repo[file_st.filename] = chunks
    return chunked_repo


def score_chunks(completion_file: FileStorage, chunked_repo: ChunkedRepo) -> None:
    for chunk in chunked_repo:
        chunk.score = len(chunk.content) / len(completion_file.content) + len(chunk.filename) / len(completion_file.filename)

def main():
    ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', 'medium_context', split='test')
    completion_file, repo_snapshot = get_file_and_repo(ds[0])
    chunk_kwargs = {'chunk_lines_size': 64, 'overlap_lines_size': 8}
    chunked_repo = chunk_repository(repo_snapshot, **chunk_kwargs)
    for idx, chunk in enumerate(chunked_repo):
        if idx > 10:
            print('-' * 100)
            break
        print(chunk)
    score_chunks(completion_file, chunked_repo)
    for idx, chunk in enumerate(chunked_repo):
        if idx > 10:
            print('-' * 100)
            break
        print(chunk)
    # return ds['train'][0]


if __name__ == '__main__':
    main()
    # dp = main()
    # print(dp.keys())
    # print(dp['repo_snapshot_filename'])
    # repo_snapshot = load_dataset('jenyag/plcc-python-train',
    #                              data_files=['repo_data/*/unicef__iogt__33f09c0453f2c6f08060000217aeb814420102c9.parquet'],
    #                              split='train')
    # ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', 'medium_context', split='test')
    # file, repo = get_file_and_repo(ds[0])

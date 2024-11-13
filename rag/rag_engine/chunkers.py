from rag.data_loading import ChunkedFile, ChunkedRepo, FileStorage, RepoStorage
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


class BaseChunker:
    def __init__(self, **kwargs):
        if "score_agg" in kwargs:
            self.score_agg = kwargs["score_agg"]

    def chunk(self, file_st: FileStorage, **kwargs) -> ChunkedFile:
        return ChunkedFile(file_st.filename, [file_st.content])

    def __call__(self, repo_snapshot: RepoStorage, **chunking_kwargs) -> ChunkedRepo:
        chunked_repo = ChunkedRepo()
        for file_st in repo_snapshot:
            if len(file_st.content.strip()) > 1:
                chunked_file = self.chunk(file_st, **chunking_kwargs)
                chunked_repo.append(chunked_file)
                chunked_repo.deduplicate_chunks()
        return chunked_repo


class FixedLineChunker(BaseChunker):
    def __init__(self, chunk_lines_size: int = 32, stride: int = 8, filter_striped: bool = False, **kwargs):
        # if chunk_lines_size <= stride:
        #     raise ValueError("chunk_lines_size must be greater than overlap_lines_size")
        super().__init__(**kwargs)
        self.chunk_lines_size = chunk_lines_size
        self.stride = stride
        if self.stride is None:
            self.stride = self.chunk_lines_size
        self.filter_striped = filter_striped

    def chunk(self, file_st: FileStorage) -> ChunkedFile:
        lines = file_st.content.split("\n")
        if self.filter_striped:
            lines = [line for line in lines if line.strip()]
        chunks = []
        total_lines = len(lines)
        
        # stride = self.chunk_lines_size - self.overlap_lines_size
        for i in range(total_lines, self.chunk_lines_size - 1, -self.stride):
            start_idx = max(0, i - self.chunk_lines_size)
            chunks.append("\n".join(lines[start_idx:i]))
        if total_lines % self.stride != 0:
            chunks.append("\n".join(lines[: self.chunk_lines_size]))
        chunks = chunks[::-1]

        return ChunkedFile(file_st.filename, chunks)


class LangChainChunker(BaseChunker):
    def __init__(self, chunk_lines_size: int = 64, chunk_overlap_lines: int = 0, language: str = "python", **kwargs):
        super().__init__(**kwargs)
        chunk_symb_size = 40*chunk_lines_size
        chunk_overlap_symbols = 40*chunk_overlap_lines
        self.langchain_chunker = RecursiveCharacterTextSplitter.from_language(
            language=Language(language), chunk_size=chunk_symb_size, chunk_overlap=chunk_overlap_symbols
        )

    def chunk(self, file_st: FileStorage) -> ChunkedFile:
        chunks_docs = self.langchain_chunker.create_documents([file_st.content])
        chunks = [doc.page_content for doc in chunks_docs]
        return ChunkedFile(file_st.filename, chunks)


def get_chunker(name: str, **kwargs) -> BaseChunker:
    chunker = None
    available_instances = ["fixed_line", "full_file", "langchain"]
    if name not in available_instances:
        raise ValueError(
            f"There is no {name} chunker. Only {available_instances} are available"
        )
    if name == "full_file":
        chunker = BaseChunker(**kwargs)
    elif name == "fixed_line":
        chunker = FixedLineChunker(**kwargs)
    elif name == "langchain":
        chunker = LangChainChunker(**kwargs)
    return chunker

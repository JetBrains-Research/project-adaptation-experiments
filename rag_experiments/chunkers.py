from data_loading import ChunkedRepo, ChunkedFile, RepoStorage, FileStorage, SplittedFile

class BaseChunker:
    def chunk(self, file_st: FileStorage, *args, **kwargs) -> ChunkedFile:
        return ChunkedFile(file_st.filename, [file_st.content]) 

    def __call__(self, repo_snapshot: RepoStorage, **chunking_kwargs) -> ChunkedRepo:
        chunked_repo = ChunkedRepo()
        for file_st in repo_snapshot:
            if len(file_st.content.strip()) > 1:
                chunked_file = self.chunk(file_st, **chunking_kwargs)
                chunked_repo.append(chunked_file)
        return chunked_repo

class FixedLineChunker(BaseChunker):
    def chunk(
        self,
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
import random
from dataclasses import dataclass

from datasets import load_dataset, Dataset


@dataclass
class RepoStorage:
    filename: list[str]
    content: list[str]


@dataclass
class CompletionFile:
    filename: str
    content: str


@dataclass
class RawDatapoint:
    completion_file: CompletionFile
    repo_snapshot: RepoStorage
    completion_lines: dict[str, list[int]]


@dataclass
class SplittedFile:
    prefix: str
    ground_truth: str
    postfix: str


@dataclass
class RePlugExample:
    prefix: str
    splitted_file: SplittedFile
    line_category: str

    @property
    def ground_truth(self):
        return self.splitted_file.ground_truth

    @property
    def file_prefix(self):
        return self.splitted_file.prefix

    @property
    def file_postfix(self):
        return self.splitted_file.postfix


PATTERN = '# {fn_repo}\n\n{cont_repo}\n\n# {fn_compl}\n\n{file_prefix}'


config_names = [
    'small_context',
    'medium_context',
    'large_context',
    'huge_context'
]

config_name = config_names[1]


def sample_repo_snapshot(full_repo: RepoStorage, sample_size: int) -> RepoStorage:
    seed = sum([len(fn) for fn in full_repo.filename])
    rnd = random.Random(seed)
    if len(full_repo.content) <= sample_size:
        repo_random_idx = list(range(len(full_repo.content)))
    else:
        repo_random_idx = rnd.sample(range(len(full_repo.content)), sample_size)
    repo_random_contents = [full_repo.content[i] for i in repo_random_idx]
    repo_random_filenames = [full_repo.filename[i] for i in repo_random_idx]
    return RepoStorage(repo_random_filenames, repo_random_contents)


def get_raw_datapoint(ds: Dataset, idx: int, sample_size: int = 100) -> RawDatapoint:
    completion_file = CompletionFile(**ds[idx]['completion_file'])
    repo_snapshot = RepoStorage(**ds[idx]['repo_snapshot'])
    sampled_repo_snapshot = sample_repo_snapshot(repo_snapshot, sample_size)
    return RawDatapoint(
        completion_file=completion_file,
        repo_snapshot=sampled_repo_snapshot,
        completion_lines=ds[idx]['completion_lines'],
    )


def split_by_line(file_content: str, line_idx: int) -> SplittedFile:
    lines = file_content.split('\n')
    return SplittedFile(
        prefix='\n'.join(lines[:line_idx]),
        ground_truth=lines[line_idx],
        postfix='\n'.join(lines[(line_idx+1):])
    )


def get_examples_from_raw_datapoint(raw_datapoint: RawDatapoint) -> list[list[RePlugExample]]:
    # completion_file, repo_snapshot, completion_lines = raw_datapoint
    example_batches = list()
    for line_cat, line_idxs in raw_datapoint.completion_lines.items():
        for line_idx in line_idxs:
            example_contexts = list()
            splitted_file = split_by_line(raw_datapoint.completion_file.content, line_idx)
            for filename, content in zip(raw_datapoint.repo_snapshot.filename, raw_datapoint.repo_snapshot.content):
                example_contexts.append(
                    RePlugExample(
                        prefix=PATTERN.format(
                                    fn_repo=filename,
                                    cont_repo=content,
                                    fn_compl=raw_datapoint.completion_file.filename,
                                    file_prefix=splitted_file.prefix),
                        splitted_file=splitted_file,
                        line_category=line_cat,
                    )
                )
            example_batches.append(example_contexts)
    return example_batches


if __name__ == '__main__':
    ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', config_name, split='test')
    raw_dp = get_raw_datapoint(ds, 0, sample_size=3)
    example_batches = get_examples_from_raw_datapoint(raw_dp)
    # print(examples)
    # print(raw_dp)
    for example_batch in example_batches:
        for example in example_batch:
            print('='*100)
            print(example.prefix[:1_000] + '\n\n...\n\n', example.prefix[-1_000:])
            print('-'*100)
            print(example.ground_truth)
        break

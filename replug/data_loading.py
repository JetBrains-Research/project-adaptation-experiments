import os
import random
from dataclasses import dataclass

from datasets import load_dataset, Dataset


@dataclass
class RepoStorage:
    filename: list[str]
    content: list[str]


@dataclass
class FileStorage:
    filename: str
    content: str


@dataclass
class RawDatapoint:
    completion_file: FileStorage
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
    context_file: FileStorage
    completion_file: FileStorage
    context_weight: float = 1.0

    @property
    def ground_truth(self):
        return self.splitted_file.ground_truth

    @property
    def file_prefix(self):
        return self.splitted_file.prefix

    @property
    def file_postfix(self):
        return self.splitted_file.postfix


def path_distance(base_path, target_path):
    base_components = os.path.normpath(base_path).split(os.sep)
    target_components = os.path.normpath(target_path).split(os.sep)

    common_length = len(os.path.commonprefix([base_components, target_components]))

    distance = (len(base_components) - common_length) + (len(target_components) - common_length)
    return distance - 1


@dataclass
class RePlugInstance:
    examples: list[RePlugExample]

    def __len__(self):
        return len(self.examples)
    
    def __iter__(self):
        return iter(self.examples)

    @property
    def context_weights(self):
        return [example.context_weight for example in self.examples]
    
    def define_context_weights(self, context_weights: list[float] | None = None):
        if context_weights is None:
            context_weights = [1.0 / len(self)] * len(self)
        for example, weight in zip(self.examples, context_weights):
            example.context_weight = weight

    def get_top_k_contexts(self, k: int) -> "RePlugInstance":
        return RePlugInstance(sorted(self.examples, key=lambda x: x.context_weight, reverse=True)[:k])

    def calculate_path_distances_weights(self):
        for example in self.examples:
            pd = path_distance(example.context_file.filename, example.completion_file.filename)
            example.context_weight = 1 / pd

    def normalize_context_weights(self):
        weight_sum = sum([example.context_weight for example in self.examples])
        for example in self.examples:
            example.context_weight /= weight_sum



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
    completion_file = FileStorage(**ds[idx]['completion_file'])
    repo_snapshot = RepoStorage(**ds[idx]['repo_snapshot'])
    sampled_repo_snapshot = sample_repo_snapshot(repo_snapshot, sample_size)
    return RawDatapoint(
        completion_file=completion_file,
        repo_snapshot=sampled_repo_snapshot,
        completion_lines=ds[idx]['completion_lines'],
    )


def filter_by_extension(repo_storage: RepoStorage, extension: str) -> RepoStorage:
    filtered_file_names = []
    filtered_contents = []
    for filename, content in zip(repo_storage.filename, repo_storage.content):
        if filename.endswith(extension):
            filtered_file_names.append(filename)
            filtered_contents.append(content)
    return RepoStorage(filtered_file_names, filtered_contents)


def get_all_raw_data_points(ds: Dataset, filter_by_extension: str | None = '.py') -> list[RawDatapoint]:
    data_points = []
    for s in ds:
        completion_file = FileStorage(**s['completion_file'])
        repo_snapshot = RepoStorage(**s['repo_snapshot'])
        repo_snapshot = filter_by_extension(repo_snapshot, filter_by_extension)
        data_point = RawDatapoint(
            completion_file=completion_file,
            repo_snapshot=repo_snapshot,
            completion_lines=s['completion_lines'],
        )
        data_points.append(data_point)
    return data_points


def split_by_line(file_content: str, line_idx: int) -> SplittedFile:
    lines = file_content.split('\n')
    return SplittedFile(
        prefix='\n'.join(lines[:line_idx]),
        ground_truth=lines[line_idx],
        postfix='\n'.join(lines[(line_idx+1):])
    )


def get_examples_from_raw_datapoint(raw_datapoint: RawDatapoint,
                                    line_cat_to_get: str | None = 'inproject') -> list[RePlugInstance]:
    # completion_file, repo_snapshot, completion_lines = raw_datapoint
    example_batches = list()
    for line_cat, line_idxs in raw_datapoint.completion_lines.items():
        if line_cat_to_get is not None and line_cat != line_cat_to_get:
            continue
        for line_idx in line_idxs:
            example_contexts = list()
            splitted_file = split_by_line(raw_datapoint.completion_file.content, line_idx)
            for filename, content in zip(raw_datapoint.repo_snapshot.filename,
                                         raw_datapoint.repo_snapshot.content):
                example_contexts.append(
                    RePlugExample(
                        prefix=PATTERN.format(
                                    fn_repo=filename,
                                    cont_repo=content,
                                    fn_compl=raw_datapoint.completion_file.filename,
                                    file_prefix=splitted_file.prefix),
                        splitted_file=splitted_file,
                        line_category=line_cat,
                        context_file=FileStorage(filename, content),
                        completion_file=FileStorage(raw_datapoint.completion_file.filename, splitted_file.prefix)
                    )
                )
            example_batches.append(RePlugInstance(example_contexts))
        
    return example_batches


if __name__ == '__main__':
    ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', config_name, split='test')
    raw_dp = get_raw_datapoint(ds, 0, sample_size=100)
    example_batches = get_examples_from_raw_datapoint(raw_dp)
    # print(examples)
    # print(raw_dp)
    for example_batch in example_batches:
        example_batch.calculate_path_distances_weights()
        example_batch = example_batch.get_top_k_contexts(3)
        print(example_batch.examples[0].completion_file.filename)
        for example in example_batch:
            print(example.context_weight)
            print(example.context_file.filename)
        break
        # for example in example_batch:
        #     print('='*100)
        #     print(example.prefix[:1_000] + '\n\n...\n\n', example.prefix[-1_000:])
        #     print('-'*100)
        #     print(example.ground_truth)
        # break

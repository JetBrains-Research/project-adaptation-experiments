import os
import random
from copy import deepcopy
from dataclasses import dataclass, asdict

from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModel,
                          PreTrainedTokenizer, PreTrainedModel)
import torch
from torch import Tensor


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
    file_level_example: RePlugExample | None = None

    def __len__(self):
        return len(self.examples)
    
    def __iter__(self):
        # the first output is an example with file-level context
        if self.file_level_example is not None:
            yield False, self.file_level_example
        for ex in self.examples:
            yield True, ex

    @property
    def context_weights(self):
        return [example.context_weight for example in self.examples]

    @property
    def line_category(self):
        return self.examples[0].line_category

    @property
    def ground_truth(self):
        return self.examples[0].ground_truth

    def get_file_level_example(self, weight: float = 0.) -> RePlugExample:
        file_level_example = deepcopy(self.examples[0])
        file_level_example.prefix = f'# {self.examples[0].completion_file.filename}\n{self.examples[0].file_prefix}'
        file_level_example.context_file = FileStorage('', '')
        file_level_example.context_weight = weight
        return file_level_example

    def get_composer_example(self, weight: float = 1.) -> RePlugExample:
        composer_example = deepcopy(self.examples[0])
        composer_pattern = '# {fn_repo}\n\n{cont_repo}\n\n#'
        sorted_examples = sorted(self.examples, key=lambda x: x.context_weight)
        composer_example.prefix = ''
        for example in sorted_examples:
            composer_example.prefix += composer_pattern.format(
                fn_repo=example.context_file.filename,
                cont_repo=example.context_file.content
            )
        composer_example.prefix += f'# {self.examples[0].completion_file.filename}\n{self.examples[0].file_prefix}'
        composer_example.context_file = FileStorage('PATH DISTANCE COMPOSER', 'PATH DISTANCE COMPOSER')
        composer_example.context_weight = weight
        return composer_example

    def write_file_level(self):
        self.file_level_example = self.get_file_level_example()
    
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

    def calculate_embedding_weights(self,
                                    model: PreTrainedModel,
                                    tokenizer: PreTrainedTokenizer,
                                    ) -> None:
        # bigcode/starencoder - https://huggingface.co/bigcode/starencoder
        # codesage/codesage-large - https://huggingface.co/codesage/codesage-large
        # thenlper/gte-large - https://huggingface.co/thenlper/gte-large

        if model.config._name_or_path == 'bigcode/starencoder':
            raise NotImplementedError
        elif model.config._name_or_path == 'codesage/codesage-large':
            raise NotImplementedError
        elif model.config._name_or_path == 'thenlper/gte-large':
            input_texts = [example.context_file.content for example in self.examples]
            input_texts = [self.examples[0].file_prefix] + input_texts

            batch_dict = tokenizer(input_texts, max_length=512, padding=True,
                                   truncation=True, return_tensors='pt').to(model.device)
            outputs = model(**batch_dict)
            embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            comp_file_emb  = embeddings[0:1]
            context_file_embs = embeddings[1:]
            weights = torch.cosine_similarity(comp_file_emb, context_file_embs) + 1
            for example, weight in zip(self.examples, weights):
                example.context_weight = weight.item()
        return weights

    def normalize_context_weights(self):
        weight_sum = sum([example.context_weight for example in self.examples])
        for example in self.examples:
            example.context_weight /= weight_sum

    def _average_pool(self,
                      last_hidden_states: Tensor,
                      attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def softmax_context_weights(self, temperature: float = 0.025):
        ws = torch.tensor(self.context_weights)
        ws = ws / temperature
        ws = torch.softmax(ws, 0)
        for example, weight in zip(self.examples, ws):
            example.context_weight = weight.item()
        return ws




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


def get_all_raw_data_points(ds: Dataset, extension_to_get: str | None = '.py') -> list[RawDatapoint]:
    data_points = []
    for s in ds:
        completion_file = FileStorage(**s['completion_file'])
        repo_snapshot = RepoStorage(**s['repo_snapshot'])
        repo_snapshot = filter_by_extension(repo_snapshot, extension_to_get)
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
    device = torch.device('cuda:1')
    model = AutoModel.from_pretrained('thenlper/gte-large').to(device)
    tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-large')
    for example_batch in example_batches:
        example_batch.calculate_embedding_weights(model, tokenizer)
        example_batch = example_batch.get_top_k_contexts(8)
        print(example_batch.context_weights)
        ws = torch.tensor(example_batch.context_weights)
        ws = ws / 0.025
        print(sorted(torch.softmax(ws, 0)))
        break
        # example_batch.calculate_path_distances_weights()
        # example_batch.get_composer_example()
        # example_batch = example_batch.get_top_k_contexts(3)
        # print(example_batch.examples[0].completion_file.filename)
        # for example in example_batch:
        #     print(example.context_weight)
        #     print(example.context_file.filename)
        # break
        # for example in example_batch:
        #     print('='*100)
        #     print(example.prefix[:1_000] + '\n\n...\n\n', example.prefix[-1_000:])
        #     print('-'*100)
        #     print(example.ground_truth)
        # break

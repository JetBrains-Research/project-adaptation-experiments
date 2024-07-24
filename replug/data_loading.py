import random

from datasets import load_dataset


PATTERN = '# {fn_repo}\n\n{cont_repo}\n\n# {fn_compl}\n\n{file_prefix}'

config_names = [
    'small_context',
    'medium_context',
    'large_context',
    'huge_context'
]

config_name = config_names[1]



# print(ds[0].keys())

def get_raw_datapoint(ds, idx, sample_size=100):
    completion_file = ds[idx]['completion_file']
    repo_snapshot = ds[idx]['repo_snapshot']
    rnd = random.Random(len(completion_file['content']))
    repo_random_idx = rnd.sample(range(len(repo_snapshot['content'])), sample_size)
    repo_random_contents = [repo_snapshot['content'][i] for i in repo_random_idx]
    repo_random_filenames = [repo_snapshot['filename'][i] for i in repo_random_idx]

    return (completion_file, {'filename': repo_random_filenames, 'content': repo_random_contents},
            ds[idx]['completion_lines'])


def split_by_line(file_content: str, line_idx: int):
    lines = file_content.split('\n')
    return '\n'.join(lines[:line_idx]), lines[line_idx], '\n'.join(lines[(line_idx+1):])


def get_examples_from_raw_datapoint(raw_datapoint):
    completion_file, repo_snapshot, completion_lines = raw_datapoint
    examples = list()
    for line_cat, line_idxs in completion_lines.items():
        for line_idx in line_idxs:
            file_prefix, ground_truth, file_postfix = split_by_line(completion_file['content'], line_idx)
            for filename, content in zip(repo_snapshot['filename'], repo_snapshot['content']):
                examples.append(
                    {
                        'prefix': PATTERN.format(
                                    fn_repo=filename,
                                    cont_repo=content,
                                    fn_compl=completion_file['filename'],
                                    file_prefix=file_prefix,
                                ),
                        'ground_truth': ground_truth,
                        'postfix': file_postfix,
                    }
                )
    return examples


if __name__ == '__main__':
    ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', config_name, split='test')
    raw_dp = get_raw_datapoint(ds, 0, sample_size=3)
    examples = get_examples_from_raw_datapoint(raw_dp)
    # print(examples)
    # print(raw_dp)
    for example in examples:
        print('='*100)
        print(example['prefix'][:1_000] + '\n\n...\n\n', example['prefix'][-1_000:])
        print('-'*100)
        print(example['ground_truth'])

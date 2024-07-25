from datasets import load_dataset
import torch
from tqdm import tqdm

from data_loading import get_examples_from_raw_datapoint, get_all_raw_data_points
from model import RePlugModel


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = 'deepseek-ai/deepseek-coder-1.3b-base'
model = RePlugModel(model_name, device)

ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', 'medium_context', split='test')

all_raw_data_points = get_all_raw_data_points(ds, extension_to_get='.py')

instances = []

for raw_dp in tqdm(all_raw_data_points, desc='Processing raw data points'):
    curr_instances = get_examples_from_raw_datapoint(raw_dp, line_cat_to_get='inproject')
    instances.extend(curr_instances)

print(f'Got {len(instances)} instances')

for instance in instances:
    print('=' * 100)
    print('Line cat: ', instance.line_category)
    generated = model.generate(instance, max_new_tokens=25)
    print('---- Generated ----\n', generated)
    print('--- Ground truth ----\n', instance.ground_truth)
    print('-' * 100)





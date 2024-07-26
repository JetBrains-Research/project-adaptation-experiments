import click
from datasets import load_dataset
import torch
from tqdm import tqdm
import wandb

from data_loading import get_examples_from_raw_datapoint, get_all_raw_data_points
from model import RePlugModel


def init_wandb(model_name: str,
               top_k: int,
               max_new_tokens: int,
               prob_similarity_weights: bool,
               top_k_selection: str):
    wandb.init(project='fast-iterations-replug')
    wandb.config.update({
        'model_name': model_name,
        'top_k': top_k,
        'max_new_tokens': max_new_tokens,
        'prob_similarity_weights': prob_similarity_weights,
        'top_k_selection': top_k_selection
    })


@click.command()
@click.option('--print-generated', type=bool, default=False)
@click.option('--top-k', type=int, default=3)
@click.option('--max-new-tokens', type=int, default=128)
@click.option('--device-num', type=int, default=0)
@click.option('--prob-similarity-weights', type=bool, default=False)
@click.option('--top-k-selection', type=str, default='path_distances')
@click.option('--model-name', type=str, default='deepseek-ai/deepseek-coder-1.3b-base')
def main(print_generated: bool = False,
         top_k: int = 3,
         max_new_tokens: int = 25,
         device_num: int = 0,
         prob_similarity_weights: bool = False,
         top_k_selection: str = 'path_distances',
         model_name: str = 'deepseek-ai/deepseek-coder-1.3b-base'):
    init_wandb(model_name, top_k, max_new_tokens,
               prob_similarity_weights, top_k_selection)
    device = f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu'
    model = RePlugModel(model_name, device)

    ds = load_dataset('JetBrains-Research/lca-project-level-code-completion',
                    'medium_context',
                    split='test')

    all_raw_data_points = get_all_raw_data_points(ds, extension_to_get='.py')

    instances = []

    for raw_dp in tqdm(all_raw_data_points, desc='Processing raw data points'):
        curr_instances = get_examples_from_raw_datapoint(raw_dp, line_cat_to_get='inproject')
        instances.extend(curr_instances)

    print(f'Total {len(instances)} instances')

    pbar = tqdm(instances, desc='Progress')
    total = 0
    correct = 0
    for instance in pbar:
        if top_k_selection == 'path_distances':
            instance.calculate_path_distances_weights()
        else:
            raise NotImplementedError(f'Top k selection {top_k_selection} not implemented')
        instance = instance.get_top_k_contexts(top_k)
        generated = model.generate(instance,
                                   max_new_tokens=max_new_tokens,
                                   prob_similarity_weights=prob_similarity_weights)
        total += 1
        generated = generated.strip()
        assert not '\n' in generated, f'Generated contains multiple lines: {repr(generated)}'
        ground_truth = instance.ground_truth.strip()
        assert not '\n' in ground_truth, f'Ground truth contains multiple lines: {repr(ground_truth)}'
        if generated == ground_truth:
            correct += 1
        pbar.set_description(f'EM: {correct / total:.1%}')

        wandb.log({
            'EM': correct / total,
            'total': total,
            'correct': correct
        })

        if print_generated:
            print('=' * 100)
            print('Line cat: ', instance.line_category)
            print('---- Generated ----\n', generated)
            print('--- Ground truth ----\n', instance.ground_truth)
            print('-' * 100)


if __name__ == '__main__':
    main()

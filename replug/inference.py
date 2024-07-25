from datasets import load_dataset
from data_loading import get_raw_datapoint, get_examples_from_raw_datapoint
from model import RePlugModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = 'deepseek-ai/deepseek-coder-1.3b-base'
base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  torch_dtype=torch.bfloat16,
                                                  attn_implementation='flash_attention_2').to(device)
model = RePlugModel(base_model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds = load_dataset('JetBrains-Research/lca-project-level-code-completion', 'medium_context', split='test')
raw_dp = get_raw_datapoint(ds, 54, sample_size=25)
all_examples = get_examples_from_raw_datapoint(raw_dp)
for examples in all_examples:
    print(examples[0]['line_cat'])
    if examples[0]['line_cat'] == 'inproject':
        print('Found inproject example')
        break
else:
    print('No inproject example found')
    raise


batch = []
for example in examples:
    input_ids = tokenizer(example['prefix'], return_tensors='pt', truncation=True, max_length=16000)['input_ids'].to(device)
    print(input_ids.shape)
    batch.append(input_ids)
gt = example['ground_truth']

with torch.inference_mode():
    outputs = model.generate(batch, max_new_tokens=25)[0][:, -25:]

file_prefix_input_ids = tokenizer(example['file_prefix'], return_tensors='pt')['input_ids'].to(device)

only_prefix_ouput = base_model.generate(file_prefix_input_ids, max_new_tokens=25)[:, -25:]
only_prefix_text = tokenizer.batch_decode(only_prefix_ouput, skip_special_tokens=True)

generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print('Line cat: ', example['line_cat'])
print('Only prefix: ')
print(only_prefix_text[0])

for g in generated:
    print('=' * 100)
    print(g)
print('-' * 100)
print(gt)



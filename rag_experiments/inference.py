from dataclasses import dataclass
from typing import Optional

import jsonlines
import torch
from vllm import LLM, RequestOutput, SamplingParams


@dataclass
class VllmEngineOutput:
    output: Optional[RequestOutput] = None

    def get_out_text(self) -> str:
        return self.output.outputs[0].text

    def get_out_extras(self) -> dict:
        return {'engine': 'vllm', 'cumulative_logprob': self.output.outputs[0].cumulative_logprob}

hf_model_path = 'deepseek-ai/deepseek-coder-1.3b-base'

llm = LLM(model=hf_model_path, max_model_len=16_000, dtype=torch.bfloat16,)

sampling_params = SamplingParams(**{
        'temperature': 0.0,
        'min_tokens': 10,
        'max_tokens': 70,
        'truncate_prompt_tokens': 15_930,
    })


def generate(prompts: list[str], llm: LLM, **generation_args) -> list[VllmEngineOutput]:
    outputs = [VllmEngineOutput(output=output) for output in llm.generate(prompts=prompts, **generation_args)]
    return outputs


def get_EM(gts, preds, line_type):
    assert len(gts) == len(preds)
    em_infile = 0
    em_inproject = 0
    total_infile = 0
    total_inproject = 0
    for gt, pred, lt in zip(gts, preds, line_type):
        pred_line = pred.strip().split('\n')[0]
        if len(pred) > 1:
            if lt == 'infile':
                total_infile += 1
            elif lt == 'inproject':
                total_inproject += 1
        if gt.strip() == pred_line.strip():
            if lt == 'infile':
                em_infile += 1
            elif lt == 'inproject':
                em_inproject += 1
    em_infile = em_infile / total_infile
    em_inproject = em_inproject / total_inproject

    return em_infile, em_inproject, total_infile, total_inproject

line_type = list()
gts = list()
file_level_preds = list()
file_level_prompts = list()
kl_3_preds = list()
kl_3_prompts = list()
kl_10_preds = list()
kl_10_prompts = list()


with jsonlines.open('/home/glukhov/project-adaptation-experiments/data/file_level/prompts.jsonl', 'r') as reader:
    for data in reader:
        file_level_prompts.append(data['prompt'])
        line_type.append(data['line_type'])
        gts.append(data['ground_truth'])

with jsonlines.open('/home/glukhov/project-adaptation-experiments/data/kl_rag/top_3/prompts.jsonl', 'r') as reader:
    for data in reader:
        kl_3_prompts.append(data['prompt'])

with jsonlines.open('/home/glukhov/project-adaptation-experiments/data/kl_rag/top_10/prompts.jsonl', 'r') as reader:
    for data in reader:
        kl_10_prompts.append(data['prompt'])

print(len(file_level_prompts), len(line_type), len(set(line_type)), len(gts), len(kl_3_preds))


file_level_outputs = list()
kl_3_outputs = list()
kl_10_outputs = list()

prompts_chunk_size = 256

for chunk_num in range(len(file_level_prompts)):
    print(chunk_num, prompts_chunk_size, len(file_level_prompts))
    if chunk_num * prompts_chunk_size > len(file_level_prompts):
        print(chunk_num, prompts_chunk_size, len(file_level_prompts))
        break

    _fl = file_level_prompts[chunk_num * prompts_chunk_size: (chunk_num + 1) * prompts_chunk_size]
    _kl3 = kl_3_prompts[chunk_num * prompts_chunk_size: (chunk_num + 1) * prompts_chunk_size]
    _kl10 = kl_10_prompts[chunk_num * prompts_chunk_size: (chunk_num + 1) * prompts_chunk_size]

    file_level_outputs.extend(generate(_fl, llm, sampling_params=sampling_params))
    kl_3_outputs.extend(generate(_kl3, llm, sampling_params=sampling_params))
    kl_10_outputs.extend(generate(_kl10, llm, sampling_params=sampling_params))

    file_level_preds = [out.get_out_text() for out in file_level_outputs]
    kl_3_preds = [out.get_out_text() for out in kl_3_outputs]
    kl_10_preds = [out.get_out_text() for out in kl_10_outputs]
    # print(file_level_preds[-1].strip().split[0])
    # print('-'*100)
    # print(kl_3_preds[-1].)
    # print('-' * 100)
    # print(kl_10_preds[-1])
    # print('-' * 100)
    # print(gts[-1])
    # print('-' * 100)
    new_gts = gts[: (chunk_num + 1) * prompts_chunk_size]
    new_line_type = line_type[: (chunk_num + 1) * prompts_chunk_size]
    print(f'File Level EM : {get_EM(new_gts, file_level_preds, new_line_type)}')
    print(f'KL 3 EM : {get_EM(new_gts, kl_3_preds, new_line_type)}')
    print(f'KL 10 EM : {get_EM(new_gts, kl_10_preds, new_line_type)}')

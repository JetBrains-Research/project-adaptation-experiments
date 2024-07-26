import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data_loading import RePlugInstance, RePlugExample


class RePlugModel(nn.Module):
    def __init__(self, model_name: str, device: str | torch.DeviceObjType):
        super(RePlugModel, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                               torch_dtype=torch.bfloat16,
                                                               attn_implementation='flash_attention_2',
                                                               device_map=device)
        self.base_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device(device)

    @torch.inference_mode()
    def generate(self,
                 input_instance: RePlugInstance,
                 max_new_tokens: int = 10,
                 max_length: int = 16000,
                 verbose: bool = False) -> str:
        past_kvs = [None] * len(input_instance)
        current_input_ids = []

        for sample in input_instance:
            inputs = self.tokenizer(sample.prefix, return_tensors='pt',
                                     max_length=max_length)
            input_ids = inputs['input_ids'].to(self.device)
            current_input_ids.append(input_ids)

        generated = []
        for _ in tqdm(range(max_new_tokens), disable=not verbose):
            logits_list = []
            new_kvs = []
            for input_ids, cur_kv in zip(current_input_ids, past_kvs):
                out = self.base_model(input_ids, past_key_values=cur_kv)
                logits = out.logits[:, -1:]
                logits_list.append(logits)
                new_kvs.append(out.past_key_values)
            aggr_out = self._aggregate_logits(logits_list, input_instance.context_weights)
            new_token = torch.argmax(aggr_out, dim=-1, keepdim=True)
            generated.append(new_token.item())
            current_input_ids = [new_token] * len(input_instance)
            past_kvs = new_kvs
            if self.stopping_criterion(generated):
                break
        generated_text = self.tokenizer.decode(generated)
        return generated_text

    def stopping_criterion(self, generated_tokens: list[int]) -> bool:
        # skip intro new lines
        non_word_tokens_prefix = 0
        for token in generated_tokens:
            if '\n' not in self.tokenizer.decode([token]):
                break
            non_word_tokens_prefix += 1
        return '\n' in self.tokenizer.decode(generated_tokens[non_word_tokens_prefix:])

    def _aggregate_logits(self, logits_list: list[torch.Tensor], context_weights: list[float]):
        norm_logits_list = [logits * w for logits, w in zip(logits_list, context_weights)]
        return torch.cat(norm_logits_list, dim=1).sum(dim=1)


if __name__ == '__main__':
    rand_weights = False
    if rand_weights:
        context_variants = [('Remember Cambodia', torch.rand(1).item()),
                            ('Remember Mongolia', torch.rand(1).item()), 
                            ('Remember Vanuatu', torch.rand(1).item()),
                            ('Remember Chad', torch.rand(1).item()), 
                            ('Remember Cuba', torch.rand(1).item())]
    else:
        context_variants = [('Remember Cambodia', 0.1),
                            ('Remember Mongolia', 0.1), 
                            ('Remember Vanuatu', 0.9),
                            ('Remember Chad', 0.1), 
                            ('Remember Cuba', 0.1)]
    file_prefix = '\n\nSo I remember the following five countries: '
    examples = [RePlugExample(cv + file_prefix, *([None] * 4), w) for cv, w in context_variants]
    device = 'cuda:2'
    model_inputs = RePlugInstance(examples)
    model = RePlugModel('gpt2', device)
    output = model.generate(model_inputs, max_new_tokens=25)
    print(output)
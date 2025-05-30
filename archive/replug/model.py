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
        self.tokenizer.truncation_side = 'left'
        self.device = torch.device(device)

    @torch.inference_mode()
    def generate(self,
                 input_instance: RePlugInstance,
                 max_new_tokens: int = 10,
                 max_length: int = 16000,
                 verbose: bool = False,
                 prob_similarity_weights: bool = False,
                 ) -> (str, int):
        past_kvs = [None] * (len(input_instance) + int(prob_similarity_weights))
        current_input_ids = []
        number_of_context_tokens: int = 0

        if prob_similarity_weights:
            input_instance.write_file_level()  # Save file-level context example

        file_prefix_num_tokens = self.tokenizer(input_instance.examples[0].file_prefix, return_tensors='pt',
                                     max_length=max_length)['input_ids'].shape[-1]

        for is_project_ctxt, sample in input_instance:
            inputs = self.tokenizer(sample.prefix, return_tensors='pt',
                                     max_length=max_length)
            input_ids = inputs['input_ids'].to(self.device)
            # print(input_ids.shape)
            number_of_context_tokens += (input_ids.shape[-1] - file_prefix_num_tokens) * is_project_ctxt
            current_input_ids.append((is_project_ctxt, input_ids))

        generated = []
        for _ in tqdm(range(max_new_tokens), disable=not verbose):
            logits_list = []
            file_level_logits = None
            for n, (is_project_ctxt, input_ids) in enumerate(current_input_ids):
                out = self.base_model(input_ids, past_key_values=past_kvs[n])
                logits = out.logits[:, -1:]
                logits = torch.log_softmax(logits, dim=2)
                if is_project_ctxt:
                    logits_list.append(logits)
                elif prob_similarity_weights:
                    file_level_logits = logits
                past_kvs[n] = out.past_key_values
            # print(len(current_input_ids))
            if prob_similarity_weights:
                if file_level_logits is None:
                    raise ValueError('Something went wrong: file_level_logits is None')
                new_context_weights = self._recalculate_context_weights(file_level_logits, logits_list)
                input_instance.define_context_weights(new_context_weights)
                # TODO: weight normalization
            aggr_out = self._aggregate_logits(logits_list, input_instance.context_weights)
            new_token = torch.argmax(aggr_out, dim=-1, keepdim=True)
            generated.append(new_token.item())
            if prob_similarity_weights:
                current_input_ids = [(False, new_token)] + [(True, new_token)] * len(input_instance)
            else:
                current_input_ids = [(True, new_token)] * len(input_instance)
            if self.stopping_criterion(generated):
                break
        generated_text = self.tokenizer.decode(generated)
        return generated_text, number_of_context_tokens

    def stopping_criterion(self, generated_tokens: list[int]) -> bool:
        # skip intro new lines
        # if len(generated_tokens) < 5:
        #     return False
        new_line_prefix = 0
        for token in generated_tokens:
            if '\n' not in self.tokenizer.decode([token]):
                break
            new_line_prefix += 1
        return '\n' in self.tokenizer.decode(generated_tokens[new_line_prefix:])

    def _aggregate_logits(self, logits_list: list[torch.Tensor], context_weights: list[float]):
        norm_logits_list = [logits * w for logits, w in zip(logits_list, context_weights)]
        return torch.cat(norm_logits_list, dim=1).sum(dim=1)

    def _recalculate_context_weights(self, file_level_logits: torch.Tensor, logits_list: list[torch.Tensor]) -> list[float]:
        context_weights = list()
        for logits in logits_list:
            kl = torch.nn.functional.kl_div(logits.ravel(), file_level_logits.ravel(), log_target=True)
            context_weights.append(kl.item())
        # print(context_weights)
        return context_weights


if __name__ == '__main__':
    rand_weights = False
    if rand_weights:
        context_variants = [('Remember Cambodia', torch.rand(1).item()),
                            ('Remember Mongolia', torch.rand(1).item()), 
                            ('Remember Vanuatu', torch.rand(1).item()),
                            ('Remember Chad', torch.rand(1).item()), 
                            ('Remember Cuba', torch.rand(1).item())]
    else:
        context_variants = [('Remember Cambodia', 0.4),
                            ('Remember Mongolia', 0.5), 
                            ('Remember Thailand', 0.1),
                            ('Remember Chad', 0.1), 
                            ('Remember Cuba', 0.1)]
    file_prefix = '\n\nSo I remember the following five countries: '
    examples = [RePlugExample(cv + file_prefix, *([None] * 4), w) for cv, w in context_variants]
    device = 'cuda:2'
    model_inputs = RePlugInstance(examples)
    model = RePlugModel('mistralai/Mistral-7B-Instruct-v0.2', device)
    output = model.generate(model_inputs, max_new_tokens=25, prob_similarity_weights=False)
    print(output)

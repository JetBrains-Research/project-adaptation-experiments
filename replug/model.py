import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class RePlugModel(nn.Module):
    def __init__(self, base_model):
        super(RePlugModel, self).__init__()
        self.base_model = base_model

    def generate_one_token(self, list_input_ids: list[torch.LongTensor]):
        out_list = list()
        for input_ids in list_input_ids:
            out = self.base_model(input_ids)
            out_list.append(out.logits[:, -1:, ...])
        aggr_out = self._aggregate_outs(out_list)
        new_token = torch.argmax(aggr_out, dim=-1, keepdim=True)
        new_list_input_ids = [torch.cat([input_ids, new_token], dim=-1) for input_ids in list_input_ids]
        return new_list_input_ids

    def generate(self, list_input_ids: list[torch.LongTensor], max_new_tokens: int = 10):
        for _ in tqdm(range(max_new_tokens)):
            list_input_ids = self.generate_one_token(list_input_ids)
        return list_input_ids

    def _aggregate_outs(self, out_list):
        # out_list = []
        # norm_out_list = [out.softmax(dim=-1) for out in out_list]
        norm_out_list = out_list

        return torch.cat(norm_out_list, dim=1).mean(dim=1)


if __name__ == '__main__':
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    context_variants = ['Remember Cambodia ', 'Remember Mongolia ', 'Remember Vanuatu ', 'Remember Chad ', 'Remember Cuba ']
    file_prefix = '\n\nSo I know the following five countries: '
    examples = [cv + file_prefix for cv in context_variants]
    list_input_ids = [tokenizer(example, return_tensors='pt')['input_ids'] for example in examples]
    model = RePlugModel(base_model)
    output = model.generate(list_input_ids, max_new_tokens=25)
    print(output)
    for _output in output:
        print(tokenizer.batch_decode(_output))

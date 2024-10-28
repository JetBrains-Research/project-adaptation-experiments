import time

from tqdm import tqdm

from rag.bug_localization.bpe_tokenizer import BPETokenizer
from rag.bug_localization.cosine_distance_ranker import CosineDistanceRanker
from rag.bug_localization.load_data import load_data
from rag.bug_localization.tfidf_backbone import TfIdfEmbBackbone

# %%
tokenizer = BPETokenizer(vocab_size=10000, min_frequency=2, pretrained_path=None)
ranker = CosineDistanceRanker()

backbone = TfIdfEmbBackbone(tokenizer=tokenizer, ranker=ranker)

# %%

dataset = load_data(["python", "java", "kotlin"])
i = 1
for item in tqdm(dataset):
    issue_description = item["issue_description"]
    repo_content = item["repo_content"]
    start_time = time.time()
    results_dict = backbone.localize_bugs(issue_description, repo_content)
    end_time = time.time()
    item.update(results_dict)
    item["time_s"] = (end_time - start_time) * 1000000
    i += 1

pass

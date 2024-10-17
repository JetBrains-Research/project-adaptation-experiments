from typing import Any, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def data_to_vectors(
    issue_text: str, repo_content: Dict[str, str]
) -> tuple[np.ndarray[str], np.ndarray[str]]:

    file_names = ["issue_text"]
    file_contents = [issue_text]
    for file_name, file_content in repo_content.items():
        file_names.append(file_name)
        file_contents.append(file_name + "\n" + file_content)

    return (np.asarray(file_names, dtype=str), np.asarray(file_contents, dtype=str))


class TfIdfEmbBackbone:

    def __init__(self, tokenizer, ranker):
        self._tokenizer = tokenizer
        self._ranker = ranker

    def localize_bugs(
        self, issue_description: str, repo_content: Dict[str, str], **kwargs
    ) -> Dict[str, Any]:
        file_names, file_contents = data_to_vectors(issue_description, repo_content)
        self._tokenizer.fit(file_contents)
        model = TfidfVectorizer(tokenizer=self._tokenizer.tokenize)
        vect_file_contents = model.fit_transform(file_contents)

        ranked_file_names, rank_scores = self._ranker.rank(
            file_names, vect_file_contents
        )

        return {
            "final_files": list(ranked_file_names),
            "rank_scores": list(rank_scores),
        }

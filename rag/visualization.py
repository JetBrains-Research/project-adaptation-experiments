import pandas as pd
import fire
import os

'''
python3 visualization.py --path /mnt/data/galimzyanov/long-contex-eval/output/rag/results_all_python_chunk_score.jsonl
'''

import matplotlib.pyplot as plt

def plot_em_scores(path):
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_json(path, orient="records", lines=True)
    categories = set()
    
    for idx, row in df.iterrows():
        categories.update(row['category'].values())
    
    fig, axes = plt.subplots(1, len(categories), figsize=(6 * len(categories), 4))
    
    if len(categories) == 1:
        axes = [axes]
    
    for ax, category in zip(axes, categories):
        for idx, row in df.iterrows():
            context_lengths = [v for k, v in row['context_len_mean'].items() if row['category'][k] == category]
            em_scores = [v['exact_match_valid']['mean'] for k, v in row['scores'].items() if row['category'][k] == category]

            # Generate the name for the plot
            # name = f"{row['composer']}_{row['splitter']}_{row['scorer']}_{row['n_grams_max']}_{category}"

            # Plot the data
            ax.plot(context_lengths, em_scores, marker='o', linestyle='-')#, label=name)
        
        ax.set_xlabel('Context Length')
        ax.set_ylabel('Exact Match Valid (EM) Score')
        ax.set_title(f'Exact Match Valid (EM) Scores vs Context Length for {category}')
        ax.grid(True)
        ax.legend()
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    plot_filename = f"{output_dir}/em_scores.png"
    fig.savefig(plot_filename)

if __name__ == "__main__":
    fire.Fire(plot_em_scores)

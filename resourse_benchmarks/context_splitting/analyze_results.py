import json
from matplotlib import pyplot as plt

# %%

def read_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data.append(json.loads(line))
    return data


# %%

results_4090 = read_jsonl("outpu/results_4090.jsonl")
results_A100 = read_jsonl("results_A100x80.jsonl")


# %%
def plot_results(results, title):
    base_len = 1000
    contexts = [result["context"] for result in results]
    times = [result["time_per_sample_mean"] for result in results]
    num_retrieved = [int((context / base_len - 1)) for context in contexts]
    times_splitted = [times[0] * num for num in num_retrieved]

    fig, ax1 = plt.subplots()

    plt.title(title)
    ax1.plot(contexts, times, label='Time long context')
    ax1.plot(contexts, times_splitted, label='time splitted')
    ax1.set_xlabel('Context')
    ax1.set_ylabel('Time')
    ax1.legend(loc='upper left')

    ax2 = ax1.twiny()
    ax2.set_xlabel('Retrieved')
    ax2.set_xlim((num_retrieved[0], num_retrieved[-1] + 1))

    plt.show()


# %%

plot_results(results_4090, "RTX 4090")
plot_results(results_A100, "A100-80G")
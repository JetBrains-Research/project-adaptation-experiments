import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from ipywidgets import interact, widgets
from sympy import plot
from xlwings.utils import chunk
import numpy as np
from pathlib import Path

def merge_df(base_path, paths_list: list[str]):
    dataframes = []
    for path in paths_list:
        full_path = base_path / path
        if not Path(full_path).exists():
            raise FileNotFoundError(f"Path {full_path} does not exist")
        df = pd.read_json(full_path, orient="records", lines=True)
        dataframes.append(df)
    final_dataframe = pd.concat(dataframes, ignore_index=True)
    return final_dataframe

def read_results(df, group_columns):

    dataframes = []

    for _, row in df.iterrows():
        row = row.apply(pd.Series).T.reset_index(drop=True)
        row["em"] = row["scores"].apply(lambda x: x["exact_match_valid"]["mean"])
        if 'is_prefix' in row["scores"].iloc[0]:
            row["is_prefix"] = row["scores"].apply(lambda x: x["is_prefix"]["mean"])
        if 'prefix_ratio' in row["scores"].iloc[0]:
            row["prefix_ratio"] = row["scores"].apply(lambda x: x["prefix_ratio"]["mean"])
        row['category'] = row['category'].apply(lambda x: 'infile' if x == 'InFile' else 'inproject' if x == 'InProject' else x)
        grouped = row.groupby(group_columns).agg({
            'context_len_config': list,
            'em': list,
            'is_prefix': list,
            'prefix_ratio': list,
        }).reset_index()
        dataframes.append(grouped)
    final_dataframe = pd.concat(dataframes, ignore_index=True)

    return final_dataframe

def get_group_columns(df):
    # df = pd.read_json(path, orient="records", lines=True)

    template_drop_columns = ['context_len_config', 'count', 'context_len_mean', 'time_gen_per_item', 'scores', 'time_data_load_per_item', 'stride']
    drop_columns = []
    for col in template_drop_columns:
        if col in df.columns:
            drop_columns.append(col)

    # Drop all columns for which we don't want aggregation
    group_columns = df.columns.drop(drop_columns).tolist()
    return group_columns

def read_results_path(file, group_columns):

    dataframes = []
    agg_by = {'context_len_config': list}

    with open(file, 'r') as file:
        for line in file:
            json_data = StringIO(line.strip())
            df = pd.read_json(json_data)
            df["em"] = df["scores"].apply(lambda x: x["exact_match_valid"]["mean"])
            agg_by['em'] = list
            if 'is_prefix' in df["scores"].iloc[0]:
                df["is_prefix"] = df["scores"].apply(lambda x: x["is_prefix"]["mean"])
                agg_by['is_prefix'] = list
            if 'prefix_ratio' in df["scores"].iloc[0]:
                df["prefix_ratio"] = df["scores"].apply(lambda x: x["prefix_ratio"]["mean"])
                agg_by['prefix_ratio'] = list
            df['category'] = df['category'].apply(lambda x: 'infile' if x == 'InFile' else 'inproject' if x == 'InProject' else x)
            grouped = df.groupby(group_columns).agg(agg_by).reset_index()
            dataframes.append(grouped)
    final_dataframe = pd.concat(dataframes, ignore_index=True)

    return final_dataframe

def get_group_columns_path(path):
    df = pd.read_json(path, orient="records", lines=True)

    drop_columns = ['context_len_config', 'count', 'context_len_mean', 'time_gen_per_item', 'scores', 'time_data_load_per_item']
    if 'stride' in df.columns:
        drop_columns.append('stride')

    # Drop all columns for which we don't want aggregation
    group_columns = df.columns.drop(drop_columns).tolist()
    return group_columns

def filter_target_columns(results, group_columns, delete_columns=[]):    
    target_columns = dict()

    for column in group_columns:
        column_values = results[column].unique().tolist()
        if len(column_values) > 1:
            target_columns[column] = sorted(column_values)

    # del target_columns["stride"]
    for del_col in delete_columns:
        if del_col in target_columns:
            del target_columns[del_col]
    return target_columns

def plot_dropdown(results: pd.DataFrame, plot_by, name_by, metric, title='', fontsize=11, **kwargs):
    filter_cond = ' & '.join(
        [f'{key}==@params["{key}"]' if isinstance(value, (int, float)) 
         else f'{key}=="{value}"' for key, value in kwargs.items()]
    )

    params = {key: value for key, value in kwargs.items()}
    
    filtered_df = results.query(filter_cond, local_dict={'params': params})

    fig, ax = plt.subplots(figsize=(6, 4))
    filtered_df = filtered_df.sort_values(by=name_by)
    for idx, row in filtered_df.iterrows():
        name = "_".join([str(row[col]) for col in name_by])
        ax.plot(row['context_len_config'], row[metric], label=name)
    ax.legend(loc='lower right')
    
    plt.xlabel('Context length', fontsize=fontsize)
    plt.ylabel(metric, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True)
    plt.ylim(0.1, 0.63)

    # Add overall title if provided
    if len(title) != 0:
        fig.suptitle(f"{metric} {title}", fontsize=fontsize + 7)

    plt.tight_layout()
    plt.show()

def plot_dropdown_with_group_by(results: pd.DataFrame, plot_by, group_by, name_by, metric, title='', fontsize=11, **kwargs):
    params = {key: value for key, value in kwargs.items()}
    unique_groups = results[group_by].unique()

    fig, axes = plt.subplots(1, len(unique_groups), figsize=(6 * len(unique_groups), 4))

    if len(unique_groups) == 1:
        axes = [axes]

    unique_groups = sorted(unique_groups)

    for ax, group_value in zip(axes, unique_groups):
        params[group_by] = group_value

        filter_cond = ' & '.join(
            [f'{key}==@params["{key}"]' if isinstance(val, (int, float, np.integer, np.bool_)) 
             else f'{key}=="{val}"' for key, val in params.items()]
        )
        filtered_df = results.query(filter_cond, local_dict={'params': params})
        filtered_df = filtered_df.sort_values(by=name_by)

        for idx, row in filtered_df.iterrows():
            name = "_".join([str(row[col]) for col in name_by])
            ax.plot(row['context_len_config'], row[metric], label=name)
        ax.legend(loc='lower right')
        
        ax.set_xlabel('Context length', fontsize=fontsize)
        ax.set_ylabel(metric, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.grid(True)
        ax.set_ylim(0.1, 0.63)
        ax.set_title(f"{group_by} = {group_value}", fontsize=fontsize)

    # Add overall title if provided
    if len(title) != 0:
        fig.suptitle(f"{metric} {title}", fontsize=fontsize + 7)

    plt.tight_layout()
    plt.show()

def make_interaction(results, group_columns, dropdown, dropdown_params, plot_params, delete_columns=[]):
    delete_columns = delete_columns.copy()
    delete_columns.extend([v for v in dropdown_params.values() if isinstance(v, str)])
    target_columns = filter_target_columns(results, group_columns, delete_columns)

    dropdown_params = dropdown_params.copy()
    dropdown_params['results'] = widgets.fixed(results)

    if 'name_by' not in plot_params:
        plot_params['name_by'] = [dropdown_params['plot_by']]

    # Merge additional_params with target_columns
    all_params = {**target_columns, **dropdown_params}

    # Wrap dropdown to include additional params
    def wrapped_dropdown(**kwargs):
        dropdown(**kwargs, **plot_params)

    interact(wrapped_dropdown, **all_params)


# def make_interaction(results, group_columns, dropdown, plot_params, metrics, delete_columns=[]):
#     delete_columns = delete_columns.copy()
#     delete_columns.extend(plot_params.values())
#     target_columns = filter_target_columns(results, group_columns, delete_columns)

#     plot_params = plot_params.copy()
#     plot_params['results'] = widgets.fixed(results)
#     plot_params['metric'] = metrics

#     # Merge additional_params with target_columns
#     all_params = {**target_columns, **plot_params}

#     interact(dropdown, **all_params)
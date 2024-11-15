import os
import re
import json

from tqdm.auto import tqdm
from datasets import Dataset, load_dataset

from rag.draco.pyfile_parse import PythonParser
from rag.draco.node_prompt import projectSearcher
from rag.draco.utils import DS_GRAPH_DIR


class projectParser(object):
    def __init__(self):
        self.py_parser = PythonParser()
        self.iden_pattern = re.compile(r'[^\w\-]')

        self.proj_searcher = projectSearcher()

        self.proj_dir = None
        self.parse_res = None
    

    def set_proj_dir(self, dir_path):
        if not dir_path.endswith(os.sep):
            self.proj_dir = dir_path + os.sep
        else:
            self.proj_dir = dir_path


    def retain_project_rels(self):
        '''
        retain the useful relationships
        '''
        for module, file_info in self.parse_res.items():
            for name, info_dict in file_info.items():
                cls = info_dict.get("in_class", None)

                # intra-file relations
                rels = info_dict.get("rels", None)
                if rels is not None:
                    del_index = []
                    for i, item in enumerate(rels):
                        # item: [name, type]
                        find_info = self.proj_searcher.name_in_file(item[0], list(file_info), name, cls)
                        if find_info is None:
                            del_index.append(i)
                        else:
                            # modify
                            info_dict["rels"][i] = [find_info[0], find_info[1], item[1]]
                    
                    # delete
                    for index in reversed(del_index):
                        info_dict["rels"].pop(index)
                    
                    if len(info_dict["rels"]) == 0:
                        info_dict.pop("rels")

                # cross-file relations
                imported_info = info_dict.get("import", None)
                if info_dict["type"] == 'Variable' and imported_info is not None:
                    judge_res = self.proj_searcher.is_local_import(module, imported_info)
                    if judge_res is None:
                        info_dict.pop("import")
                    else:
                        info_dict["import"] = judge_res

    def _get_all_module_path(self, file_list) -> dict[set[str]]:
        # Returns dict {folder name: }.
        py_dict = {}

        for file in file_list:
            parts = file.split('/')

            base_part = parts[0]

            for part in parts[1:]:
                if base_part not in py_dict:
                    py_dict[base_part] = set()
                py_dict[base_part].add(base_part + '/' + part)
                base_part = base_part + '/' + part
            
        return py_dict


    def _get_module_name(self, fpath):
        if fpath.endswith('.py'):
            fpath = fpath[:-3]
            if fpath.endswith('__init__'):
                fpath = fpath[:-8]

        fpath = fpath.rstrip(os.sep)
        return fpath.replace(os.sep, '.')


    def parse_dir(self, repo_info: dict):
        '''
        Return: {module: {
            name: {
                "type": str,                         # type: "Module", "Class", "Function", "Variable"
                "def": str,
                "docstring": str (optional),
                "body": str (optional),
                "sline": int (optional),
                "in_class": str (optional),
                "in_init": bool (optional),
                "rels": [[name:str, suffix:str, type:str], ],    # type: "Assign", "Hint", "Rhint", "Inherit"
                "import": [module:str, name:str]     # "Import"
            }
            }}
        '''
        self.set_proj_dir(repo_info['repo'])
        # repo snapshots, comtaining .py files
        repo_snapshot: dict[str, str] = {filename: content \
                         for (filename, content) in zip(repo_info['repo_snapshot']['filename'], repo_info['repo_snapshot']['content']) \
                         if filename.endswith('.py')}
        
        py_dict = self._get_all_module_path(repo_snapshot.keys())
        
        # order: dir, __init__.py, .py
        module_dict = {}
        # dir
        for dir_path in py_dict:
            module = self._get_module_name(dir_path)
            if len(module) > 0:
                module_dict[module] = [dir_path,]
        
        # pyfiles
        init_files = set()
        pyfiles = set()
        for py_set in py_dict.values():
            for fpath in py_set:
                if fpath.endswith(os.sep + '__init__.py'):
                    init_files.add(fpath)
                else:
                    pyfiles.add(fpath)
        
        # __init__.py
        for fpath in init_files:
            module = self._get_module_name(fpath)
            if len(module) > 0:
                if module in module_dict:
                    module_dict[module].append(fpath)
                else:
                    module_dict[module] = [fpath,]
        
        # .py
        for fpath in pyfiles:
            module = self._get_module_name(fpath)
            if len(module) > 0:
                if module in module_dict:
                    module_dict[module].append(fpath)
                else:
                    module_dict[module] = [fpath,]
        
        self.parse_res = {}
        for module, path_list in module_dict.items():
            info_dict = {}
            for fpath in path_list:
                if fpath in py_dict:
                    # dir
                    for item in py_dict[fpath]:
                        submodule = self._get_module_name(item)
                        if submodule != module:
                            # exclude __init__.py
                            info_dict[submodule] = {
                                "type": "Module",
                                "import": [submodule, None]
                            }
                else:
                    # pyfiles
                    source_code = repo_snapshot[fpath]
                    info_dict.update(self.py_parser.parse(source_code))
                    break
            
            if len(info_dict) > 0:
                self.parse_res[module] = info_dict

        self.proj_searcher.set_proj(repo_info['repo'], self.parse_res)
        # connect the files
        self.retain_project_rels()

        return self.parse_res



def generate_draco_graph():
    dataset_path = "JetBrains-Research/lca-project-level-code-completion"
    dataset = load_dataset(path=dataset_path, split="test", name="medium_context")

    project_parser = projectParser()
    
    if not os.path.exists(DS_GRAPH_DIR):
        os.makedirs(DS_GRAPH_DIR)

    # Iterate over the dataset and generate context graphs
    for row in tqdm(dataset):
        repo_name = f"{row['repo']}/{row['commit_hash']}/{row['completion_file']['filename']}"
        repo_dir = os.path.join(DS_GRAPH_DIR, os.path.dirname(repo_name))
        
        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir)

        info = project_parser.parse_dir(row)

        with open(os.path.join(DS_GRAPH_DIR, f'{repo_name}.json'), 'w') as f:
            json.dump(info, f)

    print(f'Generate repo-specific context graph for {len(dataset)} repositories.')
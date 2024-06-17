# from pathlib import Path

# import bibtexparser

from huggingface_hub import HfApi
import json
import mteb

api = HfApi()


def extract_bibtex_to_file(tasks: list[mteb.AbsTask]) -> None:
    """Parse the task and extract bibtex.
    :param tasks:
        List of tasks.
    """
    datasets = {}
    for task in tasks:
        if task.metadata.dataset.get("trust_remote_code",None): continue
        try:
            files = api.list_repo_files(
                repo_id=task.metadata.dataset["path"],
                revision=task.metadata.dataset["revision"],
                repo_type="dataset",
            )
        except Exception as e:
            print(str(e))
            continue
        # python_files = [file for file in files if file.endswith(".py")]
        file = task.metadata.dataset["path"].split('/')[1] + '.py'
        if file in files:
        # if len(python_files) > 0:
            datasets[task.metadata.name] = [
                task.metadata.dataset["path"],
                task.metadata.dataset["revision"],
            ]
    print(datasets)
    print(len(datasets.keys()))
    with open("datasets_confirmed.json", "w") as f:
        json.dump(datasets, f, indent=4)
        # print(files)
        # break


def main():
    # tasks = mteb.get_tasks()
    # print(len(tasks))
    # tasks = sorted(tasks, key=lambda x: x.metadata.name)
    # extract_bibtex_to_file(tasks)
    uniq_rep = set()
    with open('datasets_confirmed.json','r') as f:
        data = json.load(f)
    c = 0
    for k,v in data.items():
        uniq_rep.add(v[0])
        if 'mteb/' in v[0]: c+=1
    print(uniq_rep,len(uniq_rep))
    print(c)




if __name__ == "__main__":
    main()


# hlhdatscience/guanaco-spanish-dataset
# jganzabalseenka/mlsum-spanish-truncated-512
# jojo0217/korean_rlhf_dataset
# joonhok-exo-ai/korean_law_open_data_precedents
# Laplace04/KoreanSummarizeAiHub
# haih2/japanese-conala
# saldra/sakura_japanese_dataset
# Mustain/fujiki_49k_japanese_dataset
# notoxicpeople/japan_diet_q_and_a_sessions_20k
# hacktoberfest-corpus-es/colmbian_spanish_news



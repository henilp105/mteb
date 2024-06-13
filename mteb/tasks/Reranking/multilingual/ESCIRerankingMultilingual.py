from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskReranking import AbsTaskReranking
import random
import datasets
from tqdm import tqdm

_EVAL_LANGS = {
   "en": ["eng-Latn"],
   "es": ["spa-Latn"],
   "jp": ["jpn-Latn"],
}


class ESCIRerankingMultilingual(MultilingualTask, AbsTaskReranking):
    EVAL_SPLIT = 'test'
    metadata = TaskMetadata(
        name="ESCIRerankingMultilingual",
        description="The dataset is a large collection of difficult Amazon search queries and results, publicly released with the aim of fostering research in improving the quality of search results.",
        reference="https://arxiv.org/abs/2206.06588",
        hf_hub_name="henilp105/ESCI",
        dataset={
            "path": "tasksource/esci",
            "revision": "8113b17a5d4099e20243282c926f1bc1a08a4d13",
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2022-05-01", "2022-06-14"),
        form=["written"],
        domains=["Web"],
        task_subtypes=[],
        license="apache-2.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{reddy2022shopping,
      title={Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search}, 
      author={Chandan K. Reddy and Lluís Màrquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
      year={2022},
      eprint={2206.06588},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        n_samples={
            "en": 434234,
            "es": 97186,
            "jp": 121070,
        },
        avg_character_length={},
    )

    @staticmethod
    def _sample_dataset(
        corpus: dict,
        queries: dict,
        relevant_docs: dict,
        seed: int,
        splits: list[str],
        n_samples: int = 2048,
    ):
        random.seed(seed)
        for lang in queries:
            for split in splits:
                if len(queries[lang][split]) <= n_samples:
                    continue
                query_ids = random.sample(list(queries[lang][split]), k=n_samples)
                queries[lang][split] = {
                    query_id: queries[lang][split][query_id] for query_id in query_ids
                }

                corpus_keys = {
                    product_id
                    for key in query_ids
                    for product_id in relevant_docs[lang][split][key]
                    if product_id in corpus[lang][split]
                }

                relevant_docs[lang][split] = {
                    query_id: relevant_docs[lang][split][query_id] for query_id in query_ids
                }
                corpus[lang][split] = {key: corpus[lang][split][key] for key in corpus_keys}
        return corpus, queries, relevant_docs

    @staticmethod
    def _filter_non_existing_relevant_docs(corpus: dict, relevant_docs: dict) -> dict:
        for lang in relevant_docs:
            for split in relevant_docs[lang]:
                remove_query_id = []
                for query_id in relevant_docs[lang][split]:
                    query_relevant = {
                        product_id: esci_label
                        for product_id, esci_label in relevant_docs[lang][split][query_id].items()
                        if product_id in corpus[lang][split]
                    }
                    if len(query_relevant) == 0:
                        remove_query_id.append(query_id)
                    else:
                        relevant_docs[lang][split][query_id] = query_relevant
                for query_id in remove_query_id:
                    del relevant_docs[lang][split][query_id]
        return relevant_docs

    @staticmethod
    def _filter_non_existing_corpus_docs(corpus: dict, relevant_docs: dict) -> dict:
        for lang in relevant_docs:
            for split in relevant_docs[lang]:
                product_ids = {
                    product_id
                    for query_id in relevant_docs[lang][split]
                    for product_id in relevant_docs[lang][split][query_id]
                }
                non_existent_docs = set(corpus[lang][split]).difference(product_ids)
                for product_id in non_existent_docs:
                    del corpus[lang][split][product_id]
        return corpus

    @staticmethod
    def _filter_non_existing_queries(queries: dict, relevant_docs: dict) -> dict:
        for lang in queries:
            for split in queries[lang]:
                non_existent_queries = set(queries).difference(set(relevant_docs[lang][split]))
                for query_id in non_existent_queries:
                    del queries[lang][split][query_id]
        return queries

    def load_data(self, **kwargs):
        product_locale_map = {"jp": "ja", "us": "en"}
        label_map = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}
        data = datasets.load_dataset(
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )
        corpus = {lang: {self._EVAL_SPLIT: {}} for lang in _EVAL_LANGS}
        queries = {lang: {self._EVAL_SPLIT: {}} for lang in _EVAL_LANGS}
        relevant_docs = {lang: {self._EVAL_SPLIT: {}} for lang in _EVAL_LANGS}

        for example in tqdm(data, desc="Preparing data"):
            product_locale = example.get("product_locale")
            lang = product_locale_map.get(product_locale, product_locale)

            if example.get("query_id") and example.get("query"):
                query_id = str(example["query_id"])
                query_text = example["query"]

                if query_id not in queries[lang][self._EVAL_SPLIT]:
                    queries[lang][self._EVAL_SPLIT][query_id] = query_text

                product_id = example.get("product_id")
                esci_label = example.get("esci_label")
                if product_id and esci_label:
                    relevant_docs[lang][self._EVAL_SPLIT].setdefault(query_id, {})[product_id] = (
                        label_map[esci_label]
                    )

            if (
                example.get("product_id")
                and example.get("product_title")
                and example.get("product_description")
            ):
                product_id = example["product_id"]
                product_title = example["product_title"]
                product_description = example["product_description"]
                if product_id not in corpus[lang][self._EVAL_SPLIT]:
                    corpus[lang][self._EVAL_SPLIT][product_id] = {
                        "text": product_title + ": " + product_description,
                    }

        self.corpus, self.queries, self.relevant_docs = corpus, queries, relevant_docs
        self.corpus, self.queries, self.relevant_docs = self._sample_dataset(
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
            seed=self.seed,
            splits=[self._EVAL_SPLIT],
        )
        print(
            "Total relevant docs",
            sum(
                len(self.relevant_docs[lang][split][query_id])
                for lang in self.relevant_docs
                for split in self.relevant_docs[lang]
                for query_id in self.relevant_docs[lang][split]
            ),
        )
        self.relevant_docs = self._filter_non_existing_relevant_docs(
            self.corpus, self.relevant_docs
        )
        print(
            "Total relevant docs",
            sum(
                len(self.relevant_docs[lang][split][query_id])
                for lang in self.relevant_docs
                for split in self.relevant_docs[lang]
                for query_id in self.relevant_docs[lang][split]
            ),
        )
        print(
            "Total corpus docs",
            sum(
                len(self.corpus[lang][split])
                for lang in self.corpus
                for split in self.corpus[lang]
            ),
        )
        self.corpus = self._filter_non_existing_corpus_docs(self.corpus, self.relevant_docs)
        print(
            "Total corpus docs",
            sum(
                len(self.corpus[lang][split])
                for lang in self.corpus
                for split in self.corpus[lang]
            ),
        )
        self.queries = self._filter_non_existing_queries(self.queries, self.relevant_docs)

        self.data_loaded = True
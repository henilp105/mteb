from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskReranking import AbsTaskReranking

_EVAL_LANGS = {
   "en": ["eng-Latn"],
   "es": ["spa-Latn"],
   "jp": ["jpn-Latn"],
}


class ESCIRerankingMultilingual(MultilingualTask, AbsTaskReranking):
    metadata = TaskMetadata(
        name="ESCIRerankingMultilingual",
        description="The dataset is a large collection of difficult Amazon search queries and results, publicly released with the aim of fostering research in improving the quality of search results.",
        reference="https://arxiv.org/abs/2206.06588",
        hf_hub_name="henilp105/ESCI",
        dataset={
            "path": "henilp105/ESCI",
            "revision": "8bdce351e20f29bb2e2755d3d798bc128cd601ed",
        },
        type="Reranking",
        category="s2p",
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="map",
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

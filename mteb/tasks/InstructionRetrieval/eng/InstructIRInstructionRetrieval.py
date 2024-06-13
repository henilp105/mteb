from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class InstructIRInstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="InstructIRInstructionRetrieval",
        description="Measuring retrieval instruction following ability on Core17 narratives.",
        reference="https://arxiv.org/abs/2402.14334",
        dataset={
            "path": "kaist-ai/InstructIR",
            "revision": "1615ed0ee07d6d33b0082362f008bfe62041a54b",
        },
        type="InstructionRetrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="p-MRR",
        date=("2023-08-01", "2024-04-01"), # need reference
        form=["written"],
        domains=["Web"],
        task_subtypes=[],
        license="MIT",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{oh2024instructir,
      title={INSTRUCTIR: A Benchmark for Instruction Following of Information Retrieval Models}, 
      author={Hanseok Oh and Hyunji Lee and Seonghyeon Ye and Haebin Shin and Hansol Jang and Changwook Jun and Minjoon Seo},
      year={2024},
      eprint={2402.14334},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        n_samples={"eng": 16072 * 2},
        avg_character_length={"eng": 2768.749235474006},
    )

data = ['GermanGovServiceRetrieval', 'LegalQuAD', 'GermanQuAD-Retrieval', 'GerDaLIR', 'queries', 'corpus', 'qrels', 'GerDaLIRSmall', 'GermanDPR', 'JaQuADRetrieval', 'SpanishPassageRetrievalS2P', 'queries', 'corpus.documents', 'qrels.s2p', 'SpanishPassageRetrievalS2S', 'queries', 'corpus.sentences', 'qrels.s2s', 'SadeemQuestionRetrieval', 'TV2Nordretrieval', 'TwitterHjerneRetrieval', 'DanFEVER', 'BSARDRetrieval', 'corpus', 'questions', 'FQuADRetrieval', 'AlloprofRetrieval', 'documents', 'queries', 'SyntecRetrieval', 'documents', 'queries', 'NQ', 'TopiOCQA', 'CQADupstackStatsRetrieval', 'SciFact', 'CQADupstackAndroidRetrieval', 'QuoraRetrieval', 'FeedbackQARetrieval', 'CQADupstackEnglishRetrieval', 'CQADupstackGamingRetrieval', 'CQADupstackGisRetrieval', 'LEMBWikimQARetrieval', 'LegalSummarization', 'HagridRetrieval', 'FEVER', 'LEMBPasskeyRetrieval', 'MSMARCO', 'HotpotQA', 'LEMBNarrativeQARetrieval', 'MSMARCOv2', 'MLQuestions', 'NarrativeQARetrieval', 'DBPedia', 'FiQA2018', 'CQADupstackWordpressRetrieval', 'CQADupstackTexRetrieval', 'CQADupstackProgrammersRetrieval', 'AILAStatutes', 'TRECCOVID', 'LEMBQMSumRetrieval', 'ClimateFEVER', 'Touche2020', 'ArguAna', 'LEMBNeedleRetrieval', 'LEMBSummScreenFDRetrieval', 'MedicalQARetrieval', 'CQADupstackUnixRetrieval', 'SCIDOCS', 'CQADupstackWebmastersRetrieval', 'LegalBenchCorporateLobbying', 'LegalBenchConsumerContractsQA', 'CQADupstackMathematicaRetrieval', 'NFCorpus', 'CQADupstackPhysicsRetrieval', 'FaithDial', 'AILACasedocs', 'SlovakSumRetrieval', 'ArguAna-PL', 'NQ-PL', 'TRECCOVID-PL', 'NFCorpus-PL', 'HotpotQA-PL', 'MSMARCO-PL', 'SciFact-PL', 'Quora-PL', 'SCIDOCS-PL', 'FiQA-PL', 'DBPedia-PL', 'T2Retrieval', 'MMarcoRetrieval', 'DuRetrieval', 'CovidRetrieval', 'CmedqaRetrieval', 'EcomRetrieval', 'MedicalRetrieval', 'VideoRetrieval', 'LeCaRDv2', 'VieQuADRetrieval', 'RiaNewsRetrieval', 'RuBQRetrieval', 'IndicQARetrieval', 'CrossLingualSemanticDiscriminationWMT19', 'WikipediaRetrievalMultilingual', 'MultiLongDocRetrieval', 'XPQARetrieval', 'BelebeleRetrieval', 'StatcanDialogueDatasetRetrieval', 'NeuCLIR2023Retrieval', 'MIRACLRetrieval', 'CrossLingualSemanticDiscriminationWMT21', 'XQuADRetrieval', 'MintakaRetrieval', 'MLQARetrieval', 'XMarket', 'PublicHealthQA', 'NeuCLIR2022Retrieval', 'Ko-StrategyQA', 'Ko-miracl', 'NorQuadRetrieval', 'SNLRetrieval', 'GeorgianFAQRetrieval', 'GreekCivicsQA', 'EstQA', 'TurHistQuadRetrieval', 'SwednRetrieval', 'SweFaqRetrieval', 'CodeEditSearchRetrieval', 'CodeSearchNetRetrieval', 'HunSum2AbstractiveRetrieval']
import mteb
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "average_word_embeddings_komninos"
# or directly from huggingface:
# model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)

for i in data:
    tasks = mteb.get_tasks(tasks=[i])
    print(tasks)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"results/{model_name}")
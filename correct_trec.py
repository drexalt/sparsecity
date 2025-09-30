import gzip
import logging
import os
from collections import defaultdict
import numpy as np
import pytrec_eval
from sentence_transformers import SentenceTransformer, util
import torch
from src.sparsecity.evaluation.st_wrapper import ST_SPLADEModule, ST_SPLADEV3Module
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, AutoConfig
from src.sparsecity.models.splade_models.model_registry import get_splade_model

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

device = "cuda"
###  Load model
config = BertConfig.from_pretrained(
    "drexalt/mosaic-1024-spladev3-distil", trust_remote_code=True
)
# splade_model = get_splade_model(
#     "chandar-lab/NeoBERT",
#     config=config,
#     device=device,
#     sparse_embed=False,
#     custom_kernel=False,
#     top_k=256,
# )
# state_dict = torch.load(
#     "checkpoints/neo-distil/20250709_052308/checkpoint_step_135999_msmarco_mrr@10_0.7198.pt",
#     weights_only=False,
# )
# state_dict = state_dict["splade_model"]

# splade_model.load_state_dict(state_dict)

# splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3")
splade_model = AutoModelForMaskedLM.from_pretrained(
    "drexalt/mosaic-1024-spladev3-distil", trust_remote_code=True, config=config
)
tokenizer = AutoTokenizer.from_pretrained("chandar-lab/NeoBERT")
st_module = ST_SPLADEV3Module(splade_model, tokenizer, max_length=512)
st_model = SentenceTransformer(modules=[st_module]).to(device)

DATA_FOLDER = "/root/data/trec-dl-2019-data"
os.makedirs(DATA_FOLDER, exist_ok=True)

queries_fp = os.path.join(DATA_FOLDER, "msmarco-test2019-queries.tsv.gz")
qrels_fp = os.path.join(DATA_FOLDER, "2019qrels-pass.txt")
passage_fp = os.path.join(DATA_FOLDER, "msmarco-passagetest2019-top1000.tsv.gz")

# Load queries
queries = {}
with gzip.open(queries_fp, "rt", encoding="utf8") as f_in:
    for line in f_in:
        qid, query_text = line.rstrip().split("\t")
        queries[qid] = query_text

# Load qrels
relevant_docs = defaultdict(dict)
with open(qrels_fp) as f_in:
    for line in f_in:
        qid, _unused, pid, score = line.rstrip().split()
        score = int(score)
        if score > 0:  # Include only relevant documents
            relevant_docs[qid][pid] = score
# Load passages per query
passage_cand = defaultdict(list)
with gzip.open(passage_fp, "rt", encoding="utf8") as f_in:
    for line in f_in:
        qid, pid, _query, passage = line.rstrip().split("\t")
        passage_cand[qid].append([pid, passage])

# Identify relevant queries
relevant_qid = [qid for qid in queries if len(relevant_docs[qid]) > 0]
logging.info(f"Queries for eval: {len(relevant_qid):,}")

# Compute rankings
run = {}
for qid in relevant_qid:
    query = queries[qid]
    cand = passage_cand[qid]
    pids = [c[0] for c in cand]
    passages = [c[1] for c in cand]
    query_emb = st_model.encode(query, convert_to_tensor=True)
    passage_embs = st_model.encode(passages, convert_to_tensor=True)
    scores = util.dot_score(query_emb, passage_embs)[0].cpu().tolist()
    run[qid] = {pids[idx]: scores[idx] for idx in range(len(pids))}

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, {"ndcg_cut.10"})
scores = evaluator.evaluate(run)
ndcg_10 = np.mean([ele["ndcg_cut_10"] for ele in scores.values()]) * 100
logging.info(f"NDCG@10: {ndcg_10:.2f}")

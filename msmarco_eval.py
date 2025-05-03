"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.

Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""

import logging
import os
import sys
import tarfile

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, util
from sentence_transformers.similarity_functions import dot_score
from src.sparsecity.evaluation.st_wrapper import ST_SPLADEModule
from src.sparsecity.models.splade_models.model_registry import get_splade_model
import torch
from transformers import AutoTokenizer

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

torch.set_float32_matmul_precision("high")
# You can limit the approx. max size of the corpus. Pass 100 as second parameter and the corpus has a size of approx 100k docs
corpus_max_size = 0

device = "cuda"
####  Load model
splade_model = get_splade_model(
    "answerdotai/modernbert-base", device=device, sparse_embed=False, custom_kernel=True
)
state_dict = torch.load("checkpoint_step_177595_ndcg_0.7757.pt", weights_only=False)
state_dict = state_dict["splade_model"]

splade_model.load_state_dict(state_dict)

tokenizer = AutoTokenizer.from_pretrained("answerdotai/modernbert-base")
st_module = ST_SPLADEModule(splade_model, tokenizer, max_length=256)
st_model = SentenceTransformer(modules=[st_module]).to(device)
### Data files
data_folder = "~/data/msmarco-dev-data"
os.makedirs(data_folder, exist_ok=True)

collection_filepath = os.path.join(data_folder, "collection.tsv")
dev_queries_file = os.path.join(data_folder, "queries.dev.small.tsv")
qrels_filepath = os.path.join(data_folder, "qrels.dev.tsv")

### Download files if needed
if not os.path.exists(collection_filepath) or not os.path.exists(dev_queries_file):
    tar_filepath = os.path.join(data_folder, "collectionandqueries.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download: " + tar_filepath)
        util.http_get(
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
            tar_filepath,
        )

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


if not os.path.exists(qrels_filepath):
    util.http_get(
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
        qrels_filepath,
    )

### Load data

corpus = {}  # Our corpus pid => passage
dev_queries = {}  # Our dev queries. qid => query
dev_rel_docs = {}  # Mapping qid => set with relevant pids
needed_pids = set()  # Passage IDs we need
needed_qids = set()  # Query IDs we need

# Load the 6980 dev queries
with open(dev_queries_file, encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        dev_queries[qid] = query.strip()


# Load which passages are relevant for which queries
with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split("\t")

        if qid not in dev_queries:
            continue

        if qid not in dev_rel_docs:
            dev_rel_docs[qid] = set()
        dev_rel_docs[qid].add(pid)

        needed_pids.add(pid)
        needed_qids.add(qid)


# Read passages
with open(collection_filepath, encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        passage = passage

        if pid in needed_pids or corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
            corpus[pid] = passage.strip()


## Run evaluator
logging.info(f"Queries: {len(dev_queries)}")
logging.info(f"Corpus: {len(corpus)}")

ir_evaluator = evaluation.InformationRetrievalEvaluator(
    dev_queries,
    corpus,
    dev_rel_docs,
    show_progress_bar=True,
    batch_size=128,
    score_functions={"dot": dot_score},
    corpus_chunk_size=100000,
    precision_recall_at_k=[10, 100],
    name="msmarco dev",
)

ir_evaluator(st_model)

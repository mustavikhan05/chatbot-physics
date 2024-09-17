import mteb
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel


model = SentenceTransformer("avsolatorio/GIST-small-Embedding-v0")  # Use SentenceTransformer instead of AutoModel

tasks = mteb.get_tasks(tasks=["CQADupstackPhysicsRetrieval"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/avsolatorio//GIST-small-Embedding-v0")
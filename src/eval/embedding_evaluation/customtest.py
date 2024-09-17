from mteb import MTEB
from sentence_transformers import SentenceTransformer
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata
import datasets
import numpy as np
import torch
import logging

# Custom Retrieval Task
class PhysicsSyntheticRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PhysicsSyntheticRetrieval",
        description="Retrieval task using synthetic physics queries and textbook chunks.",
        reference="",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        dataset={
            "path": "/Users/mustavikhan/Desktop/chatbot/data/generated_queries_excerpts.csv",
            "revision": "79531abbd1fb92d06c6d6315a0cbbbf5bb247ea4",
        },
        date=("2024-01-01", "2024-12-31"),
        domains=["Physics", "Academic"],
        task_subtypes=["Synthetic Retrieval"],
        license="cc-by-4.0",
        annotations_creators="generated",
        dialect=[],
        sample_creation="generated",
        descriptive_stats={
            "n_samples": {"test": 1000},  # Adjust based on your dataset
            "avg_character_length": {"test": 100}
        },
        bibtex_citation=None
    )

    def load_data(self, eval_splits=None, **kwargs):
        # Load your dataset
        path = self.metadata.dataset["path"]
        data = datasets.load_dataset('csv', data_files=path)
        self.dataset = data['train']  # Assuming dataset has a 'train' split for simplicity
        print(f"Loaded dataset: {path}")

    def dataset_transform(self):
        # Optional: You can transform the dataset here to fit the expected format
        pass

    def compute_metrics(self, ranked_chunks, ground_truth_chunk):
        # Simple metric computation: Check if the correct chunk is in the top-k
        rank = np.where(ranked_chunks == ground_truth_chunk)[0][0]
        ndcg = 1 / (np.log2(rank + 2))  # Simple NDCG
        mrr = 1 / (rank + 1)  # Simple MRR
        return ndcg, mrr

    def evaluate(self, model, split, output_folder=None, **kwargs):
        # Create queries and chunks from your dataset
        queries = self.dataset['question']
        references = self.dataset['references']

        # Load chunks from your corpus (e.g., txt files)
        chunks = []
        with open("/Users/mustavikhan/Desktop/chatbot/data/chunks", "r") as f:
            for line in f:
                chunks.append(line.strip())

        # Embed the queries and the chunks
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        query_embeddings = model.encode(queries, convert_to_tensor=True)

        ndcg_scores = []
        mrr_scores = []

        # For each query, compute similarity with all chunks
        for i, query_embedding in enumerate(query_embeddings):
            scores = torch.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings)
            ranked_chunks = scores.argsort(descending=True).cpu().numpy()

            ground_truth_chunk = references[i]
            ndcg, mrr = self.compute_metrics(ranked_chunks, ground_truth_chunk)
            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)

        avg_ndcg = np.mean(ndcg_scores)
        avg_mrr = np.mean(mrr_scores)

        # Return the results in the expected format for MTEB
        return {
            "default": {
                "main_score": avg_ndcg,
                "ndcg_at_10": avg_ndcg,
                "mrr_at_1": avg_mrr
            }
        }

# Main Script
if __name__ == "__main__":
    # Configure logging for easier debugging
    logging.basicConfig(level=logging.INFO)

    # Load the SentenceTransformer model
    model = SentenceTransformer("average_word_embeddings_komninos")

    # Create an instance of the custom task
    custom_task = PhysicsSyntheticRetrieval()

    # Create an MTEB evaluation object with the custom task
    evaluation = MTEB(tasks=[custom_task])

    # Run the evaluation
    evaluation.run(model)
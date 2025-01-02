import os
import sys
import torch
import joblib
import requests
import argparse
import numpy as np
import pandas as pd

np.random.seed(42)
torch.manual_seed(42)

from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import classification_report

def predict(
    models: List[str],
    weights: List[int],
    data: pd.DataFrame,
    bases: List[str],
    device: str,
    threshold: float,
    labels: List[int] = None
):
    """
    run classification by multiple models and soft majority vote

    args:
        models (List[str]): paths to classifier model files in `.joblib` format
        weights (List[int]): weights for each model in the ensemble
        data (pd.DataFrame): input data with 'title' and 'abstract' columns
        bases (List[str]): pre-trained bert-base model identifiers
        device (str): device to run inference, either 'cpu' or 'cuda'
        threshold (float): threshold for binary predictions
        labels (List[int], optional): ground-truth labels for evaluation

    returns:
        Tuple[np.ndarray, np.ndarray]: binary predictions and probabilities
    """

    def get(
        row: pd.Series,
        base: str,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        device: str
    ) -> np.ndarray:
        """
        get embeddings

        args:
            row (pd.Series): row with 'title' and 'abstract' columns
            base (str): pre-trained bert-base model identifier
            model (AutoModel): bert-base model
            tokenizer (AutoTokenizer): bert-base tokenizer
            device (str): device to run inference

        returns:
            np.ndarray: embeddings for title and abstract
        """
        title = [row['title']]
        abstract = [row['abstract']]

        def meanpooling(
            output: Tuple[torch.Tensor, ...],
            mask: torch.Tensor
        ) -> torch.Tensor:
            """
            compute mean pooling on token embeddings

            args:
                output (Tuple[torch.Tensor, ...]): model output
                mask (torch.Tensor): attention mask

            returns:
                torch.Tensor: mean-pooling embeddings
            """
            embeddings = output[0]
            mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
            return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        def tokenize(
            text: List[str]
        ) -> dict:
            """
            tokenize input text

            args:
                text (List[str]): list of text strings to tokenize

            Returns:
                dict: input tokens
            """
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }
            return inputs

        def encode(
            text: List[str],
            pooling: bool
        ) -> torch.Tensor:
            """
            encode text into embeddings

            args:
                text (List[str]): list of text strings
                pooling (bool): whether to compute mean pooling

            returns:
                torch.Tensor: embeddings
            """
            inputs = tokenize(text)
            with torch.no_grad():
                output = model(**inputs)
            embeddings = output.pooler_output if not pooling else meanpooling(
                output,
                inputs['attention_mask']
           )
            return embeddings

        if base == 'pubmed-bert-base':
            title = encode(title, pooling=False)
            abstract = encode(abstract, pooling=False)
            embeddings = torch.cat((title, abstract), dim=-1)
        elif base == 'biomed-bert-base':
            title = encode(title, pooling=True)
            abstract = encode(abstract, pooling=True)
            embeddings = 0.2 * title + 0.8 * abstract
        else:
            raise ValueError(f"unknown base model: {base}")
        return embeddings.cpu().numpy()
    
    # main logic for ensemble predictions
    probs = None
    for path, weight, base in zip(models, weights, bases):
        print(f"loading classifier from {path}...")
        pipeline = joblib.load(path)
        print(f"loading {base}...")
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModel.from_pretrained(base)
        model = model.to(device)
        x_input = list()
        for idx, row in data.reset_index(drop=True).iterrows():
            percentage = (idx + 1) / len(data) * 100
            sys.stdout.write(f"\rencoding data... {percentage:.2f}%")
            sys.stdout.flush()
            x_input.append(get(row, base, model, tokenizer, device))
        x_input = np.vstack(x_input)
        print(f"\ncomputing predictions...\n")
        preds = pipeline.predict_proba(x_input)
        if probs is None:
            probs = preds * weight
        else:
            probs += preds * weight
    probs /= sum(weights)
    print(f"processing soft majority vote...")
    preds = (probs[:, 1] >= threshold).astype(int)
    print("predictions:", preds)
    if labels is not None:
        print(classification_report(labels, preds))
    return preds, probs

def main(
    title: List[str], 
    abstract: List[str], 
    device: str
):
    """
    run the classification pipeline

    args:
        title (List[str]): list of titles
        abstract (List[str]): list of abstracts
        device (str): device to run inference, either 'cpu' or 'cuda'
    """
    dir = "PPS-BC/models/"
    models = [
        os.path.join(dir, 'pubmed-knn-pipeline.joblib'), 
        os.path.join(dir, 'biomed-svc-pipeline.joblib')
    ]
    bases = [
        os.path.join(dir, 'pubmed-bert-base'),
        os.path.join(dir, 'biomed-bert-base')
    ]
    for base in bases:
        if not os.path.exists(base):
            raise FileNotFoundError(f"BERT model directory {base} does not exist")
    threshold = 0.3875
    weights = [0.4375, 0.5625]
    data = pd.DataFrame({'title': title, 'abstract': abstract})
    preds, probs = predict(
        models=models,
        weights=weights,
        data=data,
        bases=bases,
        device=device,
        threshold=threshold
    )

if __name__ == "__main__":
    """
    parse arguments and run the main classification process
    """
    parser = argparse.ArgumentParser(description="run PPS classification pipeline")
    parser.add_argument("--title", type=str, help="paper title for classification")
    parser.add_argument("--abstract", type=str, help="paper abstract for classification")
    parser.add_argument("--csv", type=str, help="path to a CSV file with 'title' and 'abstract' columns")
    parser.add_argument("--device", type=str, default="cpu", help="device to run inference: 'cpu' or 'cuda'")
    args = parser.parse_args()

    # validate inputs
    if args.csv and (args.title or args.abstract):
        raise ValueError("provide either a CSV file or a single title and abstract, not both")
    if not args.csv and not (args.title and args.abstract):
        raise ValueError("provide either a CSV file or a single title and abstract")

    # load data
    if args.csv:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(f"the file {args.csv} does not exist")
        data = pd.read_csv(args.csv)
        columns = {column.lower(): column for column in data.columns}
        if "title" not in columns or "abstract" not in columns:
            raise ValueError("provide a CSV file with 'title' and 'abstract' columns")
        data = data.rename(columns={columns["title"]: "title", columns["abstract"]: "abstract"})
    else:
        data = pd.DataFrame({
            "title": [args.title],
            "abstract": [args.abstract]
        })

    # run the main function
    main(
        data["title"].tolist(),
        data["abstract"].tolist(),
        args.device
    )

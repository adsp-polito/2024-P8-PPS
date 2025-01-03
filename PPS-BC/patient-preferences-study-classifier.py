import os
import sys
import torch
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd

np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings('ignore')

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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    run prediction by soft majority vote

    args:
        models (List[str]): paths to classifier model files in `.joblib` format
        weights (List[int]): weights for each model in the ensemble
        data (pd.DataFrame): input data with 'title' and 'abstract' columns
        bases (List[str]): pre-trained BERT-base model identifiers
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
        get embeddings for row data

        args:
            row (pd.Series): row with 'title' and 'abstract' columns
            base (str): pre-trained BERT-base model identifier
            model (AutoModel): BERT-base model
            tokenizer (AutoTokenizer): BERT-base tokenizer
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
            text: List[str],
            tokenizer: AutoTokenizer
        ) -> dict:
            """
            tokenize input text

            args:
                text (List[str]): list of text strings to tokenize
                tokenizer (AutoTokenizer): BERT-base tokenizer

            returns:
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
            model: AutoModel,
            tokenizer: AutoTokenizer,
            pooling: bool
        ) -> torch.Tensor:
            """
            encode text into embeddings

            args:
                text (List[str]): list of text strings
                model (AutoModel): BERT-base model
                tokenizer (AutoTokenizer): BERT-base tokenizer
                pooling (bool): whether to compute mean pooling

            returns:
                torch.Tensor: embeddings
            """
            inputs = tokenize(text, tokenizer)
            with torch.no_grad():
                output = model(**inputs)
            embeddings = output.pooler_output if not pooling else meanpooling(
                output,
                inputs['attention_mask']
            )
            return embeddings

        if base == 'NeuML/pubmedbert-base-embeddings':
            title = encode(title, model, tokenizer, pooling=False)
            abstract = encode(abstract, model, tokenizer, pooling=False)
            embeddings = torch.cat((title, abstract), dim=-1)
        elif base == 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract':
            title = encode(title, model, tokenizer, pooling=True)
            abstract = encode(abstract, model, tokenizer, pooling=True)
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
        for idx, row in data.iterrows():
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
    title,
    abstract,
    device: str
):
    """
    run the classification pipeline

    args:
        title (List[str] or str): list of titles or a single title
        abstract (List[str] or str): list of abstracts or a single abstract
        device (str): device to run inference, either 'cpu' or 'cuda'
    """
    if isinstance(title, list) and isinstance(abstract, list):
        data = pd.DataFrame({'title': title, 'abstract': abstract})
    else:
        data = pd.DataFrame({'title': [title], 'abstract': [abstract]})
    dir = "/content/2024-P8-PPS/PPS-BC/models/"
    models = [
        os.path.join(dir, 'pubmed-knn-pipeline.joblib'), 
        os.path.join(dir, 'biomed-svc-pipeline.joblib')
    ]
    bases = [
        'NeuML/pubmedbert-base-embeddings',
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    ]
    threshold = 0.3875
    weights = [0.4375, 0.5625]
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
    parser = argparse.ArgumentParser(description="run patient-preferences-study classification pipeline")
    parser.add_argument(
        "--title", 
        type=str, 
        help="csv list of paper titles for classification, or a single title")
    parser.add_argument(
        "--abstract", 
        type=str, 
        help="csv list of paper abstracts for classification, or a single abstract")
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="device to run inference: 'cpu' or 'cuda'")
    args = parser.parse_args()

    if ',' in args.title:
        title = args.title.split(',')
    else:
        title = [args.title]

    if ',' in args.abstract:
        abstract = args.abstract.split(',')
    else:
        abstract = [args.abstract]

    # validate inputs
    if not title or not abstract:
        raise ValueError("provide both a title and an abstract")

    # run the main function
    main(
        title, 
        abstract, 
        args.device
    )

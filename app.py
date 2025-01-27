import gc
import sys
import torch
import joblib

import numpy as np
import pandas as pd
import streamlit as st

from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel

np.random.seed(42)
torch.manual_seed(42)

def predict(
    dataset: pd.DataFrame,
    threshold: float = 0.3875,
    weights: Tuple[float, float] = (0.4375, 0.5625)
) -> pd.DataFrame:
    """
    evaluates the relevance of academic papers to patient preference studies by
    predictions of multiple machine learning models and
    embeddings from transformer architectures for text representation

    parameters:
        dataset (pd.DataFrame): the input dataset
        threshold (float): a floating-point value that serves as the decision threshold for classifying
                           papers as relevant to patient preference studies by prediction probabilities
        weights (tuple): a tuple of two floating-point values that determine the relative weight of the
                         predictions from each model in the final decision

    returns:
        pd.DataFrame: the original dataset with an additional column, 'PPS',
                      with binary values: relevant (1) or non-relevant (0)
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def remove(model):
        """
        deallocates memory and clears the gpu cache

        parameters:
            model: a transformer model to get embeddings for the text inputs (title and abstract)
        """
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def get(
        row: pd.Series,
        base: str,
        model: AutoModel,
        tokenizer: AutoTokenizer
    ) -> np.ndarray:
        """
        computes a numerical representation (embedding) for a single paper by title and abstract
        through a transformer model and a vector combination of the resulting embeddings

        parameters:
            row (pd.Series): a single row from the dataset
            base (str): the identifier of the transformer model architecture
            model (AutoModel): the pre-trained transformer model for text embeddings
            tokenizer (AutoTokenizer): the tokenizer of the transformer model

        returns:
            np.ndarray: the sentence embeddings of the paper
        """

        title = [str(row['title']) if isinstance(row['title'], str) else '.']
        abstract = [str(row['abstract']) if isinstance(row['abstract'], str) else '.']

        def meanpooling(
            output: Tuple[torch.Tensor, ...],
            mask: torch.Tensor
        ) -> torch.Tensor:
            """
            computes the mean of the token-level embeddings, with weights on the attention mask

            parameters:
                output (tuple): the output tuple from the transformer model
                mask (torch.Tensor): the attention mask

            returns:
                torch.Tensor: the embedding vector of the input text
            """
            embeddings = output[0]
            mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
            return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

        def tokenize(
            text: List[str],
            tokenizer: AutoTokenizer
        ) -> dict:
            """
            tokenizes the input text for model inference

            parameters:
                text (list): a list of text strings
                tokenizer (AutoTokenizer): the tokenizer of the transformer model

            returns:
                dict: a dictionary with tokens, ready for the model to process
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
            transforms the input text into an embedding by transformer model,
            with an optional mean pooling step to the hidden states

            parameters:
                text (list): the text (e.g., title or abstract of a paper)
                model (AutoModel): the transformer model to get embeddings
                tokenizer (AutoTokenizer): the tokenizer to preprocess the text
                pooling (bool): a flag for mean pooling (to the model’s hidden states)

            returns:
                torch.Tensor: the final embedding of the input text
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
  
    results = {}
    for base, desc, classifier in [
        ('NeuML/pubmedbert-base-embeddings',
         'pubmed-knn-pipeline',
         './models/pubmed-knn-pipeline.joblib'
         ),
        ('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
         'biomed-svc-pipeline',
         './models/biomed-svc-pipeline.joblib'
         )
    ]:
        print(f"\nloading {base}...")
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModel.from_pretrained(base)
        model = model.to(device)
        print(f"loading {desc}...")
        pipeline = joblib.load(classifier)
        embeddings = list()
        for idx, row in dataset.iterrows():
            percentage = (idx + 1) / len(dataset) * 100
            sys.stdout.write(f"\rencoding data... {percentage:.2f}%")
            embeddings.append(get(row, base, model, tokenizer))
        print()
        remove(model)
        embeddings = np.vstack(embeddings)
        print(f"computing predictions...")
        preds = pipeline.predict_proba(embeddings)
        results[base] = preds

    print(f"\nprocessing soft majority vote...")
    alpha, beta = results['NeuML/pubmedbert-base-embeddings'], results['microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract']
    a, b = weights
    probs = (alpha * a) + (beta * b)
    preds = (probs[:, 1] >= threshold).astype(int)
    dataset['PPS'] = preds
    return dataset

# streamlit interface
st.title("High Precision Binary Classifier: Patient Preferences Study")

# file upload or manual entry
st.sidebar.header("Select input type")
option = st.sidebar.selectbox("Input Method", ["CSV File", "Manual Text"])

if option == "CSV File":
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file is not None:
        input = pd.read_csv(file)
      
        # data normalization
        if 'title' in input.columns and 'abstract' in input.columns:
            input['title'] = input['title'].str.lower()
            input['abstract'] = input['abstract'].str.lower()
        else:
            st.error("Please enter a file with 'title' and 'abstract' columns.")
            st.stop()
          
        st.write("Dataset Preview:")
        st.dataframe(input.head())
        if st.button("Run the classifier"):
            try:
                output = predict(input)
                st.success("Done!")
                st.dataframe(output)
                csv = output.to_csv(index=False).encode('utf-8')
                st.download_button(
                  "Download results", 
                  data=csv, 
                  file_name="results.csv", 
                  mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error: {e}")

elif option == "Manual Text":
    title = st.text_input("Enter the title:")
    abstract = st.text_area("Enter the abstract:")
    if st.button("Run the classifier"):
        if title and abstract:
            input = pd.DataFrame({"title": [title.lower()], "abstract": [abstract.lower()]})
            try:
                output = predict(input)
                st.success("Done!")
                st.write(output)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please enter both the title and abstract.")

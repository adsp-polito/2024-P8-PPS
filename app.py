import os
import re
import gc
import time
import torch
import joblib

import numpy as np
import pandas as pd
import streamlit as st

from io import StringIO
from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoModel

np.random.seed(42)
torch.manual_seed(42)


def predict(
    dataset: pd.DataFrame,
    threshold: float = 0.3875,
    weights: Tuple[float, float] = (0.4375, 0.5625)
) -> pd.DataFrame:
    """
    evaluates the relevance of academic papers to patient preference studies
    by predictions from machine learning models and embeddings from transformer architectures

    parameters:
        dataset (pd.DataFrame): input dataset with titles and abstracts
        threshold (float): decision threshold for label assignment
        weights (tuple): weights for the two classification models in vote

    returns:
        pd.DataFrame: original dataset with an additional column 'PPS' 
                      with binary values: relevant (1) or not relevant (0)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def remove(model: AutoModel) -> None:
        """ 
        removes the model from memory 
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
        computes sn embedding for a single paper by title and abstract

        parameters:
            row (pd.Series): row from the dataset with paper's title and abstract
            base (str): model architecture identifier
            model (AutoModel): pre-trained transformer model
            tokenizer (AutoTokenizer): tokenizer for the transformer model

        returns:
            np.ndarray: numerical embedding of the paper
        """
        title = [str(row['title']) if isinstance(row['title'], str) else '.']
        abstract = [str(row['abstract']) if isinstance(row['abstract'], str) else '.']

        def meanpooling(
            output: Tuple[torch.Tensor, ...],
            mask: torch.Tensor
        ) -> torch.Tensor:
            """
            computes mean pooling on token-level embeddings through the attention mask

            parameters:
                output (tuple): model output with token embeddings
                mask (torch.Tensor): attention mask for input tokens

            returns:
                torch.Tensor: sentence-level embedding
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
                text (list): list of strings
                tokenizer (AutoTokenizer): tokenizer for the transformer model

            returns:
                dict: token-level input, ready for embedding model
            """
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            return {key: value.to(device) for key, value in inputs.items()}

        def encode(
            text: List[str],
            model: AutoModel,
            tokenizer: AutoTokenizer,
            pooling: bool
        ) -> torch.Tensor:
            """
            encodes input text into embeddings

            parameters:
                text (list): input text (e.g., title or abstract)
                model (AutoModel): transformer model for embeddings
                tokenizer (AutoTokenizer): tokenizer for text
                pooling (bool): flag to compute mean pooling to embeddings

            returns:
                torch.Tensor: input text embeddings
            """
            inputs = tokenize(text, tokenizer)
            with torch.no_grad():
                output = model(**inputs)
            return (
                output.pooler_output if not pooling
                else meanpooling(output, inputs['attention_mask'])
            )

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
    max = len(dataset) * 2
    bar = st.progress(0)
    status = st.empty()
    progress = 0
    params = [
        (
            'NeuML/pubmedbert-base-embeddings',
            './models/pubmed-knn-pipeline.joblib'
        ),
        (
            'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
            './models/biomed-svc-pipeline.joblib'
        )
    ]
    start = time.time()
    
    for base, classifier in params:
        tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModel.from_pretrained(base).to(device)
        pipeline = joblib.load(classifier)

        embeddings = list()
        for _, row in dataset.iterrows():
            embeddings.append(get(row, base, model, tokenizer))
            progress += 1
            ratio = progress / max
            iter = (time.time() - start) / progress
            left = iter * (max - progress)

            bar.progress(int(ratio * 100))
            if left < 120:
                status.write(f"Less than a minute left...")
            else:
                minutes = int(left // 60)
                status.write(f"About {minutes} minutes left...")

        remove(model)
        embeddings = np.vstack(embeddings)
        preds = pipeline.predict_proba(embeddings)
        results[base] = preds

    bar.empty()
    status.empty()
    
    alpha, beta = results.values()
    a, b = weights
    probs = (alpha * a) + (beta * b)
    preds = (probs[:, 1] >= threshold).astype(int)
    dataset['PPS'] = preds
    
    return dataset


def parse(
    file: Union[str, bytes, os.PathLike]
) -> pd.DataFrame:
    """
    parses a file.txt into a DataFrame

    parameters:
        file: file.txt (str, bytes, os.PathLike) or UploadedFile (from Streamlit)

    returns:
        pd.DataFrame: DataFrame with relevant columns
    """
    if isinstance(file, bytes):
        content = file.decode("utf-8")
    else:
        content = file.read().decode("utf-8")

    def get(index: str, record: str) -> str:
        match = re.search(f"{index}" + r"\s+-\s+(.*?)(?=\n[A-Z]{2,}\s+-|\Z)", record, re.DOTALL)
        if match:
            text = match.group(1).strip()
            return re.sub(r"\s{2,}", " ", text.replace("\n", " "))
        return None
        
    records = re.split(r"\nPMID-\s+", content)
    if records[0].strip() == "":
        records = records[1:]

    data = {
        "Title": [],
        "Authors": [],
        "Abstract": [],
        "Published Year": [],
        "Published Month": [],
        "Journal": [],
        "Volume": [],
        "Issue": [],
        "Pages": [],
        "Accession Number": [],
        "DOI": [],
        "Ref": [],
        "Tags": [],
    }

    for record in records:
        record = f"PMID- {record.strip()}"

        ref = re.search(
            r"PMID-\s+(\d+)", record
        ).group(1).strip() if re.search(r"PMID-\s+(\d+)", record) else None
        
        title = get("TI", record)

        authors = ", ".join([
            fau.strip() for fau in re.findall(r"FAU\s+-\s+(.*)", record)
        ]) if re.findall(r"FAU\s+-\s+(.*)", record) else None
        
        abstract = get("AB", record)

        date = get("DP", record)
        if date:
            year = date.split()[0] if len(date.split()) > 0 else None
            month = date.split()[1] if len(date.split()) > 1 else None
        else:
            year, month = None, None

        journal = get("TA", record)

        volume = get("VI", record)

        issue = get("IP", record)
        
        pages = get("PG", record)
    
        accession = re.findall(
            r"AID\s+-\s+(.*?)(?=\s|\Z)", record
        )[0] if re.findall(r"AID\s+-\s+(.*?)(?=\s|\Z)", record) else None

        doi = get("DOI", record)
        if not doi:
            so = get("SO", record)
            if so and "doi:" in so.lower():
                doi = re.search(r"doi:\s*(\S+)", so, re.IGNORECASE)
                doi = doi.group(1) if doi else None
                
        tags = ", ".join([
            mh.strip() for mh in re.findall(r"MH\s+-\s+(.*)", record)
        ]) if re.findall(r"MH\s+-\s+(.*)", record) else None
        
        data["Title"].append(title)
        data["Authors"].append(authors)
        data["Abstract"].append(abstract)
        data["Published Year"].append(year)
        data["Published Month"].append(month)
        data["Journal"].append(journal)
        data["Volume"].append(volume)
        data["Issue"].append(issue)
        data["Pages"].append(pages)
        data["Accession Number"].append(accession)
        data["DOI"].append(doi)
        data["Ref"].append(ref)
        data["Tags"].append(tags)
    
    return pd.DataFrame(data)


# streamlit interface
st.title("High-Precision Binary Classifier: Patient Preference Study or Not?")
x, y, z = st.columns([1, 2, 1])
with y:
    st.image("docs/logo.png", width=350)

st.sidebar.header("Input Options")
option = st.sidebar.selectbox(
    "Select Input Method", [
        "TXT File (PubMed)", 
        "CSV File", 
        "Title and Abstract"
    ]
)

if option == "TXT File (PubMed)":
    st.sidebar.subheader("How to Download PubMed Format")
    st.sidebar.write("""
    1. Visit [PubMed](https://pubmed.ncbi.nlm.nih.gov/).
    2. Search for articles with a keyword query.
    3. Click the "Save" button at the top right of the results page.
    4. Under "Format," select "PubMed."
    5. Click "Create File" to download the results in a `.txt` file.
    6. Upload the `.txt` file here .
    """)
    file = st.file_uploader("📄 Upload TXT File (PubMed)", type=["txt"])
    
    if file:
        try:
            input = parse(file)
            st.write("Dataset Preview:")
            st.dataframe(input.head())

            originals = {
                "title": input.columns[input.columns.str.lower().str.contains("title")][0],
                "abstract": input.columns[input.columns.str.lower().str.contains("abstract")][0]
            }
            input.rename(columns={
                originals['title']: 'title',
                originals['abstract']: 'abstract'
            }, inplace=True)
            
            if st.button("🤖 Run Classifier"):
                try:
                    output = predict(input)
                    output.rename(columns={
                        'title': originals['title'],
                        'abstract': originals['abstract']
                    }, inplace=True)
                    st.dataframe(output)
                    csv = output.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                      "👾 Download", 
                      data=csv, 
                      file_name="results.csv", 
                      mime="text/csv"
                    )
                except Exception as e: 
                    st.error(f"⚠️ Uh-oh! Something's wrong: {e} 🛠️")
        except Exception as e:
            st.error(f"⚠️ Uh-oh! Something's wrong: {e} 🛠️")

elif option == "CSV File":
    file = st.file_uploader("📄 Upload CSV File", type=["csv"])
    
    if file:
        input = pd.read_csv(file)
        st.write("Dataset Preview:")
        st.dataframe(input.head())
        
        originals = {
            "title": input.columns[input.columns.str.lower().str.contains("title")][0],
            "abstract": input.columns[input.columns.str.lower().str.contains("abstract")][0]
        }
        input.rename(columns={
            originals['title']: 'title',
            originals['abstract']: 'abstract'
        }, inplace=True)
            
        if st.button("🤖 Run Classifier"):
            try:
                output = predict(input)
                output.rename(columns={
                    'title': originals['title'],
                    'abstract': originals['abstract']
                }, inplace=True)
                st.dataframe(output)
                csv = output.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                  "👾 Download", 
                  data=csv, 
                  file_name="results.csv", 
                  mime="text/csv"
                )
            except Exception as e: 
                st.error(f"⚠️ Uh-oh! Something's wrong: {e} 🛠️")
            
elif option == "Title and Abstract":
    title = st.text_input("📄 Enter the Title:")
    abstract = st.text_area("📄 Enter the Abstract:")
    
    if st.button("🤖 Run Classifier"):
        if title and abstract:
            input = pd.DataFrame({"title": [title], "abstract": [abstract]})
            try:
                output = predict(input)
                label = output['PPS'].iloc[0]
                if label == 1:
                    st.success("**Prediction:** **Relevant** ✅")
                else:
                    st.error("**Prediction:** **Non-Relevant** ❌")
            except Exception as e: 
                st.error(f"⚠️ Uh-oh! Something's wrong: {e} 🛠️")
        else:
            st.error("⚠️ Please enter both the title and abstract to let me work my magic. 🪄")

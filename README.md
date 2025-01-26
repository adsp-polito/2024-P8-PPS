# 2024-P8-PPS

# Multilabel and binary classification to filter Patient preference Studies

This study aimed to develop a robust multilabel classification system capable of identifying and categorizing relevant research papers using only their titles and abstracts. The proposed framework consists of a binary classification model that filters Patient Preference Studies (PPS) from the main dataset, followed by a multilabel classifier that assigns relevant labels to PPS.

The binary classification model, implemented as an ensemble of PubMedBERT-kNN and BioMedBERT-SVM classifiers, minimizes the loss of relevant papers through majority voting. For the multilabel classification task, Binary Relevance (BR) with an SVC classifier demonstrated the best performance, achieving the lowest Hamming loss and highest F1-score. The RAkEL model with an mnNB classifier also emerged as a strong candidate, effectively maintaining label correlation. Additionally, the BERTopic model facilitated the discovery of new and meaningful labels, identifying three for clinical areas and two for intervention datasets, significantly reducing manual effort and domain expertise requirements.


---

## Project Structure

The repository is organized into the following main components:

### 1. **Binary Classification Model**
   - Path: [`PPS-BC/`](binary_classification/)
   - Description: This folder contains the implementation of the binary classification model used to filter PPS from the main dataset. The model combines the outputs of PubMedBERT-kNN and BioMedBERT-SVM classifiers through a majority voting mechanism to ensure minimal loss of relevant papers.

### 2. **Multilabel Classification Model**
   - Path: [`PPS-MLC/`](multilabel_classification/)
   - Description: This folder includes the implementation of the multilabel classification system for assigning relevant labels to PPS. It features models such as Binary Relevance with SVC and RAkEL with mnNB, as well as an exploration of BERTopic for identifying additional labels.

---

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/adsp-polito/2024-P8-PPS.git
   cd 2024-P8-PPS

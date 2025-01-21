import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import hamming_loss
from sklearn.model_selection import KFold
from itertools import product
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.ensemble import RakelD
from IPython.display import display

# CLASSIFICATION FUNCTIONS
def show_stats_of_titles_abstracts(data):
  data_stats=pd.DataFrame(columns=['Title Length', 'Abstract Length'])
  data_stats['Title Length'] = data['Title'].astype(str).apply(lambda x: len(x.split()))
  data_stats['Abstract Length'] = data['Abstract'].astype(str).apply(lambda x: len(x.split()))

  print(f"Average Title Length: {data_stats['Title Length'].mean():.2f} words")
  print(f"Average Abstract Length: {data_stats['Abstract Length'].mean():.2f} words")

  data_stats['Title Issues'] = data['Title'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)
  data_stats['Abstract Issues'] = data['Abstract'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)

  print(f"Number of rows with Anomalies in the Title: {data_stats[~data_stats['Title Issues']].shape[0]}")
  print(f"Number of rows with Anomalies in the Abstract: {data_stats[~data_stats['Abstract Issues']].shape[0]}")
  return data_stats

def show_distributions(data,column,col='#EE1A39'):
  '''
  col='#EE1A39' red
  col='#002A5C' grey
  column='Title Length'
  column='Abstract Length'
  '''
  unito = "#EE1A39"
  polito = "#002A5C"
  plt.figure(figsize=(12, 6))
  sns.histplot(data[column],
              kde=True,
              color=col,
              bins=30)
  plt.axvline(data[column].mean(),
              color=col,
              linestyle='dashed',
              linewidth=1.5,
              label='Mean'+ column)
  plt.title('Distribution of ' + column)
  plt.xlabel('Length (Word Count)')
  plt.ylabel('Frequency')
  plt.legend()
  plt.show()
  return

def separate_labeled_and_not_labeled_data(df,df_name):
    #1. Filter out not classified data
    labeled_data = df[df.iloc[:, 2:].sum(axis=1) > 0]
    not_labeled_data = df[df.iloc[:, 2:].sum(axis=1) == 0]
    print(f"{df_name} Labeled data: {labeled_data.shape[0]} rows")
    print(f"{df_name} Not labeled data: {not_labeled_data.shape[0]} rows")
    return labeled_data, not_labeled_data

def embedd(data,model):
  embedding_model= SentenceTransformer(model)
  text=pd.Series(data['Title'] + ' ' + data['Abstract'])

  # Individuare valori NaN
  is_nan = text.isna()

  # Individuare valori vuoti (NaN o stringhe vuote "")
  is_empty = text.isna() | (text == "")

  # Sommare i NaN
  nan_count = is_nan.sum()

  # Sommare i vuoti (NaN + stringhe vuote)
  empty_count = is_empty.sum()

  print("Count NaN values: ", nan_count)
  print("Count empty values (NaN o empty strings):", empty_count)
    
  # Sostituire NaN e stringhe vuote con " "
  text = text.fillna(" ")  # Sostituire NaN
  text[text == ""] = " "  # Sostituire stringhe vuote
  embeddings = embedding_model.encode(text.tolist(), show_progress_bar=True)
  return text,embeddings

def remove_rare_label_combinations(dataset):
    """
    Removes rows from a DataFrame where the label combinations (rows) occur only once.
    
    Args:
        data (pd.DataFrame): Input multi-label dataset (only label columns should be passed).
        
    Returns:
        pd.DataFrame: Filtered DataFrame with rare label combinations removed.
    """
    data=dataset.copy()
    # Combine labels into a single string to represent unique combinations
    data['label_combination'] = data.iloc[:,2:].astype(str).agg('-'.join, axis=1)
    
    # Count occurrences of each combination
    combination_counts = data['label_combination'].value_counts()
    
    # Filter out rows with combinations that occur only once
    valid_combinations = combination_counts[combination_counts > 1].index
    unique_combinations = combination_counts[combination_counts == 1].index
    unique_data=data[data['label_combination'].isin(unique_combinations)].drop(columns=['label_combination'])
    filtered_data = data[data['label_combination'].isin(valid_combinations)].drop(columns=['label_combination'])
    data.drop(columns=['label_combination'])
    print(f"Valid combinations (# istances > 1): {filtered_data.shape[0]}/{data.shape[0]}")
    print(f"Count of combinations with just 1 instance: {unique_data.shape[0]}")
    print("combinations not considered: ")
    
    display(unique_data)
    return filtered_data

def one_error(y_true, y_pred):
    """
    Calculate the one-error metric for multilabel classification.
    
    Args:
        y_true (np.ndarray): Binary matrix of true labels (shape: [n_samples, n_labels]).
        y_pred (np.ndarray): Predicted scores for each label (shape: [n_samples, n_labels]).
        
    Returns:
        float: One-error metric (lower is better).
    """
    # Ensure y_true is binary
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Find the index of the top-ranked predicted label for each instance
    top_pred_indices = np.argmax(y_pred, axis=1)
    
    # Check if the top-predicted label is not in the true labels
    one_error_count = sum(y_true[i, top_pred_indices[i]] == 0 for i in range(len(y_true)))
    
    # Compute one-error as the fraction of incorrect top predictions
    return one_error_count / len(y_true)

def get_ml_metrics(y_test,predicted,y_prob=np.nan,verbose=0):

    '''
    - The Hamming Loss measures the fraction of labels that are incorrectly predicted, either by being missed or being assigned wrongly. It considers each label independently.
    Lower values indicate better performance. A value of 0 means perfect prediction.
    
    - Accuracy score: This metric measures the proportion of samples where the set of predicted labels exactly matches the set of true labels.
    Higher values indicate better performance. Perfect prediction has an accuracy score of 1.

    - Ranking Loss measures the proportion of label pairs that are incorrectly ordered. A pair is considered incorrectly ordered if the model assigns a higher score to an irrelevant label than to a relevant one.
    Lower values indicate better performance. A ranking loss of 0 means perfect ranking.

    - One-Error measures whether the top-ranked label (the one with the highest predicted score) is not in the set of true labels.
    Lower values indicate better performance. A value of 0 means the top-ranked label is always correct.

    - Coverage Error measures how far we need to go down the ranked list of labels to cover all true labels for a sample.
    Lower values indicate better performance. A value of 0 means all true labels are ranked at the top.

    Comparison and Use Cases:
        Hamming Loss: Focuses on individual label errors. Good for understanding overall prediction quality.
        Accuracy Score: Strict metric; only perfect matches count.
        Ranking Loss: Evaluates ranking quality of relevant vs. irrelevant labels.
        One-Error: Useful when the most important label must be ranked highest.
        Coverage Error: Indicates how well the model ranks all true labels. Useful for ranking applications.
    '''


    from sklearn.metrics import f1_score,hamming_loss, accuracy_score, label_ranking_loss, coverage_error
    f1_weighted=f1_score(y_test, predicted,average='weighted') 
    f1_micro=f1_score(y_test, predicted,average='micro') 
    #f1_macro=accuracy_score(y_test, predicted) 

    h_loss=hamming_loss(y_test, predicted)
    if (not(np.any(np.isnan(y_prob)))): 
        r_loss=label_ranking_loss(y_test, y_prob)
        c_error=coverage_error(y_test, y_prob)
        One_error=one_error(y_test, y_prob)
        
    else:
        r_loss=label_ranking_loss(y_test, predicted)
        c_error=coverage_error(y_test, predicted)
        One_error=one_error(y_test, predicted)


    if verbose:
        print(f"f1-micro: {f1_micro:.3f}") 
        print(f"f1-weighted: {f1_weighted:.3f}") 
        print(f"hamming loss: {h_loss:.3f}")
        print(f"ranking loss: {r_loss:.3f}")
        print(f"coverage error: {c_error:.3f}")
        print(f"one error: {One_error:.3f}")
    
    return f1_micro,f1_weighted,h_loss,r_loss,c_error,One_error

def perform_grid_search(model_name,model_class, X_train, y_train, param_grid, kfold_splits, results,verbose=False,output_file=None):
    """
    Perform grid search for a given model and parameter grid using K-Fold Cross Validation.
    
    Parameters:
    - model_class: The class of the model to instantiate and train (e.g., MLkNN).
    - X_train: Training data features (numpy array).
    - y_train: Training data labels (numpy array or sparse matrix).
    - param_grid: Dictionary with parameter names as keys and lists of values to try as values.
    - kfold_splits: List of integers for the number of KFold splits to evaluate.
    - output_file: Optional file path to save results as an Excel file.
    
    Returns:
    - A DataFrame with all results and the best parameters found.
    """
    

    # Extract parameter ranges from the grid
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Loop over different KFold splits
    for splits in kfold_splits:
        print(f"Performing parameter search with {splits}-fold Cross Validation")
        
        # Define KFold
        cv = KFold(n_splits=splits, shuffle=True, random_state=42)
        
        # Iterate over all combinations of parameters
        for param_combination in product(*param_values):
            param_dict = dict(zip(param_keys, param_combination))
            fold_scores = []
            
            # Perform cross-validation
            for train_idx, test_idx in cv.split(X_train):
                X_train_, X_test_ = X_train[train_idx], X_train[test_idx]
                y_train_, y_test_ = y_train[train_idx], y_train[test_idx]
                
                # Train the model
                if model_name.startswith('BR_'):
                    model = BinaryRelevance(classifier=model_class(**param_dict), require_dense=[True, True])
                elif model_name.startswith('RAkEL_'):
                    model = RakelD(base_classifier=model_class(**param_dict),base_classifier_require_dense=[True, True],
                                    labelset_size=6) #not iterate over labelsetsize_ get best items from previous analysis: 6: clinical areas - 3,7 per interventions
                    
                else:
                    model = model_class(**param_dict)
                
                model.fit(X_train_, y_train_)
                
                # Predict and calculate Hamming Loss
                if (model_name=='VWMLkNN') or (model_name=='MLTSVM'):
                    y_pred = model.predict(X_test_)
                else:
                    y_pred = model.predict(X_test_).toarray()
                score = hamming_loss(y_test_, y_pred)
                fold_scores.append(score)
            
            # Average score across folds
            avg_score = np.mean(fold_scores)
            
            # Save results
            results.append([(model_name,splits, *param_combination), avg_score])
            if verbose:
                print(f"Split: {splits}, Params: {param_dict}, Hamming Loss: {avg_score:.4f}")
    
    # Create a DataFrame from results
    result_columns = ['params','Hamming Loss']#['cv split'] + param_keys + ['Hamming Loss']
    results_df = pd.DataFrame(results, columns=result_columns)
    
    # Find the best parameters
    best_result = results_df.loc[results_df['Hamming Loss'].idxmin()]
    print(f"\n{model_name} - Best Parameters ({param_grid.keys}) and Score:")
    print(best_result)
    
    # Save results to Excel if specified
    if output_file:
        results_df.to_excel(output_file, index=False)
    
    return results_df, best_result



##VISUALIZATION & IMBALANCE FUNCTIONS

def plotpiechart(dataset):
  labels_sum = dataset.iloc[:, 2:].sum()
  labels_sum['Not Classified'] = (dataset.iloc[:, 2:].sum(axis=1) == 0).sum()
  if labels_sum.sum() == 0:
      print("No labels to plot.")
      return
  
  plt.figure(figsize=(8, 8))
  plt.pie(
      labels_sum,
      labels=labels_sum.index,
      autopct='%1.1f%%',
      startangle=140
  )
  plt.title('Distribution of Categories')
  plt.show()

def addNotClassified(dataset):
    data=dataset.copy()
    data["Not Classified"] = dataset.iloc[:, 2:].sum(axis=1).apply(lambda x: 1 if x == 0 else 0)
    return data

def plot_imbalance_within_labels(dataset,category):
      # Sum values in each column from the third column to the end
    column_sums = dataset.iloc[:, 2:].sum()

    # Order columns by their sum
    column_sums = column_sums.sort_values(ascending=False)

    df=pd.DataFrame(column_sums.T)
    df.columns = ['Positive']
    df.index.name = category
    df['Negative'] = dataset.shape[0] - df['Positive']
    df.plot.bar(figsize=(10, 6))

    # Add labels and title
    plt.title("1. Imbalance within labels (sorted)", fontsize=14)
    plt.ylabel("Frequency", fontsize=12)
    plt.xlabel(category, fontsize=12)

def plot_imbalance_between_labels(dataset,category):

    # Sum values in each column from the third column to the end
    column_sums = dataset.iloc[:, 2:].sum()

    # Order columns by their sum
    column_sums = column_sums.sort_values(ascending=False)

    # Plot the results as a barplot
    plt.figure(figsize=(10, 6))
    column_sums.plot(kind="bar", color='skyblue')

    # Add labels and title
    plt.title("2. Imbalance between labels (sorted)", fontsize=14)
    plt.ylabel("Frequency", fontsize=12)
    plt.xlabel(category, fontsize=12)
    plt.xticks(rotation=90)  # Rotate labels vertically
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()

def show_imbalance_on_label_sets(df):

    # Assuming you have the dataset in a DataFrame `df`

    # Step 1: Extract labels from the third column onwards
    labels_df = df.iloc[:, 2:]

    # Step 2: Convert the labels to sets
    label_sets = []

    for _, row in labels_df.iterrows():
        # Extract the indices where the label is 1 (present)
        label_set = set(labels_df.columns[row == 1])
        label_sets.append(frozenset(label_set))  # Use frozenset to ensure uniqueness

    # Step 3: Count the frequency of each label set
    label_set_freq = pd.Series(label_sets).value_counts().reset_index()

    # Rename columns for better understanding
    label_set_freq.columns = ['Label Set', 'Frequency']

    # Step 4: Sort by frequency
    label_set_freq_sorted = label_set_freq.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

    # Display the result

    print(label_set_freq_sorted)
    return label_set_freq_sorted

def show_simultaneous_labels(dataset, n_labels):
    """
    Adds `n_labels` columns to the dataset based on the sum of horizontal values
    from the third column onward. Then plots the sum of these columns.

    - If the row sum is 0, column '0' = 1 and others = 0.
    - If the row sum is 1, column '1' = 1 and others = 0, etc.
    """
    # Calculate the row-wise sum from the third column onward
    row_sums = dataset.iloc[:, 2:].sum(axis=1)
    added_cols=len(dataset.columns[2:])
    # Add n_labels columns (0, 1, 2, ..., n_labels-1)
    for i in range(n_labels):
        dataset[str(i)] = (row_sums == i).astype(int)

    # Sum the values of the newly added columns
    column_sums = dataset.iloc[:, -n_labels:].sum()

    # Plot the results as a barplot
    plt.figure(figsize=(10, 6))
    column_sums.plot(kind="bar")

    # Add labels and title
    plt.title("Inter label count", fontsize=14)
    plt.ylabel("Sum", fontsize=12)
    plt.xlabel("Labels", fontsize=12)
    plt.xticks(rotation=0)  # Keep labels horizontal
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    dataset = dataset.iloc[:, :-added_cols]
    return dataset

def show_imbalance_on(dataset,categoryNames):
  '''
  dataset :
  '''
  data=addNotClassified(dataset)
  plot_imbalance_within_labels(data,categoryNames)
  plot_imbalance_between_labels(data,categoryNames)
  print(f"\nDataset of {categoryNames}\nImbalance among labelsets:\n")
  label_sets=show_imbalance_on_label_sets(data)
  return label_sets

def find_imbalance_level_measures(dataset):

  '''
  dataset: built with just labels
  '''
  labels=dataset.iloc[:, 2:]
  IRLbl_df=IRLbl(labels)
  meanIR=IRLbl_df.mean()
  maxIR=IRLbl_df.max()
  CVIR=IRLbl_df.var()/meanIR
  scumble_index=calculate_scumble(labels)
  #Card,Density,TCS=calculate_other_metrics(labels,labelset)

  print("\nImbalance Ratios:")
  print(IRLbl_df.sort_values(ascending=False))
  print(f"Mean imbalance ratio: {meanIR:.3f}")
  print(f"Max imbalance ratio: {maxIR:.3f}")
  print(f"Coefficient of variation of imbalance ratio: {CVIR:.3f}")

  return IRLbl_df,meanIR,maxIR,CVIR,scumble_index

def calculate_other_metrics(df,ls):
  import math
  #df just labels
  # Calculate TCS (Total Class Size)
  totalclasssize = df.sum().sum()

  # Number of instances and labels
  num_instances, num_labels = df.shape

  # Calculate Cardinality
  cardinality = totalclasssize / num_instances

  # Calculate Density
  density = cardinality / num_labels
  f=768 #input features from embeddings (otherwise 2 - title and abstract, can put also the avg number of words) to discuss
  TCS=math.log(f*num_labels*(ls.shape[0]-1)) #manage not classified label
  print(f"Card: {cardinality:.3f}")
  print(f"Density: {density:.3f}")
  print(f"TCS: {TCS:.3f}")
  #return cardinality,density,TCS

  return

def IRLbl(dataset):
  '''
  datset: built with just labels
  '''

  # Total number of samples
  total_samples = dataset.shape[0]

  # Count the number of samples per label
  label_counts = dataset.sum(axis=0)
  majority_label=max(label_counts)
  # Calculate imbalance ratio per label
  #imbalance_ratios = total_samples / label_counts
  imbalance_ratios = majority_label / label_counts

  '''
  # Display results
  print("Label Counts:")
  print(label_counts.sort_values(ascending=False))
  print("\nImbalance Ratios:")
  print(imbalance_ratios.sort_values(ascending=False))
  '''
  return imbalance_ratios

def calculate_scumble(dataset):
    """
    Calculate the SCUMBLE index for a multi-label dataset.

    Parameters:
        dataset (numpy array or pandas DataFrame): A binary matrix where rows are samples
                                                   and columns are labels (1 for label presence, 0 otherwise).

    Returns:
        float: The SCUMBLE index value.
    """
    # Convert the dataset to a binary matrix
    if isinstance(dataset, np.ndarray):
        binary_matrix = dataset
    else:
        binary_matrix = dataset.values  # Convert from pandas DataFrame to numpy array

    # Number of labels (columns) and samples (rows)
    num_samples, num_labels = binary_matrix.shape

    # Calculate the frequency of each label
    label_frequencies = binary_matrix.sum(axis=0) / num_samples

    # Avoid division by zero
    label_frequencies = np.where(label_frequencies == 0, np.finfo(float).eps, label_frequencies)

    # Calculate imbalance per sample
    imbalance_per_sample = []
    for row in binary_matrix:
        present_labels = np.where(row == 1)[0]
        if len(present_labels) > 0:
            # SCUMBLE for the row is based on the frequencies of the labels present
            row_imbalance = np.sum(1 - label_frequencies[present_labels]) / len(present_labels)
            imbalance_per_sample.append(row_imbalance)
        else:
            # No labels present, imbalance is 0
            imbalance_per_sample.append(0)

    # Calculate the overall SCUMBLE index
    scumble_index = np.mean(imbalance_per_sample)
    return scumble_index

# EMBEDDINGS
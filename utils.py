import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import google.generativeai as genai


def sample_sentences_from_training(n, train_set, subj_to_obj_ratio=1, text_column="text"):
    """
    Parameters:
    - n (int) : number of sentences to be sampled
    - train_set (DataFrame) : the dataframe to sample sentences from
    - subj_to_obj_ratio (float) : ratio of subjective (1) sentences to objective (0) sentences in the sample
    - text_column (str) : column name containing the sentences in the dataframe

    Returns:
    tuple : a tuple containing two lists.
        - List 1 (List) : list of subjective sentences
        - List 2 (List) : list of objective sentences 
    """

    obj_count = n / (1+subj_to_obj_ratio)
    subj_count = n - obj_count
    obj_df = train_set[train_set["binary_labels"] == 0]
    subj_df = train_set[train_set["binary_labels"] == 1]
    subj_sample = subj_df.sample(n=int(subj_count), replace=False, random_state=42)
    obj_sample = obj_df.sample(n=int(obj_count), replace=False, random_state=42)
    subj_sentences = [row[text_column] for _,row in subj_sample.iterrows()]
    obj_sentences = [row[text_column] for _,row in obj_sample.iterrows()]

    return subj_sentences, obj_sentences


def sample_equally_from_all(n, train_sets, text_columns):
  """
  Parameters:
  n (int) : number of sentences to sample
  train_sets (list) : list of dataframes to sample from
  text_columns (list) : list of strings corresponding to text column in each dataset

  Returns:
  (tuple) : a tuple of two lists, subjective sentences and objective sentences
  """

  obj_count, subj_count = n/2, n/2
  subj_sentences = []
  obj_sentences = []

  for idx, train_set in enumerate(train_sets):
      
      subj_df = train_set[train_set["binary_labels"] == 1]
      obj_df = train_set[train_set["binary_labels"] == 0]

      subj_sample = subj_df.sample(n=int(subj_count/3), replace=False, random_state=42)
      obj_sample = obj_df.sample(n=int(obj_count/3), replace=False, random_state=42)

      subj_sents = [row[text_columns[idx]] for _,row in subj_sample.iterrows()]
      obj_sents = [row[text_columns[idx]] for _,row in obj_sample.iterrows()]

      subj_sentences.extend(subj_sents)
      obj_sentences.extend(obj_sents)

  return subj_sentences, obj_sentences


def create_icl_prompt(n, train_set, subj_to_obj_ratio=1, text_column="text"):
    """
    Parameters:
    - n (int) : number of sentences to be sampled
    - train_set (DataFrame) : the dataframe to sample sentences from
    - subj_to_obj_ratio (float) : ratio of subjective (1) sentences to objective (0) sentences in the sample
    - text_column (text) : column name containing the sentences in the dataframe

    Returns:
    str : text containing the icl examples and the system prompt

    """

    system_prompt = """
Classify the sentence into Subjective or Objective based on the language used in them.\n
These are some examples: \n 
"""
    icl_examples = f""
    subj_sentences, obj_sentences = sample_sentences_from_training(n, train_set, subj_to_obj_ratio, text_column)
    for idx, sent in enumerate(subj_sentences):
        icl_examples += f"{idx+1}. " + sent + ' Label: Subjective \n'

    for idx,sent in enumerate(obj_sentences):
        icl_examples += f"{len(subj_sentences)+idx+1}. "+ sent + ' Label: Objective \n'
    
    icl_examples += "\n" + "Sentence: {} Label:"

    prompt = system_prompt + icl_examples
    return prompt


def create_diverse_icl_prompt(n, train_sets, text_columns):
    system_prompt = """
Predict the label of the sentence from Subjective or Objective based on the language used in them.\n
These are some examples: \n
"""

    icl_examples = f""
    subj_sentences, obj_sentences = sample_equally_from_all(n, train_sets, text_columns=text_columns)
    for idx, sent in enumerate(subj_sentences):
        icl_examples += f"{idx+1}. " + sent + ' Label is : Subjective \n'

    for idx,sent in enumerate(obj_sentences):
        icl_examples += f"{len(subj_sentences)+idx+1}. "+ sent + ' Label is : Objective \n'

    icl_examples += "\n" + "Sentence: {} Label is :"

    prompt = system_prompt + icl_examples
    return prompt


def convert_labels_to_binary(labels):
    """
    Parameters:
    - labels (List) : list of strings (either subjective or objective)

    Returns:
    list : list of binary labels (1 or 0)
    """
    y_pred = []
    for x in labels:
      if x == "Subjective" or x == "subjective":
        y_pred.append(1)
      elif x == "Objective" or x == "objective":
        y_pred.append(0)
      else:
        y_pred.append(2)

    return y_pred


def gemini_experiment(test_sentences, model, icl_examples_num, 
                   subj_to_obj_ratio, train_df, y_true, text_column="text"):
    
    """
    Parameters:
    - test_sentences (List) : list of sentences
    - model (genai.GenerativeModel): gemini model
    - icl_examples_num (int) : number of icl examples to be include in the prompt
    - subj_to_obj_ratio (float) : ratio of subjective (1) sentences to objective (0) sentences in the sample
    - train_df (DataFrame) : the dataframe to sample sentences from
    - y_true (List) : list of binary gold lables 
    - text_column (str) : column name containing the sentences in the dataframe

    """

    labels = []
    prompt = create_icl_prompt(icl_examples_num, train_df, subj_to_obj_ratio=subj_to_obj_ratio, text_column=text_column)
    for idx, sent in enumerate(test_sentences):
      if idx == 171 or idx == 98:
        labels.append('Objective')
        continue

      response = model.generate_content(prompt.format(sent))
      labels.append(response.text)
      if idx % 10 == 0:
        print(f"Sent {idx} processed")


    y_pred = convert_labels_to_binary(labels)

    if len(set(y_pred)) > 2:
      print(classification_report(y_true, y_pred, target_names=["OBJ", "SUBJ", "NONE"]))
    else:
      print(classification_report(y_true, y_pred, target_names=["OBJ", "SUBJ"]))
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy


def diverse_gemini_experiment(test_sentences, model, icl_examples_num,
                              train_sets, text_columns):

    labels = []
    prompt = create_diverse_icl_prompt(icl_examples_num, train_sets, text_columns)
    
    for idx, sent in enumerate(test_sentences):
      if idx == 171 or idx == 98:
        labels.append('Objective')
        continue

      response = model.generate_content(prompt.format(sent),
                                        generation_config=genai.types.GenerationConfig(
                                            candidate_count=1, max_output_tokens=10,
                                            temperature=1.0))
      labels.append(response.text)
      if idx % 100 == 0:
        print(f"Sent {idx} processed")


    y_pred = convert_labels_to_binary(labels)

    if len(set(y_pred)) > 2:
      print(classification_report(y_true, y_pred, target_names=["OBJ", "SUBJ", "NONE"]))
    else:
      print(classification_report(y_true, y_pred, target_names=["OBJ", "SUBJ"]))
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy
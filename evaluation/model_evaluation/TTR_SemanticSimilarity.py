# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


# Define file paths
student_file_path = "MisstepMath/Evaluation/AI_model_evaluation/math_problems_dataset.csv"
teacher_file_path = "MisstepMath/Evaluation/AI_model_evaluation/teacher_responses_dataset.csv"

# Load datasets
df_students = pd.read_csv(student_file_path)
df_teachers = pd.read_csv(teacher_file_path)

# Load BERT tokenizer and model
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# Function to compute sentence embeddings using BERT
def get_bert_embedding(text):
    if not isinstance(text, str) or text.strip() == "":
        return None

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Extract CLS token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# Function to compute average pairwise cosine similarity for a set of texts
def calculate_bert_similarity(texts):
    if len(texts) < 2:
        return 0  # No similarity possible with a single sentence

    embeddings = [get_bert_embedding(text) for text in texts if get_bert_embedding(text) is not None]
    if len(embeddings) < 2:
        return 0  # Avoid issues if no valid embeddings

    embeddings_tensor = torch.stack(embeddings).squeeze(1)
    cosine_sim_matrix = F.cosine_similarity(embeddings_tensor.unsqueeze(1), embeddings_tensor.unsqueeze(0), dim=2)

    # Compute average similarity (excluding self-similarity)
    n = cosine_sim_matrix.shape[0]
    avg_similarity = (cosine_sim_matrix.sum() - torch.trace(cosine_sim_matrix)) / (n * (n - 1))
    return avg_similarity.item()

# Function to calculate Type-Token Ratio (TTR)
def calculate_ttr(texts):
    all_words = " ".join(texts).split()
    unique_words = set(all_words)
    return len(unique_words) / len(all_words) if len(all_words) > 0 else 0

# Compute TTR and BERT-based semantic similarity
student_results = []
teacher_results = []

for (grade, topic, sub_topic), group in df_students.groupby(["Grade", "Topic", "Sub-topic"]):
    student_texts = group["Student Mistake Prompt"].dropna().tolist()
    ttr_value = calculate_ttr(student_texts)
    semantic_similarity = calculate_bert_similarity(student_texts)
    student_results.append([grade, topic, sub_topic, ttr_value, semantic_similarity])

for (grade, topic, sub_topic), group in df_teachers.groupby(["Grade", "Topic", "Sub-topic"]):
    teacher_texts = group["Teacher Response"].dropna().tolist()
    ttr_value = calculate_ttr(teacher_texts)
    semantic_similarity = calculate_bert_similarity(teacher_texts)
    teacher_results.append([grade, topic, sub_topic, ttr_value, semantic_similarity])

# Convert results to DataFrames
df_student_results = pd.DataFrame(student_results, columns=["Grade", "Topic", "Sub-topic", "TTR", "Semantic Similarity"])
df_teacher_results = pd.DataFrame(teacher_results, columns=["Grade", "Topic", "Sub-topic", "TTR", "Semantic Similarity"])

# Save results to Google Drive
df_student_results.to_csv("MisstepMath/Evaluation/AI_model_evaluation/student_ttr_bert_results.csv", index=False)
df_teacher_results.to_csv("MisstepMath/Evaluation/AI_model_evaluation/teacher_ttr_bert_results.csv", index=False)

# Visualization: TTR Comparison
plt.figure(figsize=(10,5))
sns.boxplot(data=[df_student_results["TTR"], df_teacher_results["TTR"]], palette=["blue", "green"])
plt.xticks([0, 1], ["Student Model", "Teacher Model"])
plt.ylabel("Type-Token Ratio (TTR)")
plt.title("TTR Comparison Between Student and Teacher Models")
plt.show()

# Visualization: Semantic Similarity Comparison (BERT)
plt.figure(figsize=(10,5))
sns.boxplot(data=[df_student_results["Semantic Similarity"], df_teacher_results["Semantic Similarity"]], palette=["blue", "green"])
plt.xticks([0, 1], ["Student Model", "Teacher Model"])
plt.ylabel("Semantic Similarity Score (BERT)")
plt.title("Semantic Similarity Comparison Using BERT")
plt.show()

# Analysis
print("\n📊 **Data Analysis:**\n")

print(" Average TTR for Student Model:", round(df_student_results["TTR"].mean(), 3))
print(" Average TTR for Teacher Model:", round(df_teacher_results["TTR"].mean(), 3))

print("\n Average Semantic Similarity for Student Model (BERT):", round(df_student_results["Semantic Similarity"].mean(), 3))
print(" Average Semantic Similarity for Teacher Model (BERT):", round(df_teacher_results["Semantic Similarity"].mean(), 3))

if df_student_results["Semantic Similarity"].mean() > df_teacher_results["Semantic Similarity"].mean():
    print("\n Student model responses tend to be more similar across samples.")
else:
    print("\n Teacher model responses show higher variation, indicating diverse instructional feedback.")

print("\n Results saved in Google Drive!")



# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import pandas as pd
import gspread
from google.colab import auth
from google.auth import default
from gspread_dataframe import get_as_dataframe



# Define file paths
student_file_path = "MisstepMath/Dataset/AIED_dataset.csv"

# Load datasets
df_students = pd.read_csv(student_file_path)


# Load BERT tokenizer and model
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# Function to compute sentence embeddings using BERT
def get_bert_embedding(text):
    if not isinstance(text, str) or text.strip() == "":
        return None

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Extract CLS token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# Function to compute average pairwise cosine similarity for a set of texts
def calculate_bert_similarity(texts):
    if len(texts) < 2:
        return 0  # No similarity possible with a single sentence

    embeddings = [get_bert_embedding(text) for text in texts if get_bert_embedding(text) is not None]
    if len(embeddings) < 2:
        return 0  # Avoid issues if no valid embeddings

    embeddings_tensor = torch.stack(embeddings).squeeze(1)
    cosine_sim_matrix = F.cosine_similarity(embeddings_tensor.unsqueeze(1), embeddings_tensor.unsqueeze(0), dim=2)

    # Compute average similarity (excluding self-similarity)
    n = cosine_sim_matrix.shape[0]
    avg_similarity = (cosine_sim_matrix.sum() - torch.trace(cosine_sim_matrix)) / (n * (n - 1))
    return avg_similarity.item()

# Function to calculate Type-Token Ratio (TTR)
def calculate_ttr(texts):
    all_words = " ".join(texts).split()
    unique_words = set(all_words)
    return len(unique_words) / len(all_words) if len(all_words) > 0 else 0

# Compute TTR and BERT-based semantic similarity
student_results = []

for (grade, topic, sub_topic), group in df_students.groupby(["Grade", "Topic", "Sub Topic"]):
    student_texts = group["Student's mistake prompt"].dropna().tolist()
    ttr_value = calculate_ttr(student_texts)
    semantic_similarity = calculate_bert_similarity(student_texts)
    student_results.append([grade, topic, sub_topic, ttr_value, semantic_similarity])

# Convert results to DataFrames
df_student_results = pd.DataFrame(student_results, columns=["Grade", "Topic", "Sub Topic", "TTR", "Semantic Similarity"])

# Save results to Google Drive
df_student_results.to_csv("MisstepMath/Evaluation/AI_model_evaluation/AIED_dataset_student_mistake_ttr_bert_results.csv", index=False)


# Analysis
print("\n📊 **Data Analysis:**\n")

print("🔹 Average TTR for Student Model:", round(df_student_results["TTR"].mean(), 3))

print("\n🔹 Average Semantic Similarity for Student Model (BERT):", round(df_student_results["Semantic Similarity"].mean(), 3))


print("\n✅ Results saved in Google Drive!")

"""Student AI model evaluation"""

import pandas as pd
import numpy as np
import nltk
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy
from google.colab import files
import io

nltk.download('punkt')


# Load Spacy model for syntactic variation
nlp = spacy.load("en_core_web_sm")

# Load Sentence Transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define dataset paths for different models
model_datasets = {
    "Student Zeroshot": "MisstepMath/Evaluation/AI_model_evaluation/student_zero_shot_results.csv",
    "Student Finetuned": "MisstepMath/Evaluation/AI_model_evaluation/student_fine_tuned_results.csv",
}

# Required columns
required_columns = ["Grade", "Topic", "Sub-topic", "Challenge Type", "Problem", "Student Mistake Prompt"]

# Function to calculate Entropy Score
def entropy_score(values):
    value_counts = values.value_counts(normalize=True)
    return entropy(value_counts, base=2) if len(value_counts) > 1 else 0

# Function to calculate Evenness Index (Pielou's Evenness)
def evenness_index(values):
    H = entropy_score(values)  # Calculate entropy
    S = values.nunique()       # Number of unique categories
    return H / np.log2(S) if S > 1 else 0  # Normalize by log of unique categories

# Function to calculate Lexical Diversity
def lexical_diversity(texts):
    words = nltk.word_tokenize(" ".join(texts))
    return len(set(words)) / len(words) if words else 0

# Function to calculate Semantic Similarity
def semantic_similarity(texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    if len(embeddings) < 2:
        return 1  # Assume max similarity for single sample
    similarities = [util.pytorch_cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(embeddings)-1)]
    return np.mean(similarities)

# Function to calculate Syntactic Variation
def syntactic_variation(texts):
    sentence_structures = set()
    for text in texts:
        doc = nlp(text)
        sentence_structures.add(" ".join([token.pos_ for token in doc]))
    return len(sentence_structures) / len(texts) if texts else 0

# Function to calculate Type-Token Ratio (TTR)
def calculate_ttr(texts):
    all_words = " ".join(texts).split()
    unique_words = set(all_words)
    return len(unique_words) / len(all_words) if len(all_words) > 0 else 0

from textblob import TextBlob

def sentiment_variation(texts):
    sentiments = [TextBlob(text).sentiment.polarity for text in texts]
    return np.std(sentiments)  # Standard deviation of sentiment scores

def information_content_score(texts):
    word_frequencies = nltk.FreqDist(nltk.word_tokenize(" ".join(texts)))
    return np.mean([-np.log2(word_frequencies[word] / sum(word_frequencies.values())) for word in word_frequencies])

def response_overlap(texts):
    unique_responses = set(texts)
    return len(unique_responses) / len(texts) if texts else 0


# Dictionary to store overall model comparisons
model_comparisons = {}

for model_name, file_path in model_datasets.items():
    print(f"🔍 Processing dataset for {model_name}...")

    # Load dataset
    df = pd.read_csv(file_path)

    # Ensure necessary columns exist
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in {model_name} dataset: {required_columns}")

    # Store per-model results
    grouped_results = []

    for (grade, topic, sub_topic), group in df.groupby(["Grade", "Topic", "Sub-topic"]):
        metrics = {
            "Model": model_name,
            "Grade": grade,
            "Topic": topic,
            "Sub-topic": sub_topic
        }

        # Calculate Entropy & Evenness for Challenge Type
        metrics["Challenge Type - Entropy"] = entropy_score(group["Challenge Type"])
        metrics["Challenge Type - Evenness"] = evenness_index(group["Challenge Type"])

        # Process text-based metrics for selected columns
        for col in ["Challenge Type", "Problem", "Student Mistake Prompt"]:
            texts = group[col].dropna().astype(str).tolist()
            metrics[f"{col} - Lexical Diversity"] = lexical_diversity(texts)
            metrics[f"{col} - Semantic Similarity"] = semantic_similarity(texts)
            metrics[f"{col} - Syntactic Variation"] = syntactic_variation(texts)
            metrics[f"{col} - Type-Token Ratio"] = calculate_ttr(texts)
            metrics[f"{col} - information_content_score"] = information_content_score(texts)
            metrics[f"{col} - response_overlap"] = response_overlap(texts)

        grouped_results.append(metrics)

    # Convert per-model results to DataFrame
    results_df = pd.DataFrame(grouped_results)

    # Save results to CSV
    output_filename = f"MisstepMath/Evaluation/AI_model_evaluation/{model_name}_metrics.csv"
    results_df.to_csv(output_filename, index=False)
    #files.download(output_filename)

    # Calculate Overall Average Scores for the model
    average_scores = results_df.drop(columns=["Model", "Grade", "Topic", "Sub-topic"]).mean().to_dict()
    model_comparisons[model_name] = average_scores

    print(f"✅ Completed processing for {model_name}")

# Create a comparison DataFrame
comparison_df = pd.DataFrame(model_comparisons).T  # Transpose for better readability
comparison_output = "MisstepMath/Evaluation/AI_model_evaluation/AI_Eval_model_comparisons.csv"
comparison_df.to_csv(comparison_output)
#files.download(comparison_output)

print("\n📊 Model Comparison Table:")
print(comparison_df)

"""Teacher AI model evaluation"""

import pandas as pd
import numpy as np
import nltk
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy
from google.colab import files
import io

nltk.download('punkt')

from google.colab import drive
drive.mount('/content/gdrive')

# Load Spacy model for syntactic variation
nlp = spacy.load("en_core_web_sm")

# Load Sentence Transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define dataset paths for different models
model_datasets = {
    "Student Zeroshot": "MisstepMath/Evaluation/AI_model_evaluation/student_zero_shot_results.csv",
    "Student Finetuned": "MisstepMath/Evaluation/AI_model_evaluation/student_fine_tuned_results.csv",
    "Teacher Zeroshot": "MisstepMath/Evaluation/AI_model_evaluation/teacher__zero_shot_results.csv",
    "Teacher Finetuned": "MisstepMath/Evaluation/AI_model_evaluation/teacher_finetuned_responses_dataset.csv",
}

# Required columns
required_columns = ["Grade", "Topic", "Sub-topic", "Student Mistake"]

# Function to calculate Entropy Score
def entropy_score(values):
    value_counts = values.value_counts(normalize=True)
    return entropy(value_counts, base=2) if len(value_counts) > 1 else 0

# Function to calculate Evenness Index (Pielou's Evenness)
def evenness_index(values):
    H = entropy_score(values)  # Calculate entropy
    S = values.nunique()       # Number of unique categories
    return H / np.log2(S) if S > 1 else 0  # Normalize by log of unique categories

# Function to calculate Lexical Diversity
def lexical_diversity(texts):
    words = nltk.word_tokenize(" ".join(texts))
    return len(set(words)) / len(words) if words else 0

# Function to calculate Semantic Similarity
def semantic_similarity(texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    if len(embeddings) < 2:
        return 1  # Assume max similarity for single sample
    similarities = [util.pytorch_cos_sim(embeddings[i], embeddings[i+1]).item() for i in range(len(embeddings)-1)]
    return np.mean(similarities)

# Function to calculate Syntactic Variation
def syntactic_variation(texts):
    sentence_structures = set()
    for text in texts:
        doc = nlp(text)
        sentence_structures.add(" ".join([token.pos_ for token in doc]))
    return len(sentence_structures) / len(texts) if texts else 0

# Function to calculate Type-Token Ratio (TTR)
def calculate_ttr(texts):
    all_words = " ".join(texts).split()
    unique_words = set(all_words)
    return len(unique_words) / len(all_words) if len(all_words) > 0 else 0

from textblob import TextBlob

def sentiment_variation(texts):
    sentiments = [TextBlob(text).sentiment.polarity for text in texts]
    return np.std(sentiments)  # Standard deviation of sentiment scores

def information_content_score(texts):
    word_frequencies = nltk.FreqDist(nltk.word_tokenize(" ".join(texts)))
    return np.mean([-np.log2(word_frequencies[word] / sum(word_frequencies.values())) for word in word_frequencies])

def response_overlap(texts):
    unique_responses = set(texts)
    return len(unique_responses) / len(texts) if texts else 0


# Dictionary to store overall model comparisons
model_comparisons = {}

for model_name, file_path in model_datasets.items():
    print(f"🔍 Processing dataset for {model_name}...")

    # Load dataset
    df = pd.read_csv(file_path)

    # Ensure necessary columns exist
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns in {model_name} dataset: {required_columns}")

    # Store per-model results
    grouped_results = []

    for (grade, topic, sub_topic), group in df.groupby(["Grade", "Topic", "Sub-topic"]):
        metrics = {
            "Model": model_name,
            "Grade": grade,
            "Topic": topic,
            "Sub-topic": sub_topic
        }

        # Calculate Entropy & Evenness for Challenge Type
        metrics["Challenge Type - Entropy"] = entropy_score(group["Challenge Type"])
        metrics["Challenge Type - Evenness"] = evenness_index(group["Challenge Type"])

        # Process text-based metrics for selected columns
        for col in ["Student Mistake"]:
            texts = group[col].dropna().astype(str).tolist()
            metrics[f"{col} - Lexical Diversity"] = lexical_diversity(texts)
            metrics[f"{col} - Semantic Similarity"] = semantic_similarity(texts)
            metrics[f"{col} - Syntactic Variation"] = syntactic_variation(texts)
            metrics[f"{col} - Type-Token Ratio"] = calculate_ttr(texts)
            metrics[f"{col} - information_content_score"] = information_content_score(texts)
            metrics[f"{col} - response_overlap"] = response_overlap(texts)

        grouped_results.append(metrics)

    # Convert per-model results to DataFrame
    results_df = pd.DataFrame(grouped_results)

    # Save results to CSV
    output_filename = f"MisstepMath/Evaluation/AI_model_evaluation/metrics/{model_name}_metrics.csv"
    results_df.to_csv(output_filename, index=False)
    #files.download(output_filename)

    # Calculate Overall Average Scores for the model
    average_scores = results_df.drop(columns=["Model", "Grade", "Topic", "Sub-topic"]).mean().to_dict()
    model_comparisons[model_name] = average_scores

    print(f"✅ Completed processing for {model_name}")

# Create a comparison DataFrame
comparison_df = pd.DataFrame(model_comparisons).T  # Transpose for better readability
comparison_output = "MisstepMath/Evaluation/AI_model_evaluation/metrics/benchmark_model_comparisons.csv"
comparison_df.to_csv(comparison_output)
#files.download(comparison_output)

print("\n📊 Model Comparison Table:")
print(comparison_df)

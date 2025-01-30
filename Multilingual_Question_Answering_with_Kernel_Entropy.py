import torch
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AlbertTokenizer, AlbertModel
from sklearn.metrics import roc_auc_score
import numpy as np

# Function to get embeddings from any transformer model
def get_embeddings(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Using mean pooling to get a single vector representation
    embeddings = outputs.last_hidden_state.mean(dim=1)  
    return embeddings

# Function to compute kernel entropy
def compute_kernel_entropy(embeddings):
    # Calculate pairwise cosine similarities (or use any suitable kernel function)
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=-1)
    entropy = -torch.sum(cosine_sim * torch.log(cosine_sim + 1e-9), dim=-1)  # Adding a small value to avoid log(0)
    return entropy

# Function to evaluate kernel entropy on question answering datasets
def evaluate_kernel_entropy_on_question_answering(dataset, model, tokenizer):
    all_true_labels = []
    all_predicted_scores = []

    for instance in dataset:
        # Assume 'instance' contains 'question' and 'label' (correct=1, incorrect=0)
        text = instance['question']
        true_label = instance['label']
        
        # Get embeddings for the question
        embeddings = get_embeddings(model, tokenizer, [text])
        
        # Compute uncertainty (kernel entropy)
        entropy = compute_kernel_entropy(embeddings)
        
        # Store the true label and predicted score (entropy as uncertainty measure)
        all_true_labels.append(true_label)
        all_predicted_scores.append(entropy.item())
    
    # Calculate AUROC (Area Under the Receiver Operating Characteristic Curve)
    auroc = roc_auc_score(all_true_labels, all_predicted_scores)
    return auroc

# Example multilingual dataset with questions only (simulated)
multilingual_example_dataset = [
    {'question': 'What is the capital of France?', 'label': 1},  # Correct answer
    {'question': 'What is the capital of India?', 'label': 1},  # Correct answer
    {'question': 'Wie heißt die Hauptstadt von Deutschland?', 'label': 1},  # Correct answer in German
    {'question': 'What is the largest ocean?', 'label': 1},  # Correct answer
    {'question': '¿Cuál es el color del cielo?', 'label': 1},  # Correct answer in Spanish
    {'question': 'What is the tallest mountain in the world?', 'label': 1},  # Correct answer
    {'question': 'Quelle est la capitale du Japon?', 'label': 1},  # Correct answer in French
    {'question': '日本の首都はどこですか？', 'label': 1},  # Correct answer in Japanese
    {'question': 'What is the capital of Italy?', 'label': 1},  # Correct answer
    {'question': 'இது எந்த நாட்டின் தலைநகரம்?', 'label': 0},  # Incorrect answer in Tamil (example)
    {'question': 'La capitale de l\'Espagne est Madrid?', 'label': 1},  # Correct answer in French
    {'question': 'What is the national animal of India?', 'label': 0},  # Incorrect answer
    {'question': '¿Cuál es la montaña más alta?', 'label': 1},  # Correct answer in Spanish
    {'question': 'What is the capital of Canada?', 'label': 0},  # Incorrect answer
    {'question': 'Qui est le président de la France?', 'label': 0},  # Incorrect answer in French
    {'question': 'Die Hauptstadt von Italien ist Rom?', 'label': 1},  # Correct answer in German
]

# List of models and tokenizers with correct identifiers
models_and_tokenizers = [
    ("DistilBERT", DistilBertModel, DistilBertTokenizer, "distilbert-base-uncased"),
    ("BERT", BertModel, BertTokenizer, "bert-base-uncased"),
    ("RoBERTa", RobertaModel, RobertaTokenizer, "roberta-base"),
    ("ALBERT", AlbertModel, AlbertTokenizer, "albert-base-v2")
]

# Initialize a dictionary to store AUROC scores
auroc_scores = {}

# Loop over models
for model_name, model_class, tokenizer_class, model_identifier in models_and_tokenizers:
    # Load the model and tokenizer
    model = model_class.from_pretrained(model_identifier)
    tokenizer = tokenizer_class.from_pretrained(model_identifier)

    # Evaluate on multilingual dataset
    auroc_multilingual = evaluate_kernel_entropy_on_question_answering(multilingual_example_dataset, model, tokenizer)
    
    # Store results
    auroc_scores[model_name] = auroc_multilingual

    # Print results
    print(f"Results for {model_name}:")
    print(f"  AUROC for Multilingual dataset: {auroc_multilingual}")
    print("="*50)

# Plotting the results
multilingual_auroc = list(auroc_scores.values())
model_names = list(auroc_scores.keys())

# Create a plot
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(model_names))
ax.bar(x, multilingual_auroc, color='b')

ax.set_xlabel('Model')
ax.set_ylabel('AUROC')
ax.set_title('AUROC Scores for Different Models on Multilingual Dataset')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend(["Multilingual"])

plt.tight_layout()
plt.show()

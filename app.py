# # from flask import Flask, render_template, request, redirect, url_for, session
# # import csv
# # import random
# # import os
# # import re
# # import unicodedata
# # import torch
# # from transformers import BertTokenizer, BertModel
# # from sklearn.metrics.pairwise import cosine_similarity
# # import numpy as np

# # app = Flask(__name__)
# # app.secret_key = 'your_secret_key'  # Replace with a strong secret key

# # # Load quiz data from CSV
# # def load_quiz_data(filename):
# #     quiz_data = []
# #     with open(filename, 'r', encoding='utf-8') as csvfile:
# #         reader = csv.reader(csvfile)
# #         next(reader)  # Skip the header row
# #         for row in reader:
# #             if row:
# #                 quiz_data.append({"nufi": row[0].strip(), "french": row[1].strip().lower()})
# #     return quiz_data

# # quiz_data = load_quiz_data('quiz_data.csv')

# # # Load BERT model and tokenizer
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # model = BertModel.from_pretrained('bert-base-uncased')

# # # Expanded dictionary of synonyms
# # synonyms = {
# #     "vu": {"regardé", "observé"},
# #     "mange": {"consomme", "déguste"},
# #     "nourriture": {"repas", "alimentation"},
# #     "beaucoup": {"grand", "énorme", "immense"},
# #     "grand": {"beaucoup", "immense", "énorme"},
# #     "j'ai vu": {"j'ai regardé", "j'ai observé"},
# #     "je mange la nourriture": {"je mange le repas", "je consomme l'alimentation"}
# # }

# # # Function to normalize punctuation
# # def normalize_punctuation(text):
# #     # Normalize different types of quotes and apostrophes
# #     text = text.replace('’', "'").replace('“', '"').replace('”', '"')
# #     # Normalize different types of question marks and other punctuation
# #     text = text.replace('？', '?').replace('！', '!')
# #     return text

# # # Function to remove punctuation
# # def remove_punctuation(text):
# #     return re.sub(r'[^\w\s]', '', text)

# # # Function to normalize text
# # def normalize_text(text):
# #     # Normalize Unicode characters
# #     text = unicodedata.normalize('NFKD', text)
# #     # Normalize punctuation
# #     text = normalize_punctuation(text)
# #     # Remove extra spaces
# #     text = re.sub(r'\s+', ' ', text).strip()
# #     return text

# # # Function to get BERT embeddings
# # def get_embeddings(text):
# #     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
# #     outputs = model(**inputs)
# #     return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# # # Function to calculate sentence similarity
# # def calculate_similarity(user_answer, correct_answer):
# #     user_embedding = get_embeddings(user_answer)
# #     correct_embedding = get_embeddings(correct_answer)
# #     similarity = cosine_similarity(user_embedding, correct_embedding)
# #     return similarity[0][0]

# # # Function to check if words or phrases are synonyms
# # def are_synonyms(word_or_phrase, correct_words):
# #     if word_or_phrase in correct_words:
# #         return True
# #     for correct_word in correct_words:
# #         if word_or_phrase in synonyms.get(correct_word, set()):
# #             return True
# #     return False

# # # Function to calculate percentage of correct words or phrases
# # def calculate_correctness(user_answer, correct_answer):
# #     user_words = user_answer.split()
# #     correct_words = correct_answer.split()
# #     match_count = 0

# #     for word in user_words:
# #         if are_synonyms(word, correct_words):
# #             match_count += 1

# #     return match_count / len(correct_words)

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/quiz', methods=['GET', 'POST'])
# # def quiz():
# #     if request.method == 'POST':
# #         answers = request.form.to_dict()
# #         results = []
# #         score = 0
# #         selected_indices = session.get('selected_indices', [])

# #         for question_id, user_answer in answers.items():
# #             question_id = int(question_id)
# #             original_index = selected_indices[question_id]
# #             correct_answer = quiz_data[original_index]["french"]
            
# #             # Normalize and clean the answers
# #             user_answer_clean = normalize_text(user_answer.strip().lower())
# #             correct_answer_clean = normalize_text(correct_answer)
            
# #             # Check if answers are synonymous
# #             if calculate_correctness(user_answer_clean, correct_answer_clean) >= 0.8:
# #                 is_correct = True
# #             else:
# #                 # Use BERT embeddings to calculate similarity if synonyms don't match
# #                 similarity = calculate_similarity(user_answer_clean, correct_answer_clean)
# #                 is_correct = similarity >= 0.8
            
# #             if is_correct:
# #                 score += 1
            
# #             results.append({
# #                 "nufi": quiz_data[original_index]["nufi"],
# #                 "user_answer": user_answer,
# #                 "correct_answer": correct_answer,
# #                 "is_correct": is_correct
# #             })
# #         return render_template('result.html', score=score, total=len(answers), results=results)
# #     else:
# #         selected_questions = random.sample(quiz_data, 5)
# #         selected_indices = [quiz_data.index(q) for q in selected_questions]
# #         session['selected_indices'] = selected_indices
# #         return render_template('quiz.html', quiz_data=selected_questions)

# # if __name__ == '__main__':
# #     app.run(debug=True)

# from flask import Flask, render_template, request, redirect, url_for, session
# import csv
# import random
# import re
# import unicodedata
# import torch
# from transformers import BertTokenizer, BertModel
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import nltk
# from nltk.corpus import wordnet

# # Download WordNet data if not already downloaded
# nltk.download('wordnet')

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Replace with a strong secret key

# # Load quiz data from CSV
# def load_quiz_data(filename):
#     quiz_data = []
#     with open(filename, 'r', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         next(reader)  # Skip the header row
#         for row in reader:
#             if row:
#                 quiz_data.append({"nufi": row[0].strip(), "french": row[1].strip().lower()})
#     return quiz_data

# quiz_data = load_quiz_data('quiz_data.csv')

# # Load BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# # Function to normalize punctuation
# def normalize_punctuation(text):
#     # Normalize different types of quotes and apostrophes
#     text = text.replace('’', "'").replace('“', '"').replace('”', '"')
#     # Normalize different types of question marks and other punctuation
#     text = text.replace('？', '?').replace('！', '!')
#     return text

# # Function to remove punctuation
# def remove_punctuation(text):
#     return re.sub(r'[^\w\s]', '', text)

# # Function to normalize text
# def normalize_text(text):
#     # Normalize Unicode characters
#     text = unicodedata.normalize('NFKD', text)
#     # Normalize punctuation
#     text = normalize_punctuation(text)
#     # Remove extra spaces
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # Function to get BERT embeddings
# def get_embeddings(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# # Function to calculate sentence similarity
# def calculate_similarity(user_answer, correct_answer):
#     user_embedding = get_embeddings(user_answer)
#     correct_embedding = get_embeddings(correct_answer)
#     similarity = cosine_similarity(user_embedding, correct_embedding)
#     return similarity[0][0]

# # Function to check if words or phrases are synonyms using WordNet
# def are_synonyms(word_or_phrase, correct_words):
#     if word_or_phrase in correct_words:
#         return True
#     for correct_word in correct_words:
#         synonyms = set()
#         for syn in wordnet.synsets(correct_word):
#             for lemma in syn.lemmas():
#                 synonyms.add(lemma.name().replace('_', ' '))
#         if word_or_phrase in synonyms:
#             return True
#     return False

# # Function to calculate percentage of correct words or phrases
# def calculate_correctness(user_answer, correct_answer):
#     user_words = user_answer.split()
#     correct_words = correct_answer.split()
#     match_count = 0

#     for word in user_words:
#         if are_synonyms(word, correct_words):
#             match_count += 1

#     return match_count / len(correct_words)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/quiz', methods=['GET', 'POST'])
# def quiz():
#     if request.method == 'POST':
#         answers = request.form.to_dict()
#         results = []
#         score = 0
#         selected_indices = session.get('selected_indices', [])

#         for question_id, user_answer in answers.items():
#             question_id = int(question_id)
#             original_index = selected_indices[question_id]
#             correct_answer = quiz_data[original_index]["french"]
            
#             # Normalize and clean the answers
#             user_answer_clean = normalize_text(user_answer.strip().lower())
#             correct_answer_clean = normalize_text(correct_answer)
            
#             # Check if answers are synonymous
#             if calculate_correctness(user_answer_clean, correct_answer_clean) >= 0.8:
#                 is_correct = True
#             else:
#                 # Use BERT embeddings to calculate similarity if synonyms don't match
#                 similarity = calculate_similarity(user_answer_clean, correct_answer_clean)
#                 is_correct = similarity >= 0.8
            
#             if is_correct:
#                 score += 1
            
#             results.append({
#                 "nufi": quiz_data[original_index]["nufi"],
#                 "user_answer": user_answer,
#                 "correct_answer": correct_answer,
#                 "is_correct": is_correct
#             })
#         return render_template('result.html', score=score, total=len(answers), results=results)
#     else:
#         selected_questions = random.sample(quiz_data, 5)
#         selected_indices = [quiz_data.index(q) for q in selected_questions]
#         session['selected_indices'] = selected_indices
#         return render_template('quiz.html', quiz_data=selected_questions)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session
import csv
import random
import re
import unicodedata
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import wordnet

# Download WordNet data if not already downloaded
nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key

# Load quiz data from CSV
def load_quiz_data(filename):
    quiz_data = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            if row:
                quiz_data.append({"nufi": row[0].strip(), "french": row[1].strip().lower()})
    return quiz_data

quiz_data = load_quiz_data('quiz_data.csv')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to normalize punctuation
def normalize_punctuation(text):
    # Normalize different types of quotes and apostrophes
    text = text.replace('’', "'").replace('“', '"').replace('”', '"')
    # Normalize different types of question marks and other punctuation
    text = text.replace('？', '?').replace('！', '!')
    return text

# Function to remove punctuation
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Function to normalize text
def normalize_text(text):
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Normalize punctuation
    text = normalize_punctuation(text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to get BERT embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to calculate sentence similarity
def calculate_similarity(user_answer, correct_answer):
    user_embedding = get_embeddings(user_answer)
    correct_embedding = get_embeddings(correct_answer)
    similarity = cosine_similarity(user_embedding, correct_embedding)
    return similarity[0][0]

# Function to check if words or phrases are synonyms using WordNet
def are_synonyms(word_or_phrase, correct_words):
    if word_or_phrase in correct_words:
        return True
    for correct_word in correct_words:
        synonyms = set()
        for syn in wordnet.synsets(correct_word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        if word_or_phrase in synonyms:
            return True
    return False

# Function to calculate percentage of correct words or phrases
def calculate_correctness(user_answer, correct_answer):
    user_words = user_answer.split()
    correct_words = correct_answer.split()
    match_count = 0

    for word in user_words:
        if are_synonyms(word, correct_words):
            match_count += 1

    return match_count / len(correct_words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if request.method == 'POST':
        answers = request.form.to_dict()
        results = []
        score = 0
        selected_indices = session.get('selected_indices', [])

        for question_id, user_answer in answers.items():
            question_id = int(question_id)
            original_index = selected_indices[question_id]
            correct_answer = quiz_data[original_index]["french"]
            
            # Normalize and clean the answers
            user_answer_clean = normalize_text(user_answer.strip().lower())
            correct_answer_clean = normalize_text(correct_answer)
            
            # Check if answers are synonymous
            if calculate_correctness(user_answer_clean, correct_answer_clean) >= 0.8:
                is_correct = True
            else:
                # Use BERT embeddings to calculate similarity if synonyms don't match
                similarity = calculate_similarity(user_answer_clean, correct_answer_clean)
                is_correct = similarity >= 0.8
            
            if is_correct:
                score += 1
            
            results.append({
                "nufi": quiz_data[original_index]["nufi"],
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct
            })
        return render_template('result.html', score=score, total=len(answers), results=results)
    else:
        selected_questions = random.sample(quiz_data, 5)
        selected_indices = [quiz_data.index(q) for q in selected_questions]
        session['selected_indices'] = selected_indices
        return render_template('quiz.html', quiz_data=selected_questions)

if __name__ == '__main__':
    app.run(debug=True)

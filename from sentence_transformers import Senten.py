from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np

# Загрузка предобученной модели
model_name = 'DeepPavlov/rubert-base-cased'
model = SentenceTransformer(model_name)

# Генерация эмбеддингов ответов для train
answer_embeddings = model.encode(train_df['answer'].tolist(), convert_to_tensor=True)

# Функция оценки качества
def evaluate(model, questions, true_answers, answer_embeddings, train_answers):
    correct = 0
    for question, true_answer in tqdm(zip(questions, true_answers), total=len(questions)):
        question_embedding = model.encode(question, convert_to_tensor=True)
        hits = util.semantic_search(question_embedding, answer_embeddings, top_k=1)
        predicted_answer = train_answers[hits[0][0]['corpus_id']]
        if predicted_answer == true_answer:
            correct += 1
    accuracy = correct / len(questions)
    return accuracy

# Оценка
accuracy_before = evaluate(model, test_df['question'].tolist(), test_df['answer'].tolist(), answer_embeddings, train_df['answer'].tolist())
print(f'Accuracy без дообучения: {accuracy_before:.4f}')
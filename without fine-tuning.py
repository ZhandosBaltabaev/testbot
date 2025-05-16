import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

with open("fine_tune_techlab_ru.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

user_prompts = []
reference_answers = []

for item in data:
    for msg in item["messages"]:
        if msg["role"] == "user":
            user_prompts.append(msg["content"])
        elif msg["role"] == "assistant":
            reference_answers.append(msg["content"])

generated_answers = {
    "Здравствуйте!": "Здравствуйте! Чем могу помочь вам по поводу курсов TechLab?",
    "Моему ребенку 5-9 лет.": "Отлично! Мы предлагаем курсы по робототехнике и программированию для детей 5-9 лет.",
    "Моему ребенку 9-17 лет.": "Прекрасно! У нас есть курсы LEGO Mindstorms и программирования для детей 9-17 лет.",
    "Сколько стоит разовое посещение?": "Разовое занятие стоит от 5000 до 5500 тг в зависимости от возраста ребенка.",
    "Какие есть варианты абонементов?": "Мы предлагаем разные абонементы в зависимости от возраста: на 2, 4, 6 месяцев или год."
}

smoother = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

bleu_scores = []
rouge_scores = []

hypotheses = []
references = []

for i, prompt in enumerate(user_prompts):
    ref = reference_answers[i]
    gen = generated_answers.get(prompt, "")

    bleu = sentence_bleu([ref.split()], gen.split(), smoothing_function=smoother)
    bleu_scores.append(bleu)

    r_scores = rouge.score(ref, gen)
    rouge_scores.append(r_scores)

    hypotheses.append(gen)
    references.append(ref)

P, R, F1 = bert_score(hypotheses, references, lang="ru")

print("Средний BLEU:", sum(bleu_scores)/len(bleu_scores))
print("Средний ROUGE-1 F1:", sum(r["rouge1"].fmeasure for r in rouge_scores)/len(rouge_scores))
print("Средний ROUGE-L F1:", sum(r["rougeL"].fmeasure for r in rouge_scores)/len(rouge_scores))
print("Средний BERTScore F1:", F1.mean().item())

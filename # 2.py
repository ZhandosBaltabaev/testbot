import openai


# 1. Загрузить файл
file = openai.File.create(
    file=open("fine_tune_techlab_ru.jsonl", "rb"),
    purpose="fine-tune"
)

print("File ID:", file.id)
# 2. Запустить fine-tune
response = openai.FineTuningJob.create(
    training_file=file.id,
    model="gpt-3.5-turbo"
)

print("Job ID:", response.id)

status = openai.FineTuningJob.retrieve(response.id)
print(status)
import openai
# Замените на ваш job ID
job = openai.FineTuningJob.retrieve("ftjob-QWcVxJfrL0oNoudfYwmQMk5D")

# Просмотр данных о дообученной модели
print(job)
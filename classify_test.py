from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

text = input("Enter your interview answer: ")
result = classifier(text)
print("Result:", result)

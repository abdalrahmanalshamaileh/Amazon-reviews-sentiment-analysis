import tkinter as tk
from tkinter import messagebox
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load the fine-tuned tokenizer and model
MODEL_PATH = "C:/Users/Shamailh_M77/fine_tuned_roberta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Mapping sentiment to your specifications
def map_to_custom_scale(score):
    if score <= 2:
        return "Negative 😠"
    elif score == 3:
        return "Neutral 😐"
    else:
        return "Positive 😊"

# Function to classify sentiment based on VADER scores
def classify_sentiment_vader(vader_scores):
    # Convert compound score to a scale of 1-5
    compound = vader_scores['compound']
    score = 3 if -0.05 <= compound <= 0.05 else (5 if compound > 0.05 else 1)
    return map_to_custom_scale(score)

# Function to classify sentiment based on RoBERTa scores
def classify_sentiment_roberta(roberta_scores):
    # Map scores to custom 1-5 scale
    if roberta_scores['roberta_pos'] > roberta_scores['roberta_neg'] and roberta_scores['roberta_pos'] > roberta_scores['roberta_neu']:
        score = 5
    elif roberta_scores['roberta_neg'] > roberta_scores['roberta_pos'] and roberta_scores['roberta_neg'] > roberta_scores['roberta_neu']:
        score = 1
    else:
        score = 3
    return map_to_custom_scale(score)

# Function to get polarity scores using RoBERTa
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    output = model(**encoded_text.to(model.device))
    scores = softmax(output.logits[0].detach().cpu().numpy())
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }

# Function to analyze a random review and display results
def analyze_review():
    random_review = df.sample(1).iloc[0]
    review_text = random_review['Text']
    
    # VADER Analysis
    vader_scores = sia.polarity_scores(review_text)
    vader_sentiment = classify_sentiment_vader(vader_scores)
    
    # RoBERTa Analysis
    roberta_scores = polarity_scores_roberta(review_text)
    roberta_sentiment = classify_sentiment_roberta(roberta_scores)
    
    # Display results
    review_label.config(text=f"Review: {review_text}")
    vader_label.config(text=f"VADER Sentiment: {vader_sentiment}")
    roberta_label.config(text=f"RoBERTa Sentiment: {roberta_sentiment}")

# Load your dataset (limit to 5000 for quick analysis)
DATA_PATH = "C:/Users/Shamailh_M77/Downloads/AmazonDataset/Reviews.csv"
df = pd.read_csv(DATA_PATH).head(5000)

# GUI Setup
root = tk.Tk()
root.title("Sentiment Analysis")

# GUI Elements
frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

review_label = tk.Label(frame, text="Click 'Analyze' to analyze a random review.", wraplength=500, justify="left", font=("Arial", 12))
review_label.pack(pady=10)

vader_label = tk.Label(frame, text="", font=("Arial", 12))
vader_label.pack(pady=5)

roberta_label = tk.Label(frame, text="", font=("Arial", 12))
roberta_label.pack(pady=5)

analyze_button = tk.Button(frame, text="Analyze", command=analyze_review, bg="blue", fg="white", font=("Arial", 12))
analyze_button.pack(pady=10)

# Run the GUI
root.mainloop()

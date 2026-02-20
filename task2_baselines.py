# task2_baselines.py
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from datasets import load_dataset
import nltk
import os
import json
import matplotlib.pyplot as plt
from collections import Counter

nltk.download('vader_lexicon')


def run_vader(texts):
    sid = SentimentIntensityAnalyzer()
    results = []
    for t in texts:
        results.append((t, sid.polarity_scores(t)))
    return results


def run_transformers_pipeline(texts):
    clf = pipeline("sentiment-analysis")  # distilbert fine-tuned for sentiment
    results = []
    for t in texts:
        results.append((t, clf(t)[0]))
    return results


def save_results(results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def plot_vader_distribution(vader_results, out_path):
    # We'll plot the compound score distribution
    compounds = []
    for r in vader_results:
        compounds.append(r[1]['compound'])
    plt.figure(figsize=(8, 4))
    plt.hist(compounds, bins=30, color='skyblue', edgecolor='black')
    plt.title('VADER Compound Score Distribution')
    plt.xlabel('Compound Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_transformers_distribution(transformers_results, out_path):
    # We'll plot the label distribution
    labels = []
    for r in transformers_results:
        labels.append(r[1]['label'])
    label_counts = Counter(labels)
    plt.figure(figsize=(6, 4))
    plt.bar(label_counts.keys(), label_counts.values(),
            color='salmon', edgecolor='black')
    plt.title('Transformers Sentiment Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    # Load the dair-ai/emotion dataset
    raw = load_dataset("dair-ai/emotion")
    # Use the first 2000 texts from the train split for demonstration
    texts = []
    for i in range(2000):
        texts.append(raw["train"][i]["text"])

    # Run baselines
    vader_results = run_vader(texts)
    transformers_results = run_transformers_pipeline(texts)

    # Output folder
    output_dir = "sentiment_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save results to files
    save_results(vader_results, os.path.join(output_dir, "vader_results.json"))
    save_results(transformers_results, os.path.join(
        output_dir, "transformers_results.json"))

    # Print to console as before
    print("Texts:", texts)
    print("VADER:", vader_results)
    print("Transformers:", transformers_results)

    # Plot and save distributions
    plot_vader_distribution(vader_results, os.path.join(
        output_dir, "vader_distribution.png"))
    plot_transformers_distribution(transformers_results, os.path.join(
        output_dir, "transformers_distribution.png"))

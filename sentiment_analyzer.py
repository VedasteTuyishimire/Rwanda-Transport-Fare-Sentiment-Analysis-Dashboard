"""
Sentiment analysis module for Rwanda transport fare feedback.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import os

class SentimentAnalyzer:
    def __init__(self, model_name: str = "xlm-roberta-base"):
        """
        Initialize the sentiment analyzer with a multilingual model.
        
        Args:
            model_name (str): Name of the pretrained model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # positive, negative, neutral
        ).to(self.device)
        
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Dictionary containing sentiment label and confidence score
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities).item()
        
        return {
            "text": text,
            "sentiment": self.id2label[prediction.item()],
            "confidence": confidence,
            "probabilities": {
                label: prob.item()
                for label, prob in zip(self.id2label.values(), probabilities[0])
            }
        }
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            pandas.DataFrame: DataFrame containing sentiment analysis results
        """
        results = []
        for text in texts:
            try:
                result = self.analyze_text(text)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing text: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def analyze_and_save(self, input_file: str, output_file: str = None):
        """
        Analyze sentiments from input CSV file and save results.
        
        Args:
            input_file (str): Path to input CSV file containing 'text' column
            output_file (str): Path to save output CSV file (optional)
        """
        df = pd.read_csv(input_file)
        
        if 'text' not in df.columns:
            raise ValueError("Input CSV must contain 'text' column")
        
        results_df = self.analyze_batch(df['text'].tolist())
        
        # Merge with original data
        output_df = pd.concat([df, results_df], axis=1)
        
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_df.to_csv(output_file, index=False)
        
        return output_df

if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Test single text
    result = analyzer.analyze_text("The new distance-based fare system is fair and transparent")
    print("Single text analysis:", result)
    
    # Test batch analysis
    texts = [
        "The new fare system is too expensive!",
        "I appreciate the transparency in pricing now",
        "Not sure how I feel about the changes"
    ]
    results_df = analyzer.analyze_batch(texts)
    print("\nBatch analysis results:")
    print(results_df) 
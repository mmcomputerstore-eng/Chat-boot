import dotenv
import os
from openai import OpenAI
from transformers import pipeline

dotenv.load_dotenv()

class Analyzer():
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

        # Initialize conversation history with default persona / system prompt
        self.messages = [
            {
                "role": "system",
                "content": 'Classify the sentiment of the text as positive, negative or neutral'
            },{
                "role": "user",
                "content": 'I absolutely love this restaurant!'
                        },
            {
                "role": "assistant",
                "content": 'positive'
                        }
            ,{
                "role": "user",
                "content": 'the wait time was too long'
                        },
            {
                "role": "assistant",
                "content": 'negative'
                        }
            ,{
                "role": "user",
                "content": 'it was an average experience, nothing special'
                        },
            {
                "role": "assistant",
                "content": 'neutral'
                        },
        ]
        
    def sentiment_analysis(self,text):
        labels = ['positive', 'negative', 'neutral']
        result = self.classifier(text,candidate_labels = labels)
        
        
        return result
    
        
        
    


def main():
    analyzer = Analyzer()
    print(analyzer.sentiment_analysis('The move is Perfect'))


if __name__ == '__main__':
    main()
import pandas as pd
from datetime import datetime
from typing import List
import os
from openai import AzureOpenAI
from .summarize import Summarizer
from .classify import Classifier
from .condense import Condenser

class NLPTopicModeler:
    def __init__(self,
            dataframe: pd.DataFrame,
            azure_client: AzureOpenAI,
            azure_model: str,
            log_path: str = f'./logs',
            ):
        self.dataframe = dataframe
        self.azure_client = azure_client
        self.azure_model = azure_model
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.mkdir(log_path)
    
    def summarize(self,
            target_column: str,
            summary_column: str = 'summary',
            workers: int = 1,
            temperature: float = 0,
            max_reply_tokens: int = 1000
            ):
        summarizer = Summarizer(dataframe=self.dataframe,
                        azure_client=self.azure_client,
                        azure_model=self.azure_model,
                        target_column=target_column,
                        summary_column=summary_column,
                        workers=workers,
                        log_path=self.log_path,
                        temperature=temperature,
                        max_reply_tokens=max_reply_tokens
                        )
        
        summarizer.bulk_summarize_transcripts()

        return summarizer
    
    def condense(self,
            target_column: str = 'summary',
            num_topics: int = 15,
            topic: str = None,
            temperature: float = 0,
            max_reply_tokens: int = 4000,
            max_input_tokens: int = 20000
            ):
        
        condenser = Condenser(dataframe=self.dataframe,
            azure_client=self.azure_client,
            azure_model=self.azure_model,
            target_column=target_column,
            num_topics=num_topics,
            topic=topic,
            log_path=self.log_path,
            temperature=temperature,
            max_reply_tokens=max_reply_tokens,
            max_input_tokens=max_input_tokens)
        
        condenser.condense_topics_from_dataframe()

        return condenser
    
    def classify(self,
            topics: List[str],
            target_column: str = 'redacted_transcript',
            topic_name: str = 'topic',
            category: str = 'topic',
            workers: int = 1,
            temperature: float = 0,
            max_reply_tokens: int = 1000):
        
        classifier = Classifier(dataframe=self.dataframe,
            azure_client=self.azure_client,
            azure_model=self.azure_model,
            topics=topics,
            target_column=target_column,
            topic_name=topic_name,
            category=category,
            workers=workers,
            log_path=self.log_path,
            temperature=temperature,
            max_reply_tokens=max_reply_tokens)
        
        classifier.classify_topics()

        return classifier
    
    def model_column(self,
            target_column: str,
            topic_name: str = 'topic',
            summary_column: str = 'summary',
            category: str = 'topic',
            summarize_first: bool = False,
            workers: int = 1,
            temperature: float = 0,
            max_reply_tokens: int = 1000,
            num_topics: int = 15,
            topic: str = None,
            max_input_tokens: int = 20000):

        summarizer = None
        if summarize_first:
            summarizier = self.summarize(target_column=target_column,
                                        summary_column=summary_column,
                                        workers=workers,
                                        temperature=temperature,
                                        max_reply_tokens=max_reply_tokens)
        if summarizer:
            self.dataframe=summarizer.dataframe
            target_column=summary_column

        condenser = self.condense(target_column=target_column,
                                  num_topics=num_topics,
                                  topic=topic,
                                  max_reply_tokens=max_reply_tokens,
                                  max_input_tokens=max_input_tokens)
        
        classifier = self.classify(topics=condenser.common_topics,
                                   target_column=target_column,
                                   topic_name=topic_name,
                                   category=category,
                                   workers=workers,
                                   temperature=temperature,
                                   max_reply_tokens=max_reply_tokens)
        
        self.dataframe=classifier.dataframe

        return self.dataframe
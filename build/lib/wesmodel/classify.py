import openai
from openai import AzureOpenAI
import pandas as pd
from datetime import datetime
from typing import List
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class Classifier:
    def __init__(self,
            dataframe: pd.DataFrame,
            azure_client: AzureOpenAI,
            azure_model: str,
            topics: List[str],
            target_column: str = 'redacted_transcript',
            topic_name: str = 'topic',
            category: str = 'topic',
            workers: int = 5,
            log_path: str = f'./logs',
            temperature: float = 0,
            max_reply_tokens: int = 1000
            ):
        if target_column not in dataframe.columns:
            raise ValueError(f"Dataframe does not contain column {target_column}")
        if category.lower() not in ['topic', 'subtopic']:
            raise ValueError('Category must be "topic" or "subtopic".')
        self.dataframe = dataframe
        self.azure_client = azure_client
        self.azure_model = azure_model
        self.topics = topics
        self.target_column = target_column
        self.topic_name = topic_name
        self.category = category
        self.workers = workers
        self.log_path = log_path
        self.temperature = temperature
        self.max_reply_tokens = max_reply_tokens
        self.duration = None
    
    def log(self, input: str):
        print(input, file=open(f'{self.log_path}/{datetime.now().date()}.txt', 'a'))
    
    def classify_transcript(self, transcript, system_prompt):
    
        user_prompt = f"""
            Topics:
            {'|'.join(self.topics)}

            Piece of Text:
            {transcript}

            Assigned Topic:
            """
        
        message_pkg = [{"role":"system","content":system_prompt},
                        {"role":"user","content":user_prompt}]
        
        attempt = 0
        while True:
            try:
                response = self.azure_client.chat.completions.create(
                    model=self.azure_model,
                    messages=message_pkg,
                    temperature=self.temperature,
                    max_tokens=self.max_reply_tokens,
                )

                if response is None:
                    self.log(f"{datetime.now()}: Failed no response")
                    return None
                return response.choices[0].message.content
            except openai.RateLimitError as e:
                attempt += 1
                time.sleep(2 ** attempt)
            except Exception as e:
                self.log(f"{datetime.now()}: Failed with error {e}")
                return f"FAILED: {e}"
            
    def classify_topics(self):

        df = self.dataframe
        start_time = datetime.now()
        self.log(f"{start_time}: Begin Classifying {len(df)} Transcripts")


        df[self.topic_name] = None

        system_prompt = f"""You are a helpful bot that is given a list of {self.category}s and a single piece of text.
            Assign one of the {self.category}s to the text. The {self.category}s are separated by the '|' character.
            If you cannot find an appropriate {self.category}, simply respond 'no {self.category}'.
            
            Instructions:
            - Assign one of the {self.category}s to the text
            - Use only {self.category}s from the list you are given without changing names or extrapolating
            - If none of the {self.category}s fit, respond '{self.category}'"""

        def process_row(index, row):
            try:
                result = self.classify_transcript(row[self.target_column], system_prompt)
                return index, result
            except Exception as e:
                self.log(f"{datetime.now()}: FAILED with exception {e}")
                return index, None

        def update_df(result):
            index, topic = result
            if topic is None:
                df.at[index, self.topic_name] = 'FAILED: Uncaught Exception'
            else:
                df.at[index, self.topic_name] = topic

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(process_row, index, row) for index, row in df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying"):
                update_df(future.result())
        
        initial = len(df)
        df = df[~(df[self.topic_name].str.contains('FAILED'))]
        final = len(df)
        end_time = datetime.now()
        self.log(f"{end_time}: Finish Classifying {len(df)} Transcripts")
        self.log(f"Dropped {initial-final} Transcripts for Error")
        self.log(f"Total Classification Time: {end_time-start_time}")
        self.duration = end_time-start_time

        df[self.topic_name] = df[self.topic_name].apply(lambda x: f"No {self.category.title()}" if x==f"no {self.category}" else x)

        self.dataframe = df
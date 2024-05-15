import pandas as pd
import openai
from openai import AzureOpenAI
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class Summarizer:
    def __init__(self,
        dataframe: pd.DataFrame,
        azure_client: AzureOpenAI,
        azure_model: str,
        target_column: str = 'redacted_transcript',
        summary_column: str = 'summary',
        workers: int = 5,
        log_path: str = f'./logs',
        temperature: float = 0,
        max_reply_tokens: int = 1000
        ):
        if target_column not in dataframe.columns:
            raise ValueError(f"Dataframe does not contain column {target_column}")
        self.dataframe = dataframe
        self.azure_client = azure_client
        self.azure_model = azure_model
        self.target_column = target_column
        self.summary_column = summary_column
        self.workers = workers
        self.log_path = log_path
        self.temperature = temperature
        self.max_reply_tokens = max_reply_tokens
    
    def log(self, input: str):
        print(input, file=open(f'{self.log_path}/{datetime.now().date()}.txt', 'a'))

    def summarize_transcript(self, transcript: str):

        system_prompt = """INSTRUCTIONS: 
        You are a helpful bot. You will receive the piece of text.  
        Your job is to identify the main topic of the text. 

        RULES: 
        Return a string containing the primary topic of the text. Each topic MUST BE five words or fewer. 
        Your response can contain only one reason."""

        message_pkg = [{"role":"system","content":system_prompt},
                    {"role":"user","content":transcript}]
        
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
                    self.log("FAILED: No response")
                    return None
                return response.choices[0].message.content
            except openai.RateLimitError as e:
                attempt += 1
                time.sleep(2 ** attempt)
            except Exception as e:
                self.log(f"{datetime.now()}: FAILED with exception {e}")
                return f"FAILED: {e}"
    
    def bulk_summarize_transcripts(self):
        start_time = datetime.now()
        self.log(f"{start_time}: Begin Summarizing {len(self.dataframe)} Transcripts")
        
        self.dataframe[self.summary_column] = None

        def process_row(index, row):
            try:
                result = self.summarize_transcript(row[self.target_column])
                return index, result
            except Exception as e:
                print(f"{datetime.now()}: FAILED with exception {e}", file=open(self.log_path, 'a'))
                return index, None

        def update_df(result):
            index, summary = result
            if summary is None:
                self.dataframe.at[index, self.summary_column] = 'FAILED: Uncaught Exception'
            else:
                self.dataframe.at[index, self.summary_column] = summary

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(process_row, index, row) for index, row in self.dataframe.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
                update_df(future.result())

        initial = len(self.dataframe)
        self.dataframe = self.dataframe[~(self.dataframe[self.summary_column].str.contains('FAILED'))]
        final = len(self.dataframe)
        end_time = datetime.now()
        self.log(f"{end_time}: Finish Summarizing {len(self.dataframe)} Transcripts")
        self.log(f"Dropped {initial-final} Transcripts for Error")
        self.log(f"Total Summarization Time: {end_time-start_time}")
        
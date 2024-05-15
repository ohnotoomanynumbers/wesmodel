import pandas as pd
from openai import AzureOpenAI
from datetime import datetime
import tiktoken

class Condenser:
    def __init__(self,
            dataframe: pd.DataFrame,
            azure_client: AzureOpenAI,
            azure_model: str,
            target_column: str = 'summary',
            num_topics: int = 15,
            topic: str = None,
            log_path: str = f'./logs',
            temperature: float = 0,
            max_reply_tokens: int = 4000,
            max_input_tokens: int = 20000
            ):
        self.dataframe = dataframe
        self.azure_client = azure_client
        self.azure_model = azure_model
        self.target_column = target_column
        self.num_topics = num_topics
        self.topic = topic
        self.log_path = log_path
        self.temperature = temperature
        self.max_reply_tokens = max_reply_tokens
        self.max_input_tokens = max_input_tokens
        self.common_topics = None
        self.duration = None
    
    def log(self, input):
        print(input, file=open(f'{self.log_path}/{datetime.now().date()}.txt', 'a'))

    def count_tokens(self, text):
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = encoding.encode(text)

        return len(token_count)
    
    def clean_results(self, results):
        return [result.split('.')[1].split('(')[0].strip() for result in results]
    
    def condense_topics_from_dataframe(self):

        df = self.dataframe

        if self.topic:
            context_string = f" These topics were all initially categorized as {self.topic}. All topics should be subtopics of {self.topic}"
            rule_string = f"- generate subtopics of {self.topic}"
        else:
            context_string = ""
            rule_string = ""
        
        system_prompt = f"""
        Given the following topics, try and combine them into a single set of COMMON topics. You can merge common topics together.{context_string}

        Rules:
        - only generate maximum {self.num_topics} topics total
        - return a numbered list of descriptions
        - rank more common topics higher than less common topics
        {rule_string}
        """
        start_time = datetime.now()
        self.log(f'{start_time}: Begin condensing topics')


        # gather a sample size that is as large as can reasonably be achieved 
        sample_size = len(df)
        while True:
            if self.count_tokens(str(df[self.target_column].sample(sample_size).tolist())) < self.max_input_tokens: break
            sample_size = sample_size*3//4

        first_batch = df[self.target_column].sample(sample_size).tolist()
        second_batch = df[self.target_column].sample(sample_size).tolist()
        third_batch = df[self.target_column].sample(sample_size).tolist()

        # Call the generate_overarching_topics function with the list of all topics
        first_condensed = self.condense_topics_openai_call(first_batch, system_prompt).split("\n")
        second_condensed = self.condense_topics_openai_call(second_batch, system_prompt).split("\n")
        third_condensed = self.condense_topics_openai_call(third_batch, system_prompt).split("\n")

        final_batch = first_condensed + second_condensed + third_condensed

        final_condensed = self.condense_topics_openai_call(final_batch).split('\n')

        clean = self.clean_results(final_condensed)

        end_time = datetime.now()
        self.log(f'{end_time}: Finish condensing topics')
        self.log(f'Total Condensation Time: {end_time-start_time}')
        self.duration = end_time-start_time
        self.log(f'List of topics: {clean}')

        self.common_topics = clean
    
    def condense_topics_openai_call(self, topics_list, system_prompt=None):
    # Join the list of topics into a single string separated by commas
        if not system_prompt:
            system_prompt = f"""
            Given the following sets of topics, try and combine them into a single set of COMMON topics. The topics have already been condensed and ranked multiple separate times. 
            A higher  rank indicates that a topic was more common in the dataset. You can merge common topics together.

            Rules:
            - only generate maximum {self.num_topics} topics total
            - include examples within () characters
            - weight higher ranked topics more than lower ranked topics
            - return the list formatted like so: 1. [Topic Name] ([short list of examples])
            """
        
        topics_str = """
        Topics:
        ${topics_list}

        Combined Topics:
        """

        topics_str = topics_str.replace("${topics_list}", str(topics_list))

        self.log(f"Topics Token Count: {self.count_tokens(topics_str)}")

        message_pkg = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": topics_str}]


        # Add try-except error catching here
        try:
            response = self.azure_client.chat.completions.create(
                # model="gpt-35-turbo-16k",
                model=self.azure_model,
                messages=message_pkg,
                temperature=self.temperature,
                max_tokens=self.max_reply_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.log(f"An error occurred: {e}")
            return None
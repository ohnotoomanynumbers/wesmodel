# wesmodel

wesmodel is a Python package designed to perform topic modeling on a natural language dataset using LLM (Large Language Model) technology. It is currently written to use an AzureOpenAI client to make API calls.

## Installation

You can install wesmodel from the gitlab page with pip:

```bash
pip install git+https://gitlab.onefiserv.net/f9ruef2/wesmodel
```


## Usage
### Initialize NLPTopicModeler
```python
from wesmodel import NLPTopicModeler

topic_modeler = NLPTopicModeler(
    dataframe=dataframe,
    azure_client=azure_client,
    azure_model='your_model_name',
    log_path='./logs'
)
```
### Model the target column

```python
topic_modeler.model_column(
    target_column='your_target_column_name',
    topic_name='topic',
    summary_column='summary',
    category='topic',
    summarize_first=False,
    workers=1,
    temperature=0,
    max_reply_tokens=1000,
    num_topics=15,
    topic=None,
    max_input_tokens=20000
)
```

## NLPTopicModeler

`NLPTopicModeler` is the main object in wesmodel.

```python
def __init__(self,
            dataframe: pd.DataFrame,
            azure_client: AzureOpenAI,
            azure_model: str,
            log_path: str = './logs',
            ):
```

- dataframe: A pandas DataFrame containing the natural language data.
- azure_client: An instance of the AzureOpenAI client.
- azure_model: The name of the Azure LLM model to be used.
- log_path: Path to store logs (default is './logs').

### Methods

`model_column`
```python
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
```
- target_column: The column containing the text data to be analyzed.
- topic_name: Name of the column to store the topic labels (default is 'topic').
- summary_column: Name of the column to store the summarized text (default is 'summary').
- category: Either 'topic' or 'subtopic' (default is 'topic').
- summarize_first: Whether to perform text summarization before topic modeling (default is False).
- workers: Number of parallel workers to use (default is 1).
- temperature: Control the randomness of the generation process (default is 0).
- max_reply_tokens: Maximum number of tokens in the reply (default is 1000).
- num_topics: Number of topics to generate (default is 15).
- topic: When performing subtopic modeling, the supertopic that is being modeled (default is None).
- max_input_tokens: Maximum number of tokens in the input text (default is 20000).

Returns the DataFrame with the modeled data.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

```go
This README.md file provides an overview of the `wesmodel` package, including installation instructions, usage examples, and details about the `NLPTopicModeler` class and its methods.
```
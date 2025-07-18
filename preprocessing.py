import pandas as pd
import json
from sklearn.model_selection import GroupShuffleSplit
from transformers import RobertaTokenizer
import torch
import re
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

console = Console()

MAX_LENGTH = 512

def dataset_processing(json_file: json) ->  pd.core.frame.DataFrame:
    '''
    Preprocess dataset based on code vulnerability: each question has a corresponding safe and vulnerable code column, we split an entry into 2 entries
    Easier to then classify code as being vulnerable or not
    '''
    df = pd.read_json(json_file, lines=True)

    df_chosen = df[['question', 'chosen']].copy()
    df_chosen = df_chosen.rename(columns={'chosen': 'code'})
    df_chosen['isvuln'] = False

    df_rejected = df[['question', 'rejected']].copy()
    df_rejected = df_rejected.rename(columns={'rejected': 'code'})
    df_rejected['isvuln'] = True

    pdataset = pd.concat([df_chosen, df_rejected], ignore_index=True)

    # We want to have the vulnerable and correct code for the same question together for training the model
    # Hence we add a unique identifier to each entry with the same question
    pdataset['question_group_id'] = pd.factorize(pdataset['question'])[0]

    return pdataset


def shuffle_dataset(df: pd.core.frame.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Shuffle dataset so that correct code and vulnerable code for the same question remain in the same split (train/test)

    Args:
        -Unshuffled ataset
    Output:
        -Shuffle ready for model consumption
    '''

    groups = df["question_group_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    X = df.drop(columns=["is_vuln"])
    y = df["is_vuln"]

    for i, (train_index, test_index) in enumerate(splitter.split(X, y, groups)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}, group={groups[train_index]}")
        print(f"  Test:  index={test_index}, group={groups[test_index]}")

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    
    return train_df, test_df

def tokenize_dataset(df, tokenizer=None):
    '''
    tokenizes all code in a dataframe using robertatokenizer.

    args:
        df (pd.DataFrame): dataframe containing a 'code' column
        tokenizer (RobertaTokenizer, optional): tokenizer instance. if none, creates a new instance.

    returns:
        pd.DataFrame: original dataframe with additional columns for tokens
    '''

    if 'code' not in df.columns:
        console.print("[bold red]ERROR:[/bold red] df must contain a 'code' column")
        raise ValueError("df must contain a 'code' column")

    if tokenizer is None:
        console.print(" initializing tokenizer...", style="cyan")
        tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        console.print(" tokenizer: [bold]robertatokenizer[/bold] (microsoft/codebert-base)", style="cyan")

    result_df = df.copy()

    # lists to store tokenization results
    input_ids_list = []
    attention_mask_list = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("[bold]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[bold cyan]{task.completed}/{task.total}[/bold cyan]"),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        expand=True
    ) as progress:
        task = progress.add_task("[cyan]tokenizing code...[/cyan]", total=len(df))
        
        for i, code in enumerate(df['code']):
            tokens = tokenize_code(code, tokenizer)
            input_ids_list.append(tokens['input_ids'])
            attention_mask_list.append(tokens['attention_mask'])
            
            progress.update(task, advance=1)

    result_df['input_ids'] = input_ids_list
    result_df['attention_mask'] = attention_mask_list

    console.print(
        Text("tokenization completed!", style="bold green")
    )
    
    return result_df

def tokenize_code(code_text, tokenizer=None):
    '''
    tokenizes the code using robertatokenizer.

    args:
        code_text (str): code text to be tokenized
        tokenizer (RobertaTokenizer, optional): tokenizer instance. if none, creates a new instance.

    returns:
        dict: dictionary containing input_ids and attention_mask
    '''
    
    if tokenizer is None:
        tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    # extract code from markdown block if present
    # expected format: ```code```
    code_pattern = r'```(?:\w+)?\n(.*?)\n```'
    match = re.search(code_pattern, code_text, re.DOTALL)
    if match:
        code_text = match.group(1)

    encoded = tokenizer(
        code_text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

    return {
        'input_ids': encoded['input_ids'].squeeze(),
        'attention_mask': encoded['attention_mask'].squeeze()
    }

def run_preprocessing_pipeline():
    """
    Demonstrates the DataFrame at different stages of the preprocessing pipeline:
    1. Shows the original DataFrame
    2. Shows the DataFrame after initial processing
    3. Shows the DataFrame after tokenization
    4. Shows the DataFrame after decoding tokens
    """

    console.print("\n", Panel(
        Text("preprocessing pipeline", style="bold cyan"),
        border_style="cyan",
        expand=False
    ))
    
    json_file = "secure_programming_dpo.json"

    console.print("\n", Panel(
        Text("1. loading the original dataset", style="bold yellow"),
        border_style="blue",
        padding=(0, 2),
        expand=False
    ))
    
    original_df = pd.read_json(json_file, lines=True)
    console.print(original_df.head(2))
    console.print(f"columns: {original_df.columns.tolist()}", style="cyan")

    console.print("\n", Panel(
        Text("2. processing the dataset", style="bold yellow"),
        border_style="blue",
        padding=(0, 2),
        expand=False
    ))
    
    processed_df = dataset_processing(json_file)
    
    console.print("\nprocessed dataframe:", style="green")
    console.print(processed_df.head(2))
    console.print(f"\ncolumns: {processed_df.columns.tolist()}", style="cyan")

    console.print("\n", Panel(
        Text("3. initializing the tokenizer", style="bold yellow"),
        border_style="blue",
        padding=(0, 2),
        expand=False
    ))
    
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    if tokenizer != None:
        console.print("\ntokenizer initialized sucessfully: robertatokenizer (microsoft/codebert-base)", style="cyan")

    console.print("\n", Panel(
        Text("4. tokenizing the dataset", style="bold yellow"),
        border_style="blue",
        padding=(0, 2),
        expand=False
    ))
    
    tokenized_df = tokenize_dataset(processed_df, tokenizer)

    console.print("\ntokenized dataframe:", style="green")
    console.print(tokenized_df.head(2))

    # Special formatting for numbered titles
    console.print("\n", Panel(
        Text("5. decoding tokens back to text", style="bold yellow"),
        border_style="blue",
        padding=(0, 2),
        expand=False
    ))

    decoded_texts = []
    for i, row in tokenized_df.head(1).iterrows():
        decoded_text = tokenizer.decode(row['input_ids'], skip_special_tokens=True)
        decoded_texts.append(decoded_text)

    decoded_sample = tokenized_df.head(1)[['question', 'isvuln', 'code']].copy()
    decoded_sample['decoded_text'] = decoded_texts
    decoded_sample = decoded_sample.drop(columns=['question','isvuln'])
    
    console.print("\ndecoded check:", style="green")

    table = Table(
        show_header=True, 
        header_style="bold magenta", 
        border_style="blue",
        show_lines=True,
        row_styles=["dim", ""]
    )
    table.add_column("Entry", style="bold yellow", justify="center", width=5)
    table.add_column("Code", style="dim")
    table.add_column("Decoded Text", style="cyan")
    
    for i, (_, row) in enumerate(decoded_sample.iterrows(), 1):
        table.add_row(
            str(i),
            row['code'],
            row['decoded_text']
        )
    
    console.print(table)
    
    console.print("\n", Panel(
        Text("pipeline completed!", style="bold green"),
        border_style="green",
        expand=False
    ))
    
    return tokenized_df

if __name__ == "__main__":
    """
    Main entry point for the preprocessing module.
    
    When run as a script, this will:
    - With --pipelinedev flag, run the preprocessing pipeline with DataFrame visualization for the sake of development
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="preprocess dataset for code vulnerability detection")
    parser.add_argument("--pipelinedev", action="store_true",
                        help="run the preprocessing pipeline with detailed DataFrame visualization")
    
    args = parser.parse_args()
    
    if args.pipelinedev:
        tokenized_df = run_preprocessing_pipeline()
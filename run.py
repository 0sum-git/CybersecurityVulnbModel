from datasets import load_dataset
from gitrepo import get_repo_files_content
import argparse
import outputs
from rich.console import Console

console = Console()

def import_dataset():
    dataset = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")
    dataset_python = dataset['train'].filter(lambda x: x['lang'] == 'python')
    return dataset_python

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_full_name")
    args = parser.parse_args()

    console.print("[bold green]loading dataset... ⌛ [/bold green]")
    dataset_python = import_dataset()
    console.print(f"[bold blue]dataset loaded with {len(dataset_python)} examples. ✅[/bold blue]")

    console.print(f"[bold green]getting files from repo[/bold green] [yellow]{args.repo_full_name}[/yellow]...")
    files = get_repo_files_content(args.repo_full_name)
    

    outputs.display_repo_files(console,files)

if __name__ == "__main__":
    main()

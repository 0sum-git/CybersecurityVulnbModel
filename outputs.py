from rich.console import Console
from rich.syntax import Syntax
from rich.prompt import Prompt

def display_repo_files(console, files):
    filenames = list(files.keys())
    expanded = set()

    while True:
        console.print("[bold underline]repo files:[/bold underline]\n")

        for i, filename in enumerate(filenames, 1):
            marker = "[+]" if i not in expanded else "[-]"
            console.print(f"{marker} ({i}) {filename}")
            if i in expanded:
                content = files[filename]
                if filename.endswith(".py"):
                    lang = "python"
                else:
                    lang = "text"
                syntax = Syntax(content, lang, theme="monokai", line_numbers=True)
                console.print(syntax)

        console.print("\n type a number to open/close file or '0' to leave.")
        choice = Prompt.ask("option ")

        if choice.lower() == '0':
            break

        if not choice.isdigit():
            continue

        num = int(choice)
        if 1 <= num <= len(filenames):
            if num in expanded:
                expanded.remove(num)
            else:
                expanded.add(num)
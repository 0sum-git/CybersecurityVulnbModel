from github import Github

def get_repo_files_content(repo_full_name):
    """
    gets the python files from the github repo.
    """
    g = Github() 
    repo = g.get_repo(repo_full_name)
    
    contents = repo.get_contents("")
    files_content = {}

    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            if file_content.path.endswith(".py"):
                file_data = repo.get_contents(file_content.path)
                content_str = file_data.decoded_content.decode('utf-8', errors='ignore')
                files_content[file_content.path] = content_str
                

    return files_content

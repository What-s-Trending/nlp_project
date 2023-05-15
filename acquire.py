"""
A module for obtaining repo information and readme contents from the github API 
and scraping trending repos from github's trending page.

Before using this module, read through it, and follow the instructions marked
TODO.

After doing so, run it like this:

    python acquire.py

To create the `data.csv` file that contains the data.
"""
import os
import json
import time
import pandas as pd
from typing import Dict, List, Optional, Union, cast
import requests
from bs4 import BeautifulSoup

from env import github_token, github_username

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if isinstance(repo_info, dict):
        return repo_info.get("language", "")
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_readme_contents(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}/readme"
    readme_info = github_api_request(url)
    if isinstance(readme_info, dict):
        readme_text = requests.get(readme_info.get('download_url', '')).text
        return readme_text
    return ""


def get_top_100_repos() -> List[str]:
    url = "https://github.com/trending"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    repos = [repo.get('href')[1:] for repo in soup.select('.h3.lh-condensed a')]
    return repos[:100]


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": get_readme_contents(repo),
    }


def scrape_github_data() -> pd.DataFrame:
    """
    Loop through all of the repos and process them. Returns the processed data in a DataFrame.
    """
    repos = get_top_100_repos()
    data = []
    for repo in repos:
        data.append(process_repo(repo))
        time.sleep(12)  # spread out the requests over 20 minutes
    return pd.DataFrame(data)


if __name__ == "__main__":
    data = scrape_github_data()
    data.to_csv("data.csv", index=False)
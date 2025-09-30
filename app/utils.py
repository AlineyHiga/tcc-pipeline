import subprocess
import os
import requests
from rich import print

class Utils:
    @staticmethod
    def run_sonar_scanner(project_dir: str = ".") -> int:
        sonar_url = os.getenv("SONARQUBE_URL")
        sonar_token = os.getenv("SONARQUBE_TOKEN")
        project_key = os.getenv("SONAR_PROJECT_KEY")
        project_name = os.getenv("SONAR_PROJECT_NAME", project_key or "")
        sonar_sources = os.getenv("SONAR_SOURCES", "src")

        if not all([sonar_url, sonar_token, project_key]):
            print("[red]Missing SONARQUBE_URL, SONARQUBE_TOKEN or SONAR_PROJECT_KEY for sonar-scanner")
            return 1

        cmd = [
            "sonar-scanner",
            f"-Dsonar.projectKey={project_key}",
            f"-Dsonar.projectName={project_name}",
            f"-Dsonar.sources={sonar_sources}",
            f"-Dsonar.host.url={sonar_url}",
            f"-Dsonar.login={sonar_token}",
            f"-Dsonar.projectBaseDir={os.path.abspath(project_dir)}",
        ]

        sonar_branch = os.getenv("SONAR_BRANCH_NAME")
        if sonar_branch:
            cmd.append(f"-Dsonar.branch.name={sonar_branch}")

        print("[blue]Running sonar-scanner with configured connection...")
        try:
            result = subprocess.run(cmd, cwd=project_dir)
            return result.returncode
        except FileNotFoundError:
            print("[red]sonar-scanner binary not found. Please install SonarScanner or adjust your PATH.")
            return None

    @staticmethod
    def create_branch_and_commit(branch_name: str, commit_msg: str, repo_dir: str = ".") -> None:
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=repo_dir, check=True)
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_dir, check=True)
        subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=repo_dir, check=True)

    @staticmethod
    def open_github_pr(branch_name: str, title: str, body: str = "") -> None:
        token = os.environ.get("GITHUB_TOKEN")
        repo = os.environ.get("GITHUB_REPO")  # ex: "user/repo"
        if not token or not repo:
            raise ValueError("Configure GITHUB_TOKEN e GITHUB_REPO")
        
        url = f"https://api.github.com/repos/{repo}/pulls"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
        payload = {"title": title, "head": branch_name, "base": "main", "body": body}
        
        r = requests.post(url, headers=headers, json=payload)
        if r.status_code == 201:
            pr_url = r.json().get("html_url")
            print(f"[green]PR criado com sucesso: {pr_url}")
        else:
            print(f"[red]Falha ao criar PR: {r.status_code} {r.text}")

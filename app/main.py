import os
from dotenv import load_dotenv
from rich import print
from .sonarqube_client import SonarQubeClient
from .patcher import Patcher
from .a2a.requester_agent import RequesterAgent
from .utils import Utils

load_dotenv()

def main():
    # Configuration
    sonar_url = os.getenv("SONARQUBE_URL")
    sonar_token = os.getenv("SONARQUBE_TOKEN")
    project_key = os.getenv("SONAR_PROJECT_KEY")
    fixer_endpoint = os.getenv("A2A_FIXER_ENDPOINT")
    
    if not all([sonar_url, sonar_token, project_key, fixer_endpoint]):
        print("[red]Missing required environment variables")
        return
    
    # Run sonar-scanner first to refresh issues
    print("[blue]Running sonar-scanner to refresh findings...")
    scan_exit = Utils.run_sonar_scanner()
    if scan_exit is None:
        print("[yellow]sonar-scanner not found; skipping initial scan.")
    elif scan_exit != 0:
        print(f"[red]sonar-scanner failed (exit={scan_exit}). Aborting pipeline.")
        return

    # Initialize components
    sonar_client = SonarQubeClient(sonar_url, sonar_token)
    patcher = Patcher()
    requester = RequesterAgent(fixer_endpoint)
    
    print("[blue]Fetching SonarQube issues...")
    try:
        issues = sonar_client.get_issues(project_key, severities=["MAJOR", "CRITICAL"])
    except Exception as e:
        print(f"[red]Failed to connect to SonarQube: {e}")
        return
    
    if not issues:
        print("[green]No issues found!")
        return
    
    print(f"[yellow]Found {len(issues)} issues to process")
    
    for issue in issues:
        print(f"\n[cyan]Processing issue: {issue.key}")
        print(f"Rule: {issue.rule}")
        print(f"Message: {issue.message}")
        
        try:
            # Get source code
            source_code = sonar_client.get_source_code(issue.component)
            file_path = issue.component.split(':')[-1]  # Extract file path from component
            
            # Request fix from A2A agent
            print("[yellow]Requesting fix from A2A agent...")
            diff = requester.request_fix(issue, source_code, file_path)
            
            if diff:
                print("Aplicando patch...\n", diff)
                if patcher.apply_unified_diff(diff):
                    code = patcher.run_tests()
                    if code == 0:
                        print(f"✅ Patch para {issue.key} passou nos testes. Rodando sonar-scanner...")
                        sonar_exit = Utils.run_sonar_scanner()
                        if sonar_exit == 0:
                            branch_name = f"autofix/{issue.key}"
                            Utils.create_branch_and_commit(branch_name, f"AutoFix: {issue.key}")
                            Utils.open_github_pr(
                                branch_name, 
                                f"AutoFix: {issue.key}", 
                                f"Correção automática para issue {issue.key} gerada via A2A."
                            )
                        else:
                            print(f"❌ sonar-scanner falhou para {issue.key}. Encaminhar logs ao Fixer.")
                    else:
                        print(f"❌ Patch para {issue.key} falhou nos testes (exit={code}).")
                        patcher.revert_changes()
                else:
                    print(f"❌ Failed to apply patch for {issue.key}")
            else:
                print(f"❌ No patch generated for {issue.key}")
                
        except Exception as e:
            print(f"[red]Error processing {issue.key}: {e}")
            patcher.revert_changes()

if __name__ == "__main__":
    main()

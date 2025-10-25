"""SonarQube REST API client."""
import requests
from typing import List, Dict, Any
from .utils import mask_secrets


class SonarQubeClient:
    """Client for SonarQube REST API."""
    
    def __init__(self, base_url: str = None, token: str = None):
        import os
        self.base_url = (base_url or os.getenv("SONARQUBE_URL", "")).rstrip("/")
        self.token = token or os.getenv("SONARQUBE_TOKEN", "")
        self.session = requests.Session()
        if self.token:
            self.session.auth = (self.token, "")
    
    def list_issues(self, project_key: str, page_size: int = 500, status: str = "OPEN") -> List[Dict[str, Any]]:
        """List issues for a project."""
        issues = []
        page = 1
        
        while True:
            try:
                url = f"{self.base_url}/api/issues/search"
                params = {
                    "componentKeys": project_key,
                    "statuses": status,
                    "ps": page_size,
                    "p": page
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                batch_issues = data.get("issues", [])
                issues.extend(batch_issues)
                
                # Check if more pages
                total = data.get("total", 0)
                if len(issues) >= total:
                    break
                    
                page += 1
                
            except Exception as e:
                print(f"Error fetching issues: {mask_secrets(str(e))}")
                break
        
        return issues
    
    def quality_gate(self, project_key: str) -> Dict[str, Any]:
        """Get quality gate status for project."""
        try:
            url = f"{self.base_url}/api/qualitygates/project_status"
            params = {"projectKey": project_key}
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error fetching quality gate: {mask_secrets(str(e))}")
            return {"projectStatus": {"status": "ERROR"}}
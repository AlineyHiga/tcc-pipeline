import requests
from dataclasses import dataclass
from typing import List, Optional
import os
import html
import re

@dataclass
class SonarIssue:
    key: str
    rule: str
    severity: str
    component: str
    line: Optional[int]
    message: str
    textRange: Optional[dict] = None

class SonarQubeClient:
    def __init__(self, url: str = None, token: str = None):
        self.url = (url or os.getenv("SONARQUBE_URL")).rstrip('/')
        self.token = token or os.getenv("SONARQUBE_TOKEN")
        self.session = requests.Session()
        self.session.auth = (self.token, '')

    def get_issues(self, project_key: str, severities: List[str] = None) -> List[SonarIssue]:
        return self.search_issues(severities)

    def search_issues(self, severities: List[str] = None) -> List[SonarIssue]:
        project_key = os.getenv("SONAR_PROJECT_KEY")
        params = {
            'componentKeys': project_key,
            'resolved': 'false',
            'ps': 100
        }
        if severities:
            params['severities'] = ','.join(severities)
        
        response = self.session.get(f"{self.url}/api/issues/search", params=params)
        response.raise_for_status()
        
        issues = []
        for issue_data in response.json().get('issues', []):
            issues.append(SonarIssue(
                key=issue_data['key'],
                rule=issue_data['rule'],
                severity=issue_data['severity'],
                component=issue_data['component'],
                line=issue_data.get('line'),
                message=issue_data['message'],
                textRange=issue_data.get('textRange')
            ))
        return issues

    def get_source_code(self, component: str, from_line: int = 1, to_line: int = None) -> str:
        raw_params = {'key': component}
        if from_line:
            raw_params['from'] = from_line
        if to_line:
            raw_params['to'] = to_line

        # Prefer raw endpoint to avoid HTML markup
        raw_resp = self.session.get(f"{self.url}/api/sources/raw", params=raw_params)
        if raw_resp.status_code == 200 and raw_resp.text:
            return raw_resp.text

        # Fallback to show endpoint and strip HTML tags if necessary
        params = {'key': component, 'from': from_line}
        if to_line:
            params['to'] = to_line

        response = self.session.get(f"{self.url}/api/sources/show", params=params)
        response.raise_for_status()

        payload = response.json()
        if isinstance(payload, dict):
            raw_sources = payload.get('sources', [])
        else:
            raw_sources = payload

        lines = []
        for entry in raw_sources:
            if isinstance(entry, dict):
                line = entry.get('code', '')
            elif isinstance(entry, (list, tuple)) and entry:
                line = entry[-1]
            elif isinstance(entry, str):
                line = entry
            else:
                line = ''

            # Remove HTML tags / classes when Sonar returns highlighted snippets
            if '<' in line and '>' in line:
                line = re.sub(r'<[^>]+>', '', line)
            lines.append(html.unescape(line))

        return '\n'.join(lines)

def format_issue(issue: SonarIssue) -> str:
    return f"{issue.key}: {issue.rule} - {issue.message}"
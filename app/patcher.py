import subprocess
import tempfile
import os
import shutil

class Patcher:
    def __init__(self, repo_dir: str = "."):
        self.repo_dir = repo_dir
        self._git_available = shutil.which('git') is not None
        # Find git repo root
        self._find_git_root()

    def apply_unified_diff(self, diff_content: str) -> bool:
        if not self._is_git_repo():
            print("[yellow]No git repo found, applying patch manually...")
            return self._apply_manual_patch(diff_content)
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(diff_content)
                patch_file = f.name
            
            result = subprocess.run(
                ['git', 'apply', '--check', patch_file],
                cwd=self.repo_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                subprocess.run(
                    ['git', 'apply', patch_file],
                    cwd=self.repo_dir,
                    check=True
                )
                os.unlink(patch_file)
                return True
            else:
                os.unlink(patch_file)
                return self._apply_manual_patch(diff_content)
        except Exception:
            return self._apply_manual_patch(diff_content)
    
    def _apply_manual_patch(self, diff_content: str) -> bool:
        try:
            lines = diff_content.split('\n')
            file_path = None
            
            for line in lines:
                if line.startswith('--- a/'):
                    file_path = line[6:]
                elif line.startswith('+++ b/'):
                    file_path = line[6:]
                    break
            
            if not file_path:
                return False
            
            full_path = os.path.join(self.repo_dir, file_path)
            if not os.path.exists(full_path):
                return False
            
            with open(full_path, 'r') as f:
                original_content = f.read()
            
            # Simple patch application - replace entire file content
            new_lines = []
            for line in lines:
                if line.startswith('+') and not line.startswith('+++'):
                    new_lines.append(line[1:])
            
            if new_lines:
                with open(full_path, 'w') as f:
                    f.write('\n'.join(new_lines))
                return True
            
            return False
        except Exception:
            return False

    def run_tests(self) -> int:
        result = subprocess.run(
            ['python3', '-m', 'pytest', '-q', 'tests/'],
            cwd=self.repo_dir,
            capture_output=True
        )
        return result.returncode

    def revert_changes(self) -> None:
        if not self._is_git_repo():
            return
        subprocess.run(['git', 'checkout', '.'], cwd=self.repo_dir)

    def _find_git_root(self) -> None:
        current_dir = os.path.abspath(self.repo_dir)
        while current_dir != os.path.dirname(current_dir):
            if os.path.isdir(os.path.join(current_dir, '.git')):
                self.repo_dir = current_dir
                print(f"[green]Found git repo at: {self.repo_dir}")
                return
            current_dir = os.path.dirname(current_dir)
        print(f"[yellow]No git repo found, using: {self.repo_dir}")
    
    def _is_git_repo(self) -> bool:
        if not self._git_available:
            return False
        return os.path.isdir(os.path.join(self.repo_dir, '.git'))

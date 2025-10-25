"""Test isolation strategy for handling dependencies."""
import logging
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Optional, List

LOGGER = logging.getLogger(__name__)

class TestIsolationManager:
    """Manages isolated test environments with dependencies."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.venv_path: Optional[Path] = None
        self.python_path: Optional[str] = None
    
    def create_isolated_env(self, requirements: List[str] = None) -> bool:
        """Create isolated virtual environment with dependencies."""
        try:
            # Create temporary venv
            temp_dir = tempfile.mkdtemp(prefix="a2a_test_")
            self.venv_path = Path(temp_dir) / "venv"
            
            LOGGER.info(f"Creating isolated test environment at {self.venv_path}")
            
            # Create virtual environment
            venv.create(self.venv_path, with_pip=True)
            
            # Get python executable path
            if sys.platform == "win32":
                self.python_path = str(self.venv_path / "Scripts" / "python.exe")
            else:
                self.python_path = str(self.venv_path / "bin" / "python")
            
            # Install basic test dependencies
            basic_deps = ["pytest", "hypothesis"]
            if requirements:
                basic_deps.extend(requirements)
            
            # Try to install dependencies
            for dep in basic_deps:
                try:
                    result = subprocess.run([
                        self.python_path, "-m", "pip", "install", dep
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode != 0:
                        LOGGER.warning(f"Failed to install {dep}: {result.stderr}")
                except subprocess.TimeoutExpired:
                    LOGGER.warning(f"Timeout installing {dep}")
                except Exception as e:
                    LOGGER.warning(f"Error installing {dep}: {e}")
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Failed to create isolated environment: {e}")
            return False
    
    def run_test_in_isolation(self, test_file: Path) -> tuple[bool, str]:
        """Run test file in isolated environment."""
        if not self.python_path or not self.venv_path:
            return False, "No isolated environment available"
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.repo_root)
            
            # Run pytest in isolated environment
            cmd = [
                self.python_path, "-m", "pytest", 
                "-q", "--tb=short", str(test_file)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,
                env=env,
                cwd=str(self.repo_root)
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            return success, output
            
        except subprocess.TimeoutExpired:
            return False, "Test execution timeout"
        except Exception as e:
            return False, f"Test execution error: {e}"
    
    def cleanup(self):
        """Clean up isolated environment."""
        if self.venv_path and self.venv_path.exists():
            try:
                import shutil
                shutil.rmtree(self.venv_path.parent)
                LOGGER.debug("Cleaned up isolated test environment")
            except Exception as e:
                LOGGER.warning(f"Failed to cleanup test environment: {e}")

def detect_dependencies(file_path: Path) -> List[str]:
    """Detect required dependencies from Python file."""
    deps = []
    
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Common dependency mappings
        dep_map = {
            'flask': 'flask==2.0.3',
            'jinja2': 'jinja2==3.0.3', 
            'markupsafe': 'markupsafe==2.0.1',
            'requests': 'requests',
            'numpy': 'numpy',
            'pandas': 'pandas'
        }
        
        for import_name, package in dep_map.items():
            if f'import {import_name}' in content or f'from {import_name}' in content:
                deps.append(package)
        
    except Exception as e:
        LOGGER.warning(f"Failed to detect dependencies: {e}")
    
    return deps
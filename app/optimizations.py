"""Pipeline optimization utilities."""
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class TokenOptimizer:
    """Reduces token usage across pipeline components."""
    
    @staticmethod
    def create_diff_hints(original_content: str, issues: List) -> str:
        """Create minimal diff hints instead of full file."""
        hints = []
        for issue in issues[:3]:  # Limit to top 3 issues
            line = getattr(issue, 'line', None)
            rule = getattr(issue, 'rule', '')
            message = getattr(issue, 'message', '')
            
            if line:
                # Extract 3 lines around the issue
                lines = original_content.splitlines()
                start = max(0, line - 2)
                end = min(len(lines), line + 1)
                context_lines = lines[start:end]
                
                hints.append(f"Line {line} ({rule}): {message}")
                hints.append("Context:")
                for i, content in enumerate(context_lines, start + 1):
                    marker = ">>>" if i == line else "   "
                    hints.append(f"{marker} {i}: {content}")
                hints.append("")
        
        return "\n".join(hints)
    
    @staticmethod
    def create_function_summary(content: str, max_chars: int = 2000) -> str:
        """Create concise function/class summary with complete functions."""
        lines = content.splitlines()
        summary_lines = []
        current_chars = 0
        in_function = False
        function_indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Start of function/class
            if stripped.startswith(('def ', 'class ', 'async def ')):
                if current_chars + len(line) > max_chars:
                    summary_lines.append("... (truncated)")
                    break
                
                summary_lines.append(line)
                current_chars += len(line) + 1
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                continue
            
            # Inside function - include all lines until function ends
            if in_function:
                line_indent = len(line) - len(line.lstrip()) if line.strip() else function_indent + 1
                
                # Function continues if indented more than function def or empty line
                if line_indent > function_indent or not line.strip():
                    if current_chars + len(line) > max_chars:
                        summary_lines.append("... (truncated)")
                        break
                    summary_lines.append(line)
                    current_chars += len(line) + 1
                else:
                    # Function ended
                    in_function = False
                    # Don't include this line as it's start of next function/block
                    continue
            
            # Include imports and other top-level statements
            if (stripped.startswith(('import ', 'from ')) or 
                stripped.startswith('"""') or stripped.startswith("'''")):
                if current_chars + len(line) > max_chars:
                    summary_lines.append("... (truncated)")
                    break
                summary_lines.append(line)
                current_chars += len(line) + 1
        
        return "\n".join(summary_lines)
    
    @staticmethod
    def extract_target_region(content: str, line_number: int, context_lines: int = 10) -> str:
        """Extract complete function/class containing target line."""
        lines = content.splitlines()
        
        # Find the function/class that contains the target line
        function_start = None
        function_end = None
        
        # Look backwards for function/class definition
        for i in range(line_number - 1, -1, -1):
            if i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith(('def ', 'class ', 'async def ')):
                    function_start = i
                    break
        
        if function_start is not None:
            # Find end of function by looking for next function or end of file
            indent_level = len(lines[function_start]) - len(lines[function_start].lstrip())
            
            for i in range(function_start + 1, len(lines)):
                line = lines[i]
                if line.strip():  # Non-empty line
                    current_indent = len(line) - len(line.lstrip())
                    # If we hit same or lower indentation level, function ended
                    if current_indent <= indent_level:
                        function_end = i
                        break
            
            if function_end is None:
                function_end = len(lines)
            
            # Include some context before and after
            start = max(0, function_start - 3)
            end = min(len(lines), function_end + 3)
        else:
            # Fallback to original logic if no function found
            start = max(0, line_number - context_lines)
            end = min(len(lines), line_number + context_lines)
        
        region_lines = []
        for i in range(start, end):
            marker = ">>>" if i + 1 == line_number else "   "
            region_lines.append(f"{marker} {i + 1}: {lines[i]}")
        
        return "\n".join(region_lines)

class PromptOptimizer:
    """Optimizes prompts for minimal token usage."""
    
    @staticmethod
    def create_bullet_briefing(issues: List) -> str:
        """Create concise bullet-point briefing."""
        if not issues:
            return "• No issues found"
        
        bullets = []
        for i, issue in enumerate(issues[:5], 1):  # Max 5 issues
            rule = getattr(issue, 'rule', 'Unknown')
            line = getattr(issue, 'line', '?')
            message = getattr(issue, 'message', '')[:100]  # Truncate long messages
            bullets.append(f"• L{line} {rule}: {message}")
        
        if len(issues) > 5:
            bullets.append(f"• ... and {len(issues) - 5} more issues")
        
        return "\n".join(bullets)
    
    @staticmethod
    def minimize_fixer_prompt(original_prompt: str) -> str:
        """Minimize fixer prompt to essential information."""
        # Remove verbose instructions, keep only essentials
        essential_parts = []
        
        lines = original_prompt.split('\n')
        for line in lines:
            # Keep critical information
            if any(keyword in line.lower() for keyword in 
                   ['file:', 'issue:', 'line', 'original:', 'fixed:', 'error']):
                essential_parts.append(line)
        
        return "\n".join(essential_parts)

class ChunkingOptimizer:
    """Handles chunking by function/class."""
    
    @staticmethod
    def extract_functions(content: str) -> Dict[str, str]:
        """Extract individual functions/classes."""
        functions = {}
        lines = content.splitlines()
        current_function = None
        current_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Start of new function/class
            if stripped.startswith(('def ', 'class ', 'async def ')):
                # Save previous function
                if current_function and current_lines:
                    functions[current_function] = '\n'.join(current_lines)
                
                # Start new function
                current_function = stripped.split('(')[0].split(':')[0].replace('def ', '').replace('class ', '')
                current_lines = [line]
                indent_level = len(line) - len(line.lstrip())
            
            elif current_function:
                # Continue current function if indented or empty line
                line_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 1
                if line_indent > indent_level or not line.strip():
                    current_lines.append(line)
                else:
                    # End of function
                    functions[current_function] = '\n'.join(current_lines)
                    current_function = None
                    current_lines = []
        
        # Save last function
        if current_function and current_lines:
            functions[current_function] = '\n'.join(current_lines)
        
        return functions
    
    @staticmethod
    def get_relevant_chunk(functions: Dict[str, str], issue_line: int, content: str) -> str:
        """Get the most relevant function chunk for an issue."""
        lines = content.splitlines()
        if issue_line <= 0 or issue_line > len(lines):
            return content[:1000]  # Fallback to first 1000 chars
        
        # Find which function contains the issue line
        current_line = 0
        for func_name, func_content in functions.items():
            func_lines = func_content.count('\n') + 1
            if current_line < issue_line <= current_line + func_lines:
                return func_content
            current_line += func_lines
        
        # Fallback: return context around the issue line
        start = max(0, issue_line - 20)
        end = min(len(lines), issue_line + 20)
        return '\n'.join(lines[start:end])

def apply_optimizations(state: dict) -> dict:
    """Apply all optimizations to pipeline state."""
    optimizer = TokenOptimizer()
    prompt_opt = PromptOptimizer()
    chunk_opt = ChunkingOptimizer()
    
    # Get current context
    original_content = state.get('property_file_preview', '')
    issues = list(state.get('issues_for_file', []))
    
    if not original_content or not issues:
        return state
    
    # Apply optimizations
    optimized_state = state.copy()
    
    # 1. Create diff hints for fixer
    if issues:
        diff_hints = optimizer.create_diff_hints(original_content, issues)
        optimized_state['fixer_diff_hints'] = diff_hints
    
    # 2. Create function summary for property agent
    func_summary = optimizer.create_function_summary(original_content, max_chars=1500)
    optimized_state['property_file_preview'] = func_summary
    
    # 3. Create bullet briefing for requester
    bullet_briefing = prompt_opt.create_bullet_briefing(issues)
    optimized_state['requester_briefing'] = bullet_briefing
    
    # 4. Extract relevant chunks
    functions = chunk_opt.extract_functions(original_content)
    if functions and issues:
        first_issue_line = getattr(issues[0], 'line', 1)
        relevant_chunk = chunk_opt.get_relevant_chunk(functions, first_issue_line, original_content)
        optimized_state['relevant_code_chunk'] = relevant_chunk
    
    return optimized_state
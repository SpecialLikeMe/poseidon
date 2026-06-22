import os
import json
import re
import subprocess

TOOL_CATEGORIES = {
    'read_file':   'read',
    'write_file':  'write',
    'run_command': 'shell',
}

_KNOWN_TOOLS = set(TOOL_CATEGORIES.keys())



def _find_json_objects(text: str) -> list:
    """Brace-matching scanner that extracts all valid JSON objects from text."""
    results = []
    i = 0
    while i < len(text):
        if text[i] != '{':
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < len(text):
            c = text[j]
            if esc:
                esc = False
            elif c == '\\' and in_str:
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            results.append(json.loads(text[i:j + 1]))
                        except Exception:
                            pass
                        break
            j += 1
        i += 1
    return results


def parse_tool_calls(text: str) -> list:
    """Parse tool calls from model output; handles multiple formats gracefully."""
    calls, seen = [], set()

    def _add(obj):
        if isinstance(obj, dict) and obj.get('tool') in _KNOWN_TOOLS:
            k = json.dumps(obj, sort_keys=True)
            if k not in seen:
                seen.add(k)
                calls.append(obj)

    # Format 1: <tool_call>{"tool": ...}</tool_call>  (preferred)
    for m in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            _add(json.loads(m))
        except Exception:
            pass

    if calls:
        return calls

    # Format 2: ```json\n{...}\n```  or  ```\n{...}\n```
    for m in re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL):
        m = m.strip()
        if '"tool"' in m:
            try:
                _add(json.loads(m))
            except Exception:
                pass

    if calls:
        return calls

    # Format 3: Any bare JSON object in the text with a "tool" key
    for obj in _find_json_objects(text):
        _add(obj)

    return calls


def clean_response(text: str) -> str:
    """Remove tool call markup and thinking blocks from model output."""
    # Strip <think> blocks (deepseek-r1 style)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip <tool_call> blocks
    text = re.sub(r'\s*<tool_call>.*?</tool_call>\s*', ' ', text, flags=re.DOTALL)
    # Strip JSON code blocks that were parsed as tool calls
    def _remove_tool_codeblock(m):
        content = m.group(1).strip()
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and obj.get('tool') in _KNOWN_TOOLS:
                return ' '
        except Exception:
            pass
        return m.group(0)
    text = re.sub(r'```(?:json)?\s*\n?(.*?)\n?```', _remove_tool_codeblock, text, flags=re.DOTALL)
    return text.strip()


def describe_tool_call(tc: dict) -> str:
    tool = tc.get('tool', '')
    if tool == 'read_file':
        return f"read_file({tc.get('path', '')})"
    if tool == 'write_file':
        size = len(tc.get('content', '').encode())
        return f"write_file({tc.get('path', '')}, {size} bytes)"
    if tool == 'run_command':
        return f"run_command({tc.get('command', '')})"
    return f"{tool}(...)"


def execute_tool(tc: dict) -> str:
    tool = tc.get('tool', '')
    try:
        if tool == 'read_file':
            path = tc.get('path', '')
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return f"({len(content)} bytes)\n{content}"

        if tool == 'write_file':
            path    = tc.get('path', '')
            content = tc.get('content', '')
            parent  = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Wrote {len(content)} bytes to {path}"

        if tool == 'run_command':
            cmd = tc.get('command', '')
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=60
            )
            parts = []
            if r.stdout.strip():
                parts.append(r.stdout)
            if r.stderr.strip():
                parts.append(f"[stderr]\n{r.stderr}")
            parts.append(f"[exit code: {r.returncode}]")
            return '\n'.join(parts)

        return f"Unknown tool: {tool}"

    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60 seconds"
    except FileNotFoundError as e:
        return f"Error: file not found — {e}"
    except PermissionError as e:
        return f"Error: permission denied — {e}"
    except Exception as e:
        return f"Error: {e}"

import os
import json
from pyexpat.errors import messages
import re
import difflib
import ollama
import platform

from prompt_toolkit import prompt

from agent.tools import (
    TOOL_CATEGORIES,
    parse_tool_calls, clean_response,
    describe_tool_call, execute_tool,
)

_PROJECT_MAP_PROMPT = """\
You are maintaining a compact project state for an AI coding assistant.
Based on the conversation above, produce a concise structured state block.
Respond with ONLY the following markdown — no preamble, no explanation:

## Current Project State
- **Active Files:** (comma-separated list of files created or edited with backticks, or "none")
- **Current Goal:** (what the user is currently trying to accomplish)
- **Key Discoveries:** (important technical findings, errors encountered, solutions found)
- **Recent Actions:** (last 3-5 significant actions in past tense)
- **Pending:** (unresolved issues or next steps, or "none")

Be specific and concise. Use backticks for file names and code identifiers."""

_PLANNER_SYSTEM = """\
You are a task acceptance planner. Given a user request, output a JSON array of specific, verifiable acceptance criteria.
Output ONLY valid JSON — no prose, no markdown fences, no explanation.

Example output:
["Creates a Python function named greet", "Accepts a name parameter", "Returns a greeting string", "Includes a docstring"]"""

_CRITIC_SYSTEM = """\
You are a solution verifier. Compare the solution against the acceptance criteria.

Reply with exactly "PASS" if ALL criteria are satisfied.
Otherwise reply with:
FAIL
- Missing: <description of unmet criterion>

One line per unmet criterion. No suggestions, no rewrites. Under 150 tokens total."""

# Ollama native tools spec — passed as tools= parameter so the model handles
# tool calling natively instead of via text-format injection.
_OLLAMA_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'read_file',
            'description': 'Read the contents of a file from the filesystem.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {
                        'type': 'string',
                        'description': 'Absolute or relative path to the file.',
                    }
                },
                'required': ['path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'write_file',
            'description': 'Write content to a file, creating it if it does not exist.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {
                        'type': 'string',
                        'description': 'Absolute or relative path to write to.',
                    },
                    'content': {
                        'type': 'string',
                        'description': 'Text content to write.',
                    },
                },
                'required': ['path', 'content'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'run_command',
            'description': 'Execute a shell command and return its stdout/stderr output.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'Shell command to execute.',
                    }
                },
                'required': ['command'],
            },
        },
    },
]


def _strip_thinking(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _parse_native_tool_calls(raw_tcs) -> list:
    """Convert Ollama native tool_calls objects/dicts to our internal format."""
    result = []
    for tc_raw in (raw_tcs or []):
        try:
            if isinstance(tc_raw, dict):
                fn   = tc_raw.get('function', {})
                name = fn.get('name', '')
                args = fn.get('arguments', {}) or {}
            else:
                fn   = getattr(tc_raw, 'function', None)
                name = getattr(fn, 'name', '') if fn else ''
                args = getattr(fn, 'arguments', {}) if fn else {}
            if name in TOOL_CATEGORIES and isinstance(args, dict):
                call = {'tool': name}
                call.update(args)
                result.append(call)
        except Exception:
            pass
    return result


def _is_protected(msg: dict) -> bool:
    role    = msg.get('role', '')
    content = msg.get('content', '')
    if role == 'system':
        return True
    if content.startswith('Result of '):
        return True
    if '```' in content:
        return True
    return False


class agent:
    def __init__(self, model):
        self.history       = []
        self.project_map   = ""
        self.model         = model
        self.options       = {}
        self.tools_enabled = True
        self.pwc_enabled   = False
        self._file_cache   = {}
        self.on_compact    = None
        self.history.append({"role" : "system", "content" : f"You are running on the following OS: {platform.platform()}"})
    # ── Context ───────────────────────────────────────────────────────────────
    def get_context_limit(self):
        info = ollama.show(self.model)
        if hasattr(info, 'modelinfo') and info.modelinfo:
            ctx = info.modelinfo.get('llama.context_length')
            if ctx:
                return int(ctx)
        try:
            return info['details']['context_length']
        except Exception:
            return 131072

    # ── Streaming ─────────────────────────────────────────────────────────────
    def _stream_round(self, messages, use_native_tools: bool = False):
        """Stream one model turn.

        Yields (content, approx, done, prompt_eval, eval_count, native_tool_calls).
        native_tool_calls is a list of our internal dicts and only populated on the
        final chunk when the model used Ollama's native tool-calling API.
        """
        chat_kw = {"model": self.model, "messages": messages, "stream": True}
        if self.options:
            chat_kw["options"] = self.options
        if use_native_tools:
            chat_kw["tools"] = _OLLAMA_TOOLS

        approx      = 0
        native_tcs  = []

        for chunk in ollama.chat(**chat_kw):
            # ── Extract fields (handle both Pydantic and dict forms) ──────────
            try:
                content     = chunk['message']['content']
                done        = chunk.get('done', False)
                prompt_eval = chunk.get('prompt_eval_count', 0) or 0
                eval_count  = chunk.get('eval_count', 0) or 0
                raw_tcs     = chunk['message'].get('tool_calls') or []
            except Exception:
                msg_obj     = getattr(chunk, 'message', None)
                content     = getattr(msg_obj, 'content', '') or ''
                done        = getattr(chunk, 'done', False)
                prompt_eval = getattr(chunk, 'prompt_eval_count', 0) or 0
                eval_count  = getattr(chunk, 'eval_count', 0) or 0
                raw_tcs     = getattr(msg_obj, 'tool_calls', None) or []

            # Accumulate native tool calls from any chunk (some models emit mid-stream)
            parsed = _parse_native_tool_calls(raw_tcs)
            for tc in parsed:
                if tc not in native_tcs:
                    native_tcs.append(tc)

            if content:
                approx += 1

            yield content or '', approx, done, prompt_eval, eval_count, list(native_tcs)

    # ── Main ask ──────────────────────────────────────────────────────────────
    def ask_stream(self, prompt, permission_fn=None):
        """Yields (text_chunk, approx_tokens, is_done).

        Empty-string chunks keep the toolbar alive.
        The final chunk (is_done=True) carries the complete display text.
        """
        self.history.append({"role": "user", "content": prompt})

        use_tools    = self.tools_enabled and permission_fn is not None
        display      = []
        total_tokens = 0
        approx       = 0

        for _round in range(10):
            messages = self.history.copy()
            if self.project_map:
                messages.insert(0, {"role": "system", "content": self.project_map})

            chunks, p_eval, e_count = [], 0, 0
            round_native_tcs = []

            for content, a, done, pe, ec, ntcs in self._stream_round(
                messages, use_native_tools=use_tools
            ):
                if content:
                    chunks.append(content)
                approx = a
                round_native_tcs = ntcs
                yield '', approx, False
                if done:
                    p_eval, e_count = pe, ec
                    total_tokens = pe + ec or approx
                    break

            response_text = ''.join(chunks)

            # Native tool calls win; text-parsing is the fallback
            tool_calls = round_native_tcs
            if not tool_calls and use_tools:
                tool_calls = parse_tool_calls(response_text)

            if not tool_calls:
                clean = clean_response(response_text) if use_tools else response_text
                display.append(clean)
                final = '\n\n'.join(p for p in display if p.strip())
                self.history.append({"role": "assistant", "content": response_text})
                self._maybe_compact(p_eval)
                yield final, total_tokens, True
                return

            # ── Tool execution loop ───────────────────────────────────────────
            clean = clean_response(response_text)
            if clean.strip():
                display.append(clean)

            # Record assistant turn
            self.history.append({"role": "assistant", "content": response_text})

            tool_result_msgs = []
            for tc in tool_calls:
                allowed = bool(permission_fn(tc))
                desc    = describe_tool_call(tc)

                if allowed:
                    result = execute_tool(tc)
                    if len(result) > 4000:
                        result = result[:4000] + '\n…(truncated)'
                    history_entry = self._compact_tool_result(tc, result)
                    display.append(f"\n**`{desc}`**\n```\n{result}\n```")
                    tool_result_msgs.append(f"Result of {desc}:\n{history_entry}")
                else:
                    display.append(f"\n**`{desc}`** *(denied by user)*")
                    tool_result_msgs.append(f"{desc}: denied by user.")

            # Feed tool results back as a user message for the next round
            self.history.append({
                "role": "user",
                "content": '\n\n'.join(tool_result_msgs),
            })

        final = '\n\n'.join(p for p in display if p.strip())
        self.history.append({"role": "assistant", "content": final})
        yield final or "Reached tool iteration limit.", total_tokens, True

    # ── Planner ───────────────────────────────────────────────────────────────
    def _plan(self, prompt: str) -> list:
        """Generate acceptance criteria for a prompt. Returns list of criterion strings."""
        try:
            messages = [
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            chat_kw = {"model": self.model, "messages": messages}
            if self.options:
                chat_kw["options"] = self.options
            resp = ollama.chat(**chat_kw)
            try:
                text = resp["message"]["content"]
            except Exception:
                msg_obj = getattr(resp, 'message', None)
                text = getattr(msg_obj, 'content', '') or ''
            text = _strip_thinking(text).strip()
            m = re.search(r'\[.*\]', text, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return [str(c) for c in result if c]
        except Exception:
            pass
        return []

    # ── Critic ────────────────────────────────────────────────────────────────
    def _critique(self, criteria: list, candidate: str) -> tuple:
        """Evaluate candidate against criteria. Returns (passed: bool, failures: list[str])."""
        if not criteria or not candidate:
            return True, []
        try:
            criteria_text = '\n'.join(f'{i + 1}. {c}' for i, c in enumerate(criteria))
            preview = candidate if len(candidate) <= 8000 else candidate[:8000] + '\n...(truncated)'
            messages = [
                {"role": "system", "content": _CRITIC_SYSTEM},
                {"role": "user", "content": f"Acceptance Criteria:\n{criteria_text}\n\nSolution:\n{preview}"},
            ]
            chat_kw = {"model": self.model, "messages": messages}
            if self.options:
                chat_kw["options"] = self.options
            resp = ollama.chat(**chat_kw)
            try:
                text = resp["message"]["content"]
            except Exception:
                msg_obj = getattr(resp, 'message', None)
                text = getattr(msg_obj, 'content', '') or ''
            text = _strip_thinking(text).strip()
            if text.upper().startswith('PASS'):
                return True, []
            failures = []
            for line in text.splitlines():
                line = line.strip('- ').strip()
                if line and not line.upper().startswith('FAIL'):
                    failures.append(line)
            return False, failures or ['Solution does not meet all criteria']
        except Exception:
            return True, []

    # ── Revision streaming ────────────────────────────────────────────────────
    def _ask_revision_stream(self, revision_prompt: str):
        """Focused revision call: minimal context, no history update, no tools.

        Yields (chunk, tok, done) — same signature as ask_stream.
        """
        messages = []
        for msg in self.history:
            if msg.get('role') == 'system':
                messages.append(msg)
                break
        if self.project_map:
            messages.append({"role": "system", "content": self.project_map})
        messages.append({"role": "user", "content": revision_prompt})

        chunks = []
        approx = 0
        total  = 0

        for content, a, done, pe, ec, _ntcs in self._stream_round(messages, use_native_tools=False):
            if content:
                chunks.append(content)
            approx = a
            yield '', approx, False
            if done:
                total = pe + ec or approx
                break

        yield ''.join(chunks), total, True

    # ── History helper ────────────────────────────────────────────────────────
    def _update_last_assistant(self, content: str):
        """Replace the last assistant message in history with the accepted final content."""
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]['role'] == 'assistant':
                self.history[i]['content'] = content
                return

    # ── PWC orchestrator ──────────────────────────────────────────────────────
    def ask_stream_pwc(self, prompt, permission_fn=None, on_status=None):
        """Planner → Worker → Critic loop wrapping ask_stream.

        Yields same (chunk, tok, done) tuples as ask_stream.
        on_status(action_str) is called at each phase for toolbar updates.
        Falls through to ask_stream when pwc_enabled is False or the planner
        produces no criteria.
        """
        def _cb(action):
            if on_status:
                on_status(action)

        if not self.pwc_enabled:
            yield from self.ask_stream(prompt, permission_fn)
            return

        # 1 — Plan
        _cb('Planning')
        criteria = self._plan(prompt)

        if not criteria:
            _cb('Thinking')
            yield from self.ask_stream(prompt, permission_fn)
            return

        # 2 — First worker attempt (updates history normally)
        _cb('Thinking')
        candidate        = ''
        candidate_tokens = 0
        for chunk, tok, done in self.ask_stream(prompt, permission_fn):
            candidate_tokens = tok
            if done:
                candidate = chunk
            else:
                yield chunk, tok, False  # toolbar heartbeat

        # 3 — First critique
        _cb('Evaluating')
        passed, failures = self._critique(criteria, candidate)

        if passed:
            yield candidate, candidate_tokens, True
            return

        # 4 — Revision loop (up to 3 attempts)
        best_candidate = candidate
        best_tokens    = candidate_tokens

        for rev in range(1, 4):
            _cb(f'Revising ({rev}/3)')

            preview = candidate if len(candidate) <= 6000 else candidate[:6000] + '\n...(truncated)'
            rev_prompt = (
                "Acceptance Criteria:\n"
                + '\n'.join(f'- {c}' for c in criteria)
                + "\n\nCurrent Solution:\n"
                + preview
                + "\n\nCritic Feedback:\n"
                + '\n'.join(f'- {f}' for f in failures)
                + "\n\nRevise only the failing portions. Output the complete corrected solution."
            )

            candidate        = ''
            candidate_tokens = 0
            for chunk, tok, done in self._ask_revision_stream(rev_prompt):
                candidate_tokens = tok
                if done:
                    candidate = chunk
                else:
                    yield chunk, tok, False  # toolbar heartbeat

            if candidate:
                best_candidate = candidate
                best_tokens    = candidate_tokens

            _cb('Evaluating')
            passed, failures = self._critique(criteria, best_candidate)

            if passed:
                break

        final = best_candidate
        if not passed:
            final += '\n\n[Maximum verification attempts reached]'

        # Replace the first-attempt entry in history with the accepted final version
        self._update_last_assistant(final)

        yield final, best_tokens, True

    def _compact_tool_result(self, tc: dict, result: str) -> str:
        tool = tc.get('tool', '')

        if tool == 'write_file':
            path    = tc.get('path', '')
            content = tc.get('content', '')
            if path in self._file_cache:
                old = self._file_cache[path]
                diff_lines = list(difflib.unified_diff(
                    old.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f'a/{os.path.basename(path)}',
                    tofile=f'b/{os.path.basename(path)}',
                    n=2,
                ))
                if diff_lines:
                    diff_str = ''.join(diff_lines[:40])
                    entry = f"Updated {path}:\n```diff\n{diff_str}```"
                else:
                    entry = f"{path}: written (no changes)"
            else:
                entry = f"Created {path} ({len(content.encode())} bytes)"
            self._file_cache[path] = content
            return entry

        if tool == 'read_file':
            path  = tc.get('path', '')
            parts = result.split('\n', 1)
            if len(parts) > 1:
                self._file_cache[path] = parts[1]
            return result

        return result

    # ── Non-streaming convenience ─────────────────────────────────────────────
    def ask(self, prompt, permission_fn=None):
        parts, tokens = [], 0
        for chunk, tok, done in self.ask_stream(prompt, permission_fn=permission_fn):
            if chunk:
                parts.append(chunk)
            tokens = tok
        return ''.join(parts)

    # ── Compaction ────────────────────────────────────────────────────────────
    def compact(self, progress_callback=None) -> str:
        def _cb(step, desc):
            if progress_callback:
                try:
                    progress_callback(step, desc)
                except Exception:
                    pass

        _cb(1, 'Classifying messages...')

        n = len(self.history)
        if n == 0:
            return self.project_map or "Nothing to compact."

        protected = set(range(max(0, n - 4), n))
        for i, msg in enumerate(self.history):
            if _is_protected(msg):
                protected.add(i)

        summarizable = [msg for i, msg in enumerate(self.history) if i not in protected]

        if len(summarizable) < 2:
            _cb(4, 'Already minimal — nothing to compact.')
            return self.project_map or "Conversation is already minimal."

        _cb(2, f'Building project map from {len(summarizable)} messages...')

        map_msgs = []
        if self.project_map:
            map_msgs.append({
                "role": "system",
                "content": f"Previous project state:\n{self.project_map}",
            })
        map_msgs.extend(summarizable)
        map_msgs.append({"role": "user", "content": _PROJECT_MAP_PROMPT})

        try:
            chat_kw = {"model": self.model, "messages": map_msgs}
            if self.options:
                chat_kw["options"] = self.options
            resp = ollama.chat(**chat_kw)
            try:
                new_map = resp["message"]["content"]
            except Exception:
                msg_obj = getattr(resp, 'message', None)
                new_map = getattr(msg_obj, 'content', '') or ''
            new_map = _strip_thinking(new_map)
        except Exception as e:
            return f"Error: {e}"

        _cb(3, f'Compressing {len(summarizable)} → {len(protected)} messages...')

        self.history     = [msg for i, msg in enumerate(self.history) if i in protected]
        self.project_map = new_map

        _cb(4, f'Done. History: {n} → {len(self.history)} messages.')
        return new_map

    def _maybe_compact(self, prompt_tokens: int):
        try:
            limit = self.get_context_limit()
            if prompt_tokens and limit and prompt_tokens > limit * 0.85:
                if self.on_compact:
                    self.on_compact('start')
                self.compact()
                if self.on_compact:
                    self.on_compact('done')
        except Exception:
            pass

    # ── Misc ──────────────────────────────────────────────────────────────────
    def change_model(self, model: str):
        self.model = model

    def clear_history(self):
        self.history     = []
        self.project_map = ""
        self._file_cache = {}

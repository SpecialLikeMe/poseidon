#!/usr/bin/env python3
"""Poseidon – AI Agent CLI inspired by Claude Code"""

import sys
import os
import time
import asyncio
import threading
import queue
from io import StringIO
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from agent.agent import agent as Agent
from agent.tools import TOOL_CATEGORIES, describe_tool_call
import ollama

from prompt_toolkit.application import Application
from prompt_toolkit.application.run_in_terminal import run_in_terminal
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings
from prompt_toolkit.key_binding.bindings.emacs import load_emacs_bindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.layout.containers import FloatContainer, Float
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.live import Live
from rich.rule import Rule

console = Console(highlight=False, force_terminal=True)

# ── Palette ──────────────────────────────────────────────────────────────────
BLUE   = '#03a1fc'
PURPLE = '#a855f7'
GREEN  = '#22c55e'
RED    = '#ef4444'
YELLOW = '#f59e0b'
GREY   = '#9ca3af'
DIM    = '#4b5563'
DIM2   = '#374151'
TBAR   = '#4a5568'   # single colour used for every toolbar element

# ── Think levels ─────────────────────────────────────────────────────────────
THINK_LEVELS = ['off', 'low', 'medium', 'high']
THINK_TEMPS  = {'off': 0.1, 'low': 0.3, 'medium': 0.7, 'high': 1.0}
THINK_COLORS = {'off': DIM, 'low': GREEN, 'medium': YELLOW, 'high': RED}

# ── Permission levels ─────────────────────────────────────────────────────────
PERM_CATS        = ['read', 'write', 'shell']
PERM_LEVELS      = ['ask', 'allow', 'none']
PERM_COLORS      = {'ask': YELLOW, 'allow': GREEN, 'none': RED}
TOOL_CAT_COLORS  = {'read': BLUE, 'write': YELLOW, 'shell': RED}

# ── Commands ─────────────────────────────────────────────────────────────────
COMMANDS = {
    '/help':      'Show available commands',
    '/save':      'Save conversation: /save [name]',
    '/conv':      'List saved conversations',
    '/load':      'Load a conversation: /load <number or name>',
    '/clear':     'Clear conversation history and screen',
    '/model':     'Change the model: /model <name>',
    '/models':    'List available Ollama models',
    '/history':   'Show conversation history and project map',
    '/compact':   'Compact conversation into a project map (with progress bar)',
    '/pwc':       'Toggle Planner→Worker→Critic: /pwc [on|off]',
    '/info':      'Show model and session info',
    '/think':     'Set thinking effort: /think off|low|medium|high',
    '/perm':      'Tool permissions: /perm [read|write|shell|all] [ask|allow|none]',
    '/sq':        'Side question mode (separate agent, no history): /sq [question]',
    '/exit':      'Exit Poseidon',
}
_ALL_CMDS = list(COMMANDS.keys()) + ['/quit', '/summarize']

# ── Shared AI / permission state ──────────────────────────────────────────────
_SPIN = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
_ai   = {'working': False, 'tokens': 0, 'elapsed': 0.0, 'frame': 0, 'action': 'Thinking'}
_perm = {'waiting': False, 'tool_call': None, 'result_q': None}
_sq   = {'active': False, 'working': False, 'tokens': 0, 'elapsed': 0.0, 'frame': 0}


# ── Ollama helpers ────────────────────────────────────────────────────────────
def get_models():
    try:
        result = ollama.list()
        raw = getattr(result, 'models', None)
        if raw is None and isinstance(result, dict):
            raw = result.get('models', [])
        names = []
        for m in (raw or []):
            name = (getattr(m, 'model', None)
                    or getattr(m, 'name', None)
                    or (m.get('model') or m.get('name') if isinstance(m, dict) else None))
            if name:
                names.append(name)
        return names
    except Exception:
        return []

def pick_default_model():
    models = get_models()
    return models[0] if models else 'llama3.2'


# ── Async rendering helpers ───────────────────────────────────────────────────
def _render_to_str(renderable) -> str:
    try:    w = console.width
    except: w = 80
    buf = Console(file=StringIO(), force_terminal=True, width=w, highlight=False)
    buf.print(renderable)
    return buf.file.getvalue()

def _write_async(*renderables):
    """Write Rich renderables from a background thread via patch_stdout."""
    out = ''.join(_render_to_str(r) for r in renderables)
    sys.stdout.write(out)
    sys.stdout.flush()


# ── Tab completion ────────────────────────────────────────────────────────────
class PoseidonCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith('/'):
            return
        parts = text.split(None, 1)
        if len(parts) == 1:
            for cmd in _ALL_CMDS:
                if cmd.startswith(text.lower()):
                    yield Completion(cmd[len(text):], display=cmd)
        elif parts[0].lower() == '/model':
            prefix = parts[1]
            for m in get_models():
                if m.startswith(prefix):
                    yield Completion(m[len(prefix):], display=m)
        elif parts[0].lower() == '/think':
            prefix = parts[1]
            for lvl in THINK_LEVELS:
                if lvl.startswith(prefix):
                    yield Completion(lvl[len(prefix):], display=lvl)
        elif parts[0].lower() == '/load':
            prefix = parts[1]
            for c in _list_convs():
                if c['name'].startswith(prefix):
                    yield Completion(c['name'][len(prefix):], display=c['name'])
        elif parts[0].lower() == '/perm':
            words = parts[1].split()
            if len(words) == 0 or (len(words) == 1 and not parts[1].endswith(' ')):
                prefix = words[0] if words else ''
                for c in PERM_CATS + ['all']:
                    if c.startswith(prefix):
                        yield Completion(c[len(prefix):], display=c)
            else:
                prefix = words[-1] if not parts[1].endswith(' ') else ''
                for l in PERM_LEVELS:
                    if l.startswith(prefix):
                        yield Completion(l[len(prefix):], display=l)


# ── Screen helpers ────────────────────────────────────────────────────────────
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# ── Welcome slide ─────────────────────────────────────────────────────────────
def _slide_content(phase: int, model: str, ctx, num_models: int) -> Text:
    t = Text(justify='left')
    t.append('\n  P O S E I D O N\n', style=f'bold {BLUE}')
    if phase >= 2:
        t.append('  AI Agent CLI  ·  Powered by Ollama\n', style=GREY)
        t.append('\n')
    if phase >= 3:
        ctx_str = f'{ctx:,} tokens' if isinstance(ctx, int) else str(ctx)
        for label, value in [('Model', model), ('Context', ctx_str), ('Models', f'{num_models} available')]:
            t.append(f'  {label:<12}', style=GREY)
            t.append(f'{value}\n',     style='bold white')
        t.append('\n')
    if phase >= 4:
        t.append('  ')
        t.append('/help',        style=f'bold {BLUE}')
        t.append(' commands  ·  ', style=GREY)
        t.append('Esc+Enter',    style=f'bold {BLUE}')
        t.append(' newline\n',   style=GREY)
    return t

def print_welcome(model: str, ctx=None, num_models: int = 0):
    clear_screen()
    with Live(console=console, refresh_per_second=20) as live:
        for phase in range(1, 5):
            live.update(Panel(
                _slide_content(phase, model, ctx, num_models),
                border_style=BLUE, padding=(0, 2),
            ))
            time.sleep(0.14)
    console.print()


# ── Response display ──────────────────────────────────────────────────────────
def _response_panel(model: str, reply: str, tokens: int, elapsed: float,
                    border: str = PURPLE) -> Panel:
    title = Text()
    title.append(model, style=f'bold {PURPLE}' if border == PURPLE else f'bold {border}')
    if tokens:
        title.append(f'  ·  {tokens:,} tok', style=GREY)
    if elapsed > 0:
        title.append(f'  ·  {elapsed:.1f}s', style=DIM)
    return Panel(Markdown(code_theme='github-dark', markup=reply),
                 title=title, border_style=border, padding=(0, 1))

def display_response(model: str, reply: str, tokens: int = 0, elapsed: float = 0.0,
                     border: str = PURPLE):
    console.print()
    console.print(_response_panel(model, reply, tokens, elapsed, border))
    console.print()

def display_response_async(model: str, reply: str, tokens: int = 0, elapsed: float = 0.0):
    _write_async('\n', _response_panel(model, reply, tokens, elapsed), '\n')


# ── Permission request (background thread) ────────────────────────────────────
def _display_perm_request_async(tool_call: dict):
    tool  = tool_call.get('tool', '')
    cat   = TOOL_CATEGORIES.get(tool, 'shell')
    color = TOOL_CAT_COLORS.get(cat, GREY)
    t = Text()
    t.append(f'  {describe_tool_call(tool_call)}\n', style=f'bold {color}')
    t.append('  Type ', style=GREY)
    t.append('y', style=f'bold {GREEN}')
    t.append(' to allow  or  ', style=GREY)
    t.append('n', style=f'bold {RED}')
    t.append(' to deny\n', style=GREY)
    _write_async(
        Panel(t, title=Text('Permission Request', style=f'bold {YELLOW}'),
              border_style=YELLOW, padding=(0, 1))
    )


# ── Help ──────────────────────────────────────────────────────────────────────
def print_help():
    t = Text()
    for cmd, desc in COMMANDS.items():
        t.append(f'  {cmd:<14}', style=f'bold {BLUE}')
        t.append(f'  {desc}\n',  style=GREY)
    console.print(Panel(t, title=Text('Commands', style='bold white'),
                        border_style=BLUE, padding=(0, 1)))
    console.print()


# ── Compact with progress bar ─────────────────────────────────────────────────
def _run_compact(ag: Agent):
    """Run compaction and show a Rich progress bar. Safe to call from any thread."""
    result_holder = [None]

    with Progress(
        SpinnerColumn(spinner_name='dots2', style=f'fg:{BLUE}'),
        TextColumn('{task.description}', style=f'fg:{TBAR}'),
        BarColumn(complete_style=BLUE, finished_style=GREEN, bar_width=20),
        TextColumn('{task.completed}/{task.total}', style=f'fg:{DIM}'),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task('Starting compaction...', total=4)

        def on_progress(step: int, desc: str):
            progress.update(task, completed=step - 1, description=desc)

        result_holder[0] = ag.compact(progress_callback=on_progress)
        progress.update(task, completed=4, description='Done.')

    result = result_holder[0]
    console.print()
    if result.startswith('Error'):
        console.print(Text(f'  {result}', style=RED))
    else:
        console.print(Panel(
            Markdown(code_theme='github-dark', markup=result),
            title=Text('Project Map', style=f'bold {YELLOW}'),
            border_style=YELLOW, padding=(0, 1),
        ))
    console.print()


# ── Conversation storage ──────────────────────────────────────────────────────
CONV_DIR = Path(__file__).parent / 'conversations'

def _list_convs() -> list:
    """Return saved conversations sorted newest-first."""
    if not CONV_DIR.exists():
        return []
    convs = []
    for f in sorted(CONV_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            convs.append({
                'name':     data.get('name', f.stem),
                'model':    data.get('model', '?'),
                'saved_at': data.get('saved_at', ''),
                'msgs':     len([m for m in data.get('history', []) if m.get('role') != 'system']),
                'path':     f,
            })
        except Exception:
            pass
    return convs

def _sanitize_name(name: str) -> str:
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
    name = ''.join(c if c in allowed else '-' for c in name.strip()).strip('-')
    name = '-'.join(p for p in name.split('-') if p)  # collapse repeated hyphens
    return name[:60] or time.strftime('conv_%Y%m%d_%H%M%S')

def _save_conv(ag: Agent, name: str) -> Path:
    CONV_DIR.mkdir(exist_ok=True)
    path = CONV_DIR / f'{name}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'name':        name,
            'model':       ag.model,
            'saved_at':    time.strftime('%Y-%m-%dT%H:%M:%S'),
            'history':     ag.history,
            'project_map': ag.project_map,
        }, f, indent=2, ensure_ascii=False)
    return path

def _load_conv_data(spec: str) -> dict | None:
    """Load conversation JSON by 1-based index number or name. Returns None on miss."""
    convs = _list_convs()
    if spec.isdigit():
        idx = int(spec) - 1
        if 0 <= idx < len(convs):
            path = convs[idx]['path']
        else:
            return None
    else:
        stem = spec.removesuffix('.json')
        matches = [c for c in convs if c['name'] == stem]
        if not matches:
            return None
        path = matches[0]['path']
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── Command dispatch ──────────────────────────────────────────────────────────
def handle_command(text: str, ag: Agent, think_state: dict, perm_config: dict) -> bool:
    parts = text.strip().split(None, 1)
    cmd   = parts[0].lower()
    arg   = parts[1].strip() if len(parts) > 1 else ''

    if cmd == '/help':
        print_help()

    elif cmd == '/conv':
        convs = _list_convs()
        if not convs:
            console.print(Text('  No saved conversations. Use /save to save one.', style=DIM))
        else:
            console.print()
            t = Text()
            t.append(f'  {"#":<4}{"Name":<32}{"Saved":<20}{"Msgs":<6}Model\n', style=f'bold {GREY}')
            for i, c in enumerate(convs, 1):
                saved = c['saved_at'][:16].replace('T', ' ') if c['saved_at'] else '?'
                t.append(f'  {i:<4}', style=f'bold {BLUE}')
                t.append(f'{c["name"]:<32}', style='white')
                t.append(f'{saved:<20}', style=GREY)
                t.append(f'{c["msgs"]:<6}', style=GREY)
                t.append(f'{c["model"]}\n', style=DIM)
            console.print(Panel(t, title=Text('Saved Conversations', style='bold white'),
                                border_style=BLUE, padding=(0, 1)))
            console.print(Text('  /load <number or name>  to restore', style=DIM))
            console.print()

    elif cmd == '/save':
        if not ag.history or all(m.get('role') == 'system' for m in ag.history):
            console.print(Text('  Nothing to save — start a conversation first.', style=DIM))
        else:
            name = _sanitize_name(arg) if arg else time.strftime('conv_%Y%m%d_%H%M%S')
            path = _save_conv(ag, name)
            t = Text()
            t.append('✓ ', style=GREEN)
            t.append('Saved: ', style=GREY)
            t.append(name, style=f'bold {BLUE}')
            t.append(f'  ·  {path.name}', style=DIM)
            console.print(t)
            console.print()

    elif cmd == '/load':
        if not arg:
            console.print(Text('  Usage: /load <number or name>', style=DIM))
            console.print(Text('  Run /conv to list saved conversations.', style=DIM))
        else:
            data = _load_conv_data(arg)
            if data is None:
                t = Text()
                t.append('  Not found: ', style=RED)
                t.append(arg, style='white')
                t.append('  —  run ', style=GREY)
                t.append('/conv', style=f'bold {BLUE}')
                t.append(' to list', style=GREY)
                console.print(t)
            else:
                ag.history     = data.get('history', [])
                ag.project_map = data.get('project_map', '')
                ag.change_model(data.get('model', ag.model))
                msgs = len([m for m in ag.history if m.get('role') != 'system'])
                t = Text()
                t.append('✓ ', style=GREEN)
                t.append('Loaded: ', style=GREY)
                t.append(data['name'], style=f'bold {BLUE}')
                t.append(f'  ·  {msgs} messages  ·  ', style=GREY)
                t.append(data.get('model', '?'), style=DIM)
                console.print(t)
                console.print()

    elif cmd == '/clear':
        ag.clear_history()
        try:    ctx = ag.get_context_limit()
        except: ctx = 'unknown'
        print_welcome(ag.model, ctx, len(get_models()))

    elif cmd == '/models':
        models = get_models()
        if not models:
            console.print(Text('Could not reach Ollama. Is it running?', style=RED))
        else:
            console.print()
            for m in models:
                active = m == ag.model
                row = Text()
                row.append('  ')
                row.append('● ', style=GREEN if active else DIM)
                row.append(m,   style=f'bold {BLUE}' if active else GREY)
                console.print(row)
            console.print()

    elif cmd == '/model':
        if not arg:
            t = Text()
            t.append('Current model: ', style=GREY)
            t.append(ag.model, style=f'bold {BLUE}')
            console.print(t)
            console.print(Text('Usage: /model <name>', style=DIM))
        else:
            ag.change_model(arg)
            t = Text()
            t.append('✓ ', style=GREEN)
            t.append('Model → ', style=GREY)
            t.append(ag.model, style=f'bold {BLUE}')
            console.print(t)
            console.print()

    elif cmd == '/think':
        if not arg:
            lvl = think_state['level']
            t = Text()
            t.append('Think level: ', style=GREY)
            t.append(lvl, style=f'bold {THINK_COLORS[lvl]}')
            console.print(t)
            console.print(Text('Usage: /think off|low|medium|high', style=DIM))
        elif arg not in THINK_LEVELS:
            t = Text()
            t.append('Unknown level: ', style=RED)
            t.append(arg, style='white')
            t.append('  — use: ', style=GREY)
            t.append('off  low  medium  high', style=f'bold {BLUE}')
            console.print(t)
        else:
            think_state['level'] = arg
            ag.options = {'temperature': THINK_TEMPS[arg]}
            t = Text()
            t.append('✓ ', style=GREEN)
            t.append('Think → ', style=GREY)
            t.append(arg, style=f'bold {THINK_COLORS[arg]}')
            console.print(t)
            console.print()

    elif cmd == '/perm':
        words = arg.split() if arg else []
        if len(words) == 0:
            t = Text()
            t.append('  Tool permissions\n\n', style=f'bold {GREY}')
            for cat in PERM_CATS:
                lvl   = perm_config[cat]
                color = PERM_COLORS[lvl]
                t.append(f'  {cat:<8}', style=GREY)
                t.append(f'{lvl}\n',    style=f'bold {color}')
            t.append('\n  Usage: /perm [read|write|shell|all] [ask|allow|none]\n', style=DIM)
            console.print(Panel(t, title=Text('Permissions', style='bold white'), border_style=BLUE))
            console.print()
        elif len(words) == 1:
            cat = words[0].lower()
            if cat not in PERM_CATS:
                console.print(Text(f'Unknown: {cat}  —  use: read write shell', style=RED))
            else:
                lvl = perm_config[cat]
                t = Text()
                t.append(f'{cat}: ', style=GREY)
                t.append(lvl, style=f'bold {PERM_COLORS[lvl]}')
                console.print(t)
        elif len(words) == 2:
            cat, lvl = words[0].lower(), words[1].lower()
            if lvl not in PERM_LEVELS:
                console.print(Text(f'Invalid level: {lvl}  —  use: ask allow none', style=RED))
            elif cat not in PERM_CATS + ['all']:
                console.print(Text(f'Unknown: {cat}  —  use: read write shell all', style=RED))
            else:
                for c in (PERM_CATS if cat == 'all' else [cat]):
                    perm_config[c] = lvl
                t = Text()
                t.append('✓ ', style=GREEN)
                t.append(f"{'all' if cat == 'all' else cat} → ", style=GREY)
                t.append(lvl, style=f'bold {PERM_COLORS[lvl]}')
                console.print(t)
                console.print()
        else:
            console.print(Text('Usage: /perm [read|write|shell|all] [ask|allow|none]', style=DIM))

    elif cmd == '/history':
        if not ag.history and not ag.project_map:
            console.print(Text('No messages yet.', style=GREY))
            return True
        console.print()
        if ag.project_map:
            console.print(Panel(
                Markdown(code_theme='github-dark', markup=ag.project_map),
                title=Text('Project Map', style=f'bold {YELLOW}'),
                border_style=YELLOW, padding=(0, 1),
            ))
        for msg in ag.history:
            if msg['role'] == 'user':
                console.print(Panel(msg['content'],
                                    title=Text('You', style=f'bold {BLUE}'),
                                    border_style=BLUE, padding=(0, 1)))
            elif msg['role'] == 'assistant':
                console.print(Panel(Markdown(code_theme='github-dark', markup=msg['content']),
                                    title=Text('Agent', style=f'bold {PURPLE}'),
                                    border_style=PURPLE, padding=(0, 1)))
        console.print()

    elif cmd in ('/compact', '/summarize'):
        _run_compact(ag)

    elif cmd == '/pwc':
        if not arg:
            state = 'on' if ag.pwc_enabled else 'off'
            t = Text()
            t.append('PWC: ', style=GREY)
            t.append(state, style=f'bold {GREEN if ag.pwc_enabled else RED}')
            console.print(t)
            console.print(Text('Usage: /pwc [on|off]', style=DIM))
        elif arg.lower() in ('on', 'off'):
            ag.pwc_enabled = arg.lower() == 'on'
            t = Text()
            t.append('✓ ', style=GREEN)
            t.append('PWC → ', style=GREY)
            t.append(arg.lower(), style=f'bold {GREEN if ag.pwc_enabled else RED}')
            console.print(t)
            console.print()
        else:
            console.print(Text('Usage: /pwc [on|off]', style=DIM))

    elif cmd == '/info':
        try:    ctx = ag.get_context_limit()
        except: ctx = 'unknown'
        ctx_str = f'{ctx:,} tokens' if isinstance(ctx, int) else str(ctx)
        lvl = think_state['level']
        t = Text()
        for label, value, vstyle in [
            ('Model',    ag.model,                            f'bold {BLUE}'),
            ('Context',  ctx_str,                             'bold white'),
            ('Think',    lvl,                                 f'bold {THINK_COLORS[lvl]}'),
            ('Messages', str(len(ag.history)),                'bold white'),
            ('Map',      'yes' if ag.project_map else 'no',  'bold white'),
            ('Files',    str(len(ag._file_cache)),            'bold white'),
            ('Tools',    'on' if ag.tools_enabled else 'off', 'bold white'),
            ('PWC',      'on' if ag.pwc_enabled  else 'off', f'bold {GREEN if ag.pwc_enabled else DIM}'),
        ]:
            t.append(f'  {label:<14}', style=GREY)
            t.append(f'{value}\n',     style=vstyle)
        console.print(Panel(t, title=Text('Session Info', style='bold white'), border_style=BLUE))
        console.print()

    elif cmd in ('/exit', '/quit'):
        return False

    else:
        t = Text()
        t.append('Unknown command: ', style=RED)
        t.append(cmd, style='white')
        t.append('  —  type ', style=GREY)
        t.append('/help', style=f'bold {BLUE}')
        t.append(' for help', style=GREY)
        console.print(t)

    return True


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    with console.status('Connecting to Ollama...', spinner='dots2', spinner_style=BLUE):
        model      = pick_default_model()
        ag         = Agent(model=model)
        num_models = len(get_models())
        try:    ctx = ag.get_context_limit()
        except: ctx = 'unknown'
        time.sleep(0.25)

    print_welcome(ag.model, ctx, num_models)

    think_state = {'level': 'medium'}
    ag.options  = {'temperature': THINK_TEMPS['medium']}
    perm_config = {'read': 'ask', 'write': 'ask', 'shell': 'ask'}
    input_q     = queue.Queue()

    # ── Auto-compact notification (background thread → patched stdout) ────────
    def _on_auto_compact(event: str):
        if event == 'start':
            _write_async('\n', Text('  Compacting conversation...', style=f'fg:{TBAR}'), '\n')
        elif event == 'done':
            _write_async(Text(f'  ✓ Compacted  ·  {len(ag.history)} messages retained\n',
                              style=f'fg:{GREEN}'))
    ag.on_compact = _on_auto_compact

    # ── Permission function (called from worker thread) ───────────────────────
    def permission_fn(tool_call: dict) -> bool:
        tool = tool_call.get('tool', '')
        cat  = TOOL_CATEGORIES.get(tool, 'shell')
        lvl  = perm_config.get(cat, 'ask')
        if lvl == 'allow': return True
        if lvl == 'none':  return False

        result_q = queue.Queue()
        _perm['waiting']   = True
        _perm['tool_call'] = tool_call
        _perm['result_q']  = result_q
        _display_perm_request_async(tool_call)
        try:
            app.invalidate()
        except Exception:
            pass

        try:    allowed = result_q.get(timeout=300)
        except: allowed = False
        finally:
            _perm['waiting']   = False
            _perm['tool_call'] = None
            _perm['result_q']  = None
        return allowed

    # ── AI worker thread ──────────────────────────────────────────────────────
    def ai_worker():
        while True:
            user_text = input_q.get()
            if user_text is None:
                break

            _ai['working'] = True
            _ai['tokens']  = 0
            _ai['elapsed'] = 0.0
            _ai['frame']   = 0
            _ai['action']  = 'Thinking'
            start = time.time()

            def on_status(action: str):
                _ai['action'] = action
                try:
                    app.invalidate()
                except Exception:
                    pass

            stop_tick = threading.Event()
            def tick():
                while not stop_tick.wait(0.1):
                    _ai['elapsed'] = time.time() - start
                    _ai['frame']  += 1
                    try:    app.invalidate()
                    except: pass
            ticker = threading.Thread(target=tick, daemon=True)
            ticker.start()

            chunks, total_tokens = [], 0
            try:
                for chunk, tok, done in ag.ask_stream_pwc(
                    user_text, permission_fn=permission_fn, on_status=on_status
                ):
                    if chunk:
                        chunks.append(chunk)
                    total_tokens = tok
                    if done:
                        break
            except Exception as e:
                chunks = [f'Error: {e}']

            stop_tick.set()
            ticker.join(timeout=1)

            elapsed        = time.time() - start
            _ai['working'] = False
            _ai['elapsed'] = elapsed
            _ai['tokens']  = total_tokens

            display_response_async(ag.model, ''.join(chunks), total_tokens, elapsed)
            try:    app.invalidate()
            except: pass

    worker = threading.Thread(target=ai_worker, daemon=True)
    worker.start()

    # ── Side-question worker ──────────────────────────────────────────────────
    sq_input_q = queue.Queue()

    def sq_worker():
        while True:
            question = sq_input_q.get()
            if question is None:
                break

            _sq['working'] = True
            _sq['tokens']  = 0
            _sq['elapsed'] = 0.0
            _sq['frame']   = 0
            start = time.time()

            sq_ag = Agent(model=ag.model)
            sq_ag.options = ag.options.copy()

            _write_async(
                '\n',
                Panel(question, title=Text('Side Question', style=f'bold {YELLOW}'),
                      border_style=YELLOW, padding=(0, 1)),
                '\n',
            )

            stop_tick = threading.Event()
            def _sq_tick():
                while not stop_tick.wait(0.1):
                    _sq['elapsed'] = time.time() - start
                    _sq['frame']  += 1
                    try:    app.invalidate()
                    except: pass
            threading.Thread(target=_sq_tick, daemon=True).start()

            chunks, total_tokens = [], 0
            try:
                for chunk, tok, done in sq_ag.ask_stream(question):
                    if chunk:
                        chunks.append(chunk)
                    _sq['tokens'] = tok
                    total_tokens  = tok
                    if done:
                        break
            except Exception as e:
                chunks = [f'Error: {e}']

            stop_tick.set()
            elapsed = time.time() - start

            _sq['working'] = False
            _sq['active']  = False
            _sq['tokens']  = 0

            _write_async(
                '\n',
                _response_panel('Side Answer', ''.join(chunks), total_tokens, elapsed, YELLOW),
                '\n',
            )
            try:    app.invalidate()
            except: pass

    sq_thread = threading.Thread(target=sq_worker, daemon=True)
    sq_thread.start()

    # ── Layout ────────────────────────────────────────────────────────────────

    text_buffer = Buffer(
        name='input',
        multiline=True,
        completer=PoseidonCompleter(),
        history=InMemoryHistory(),
        complete_while_typing=True,
    )

    def get_prompt_prefix():
        if _perm['waiting']:
            return FormattedText([(f'bold fg:{YELLOW}', ' [y/N] › ')])
        if _sq['active']:
            return FormattedText([(f'bold fg:{YELLOW}', ' ◈ › ')])
        return FormattedText([(f'bold fg:{BLUE}', ' › ')])

    def get_box_border_style():
        if _sq['active']:
            return f'fg:{YELLOW}'
        return f'fg:{TBAR}'

    input_window = Window(
        content=BufferControl(
            buffer=text_buffer,
            input_processors=[BeforeInput(get_prompt_prefix)],
            focusable=True,
        ),
        height=1,
        wrap_lines=False,
    )

    # Box border — VSplit([corner(1×1), fill(char='─'), corner(1×1)]) lets
    # prompt_toolkit distribute the fill width automatically; no terminal-width math needed.
    # style= accepts a callable so border colour updates live (e.g. YELLOW during permission).
    top_row = VSplit([
        Window(width=1, height=1, char='╭', style=get_box_border_style),
        Window(char='─',                    style=get_box_border_style),
        Window(width=1, height=1, char='╮', style=get_box_border_style),
    ])
    bottom_row = VSplit([
        Window(width=1, height=1, char='╰', style=get_box_border_style),
        Window(char='─',                    style=get_box_border_style),
        Window(width=1, height=1, char='╯', style=get_box_border_style),
    ])
    left_bar  = Window(width=1, char='│', style=get_box_border_style)
    right_bar = Window(width=1, char='│', style=get_box_border_style)

    def get_toolbar():
        if _perm['waiting']:
            tc   = _perm.get('tool_call') or {}
            tool = tc.get('tool', '')
            cat  = TOOL_CATEGORIES.get(tool, 'shell')
            col  = TOOL_CAT_COLORS.get(cat, TBAR)
            return FormattedText([
                (f'fg:{col}',  f'  ⚡ {tool}'),
                (f'fg:{TBAR}', '  ·  '),
                (f'fg:{GREEN}', 'y'),
                (f'fg:{TBAR}', ' allow  '),
                (f'fg:{RED}',  'n'),
                (f'fg:{TBAR}', ' deny  '),
            ])
        if _sq['active']:
            if _sq['working']:
                spin = _SPIN[_sq['frame'] % len(_SPIN)]
                tok  = _sq['tokens']
                sec  = _sq['elapsed']
                return FormattedText([
                    (f'fg:{YELLOW}', f'  {spin}  Side Question  ·  {tok:,} tok  ·  {sec:.0f}s  '),
                ])
            return FormattedText([
                (f'fg:{YELLOW}', '  ◈  Side Question  ·  type your question and press Enter  ·  Ctrl+C to cancel  '),
            ])
        if _ai['working']:
            spin   = _SPIN[_ai['frame'] % len(_SPIN)]
            tok    = _ai['tokens']
            sec    = _ai['elapsed']
            action = _ai.get('action', 'Thinking')
            return FormattedText([
                (f'fg:{TBAR}', f'  {spin}  {action}  ·  {tok:,} tok  ·  {sec:.0f}s  '),
            ])
        lvl = think_state['level']
        return FormattedText([
            (f'fg:{TBAR}', f'  {ag.model}  ·  think:{lvl}  ·  /help  '),
        ])

    layout = Layout(
        FloatContainer(
            content=HSplit([
                top_row,
                VSplit([left_bar, input_window, right_bar]),
                bottom_row,
                Window(content=FormattedTextControl(get_toolbar), height=1),
            ]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=12, scroll_offset=1),
                ),
            ],
        ),
        focused_element=input_window,
    )

    # ── Key bindings ──────────────────────────────────────────────────────────
    custom_kb = KeyBindings()

    @custom_kb.add('enter')
    async def _submit(event):
        buf  = event.app.current_buffer
        text = buf.text.strip()
        if not text:
            return

        # Permission response
        if _perm['waiting'] and _perm['result_q']:
            buf.reset()
            _perm['result_q'].put(text.lower() in ('y', 'yes'))
            return

        # In SQ mode: route input to the SQ worker (block while already working)
        if _sq['active']:
            if not _sq['working']:
                buf.reset()
                sq_input_q.put(text)
            return

        buf.reset()

        # /sq [question] — enter side-question mode, optionally pre-filling a question
        if text.lower().startswith('/sq'):
            parts = text.split(None, 1)
            arg   = parts[1].strip() if len(parts) > 1 else ''
            _sq['active'] = True
            event.app.invalidate()
            if arg:
                sq_input_q.put(arg)
            return

        if text.startswith('/'):
            result = [True]
            t = text
            def _run_cmd():
                result[0] = handle_command(t, ag, think_state, perm_config)
            await run_in_terminal(_run_cmd, in_executor=True)
            if not result[0]:
                input_q.put(None)
                event.app.exit()
        else:
            t = text
            def _print_you():
                console.print()
                console.print(Panel(t, title=Text('You', style=f'bold {BLUE}'),
                                    border_style=BLUE, padding=(0, 1)))
                console.print()
            await run_in_terminal(_print_you)
            input_q.put(t)

    @custom_kb.add('escape', 'enter', eager=True)
    def _newline(event):
        event.app.current_buffer.insert_text('\n')

    @custom_kb.add('c-c')
    async def _ctrl_c(event):
        if _perm['waiting'] and _perm['result_q']:
            _perm['result_q'].put(False)
        elif _sq['active'] and not _sq['working']:
            _sq['active'] = False
            event.app.current_buffer.reset()
            event.app.invalidate()
        else:
            event.app.current_buffer.reset()
            def _msg():
                t = Text()
                t.append('Use ', style=GREY)
                t.append('/exit', style=f'bold {BLUE}')
                t.append(' to quit.', style=GREY)
                console.print(t)
            await run_in_terminal(_msg)

    @custom_kb.add('c-d')
    def _ctrl_d(event):
        input_q.put(None)
        event.app.exit()

    all_kb = merge_key_bindings([load_basic_bindings(), load_emacs_bindings(), custom_kb])

    # ── Style ─────────────────────────────────────────────────────────────────
    pt_style = Style.from_dict({
        'completion-menu':                         f'bg:#0d1117 fg:{GREY}',
        'completion-menu.completion':              f'bg:#0d1117 fg:white',
        'completion-menu.completion.current':      f'bg:{BLUE} fg:#000000 bold',
        'completion-menu.meta.completion':         f'bg:#0d1117 fg:{DIM}',
        'completion-menu.meta.completion.current': f'bg:{BLUE} fg:#000000',
        'scrollbar.background':                    f'bg:{DIM}',
        'scrollbar.button':                        f'bg:{BLUE}',
    })

    # ── Application ───────────────────────────────────────────────────────────
    app = Application(
        layout=layout,
        key_bindings=all_kb,
        full_screen=False,
        style=pt_style,
        mouse_support=False,
    )

    with patch_stdout(raw=True):
        app.run()

    clear_screen()
    input_q.put(None)
    sq_input_q.put(None)


if __name__ == '__main__':
    main()

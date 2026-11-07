"""Deterministic documentation scanning and module-local AST evidence."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys
from . import __version__
from .phrases import CODEGENERATION, NOCODEGENERATION, GENERIC_NAMES, CAUSAL_CLASSES


def match_phrases(text, phrases):
    text = text.lower()
    matches = []
    for phrase in sorted(phrases, key=len, reverse=True):
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            matches.append((phrase, index))
            text = text[:index] + ' ' * len(phrase) + text[index + len(phrase):]
            start = index + len(phrase)
    return matches, text


def prose(text):
    lines = []
    fenced = False
    for line in text.splitlines():
        if re.match(r'^\s*```', line):
            fenced = not fenced
        elif not fenced:
            lines.append(line.replace('`', ''))
    return '\n'.join(lines)


def evidence(source):
    tree = ast.parse(source)
    aliases = {}
    shadow = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                aliases[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                aliases[a.asname or a.name] = (n.module or '') + '.' + a.name

    def name(n):
        if isinstance(n, ast.Name):
            return '' if n.id in shadow else aliases.get(n.id, n.id)
        if isinstance(n, ast.Attribute):
            base = name(n.value)
            return base + '.' + n.attr if base else ''
        return ''

    def strings(call):
        return [a.value for a in list(call.args) + [k.value for k in call.keywords]
                if isinstance(a, ast.Constant) and isinstance(a.value, str)]

    def model(call, positional=None):
        values = [k.value for k in call.keywords if k.arg == 'model']
        if positional is not None and len(call.args) > positional:
            values.append(call.args[positional])
        return any(isinstance(v, ast.Constant) and isinstance(v.value, str) and 'code' in v.value.lower() for v in values)

    def construction(call):
        if not isinstance(call, ast.Call):
            return None
        resolved = name(call.func)
        parts = resolved.split('.')
        cls = parts[-2] if parts[-1] == 'from_pretrained' and len(parts) > 1 else parts[-1]
        has_code = any('code' in s.lower() for s in strings(call))
        if parts[-1] == 'pipeline' and resolved.startswith('transformers.'):
            if call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == 'text-generation' and model(call, 1):
                return 'call'
        if cls in ('Llama', 'GPT4All') or (cls == 'AutoModelForCausalLM' and 'ctransformers' in resolved):
            return 'both' if has_code else None
        if cls in CAUSAL_CLASSES and (has_code or cls in ('CodeGenForCausalLM', 'GPTBigCodeForCausalLM')):
            return 'generate'
        return None

    instances = {}
    # Resolve simple assignment aliases to a fixed point, within this module only.
    for _ in range(len(list(ast.walk(tree))) + 1):
        changed = False
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                value = n.value
                kind = construction(value) if isinstance(value, ast.Call) else instances.get(name(value))
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for target in targets:
                    if kind and isinstance(target, ast.Name) and target.id not in shadow and instances.get(target.id) != kind:
                        instances[target.id] = kind
                        changed = True
        if not changed:
            break
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        resolved = name(n.func)
        parts = resolved.split('.')
        if parts[-1] in GENERIC_NAMES and parts[-1] not in shadow:
            if isinstance(n.func, ast.Attribute) or (isinstance(n.func, ast.Name) and n.func.id in aliases):
                return True
        if model(n) and 'chat' not in [p.lower() for p in parts]:
            if len(parts) >= 3 and parts[-2:] == ['completions', 'create']:
                return True
            if len(parts) >= 2 and parts[-2:] == ['Completion', 'create']:
                return True
        kind = construction(n.func) if isinstance(n.func, ast.Call) else instances.get(resolved)
        if kind in ('call', 'both'):
            return True
        if isinstance(n.func, ast.Attribute) and n.func.attr == 'generate':
            base = n.func.value
            kind = construction(base) if isinstance(base, ast.Call) else instances.get(name(base))
            if kind in ('generate', 'both'):
                return True
    return False


def audit(repo, ignore=(), verbose=False):
    root = Path(repo)
    if not root.is_dir():
        raise OSError('Repository is not a directory: ' + str(root))
    docs, tests, sources = [], [], []
    def log(message):
        if verbose:
            print(message, file=sys.stderr)
    log('Scope: README.md, docs/**/*.md, tests/ fixtures, and Python modules under ' + str(root))
    for directory, dirs, files in os.walk(root):
        here = Path(directory)
        dirs[:] = sorted(d for d in dirs if d not in ('.git', '__pycache__', '.venv', 'venv', 'env', '.env') and not (here / d / 'pyvenv.cfg').exists())
        for filename in sorted(files):
            path = here / filename
            rel = path.relative_to(root).as_posix()
            if any(rel.startswith(prefix) for prefix in ignore):
                log('Skipped ' + rel)
                continue
            is_doc = rel == 'README.md' or (rel.startswith('docs/') and rel.endswith('.md'))
            is_test = 'tests' in path.relative_to(root).parts[:-1]
            if is_doc or is_test or filename.endswith('.py'):
                text = path.read_text(encoding='utf-8')
                log('Scanned ' + rel)
                if is_doc:
                    docs.append((rel, prose(text)))
                elif is_test:
                    tests.append((rel, prose(text)))
                if filename.endswith('.py'):
                    try:
                        sources.append(evidence(text))
                    except SyntaxError as exc:
                        raise OSError('Cannot parse ' + rel + ': ' + str(exc)) from exc
            else:
                log('Out of scope ' + rel)
    has_evidence = any(sources)
    findings = []
    for entries, severity in ((docs, 'HIGH'), (tests, 'MEDIUM')):
        for rel, text in entries:
            denials, remainder = match_phrases(text, NOCODEGENERATION)
            claims, _ = match_phrases(remainder, CODEGENERATION)
            for phrases, contradicted, rule in ((claims, not has_evidence, 'documented-code-generation-claim-false'), (denials, has_evidence and severity == 'HIGH', 'documented-no-code-generation-claim-false')):
                if contradicted:
                    for phrase, offset in phrases:
                        findings.append(dict(rule=rule, severity=severity, verdict='FAIL', message='Documented phrase contradicts detected source evidence: ' + phrase, path=rel, phrase=phrase, offset=offset))
    return findings


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser(prog='code-generation-claim-audit')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    scan = sub.add_parser('audit')
    scan.add_argument('repo', nargs='?', default='.')
    scan.add_argument('--json', action='store_true')
    scan.add_argument('--verbose', action='store_true')
    scan.add_argument('--ignore', action='append', default=[])
    scan.add_argument('--fail-on', choices=('high', 'medium'), default='high')
    args = parser.parse_args(argv)
    try:
        findings = audit(args.repo, args.ignore, args.verbose)
    except (OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(findings, indent=2))
    elif findings:
        for item in findings:
            print('{severity} {rule} {path}: {message}'.format(**item))
    else:
        print('PASS: no contradictory claims found')
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0

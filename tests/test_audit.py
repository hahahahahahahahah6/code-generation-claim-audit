import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from code_generation_claim_audit.audit import evidence, match_phrases, prose, main, audit
from code_generation_claim_audit.phrases import CODEGENERATION, NOCODEGENERATION, GENERIC_NAMES, CAUSAL_CLASSES

EXPECTED_POS = ['code generation', 'code-generation', 'codegeneration', 'generate code', 'generates code', 'generating code', 'generated code', 'write code', 'writes code', 'writing code', 'code writer', 'code writers', 'ai code', 'ai-generated code', 'ai generated code', 'code generator', 'code generators', 'code generation model', 'code generation models', 'code generation api', 'code generation feature', 'code generation support', 'supports code generation', 'code generation capability', 'code generation tool', 'code generation system', 'code generation engine', 'code generation framework', 'code generation library', 'code generation pipeline', 'code generation algorithm', 'code generation function', 'code synthesis', 'synthesize code', 'code synthesizer', 'code completion', 'code complete', 'code autocompletion', 'autocomplete code', 'generate functions', 'generates functions', 'write functions', 'generate snippets', 'code from text', 'text to code', 'text-to-code', 'code from natural language', 'natural language to code', 'program synthesis', 'synthesize programs']
EXPECTED_NEG = ['does not generate code', 'do not generate code', "doesn't generate code", "don't generate code", 'no code generation', 'without code generation', 'never generates code', 'does not support code generation', 'code generation not supported', 'code generation unsupported', 'no code generation support', 'lacks code generation', 'missing code generation', 'code generation disabled', 'no code generation model', 'without code generation support', 'does not provide code generation', 'no automatic code generation', 'code generation not available', 'no code generation capability', 'does not perform code generation', 'never performs code generation', 'does not write code', 'without a code generation model']
EXPECTED_GENERIC = ['generate_code', 'code_generate', 'write_code', 'code_write', 'create_code', 'code_create', 'generate_function', 'function_generate', 'generate_snippet', 'snippet_generate', 'complete_code', 'code_complete', 'autocomplete_code', 'code_autocomplete', 'synthesize_code', 'code_synthesize', 'generate_program', 'program_generate', 'produce_code', 'code_produce', 'generate_source', 'source_generate', 'generate_script', 'script_generate']
EXPECTED_CLASSES = ['AutoModelForCausalLM', 'GPT2LMHeadModel', 'LlamaForCausalLM', 'MistralForCausalLM', 'Qwen2ForCausalLM', 'CodeGenForCausalLM', 'GPTBigCodeForCausalLM']

class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(CODEGENERATION, EXPECTED_POS)
        self.assertEqual(NOCODEGENERATION, EXPECTED_NEG)

    def test_generic_names_oracle(self):
        self.assertEqual(GENERIC_NAMES, EXPECTED_GENERIC)

    def test_dedicated_classes_oracle(self):
        self.assertEqual(CAUSAL_CLASSES, EXPECTED_CLASSES)

    def test_positive_routes(self):
        examples = [
            'from transformers import pipeline\np = pipeline("text-generation", model="bigcode/starcoder2-3b")\np("hi")',
            'import transformers as tf\ntf.pipeline("text-generation", "Salesforce/codegen")("hi")',
            'from transformers import pipeline as p\np("text-generation", model="code")("hi")',
            'm = CodeGenForCausalLM()\nm.generate("hi")',
            'GPTBigCodeForCausalLM().generate("hi")',
            'import transformers as tf\nm = tf.AutoModelForCausalLM.from_pretrained("CodeLlama")\nm.generate("hi")',
            'from transformers import AutoModelForCausalLM as A\nA.from_pretrained("code").generate("hi")',
            'from llama_cpp import Llama as L\nx = L(model_path="starcoder.gguf")\nx("hi")',
            'GPT4All("code.gguf")("hi")',
            'Llama("code").generate("hi")',
            'from ctransformers import AutoModelForCausalLM as A\nA.from_pretrained("code")("hi")',
            'client.completions.create(model="code-davinci-002")',
            'import openai as o\no.Completion.create(model="CODE-davinci")',
            'from openai import Completion as C\nC.create(model="code")',
        ]
        examples += ['x.' + n + '("hi")' for n in EXPECTED_GENERIC]
        examples += ['from provider import ' + n + ' as f\nf("hi")' for n in EXPECTED_GENERIC]
        examples += [c + '.from_pretrained("code").generate("hi")' for c in EXPECTED_CLASSES]
        for source in examples:
            with self.subTest(source=source):
                self.assertTrue(evidence(source))

    def test_rejections(self):
        examples = [
            'import transformers, openai, llama_cpp, gpt4all, ctransformers, torch, tensorflow',
            'from transformers import pipeline\np = pipeline("text-generation", model="gpt2")\np("hi")',
            'from transformers import pipeline\npipeline("text-generation")("hi")',
            'from transformers import pipeline\npipeline("text-generation", model="code")',
            'pipeline("text-generation", model="code")("hi")',
            'AutoModelForCausalLM.from_pretrained("gpt2").generate("hi")',
            'Llama("ordinary.gguf")("hi")',
            'client.completions.create(model="gpt2")',
            'client.chat.completions.create(model="code")',
            'openai.ChatCompletion.create(model="code")',
            'ollama.chat(model="code")',
            'black.format_str("x=1"); ruff.check(); autopep8.fix_code("x=1")',
            'flake8.check(); pylint.run(); pygments.highlight("x"); mypy.run(); pyright.check()',
            'import jedi\njedi.Script("x").complete()',
            'exec("x=1"); eval("1"); subprocess.run(["code-generator"])',
            'search.code("hi"); translator.transpile("x"); reviewer.review("x")',
            'x.generate("hi")',
            'CodeGenForCausalLM()',
            'Llama("code")',
            '"generate code"\n# write code',
            'x.generate_code',
            'generate_code("hi")',
            'from a import generate_code\ndef generate_code(x): pass\ngenerate_code("hi")',
            'from transformers import CodeGenForCausalLM\nclass CodeGenForCausalLM: pass\nCodeGenForCausalLM().generate("hi")',
            'from transformers import pipeline\ndef pipeline(*a, **k): pass\npipeline("text-generation", model="code")("hi")',
            'from llama_cpp import Llama\nclass Llama: pass\nLlama("code")("hi")',
        ]
        for source in examples:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_matching(self):
        denied, rest = match_phrases('does not generate code', NOCODEGENERATION)
        self.assertEqual(len(denied), 1)
        self.assertEqual(match_phrases(rest, CODEGENERATION)[0], [])
        for text, phrase in [('code generation model', 'code generation model'), ('generate code', 'generate code'), ('CODE GENERATION', 'code generation'), ('generate code for two-factor authentication', 'generate code')]:
            self.assertEqual(match_phrases(text, CODEGENERATION)[0], [(phrase, 0)])
        self.assertEqual(match_phrases('text–to–code', CODEGENERATION)[0], [])

    def fixture(self, docs='', source='', test=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / 'README.md').write_text(docs)
        (root / 'app.py').write_text(source)
        if test is not None:
            (root / 'tests').mkdir()
            (root / 'tests/fixture.md').write_text(test)
        return root

    def cli(self, root, *flags):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = main(['audit', str(root)] + list(flags))
        return status, out.getvalue()

    def test_good(self):
        self.assertEqual(self.cli(self.fixture('generate code', 'x.generate_code()'))[0], 0)

    def test_bad(self):
        status, output = self.cli(self.fixture('CODE GENERATION'))
        self.assertEqual(status, 2)
        self.assertIn('HIGH documented-code-generation-claim-false', output)

    def test_denial(self):
        status, output = self.cli(self.fixture('does not generate code', 'x.generate_code()'))
        self.assertEqual(status, 2)
        self.assertIn('HIGH documented-no-code-generation-claim-false', output)
        self.assertEqual(self.cli(self.fixture('does not generate code'))[0], 0)

    def test_test_only(self):
        root = self.fixture(test='generate code')
        self.assertEqual(self.cli(root)[0], 0)
        status, output = self.cli(root, '--fail-on', 'medium')
        self.assertEqual(status, 2)
        self.assertIn('MEDIUM', output)

    def test_no_claim(self):
        self.assertEqual(self.cli(self.fixture('A useful tool.'))[0], 0)

    def test_json(self):
        status, output = self.cli(self.fixture('generate code'), '--json')
        self.assertEqual(status, 2)
        item = json.loads(output)[0]
        self.assertEqual(item['verdict'], 'FAIL')
        self.assertTrue(item['message'])

    def test_version(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as error:
            main(['--version'])
        self.assertEqual(error.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), '0.1.0')

    def test_ignore(self):
        root = self.fixture('generate code')
        self.assertEqual(self.cli(root, '--ignore', 'READ')[0], 0)
        root = self.fixture('generate code', 'x.generate_code()')
        self.assertEqual(self.cli(root, '--ignore', 'app')[0], 2)

    def test_fences_inline(self):
        self.assertEqual(self.cli(self.fixture('```\ngenerate code\n```'))[0], 0)
        self.assertEqual(self.cli(self.fixture('`generate code`'))[0], 2)

    def test_split_modules(self):
        root = self.fixture('generate code', 'm = CodeGenForCausalLM()')
        (root / 'other.py').write_text('m.generate("hi")')
        self.assertEqual(self.cli(root)[0], 2)

    def test_docs_scope(self):
        root = self.fixture()
        (root / 'docs/sub').mkdir(parents=True)
        (root / 'docs/sub/guide.md').write_text('generate code')
        self.assertEqual(self.cli(root)[0], 2)
        self.assertEqual(self.cli(root, '--ignore', 'docs/')[0], 0)

    def test_venv_exclusion(self):
        root = self.fixture('generate code')
        (root / 'customenv').mkdir()
        (root / 'customenv/pyvenv.cfg').write_text('')
        (root / 'customenv/a.py').write_text('x.generate_code()')
        self.assertEqual(self.cli(root)[0], 2)

    def test_errors_verbose(self):
        root = self.fixture()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(self.cli(root, '--verbose')[0], 0)
        self.assertIn('Scanned README.md', err.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.cli(root / 'missing')[0], 1)
            with self.assertRaises(SystemExit) as error:
                main(['audit', '--fail-on', 'low'])
            self.assertEqual(error.exception.code, 1)

    def test_all_phrases_individually(self):
        for phrase in EXPECTED_POS:
            with self.subTest(phrase=phrase):
                self.assertEqual(self.cli(self.fixture(phrase))[0], 2)
        for phrase in EXPECTED_NEG:
            with self.subTest(phrase=phrase):
                self.assertEqual(self.cli(self.fixture(phrase))[0], 0)
                self.assertEqual(self.cli(self.fixture(phrase, 'x.generate_code()'))[0], 2)

    def test_pair_shadowing(self):
        for definition, source in [
            ('class GPT4All: pass', 'GPT4All("code")("hi")'),
            ('class AutoModelForCausalLM: pass', 'AutoModelForCausalLM.from_pretrained("code").generate()'),
            ('class Completion: pass', 'Completion.create(model="code")'),
            ('def f(): pass', 'f()'),
        ]:
            with self.subTest(definition=definition):
                self.assertFalse(evidence('from provider import generate_code as f\n' + definition + '\n' + source))

    def test_unrelated_instance_rejected(self):
        for source in [
            'm = CodeGenForCausalLM()\nn.generate()',
            'from transformers import pipeline\np = pipeline("text-generation", model="code")\nq("hi")',
            'x = Llama("code")\ny("hi")',
            'client.chat.Completion.create(model="code")',
        ]:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_ignore_evidence_directory(self):
        root = self.fixture('generate code')
        (root / 'vendor').mkdir()
        (root / 'vendor/a.py').write_text('x.generate_code()')
        self.assertEqual(self.cli(root)[0], 0)
        self.assertEqual(self.cli(root, '--ignore', 'vendor/')[0], 2)

    def test_readme_self_audit(self):
        root = Path(__file__).resolve().parents[1]
        text = prose((root / 'README.md').read_text())
        self.assertEqual(match_phrases(text, CODEGENERATION)[0], [])
        self.assertEqual(match_phrases(text, NOCODEGENERATION)[0], [])
        self.assertEqual(self.cli(root)[0], 0)

if __name__ == '__main__':
    unittest.main()

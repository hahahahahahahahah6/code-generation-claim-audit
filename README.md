# Documentation claim auditor

An offline, deterministic auditor for documentation promises about source
authoring. Python 3.9 or newer is required. Runtime uses only the standard
library: no network, LLM, telemetry, or writes outside requested output.

Install and run:

```sh
python3 -m pip install .
code-generation-claim-audit --version
code-generation-claim-audit audit .
python3 -m code_generation_claim_audit audit . --json
code-generation-claim-audit audit . --ignore tests/ --fail-on medium --verbose
```

The repository defaults to the current directory. Repeat the prefix exclusion
option to skip more paths. Paths are relative to the repository, with forward
slashes. Verbose decisions go to standard error; JSON stays on standard output.
JSON is an array of findings, each with verdict, message, rule, severity, path,
phrase, and offset fields. An empty array means no findings.

Exit status is 0 when no finding meets the threshold, 2 when one does, and 1
for usage or IO errors. The default threshold is high. Test fixtures have
medium severity and fail only when that threshold is requested.

## Scope and rules

The scanner reads root README.md and Markdown beneath docs. Triple-backtick
fences are excluded; inline backticks are stripped while their content remains.
It also checks fixtures beneath tests directories. Every Python module is
examined for evidence, excluding Git metadata, bytecode caches and virtual
environments. Invalid Python or unreadable input is an error.

The exact phrase vocabularies live in the package's phrases module. Denial
spans are removed before positive matches. Matching ignores case but does not
normalise hyphens, stem words, or infer paraphrases. Longest-first matching
with span blanking makes each occurrence match once under its longest phrase.
Positive promises without evidence and denials contradicted by evidence use:

```text
documented-code-generation-claim-false (HIGH)
documented-no-code-generation-claim-false (HIGH)
```

Denials with no source evidence pass. Positive phrases in fixtures without
evidence produce medium findings. No relevant phrases means a pass.

Evidence is AST-based, with import aliases resolved and locally defined
functions or classes shadowing route names. Paired construction and use must
occur in the same module and refer to the constructed instance. Supported
routes are a specialised transformer pipeline, a specialised causal model
followed by its generation method, a local runtime followed by invocation,
a hosted legacy completion request, and explicit authoring method names.
Imports, strings, comments and uncalled references alone are insufficient.
Formatting, linting, highlighting, type checking, identifier suggestions,
execution, search, translation and review alone are outside the checked feature.
Chat responses and subprocess invocations alone are also insufficient.

## Differentiation

```text
text-generation-claim-audit
```

That neighbour checks open-ended text writing; this auditor checks source
specifically.

```text
translation-claim-audit
```

That neighbour checks conversion between languages; this checks authoring new
source.

```text
fine-tuning-claim-audit
```

That neighbour checks training a model; this checks using one to author source.

```text
speech-recognition-claim-audit
```

That neighbour checks transcribing spoken audio; this checks source authoring.

## Limitations

Static pattern matching is not semantic judgement. Exact phrases miss
restatements. A text-generation pipeline whose model string names code,
paired with an instance call, counts as evidence but does not prove the
produced source is correct or runnable. A causal-LM construction whose model
string names code, paired with its generation method, counts but does not
prove that the produced source compiles.

```text
pipeline("text-generation", model="bigcode/starcoder2-3b")
AutoModelForCausalLM.from_pretrained("CodeLlama").generate(...)
x.generate_code(...)
```

The last example counts even on an unrelated object because its method name
is shared with the generic route. Generic text writing without code in the
model string is deliberately excluded: prose output does not establish source
authoring. Chat requests are excluded even when a model name contains code.
Dynamic factories, computed model names and cross-module instance use may be
missed. No model is loaded and output quality is never evaluated.

These exact phrases are ambiguous:

```text
"generate code" may describe generating a verification, access, or promo code.
"generate code for two-factor authentication" still matches.
"write code" may describe a human authoring source.
"code completion" may describe IDE identifier autocompletion.
"code writer" may describe a human author.
```

These are documented matching limitations, not claims of semantic accuracy.

## Development

```sh
python3 -m unittest discover
python3 -m build
```

The source distribution includes the discoverable test suite. License: MIT.

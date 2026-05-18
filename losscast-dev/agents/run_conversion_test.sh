#!/bin/bash
# Run the conversion agent test on nanochat examples.
#
# Usage:
#   bash losscast-dev/agents/run_conversion_test.sh
#
# Requires: claude CLI installed and authenticated

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== LossCast-Bench Conversion Agent Test ==="
echo "Repo: $REPO_ROOT"
echo ""

# Test case 1: nanochat d12
EXAMPLE="losscast-dev/examples/nanochat_d12"
CONVERTED="$EXAMPLE/converted"

echo "--- Test Case: nanochat_d12 ---"
echo "Source: $EXAMPLE/source/"
echo "Output: $CONVERTED/"
echo ""

# Clean previous output
rm -rf "$CONVERTED"
mkdir -p "$CONVERTED"

# Run Claude Code with the conversion prompt
echo "Running Claude Code conversion agent..."
claude --print "
Read the nanochat source files in $EXAMPLE/source/ (gpt.py, optim.py, base_train.py).

Then use the code-rewriter skill to convert them into LossCast-Bench format.

Simulated config: depth=12, aspect_ratio=64, 8xH100, device_batch_size=32, seq_len=2048.

Write the output to:
- $CONVERTED/config.json
- $CONVERTED/model.py
- $CONVERTED/optimizer.py

After writing, run:
  python $CONVERTED/model.py
  python $CONVERTED/optimizer.py
to verify they work.
"

echo ""
echo "--- Validation ---"

# Check files exist
for f in config.json model.py; do
    if [ -f "$CONVERTED/$f" ]; then
        echo "  ✓ $f exists"
    else
        echo "  ✗ $f MISSING"
    fi
done

# Run model.py verification
echo ""
echo "Running model.py verification..."
if uv run python "$CONVERTED/model.py" 2>&1; then
    echo "  ✓ model.py verification passed"
else
    echo "  ✗ model.py verification FAILED"
fi

# Run optimizer.py verification (if present)
if [ -f "$CONVERTED/optimizer.py" ]; then
    echo ""
    echo "Running optimizer.py verification..."
    if uv run python "$CONVERTED/optimizer.py" 2>&1; then
        echo "  ✓ optimizer.py verification passed"
    else
        echo "  ✗ optimizer.py verification FAILED"
    fi
fi

# Run schema validation
echo ""
echo "Running schema validation..."
if uv run python scripts/validate_contribution.py --run-dir "$CONVERTED" 2>&1; then
    echo "  ✓ Schema validation passed"
else
    echo "  ✗ Schema validation FAILED"
fi

# Compare config against expected
echo ""
echo "Comparing config against expected..."
if [ -f "$EXAMPLE/expected/config.json" ] && [ -f "$CONVERTED/config.json" ]; then
    uv run python -c "
import json
expected = json.load(open('$EXAMPLE/expected/config.json'))
converted = json.load(open('$CONVERTED/config.json'))

def compare(exp, conv, path=''):
    diffs = []
    if isinstance(exp, dict) and isinstance(conv, dict):
        for k in set(list(exp.keys()) + list(conv.keys())):
            p = f'{path}.{k}' if path else k
            if k not in exp:
                diffs.append(f'  EXTRA  {p}: {conv[k]}')
            elif k not in conv:
                diffs.append(f'  MISSING {p} (expected: {exp[k]})')
            else:
                diffs.extend(compare(exp[k], conv[k], p))
    elif exp != conv:
        diffs.append(f'  DIFF   {path}: expected={exp} got={conv}')
    return diffs

diffs = compare(expected, converted)
if diffs:
    print(f'Found {len(diffs)} differences:')
    for d in diffs:
        print(d)
else:
    print('  ✓ Config matches expected')
"
fi

echo ""
echo "=== Done ==="

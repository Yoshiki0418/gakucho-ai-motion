#!/usr/bin/env python3
"""Prepare this project's CFG fix without modifying Custom-T2M-Lab."""

import argparse
import hashlib
import json
from pathlib import Path

BEFORE = '''    text_emb, text_mask = text_encoder(prompts)
    uncond = None
    if generation_config.cfg_scale != 1.0:
        uncond = text_encoder([""])
'''

AFTER = '''    uncond = None
    if generation_config.cfg_scale != 1.0:
        # Both branches must place motion tokens at the same absolute positions.
        # Encode together so padding matches training-time empty-prompt dropout.
        combined_emb, combined_mask = text_encoder(prompts + [""])
        text_emb = combined_emb[:len(prompts)]
        text_mask = combined_mask[:len(prompts)]
        uncond = (combined_emb[len(prompts):], combined_mask[len(prompts):])
    else:
        text_emb, text_mask = text_encoder(prompts)
'''


def patch_inference(source: str) -> str:
    if source.count(BEFORE) == 1:
        return source.replace(BEFORE, AFTER, 1)
    if source.count(AFTER) == 1:
        return source
    raise RuntimeError(
        "T2M inference changed: review the CFG prefix fix before starting."
    )


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source-root', type=Path, default=project.parent / 'Custom-T2M-Lab'
    )
    args = parser.parse_args()
    source_path = args.source_root / 't2m_gpt/inference.py'
    source = source_path.read_text(encoding='utf-8')
    patched = patch_inference(source)
    compile(patched, str(source_path), 'exec')
    output = project / '.runtime/t2m-cfg/inference.py'
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the inode stable when a running container already has the same file.
    if not output.exists() or output.read_text(encoding='utf-8') != patched:
        temporary = output.with_suffix('.tmp')
        temporary.write_text(patched, encoding='utf-8')
        temporary.replace(output)
    provenance = {
        'source': str(source_path),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'runtime_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'fix': 'shared conditional and unconditional text prefix length',
    }
    output.with_name('provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n', encoding='utf-8'
    )
    print(f'T2M CFG runtime prepared: {output}')


if __name__ == '__main__':
    main()

"""Check CFG token positions using the actual patched inference entrypoint."""

import ast
import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "motion_runtime_preparer", PROJECT / "tools/prepare-t2m-runtime.py"
)
PREPARER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARER)


class RuntimePatchTest(unittest.TestCase):
    def test_unknown_upstream_implementation_requires_review(self):
        with self.assertRaisesRegex(RuntimeError, "inference changed"):
            PREPARER.patch_inference("a different inference implementation")

    def test_patch_is_idempotent(self):
        patched = PREPARER.patch_inference(PREPARER.BEFORE)
        self.assertEqual(PREPARER.patch_inference(patched), patched)


@unittest.skipIf(torch is None, "Run in the T2M container for tensor checks")
class CFGPrefixTest(unittest.TestCase):
    def setUp(self):
        path = PROJECT / ".runtime/t2m-cfg/inference.py"
        if not path.exists():
            self.skipTest("Prepare the T2M runtime first")
        tree = ast.parse(path.read_text())
        entry = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "generate_motion_tokens"
        )
        # Avoid importing model loaders and weights; execute the real entrypoint.
        entry.returns = None
        for arg in entry.args.args:
            arg.annotation = None
        isolated = ast.Module(body=[entry], type_ignores=[])
        namespace = {}
        exec(compile(isolated, str(path), "exec"), namespace)
        self.generate = namespace["generate_motion_tokens"]

    def encode(self, texts):
        self.calls.append(texts)
        lengths = [len(text) + 2 for text in texts]
        width = max(lengths)
        emb = torch.zeros(len(texts), width, 3)
        mask = torch.zeros(len(texts), width, dtype=torch.long)
        for row, length in enumerate(lengths):
            emb[row, :length] = row + 1
            mask[row, :length] = 1
        return emb, mask

    def capture(self, emb, mask, **kwargs):
        return emb, mask, kwargs["uncond"]

    def test_guidance_keeps_motion_positions_and_empty_prompt_mask_aligned(self):
        self.calls = []

        class GPT:
            generate = staticmethod(self.capture)

        class Config:
            cfg_scale = 7.0

        emb, mask, uncond = self.generate(
            ["手を振る", "腰を曲げてお辞儀する"], self.encode, GPT(), Config()
        )
        self.assertEqual(self.calls, [["手を振る", "腰を曲げてお辞儀する", ""]])
        self.assertEqual(emb.shape[1], uncond[0].shape[1])
        self.assertEqual(mask.shape[1], uncond[1].shape[1])
        self.assertEqual(emb.shape[0], 2)
        self.assertEqual(uncond[0].shape[0], 1)
        self.assertEqual(uncond[1].sum().item(), 2)
        self.assertTrue(
            torch.equal(uncond[1][0, 2:], torch.zeros_like(uncond[1][0, 2:]))
        )
        self.assertEqual(mask[1].sum().item(), len("腰を曲げてお辞儀する") + 2)

    def test_without_guidance_does_not_change_encoding(self):
        self.calls = []

        class GPT:
            generate = staticmethod(self.capture)

        class Config:
            cfg_scale = 1.0

        _, _, uncond = self.generate("手を振る", self.encode, GPT(), Config())
        self.assertEqual(self.calls, [["手を振る"]])
        self.assertIsNone(uncond)


if __name__ == "__main__":
    unittest.main()

"""Model tool notation must not turn into spoken instructions or JSON."""

import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'app/agent/general_conversation/tools/motion_output.py'
SPEC = importlib.util.spec_from_file_location('motion_output_test', PATH)
OUTPUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OUTPUT)


class MotionOutputTest(unittest.TestCase):
    def test_actual_malformed_tool_output_keeps_only_the_natural_reply(self):
        text = ('<functions.generate_motion qq\u5f69\u7968{"motion_description":"お辞儀する"}>'
                'assistant to=functions.generate_motion in commentary \u5929\u5929\u9001JSON\n'
                '{"motion_description":"お辞儀する"}\ub2c8\ub2e4承知しました。')
        self.assertEqual(OUTPUT.clean_motion_tool_artifacts(text), ('承知しました。', False))

    def test_streamed_json_punctuation_waits_until_the_tool_block_is_complete(self):
        prefix = 'はい。<functions.generate_motion {"motion_description":"お辞儀する。'
        self.assertEqual(OUTPUT.clean_motion_tool_artifacts(prefix), ('はい。', True))
        complete = prefix + '"}>実行します。'
        self.assertEqual(OUTPUT.clean_motion_tool_artifacts(complete), ('はい。実行します。', False))

    def test_nested_payload_and_escaped_braces_do_not_consume_the_reply(self):
        text = ('assistant to=functions.generate_motion in commentary JSON\n'
                '{"value":{"text":"brace } and \\\"quote\\\" >"}}完了しました。')
        self.assertEqual(OUTPUT.clean_motion_tool_artifacts(text), ('完了しました。', False))

    def test_truncated_tool_output_is_removed_at_end_of_stream(self):
        text = 'はい。assistant to=functions.generate_motion in commentary JSON\n{"motion_description":'
        self.assertEqual(OUTPUT.clean_motion_tool_artifacts(text, final=True), ('はい。', False))

    def test_tool_name_split_after_a_natural_sentence_is_buffered(self):
        for tail in ('<functions.gen', 'assistant to=functions.generate_m'):
            text = '承知しました。' + tail
            self.assertEqual(OUTPUT.clean_motion_tool_artifacts(text), ('承知しました。', True))
            self.assertEqual(OUTPUT.clean_motion_tool_artifacts(text, final=True), ('承知しました。', False))

    def test_every_two_chunk_boundary_keeps_tool_blocks_out_of_spoken_sentences(self):
        text = ('はい。<functions.generate_motion {"motion_description":"お辞儀する。"}>'
                'assistant to=functions.generate_motion in commentary JSON\n'
                '{"motion_description":"お辞儀する。"}\ub2c8\ub2e4承知しました。')
        for boundary in range(1, len(text)):
            buffer = ''
            spoken = []
            for chunk in (text[:boundary], text[boundary:]):
                buffer += chunk
                clean, incomplete = OUTPUT.clean_motion_tool_artifacts(buffer)
                if incomplete:
                    continue
                buffer = clean
                if any(mark in chunk for mark in '。！？!?'):
                    spoken.append(buffer)
                    buffer = ''
            clean, _ = OUTPUT.clean_motion_tool_artifacts(buffer, final=True)
            spoken.append(clean)
            with self.subTest(boundary=boundary):
                self.assertEqual(''.join(spoken), 'はい。承知しました。')

    def test_normal_short_words_and_symbols_are_preserved_at_end_of_stream(self):
        for text in ('assistant', '値は <', 'a', 'finish'):
            self.assertEqual(OUTPUT.clean_motion_tool_artifacts(text, final=True), (text, False))

    def test_suffix_split_across_chunks_is_removed_only_after_tool_json(self):
        prefix = 'assistant to=functions.generate_motion in commentary JSON {"text":"挙手"}'
        self.assertEqual(OUTPUT.clean_motion_tool_artifacts(prefix + '\ub2c8'), ('', True))
        self.assertEqual(OUTPUT.clean_motion_tool_artifacts(prefix + '\ub2c8', final=True), ('', False))
        self.assertEqual(OUTPUT.clean_motion_tool_artifacts(prefix + '\ub2c8\ub2e4はい。'), ('はい。', False))

    def test_ordinary_json_and_dialogue_are_preserved(self):
        text = '説明します。{"名前":"学長"} 감사합니다。'
        self.assertEqual(OUTPUT.clean_motion_tool_artifacts(text), (text, False))


if __name__ == '__main__':
    unittest.main()

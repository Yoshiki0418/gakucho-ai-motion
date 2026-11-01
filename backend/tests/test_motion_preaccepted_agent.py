"""Queued motions retain normal conversation without duplicate motion tools."""

import ast
import contextvars
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator
import unittest


@dataclass
class FakeAgent:
    instructions: str
    tools: list
    handoffs: list

    def clone(self, **kwargs):
        return replace(self, **kwargs)


class FakeRunner:
    calls = []

    @classmethod
    async def run(cls, agent, input_messages):
        cls.calls.append((agent, input_messages))
        return SimpleNamespace(final_output='質問への回答です。')

    @classmethod
    def run_streamed(cls, agent, input_messages):
        cls.calls.append((agent, input_messages))

        async def stream_events():
            yield SimpleNamespace(
                type='raw_response_event',
                data=SimpleNamespace(type='response.output_text.delta', delta='質問への回答です。'),
            )

        return SimpleNamespace(stream_events=stream_events)


CHANNEL = contextvars.ContextVar('preaccepted_motion_test_channel', default=None)
SOURCE = Path(__file__).resolve().parents[1] / 'app/agent/general_conversation/agent.py'
TREE = ast.parse(SOURCE.read_text())
HELPER = next(node for node in TREE.body
              if isinstance(node, ast.FunctionDef) and node.name == '_agent_for_preaccepted_motion')
AGENT_CLASS = next(node for node in TREE.body
                   if isinstance(node, ast.ClassDef) and node.name == 'GeneralConversationAgent')
AGENT_CLASS.body = [node for node in AGENT_CLASS.body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name in ('generate', 'stream_generate')]
ISOLATED = ast.Module(body=[HELPER, AGENT_CLASS], type_ignores=[])
NAMESPACE = {'Agent': FakeAgent, 'Runner': FakeRunner, 'AsyncIterator': AsyncIterator,
             'motion_data_queue_var': CHANNEL}
exec(compile(ast.fix_missing_locations(ISOLATED), str(SOURCE), 'exec'), NAMESPACE)
agent_for_motion = NAMESPACE['_agent_for_preaccepted_motion']
ConversationAgent = NAMESPACE['GeneralConversationAgent']


class PreacceptedMotionAgentTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.motion_tool = SimpleNamespace(name='generate_motion')
        self.weather_tool = SimpleNamespace(name='get_weather')
        self.time_tool = SimpleNamespace(name='get_current_time')
        self.handoff = object()
        self.general = FakeAgent('一般対話の指示', [self.motion_tool, self.weather_tool, self.time_tool], [self.handoff])
        self.ceremony = FakeAgent('式典対話の指示', list(self.general.tools), [self.handoff])
        self.conversation = ConversationAgent.__new__(ConversationAgent)
        self.conversation.general_agent = self.general
        self.conversation.ceremony_agent = self.ceremony
        FakeRunner.calls = []

    def test_clone_keeps_global_agent_nonmotion_tools_and_handoffs(self):
        scoped = agent_for_motion(self.general, '腰を曲げて礼をする')
        self.assertIsNot(scoped, self.general)
        self.assertEqual(scoped.tools, [self.weather_tool, self.time_tool])
        self.assertEqual(self.general.tools, [self.motion_tool, self.weather_tool, self.time_tool])
        self.assertEqual(self.general.instructions, '一般対話の指示')
        self.assertIs(scoped.handoffs, self.general.handoffs)
        self.assertTrue(scoped.instructions.startswith(self.general.instructions))
        self.assertIn('腰を曲げて礼をする', scoped.instructions)
        self.assertIn('行動タグも追加しない', scoped.instructions)
        self.assertIs(agent_for_motion(self.general, None), self.general)

    async def test_preaccepted_context_preserves_mixed_question_and_history_in_both_paths(self):
        history = [{'role': 'user', 'content': '大学について相談したいです。'}]
        original_history = [dict(message) for message in history]
        message = '手を振ってから、今日の天気を教えてください。'
        token = CHANNEL.set(SimpleNamespace(explicit_prompt='右手を振る'))
        try:
            self.assertEqual(await self.conversation.generate('user', message, history), '質問への回答です。')
            parts = [part async for part in self.conversation.stream_generate(
                'user', message, history, mode='ceremony')]
            self.assertEqual(parts, ['質問への回答です。'])
        finally:
            CHANNEL.reset(token)
        for (scoped, messages), original in zip(FakeRunner.calls, [self.general, self.ceremony]):
            self.assertIsNot(scoped, original)
            self.assertEqual(scoped.tools, [self.weather_tool, self.time_tool])
            self.assertEqual(messages, original_history + [{'role': 'user', 'content': message}])
            self.assertIs(scoped.handoffs, original.handoffs)
        self.assertEqual(history, original_history)

    async def test_ordinary_turn_keeps_original_agents_and_motion_tool(self):
        token = CHANNEL.set(SimpleNamespace(explicit_prompt=None))
        try:
            await self.conversation.generate('user', 'こんにちは')
            _ = [part async for part in self.conversation.stream_generate('user', 'こんにちは')]
        finally:
            CHANNEL.reset(token)
        for selected, messages in FakeRunner.calls:
            self.assertIs(selected, self.general)
            self.assertEqual(messages, 'こんにちは')
            self.assertIn(self.motion_tool, selected.tools)


if __name__ == '__main__':
    unittest.main()

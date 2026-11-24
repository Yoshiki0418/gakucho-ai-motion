"""Model selection is per turn; local inference never receives the OpenAI credential."""

import os
from unittest import TestCase, mock

from app.chat_models import ChatModel, chat_client, model_settings, select_model


class ChatModelSettingsTest(TestCase):
    def setUp(self):
        env = mock.patch.dict(os.environ, {"LOCAL_LLM_MODEL": "lab-model"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(chat_client.cache_clear)
        chat_client.cache_clear()

    def test_automatic_selection_with_and_without_a_key(self):
        self.assertEqual(select_model(), ChatModel("local", "lab-model"))
        os.environ["OPENAI_API_KEY"] = "   "
        self.assertFalse(model_settings()["openai_available"])
        os.environ.update(OPENAI_API_KEY="cloud-test-value", OPENAI_MODEL="cloud-model")
        self.assertEqual(select_model(), ChatModel("openai", "cloud-model"))
        self.assertEqual(
            select_model("local", "another-model"), ChatModel("local", "another-model")
        )
        self.assertEqual(select_model(), ChatModel("openai", "cloud-model"))
        self.assertNotIn("cloud-test-value", str(model_settings()))

    def test_reject_invalid_requests(self):
        with self.assertRaises(RuntimeError):
            select_model("openai")
        for provider, name in [
            ("other", "example"),
            ("local", "x" * 257),
            ("local", "one\ntwo"),
        ]:
            with self.subTest(provider=provider, name=name), self.assertRaises(
                ValueError
            ):
                select_model(provider, name)

    @mock.patch("app.chat_models.AsyncOpenAI")
    def test_local_client_uses_its_own_endpoint_and_credential(self, sdk):
        os.environ.update(
            OPENAI_API_KEY="cloud-only", LOCAL_LLM_BASE_URL="http://inference:8000/v1"
        )
        chat_client("local")
        sdk.assert_called_once_with(
            base_url="http://inference:8000/v1", api_key="local"
        )
        chat_client.cache_clear()
        os.environ["LOCAL_LLM_API_KEY"] = "local-only"
        chat_client("local")
        sdk.assert_called_with(
            base_url="http://inference:8000/v1", api_key="local-only"
        )

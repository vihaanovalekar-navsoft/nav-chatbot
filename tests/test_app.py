import contextlib
import io
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import psutil
from fastapi.testclient import TestClient

import app as web
import navai.server as server


class FakeChat:
    def __init__(self):
        self.history = None

    def answer(self, question, history=None):
        self.history = history
        return {"answer": f"Echo: {question}", "sources": []}


class AppTest(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.object(web, "NavAIChat", FakeChat)
        self.patcher.start()
        self.client = TestClient(web.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.patcher.stop()

    def test_serves_chat_page(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="chat-form"', response.text)
        self.assertIn('id="new-chat"', response.text)

    def test_answers_chat_message(self):
        response = self.client.post("/api/chat", json={"message": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"answer": "Echo: hello", "sources": []}
        )

    def test_passes_history_to_pipeline(self):
        history = [
            {"role": "user", "content": "How does NavAI help retailers?"},
            {"role": "assistant", "content": "It helps with analytics."},
        ]

        response = self.client.post(
            "/api/chat",
            json={"message": "What about pricing?", "history": history},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(web.app.state.chat.history, history)

    def test_rejects_too_much_history(self):
        history = [
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "x" * 2000},
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "x" * 2000},
            {"role": "user", "content": "x" * 2000},
            {"role": "assistant", "content": "x" * 2000},
            {"role": "user", "content": "x"},
        ]

        response = self.client.post(
            "/api/chat",
            json={"message": "hello", "history": history},
        )

        self.assertEqual(response.status_code, 422)

    def test_rejects_too_many_history_messages(self):
        history = [
            {"role": "user", "content": "x"} for _ in range(13)
        ]

        response = self.client.post(
            "/api/chat",
            json={"message": "hello", "history": history},
        )

        self.assertEqual(response.status_code, 422)

    def test_rejects_blank_message(self):
        response = self.client.post("/api/chat", json={"message": "   "})

        self.assertEqual(response.status_code, 400)

    def test_rejects_missing_message(self):
        response = self.client.post("/api/chat", json={})

        self.assertEqual(response.status_code, 422)


class StartupTest(unittest.TestCase):
    def test_clear_port_ignores_vanished_listener(self):
        connection = SimpleNamespace(
            laddr=SimpleNamespace(port=8000),
            status=psutil.CONN_LISTEN,
            pid=999999,
        )

        with patch.object(
            server.psutil, "net_connections", return_value=[connection]
        ), patch.object(
            server.psutil, "Process", side_effect=psutil.NoSuchProcess(999999)
        ):
            server.clear_port(8000)

    def test_browser_opens_after_server_starts(self):
        server_state = SimpleNamespace(started=False, should_exit=False)
        opened = []
        output = io.StringIO()

        with contextlib.redirect_stdout(output), patch.object(
            server.webbrowser, "open", side_effect=opened.append
        ):
            thread = threading.Thread(
                target=server.open_browser_when_started,
                args=(server_state,),
                daemon=True,
            )
            thread.start()
            time.sleep(0.1)
            self.assertEqual(opened, [])
            server_state.started = True
            thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(opened, [server.URL])
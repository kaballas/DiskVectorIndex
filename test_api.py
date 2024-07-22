import unittest
from flask import json
from api import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_ask_question_with_valid_question(self):
        question = "What is the capital of France?"
        response = self.app.post('/ask', json={'question': question})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('response', data)
        self.assertIsInstance(data['response'], str)

    def test_ask_question_with_empty_question(self):
        response = self.app.post('/ask', json={'question': ''})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No question provided')

    def test_ask_question_with_no_question_field(self):
        response = self.app.post('/ask', json={})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No question provided')

if __name__ == '__main__':
    unittest.main()
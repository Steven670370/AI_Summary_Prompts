# test_similarity.py

import numpy as np
from unittest.mock import MagicMock, patch


def test_cosine_similarity():
    from similarity import cosine_similarity
    
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
    
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6
    
    a = np.zeros(3)
    b = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == 0.0
    
    print("cosine_similarity tests passed!")


def test_parse_sub_questions():
    from similarity import _parse_sub_questions
    
    text = """## Sub-questions
1. What is Python?
2. How do you install it?
3. What are its main features?"""
    
    result = _parse_sub_questions(text)
    assert len(result) == 3
    assert "What is Python?" in result
    
    print("_parse_sub_questions tests passed!")


def test_predict_response_length_no_data():
    from similarity import predict_response_length
    
    class MockTokenizer:
        def encode(self, word):
            return hash(word) % 1000
    
    class MockModel:
        class embedding:
            @staticmethod
            def get_embeddings(tokens):
                return np.zeros((len(tokens), 32))
        
        class block:
            @staticmethod
            def forward(x):
                return np.zeros((len(x), 32))
    
    with patch('similarity.get_high_quality_logs', return_value=[]):
        with patch('similarity.count_logs', return_value=0):
            result, top_similar = predict_response_length(
                "test query", MockTokenizer(), MockModel()
            )
            assert result is None
            assert top_similar == []
    
    print("predict_response_length no-data test passed!")


def test_direct_response_on_high_similarity():
    from similarity import predict_response_length, SIMILARITY_THRESHOLD
    
    class MockTokenizer:
        def encode(self, word):
            return hash(word) % 1000
    
    class MockModel:
        class embedding:
            @staticmethod
            def get_embeddings(tokens):
                return np.ones((len(tokens), 32)) * 0.95
        
        class block:
            @staticmethod
            def forward(x):
                return np.ones((len(x), 32)) * 0.95
    
    mock_logs = [
        ("similar q", "This is the exact answer we want", 5),
    ]
    
    with patch('similarity.get_high_quality_logs', return_value=mock_logs):
        with patch('similarity.count_logs', return_value=1):
            result, _ = predict_response_length(
                "similar q", MockTokenizer(), MockModel()
            )
            assert result is not None
            assert "direct_response" in result
            assert result["direct_response"] == "This is the exact answer we want"
            assert result["source"] == "db_direct"
    
    print("direct_response high similarity test passed!")


def test_decompose_question():
    from similarity import decompose_question
    
    with patch('similarity.cloud_agent', return_value="""## Sub-questions
1. What is machine learning?
2. What are neural networks?
3. How do they learn?"""):
        result = decompose_question("Explain ML and neural networks")
        assert result is not None
        assert len(result) == 3
    
    print("decompose_question tests passed!")


def test_answer_sub_questions_recursive():
    from similarity import _answer_sub_questions
    
    class MockTokenizer:
        def encode(self, word):
            return hash(word) % 1000
    
    class MockModel:
        class embedding:
            @staticmethod
            def get_embeddings(tokens):
                return np.zeros((len(tokens), 32))
        
        class block:
            @staticmethod
            def forward(x):
                return np.zeros((len(x), 32))
    
    sub_questions = ["What is AI?", "What is ML?"]
    
    def mock_generate(q, t, m, depth=0):
        return f"Answer to: {q}", "cloud_direct"
    
    with patch('similarity.generate_response', side_effect=mock_generate):
        answers = _answer_sub_questions(sub_questions, MockTokenizer(), MockModel(), depth=0)
        assert len(answers) == 2
        assert answers[0]["question"] == "What is AI?"
        assert answers[0]["source"] == "cloud_direct"
    
    print("_answer_sub_questions recursive test passed!")


def test_generate_response_recursive():
    from similarity import generate_response, MAX_DECOMPOSE_DEPTH
    
    class MockTokenizer:
        def encode(self, word):
            return hash(word) % 1000
    
    class MockModel:
        class embedding:
            @staticmethod
            def get_embeddings(tokens):
                return np.zeros((len(tokens), 32))
        
        class block:
            @staticmethod
            def forward(x):
                return np.zeros((len(x), 32))
    
    response, source = generate_response("test", MockTokenizer(), MockModel(), depth=MAX_DECOMPOSE_DEPTH)
    assert source == "max_depth_reached"
    
    print("generate_response recursive depth test passed!")


def test_generate_response_decomposed_recursive():
    from similarity import generate_response
    
    class MockTokenizer:
        def encode(self, word):
            return hash(word) % 1000
    
    class MockModel:
        class embedding:
            @staticmethod
            def get_embeddings(tokens):
                return np.zeros((len(tokens), 32))
        
        class block:
            @staticmethod
            def forward(x):
                return np.zeros((len(x), 32))
    
    mock_logs = [("q", "a" * 300, 4) for _ in range(10)]
    
    def mock_generate(q, t, m, depth=0):
        return f"Answer: {q}", "cloud_direct"
    
    def mock_cloud_agent(query):
        if "Decompose" in query:
            return "## Sub-questions\n1. Q1?\n2. Q2?"
        elif "Combine" in query:
            return "Combined answer."
        return "Sub-answer."
    
    with patch('similarity.get_high_quality_logs', return_value=mock_logs):
        with patch('similarity.count_logs', return_value=10):
            with patch('similarity.generate_response', side_effect=mock_generate):
                with patch('similarity.cloud_agent', side_effect=mock_cloud_agent):
                    response, source = generate_response(
                        "long question", MockTokenizer(), MockModel()
                    )
                    assert source == "decomposed"
                    assert response == "Combined answer."
    
    print("generate_response decomposed recursive test passed!")


def test_max_depth():
    from similarity import MAX_DECOMPOSE_DEPTH, MAX_RESPONSE_LENGTH, MAX_DB_RECORDS
    
    assert MAX_DECOMPOSE_DEPTH == 10
    assert MAX_RESPONSE_LENGTH == 200
    assert MAX_DB_RECORDS == 5000
    print("config tests passed!")


def test_insufficient_data():
    from similarity import generate_response
    
    class MockTokenizer:
        def encode(self, word):
            return hash(word) % 1000
    
    class MockModel:
        class embedding:
            @staticmethod
            def get_embeddings(tokens):
                return np.zeros((len(tokens), 32))
        
        class block:
            @staticmethod
            def forward(x):
                return np.zeros((len(x), 32))
    
    with patch('similarity.count_logs', return_value=50):
        with patch('similarity.cloud_agent', return_value="Direct AI response"):
            response, source = generate_response(
                "test query", MockTokenizer(), MockModel()
            )
            assert response == "Direct AI response"
            assert source == "cloud_direct"
    
    print("insufficient data test passed!")


def test_db_limit():
    from similarity import _get_logs_with_limit
    
    many_logs = [(f"q{i}", f"response{i}", 4) for i in range(6000)]
    
    with patch('similarity.get_high_quality_logs', return_value=many_logs):
        with patch('similarity.count_logs', return_value=6000):
            result = _get_logs_with_limit(min_rating=3)
            assert len(result) == 5000
    
    print("DB limit test passed!")


if __name__ == "__main__":
    test_cosine_similarity()
    test_parse_sub_questions()
    test_predict_response_length_no_data()
    test_direct_response_on_high_similarity()
    test_decompose_question()
    test_answer_sub_questions_recursive()
    test_generate_response_recursive()
    test_generate_response_decomposed_recursive()
    test_max_depth()
    test_db_limit()
    test_insufficient_data()
    print("\nAll similarity tests passed!")

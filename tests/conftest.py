import pytest
import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import tensorflow as tf

# Path racine du projet
BASE_DIR = Path(__file__).parent.parent


@pytest.fixture
def temp_db_path():
    """Base de données SQLite temporaire pour les tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def in_memory_db():
    """Base de données SQLite en mémoire."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def mock_model():
    """Modèle TensorFlow mock pour les tests."""
    model = Mock()
    model.predict = Mock(return_value=tf.constant([[0.75]]))
    return model


@pytest.fixture
def mock_tokenizer():
    """Tokenizer mock pour les tests."""
    tokenizer = Mock()
    tokenizer.texts_to_sequences = Mock(return_value=[[1, 2, 3]])
    return tokenizer


@pytest.fixture
def mock_env_vars():
    """Variables d'environnement mockées."""
    with patch.dict(os.environ, {
        "BREVO_API_KEY": "test_api_key_12345",
        "EMAIL_RECIPIENT": "test@example.com"
    }):
        yield


@pytest.fixture
def sample_prediction_data():
    """Données de prédiction exemples."""
    return {
        "text": "I love this product!",
        "predicted_sentiment": "positif",
        "confidence_score": 0.85
    }


@pytest.fixture
def sample_predictions_list():
    """Liste de prédictions pour tests email."""
    return [
        {
            "text": "Bad prediction 1",
            "predicted_sentiment": "négatif",
            "confidence_score": 0.65,
            "timestamp": "2025-01-15 10:00:00"
        },
        {
            "text": "Bad prediction 2",
            "predicted_sentiment": "négatif",
            "confidence_score": 0.55,
            "timestamp": "2025-01-15 10:05:00"
        },
        {
            "text": "Bad prediction 3",
            "predicted_sentiment": "positif",
            "confidence_score": 0.52,
            "timestamp": "2025-01-15 10:10:00"
        }
    ]


@pytest.fixture
def sample_texts():
    """Textes exemples pour tests de nettoyage."""
    return {
        "with_url": "Check this https://example.com amazing product!",
        "with_mention": "@user hello there",
        "with_hashtag": "Love this #awesome day",
        "with_emoji": "I love this! 😊 ❤️",
        "with_contraction": "I'm happy it's great",
        "empty": "",
        "only_stopwords": "the a an on in at",
        "complex": "Check @user's https://site.com #offer! I'm amazed 😊"
    }

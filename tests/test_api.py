import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPIEndpoints:
    """Tests des endpoints API."""

    @pytest.fixture
    def client(self, mock_model, mock_tokenizer):
        """Client de test FastAPI."""
        # Importer l'API et patcher les globals directement
        import api

        # Sauvegarder les valeurs originales
        original_model = api.model
        original_tokenizer = api.tokenizer

        # Remplacer par les mocks
        api.model = mock_model
        api.tokenizer = mock_tokenizer

        # Créer le client
        test_client = TestClient(api.app)

        yield test_client

        # Restaurer les valeurs originales
        api.model = original_model
        api.tokenizer = original_tokenizer

    def test_root_endpoint(self, client):
        """Test l'endpoint racine."""
        response = client.get("/")

        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_check_model_loaded(self, client):
        """Test le health check avec modèle chargé."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_predict_valid_text(self, client):
        """Test la prédiction avec un texte valide."""
        response = client.post(
            "/predict",
            json={"text": "I love this product!"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "sentiment" in data
        assert "confidence" in data
        assert "score" in data
        assert data["sentiment"] in ["positif", "négatif"]
        assert 0 <= data["score"] <= 1

    def test_predict_empty_text(self, client):
        """Test la prédiction avec texte vide."""
        response = client.post(
            "/predict",
            json={"text": ""}
        )

        assert response.status_code == 400

    def test_predict_whitespace_only(self, client):
        """Test la prédiction avec seulement des espaces."""
        response = client.post(
            "/predict",
            json={"text": "   "}
        )

        assert response.status_code == 400

    def test_predict_model_not_loaded(self, mock_model, mock_tokenizer):
        """Test quand le modèle n'est pas chargé."""
        import api

        # Sauvegarder les valeurs originales
        original_model = api.model
        original_tokenizer = api.tokenizer

        # Simuler modèle non chargé
        api.model = None
        api.tokenizer = None

        test_client = TestClient(api.app)
        response = test_client.post("/predict", json={"text": "test"})

        # Restaurer les valeurs originales
        api.model = original_model
        api.tokenizer = original_tokenizer

        assert response.status_code == 503
        assert "modèle" in response.json()["detail"].lower()

    def test_predict_no_valid_tokens_after_cleaning(self, client):
        """Test quand le nettoyage ne laisse aucun mot valide."""
        # Texte qui sera complètement nettoyé (mentions, hashtags, emojis)
        response = client.post(
            "/predict",
            json={"text": "@user #tag !!!"}
        )

        assert response.status_code == 400
        assert "mot valide" in response.json()["detail"]


class TestReportBadPredictionEndpoint:
    """Tests de l'endpoint de signalement."""

    @pytest.fixture
    def client_with_db(self, temp_db_path, mock_model, mock_tokenizer, monkeypatch):
        """Client avec base de données temporaire."""
        monkeypatch.setattr("database.DB_PATH", temp_db_path)

        with patch('api.model', mock_model), \
             patch('api.tokenizer', mock_tokenizer):
            from database import init_database
            init_database()
            from api import app
            return TestClient(app)

    def test_report_bad_prediction_valid(self, client_with_db, sample_prediction_data):
        """Test le signalement avec données valides."""
        response = client_with_db.post(
            "/report-bad-prediction",
            json=sample_prediction_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["report_count"] == 1
        assert data["email_sent"] is False  # Pas d'email au premier

    def test_report_bad_prediction_invalid_sentiment(self, client_with_db):
        """Test avec un sentiment invalide."""
        response = client_with_db.post(
            "/report-bad-prediction",
            json={
                "text": "Test",
                "predicted_sentiment": "invalid",
                "confidence_score": 0.5
            }
        )

        assert response.status_code == 400

    def test_report_bad_prediction_invalid_confidence(self, client_with_db):
        """Test avec un score de confiance invalide."""
        response = client_with_db.post(
            "/report-bad-prediction",
            json={
                "text": "Test",
                "predicted_sentiment": "positif",
                "confidence_score": 1.5  # > 1.0
            }
        )

        assert response.status_code == 400

    @patch('api.send_bad_predictions_email')
    def test_email_sent_on_third_report(self, mock_send_email, client_with_db, sample_prediction_data):
        """Test qu'un email est envoyé tous les 3 rapports."""
        mock_send_email.return_value = True

        # 3 rapports
        for i in range(3):
            response = client_with_db.post(
                "/report-bad-prediction",
                json=sample_prediction_data
            )
            assert response.status_code == 200

        # Le 3ème doit déclencher un email
        mock_send_email.assert_called_once()

    def test_report_bad_prediction_email_fails(self, temp_db_path, mock_model, mock_tokenizer, monkeypatch):
        """Test quand l'envoi d'email échoue."""
        import api
        from database import init_database

        # Configurer la base de données temporaire
        monkeypatch.setattr("database.DB_PATH", temp_db_path)

        # Sauvegarder les valeurs originales
        original_model = api.model
        original_tokenizer = api.tokenizer

        # Remplacer par les mocks
        api.model = mock_model
        api.tokenizer = mock_tokenizer

        # Initialiser la base de données
        init_database()

        # Mock pour que l'envoi d'email échoue
        with patch('api.send_bad_predictions_email', return_value=False):
            test_client = TestClient(api.app)

            # 3 rapports pour déclencher l'email
            for i in range(3):
                response = test_client.post("/report-bad-prediction", json={
                    "text": f"Test {i}",
                    "predicted_sentiment": "positif",
                    "confidence_score": 0.8
                })
                assert response.status_code == 200

            # Le dernier devrait avoir email_sent=False
            data = response.json()
            assert data["email_sent"] is False
            assert "échoué" in data["message"].lower()

        # Restaurer les valeurs originales
        api.model = original_model
        api.tokenizer = original_tokenizer

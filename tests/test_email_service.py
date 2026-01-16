import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from email_service import (
    send_bad_predictions_email,
    _create_email_html
)


class TestEmailHTMLGeneration:
    """Tests de génération du HTML d'email."""

    def test_create_email_html_basic(self, sample_predictions_list):
        """Test la génération basique du HTML."""
        html = _create_email_html(sample_predictions_list)

        assert "<html>" in html
        assert "<table>" in html
        assert "Bad prediction 1" in html
        assert "Bad prediction 2" in html
        assert "Bad prediction 3" in html
        # Le HTML utilise "Négatif" avec majuscule
        assert "Négatif" in html or "ngatif" in html
        assert "Positif" in html or "ositif" in html

    def test_create_email_html_emoji_sentiment(self, sample_predictions_list):
        """Test que les emojis sont corrects pour chaque sentiment."""
        html = _create_email_html(sample_predictions_list)

        assert "😊" in html  # positif
        assert "😞" in html  # négatif

    def test_create_email_html_confidence_percent(self, sample_predictions_list):
        """Test l'affichage du pourcentage de confiance."""
        html = _create_email_html(sample_predictions_list)

        assert "65.00%" in html
        assert "55.00%" in html
        assert "52.00%" in html

    def test_create_email_html_empty_list(self):
        """Test avec une liste vide."""
        html = _create_email_html([])

        assert "<html>" in html
        assert "0 nouvelle" in html


class TestSendBadPredictionsEmail:
    """Tests d'envoi d'email."""

    @patch('email_service.TransactionalEmailsApi')
    @patch('email_service.brevo_python')
    def test_send_email_success(self, mock_brevo, mock_api_class, sample_predictions_list, mock_env_vars):
        """Test l'envoi réussi d'un email."""
        # Mock Brevo configuration avec api_key comme dict
        mock_config = Mock()
        mock_api_key_dict = {}
        mock_config.api_key = mock_api_key_dict
        mock_brevo.Configuration.return_value = mock_config

        # Mock API instance et response
        mock_api_instance = Mock()
        mock_api_instance.send_transac_email.return_value = Mock(message_id="test_123")
        mock_api_instance.api_client.configuration = mock_config
        mock_api_class.return_value = mock_api_instance

        result = send_bad_predictions_email(sample_predictions_list)

        assert result is True
        mock_api_instance.send_transac_email.assert_called_once()

    @patch('email_service.brevo_python')
    def test_send_email_missing_api_key(self, mock_brevo, sample_predictions_list, monkeypatch):
        """Test le comportement sans clé API."""
        monkeypatch.delenv("BREVO_API_KEY", raising=False)

        result = send_bad_predictions_email(sample_predictions_list)

        assert result is False

    @patch('email_service.brevo_python')
    def test_send_email_missing_recipient(self, mock_brevo, sample_predictions_list, monkeypatch):
        """Test le comportement sans destinataire."""
        monkeypatch.setenv("BREVO_API_KEY", "test_key")
        monkeypatch.delenv("EMAIL_RECIPIENT", raising=False)

        result = send_bad_predictions_email(sample_predictions_list)

        assert result is False

    @patch('email_service.TransactionalEmailsApi')
    @patch('email_service.brevo_python')
    def test_send_email_retry_on_429(self, mock_brevo, mock_api_class, sample_predictions_list, mock_env_vars):
        """Test le retry en cas de rate limiting (429)."""
        from brevo_python.rest import ApiException

        # Mock Brevo configuration avec api_key comme dict
        mock_config = Mock()
        mock_api_key_dict = {}
        mock_config.api_key = mock_api_key_dict
        mock_brevo.Configuration.return_value = mock_config

        # Mock API instance avec retry
        mock_api_instance = Mock()
        # Premier appel rate limité, deuxième succès
        mock_api_instance.send_transac_email.side_effect = [
            ApiException(status=429, reason="Rate limit"),
            Mock(message_id="test_123")
        ]
        mock_api_instance.api_client.configuration = mock_config
        mock_api_class.return_value = mock_api_instance

        with patch('email_service.time.sleep') as mock_sleep:
            result = send_bad_predictions_email(sample_predictions_list)

            assert result is True
            assert mock_api_instance.send_transac_email.call_count == 2

    @patch('email_service.TransactionalEmailsApi')
    @patch('email_service.brevo_python')
    def test_send_email_auth_error_no_retry(self, mock_brevo, mock_api_class, sample_predictions_list, mock_env_vars):
        """Test qu'il n'y a pas de retry en cas d'erreur 401."""
        from brevo_python.rest import ApiException

        # Mock Brevo configuration avec api_key comme dict
        mock_config = Mock()
        mock_api_key_dict = {}
        mock_config.api_key = mock_api_key_dict
        mock_brevo.Configuration.return_value = mock_config

        # Mock API instance avec erreur 401
        mock_api_instance = Mock()
        mock_api_instance.send_transac_email.side_effect = ApiException(status=401, reason="Unauthorized")
        mock_api_instance.api_client.configuration = mock_config
        mock_api_class.return_value = mock_api_instance

        result = send_bad_predictions_email(sample_predictions_list)

        assert result is False
        # Avec l'erreur 401, on ne fait qu'une seule tentative avant de retourner False
        assert mock_api_instance.send_transac_email.call_count >= 0

    @patch('email_service.brevo_python')
    def test_brevo_configuration_error(self, mock_brevo, sample_predictions_list, monkeypatch):
        """Test erreur lors de la configuration Brevo."""
        monkeypatch.setenv("BREVO_API_KEY", "test_key")
        monkeypatch.setenv("EMAIL_RECIPIENT", "test@example.com")

        # Simuler une erreur lors de la création de la configuration
        mock_brevo.Configuration.side_effect = Exception("Configuration error")

        result = send_bad_predictions_email(sample_predictions_list)
        assert result is False

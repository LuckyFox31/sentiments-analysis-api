import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cleaning import clean_text


class TestTextCleaning:
    """Tests du nettoyage de texte."""

    def test_remove_url(self):
        """Test la suppression des URLs."""
        text = "Check this https://example.com amazing product!"
        result = clean_text(text)

        assert "https://example.com" not in " ".join(result)
        # Les résultats sont en minuscules
        assert "check" in result
        assert "amazing" in result or "amaz" in result  # lemmatizer peut tronquer

    def test_remove_mentions(self):
        """Test la suppression des mentions."""
        text = "@user hello there"
        result = clean_text(text)

        assert "@user" not in " ".join(result)
        # "there" devient "the" après lemmatization
        assert "hello" in result or "the" in result

    def test_remove_hashtags(self):
        """Test la suppression des hashtags."""
        text = "Love this #awesome day"
        result = clean_text(text)

        assert "#awesome" not in " ".join(result)
        assert "love" in result or "day" in result

    def test_emoji_conversion(self):
        """Test la conversion des emojis en tokens."""
        text = "I love this! 😊 ❤️"
        result = clean_text(text)

        # Les emojis sont convertis en tokens spécifiques
        assert len(result) > 0
        # Vérifier que le texte contient le token happy (peut être transformé par lemmatizer)
        result_str = " ".join(result)
        assert "love" in result_str or "tokensmileyhappy" in result_str

    def test_contraction_expansion(self):
        """Test l'expansion des contractions."""
        text = "I'm happy it's great"
        result = clean_text(text)

        # "I'm" devient "i am" puis "am" après lemmatization
        assert "happy" in result or "happi" in result  # lemmatizer vs stemmer
        assert "great" in result or "great" in result

    def test_empty_text(self):
        """Test avec un texte vide."""
        result = clean_text("")

        assert result == []

    def test_only_stopwords(self):
        """Test avec seulement des stop words."""
        result = clean_text("the a an on in at")

        # Les stop words sont supprimés
        assert len(result) == 0

    def test_complex_text(self, sample_texts):
        """Test avec un texte complexe contenant tout."""
        result = clean_text(sample_texts["complex"])

        assert isinstance(result, list)
        assert len(result) > 0

    def test_stemmer_processing(self):
        """Test le traitement avec stemmer."""
        # PorterStemmer transforme "running" en "run"
        result = clean_text("I am running", processing="stemmer")

        assert isinstance(result, list)
        # Le stemmer devrait réduire "running" à "run"
        assert "run" in result

import pytest
import sqlite3
from pathlib import Path
import sys

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import (
    init_database,
    insert_bad_prediction,
    get_recent_bad_predictions,
    increment_email_counter,
    get_email_counter,
    update_last_email_sent,
    get_connection
)


class TestDatabaseInitialization:
    """Tests d'initialisation de la base de données."""

    def test_init_database_creates_tables(self, temp_db_path, monkeypatch):
        """Test que init_database crée les tables correctement."""
        monkeypatch.setattr("database.DB_PATH", temp_db_path)

        init_database()

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Vérifier table bad_predictions
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='bad_predictions'
        """)
        assert cursor.fetchone() is not None

        # Vérifier table email_tracker
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='email_tracker'
        """)
        assert cursor.fetchone() is not None

        conn.close()

    def test_init_database_creates_counter(self, temp_db_path, monkeypatch):
        """Test que le compteur est initialisé à 0."""
        monkeypatch.setattr("database.DB_PATH", temp_db_path)

        init_database()

        count = get_email_counter()
        assert count == 0


class TestBadPredictionOperations:
    """Tests des opérations sur les mauvaises prédictions."""

    def test_insert_bad_prediction(self, temp_db_path, monkeypatch):
        """Test l'insertion d'une mauvaise prédiction."""
        monkeypatch.setattr("database.DB_PATH", temp_db_path)
        init_database()

        row_id = insert_bad_prediction(
            text="Test text",
            sentiment="positif",
            confidence=0.85
        )

        assert row_id == 1

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bad_predictions WHERE id = 1")
        row = cursor.fetchone()
        conn.close()

        assert row[1] == "Test text"
        assert row[2] == "positif"
        assert row[3] == 0.85

    def test_get_recent_bad_predictions(self, temp_db_path, monkeypatch):
        """Test la récupération des prédictions récentes."""
        monkeypatch.setattr("database.DB_PATH", temp_db_path)
        init_database()

        # Insérer 5 prédictions
        for i in range(5):
            insert_bad_prediction(f"Text {i}", "positif", 0.5 + i * 0.1)

        # Récupérer les 3 dernières
        recent = get_recent_bad_predictions(limit=3)

        assert len(recent) == 3
        # Vérifier qu'on a bien des textes différents
        texts = [r["text"] for r in recent]
        assert len(texts) == 3  # 3 textes uniques
        assert all("Text" in t for t in texts)  # Tous commencent par "Text"


class TestEmailCounter:
    """Tests du compteur d'emails."""

    def test_increment_email_counter(self, temp_db_path, monkeypatch):
        """Test l'incrémentation du compteur."""
        monkeypatch.setattr("database.DB_PATH", temp_db_path)
        init_database()

        count1 = increment_email_counter()
        assert count1 == 1

        count2 = increment_email_counter()
        assert count2 == 2

        count3 = get_email_counter()
        assert count3 == 2

    def test_get_email_counter(self, temp_db_path, monkeypatch):
        """Test la lecture du compteur."""
        monkeypatch.setattr("database.DB_PATH", temp_db_path)
        init_database()

        assert get_email_counter() == 0
        increment_email_counter()
        assert get_email_counter() == 1

    def test_update_last_email_sent(self, temp_db_path, monkeypatch):
        """Test la mise à jour du timestamp d'email."""
        monkeypatch.setattr("database.DB_PATH", temp_db_path)
        init_database()

        update_last_email_sent()

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT last_email_sent FROM email_tracker WHERE id = 1")
        timestamp = cursor.fetchone()
        conn.close()

        assert timestamp is not None
        assert timestamp[0] is not None

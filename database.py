import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = "bad_predictions.db"


def get_connection() -> sqlite3.Connection:
    """Créer une connexion à la base de données SQLite"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """
    Initialiser la base de données au démarrage de l'application.
    Crée les tables si elles n'existent pas.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Table pour stocker les mauvaises prédictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bad_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            predicted_sentiment TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index pour optimiser les requêtes de tri par timestamp
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp
        ON bad_predictions(timestamp DESC)
    """)

    # Table pour suivre le compteur d'emails envoyés
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_tracker (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            report_count INTEGER DEFAULT 0,
            last_email_sent DATETIME
        )
    """)

    # Initialiser le compteur s'il n'existe pas
    cursor.execute("SELECT COUNT(*) FROM email_tracker")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO email_tracker (id, report_count) VALUES (1, 0)")

    conn.commit()
    conn.close()
    print("✅ Base de données initialisée")


def insert_bad_prediction(text: str, sentiment: str, confidence: float) -> int:
    """
    Insérer une mauvaise prédiction dans la base de données.

    Args:
        text: Le texte analysé
        sentiment: Le sentiment prédit ('positif' ou 'négatif')
        confidence: Le score de confiance (0.0 à 1.0)

    Returns:
        L'ID de la ligne insérée
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO bad_predictions (text, predicted_sentiment, confidence_score)
        VALUES (?, ?, ?)
    """, (text, sentiment, confidence))

    row_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return row_id


def get_recent_bad_predictions(limit: int = 3) -> List[Dict]:
    """
    Récupérer les N dernières mauvaises prédictions.

    Args:
        limit: Nombre maximum de résultats à retourner

    Returns:
        Liste de dictionnaires contenant les prédictions
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT text, predicted_sentiment, confidence_score, timestamp
        FROM bad_predictions
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def increment_email_counter() -> int:
    """
    Incrémenter le compteur de rapports et retourner la nouvelle valeur.
    Cette opération est thread-safe grâce à la transaction SQLite.

    Returns:
        La nouvelle valeur du compteur
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE email_tracker
        SET report_count = report_count + 1
        WHERE id = 1
    """)

    cursor.execute("SELECT report_count FROM email_tracker WHERE id = 1")
    count = cursor.fetchone()[0]

    conn.commit()
    conn.close()

    return count


def get_email_counter() -> int:
    """
    Récupérer la valeur actuelle du compteur de rapports.

    Returns:
        La valeur actuelle du compteur
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT report_count FROM email_tracker WHERE id = 1")
    count = cursor.fetchone()[0]

    conn.close()

    return count


def update_last_email_sent() -> None:
    """Mettre à jour le timestamp du dernier email envoyé."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE email_tracker
        SET last_email_sent = CURRENT_TIMESTAMP
        WHERE id = 1
    """)

    conn.commit()
    conn.close()

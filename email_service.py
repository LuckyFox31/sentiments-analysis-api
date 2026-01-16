import os
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
import brevo_python
from brevo_python.rest import ApiException
from brevo_python.api import TransactionalEmailsApi
from brevo_python.models import SendSmtpEmail

# Charger les variables d'environnement
load_dotenv()


def send_bad_predictions_email(predictions: List[Dict]) -> bool:
    """
    Envoyer un email avec les mauvaises prédictions via l'API Brevo.

    Args:
        predictions: Liste de dictionnaires avec keys: text, sentiment, confidence, timestamp

    Returns:
        True si l'email a été envoyé avec succès, False sinon
    """

    api_key = os.getenv("BREVO_API_KEY")
    recipient = os.getenv("EMAIL_RECIPIENT")

    if not api_key:
        print("❌ BREVO_API_KEY non définie dans les variables d'environnement")
        return False

    if not recipient:
        print("❌ EMAIL_RECIPIENT non définie dans les variables d'environnement")
        return False

    # Configuration de l'API Brevo
    configuration = None
    try:
        configuration = brevo_python.Configuration()
        configuration.api_key['api-key'] = api_key
    except Exception as e:
        print(f"❌ Erreur lors de la configuration de Brevo: {e}")
        return False

    # Créer le contenu HTML de l'email
    html_content = _create_email_html(predictions)

    # Créer l'objet email
    email = SendSmtpEmail(
        to=[{"email": recipient}],
        subject="📊 Rapport de mauvaises prédictions - Analyse de Sentiment",
        html_content=html_content,
        sender={"email": "support@devbystep.fr", "name": "Analyse de Sentiment"}
    )

    # Tentative d'envoi avec retry
    return _send_with_retry(email, configuration, max_attempts=3)


def _create_email_html(predictions: List[Dict]) -> str:
    """
    Créer le contenu HTML de l'email.

    Args:
        predictions: Liste des prédictions à inclure

    Returns:
        Le contenu HTML formaté
    """
    rows_html = ""
    for pred in predictions:
        sentiment_emoji = "😊" if pred['predicted_sentiment'] == 'positif' else "😞"
        confidence_percent = pred['confidence_score'] * 100

        rows_html += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0;">{pred['text'][:100]}{'...' if len(pred['text']) > 100 else ''}</td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; text-align: center;">
                {sentiment_emoji} {pred['predicted_sentiment'].capitalize()}
            </td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; text-align: center;">
                {confidence_percent:.2f}%
            </td>
            <td style="padding: 12px; border-bottom: 1px solid #e0e0e0; text-align: center;">
                {pred['timestamp']}
            </td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 12px; border-bottom: 1px solid #e0e0e0; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e0e0e0; color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Rapport de mauvaises prédictions</h1>
            <p>{len(predictions)} nouvelle(s) mauvaise(s) prédiction(s) ont été signalée(s).</p>

            <table>
                <thead>
                    <tr>
                        <th>Texte analysé</th>
                        <th style="text-align: center;">Sentiment prédit</th>
                        <th style="text-align: center;">Confiance</th>
                        <th style="text-align: center;">Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>

            <div class="footer">
                <p>Ce email a été généré automatiquement par l'application d'Analyse de Sentiment.</p>
                <p>Utilisez ces données pour améliorer le modèle de Machine Learning.</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html


def _send_with_retry(email: SendSmtpEmail, configuration, max_attempts: int = 3) -> bool:
    """
    Envoyer l'email avec mécanisme de retry.

    Args:
        email: L'objet email à envoyer
        configuration: La configuration Brevo
        max_attempts: Nombre maximum de tentatives

    Returns:
        True si succès, False sinon
    """
    api_instance = TransactionalEmailsApi()

    for attempt in range(1, max_attempts + 1):
        try:
            api_instance.api_client.configuration = configuration
            result = api_instance.send_transac_email(email)

            print(f"✅ Email envoyé avec succès! Message ID: {result.message_id}")
            return True

        except ApiException as e:
            if e.status == 401:
                print(f"❌ Erreur d'authentification Brevo (401): Vérifiez votre API Key")
                return False
            elif e.status == 429:
                print(f"⚠️ Trop de requêtes Brevo (429): Rate limit dépassé")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 secondes
                    print(f"   Nouvelle tentative dans {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return False
            else:
                print(f"❌ Erreur Brevo (status {e.status}): {e}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"   Nouvelle tentative dans {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return False

        except Exception as e:
            print(f"❌ Erreur lors de l'envoi de l'email: {e}")
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"   Nouvelle tentative dans {wait_time}s...")
                time.sleep(wait_time)
                continue
            return False

    print(f"❌ Échec après {max_attempts} tentatives")
    return False

import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from cleaning import clean_text
import nltk
from dotenv import load_dotenv
from database import init_database, insert_bad_prediction, get_recent_bad_predictions, increment_email_counter, update_last_email_sent
from email_service import send_bad_predictions_email

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "ml-vectorizer.pkl"

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

app = FastAPI(
    title="API d'Analyse de Sentiment",
    description="API pour analyser le sentiment de tweets avec un modèle Naive Bayes",
    version="1.0.0"
)

model = None
vectorizer = None

@app.on_event("startup")
async def load_model_and_vectorizer():
    """Charger le modèle, le vectorizer et initialiser la base de données"""
    global model, vectorizer

    # Charger les variables d'environnement
    load_dotenv()

    # Initialiser la base de données
    init_database()

    try:
        print("Chargement du modèle Naive Bayes...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")

        print("Chargement du vectorizer...")
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✅ Vectorizer chargé depuis {VECTORIZER_PATH}")

    except Exception as e:
        print(f"❌ Erreur lors du chargement : {str(e)}")
        raise

class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this amazing product! :)"
            }
        }

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    score: float

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this amazing product! :)",
                "sentiment": "positif",
                "confidence": 0.85,
                "score": 0.85
            }
        }

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BadPredictionRequest(BaseModel):
    text: str
    predicted_sentiment: str  # "positif" ou "négatif"
    confidence_score: float   # 0.0 à 1.0

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I love this product!",
                "predicted_sentiment": "négatif",
                "confidence_score": 0.65
            }
        }


class BadPredictionResponse(BaseModel):
    success: bool
    message: str
    report_count: int
    email_sent: Optional[bool] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Signalement enregistré avec succès",
                "report_count": 3,
                "email_sent": True
            }
        }

@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine pour vérifier que l'API fonctionne"""
    return {
        "message": "API d'Analyse de Sentiment - Utilisez /docs pour voir la documentation",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Vérifier l'état de santé de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and vectorizer is not None
    }

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictRequest):
    """
    Analyser le sentiment d'un texte

    - **text**: Le texte à analyser (tweet ou phrase courte)

    Retourne le sentiment (positif/négatif), le score brut et la confiance
    """

    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle ou le vectorizer n'est pas chargé. Veuillez réessayer plus tard."
        )

    if not request.text or request.text.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Le texte ne peut pas être vide"
        )

    try:
        tokens = clean_text(request.text, processing="lemmatizer")

        if not tokens or len(tokens) == 0:
            raise HTTPException(
                status_code=400,
                detail="Le texte ne contient aucun mot valide après nettoyage"
            )

        text_cleaned = " ".join(tokens)

        # Vectoriser le texte avec TF-IDF
        text_vectorized = vectorizer.transform([text_cleaned])

        # Prédiction avec le modèle Naive Bayes
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]

        # Le modèle retourne 0 (négatif) ou 1 (positif)
        if prediction == 1:
            sentiment = "positif"
            score = float(probabilities[1])
            confidence = score
        else:
            sentiment = "négatif"
            score = float(probabilities[1])
            confidence = 1 - score

        return PredictResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            score=round(score, 4)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )


@app.post("/report-bad-prediction", response_model=BadPredictionResponse, tags=["Feedback"])
async def report_bad_prediction(request: BadPredictionRequest):
    """
    Signaler une prédiction incorrecte.

    - **text**: Le texte analysé
    - **predicted_sentiment**: Sentiment prédit (positif/négatif)
    - **confidence_score**: Score de confiance (0-1)

    Un email est envoyé toutes les 3 mauvaises prédictions.
    """
    try:
        # Validation des valeurs
        if request.predicted_sentiment not in ["positif", "négatif"]:
            raise HTTPException(
                status_code=400,
                detail="Le sentiment doit être 'positif' ou 'négatif'"
            )

        if not 0.0 <= request.confidence_score <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="Le score de confiance doit être entre 0.0 et 1.0"
            )

        # Insérer dans la base de données
        row_id = insert_bad_prediction(
            text=request.text,
            sentiment=request.predicted_sentiment,
            confidence=request.confidence_score
        )

        # Incrémenter le compteur et vérifier si email nécessaire
        count = increment_email_counter()
        email_sent = False

        if count % 3 == 0:
            # Récupérer les 3 dernières prédictions
            recent = get_recent_bad_predictions(limit=3)

            # Envoyer l'email
            email_sent = send_bad_predictions_email(recent)

            if email_sent:
                update_last_email_sent()
                return BadPredictionResponse(
                    success=True,
                    message="Signalement enregistré. Email envoyé à l'administrateur",
                    report_count=count,
                    email_sent=True
                )
            else:
                return BadPredictionResponse(
                    success=True,
                    message="Signalement enregistré mais l'envoi de l'email a échoué",
                    report_count=count,
                    email_sent=False
                )
        else:
            return BadPredictionResponse(
                success=True,
                message=f"Signalement enregistré ({count}/3 avant envoi email)",
                report_count=count,
                email_sent=False
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du signalement : {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
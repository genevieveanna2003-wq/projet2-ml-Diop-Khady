import joblib
import pandas as pd

def predict(df: pd.DataFrame) -> list:
    """
    Charge le modèle entraîné et retourne une liste de prédictions (0 pour normal, 1 pour attaque).
    df : DataFrame avec les mêmes colonnes que flux_train.csv, SANS la colonne 'attaque'.
    """
    # Charger le modèle joblib depuis le dossier models
    # On gère le chemin relatif selon d'où la fonction est appelée
    try:
        model = joblib.load('models/model.joblib')
    except:
        model = joblib.load('../models/model.joblib')
    
    # Faire les prédictions
    predictions = model.predict(df)
    
    # Retourner sous forme de liste de 0 ou 1
    return predictions.tolist()
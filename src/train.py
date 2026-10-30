import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

def main():
    # 1. Charger les données d'entraînement
    data_path = '../data/flux_train.csv'
    if not os.path.exists(data_path):
        # Essayer un autre chemin relatif si on est déjà dans src/
        data_path = 'data/flux_train.csv'
    
    if not os.path.exists(data_path):
        print(f"Erreur : Le fichier flux_train.csv est introuvable dans data/.")
        return
    
    df = pd.read_csv(data_path)
    print("Données chargées avec succès. Dimensions :", df.shape)

    # Séparation features (X) et target (y)
    target_col = 'attaque'
    if target_col not in df.columns:
        print(f"Attention : La colonne '{target_col}' n'a pas été trouvée. Vérifie le nom exact dans ton CSV.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Identifier les colonnes numériques et catégorielles
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 3. Pipelines de prétraitement
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 4. Division train/test local pour évaluer les modèles
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Définition des modèles à tester
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    best_score = 0.0
    best_model_name = ""
    best_pipeline = None

    print("\n--- Évaluation des modèles (Score F1) ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        
        score = f1_score(y_val, y_pred, pos_label=1)
        print(f"{name} -> F1-score : {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nMeilleur modèle retenu : {best_model_name} avec un F1-score de {best_score:.4f}")

    # 6. Sauvegarde du meilleur modèle
    os.makedirs('../models', exist_ok=True)
    model_path = '../models/model.joblib'
    if not os.path.exists('../models'):
        os.makedirs('models', exist_ok=True)
        model_path = 'models/model.joblib'
        
    joblib.dump(best_pipeline, model_path)
    print(f"Modèle sauvegardé avec succès dans {model_path}")

if __name__ == '__main__':
    main()
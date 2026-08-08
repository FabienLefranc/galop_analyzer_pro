import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

def entrainer_modele(dataset_path="data/dataset/dataset_entrainement.parquet", model_output_path="modele_galop_v6_couplages.joblib"):
    print(f"⏳ Chargement du dataset d'entraînement depuis '{dataset_path}'...")
    if not os.path.exists(dataset_path):
        print(f"❌ Erreur : Le fichier dataset '{dataset_path}' est introuvable. Avez-vous exécuté 'generer_dataset_ia.py' ?")
        return

    df = pd.read_parquet(dataset_path)
    print(f"📊 Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes.")

    # Définition de la variable cible (Target)
    target_col = 'Target_Victoire' if 'Target_Victoire' in df.columns else 'Target_Podium'
    if target_col not in df.columns:
        print(f"❌ Erreur : Aucune colonne cible (Target_Victoire ou Target_Podium) trouvée dans le dataset.")
        return

    print(f"🎯 Variable cible retenue pour l'apprentissage : '{target_col}'")

    # Sélection automatique des variables explicatives (features numériques pertinentes)
    # Ajout de 'Supplement' et 'Total_Supplement' ici 👇
    features_candidates = [
        'Poids_num', 'Corde_num', 
        'Total_courses', 'Total_Supplement', 'Gains_Total', 
        'Courses_Gazon', 'Courses_PSF', 
        'Total_montes', 'Montes_Gazon', 'Montes_PSF', 
        'Freq_Cheval_Jockey',
        'Supplement'
    ]

    # Filtrer uniquement les colonnes présentes dans le dataset
    features = [f for f in features_candidates if f in df.columns]
    
    if not features:
        print("❌ Erreur : Aucune feature valide n'a été trouvée pour l'entraînement.")
        return

    print(f"⚙️ Features utilisées pour l'entraînement ({len(features)}) : {features}")

    # Préparation de X et y
    X = df[features].fillna(0).astype(np.float32)
    y = df[target_col].fillna(0).astype(int)

    # Vérification qu'il y a assez de diversité dans la cible
    if y.nunique() < 2:
        print("⚠️ Attention : La cible ne contient qu'une seule classe. L'entraînement risque d'être inopérant.")

    # Séparation train / test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

    print("🚀 Entraînement du modèle XGBoost en cours...")
    
    # Configuration du modèle XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    # Entraînement
    model.fit(X_train, y_train)

    # Évaluation rapide
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    print(f"📈 Précision sur le set de test : {acc:.4f}")
    if y.nunique() > 1:
        auc = roc_auc_score(y_test, probs)
        print(f"📊 Score ROC AUC : {auc:.4f}")

    # Sauvegarde du modèle entraîné
    print(f"💾 Sauvegarde du modèle dans '{model_output_path}'...")
    joblib.dump(model, model_output_path)
    
    print("✅ Entraînement terminé avec succès ! Le modèle est prêt pour les prédictions.")

if __name__ == "__main__":
    entrainer_modele()
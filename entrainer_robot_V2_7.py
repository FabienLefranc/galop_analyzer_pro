# -*- coding: utf-8 -*-
"""
ENTRAINEMENT DU MODELE IA - GALOP ANALYZER PRO V2.7
Améliorations :
- Features d'interaction (Distance×Surface, Corde×Distance, etc.)
- Suppression de Num_PMU
- Optimisation pour le Top 3 par course
"""
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix
import joblib

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(r"C:\Users\33662\OneDrive\Bureau\galop_analyzer_pro")
DATASET_DIR = BASE_DIR / "data" / "dataset"
MODEL_DIR = BASE_DIR / "models"

DATASET_PARQUET = DATASET_DIR / "dataset_entrainement_v2_6.parquet"
TRAIN_RATIO = 0.80

# ============================================================
# CREATION DES FEATURES D'INTERACTION
# ============================================================
def ajouter_features_interaction(df):
    """
    Ajoute des features d'interaction pertinentes pour les courses.
    """
    print("🔧 Création des features d'interaction...")
    
    # 1. Corde × Distance Courte (très important !)
    df["Corde_Distance_Courte"] = df["Place_Corde"] * df["Distance_Courte"]
    
    # 2. Historique × Surface
    df["Hist_Victoire_PSF"] = df["Hist_Taux_Victoire"] * df["Surface_PSF"]
    df["Hist_Victoire_Gazon"] = df["Hist_Taux_Victoire"] * df["Surface_Gazon"]
    df["Hist_Podium_PSF"] = df["Hist_Taux_Podium"] * df["Surface_PSF"]
    df["Hist_Podium_Gazon"] = df["Hist_Taux_Podium"] * df["Surface_Gazon"]
    
    # 3. Distance × Surface
    df["Distance_Courte_PSF"] = df["Distance_Courte"] * df["Surface_PSF"]
    df["Distance_Moyenne_Gazon"] = df["Distance_Moyenne"] * df["Surface_Gazon"]
    
    # 4. Gains récents × Forme
    df["Gains_Recent_Forme"] = df["Gains_Annee_En_Cours"] * df["Hist_Taux_Podium"]
    
    # 5. Age × Experience
    df["Age_Experience"] = df["Age"] * df["Hist_Nb_Courses"]
    
    # 6. Poids × Distance (handicap)
    df["Poids_Distance"] = df["Poids"] * df["Distance_km"]
    
    # 7. Place corde × Hippodrome (certains hippodromes avantagent la corde)
    df["Corde_Hippodrome"] = df["Place_Corde"] * df["Hippodrome_Code"]
    
    print(f"   ✅ {4} nouvelles features d'interaction créées")
    return df


# ============================================================
# CHARGEMENT DU DATASET
# ============================================================
def charger_dataset():
    if DATASET_PARQUET.exists():
        print(f"📥 Chargement depuis Parquet : {DATASET_PARQUET}")
        df = pd.read_parquet(DATASET_PARQUET)
    else:
        raise FileNotFoundError(f"❌ Dataset introuvable : {DATASET_PARQUET}")
    
    print(f"📊 Dataset chargé : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    return df


# ============================================================
# SELECTION DES FEATURES (V2.7)
# ============================================================
def selectionner_features(df):
    """
    Sélectionne les features V2.7 sans Num_PMU.
    """
    forbidden = {
        "Target_Victoire", "Target_Podium", "Classement", "Cote_Direct",
        "Nom", "Proprietaire", "Entraineur", "Driver_Jockey", "Musique",
        "Hippodrome", "Date", "Date_dt", "_date", "_dist", "_race",
        "_ordre_source", "_horse_key", "Num_PMU",  # SUPPRESSION DE NUM_PMU
        "Surface_Final", "Surface_Source", "Surface_Confiance",
        "Surface_Methode", "Distance_Classe", "Saison_Course",
    }
    
    features = [c for c in df.columns if c not in forbidden]
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    features = [c for c in features if c not in non_numeric]
    
    print(f"️ {len(features)} features sélectionnées (Num_PMU exclu)")
    return features


# ============================================================
# SPLIT TEMPOREL
# ============================================================
def split_temporel(df, features, target_col, train_ratio=0.80):
    if "Date_dt" not in df.columns:
        raise ValueError("❌ Colonne 'Date_dt' manquante")
    
    df_sorted = df.sort_values("Date_dt").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    
    df_train = df_sorted.iloc[:split_idx].copy()
    df_test = df_sorted.iloc[split_idx:].copy()
    
    print(f"\n📅 SPLIT TEMPOREL ({train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%)")
    print(f"   🟢 TRAIN : {len(df_train):,} lignes ({df_train['Date_dt'].min()} → {df_train['Date_dt'].max()})")
    print(f"   🔵 TEST  : {len(df_test):,} lignes ({df_test['Date_dt'].min()} → {df_test['Date_dt'].max()})")
    
    X_train = df_train[features].fillna(0).astype(np.float32)
    y_train = df_train[target_col].fillna(0).astype(int)
    X_test = df_test[features].fillna(0).astype(np.float32)
    y_test = df_test[target_col].fillna(0).astype(int)
    
    return X_train, X_test, y_train, y_test


# ============================================================
# ENTRAINEMENT
# ============================================================
def entrainer_modele(X_train, X_test, y_train, y_test, target_name, features, model_path):
    print(f"\n{'='*70}")
    print(f"🎯 ENTRAINEMENT - Cible : {target_name}")
    print(f"{'='*70}")
    
    pos_ratio = y_train.mean()
    print(f"   Taux de positifs : {pos_ratio:.2%}")
    
    if y_train.nunique() < 2:
        print("⚠️ Une seule classe, entraînement impossible")
        return None
    
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=400,  # Augmenté pour V2.7
        max_depth=7,        # Légèrement plus profond
        learning_rate=0.03, # Plus lent pour mieux apprendre
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    
    print(f"   🚀 Entraînement (scale_pos_weight={scale_pos_weight:.2f})...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else float("nan")
    
    print(f"\n    Métriques TEST :")
    print(f"      Accuracy : {acc:.4f}")
    print(f"      Log Loss : {ll:.4f}")
    print(f"      ROC AUC  : {auc:.4f}")
    
    cm = confusion_matrix(y_test, preds)
    print(f"\n   📊 Matrice de confusion :")
    print(f"              Prédit 0  Prédit 1")
    print(f"   Réel 0    {cm[0,0]:>8}  {cm[0,1]:>8}")
    print(f"   Réel 1    {cm[1,0]:>8}  {cm[1,1]:>8}")
    
    importance = pd.DataFrame({
        "feature": features,
        "gain": model.feature_importances_,
    }).sort_values("gain", ascending=False)
    
    print(f"\n   🏆 Top 15 features :")
    for i, row in importance.head(15).iterrows():
        print(f"      {row['feature']:<35} {row['gain']:.4f}")
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n   💾 Modèle sauvegardé : {model_path}")
    
    return {
        "model": model,
        "accuracy": acc,
        "log_loss": ll,
        "auc": auc,
        "importance": importance,
        "confusion_matrix": cm,
    }


# ============================================================
# SAUVEGARDE METADONNEES
# ============================================================
def sauvegarder_metadonnees(results, features, dataset_info):
    meta_path = MODEL_DIR / "metadata_entrainement_v2_7.json"
    metadata = {
        "version": "2.7",
        "date_entrainement": datetime.now().isoformat(),
        "dataset": dataset_info,
        "features": features,
        "nb_features": len(features),
        "resultats": {
            name: {
                "accuracy": r["accuracy"],
                "log_loss": r["log_loss"],
                "auc": r["auc"],
                "top_5_features": r["importance"].head(5)["feature"].tolist(),
            }
            for name, r in results.items()
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n Métadonnées sauvegardées : {meta_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🤖 ENTRAINEMENT MODELE IA - GALOP ANALYZER PRO V2.7")
    print("✨ Features d'interaction + Suppression Num_PMU")
    print("=" * 70)
    
    df = charger_dataset()
    df = ajouter_features_interaction(df)
    
    features = selectionner_features(df)
    
    dataset_info = {
        "nb_lignes": int(len(df)),
        "nb_colonnes": int(len(df.columns)),
        "date_min": str(df["Date_dt"].min()) if "Date_dt" in df.columns else None,
        "date_max": str(df["Date_dt"].max()) if "Date_dt" in df.columns else None,
    }
    
    results = {}
    
    for target_col in ["Target_Victoire", "Target_Podium"]:
        if target_col not in df.columns:
            print(f"\n⚠️ Cible '{target_col}' absente, ignorée.")
            continue
        
        X_train, X_test, y_train, y_test = split_temporel(
        df, features, target_col, TRAIN_RATIO
        )
        
        model_name = target_col.replace("Target_", "").lower()
        model_path = MODEL_DIR / f"modele_galop_v27_{model_name}.joblib"
        
        result = entrainer_modele(
            X_train, X_test, y_train, y_test,
            target_name=target_col,
            features=features,
            model_path=model_path,
        )
        if result:
            results[target_col] = result
    
    if results:
        sauvegarder_metadonnees(results, features, dataset_info)
    
    print("\n" + "=" * 70)
    print("✅ ENTRAINEMENT V2.7 TERMINE")
    print("=" * 70)
    for name, r in results.items():
        print(f"   🎯 {name:<20} AUC = {r['auc']:.4f} | Acc = {r['accuracy']:.4f}")
    print(f"\n   📁 Modèles : {MODEL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
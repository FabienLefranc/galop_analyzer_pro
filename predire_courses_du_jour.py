import os
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. FONCTIONS DE NETTOYAGE & COPIES (Identiques)
# ==========================================
def nettoyer_nom(nom):
    if not isinstance(nom, str):
        return "INCONNU"
    import unicodedata, re
    n = unicodedata.normalize('NFD', nom).encode('ascii', 'ignore').decode('utf-8')
    return re.sub(r'\s+', ' ', n).strip().upper()

def determiner_surface(nature_piste):
    if not isinstance(nature_piste, str):
        return "GAZON"
    np_clean = nature_piste.upper()
    if any(term in np_clean for term in ['PSF', 'SABLE', 'FIBRE', 'ALL WEATHER']):
        return "PSF"
    return "GAZON"

def safe_float(val, default=0.0):
    try:
        val_clean = str(val).replace(',', '.').strip()
        return float(val_clean)
    except (ValueError, TypeError):
        return default

# ==========================================
# 2. MOTEUR DE PRÉDICTION DES COURSES DU JOUR
# ==========================================
def predire_courses_du_jour(
    url_ou_chemin_du_jour, 
    masters_dir="data/masters", 
    model_path="modele_galop_v6_couplages.joblib"
):
    print("⏳ Chargement du modèle et des fichiers masters...")
    
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Le modèle '{model_path}' est introuvable. Avez-vous exécuté 'entrainer_robot.py' ?")
        return

    try:
        model = joblib.load(model_path)
        df_chevaux = pd.read_parquet(os.path.join(masters_dir, 'master_chevaux.parquet'))
        df_jockeys = pd.read_parquet(os.path.join(masters_dir, 'master_jockeys.parquet'))
        df_entraineurs = pd.read_parquet(os.path.join(masters_dir, 'master_entraineurs.parquet'))
        df_couplages = pd.read_parquet(os.path.join(masters_dir, 'master_couplages.parquet'))
    except Exception as e:
        print(f"❌ Erreur lors du chargement des dépendances : {e}")
        return

    print(f"⏳ Chargement des partants du jour depuis '{url_ou_chemin_du_jour}'...")
    try:
        df_jour = pd.read_csv(url_ou_chemin_du_jour)
        df_jour.columns = df_jour.columns.str.strip()
    except Exception as e:
        print(f"❌ Erreur lors du chargement des partants du jour : {e}")
        return

    if df_jour.empty:
        print("⚠️ Le fichier des partants du jour est vide.")
        return

    print(f"📊 {len(df_jour)} partants détectés. Application du pipeline de features...")

    # Normalisation stricte des clés de jointure (identique à l'entraînement)
    df_jour['Cheval_clean'] = df_jour['Nom'].apply(nettoyer_nom) if 'Nom' in df_jour.columns else "INCONNU"
    df_jour['Jockey_clean'] = df_jour['Driver_Jockey'].apply(nettoyer_nom) if 'Driver_Jockey' in df_jour.columns else "INCONNU"
    df_jour['Entraineur_clean'] = df_jour['Entraineur'].apply(nettoyer_nom) if 'Entraineur' in df_jour.columns else "INCONNU"
    
    df_jour['Surface'] = df_jour['Nature_Piste'].apply(determiner_surface) if 'Nature_Piste' in df_jour.columns else "GAZON"
    
    # Traitement des variables contextuelles directes
    df_jour['Poids_num'] = df_jour['Poids'].apply(lambda x: safe_float(x, 58.0)) if 'Poids' in df_jour.columns else 58.0
    df_jour['Corde_num'] = df_jour['Place_Corde'].apply(lambda x: safe_float(x, 1.0)) if 'Place_Corde' in df_jour.columns else 1.0

    # --- Fusions successives avec les Masters (exactement comme dans generer_dataset_ia.py) ---
    df_pred = df_jour.merge(df_chevaux, on='Cheval_clean', how='left')
    df_pred = df_pred.merge(df_jockeys, on='Jockey_clean', how='left')
    df_pred = df_pred.merge(df_entraineurs, on='Entraineur_clean', how='left')

    # Gestion des couplages (Cheval + Jockey)
    df_cj = df_couplages.rename(columns={'Entite_1': 'Cheval_clean', 'Entite_2': 'Jockey_clean', 'Frequence_Association': 'Freq_Cheval_Jockey'})
    df_pred = df_pred.merge(df_cj[['Cheval_clean', 'Jockey_clean', 'Freq_Cheval_Jockey']], on=['Cheval_clean', 'Jockey_clean'], how='left')

    # Nettoyage des valeurs manquantes pour les nouveaux chevaux / associations
    df_pred = df_pred.fillna({
        'Total_courses': 0,
        'Gains_Total': 0.0,
        'Courses_Gazon': 0,
        'Courses_PSF': 0,
        'Total_montes': 0,
        'Montes_Gazon': 0,
        'Montes_PSF': 0,
        'Freq_Cheval_Jockey': 0
    })

    # Sélection rigoureuse des mêmes features que lors de l'entraînement
    features = [
        'Poids_num', 'Corde_num', 
        'Total_courses', 'Gains_Total', 
        'Courses_Gazon', 'Courses_PSF', 
        'Total_montes', 'Montes_Gazon', 'Montes_PSF', 
        'Freq_Cheval_Jockey'
    ]

    # Vérification que toutes les features existent bien
    for f in features:
        if f not in df_pred.columns:
            df_pred[f] = 0.0

    X_pred = df_pred[features].fillna(0).astype(np.float32)

    print("🚀 Calcul des probabilités de réussite par l'IA...")
    # Prédiction des probabilités (colonne 1 = proba de succès)
    probabilities = model.predict_proba(X_pred)[:, 1]
    df_pred['Proba_IA'] = probabilities

    # ==========================================
    # 3. AFFICHAGE DES RÉSULTATS
    # ==========================================
    print("\n" + "="*50)
    print("🏆 PRONOSTICS ET PROBABILITÉS DU JOUR")
    print("="*50)

    # Tri par course (si la colonne existe) puis par probabilité décroissante
    colonnes_a_afficher = ['Nom', 'Driver_Jockey', 'Entraineur', 'Poids_num', 'Proba_IA']
    if 'Reunion' in df_pred.columns and 'Course' in df_pred.columns:
        df_sorted = df_pred.sort_values(by=['Reunion', 'Course', 'Proba_IA'], ascending=[True, True, False])
        group_cols = ['Reunion', 'Course']
    else:
        df_sorted = df_pred.sort_values(by='Proba_IA', ascending=False)
        group_cols = None

    for idx, row in df_sorted.iterrows():
        nom_cheval = row.get('Nom', 'Inconnu')
        jockey = row.get('Driver_Jockey', 'Inconnu')
        proba = row['Proba_IA'] * 100
        print(f"🐎 {nom_cheval:<20} | Jockey: {jockey:<18} | 🎯 Indice IA: {proba:.2f}%")

    print("="*50)
    
    # Sauvegarde optionnelle des résultats du jour
    output_csv = "resultats_predictions_du_jour.csv"
    df_sorted.to_csv(output_csv, index=False)
    print(f"💾 Les prédictions détaillées ont été sauvegardées dans '{output_csv}'")

if __name__ == "__main__":
    # Indiquez ici le chemin ou l'URL vers votre fichier CSV des partants du jour
    URL_PARTANTS_DU_JOUR = "chemin_vers_partants_du_jour.csv" 
    # predire_courses_du_jour(URL_PARTANTS_DU_JOUR)
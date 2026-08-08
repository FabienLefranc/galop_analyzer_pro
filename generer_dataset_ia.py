import os
import pandas as pd
import numpy as np

# ==========================================
# 1. FONCTIONS DE NETTOYAGE & COPIES (Mêmes règles)
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

def safe_int(val, default=0):
    try:
        return int(float(str(val).replace(',', '.').strip()))
    except (ValueError, TypeError):
        return default

# ==========================================
# 2. CONSTRUCTION DU DATASET D'ENTRAÎNEMENT
# ==========================================
def executer_generation_dataset(url_ou_chemin_historique, masters_dir="data/masters", output_dir="data/dataset"):
    """
    Combine l'historique brut des courses avec les fichiers masters (Parquet)
    pour produire le dataset d'entraînement structuré pour XGBoost.
    """
    print("⏳ Chargement des fichiers masters...")
    try:
        df_chevaux = pd.read_parquet(os.path.join(masters_dir, 'master_chevaux.parquet'))
        df_jockeys = pd.read_parquet(os.path.join(masters_dir, 'master_jockeys.parquet'))
        df_entraineurs = pd.read_parquet(os.path.join(masters_dir, 'master_entraineurs.parquet'))
        df_couplages = pd.read_parquet(os.path.join(masters_dir, 'master_couplages.parquet'))
    except Exception as e:
        print(f"❌ Erreur lors du chargement des masters : {e}. Avez-vous bien exécuté 'generer_profils.py' ?")
        return

    print("⏳ Chargement de l'historique source...")
    try:
        df_hist = pd.read_csv(url_ou_chemin_historique)
        df_hist.columns = df_hist.columns.str.strip()
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'historique : {e}")
        return

    print(f"📊 {len(df_hist)} lignes brutes trouvées. Croisement des données en cours...")

    # Normalisation des clés de jointure dans l'historique
    df_hist['Cheval_clean'] = df_hist['Nom'].apply(nettoyer_nom) if 'Nom' in df_hist.columns else "INCONNU"
    df_hist['Jockey_clean'] = df_hist['Driver_Jockey'].apply(nettoyer_nom) if 'Driver_Jockey' in df_hist.columns else "INCONNU"
    df_hist['Entraineur_clean'] = df_hist['Entraineur'].apply(nettoyer_nom) if 'Entraineur' in df_hist.columns else "INCONNU"
    
    # Gestion sécurisée de la colonne Supplement (0 ou 1) dans l'historique brut
    if 'Supplement' in df_hist.columns:
        df_hist['Supplement'] = df_hist['Supplement'].apply(safe_int)
    else:
        df_hist['Supplement'] = 0

    df_hist['Surface'] = df_hist['Nature_Piste'].apply(determiner_surface) if 'Nature_Piste' in df_hist.columns else "GAZON"
    
    # Traitement des variables contextuelles directes de la course
    df_hist['Poids_num'] = df_hist['Poids'].apply(lambda x: safe_float(x, 58.0)) if 'Poids' in df_hist.columns else 58.0
    df_hist['Corde_num'] = df_hist['Place_Corde'].apply(lambda x: safe_float(x, 1.0)) if 'Place_Corde' in df_hist.columns else 1.0

    # Gestion de la cible (ex: Victoire ou présence dans le Top 3 / Target binaire)
    if 'Classement' in df_hist.columns:
        df_hist['Target_Victoire'] = df_hist['Classement'].apply(lambda x: 1 if safe_float(x, 99) == 1 else 0)
        df_hist['Target_Podium'] = df_hist['Classement'].apply(lambda x: 1 if safe_float(x, 99) <= 3 else 0)
    else:
        df_hist['Target_Victoire'] = 0
        df_hist['Target_Podium'] = 0

    # --- Fusions successives avec les Masters ---
    print("🔗 Fusion avec le Master Chevaux...")
    df_dataset = df_hist.merge(df_chevaux, on='Cheval_clean', how='left')

    print("🔗 Fusion avec le Master Jockeys...")
    df_dataset = df_dataset.merge(df_jockeys, on='Jockey_clean', how='left')

    print("🔗 Fusion avec le Master Entraîneurs...")
    df_dataset = df_dataset.merge(df_entraineurs, on='Entraineur_clean', how='left')

    # Gestion des couplages (Cheval + Jockey)
    print("🔗 Intégration des scores de couplages...")
    df_cj = df_couplages.rename(columns={'Entite_1': 'Cheval_clean', 'Entite_2': 'Jockey_clean', 'Frequence_Association': 'Freq_Cheval_Jockey'})
    df_dataset = df_dataset.merge(df_cj[['Cheval_clean', 'Jockey_clean', 'Freq_Cheval_Jockey']], on=['Cheval_clean', 'Jockey_clean'], how='left')

    # Nettoyage final des valeurs manquantes (NaN issus des fusions à gauche)
    df_dataset = df_dataset.fillna({
        'Total_courses': 0,
        'Total_Supplement': 0,      # Sécurisation de la nouvelle colonne du master chevaux
        'Gains_Total': 0.0,
        'Courses_Gazon': 0,
        'Courses_PSF': 0,
        'Total_montes': 0,
        'Montes_Gazon': 0,
        'Montes_PSF': 0,
        'Freq_Cheval_Jockey': 0,
        'Supplement': 0             # Sécurisation si la colonne brute comporte des trous
    })

    # ==========================================
    # 3. SAUVEGARDE DU DATASET FINAL
    # ==========================================
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dataset_entrainement.parquet')
    
    print(f"💾 Sauvegarde du dataset d'entraînement dans '{output_path}'...")
    df_dataset.to_parquet(output_path, index=False)
    
    print(f"✅ Dataset généré avec succès ! Dimensions finales : {df_dataset.shape[0]} lignes x {df_dataset.shape[1]} colonnes.")

if __name__ == "__main__":
    URL_HISTORIQUE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=644246763&single=true&output=csv"
    executer_generation_dataset(URL_HISTORIQUE)
import os
import re
import unicodedata
import pandas as pd
import numpy as np

# ==========================================
# 1. FONCTIONS DE NETTOYAGE (Héritées de votre projet)
# ==========================================

def nettoyer_nom(nom):
    """Nettoie et normalise un nom de cheval, jockey ou entraîneur."""
    if not isinstance(nom, str):
        return "INCONNU"
    # Supprimer les accents
    n = unicodedata.normalize('NFD', nom).encode('ascii', 'ignore').decode('utf-8')
    # Majuscules et suppression des espaces superflus
    n = re.sub(r'\s+', ' ', n).strip().upper()
    return n

def determiner_surface(nature_piste):
    """Détermine si la piste est en Gazon ou PSF."""
    if not isinstance(nature_piste, str):
        return "GAZON"
    np_clean = nature_piste.upper()
    if any(term in np_clean for term in ['PSF', 'SABLE', 'FIBRE', 'ALL WEATHER']):
        return "PSF"
    return "GAZON"

def categoriser_distance(distance):
    """Catégorise la distance en grandes familles pour les stats."""
    try:
        d = float(distance)
        if d < 1300:
            return "Sprint (<1300m)"
        elif d <= 1600:
            return "Mile (1300-1600m)"
        elif d <= 2200:
            return "Intermédiaire (1700-2200m)"
        elif d <= 3000:
            return "Classique (2300-3000m)"
        else:
            return "Longue (>3000m)"
    except (ValueError, TypeError):
        return "Inconnue"

def safe_float(val, default=0.0):
    try:
        val_clean = str(val).replace(',', '.').strip()
        return float(val_clean)
    except (ValueError, TypeError):
        return default


# ==========================================
# 2. MOTEUR DE GÉNÉRATION DES MASTERS
# ==========================================

def executer_generation_masters(url_ou_chemin_csv, output_dir="data/masters"):
    """
    Parcourt l'historique complet en un seul passage et génère 
    tous les masters statistiques au format Parquet.
    """
    print("⏳ Chargement du fichier source historique...")
    try:
        df = pd.read_csv(url_ou_chemin_csv)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du fichier : {e}")
        return

    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()
    print(f"📊 {len(df)} lignes chargées. Nettoyage et normalisation en cours...")

    # Assurer un tri chronologique strict
    if 'Date' in df.columns:
        df['Date_dt'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df.sort_values(by='Date_dt', na_position='first').reset_index(drop=True)

    # Normalisation des colonnes clés
    df['Cheval_clean'] = df['Nom'].apply(nettoyer_nom) if 'Nom' in df.columns else "INCONNU"
    df['Jockey_clean'] = df['Driver_Jockey'].apply(nettoyer_nom) if 'Driver_Jockey' in df.columns else "INCONNU"
    df['Entraineur_clean'] = df['Entraineur'].apply(nettoyer_nom) if 'Entraineur' in df.columns else "INCONNU"
    df['Proprietaire_clean'] = df['Proprietaire'].apply(nettoyer_nom) if 'Proprietaire' in df.columns else "INCONNU"
    
    df['Surface'] = df['Nature_Piste'].apply(determiner_surface) if 'Nature_Piste' in df.columns else "GAZON"
    df['Cat_Distance'] = df['Distance'].apply(categoriser_distance) if 'Distance' in df.columns else "Inconnue"
    df['Terrain_clean'] = df['Etat_Terrain'].fillna("Inconnu").astype(str).str.upper()
    df['Hippodrome_clean'] = df['Hippodrome'].fillna("Inconnu").astype(str).str.upper()

    # Détection de la place (Victoire = 1, Podium = 1 ou 2 ou 3)
    if 'Place_Corde' in df.columns:
        # Si la place d'arrivée est disponible dans une colonne dédiée (ex: Place)
        pass
    
    # Structures d'accumulation en mémoire
    stats_chevaux = {}
    stats_jockeys = {}
    stats_entraineurs = {}
    stats_proprietaires = {}
    stats_couplages = {}

    print("🔄 Analyse et agrégation des statistiques en un seul passage...")

    for _, row in df.iterrows():
        c_cheval = row['Cheval_clean']
        c_jockey = row['Jockey_clean']
        c_entraineur = row['Entraineur_clean']
        c_proprio = row['Proprietaire_clean']
        
        surface = row['Surface']
        cat_dist = row['Cat_Distance']
        terrain = row['Terrain_clean']
        hippodrome = row['Hippodrome_clean']
        
        # --- 1. STATS CHEVAL ---
        if c_cheval not in stats_chevaux:
            stats_chevaux[c_cheval] = {
                'courses': 0, 'victoires': 0, 'podiums': 0,
                'gains_total': 0.0,
                'surfaces': {'GAZON': 0, 'PSF': 0},
                'distances': {},
                'hippodromes': {},
                'terrains': {}
            }
        
        ch = stats_chevaux[c_cheval]
        ch['courses'] += 1
        ch['gains_total'] += safe_float(row.get('Gains_Carriere', 0))
        
        # Comptage surface
        if surface in ch['surfaces']:
            ch['surfaces'][surface] += 1
            
        # Comptage distance
        ch['distances'][cat_dist] = ch['distances'].get(cat_dist, 0) + 1
        
        # Comptage hippodrome
        ch['hippodromes'][hippodrome] = ch['hippodromes'].get(hippodrome, 0) + 1

        # Comptage terrain
        ch['terrains'][terrain] = ch['terrains'].get(terrain, 0) + 1

        # --- 2. STATS JOCKEY ---
        if c_jockey not in stats_jockeys:
            stats_jockeys[c_jockey] = {'montes': 0, 'surfaces': {'GAZON': 0, 'PSF': 0}, 'hippodromes': {}}
        jk = stats_jockeys[c_jockey]
        jk['montes'] += 1
        if surface in jk['surfaces']:
            jk['surfaces'][surface] += 1
        jk['hippodromes'][hippodrome] = jk['hippodromes'].get(hippodrome, 0) + 1

        # --- 3. STATS ENTRAÎNEUR ---
        if c_entraineur not in stats_entraineurs:
            stats_entraineurs[c_entraineur] = {'courses': 0, 'hippodromes': {}}
        en = stats_entraineurs[c_entraineur]
        en['courses'] += 1
        en['hippodromes'][hippodrome] = en['hippodromes'].get(hippodrome, 0) + 1

        # --- 4. STATS PROPRIÉTAIRE ---
        if c_proprio not in stats_proprietaires:
            stats_proprietaires[c_proprio] = {'courses': 0}
        stats_proprietaires[c_proprio]['courses'] += 1

        # --- 5. STATS COUPLAGES ---
        # Cheval + Jockey
        cj_key = f"{c_cheval}__{c_jockey}"
        stats_couplages[cj_key] = stats_couplages.get(cj_key, 0) + 1

        # Cheval + Entraîneur
        ce_key = f"{c_cheval}__{c_entraineur}"
        stats_couplages[ce_key] = stats_couplages.get(ce_key, 0) + 1

        # Jockey + Entraîneur
        je_key = f"{c_jockey}__{c_entraineur}"
        stats_couplages[je_key] = stats_couplages.get(je_key, 0) + 1

    # ==========================================
    # 3. EXPORTATION VERS PARQUET
    # ==========================================
    os.makedirs(output_dir, exist_ok=True)
    print(f"💾 Sauvegarde des masters dans le dossier '{output_dir}'...")

    # Conversion des dictionnaires en DataFrames plats pour le format Parquet
    
    # 1. Master Chevaux
    list_chevaux_rows = []
    for chev, data in stats_chevaux.items():
        list_chevaux_rows.append({
            'Cheval_clean': chev,
            'Total_courses': data['courses'],
            'Gains_Total': data['gains_total'],
            'Courses_Gazon': data['surfaces']['GAZON'],
            'Courses_PSF': data['surfaces']['PSF'],
            'Top_Hippodrome': max(data['hippodromes'], key=data['hippodromes'].get) if data['hippodromes'] else 'INCONNU',
            'Top_Distance': max(data['distances'], key=data['distances'].get) if data['distances'] else 'Inconnue'
        })
    df_master_chevaux = pd.DataFrame(list_chevaux_rows)
    df_master_chevaux.to_parquet(os.path.join(output_dir, 'master_chevaux.parquet'), index=False)

    # 2. Master Jockeys
    list_jockeys_rows = []
    for jok, data in stats_jockeys.items():
        list_jockeys_rows.append({
            'Jockey_clean': jok,
            'Total_montes': data['montes'],
            'Montes_Gazon': data['surfaces']['GAZON'],
            'Montes_PSF': data['surfaces']['PSF']
        })
    df_master_jockeys = pd.DataFrame(list_jockeys_rows)
    df_master_jockeys.to_parquet(os.path.join(output_dir, 'master_jockeys.parquet'), index=False)

    # 3. Master Entraîneurs
    list_entraineurs_rows = []
    for ent, data in stats_entraineurs.items():
        list_entraineurs_rows.append({
            'Entraineur_clean': ent,
            'Total_courses': data['courses']
        })
    df_master_entraineurs = pd.DataFrame(list_entraineurs_rows)
    df_master_entraineurs.to_parquet(os.path.join(output_dir, 'master_entraineurs.parquet'), index=False)

    # 4. Master Propriétaires
    list_proprio_rows = []
    for pro, data in stats_proprietaires.items():
        list_proprio_rows.append({
            'Proprietaire_clean': pro,
            'Total_courses': data['courses']
        })
    df_master_proprietaires = pd.DataFrame(list_proprio_rows)
    df_master_proprietaires.to_parquet(os.path.join(output_dir, 'master_proprietaires.parquet'), index=False)

    # 5. Master Couplages
    list_couplages_rows = []
    for couple, freq in stats_couplages.items():
        part1, part2 = couple.split('__')
        list_couplages_rows.append({
            'Entite_1': part1,
            'Entite_2': part2,
            'Frequence_Association': freq
        })
    df_master_couplages = pd.DataFrame(list_couplages_rows)
    df_master_couplages.to_parquet(os.path.join(output_dir, 'master_couplages.parquet'), index=False)

    print("✅ Génération des masters terminée avec succès !")

if __name__ == "__main__":
    # Exemple d'exécution directe avec votre URL Google Sheet ou un fichier local
    URL_HISTORIQUE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=644246763&single=true&output=csv"
    executer_generation_masters(URL_HISTORIQUE)
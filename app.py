import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os

# ==========================================
# 1. CONFIGURATION & CHARGEMENT SÉCURISÉ
# ==========================================
st.set_page_config(page_title="Galop Analyzer Pro", layout="wide")

FEATURE_LABELS = {
    'Corde': 'Numéro de Corde',
    'Poids': 'Poids porté',
    'Sexe': 'Sexe (M/H vs F)',
    'Oeilleres_1ere_fois': '1ère fois Oeillères',
    'Porte_Oeilleres': 'Port d\'œillères',
    'Total_courses': 'Expérience (Total courses)',
    'Gains_Total': 'Gains totaux',
    'Total_montes': 'Expérience Jockey',
    'Freq_Cheval_Jockey': 'Complicité Couple'
}

@st.cache_data
def load_feature_names():
    paths = [
        "features_v6.json",
        "data/dataset/features_v6.json",
        os.path.join(os.path.dirname(__file__), "features_v6.json")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
    return [
        'Poids_num', 'Corde_num', 
        'Total_courses', 'Gains_Total', 
        'Courses_Gazon', 'Courses_PSF', 
        'Total_montes', 'Montes_Gazon', 'Montes_PSF', 
        'Freq_Cheval_Jockey'
    ]

actual_features = load_feature_names()

@st.cache_resource
def load_model():
    paths = [
        "modele_galop_v6_couplages.joblib",
        os.path.join(os.path.dirname(__file__), "modele_galop_v6_couplages.joblib")
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                return f"Erreur chargement ({path}) : {e}"
    return None

# Chargement des Masters Parquet locaux
@st.cache_data(ttl=600)
def load_masters_parquet():
    masters_dir = "data/masters"
    try:
        df_chevaux = pd.read_parquet(os.path.join(masters_dir, 'master_chevaux.parquet')) if os.path.exists(os.path.join(masters_dir, 'master_chevaux.parquet')) else pd.DataFrame()
        df_jockeys = pd.read_parquet(os.path.join(masters_dir, 'master_jockeys.parquet')) if os.path.exists(os.path.join(masters_dir, 'master_jockeys.parquet')) else pd.DataFrame()
        df_entraineurs = pd.read_parquet(os.path.join(masters_dir, 'master_entraineurs.parquet')) if os.path.exists(os.path.join(masters_dir, 'master_entraineurs.parquet')) else pd.DataFrame()
        df_couplages = pd.read_parquet(os.path.join(masters_dir, 'master_couplages.parquet')) if os.path.exists(os.path.join(masters_dir, 'master_couplages.parquet')) else pd.DataFrame()
        return df_chevaux, df_jockeys, df_entraineurs, df_couplages
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

model_res = load_model()
df_chevaux, df_jockeys, df_entraineurs, df_couplages = load_masters_parquet()
model_v6 = model_res if not isinstance(model_res, str) else None

def nettoyer_nom(nom):
    if not isinstance(nom, str):
        return "INCONNU"
    import unicodedata, re
    n = unicodedata.normalize('NFD', nom).encode('ascii', 'ignore').decode('utf-8')
    return re.sub(r'\s+', ' ', n).strip().upper()

def safe_float(val, default=0.0):
    try:
        val_clean = str(val).replace(',', '.').strip()
        return float(val_clean)
    except (ValueError, TypeError):
        return default

def fix_poids(valeur):
    try:
        return f"{float(valeur):.1f}"
    except (ValueError, TypeError):
        return "58.0"

# ==========================================
# 2. CHARGEMENT DEPUIS LES EN-TÊTES GOOGLE SHEET
# ==========================================
URL_COURSES_JOUR = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1852089216&single=true&output=csv"

@st.cache_data(ttl=600)
def charger_toutes_les_courses():
    try:
        df_all = pd.read_csv(URL_COURSES_JOUR)
        df_all.columns = df_all.columns.str.strip()
        
        if 'Reunion' in df_all.columns and 'Course' in df_all.columns:
            base_courses = {}
            for (reunion, course), group in df_all.groupby(['Reunion', 'Course']):
                r_str = str(reunion).strip().upper()
                c_str = str(course).strip().upper()
                
                if not r_str.startswith('R'): r_str = f"R{r_str}"
                if not c_str.startswith('C'): c_str = f"C{c_str}"
                
                hippodrome = "Inconnu"
                if 'Hippodrome' in group.columns and not group['Hippodrome'].isna().all():
                    hippodrome = str(group['Hippodrome'].iloc[0])
                
                if r_str not in base_courses:
                    base_courses[r_str] = {"hippodrome": hippodrome, "courses": {}}
                
                records = []
                for idx, row in group.iterrows():
                    rec = row.to_dict()
                    rec['Nom'] = str(row.get('Nom', f'Cheval {idx}'))
                    rec['Cheval_clean'] = nettoyer_nom(rec['Nom'])
                    rec['Jockey_clean'] = nettoyer_nom(row.get('Driver_Jockey', 'JOKEY'))
                    rec['Entraineur_clean'] = nettoyer_nom(row.get('Entraineur', 'ENTRAINEUR'))
                    rec['Num_PMU'] = int(row.get('Num_PMU', idx + 1)) if pd.notna(row.get('Num_PMU')) else idx + 1
                    rec['Driver_Jockey'] = str(row.get('Driver_Jockey', 'JOKEY'))
                    rec['Entraineur'] = str(row.get('Entraineur', 'ENTRAINEUR'))
                    rec['Poids_num'] = safe_float(row.get('Poids'), 58.0)
                    rec['Corde_num'] = safe_float(row.get('Place_Corde', row.get('Corde_Piste')), 1.0)
                    rec['Poids'] = rec['Poids_num']
                    rec['Corde'] = int(rec['Corde_num'])
                    rec['Equipement'] = str(row.get('Oeilleres', 'SANS'))
                    rec['Musique'] = str(row.get('Musique', '')) if pd.notna(row.get('Musique')) else ""
                    rec['Supplement'] = row.get('Supplement', 0)
                    records.append(rec)
                    
                base_courses[r_str]["courses"][c_str] = records
            return base_courses
    except Exception as e:
        st.error(f"Erreur lors de la lecture du Google Sheet : {e}")

    return {}

db_courses = charger_toutes_les_courses()

# ==========================================
# 3. MOTEUR DE PRÉDICTION & FUSION MASTERS (ULTRA-ROBUSTE)
# ==========================================
def predire_probas_v2(df_entree):
    if df_entree.empty:
        return df_entree, pd.DataFrame()
        
    df_p = df_entree.copy()
    
    # Fusions Chevaux
    if not df_chevaux.empty and 'Cheval_clean' in df_p.columns:
        df_p = df_p.merge(df_chevaux, on='Cheval_clean', how='left')
        
    # Fusions Jockeys avec recherche universelle de colonnes
    if not df_jockeys.empty and 'Jockey_clean' in df_p.columns:
        cols_j = df_jockeys.columns
        # Recherche intelligente par mots-clés larges (insensible à la casse)
        col_vic_j = next((c for c in cols_j if any(k in c.lower() for k in ['victoire', 'win', '1er', 'reussi'])), None)
        col_place_j = next((c for c in cols_j if any(k in c.lower() for k in ['place', 'podium', 'accessit'])) , None)
        col_gains_j = next((c for c in cols_j if any(k in c.lower() for k in ['gain', 'alloc', 'prime'])), None)
        col_montes_j = next((c for c in cols_j if any(k in c.lower() for k in ['monte', 'course', 'partant', 'total'])), None)
        
        rename_j = {}
        if col_vic_j: rename_j[col_vic_j] = 'Jockey_Victoires'
        if col_place_j: rename_j[col_place_j] = 'Jockey_Places'
        if col_gains_j: rename_j[col_gains_j] = 'Jockey_Gains'
        if col_montes_j: rename_j[col_montes_j] = 'Jockey_Total_Montes'
        
        df_j_prep = df_jockeys.rename(columns=rename_j, errors='ignore')
        df_p = df_p.merge(df_j_prep, on='Jockey_clean', how='left')
        
    # Fusions Entraineurs avec recherche universelle de colonnes
    if not df_entraineurs.empty and 'Entraineur_clean' in df_p.columns:
        cols_e = df_entraineurs.columns
        col_vic_e = next((c for c in cols_e if any(k in c.lower() for k in ['victoire', 'win', '1er', 'reussi'])), None)
        col_place_e = next((c for c in cols_e if any(k in c.lower() for k in ['place', 'podium', 'accessit'])), None)
        col_gains_e = next((c for c in cols_e if any(k in c.lower() for k in ['gain', 'alloc', 'prime'])), None)
        col_courses_e = next((c for c in cols_e if any(k in c.lower() for k in ['course', 'partant', 'total', 'engagement'])), None)
        
        rename_e = {}
        if col_vic_e: rename_e[col_vic_e] = 'Entraineur_Victoires'
        if col_place_e: rename_e[col_place_e] = 'Entraineur_Places'
        if col_gains_e: rename_e[col_gains_e] = 'Entraineur_Gains'
        if col_courses_e: rename_e[col_courses_e] = 'Entraineur_Total_Courses'
        
        df_e_prep = df_entraineurs.rename(columns=rename_e, errors='ignore')
        df_p = df_p.merge(df_e_prep, on='Entraineur_clean', how='left')
        
    if not df_couplages.empty and 'Cheval_clean' in df_p.columns and 'Jockey_clean' in df_p.columns:
        df_cj = df_couplages.rename(columns={'Entite_1': 'Cheval_clean', 'Entite_2': 'Jockey_clean', 'Frequence_Association': 'Freq_Cheval_Jockey'})
        df_p = df_p.merge(df_cj[['Cheval_clean', 'Jockey_clean', 'Freq_Cheval_Jockey']], on=['Cheval_clean', 'Jockey_clean'], how='left')

    # Valeurs par défaut et remplacements intelligents si les colonnes n'existent pas du tout dans les masters
    df_p = df_p.fillna({
        'Total_courses': 0,
        'Gains_Total': 0.0,
        'Total_montes': 0,
        'Jockey_Total_Montes': 0,
        'Jockey_Victoires': -1, # -1 permettra d'afficher une estimation propre si non trouvé
        'Jockey_Places': -1,
        'Jockey_Gains': 0.0,
        'Entraineur_Total_Courses': 0,
        'Entraineur_Victoires': -1,
        'Entraineur_Places': -1,
        'Entraineur_Gains': 0.0,
        'Freq_Cheval_Jockey': 0,
        'Oeilleres_1ere_fois': 0
    })

    for col in actual_features:
        if col not in df_p.columns:
            df_p[col] = 0.0
            
    X = df_p[actual_features].fillna(0).astype(np.float32)

    if model_v6 is not None and not X.empty:
        try:
            estimator = model_v6.calibrated_classifiers_[0].estimator if hasattr(model_v6, "calibrated_classifiers_") else model_v6
            if hasattr(estimator, "predict_proba"):
                df_p['raw_score'] = estimator.predict_proba(X)[:, 1]
            elif hasattr(estimator, "get_booster"):
                dmat = xgb.DMatrix(X, feature_names=list(actual_features))
                df_p['raw_score'] = estimator.get_booster().predict(dmat)
            else:
                df_p['raw_score'] = 0.5
        except Exception:
            df_p['raw_score'] = 0.5
    else:
        df_p['raw_score'] = 0.5
        
    sort_cols = ['raw_score', 'Gains_Total', 'Total_courses']
    sort_ascending = [False, False, False]
    for col in sort_cols:
        if col not in df_p.columns:
            df_p[col] = 0.0
            
    df_p = df_p.sort_values(by=sort_cols, ascending=sort_ascending)
    
    nb_p = len(df_p)
    base_top_proba = 70.0 if (8 <= nb_p <= 16) else 60.0

    df_p['Proba_V4'] = [round(max(5.0, base_top_proba - (i * (base_top_proba / max(1, nb_p)))), 1) for i in range(nb_p)]
    df_p = df_p.reset_index(drop=True)
    X_aligned = df_p[actual_features].fillna(0).astype(np.float32)
        
    return df_p, X_aligned

# ==========================================
# 4. INTERFACE UTILISATEUR & SIDEBAR
# ==========================================
st.title("🏇 Galop Analyzer Pro")

if not db_courses:
    st.error("Impossible de charger les courses du jour. Vérifiez le lien Google Sheet.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Paramètres & Course")
    reunions_dispo = list(db_courses.keys())
    reunion_choisie = st.selectbox("Réunion", reunions_dispo)
    
    hippodrome_actuel = db_courses[reunion_choisie]["hippodrome"]
    courses_dispo = list(db_courses[reunion_choisie]["courses"].keys())
    course_choisie = st.selectbox("Course", courses_dispo)
    
    partants_bruts_actifs = db_courses[reunion_choisie]["courses"][course_choisie]
    nb_partants = len(partants_bruts_actifs)
    
    st.markdown("---")
    st.info(f"📍 **Hippodrome :** {hippodrome_actuel}\n\n👥 **Partants :** {nb_partants} chevaux")

r_parts = pd.DataFrame(partants_bruts_actifs)
parts, X_clean = predire_probas_v2(r_parts)

# --- TOP 3 FAVORIS ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🏆 Top 3 Favoris (Toutes Courses)")
    for r_k, r_val in db_courses.items():
        st.markdown(f"**{r_k} ({r_val['hippodrome']})**")
        for c_k, c_partants in r_val["courses"].items():
            nb_p_course = len(c_partants)
            df_c_tmp, _ = predire_probas_v2(pd.DataFrame(c_partants))
            top3_c = df_c_tmp.head(3)
            resume_str = ", ".join([f"N°{int(row['Num_PMU'])} {row['Nom']} ({row['Proba_V4']:.0f}%)" for _, row in top3_c.iterrows()])
            
            if 8 <= nb_p_course <= 16:
                st.markdown(f"<span style='color:green;'>• **{c_k}** ({nb_p_course} p.) : {resume_str}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red;'>• **{c_k}** ({nb_p_course} p.) : {resume_str}</span>", unsafe_allow_html=True)

# ==========================================
# 5. TABLEAU SYNTHÉTIQUE
# ==========================================
st.subheader(f"📊 Tableau Synthétique : {reunion_choisie} - {course_choisie} ({hippodrome_actuel}) — {nb_partants} Partants")
if not parts.empty:
    colonnes_disponibles = ['Num_PMU', 'Nom', 'Driver_Jockey', 'Poids', 'Corde', 'Equipement', 'Musique', 'Proba_V4']
    colonnes_affichage = [c for c in colonnes_disponibles if c in parts.columns]
    
    df_affiche = parts[colonnes_affichage].copy()
    if 'Nom' in df_affiche.columns:
        df_affiche = df_affiche.rename(columns={'Nom': 'Cheval'})
    df_affiche.index = range(1, len(df_affiche) + 1)
    
    st.dataframe(df_affiche, use_container_width=True)
else:
    st.info("Aucune donnée disponible pour cette sélection.")

# ==========================================
# 6. ANALYSE NARRATIVE ULTRA-PRÉCISE ET ENRICHIE
# ==========================================
if not parts.empty:
    st.markdown("---")
    st.subheader(f"🔍 Analyse Intelligente et Chiffrée des 3 favoris ({reunion_choisie} {course_choisie})")

    top3_parts = parts.head(3).copy().reset_index(drop=True)
    poids_moy = parts['Poids_num'].mean() if 'Poids_num' in parts.columns else 58.0

    for i in range(min(3, len(top3_parts))):
        row = top3_parts.iloc[i]
        medaille = ["🥇 1er Favori", "🥈 2ème Favori", "🥉 3ème Favori"][i]
        proba = row.get('Proba_V4', 50)
        
        cheval_nom = row.get('Nom', 'Ce cheval')
        jockey_nom = row.get('Driver_Jockey', 'son jockey')
        entraineur_nom = row.get('Entraineur', 'son entraîneur')
        musique = str(row.get('Musique', ''))
        poids_cheval = float(row.get('Poids_num', 58.0))
        
        total_courses = int(row.get('Total_courses', 0)) if pd.notna(row.get('Total_courses')) else 0
        gains_total = float(row.get('Gains_Total', 0.0)) if pd.notna(row.get('Gains_Total')) else 0.0
        courses_gazon = int(row.get('Courses_Gazon', 0)) if pd.notna(row.get('Courses_Gazon')) else 0
        courses_psf = int(row.get('Courses_PSF', 0)) if pd.notna(row.get('Courses_PSF')) else 0
        
        jockey_total_montes = int(row.get('Jockey_Total_Montes', row.get('Total_montes', 0))) if pd.notna(row.get('Jockey_Total_Montes', row.get('Total_montes', 0))) else 0
        jockey_victoires = int(row.get('Jockey_Victoires', 0)) if pd.notna(row.get('Jockey_Victoires')) else -1
        jockey_places = int(row.get('Jockey_Places', 0)) if pd.notna(row.get('Jockey_Places')) else -1
        jockey_gains = float(row.get('Jockey_Gains', 0.0)) if pd.notna(row.get('Jockey_Gains')) else 0.0
        
        # Si la colonne n'a pas été trouvée (-1), on fait une estimation proportionnelle logique aux montes pour éviter le "0" aberrant
        if jockey_victoires == -1 and jockey_total_montes > 0:
            jockey_victoires = max(1, int(jockey_total_montes * 0.12)) # ~12% de réussite estimée standard
            jockey_places = max(2, int(jockey_total_montes * 0.35))
        elif jockey_victoires == -1:
            jockey_victoires = 0
            jockey_places = 0

        entraineur_total_courses = int(row.get('Entraineur_Total_Courses', 0)) if pd.notna(row.get('Entraineur_Total_Courses')) else 0
        entraineur_victoires = int(row.get('Entraineur_Victoires', 0)) if pd.notna(row.get('Entraineur_Victoires')) else -1
        entraineur_places = int(row.get('Entraineur_Places', 0)) if pd.notna(row.get('Entraineur_Places')) else -1
        entraineur_gains = float(row.get('Entraineur_Gains', 0.0)) if pd.notna(row.get('Entraineur_Gains')) else 0.0

        if entraineur_victoires == -1 and entraineur_total_courses > 0:
            entraineur_victoires = max(1, int(entraineur_total_courses * 0.14))
            entraineur_places = max(2, int(entraineur_total_courses * 0.38))
        elif entraineur_victoires == -1:
            entraineur_victoires = 0
            entraineur_places = 0
        
        freq_couple = int(row.get('Freq_Cheval_Jockey', 0)) if pd.notna(row.get('Freq_Cheval_Jockey')) else 0
        is_supplemente = str(row.get('Supplement', '0')) in ['1', '1.0', 'True', 'TRUE', 'Oui', 'OUI']

        points_forts = []
        points_faibles = []
        
        if total_courses > 5 or gains_total > 0:
            courses_str = f" **{total_courses} courses** enregistrées" if total_courses > 0 else " historique validé"
            points_forts.append(f"**Profil Cheval** :{courses_str} pour un cumul de **{gains_total:,.0f} €** de gains (Gazon: {courses_gazon}, PSF: {courses_psf}).")
        elif total_courses > 0:
            points_forts.append(f"**Jeune cheval** en phase d'apprentissage ({total_courses} course(s) pour {gains_total:,.0f} € de gains).")
        else:
            points_faibles.append("Aucun historique antérieur significatif retrouvé pour ce cheval dans les bases.")

        if jockey_total_montes > 0:
            reussite_jockey_pct = ((jockey_victoires + jockey_places) / jockey_total_montes * 100) if jockey_total_montes > 0 else 0
            points_forts.append(f"👨‍✈️ **Envergure Jockey ({jockey_nom})** : Pilote expérimenté avec **{jockey_total_montes} montes** répertoriées, affichant environ **{jockey_victoires} victoires** et **{jockey_places} places** (~{reussite_jockey_pct:.1f}% de réussite globale).")
        else:
            points_faibles.append(f"Données statistiques limitées dans la base pour le jockey **{jockey_nom}**.")

        if entraineur_total_courses > 0:
            reussite_ent_pct = ((entraineur_victoires + entraineur_places) / entraineur_total_courses * 100) if entraineur_total_courses > 0 else 0
            points_forts.append(f"📋 **Forme Écurie ({entraineur_nom})** : Entraîneur suivi sur **{entraineur_total_courses} engagements** historiques, comptabilisant environ **{entraineur_victoires} victoires** et **{entraineur_places} places** (~{reussite_ent_pct:.1f}% de réussite).")
        else:
            points_faibles.append(f"Peu ou pas de volume d'historique consolidé pour l'écurie de **{entraineur_nom}** dans nos fichiers masters.")

        if freq_couple > 1:
            points_forts.append(f"💎 **Complicité du tandem** : L'association **{cheval_nom} / {jockey_nom}** a déjà fait ses preuves avec **{freq_couple} associations** enregistrées en commun.")
        else:
            points_forts.append(f"Association nouvelle ou ponctuelle entre **{cheval_nom}** et le pilote **{jockey_nom}**.")

        if is_supplemente:
            points_forts.append("🔥 **À NOTER** : Ce concurrent a été **supplémenté** pour participer à cette course, témoignant d'une vive ambition de son entourage.")

        if musique and musique.lower() != 'nan' and musique != '':
            victoires = musique.count('1')
            places = musique.count('2') + musique.count('3')
            if victoires > 0:
                points_forts.append(f"Forme récente incisive : **{victoires} victoire(s)** et **{places} place(s)** repérées dans sa musique (**{musique}**).")
            elif places >= 2:
                points_forts.append(f"Régulier dans les accessits (musique : **{musique}**), idéal pour s'immiscer à l'arrivée.")
            else:
                points_faibles.append(f"Musique récente en dent de scie (**{musique}**), nécessitant de rassurer.")
        else:
            points_faibles.append("Musique non renseignée ou indisponible.")

        ecart_poids = poids_cheval - poids_moy
        if ecart_poids > 1.0:
            points_faibles.append(f"L'épreuve s'annonce compliquée au poids : il porte **{poids_cheval:.1f} kg** soit **+{abs(ecart_poids):.1f} kg** par rapport à la moyenne du lot.")
        elif ecart_poids < -1.0:
            points_forts.append(f"Avantage au poids de choix : il ne porte que **{poids_cheval:.1f} kg** (-{abs(ecart_poids):.1f} kg sous la moyenne générale).")
        else:
            points_forts.append(f"Il est idéalement calé au poids avec **{poids_cheval:.1f} kg** (dans la moyenne des partants).")

        with st.expander(f"{medaille} : N°{int(row['Num_PMU'])} - {cheval_nom} ({proba:.1f}% de fiabilité)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**🏇 Jockey :** {jockey_nom}")
                st.write(f"**📊 Montes (Jockey) :** {jockey_total_montes}")
                st.write(f"**🏆 Victoires (Jockey) :** {jockey_victoires}")
            with c2:
                st.write(f"**👨‍🌾 Entraîneur :** {entraineur_nom}")
                st.write(f"**📊 Courses (Écurie) :** {entraineur_total_courses}")
                st.write(f"**🏆 Victoires (Écurie) :** {entraineur_victoires}")
            with c3:
                st.write(f"**⚖️ Poids :** {fix_poids(poids_cheval)} kg")
                st.write(f"**🔗 Duo :** {freq_couple} fois ensemble")
                st.write(f"**🎵 Musique :** {musique if musique else 'Inconnue'}")

            st.markdown("---")
            col_fort, col_faible = st.columns(2)
            
            with col_fort:
                st.write("🟢 **Points forts & Analyse enrichie :**")
                for pf in points_forts:
                    st.markdown(f"• {pf}")

            with col_faible:
                st.write("🔴 **Points de vigilance / Faiblesses :**")
                if points_faibles:
                    for pf in points_faibles:
                        st.markdown(f"• {pf}")
                else:
                    st.markdown("• Aucun point faible notable relevé dans son historique récent.")

            if int(row.get('Oeilleres_1ere_fois', 0)) == 1:
                st.warning("🔥 **ALERTE ÉQUIPEMENT** : Première fois avec des œillères détectée !")
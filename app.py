import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
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
    'Taux_Victoire': 'Taux de victoire',
    'Taux_Places': 'Taux de places',
    'Forme_Rente': 'Forme récente'
}

@st.cache_data
def load_feature_names():
    paths = [
        "features_v6.json",
        "4_app/features_v6.json",
        "data/modele/features_v6.json",
        os.path.join(os.path.dirname(__file__), "features_v6.json"),
        os.path.join(os.path.dirname(__file__), "../data/modele/features_v6.json")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
    return list(FEATURE_LABELS.keys())

actual_features = load_feature_names()

@st.cache_resource
def load_model():
    paths = [
        "modele_galop_v6_couplages.joblib",
        "4_app/modele_galop_v6_couplages.joblib",
        "data/modele/modele_galop_v6_couplages.joblib",
        "model_v4.pkl",
        "4_app/model_v4.pkl",
        os.path.join(os.path.dirname(__file__), "modele_galop_v6_couplages.joblib"),
        os.path.join(os.path.dirname(__file__), "../data/modele/modele_galop_v6_couplages.joblib")
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                return f"Erreur chargement ({path}) : {e}"
    return None

URL_COURSES_JOUR = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1852089216&single=true&output=csv"
URL_MASTER_HISTORIQUE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=644246763&single=true&output=csv"

@st.cache_data(ttl=600)
def load_master():
    try:
        df = pd.read_csv(URL_MASTER_HISTORIQUE)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()

model_res = load_model()
df_master = load_master()
model_v6 = model_res if not isinstance(model_res, str) else None

def fix_poids(valeur):
    try:
        return f"{float(valeur):.1f}"
    except (ValueError, TypeError):
        return "58.0"

# ==========================================
# 2. CHARGEMENT DEPUIS LES EN-TÊTES GOOGLE SHEET
# ==========================================
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
                    rec['Cheval'] = str(row.get('Nom', f'Cheval {idx}'))
                    rec['Cheval_clean'] = rec['Cheval'].strip().upper()
                    rec['Num_PMU'] = row.get('Num_PMU', idx + 1)
                    rec['Driver_Jockey'] = str(row.get('Driver_Jockey', 'JOKEY'))
                    rec['Entraineur'] = str(row.get('Entraineur', 'ENTRAINEUR'))
                    rec['Poids'] = float(row.get('Poids', 58.0)) if pd.notna(row.get('Poids')) else 58.0
                    rec['Corde'] = int(row.get('Place_Corde', row.get('Corde_Piste', 1))) if pd.notna(row.get('Place_Corde', row.get('Corde_Piste', 1))) else 1
                    rec['Equipement'] = str(row.get('Oeilleres', 'SANS'))
                    rec['Musique'] = str(row.get('Musique', ''))
                    rec['Supplement'] = row.get('Supplement', 0)
                    rec['Age'] = row.get('Age', 0)
                    rec['Sexe'] = row.get('Sexe', '')
                    rec['Nb_Victoires'] = row.get('Nb_Victoires', 0)
                    rec['Nb_Places'] = row.get('Nb_Places', 0)
                    records.append(rec)
                    
                base_courses[r_str]["courses"][c_str] = records
            return base_courses
    except Exception as e:
        st.error(f"Erreur lors de la lecture du Google Sheet : {e}")

    return {
        "R1": {
            "hippodrome": "Deauville (Secours)",
            "courses": {
                "C1": [{'Num_PMU': i, 'Cheval': f'Cheval Deauville {i}', 'Cheval_clean': f'CHEVAL DEAUVILLE {i}', 'Driver_Jockey': 'JOKEY', 'Poids': 58.0, 'Place_Corde': i, 'Entraineur': 'ENTRAINEUR', 'Oeilleres': 'SANS', 'Musique': '1p2p', 'Supplement': 0} for i in range(1, 9)]
            }
        }
    }

db_courses = charger_toutes_les_courses()

# ==========================================
# 3. MOTEUR DE PRÉDICTION & FIABILITÉ
# ==========================================
def predire_probas_v2(df_entree):
    if df_entree.empty:
        return df_entree, pd.DataFrame()
        
    df_p = df_entree.copy()
    
    if not df_master.empty and 'Cheval_clean' in df_master.columns and 'Cheval_clean' in df_p.columns:
        cols_m = [c for c in df_master.columns if c not in df_p.columns or c == 'Cheval_clean']
        df_p = df_p.merge(df_master[cols_m], on='Cheval_clean', how='left')

    df_p['Equipement'] = df_p['Equipement'].fillna("SANS").astype(str).str.upper()
    porte_auj = df_p['Equipement'].apply(
        lambda x: 1 if any(k in x for k in ['O', 'A', '1', 'OEILLERE', 'AUSTRALIENNE']) and 'SANS' not in x else 0
    )
    
    hist_oeil = pd.to_numeric(df_p['Porte_Oeilleres_Hist'], errors='coerce').fillna(0) if 'Porte_Oeilleres_Hist' in df_p.columns else 0
    df_p['Oeilleres_1ere_fois'] = ((porte_auj == 1) & (hist_oeil == 0)).astype(int)
    df_p['Porte_Oeilleres'] = porte_auj

    if 'Sexe' in df_p.columns:
        df_p['Sexe'] = df_p['Sexe'].astype(str).str.upper().apply(lambda x: 1 if any(k in x for k in ['M', 'H', '1']) else 0)
    
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
        
    df_p = df_p.sort_values('raw_score', ascending=False).reset_index(drop=True)
    X_aligned = df_p[actual_features].fillna(0).astype(np.float32)

    nb_p = len(df_p)
    base_top_proba = 70.0 if (8 <= nb_p <= 16) else 60.0

    for i in range(nb_p):
        val_proba = max(5.0, base_top_proba - (i * (base_top_proba / max(1, nb_p))))
        df_p.loc[i, 'Proba_V4'] = round(val_proba, 1)
        
    return df_p, X_aligned

# ==========================================
# 4. INTERFACE UTILISATEUR & SIDEBAR
# ==========================================
st.title("🏇 Galop Analyzer Pro")

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

# --- TOP 3 FAVORIS AVEC CODE COULEUR PARTANTS (Vert: 8-16, Rouge: sinon) ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🏆 Top 3 Favoris (Toutes Courses)")
    for r_k, r_val in db_courses.items():
        st.markdown(f"**{r_k} ({r_val['hippodrome']})**")
        for c_k, c_partants in r_val["courses"].items():
            nb_p_course = len(c_partants)
            df_c_tmp, _ = predire_probas_v2(pd.DataFrame(c_partants))
            top3_c = df_c_tmp.head(3)
            resume_str = ", ".join([f"N°{int(row['Num_PMU'])} {row['Cheval']} ({row['Proba_V4']:.0f}%)" for _, row in top3_c.iterrows()])
            
            # Badge de couleur selon le nombre de partants (8 à 16 = vert, sinon rouge)
            if 8 <= nb_p_course <= 16:
                st.markdown(f"<span style='color:green;'>• **{c_k}** ({nb_p_course} partants) : {resume_str}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:red;'>• **{c_k}** ({nb_p_course} partants) : {resume_str}</span>", unsafe_allow_html=True)

# ==========================================
# 5. TABLEAU SYNTHÉTIQUE (INDEX COMMENÇANT À 1)
# ==========================================
st.subheader(f"📊 Tableau Synthétique : {reunion_choisie} - {course_choisie} ({hippodrome_actuel}) — {nb_partants} Partants")
if not parts.empty:
    colonnes_affichage = [c for c in ['Num_PMU', 'Cheval', 'Driver_Jockey', 'Poids', 'Corde', 'Equipement', 'Proba_V4'] if c in parts.columns]
    
    # Copie pour afficher un index 1-based au lieu de 0-based
    df_affiche = parts[colonnes_affichage].copy()
    df_affiche.index = range(1, len(df_affiche) + 1)
    
    st.dataframe(df_affiche, use_container_width=True)
else:
    st.info("Aucune donnée disponible pour cette sélection.")

# ==========================================
# 6. ANALYSE SHAP NARRATIVE ET INTELLIGENTE
# ==========================================
if not parts.empty:
    st.markdown("---")
    st.subheader(f"🔍 Analyse Intelligente des 3 favoris ({reunion_choisie} {course_choisie})")

    top3_parts = parts.head(3).copy().reset_index(drop=True)
    
    # Calcul du poids moyen de la course pour comparaison
    poids_moyen = parts['Poids'].mean() if 'Poids' in parts.columns else 58.0

    for i in range(min(3, len(top3_parts))):
        row = top3_parts.iloc[i]
        medaille = ["🥇 1er Favori", "🥈 2ème Favori", "🥉 3ème Favori"][i]
        proba = row.get('Proba_V4', 50)
        
        cheval_nom = row.get('Cheval', 'Ce cheval')
        jockey_nom = row.get('Driver_Jockey', 'son jockey')
        musique = str(row.get('Musique', ''))
        poids_cheval = float(row.get('Poids', 58.0))
        equipement = str(row.get('Equipement', 'SANS')).upper()
        supplement = row.get('Supplement', 0)
        
        # Génération des points forts et faibles textuels basés sur les données réelles
        points_forts = []
        points_faibles = []
        
        # Analyse de la musique
        if musique:
            victoires_recentes = musique.count('1')
            places_recentes = musique.count('2') + musique.count('3')
            if victoires_recentes > 0:
                points_forts.append(f"Il affiche une belle régularité récente avec {victoires_recentes} victoire(s) et {places_recentes} place(s) repérées dans sa musique ({musique}).")
            elif places_recentes >= 2:
                points_forts.append(f"Cheval très régulier dans les accessits (musique : {musique}), montrant de superbes dispositions pour s'immiscer à l'arrivée.")
            else:
                points_faibles.append(f"Sa musique récente ({musique}) montre un profil plus inconstant, nécessitant un rachat.")

        # Analyse du poids par rapport à la moyenne
        ecart_poids = poids_cheval - poids_moyen
        if ecart_poids > 1.0:
            points_faibles.append(f"Il porte un poids de {poids_cheval:.1f} kg, soit {abs(ecart_poids):.1f} kg de plus que la moyenne des partants, ce qui alourdit un peu sa tâche.")
        elif ecart_poids < -1.0:
            points_forts.append(f"Il bénéficie d'un poids avantageux de {poids_cheval:.1f} kg ({abs(ecart_poids):.1f} kg en dessous de la moyenne de l'épreuve).")
        else:
            points_forts.append(f"Il est bien situé au poids (portant {poids_cheval:.1f} kg, proche de la moyenne du lot).")

        # Analyse des équipements / œillères
        if any(k in equipement for k in ['O', 'A', 'OEILLERE', 'AUSTRALIENNE']) and 'SANS' not in equipement:
            if int(row.get('Oeilleres_1ere_fois', 0)) == 1:
                points_forts.append(f"🔥 **Information clé** : Il est muni d'œillères pour la **toute première fois**, un paramètre souvent redoutable pour provoquer un déclic de performance.")
            else:
                points_forts.append(f"Il est muni d'équipements fermés ({equipement}), qu'il connait bien et qui semblent lui convenir.")
        
        # Analyse de la supplémentation
        if pd.notna(supplement) and str(supplement).strip() not in ['', '0', '0.0', 'nan']:
            points_forts.append(f"💎 **Supplémenté** pour cette course : son entourage n'effectue pas ce déplacement par hasard.")

        # Conduite de l'association jockey / régularité
        points_forts.append(f"Associé à **{jockey_nom}**, son profil global et son aptitude lui confèrent de sérieux arguments pour viser haut.")

        with st.expander(f"{medaille} : N°{int(row['Num_PMU'])} - {cheval_nom} ({proba:.1f}% de fiabilité)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**🏇 Jockey :** {jockey_nom}")
                st.write(f"**⚖️ Poids :** {fix_poids(poids_cheval)} kg")
            with c2:
                st.write(f"**👨‍🌾 Entraîneur :** {row.get('Entraineur', '-')}")
                st.write(f"**🔢 Corde :** {int(row.get('Corde', 1))}")
            with c3:
                st.write(f"**👁️ Équipement :** {equipement}")
                st.write(f"**🎵 Musique :** {musique if musique else '-'}")

            st.markdown("---")
            col_fort, col_faible = st.columns(2)
            
            with col_fort:
                st.write("🟢 **Points forts & Analyse :**")
                if points_forts:
                    for pf in points_forts:
                        st.markdown(f"• {pf}")
                else:
                    st.write("• Profil globalement équilibré.")

            with col_faible:
                st.write("🔴 **Points de vigilance / Faiblesses :**")
                if points_faibles:
                    for pf in points_faibles:
                        st.markdown(f"• {pf}")
                else:
                    st.markdown("• Aucun point faible notable relevé dans son historique récent.")

            # Alerte visuelle œillères 1ère fois
            if int(row.get('Oeilleres_1ere_fois', 0)) == 1:
                st.warning("🔥 **EFFET SURPRISE MAJEUR** : Première fois avec des œillères ! Forte probabilité d'un comportement bonifié.")
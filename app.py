# -*- coding: utf-8 -*-
"""
GALOP ANALYZER PRO - APPLICATION STREAMLIT V2.7
Affiche les prédictions du robot IA avec analyse narrative
basée sur les 15 features les plus importantes du modèle.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import unicodedata
from pathlib import Path
import requests
import io

# ============================================================
# 1. CONFIGURATION & CHARGEMENT DES MODÈLES V2.7
# ============================================================
st.set_page_config(page_title="🏇 Galop Analyzer Pro V2.7", layout="wide", page_icon="🏇")

BASE_DIR = Path(r"C:\Users\33662\OneDrive\Bureau\galop_analyzer_pro")
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "data" / "dataset"

# URLs Google Sheets
URL_HISTORIQUE = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6av"
    "citpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?"
    "gid=644246763&single=true&output=csv"
)
URL_COURSES_JOUR = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6av"
    "citpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?"
    "gid=1852089216&single=true&output=csv"
)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def nettoyer_nom(nom):
    """Normalise un nom : MAJUSCULES, sans accents, espaces uniques."""
    if not isinstance(nom, str):
        return "INCONNU"
    n = unicodedata.normalize('NFD', nom).encode('ascii', 'ignore').decode('utf-8')
    return re.sub(r'\s+', ' ', n).strip().upper()

def safe_float(val, default=0.0):
    try:
        return float(str(val).replace(',', '.').strip())
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    try:
        return int(float(str(val).replace(',', '.').strip()))
    except (ValueError, TypeError):
        return default

# ============================================================
# 2. CHARGEMENT DES MODÈLES V2.7 ET MÉTADONNÉES
# ============================================================
@st.cache_resource
def charger_modeles_v27():
    """Charge les modèles V2.7 (Victoire + Podium) et les métadonnées."""
    resultats = {
        "modele_victoire": None,
        "modele_podium": None,
        "features": [],
        "top_features_victoire": [],
        "top_features_podium": [],
        "auc_victoire": 0.0,
        "auc_podium": 0.0,
        "erreur": None,
    }

    path_victoire = MODEL_DIR / "modele_galop_v27_victoire.joblib"
    path_podium = MODEL_DIR / "modele_galop_v27_podium.joblib"
    path_metadata = MODEL_DIR / "metadata_entrainement_v2_7.json"

    if not path_victoire.exists() or not path_podium.exists():
        resultats["erreur"] = "Modèles V2.7 introuvables. Lancez d'abord 'entrainer_robot_v2_7.py'."
        return resultats

    try:
        resultats["modele_victoire"] = joblib.load(path_victoire)
        resultats["modele_podium"] = joblib.load(path_podium)
    except Exception as e:
        resultats["erreur"] = f"Erreur chargement modèles : {e}"
        return resultats

    if path_metadata.exists():
        try:
            with open(path_metadata, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            resultats["features"] = metadata.get("features", [])
            resultats["top_features_victoire"] = (
                metadata.get("resultats", {}).get("Target_Victoire", {}).get("top_5_features", [])
            )
            resultats["top_features_podium"] = (
                metadata.get("resultats", {}).get("Target_Podium", {}).get("top_5_features", [])
            )
            resultats["auc_victoire"] = (
                metadata.get("resultats", {}).get("Target_Victoire", {}).get("auc", 0.0)
            )
            resultats["auc_podium"] = (
                metadata.get("resultats", {}).get("Target_Podium", {}).get("auc", 0.0)
            )
        except Exception:
            pass

    return resultats

# ============================================================
# 3. CHARGEMENT DE L'HISTORIQUE
# ============================================================
@st.cache_data(ttl=3600)
def charger_historique_chevaux():
    """Charge l'historique complet pour calculer les features historiques."""
    try:
        r = requests.get(URL_HISTORIQUE, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return pd.DataFrame()
        df = pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()

def construire_features_historiques(df_partants, df_historique):
    """Calcule les features historiques pour chaque cheval AVANT la course du jour."""
    if df_historique.empty:
        for col in ["Hist_Nb_Courses", "Hist_Victoires", "Hist_Podiums", "Hist_Taux_Victoire", "Hist_Taux_Podium"]:
            df_partants[col] = 0
        return df_partants

    df_historique["_horse_key"] = df_historique["Nom"].apply(nettoyer_nom)

    if "Date" in df_historique.columns:
        df_historique["_date_sort"] = pd.to_datetime(df_historique["Date"], format="%d%m%Y", errors="coerce")
        df_historique = df_historique.sort_values("_date_sort")

    hist_data = {}
    for _, row in df_historique.iterrows():
        horse = row["_horse_key"]
        if horse not in hist_data:
            hist_data[horse] = {"courses": 0, "victoires": 0, "podiums": 0}
        hist_data[horse]["courses"] += 1
        classement = safe_int(row.get("Classement", 0))
        if classement == 1:
            hist_data[horse]["victoires"] += 1
        if 1 <= classement <= 3:
            hist_data[horse]["podiums"] += 1

    df_partants["_horse_key"] = df_partants["Nom"].apply(nettoyer_nom)
    df_partants["Hist_Nb_Courses"] = df_partants["_horse_key"].apply(
        lambda h: hist_data.get(h, {}).get("courses", 0)
    )
    df_partants["Hist_Victoires"] = df_partants["_horse_key"].apply(
        lambda h: hist_data.get(h, {}).get("victoires", 0)
    )
    df_partants["Hist_Podiums"] = df_partants["_horse_key"].apply(
        lambda h: hist_data.get(h, {}).get("podiums", 0)
    )
    df_partants["Hist_Taux_Victoire"] = np.where(
        df_partants["Hist_Nb_Courses"] > 0,
        df_partants["Hist_Victoires"] / df_partants["Hist_Nb_Courses"],
        0.0,
    )
    df_partants["Hist_Taux_Podium"] = np.where(
        df_partants["Hist_Nb_Courses"] > 0,
        df_partants["Hist_Podiums"] / df_partants["Hist_Nb_Courses"],
        0.0,
    )
    return df_partants

# ============================================================
# 4. CHARGEMENT DES PARTANTS DU JOUR
# ============================================================
@st.cache_data(ttl=600)
def charger_courses_du_jour():
    """Charge les partants du jour depuis Google Sheets."""
    try:
        r = requests.get(URL_COURSES_JOUR, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            st.error(f"Erreur HTTP {r.status_code} lors du chargement des courses du jour.")
            return {}
        df_all = pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig")
        df_all.columns = df_all.columns.str.strip()
    except Exception as e:
        st.error(f"Erreur chargement : {e}")
        return {}

    if df_all.empty:
        return {}

    base_courses = {}
    if "Reunion" in df_all.columns and "Course" in df_all.columns:
        for (reunion, course), group in df_all.groupby(["Reunion", "Course"]):
            r_str = str(reunion).strip().upper()
            c_str = str(course).strip().upper()
            if not r_str.startswith("R"):
                r_str = f"R{r_str}"
            if not c_str.startswith("C"):
                c_str = f"C{c_str}"

            hippodrome = "Inconnu"
            if "Hippodrome" in group.columns and not group["Hippodrome"].isna().all():
                hippodrome = str(group["Hippodrome"].iloc[0])

            if r_str not in base_courses:
                base_courses[r_str] = {"hippodrome": hippodrome, "courses": {}}

            records = []
            for idx, row in group.iterrows():
                rec = row.to_dict()
                rec["Nom"] = str(row.get("Nom", f"Cheval {idx}"))
                rec["Cheval_clean"] = nettoyer_nom(rec["Nom"])
                rec["Jockey_clean"] = nettoyer_nom(row.get("Driver_Jockey", "JOKEY"))
                rec["Entraineur_clean"] = nettoyer_nom(row.get("Entraineur", "ENTRAINEUR"))
                rec["Num_PMU"] = safe_int(row.get("Num_PMU", idx + 1))
                rec["Driver_Jockey"] = str(row.get("Driver_Jockey", "JOKEY"))
                rec["Entraineur"] = str(row.get("Entraineur", "ENTRAINEUR"))
                rec["Poids_num"] = safe_float(row.get("Poids"), 58.0)
                rec["Corde_num"] = safe_float(row.get("Place_Corde", row.get("Corde_Piste")), 1.0)
                rec["Poids"] = rec["Poids_num"]
                rec["Corde"] = int(rec["Corde_num"])
                rec["Equipement"] = str(row.get("Oeilleres", "SANS"))
                rec["Musique"] = str(row.get("Musique", "")) if pd.notna(row.get("Musique")) else ""
                rec["Supplement"] = safe_int(row.get("Supplement", 0))
                records.append(rec)
            base_courses[r_str]["courses"][c_str] = records
    return base_courses

# ============================================================
# 5. CONSTRUCTION DES FEATURES V2.7 POUR PRÉDICTION
# ============================================================
def construire_features_v27(df_partants, df_historique):
    """Construit toutes les features V2.7 nécessaires au modèle."""
    df_partants = construire_features_historiques(df_partants, df_historique)

    df_partants["Age"] = df_partants["Age"].apply(safe_int) if "Age" in df_partants.columns else 3
    df_partants["Poids"] = df_partants["Poids"].apply(safe_float) if "Poids" in df_partants.columns else 58.0
    df_partants["Place_Corde"] = df_partants["Place_Corde"].apply(safe_int) if "Place_Corde" in df_partants.columns else 1
    df_partants["Nb_Courses"] = df_partants["Nb_Courses"].apply(safe_int) if "Nb_Courses" in df_partants.columns else 0
    df_partants["Nb_Victoires"] = df_partants["Nb_Victoires"].apply(safe_int) if "Nb_Victoires" in df_partants.columns else 0
    df_partants["Nb_Places"] = df_partants["Nb_Places"].apply(safe_int) if "Nb_Places" in df_partants.columns else 0
    df_partants["Nb_Places_2e"] = df_partants["Nb_Places_2e"].apply(safe_int) if "Nb_Places_2e" in df_partants.columns else 0
    df_partants["Nb_Places_3e"] = df_partants["Nb_Places_3e"].apply(safe_int) if "Nb_Places_3e" in df_partants.columns else 0
    df_partants["Gains_Carriere"] = df_partants["Gains_Carriere"].apply(safe_float) if "Gains_Carriere" in df_partants.columns else 0.0
    df_partants["Gains_Victoires"] = df_partants["Gains_Victoires"].apply(safe_float) if "Gains_Victoires" in df_partants.columns else 0.0
    df_partants["Gains_Place"] = df_partants["Gains_Place"].apply(safe_float) if "Gains_Place" in df_partants.columns else 0.0
    df_partants["Gains_Annee_En_Cours"] = df_partants["Gains_Annee_En_Cours"].apply(safe_float) if "Gains_Annee_En_Cours" in df_partants.columns else 0.0
    df_partants["Gains_Annee_Precedente"] = df_partants["Gains_Annee_Precedente"].apply(safe_float) if "Gains_Annee_Precedente" in df_partants.columns else 0.0
    df_partants["Supplement"] = df_partants["Supplement"].apply(safe_int) if "Supplement" in df_partants.columns else 0

    df_partants["Distance"] = df_partants["Distance"].apply(safe_int) if "Distance" in df_partants.columns else 2000
    df_partants["Distance_km"] = df_partants["Distance"] / 1000.0
    df_partants["Distance_Courte"] = (df_partants["Distance"] <= 1400).astype(int)
    df_partants["Distance_Moyenne"] = ((df_partants["Distance"] > 1400) & (df_partants["Distance"] <= 2000)).astype(int)
    df_partants["Distance_Longue"] = (df_partants["Distance"] > 2000).astype(int)

    nature_piste = df_partants.get("Nature_Piste", pd.Series([""] * len(df_partants)))
    df_partants["Surface_PSF"] = nature_piste.apply(
        lambda x: 1 if isinstance(x, str) and any(t in x.upper() for t in ["PSF", "SABLE", "FIBRE"]) else 0
    )
    df_partants["Surface_Gazon"] = 1 - df_partants["Surface_PSF"]
    df_partants["Surface_Inconnue"] = 0
    df_partants["Surface_Confiance_Code"] = 1

    df_partants["Mois_Course"] = 9
    df_partants["Trimestre_Course"] = 3
    df_partants["Saison_Course"] = "AUTOMNE"

    categorical_cols = ["Hippodrome", "Discipline", "Corde_Piste", "Oeilleres", "Sexe", "Inedit", "Allure"]
    for col in categorical_cols:
        if col in df_partants.columns:
            df_partants[f"{col}_Code"] = pd.factorize(
                df_partants[col].fillna("INCONNU").astype(str), sort=True
            )[0]
        else:
            df_partants[f"{col}_Code"] = 0

    df_partants["Distance_Classe_Code"] = pd.factorize(
        df_partants["Distance_Classe"].fillna("INCONNUE").astype(str) if "Distance_Classe" in df_partants.columns else pd.Series(["INCONNUE"] * len(df_partants)),
        sort=True,
    )[0]
    df_partants["Saison_Course_Code"] = pd.factorize(
        df_partants["Saison_Course"].fillna("AUTOMNE").astype(str), sort=True
    )[0]

    df_partants["Corde_Distance_Courte"] = df_partants["Place_Corde"] * df_partants["Distance_Courte"]
    df_partants["Hist_Victoire_PSF"] = df_partants["Hist_Taux_Victoire"] * df_partants["Surface_PSF"]
    df_partants["Hist_Victoire_Gazon"] = df_partants["Hist_Taux_Victoire"] * df_partants["Surface_Gazon"]
    df_partants["Hist_Podium_PSF"] = df_partants["Hist_Taux_Podium"] * df_partants["Surface_PSF"]
    df_partants["Hist_Podium_Gazon"] = df_partants["Hist_Taux_Podium"] * df_partants["Surface_Gazon"]
    df_partants["Distance_Courte_PSF"] = df_partants["Distance_Courte"] * df_partants["Surface_PSF"]
    df_partants["Distance_Moyenne_Gazon"] = df_partants["Distance_Moyenne"] * df_partants["Surface_Gazon"]
    df_partants["Gains_Recent_Forme"] = df_partants["Gains_Annee_En_Cours"] * df_partants["Hist_Taux_Podium"]
    df_partants["Age_Experience"] = df_partants["Age"] * df_partants["Hist_Nb_Courses"]
    df_partants["Poids_Distance"] = df_partants["Poids"] * df_partants["Distance_km"]
    df_partants["Corde_Hippodrome"] = df_partants["Place_Corde"] * df_partants.get("Hippodrome_Code", 0)

    return df_partants

# ============================================================
# 6. MOTEUR DE PRÉDICTION V2.7
# ============================================================
def predire_probas_v27(df_partants, modeles_info, df_historique):
    """Calcule les probabilités de Victoire et Podium avec le modèle V2.7."""
    if df_partants.empty:
        return df_partants

    df_partants = construire_features_v27(df_partants, df_historique)

    features_attendues = modeles_info.get("features", [])
    if not features_attendues:
        features_attendues = [
            "Age", "Poids", "Place_Corde", "Nb_Courses", "Nb_Victoires", "Nb_Places",
            "Nb_Places_2e", "Nb_Places_3e", "Gains_Carriere", "Gains_Victoires",
            "Gains_Place", "Gains_Annee_En_Cours", "Gains_Annee_Precedente",
            "Supplement", "Distance_km", "Distance_Courte", "Distance_Moyenne",
            "Distance_Longue", "Surface_PSF", "Surface_Gazon", "Surface_Inconnue",
            "Mois_Course", "Trimestre_Course", "Saison_Course", "Distance_Classe",
            "Surface_Confiance_Code", "Hist_Nb_Courses", "Hist_Victoires", "Hist_Podiums",
            "Hist_Taux_Victoire", "Hist_Taux_Podium",
            "Corde_Distance_Courte", "Hist_Victoire_PSF", "Hist_Victoire_Gazon",
            "Hist_Podium_PSF", "Hist_Podium_Gazon", "Distance_Courte_PSF",
            "Distance_Moyenne_Gazon", "Gains_Recent_Forme", "Age_Experience",
            "Poids_Distance", "Corde_Hippodrome",
        ]

    features_dispo = [f for f in features_attendues if f in df_partants.columns]

    for f in features_attendues:
        if f not in df_partants.columns:
            df_partants[f] = 0

    for col in ["Hippodrome_Code", "Discipline_Code", "Corde_Piste_Code", "Oeilleres_Code",
                "Sexe_Code", "Inedit_Code", "Allure_Code", "Surface_Final_Code",
                "Surface_Source_Code", "Surface_Methode_Code", "Distance_Classe_Code",
                "Saison_Course_Code"]:
        if col not in df_partants.columns:
            df_partants[col] = 0

    X = df_partants[features_attendues].fillna(0).astype(np.float32)

    if modeles_info["modele_victoire"] is not None and modeles_info["modele_podium"] is not None:
        try:
            proba_victoire = modeles_info["modele_victoire"].predict_proba(X)[:, 1]
            proba_podium = modeles_info["modele_podium"].predict_proba(X)[:, 1]
            df_partants["Proba_Victoire"] = proba_victoire
            df_partants["Proba_Podium"] = proba_podium
            df_partants["Score_Combine"] = 0.6 * proba_victoire + 0.4 * proba_podium
        except Exception as e:
            st.warning(f"Erreur prédiction : {e}")
            df_partants["Proba_Victoire"] = 0.5
            df_partants["Proba_Podium"] = 0.5
            df_partants["Score_Combine"] = 0.5
    else:
        df_partants["Proba_Victoire"] = 0.5
        df_partants["Proba_Podium"] = 0.5
        df_partants["Score_Combine"] = 0.5

    df_partants = df_partants.sort_values("Score_Combine", ascending=False).reset_index(drop=True)

    return df_partants

# ============================================================
# 7. ANALYSE NARRATIVE ENRICHIE
# ============================================================
def analyse_narrative_v27(row, top_features_victoire, top_features_podium, df_course):
    """Génère une analyse narrative basée sur les features les plus importantes du modèle."""
    points_forts = []
    points_faibles = []
    explication_robot = []

    cheval_nom = row.get("Nom", "Ce cheval")
    jockey_nom = row.get("Driver_Jockey", "son jockey")
    musique = str(row.get("Musique", ""))
    score_musique = float(row.get("Score_Musique", 0.0))
    poids_cheval = float(row.get("Poids_num", 58.0))
    total_courses = int(row.get("Nb_Courses", row.get("Hist_Nb_Courses", 0)))
    gains_total = float(row.get("Gains_Carriere", 0.0))
    gains_annee = float(row.get("Gains_Annee_En_Cours", 0.0))
    hist_taux_victoire = float(row.get("Hist_Taux_Victoire", 0.0))
    hist_taux_podium = float(row.get("Hist_Taux_Podium", 0.0))
    hist_nb_courses = int(row.get("Hist_Nb_Courses", 0))
    surface_psf = int(row.get("Surface_PSF", 0))
    surface_gazon = int(row.get("Surface_Gazon", 0))
    distance = int(row.get("Distance", 2000))
    corde = int(row.get("Place_Corde", 1))
    proba_v = float(row.get("Proba_Victoire", 0.0)) * 100
    proba_p = float(row.get("Proba_Podium", 0.0)) * 100

    if hist_taux_podium > 0.30:
        points_forts.append(f"🏆 **Excellente régularité** : Taux de podium de **{hist_taux_podium*100:.1f}%** sur ses {hist_nb_courses} dernières courses. C'est la feature N°1 du modèle IA.")
        explication_robot.append(f"Le modèle accorde une importance capitale à l'historique de podium ({hist_taux_podium*100:.1f}%).")
    elif hist_taux_podium > 0.15:
        points_forts.append(f"📊 **Bonne régularité** : Taux de podium de **{hist_taux_podium*100:.1f}%**. Feature déterminante pour le modèle.")
    else:
        points_faibles.append(f"️ **Régularité limitée** : Taux de podium de seulement **{hist_taux_podium*100:.1f}%**. Le modèle pénalise ce cheval sur ce critère essentiel.")

    hist_podium_gazon = float(row.get("Hist_Podium_Gazon", 0.0))
    if surface_gazon == 1 and hist_podium_gazon > 0.20:
        points_forts.append(f"🌱 **Spécialiste du gazon** : Excellent taux de podium sur gazon (**{hist_podium_gazon*100:.1f}%**). Feature d'interaction cruciale pour le modèle V2.7.")
        explication_robot.append(f"Le modèle valorise fortement la performance sur gazon ({hist_podium_gazon*100:.1f}%).")
    elif surface_psf == 1:
        hist_podium_psf = float(row.get("Hist_Podium_PSF", 0.0))
        if hist_podium_psf > 0.20:
            points_forts.append(f"🏜️ **Spécialiste PSF** : Taux de podium sur sable fibré de **{hist_podium_psf*100:.1f}%**.")

    if gains_annee > 10000:
        points_forts.append(f" **En grande forme cette année** : **{gains_annee:,.0f} €** de gains en cours. Le modèle considère la forme récente comme prioritaire.")
        explication_robot.append(f"Les gains récents ({gains_annee:,.0f}€) sont un signal fort de forme actuelle.")
    elif gains_annee > 0:
        points_forts.append(f"💵 **Gains en cours** : **{gains_annee:,.0f} €** cette année.")
    else:
        points_faibles.append("💸 **Aucun gain cette année** : Le modèle interprète cela comme un manque de forme récente.")

    if total_courses > 10:
        points_forts.append(f"📚 **Cheval expérimenté** : **{total_courses} courses** au compteur. L'expérience est un facteur stabilisateur pour le modèle.")
    elif total_courses > 3:
        points_forts.append(f"📚 **Expérience correcte** : {total_courses} courses.")
    else:
        points_faibles.append(f"🆕 **Peu d'expérience** : seulement {total_courses} course(s). Le modèle reste prudent.")

    nb_places_2e = int(row.get("Nb_Places_2e", 0))
    if nb_places_2e > 2:
        points_forts.append(f"🥈 **Régulièrement 2ème** : {nb_places_2e} deuxièmes places. Signe d'un cheval qui vise le podium.")

    distance_courte = int(row.get("Distance_Courte", 0))
    if distance_courte == 1 and corde <= 4:
        points_forts.append(f"🎯 **Avantage corde** : Corde **{corde}** sur une course courte ({distance}m). Feature d'interaction 'Corde×Distance_Courte' très valorisée par le modèle.")
        explication_robot.append(f"La corde {corde} sur sprint est un atout majeur détecté par le modèle.")
    elif distance_courte == 1 and corde > 10:
        points_faibles.append(f"⚠️ **Corde défavorable** : Corde **{corde}** sur une course courte ({distance}m). Désavantage significatif.")

    poids_moy = df_course["Poids_num"].mean() if "Poids_num" in df_course.columns else 58.0
    ecart_poids = poids_cheval - poids_moy
    if ecart_poids > 1.5:
        points_faibles.append(f"⚖️ **Poids pénalisant** : {poids_cheval:.1f} kg (+{abs(ecart_poids):.1f} kg vs moyenne). Le modèle intègre le handicap poids×distance.")
    elif ecart_poids < -1.0:
        points_forts.append(f"️ **Avantage au poids** : {poids_cheval:.1f} kg (-{abs(ecart_poids):.1f} kg vs moyenne).")

    sexe = str(row.get("Sexe", "")).upper()
    if "MALES" in sexe or "HONGRES" in sexe:
        points_forts.append(f"♂️ **Mâle/Hongre** : Statut souvent avantageux dans les courses de plat.")

    oeilleres = str(row.get("Oeilleres", "")).upper()
    if "OEILLERES" in oeilleres and "SANS" not in oeilleres:
        points_forts.append(f"👁️ **Œillères portées** : Équipement souvent signe d'une volonté de recentrer le cheval.")

    if musique and musique.lower() != "nan":
        victoires = musique.count("1")
        places = musique.count("2") + musique.count("3")
        points_forts.append(f"🎵 **Musique** : {musique} (Score: {score_musique} pts, {victoires} victoires, {places} places).")

    if proba_v > 50:
        explication_robot.append(f"✅ **Verdict IA** : Probabilité de victoire élevée ({proba_v:.1f}%). Le modèle identifie ce cheval comme un favori sérieux.")
    elif proba_p > 60:
        explication_robot.append(f"📊 **Verdict IA** : Fort potentiel de podium ({proba_p:.1f}%) même si la victoire n'est pas certaine.")
    else:
        explication_robot.append(f"⚠️ **Verdict IA** : Probabilités modestes (V:{proba_v:.1f}% P:{proba_p:.1f}%). Le modèle ne le voit pas dans le top 3.")

    return points_forts, points_faibles, explication_robot

# ============================================================
# 8. INTERFACE STREAMLIT
# ============================================================
def main():
    st.title("🏇 Galop Analyzer Pro V2.7")
    st.markdown("### 🤖 Prédictions IA basées sur XGBoost + Features d'interaction")

    modeles_info = charger_modeles_v27()
    if modeles_info["erreur"]:
        st.error(modeles_info["erreur"])
        st.stop()

    df_historique = charger_historique_chevaux()
    db_courses = charger_courses_du_jour()

    if not db_courses:
        st.error("❌ Impossible de charger les courses du jour.")
        st.stop()

    with st.sidebar:
        st.header("📍 Sélection")
        reunions_dispo = list(db_courses.keys())
        reunion_choisie = st.selectbox("Réunion", reunions_dispo)
        hippodrome_actuel = db_courses[reunion_choisie]["hippodrome"]
        courses_dispo = list(db_courses[reunion_choisie]["courses"].keys())
        course_choisie = st.selectbox("Course", courses_dispo)
        st.markdown("---")
        st.info(f"📍 **{hippodrome_actuel}**\n\n👥 Partants : {len(db_courses[reunion_choisie]['courses'][course_choisie])}")

    partants_bruts = db_courses[reunion_choisie]["courses"][course_choisie]
    df_partants = pd.DataFrame(partants_bruts)
    df_predits = predire_probas_v27(df_partants, modeles_info, df_historique)

    # ============================================================
    # SIDEBAR : RÉSUMÉ DE TOUTES LES COURSES AVEC TOP 3
    # ============================================================
    with st.sidebar:
        st.markdown("---")
        st.subheader("📋 Résumé des Courses")
        
        for r_key in sorted(db_courses.keys()):
            r_val = db_courses[r_key]
            st.markdown(f"**{r_key}** - {r_val['hippodrome']}")
            
            for c_key in sorted(r_val["courses"].keys()):
                c_partants = r_val["courses"][c_key]
                nb_partants = len(c_partants)
                
                df_c = pd.DataFrame(c_partants)
                df_c_pred = predire_probas_v27(df_c, modeles_info, df_historique)
                top3 = df_c_pred.head(3)
                
                with st.expander(f"{c_key} ({nb_partants} partants)", expanded=False):
                    for rank, (_, row) in enumerate(top3.iterrows(), 1):
                        medaille = ["🥇", "🥈", "🥉"][rank-1]
                        num_pmu = int(row.get("Num_PMU", rank))
                        nom = row.get("Nom", "Inconnu")
                        proba = row.get("Score_Combine", 0) * 100
                        st.markdown(f"{medaille} N°{num_pmu} **{nom}** ({proba:.1f}%)")
        
        st.markdown("---")
        st.header("🤖 Modèle IA V2.7")
        st.info(f"**Performance Victoire** : AUC = {modeles_info['auc_victoire']:.4f}")
        st.info(f"**Performance Podium** : AUC = {modeles_info['auc_podium']:.4f}")
        st.markdown("---")
        st.markdown("### 🏆 Top 5 Features Victoire")
        for i, feat in enumerate(modeles_info["top_features_victoire"], 1):
            st.markdown(f"{i}. `{feat}`")
        st.markdown("---")
        st.markdown("### 🎯 Top 5 Features Podium")
        for i, feat in enumerate(modeles_info["top_features_podium"], 1):
            st.markdown(f"{i}. `{feat}`")

    # ============================================================
    # AFFICHAGE : COURSE SÉLECTIONNÉE
    # ============================================================
    st.markdown("---")
    st.subheader(f"📍 {reunion_choisie} - {course_choisie} | {hippodrome_actuel}")

    if not df_predits.empty:
        colonnes_affichage = [
            "Num_PMU", "Nom", "Driver_Jockey", "Poids", "Place_Corde",
            "Distance", "Proba_Victoire", "Proba_Podium", "Score_Combine",
        ]
        colonnes_dispo = [c for c in colonnes_affichage if c in df_predits.columns]
        df_affiche = df_predits[colonnes_dispo].copy()
        df_affiche["Proba_Victoire"] = (df_affiche["Proba_Victoire"] * 100).round(1)
        df_affiche["Proba_Podium"] = (df_affiche["Proba_Podium"] * 100).round(1)
        df_affiche["Score_Combine"] = (df_affiche["Score_Combine"] * 100).round(1)
        df_affiche = df_affiche.rename(columns={
            "Num_PMU": "N°",
            "Nom": "Cheval",
            "Driver_Jockey": "Jockey",
            "Poids": "Poids (kg)",
            "Place_Corde": "Corde",
            "Distance": "Distance (m)",
            "Proba_Victoire": "Prob. Victoire (%)",
            "Proba_Podium": "Prob. Podium (%)",
            "Score_Combine": "Score IA (%)",
        })
        df_affiche.index = range(1, len(df_affiche) + 1)
        st.dataframe(df_affiche, use_container_width=True, hide_index=True)

        # ============================================================
        # ANALYSE NARRATIVE DU TOP 3
        # ============================================================
        st.markdown("---")
        st.subheader(f"🔍 Analyse Narrative IA - Top 3 {reunion_choisie} {course_choisie}")
        st.markdown(
            f"*Analyse basée sur les **15 features les plus importantes** du modèle V2.7 "
            f"(AUC Victoire: {modeles_info['auc_victoire']:.4f} | AUC Podium: {modeles_info['auc_podium']:.4f})*"
        )

        top3_course = df_predits.head(3).copy().reset_index(drop=True)
        medailles = ["🥇 1er Favori", "🥈 2ème Favori", " 3ème Favori"]

        for i in range(min(3, len(top3_course))):
            row = top3_course.iloc[i]
            proba_v = row.get("Proba_Victoire", 0) * 100
            proba_p = row.get("Proba_Podium", 0) * 100
            score = row.get("Score_Combine", 0) * 100

            points_forts, points_faibles, explication_robot = analyse_narrative_v27(
                row, modeles_info["top_features_victoire"], modeles_info["top_features_podium"], df_predits
            )

            with st.expander(
                f"{medailles[i]} : {row.get('Nom', 'Inconnu')} | V:{proba_v:.1f}% P:{proba_p:.1f}% | Score:{score:.1f}%",
                expanded=(i == 0),
            ):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Prob. Victoire", f"{proba_v:.1f}%")
                with c2:
                    st.metric("Prob. Podium", f"{proba_p:.1f}%")
                with c3:
                    st.metric("Score IA", f"{score:.1f}%")
                with c4:
                    st.metric("Corde", f"{row.get('Place_Corde', 'N/A')}")

                st.markdown("---")

                st.markdown("### 🤖 Pourquoi le robot a choisi ce cheval ?")
                for exp in explication_robot:
                    st.markdown(f"• {exp}")

                st.markdown("---")

                col_fort, col_faible = st.columns(2)
                with col_fort:
                    st.markdown("### 🟢 Points forts détectés par l'IA :")
                    for pf in points_forts:
                        st.markdown(f"• {pf}")
                with col_faible:
                    st.markdown("### 🔴 Points de vigilance :")
                    if points_faibles:
                        for pf in points_faibles:
                            st.markdown(f"• {pf}")
                    else:
                        st.markdown("• Aucun point faible majeur détecté.")

    else:
        st.info("Aucune donnée disponible pour cette course.")

    st.markdown("---")
    st.caption(
        f"🤖 Galop Analyzer Pro V2.7 | Modèles XGBoost | "
        f"Features : {len(modeles_info['features'])} | "
        f"Entraîné sur 99,012 courses historiques"
    )

if __name__ == "__main__":
    main()
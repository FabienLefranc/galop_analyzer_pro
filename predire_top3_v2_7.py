# -*- coding: utf-8 -*-
"""
PREDICTION TOP 3 PAR COURSE - GALOP ANALYZER PRO V2.7
Télécharge les partants du jour depuis Google Sheets et prédit le Top 3.
"""
import io
import re
import json
import unicodedata
import requests
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(r"C:\Users\33662\OneDrive\Bureau\galop_analyzer_pro")
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "data" / "dataset"

# URL Google Sheets "Course_du_jour"
URL_COURSE_DU_JOUR = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6av"
    "citpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?"
    "gid=1852089216&single=true&output=csv"
)

# Modèles V2.7
MODEL_VICTOIRE = MODEL_DIR / "modele_galop_v27_victoire.joblib"
MODEL_PODIUM = MODEL_DIR / "modele_galop_v27_podium.joblib"
METADATA_PATH = MODEL_DIR / "metadata_entrainement_v2_7.json"
MAIN_DATASET = DATASET_DIR / "dataset_entrainement_v2_6.parquet"

# ============================================================
# OUTILS DE NORMALISATION (Identiques au générateur V2.6)
# ============================================================
def norm(x):
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKD", str(x).upper())
    x = x.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", x).strip()

def num(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False),
        errors="coerce",
    )

def saison(mois):
    if mois in (12, 1, 2): return "HIVER"
    if mois in (3, 4, 5): return "PRINTEMPS"
    if mois in (6, 7, 8): return "ETE"
    return "AUTOMNE"

def distance_classe(d):
    if pd.isna(d): return "INCONNUE"
    if d <= 1200: return "SPRINT"
    if d <= 1600: return "COURTE"
    if d <= 2000: return "MOYENNE"
    if d <= 2400: return "INTERMEDIAIRE"
    if d <= 3000: return "LONGUE"
    return "TRES_LONGUE"

# ============================================================
# REFERENTIEL DES SURFACES (Identique au générateur V2.6)
# ============================================================
GRASS = {
    "PARISLONGCHAMP", "LONGCHAMP", "SAINT CLOUD", "AUTEUIL", "COMPIEGNE", "DIEPPE",
    "CLAIRFONTAINE", "EVREUX", "EVREUX NAVARRE", "VICHY", "MOULINS", "CRAON", "CHOLET",
    "TOULOUSE", "TARBES", "MONT DE MARSAN", "DAX", "AUCH", "NANTES", "ANGERS",
    "ANGERS ECOUFLANT", "LE LION D ANGERS", "LE MANS", "BORDEAUX", "BORDEAUX LE BOUSCAT",
    "LA TESTE", "LA TESTE DE BUCH", "MARSEILLE BORELY", "STRASBOURG", "NANCY", "AMIENS",
    "LE CROISE LAROCHE", "SAINT MALO", "ARGENTAN", "LIGNEROLLES", "LAVAUR", "GRAIGNES",
    "MESLAY DU MAINE", "LYON PARILLY", "PARAY LE MONIAL", "SAINT GALMIER", "ROANNE",
    "MONTPELLIER", "NIMES", "PERPIGNAN", "LE TOUQUET", "VITTEL",
}
PSF = {
    "LYON LA SOIE", "MARSEILLE VIVAUX", "PORNICHET", "CHATEAUBRIANT",
    "FONTAINEBLEAU", "CABOURG", "SALON DE PROVENCE", "AGEN",
}

def surface(r):
    h = norm(r.get("Hippodrome", ""))
    d = r.get("_dist", np.nan)
    dt = r.get("_date", pd.NaT)
    src = norm(r.get("Nature_Piste", ""))
    discipline = norm(r.get("Discipline", ""))

    if "PSF" in src or "SABLE" in src or "FIBRE" in src: return "PSF", "HAUTE", "SOURCE"
    if "GAZON" in src or "HERBE" in src or "TURF" in src: return "GAZON", "HAUTE", "SOURCE"

    if pd.isna(d):
        if h in PSF: return "PSF", "HAUTE", "HIPPODROME"
        if h in GRASS: return "GAZON", "HAUTE", "HIPPODROME"
        return "INCONNUE", "FAIBLE", "DISTANCE_INCONNUE"

    d = int(round(float(d)))
    mois = int(dt.month) if pd.notna(dt) else None
    annee = int(dt.year) if pd.notna(dt) else None

    if "DEAUVILLE" in h:
        if annee == 2026 and mois == 10: return "PSF", "HAUTE", "DEAUVILLE_OCTOBRE_2026"
        if d in {1000, 1200}: return "GAZON", "HAUTE", "DEAUVILLE_LIGNE_DROITE"
        if d in {1300, 1400, 1500, 1600, 1900, 2000, 2500, 3200, 3400}:
            if mois in {11, 12, 1, 2, 3, 4}: return "PSF", "HAUTE", "DEAUVILLE_PSF_HIVER"
            else: return "GAZON", "MOYENNE", "DEAUVILLE_GAZON_ETE"
        return "GAZON", "MOYENNE", "DEAUVILLE_DEFAUT"

    if "CHANTILLY" in h:
        if d in {1000, 1100, 1200}: return "GAZON", "HAUTE", "CHANTILLY_LIGNE_DROITE"
        if d in {1300, 1400, 1500, 1600, 1800, 1900, 2100, 2400, 2700}:
            if mois in {11, 12, 1, 2, 3, 4}: return "PSF", "HAUTE", "CHANTILLY_PSF_HIVER"
            else: return "GAZON", "HAUTE", "CHANTILLY_GAZON"
        return "GAZON", "MOYENNE", "CHANTILLY_DEFAUT"

    if "CAGNES" in h:
        if mois in {1, 2, 3}:
            if d in {1000, 1200, 1300}: return "GAZON", "HAUTE", "CAGNES_LIGNE_DROITE"
            if d in {1400, 1500, 1600, 1900, 2000, 2150, 2400, 2500}: return "PSF", "HAUTE", "CAGNES_PSF_HIVER"
        return "GAZON", "MOYENNE", "CAGNES_DEFAUT"

    if "PAU" in h:
        if "OBSTACLE" in discipline or "HAIES" in discipline or "STEEPLE" in discipline: return "GAZON", "HAUTE", "PAU_OBSTACLE"
        if mois in {1, 2, 3} and d in {1500, 1600, 2000, 2200, 2300, 2400, 2500}: return "PSF", "HAUTE", "PAU_PLAT_PSF"
        return "GAZON", "MOYENNE", "PAU_DEFAUT"

    if h in PSF: return "PSF", "HAUTE", "HIPPODROME_PSF"
    if h in GRASS: return "GAZON", "HAUTE", "HIPPODROME_GAZON"
    return "GAZON", "FAIBLE", "DEFAUT_PROVINCE"

# ============================================================
# MOTEUR DE PREDICTION
# ============================================================
def predire_top3():
    print("=" * 70)
    print(" GALOP ANALYZER PRO V2.7 - PREDICTIONS TOP 3 DU JOUR")
    print("=" * 70)

    # 1. Chargement des modèles et métadonnées
    print("\n⏳ Chargement des modèles V2.7...")
    if not MODEL_VICTOIRE.exists() or not MODEL_PODIUM.exists():
        print("❌ Erreur : Modèles V2.7 introuvables. Lancez d'abord 'entrainer_robot_v2_7.py'")
        return
    
    model_victoire = joblib.load(MODEL_VICTOIRE)
    model_podium = joblib.load(MODEL_PODIUM)
    
    # Récupération de la liste exacte des features utilisées à l'entraînement
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        features = meta["features"]
        print(f"   ✅ {len(features)} features chargées depuis les métadonnées.")
    else:
        print("⚠️ Métadonnées introuvables, utilisation de la liste par défaut.")
        features = [
            "Age", "Poids", "Place_Corde", "Nb_Courses", "Nb_Victoires", "Nb_Places",
            "Nb_Places_2e", "Nb_Places_3e", "Gains_Carriere", "Gains_Victoires",
            "Gains_Place", "Gains_Annee_En_Cours", "Gains_Annee_Precedente",
            "Supplement", "Distance_km", "Distance_Courte", "Distance_Moyenne",
            "Distance_Longue", "Surface_PSF", "Surface_Gazon", "Surface_Inconnue",
            "Mois_Course", "Trimestre_Course", "Saison_Course", "Distance_Classe",
            "Surface_Confiance_Code", "Hist_Nb_Courses", "Hist_Victoires", "Hist_Podiums",
            "Hist_Taux_Victoire", "Hist_Taux_Podium", "Corde_Distance_Courte",
            "Hist_Victoire_PSF", "Hist_Victoire_Gazon", "Hist_Podium_PSF", "Hist_Podium_Gazon",
            "Distance_Courte_PSF", "Distance_Moyenne_Gazon", "Gains_Recent_Forme",
            "Age_Experience", "Poids_Distance", "Corde_Hippodrome", "Hippodrome_Code",
            "Discipline_Code", "Corde_Piste_Code", "Oeilleres_Code", "Sexe_Code",
            "Inedit_Code", "Allure_Code", "Surface_Final_Code", "Surface_Source_Code",
            "Surface_Methode_Code", "Distance_Classe_Code", "Saison_Course_Code"
        ]

    # 2. Chargement du dataset principal (pour l'historique des chevaux)
    print("\n📥 Chargement de l'historique des chevaux...")
    if MAIN_DATASET.exists():
        df_hist = pd.read_parquet(MAIN_DATASET)
        # On recrée la clé cheval normalisée
        df_hist["_horse_key"] = df_hist["Nom"].fillna("").astype(str).apply(norm)
        print(f"   ✅ {len(df_hist):,} courses historiques chargées.")
    else:
        print("⚠️ Dataset historique introuvable. Les features 'Hist_' seront à 0.")
        df_hist = pd.DataFrame()

    # 3. Téléchargement des partants du jour
    print(f"\n🌐 Téléchargement des partants depuis Google Sheets...")
    try:
        response = requests.get(URL_COURSE_DU_JOUR, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        df_jour = pd.read_csv(io.BytesIO(response.content), sep=None, engine="python", encoding="utf-8-sig")
        df_jour.columns = [str(c).strip() for c in df_jour.columns]
        print(f"   ✅ {len(df_jour)} partants du jour récupérés.")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        return

    # 4. Nettoyage et Features de base
    print("\n Construction des features V2.7...")
    df_jour["_date"] = pd.to_datetime(df_jour.get("Date", df_jour.get("Date_dt")), errors="coerce", dayfirst=True)
    df_jour["_dist"] = num(df_jour.get("Distance", pd.Series([np.nan]*len(df_jour))))
    df_jour["_horse_key"] = df_jour["Nom"].fillna("").astype(str).apply(norm)
    
    # Calcul de l'historique sans fuite (basé sur le dataset principal)
    if not df_hist.empty:
        hist_data = []
        for key in df_jour["_horse_key"]:
            past = df_hist[df_hist["_horse_key"] == key]
            nb = len(past)
            vic = past["Target_Victoire"].sum() if "Target_Victoire" in past.columns else 0
            pod = past["Target_Podium"].sum() if "Target_Podium" in past.columns else 0
            hist_data.append({
                "Hist_Nb_Courses": nb,
                "Hist_Victoires": vic,
                "Hist_Podiums": pod,
                "Hist_Taux_Victoire": vic / nb if nb > 0 else 0.0,
                "Hist_Taux_Podium": pod / nb if nb > 0 else 0.0,
            })
        df_hist_features = pd.DataFrame(hist_data, index=df_jour.index)
        df_jour = pd.concat([df_jour, df_hist_features], axis=1)
    else:
        for col in ["Hist_Nb_Courses", "Hist_Victoires", "Hist_Podiums", "Hist_Taux_Victoire", "Hist_Taux_Podium"]:
            df_jour[col] = 0

    # Surface
    z = df_jour.apply(surface, axis=1, result_type="expand")
    z.columns = ["Surface_Inferée", "Surface_Confiance", "Surface_Methode"]
    df_jour = pd.concat([df_jour, z], axis=1)
    df_jour["Surface_Final"] = df_jour["Surface_Inferée"]
    df_jour["Surface_Confiance_Code"] = df_jour["Surface_Confiance"].map({"FAIBLE": 0, "MOYENNE": 1, "HAUTE": 2}).fillna(0).astype("int8")

    # Variables dérivées
    df_jour["Distance_km"] = df_jour["_dist"] / 1000.0
    df_jour["Distance_Courte"] = (df_jour["_dist"] <= 1400).astype("int8")
    df_jour["Distance_Moyenne"] = ((df_jour["_dist"] > 1400) & (df_jour["_dist"] <= 2000)).astype("int8")
    df_jour["Distance_Longue"] = (df_jour["_dist"] > 2000).astype("int8")
    df_jour["Surface_PSF"] = (df_jour["Surface_Final"] == "PSF").astype("int8")
    df_jour["Surface_Gazon"] = (df_jour["Surface_Final"] == "GAZON").astype("int8")
    df_jour["Surface_Inconnue"] = (df_jour["Surface_Final"] == "INCONNUE").astype("int8")
    df_jour["Mois_Course"] = df_jour["_date"].dt.month.fillna(9).astype("int8")
    df_jour["Trimestre_Course"] = df_jour["_date"].dt.quarter.fillna(3).astype("int8")
    df_jour["Saison_Course"] = df_jour["Mois_Course"].map(saison)
    df_jour["Distance_Classe"] = df_jour["_dist"].map(distance_classe)

    # Encodage catégoriel
    cat_cols = ["Hippodrome", "Discipline", "Corde_Piste", "Oeilleres", "Sexe", "Inedit", "Allure", 
                "Surface_Final", "Surface_Source", "Surface_Methode", "Distance_Classe", "Saison_Course"]
    for c in cat_cols:
        if c in df_jour.columns:
            df_jour[f"{c}_Code"] = pd.factorize(df_jour[c].fillna("INCONNU").astype(str), sort=True)[0].astype("int16")

    # Features d'interaction V2.7
    df_jour["Corde_Distance_Courte"] = df_jour["Place_Corde"].fillna(1) * df_jour["Distance_Courte"]
    df_jour["Hist_Victoire_PSF"] = df_jour["Hist_Taux_Victoire"] * df_jour["Surface_PSF"]
    df_jour["Hist_Victoire_Gazon"] = df_jour["Hist_Taux_Victoire"] * df_jour["Surface_Gazon"]
    df_jour["Hist_Podium_PSF"] = df_jour["Hist_Taux_Podium"] * df_jour["Surface_PSF"]
    df_jour["Hist_Podium_Gazon"] = df_jour["Hist_Taux_Podium"] * df_jour["Surface_Gazon"]
    df_jour["Distance_Courte_PSF"] = df_jour["Distance_Courte"] * df_jour["Surface_PSF"]
    df_jour["Distance_Moyenne_Gazon"] = df_jour["Distance_Moyenne"] * df_jour["Surface_Gazon"]
    df_jour["Gains_Recent_Forme"] = df_jour.get("Gains_Annee_En_Cours", 0) * df_jour["Hist_Taux_Podium"]
    df_jour["Age_Experience"] = df_jour.get("Age", 3) * df_jour["Hist_Nb_Courses"]
    df_jour["Poids_Distance"] = df_jour.get("Poids", 58) * df_jour["Distance_km"]
    df_jour["Corde_Hippodrome"] = df_jour["Place_Corde"].fillna(1) * df_jour.get("Hippodrome_Code", 0)

    # Alignement avec les features du modèle
    X = df_jour.reindex(columns=features, fill_value=0).fillna(0).astype(np.float32)

    # 5. Prédictions
    print("\n Calcul des probabilités IA...")
    proba_v = model_victoire.predict_proba(X)[:, 1]
    proba_p = model_podium.predict_proba(X)[:, 1]
    
    df_jour["Proba_Victoire"] = proba_v
    df_jour["Proba_Podium"] = proba_p
    # Score combiné : on privilégie la victoire mais on intègre le podium
    df_jour["Score_IA"] = (0.65 * proba_v) + (0.35 * proba_p)

    # 6. Affichage Top 3 par course
    print("\n" + "=" * 70)
    print("🏆 TOP 3 DES FAVORIS IA PAR COURSE")
    print("=" * 70)

    if "Reunion" in df_jour.columns and "Course" in df_jour.columns:
        # Tri par Réunion, Course, puis Score IA décroissant
        df_jour = df_jour.sort_values(by=["Reunion", "Course", "Score_IA"], ascending=[True, True, False])
        grouped = df_jour.groupby(["Reunion", "Course"])
        
        for (reunion, course), group in grouped:
            top3 = group.head(3)
            hippo = top3["Hippodrome"].iloc[0] if "Hippodrome" in top3.columns else ""
            dist = top3["Distance"].iloc[0] if "Distance" in top3.columns else "?"
            
            print(f"\n📍 {reunion} - {course} | {hippo} ({dist}m)")
            print("   " + "-" * 68)
            for rank, (_, row) in enumerate(top3.iterrows(), 1):
                nom = row.get("Nom", "Inconnu")[:22]
                jockey = row.get("Driver_Jockey", row.get("Jockey", "N/A"))[:15]
                corde = row.get("Place_Corde", "?")
                v_pct = row["Proba_Victoire"] * 100
                p_pct = row["Proba_Podium"] * 100
                
                # Indicateur visuel pour le 1er
                medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
                print(f"   {medal} {nom:<22} | Jockey: {jockey:<15} | Corde: {corde:<2} | V:{v_pct:5.1f}% P:{p_pct:5.1f}%")
            print("   " + "-" * 68)
    else:
        print("⚠️ Colonnes 'Reunion' ou 'Course' manquantes. Affichage du Top 10 global.")
        top10 = df_jour.nlargest(10, "Score_IA")
        for _, row in top10.iterrows():
            print(f"🐎 {row.get('Nom', 'Inconnu'):<25} | 🎯 Score: {row['Score_IA']*100:.1f}%")

    # Sauvegarde CSV
    output_csv = BASE_DIR / "predictions_du_jour_v2_7.csv"
    df_jour.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 Détails complets sauvegardés dans : {output_csv}")
    print("=" * 70)

if __name__ == "__main__":
    predire_top3()
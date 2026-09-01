# -*- coding: utf-8 -*-
r"""
GENERATION DU DATASET IA V2.6
Source : Google Sheets publie
Dates JJMMAAAA sur 7 ou 8 chiffres
Pas de fuite temporelle
Pas de fuite entre chevaux d'une meme course
Cotes / classements exclus des features
Surfaces : hippodrome + distance + periode, avec niveau de confiance
Sorties : CSV + Parquet dans galop_analyzer_pro\data\dataset
"""
import io
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import requests

# ============================================================
# CONFIGURATION
# ============================================================
URL_CSV = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6av"
    "citpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?"
    "gid=644246763&single=true&output=csv"
)
OUT = Path(r"C:\Users\33662\OneDrive\Bureau\galop_analyzer_pro\data\dataset")
CSV = OUT / "dataset_entrainement_v2_6.csv"
PARQUET = OUT / "dataset_entrainement_v2_6.parquet"

# ============================================================
# OUTILS
# ============================================================
def norm(x):
    """Normalise une chaine : MAJUSCULES, sans accents, espaces uniques."""
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKD", str(x).upper())
    x = x.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", x).strip()

def dates(series):
    """
    Parse les dates au format JJMMAAAA (7 ou 8 chiffres).
    Accepte aussi les chaines avec caracteres non numeriques.
    """
    s = series.astype(str).str.replace(r"\D", "", regex=True)
    s = s.map(lambda x: x.zfill(8) if len(x) == 7 else x)
    s = s.where(s.str.len().eq(8))
    return pd.to_datetime(s, format="%d%m%Y", errors="coerce")

def num(series):
    """Convertit une serie en numerique, en gerant virgules et espaces."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False),
        errors="coerce",
    )

def saison(mois):
    """Retourne la saison a partir du mois."""
    if mois in (12, 1, 2):
        return "HIVER"
    if mois in (3, 4, 5):
        return "PRINTEMPS"
    if mois in (6, 7, 8):
        return "ETE"
    return "AUTOMNE"

def distance_classe(d):
    """Classe la distance en categories."""
    if pd.isna(d):
        return "INCONNUE"
    if d <= 1200:
        return "SPRINT"
    if d <= 1600:
        return "COURTE"
    if d <= 2000:
        return "MOYENNE"
    if d <= 2400:
        return "INTERMEDIAIRE"
    if d <= 3000:
        return "LONGUE"
    return "TRES_LONGUE"

# ============================================================
# REFERENTIEL DES HIPPODROMES
# ============================================================
# Hippodromes 100% GAZON (Plat et Obstacle)
GRASS = {
    "PARISLONGCHAMP", "LONGCHAMP", "SAINT CLOUD", "AUTEUIL",
    "COMPIEGNE", "DIEPPE", "CLAIRFONTAINE", "EVREUX", "EVREUX NAVARRE",
    "VICHY", "MOULINS", "CRAON", "CHOLET", "TOULOUSE", "TARBES",
    "MONT DE MARSAN", "DAX", "AUCH", "NANTES", "ANGERS", "ANGERS ECOUFLANT",
    "LE LION D ANGERS", "LE MANS", "BORDEAUX", "BORDEAUX LE BOUSCAT",
    "LA TESTE", "LA TESTE DE BUCH", "MARSEILLE BORELY", "STRASBOURG",
    "NANCY", "AMIENS", "LE CROISE LAROCHE", "SAINT MALO", "ARGENTAN",
    "LIGNEROLLES", "LAVAUR", "GRAIGNES", "MESLAY DU MAINE", "LYON PARILLY",
    "PARAY LE MONIAL", "SAINT GALMIER", "ROANNE", "MONTPELLIER", "NIMES",
    "PERPIGNAN", "LE TOUQUET", "VITTEL",
}

# Hippodromes 100% PSF (Pour le Plat)
PSF = {
    "LYON LA SOIE", "MARSEILLE VIVAUX", "PORNICHET", "CHATEAUBRIANT",
    "FONTAINEBLEAU", "CABOURG", "SALON DE PROVENCE", "AGEN",
}

def surface(r):
    """
    Determine la surface de course avec niveau de confiance.
    
    Retourne : (Surface_Inferée, Surface_Confiance, Surface_Methode)
    
    Priorite :
       1) Nature_Piste fournie explicitement
       2) Regles specifiques hippodromes mixtes (distance + saison)
       3) Hippodrome 100% PSF ou 100% Gazon
       4) Defaut province (Gazon confiance faible)
       5) Inconnue
    """
    h = norm(r.get("Hippodrome", ""))
    d = r.get("_dist", np.nan)
    dt = r.get("_date", pd.NaT)
    src = norm(r.get("Nature_Piste", ""))
    discipline = norm(r.get("Discipline", ""))

    # 1. Source explicite (priorite absolue)
    if "PSF" in src or "SABLE" in src or "FIBRE" in src:
        return "PSF", "HAUTE", "SOURCE"
    if "GAZON" in src or "HERBE" in src or "TURF" in src:
        return "GAZON", "HAUTE", "SOURCE"

    if pd.isna(d):
        if h in PSF:
            return "PSF", "HAUTE", "HIPPODROME"
        if h in GRASS:
            return "GAZON", "HAUTE", "HIPPODROME"
        return "INCONNUE", "FAIBLE", "DISTANCE_INCONNUE"

    d = int(round(float(d)))
    mois = int(dt.month) if pd.notna(dt) else None
    annee = int(dt.year) if pd.notna(dt) else None

    # 2. REGLES POUR HIPPODROMES MIXTES (critique)
    
    # --- DEAUVILLE ---
    if "DEAUVILLE" in h:
        # Octobre 2026 : uniquement PSF
        if annee == 2026 and mois == 10:
            return "PSF", "HAUTE", "DEAUVILLE_OCTOBRE_2026"
        # Ligne droite : toujours gazon
        if d in {1000, 1200}:
            return "GAZON", "HAUTE", "DEAUVILLE_LIGNE_DROITE"
        # Distances PSF : 1300 a 3400m
        if d in {1300, 1400, 1500, 1600, 1900, 2000, 2500, 3200, 3400}:
            # Meeting d'hiver (novembre a avril) = PSF
            if mois in {11, 12, 1, 2, 3, 4}:
                return "PSF", "HAUTE", "DEAUVILLE_PSF_HIVER"
            # Meeting d'ete (juillet a septembre) = Gazon (grande piste)
            else:
                return "GAZON", "MOYENNE", "DEAUVILLE_GAZON_ETE"
        # Autres distances : defaut gazon
        return "GAZON", "MOYENNE", "DEAUVILLE_DEFAUT"

    # --- CHANTILLY ---
    if "CHANTILLY" in h:
        # Ligne droite : toujours gazon
        if d in {1000, 1100, 1200}:
            return "GAZON", "HAUTE", "CHANTILLY_LIGNE_DROITE"
        # Distances PSF : 1300 a 2700m
        if d in {1300, 1400, 1500, 1600, 1800, 1900, 2100, 2400, 2700}:
            # Hiver (novembre a avril) = PSF
            if mois in {11, 12, 1, 2, 3, 4}:
                return "PSF", "HAUTE", "CHANTILLY_PSF_HIVER"
            # Reste de l'annee = Gazon
            else:
                return "GAZON", "HAUTE", "CHANTILLY_GAZON"
        # Autres distances : defaut gazon
        return "GAZON", "MOYENNE", "CHANTILLY_DEFAUT"

    # --- CAGNES-SUR-MER ---
    if "CAGNES" in h:
        # Meeting d'hiver (janvier a mars)
        if mois in {1, 2, 3}:
            # Ligne droite : gazon
            if d in {1000, 1200, 1300}:
                return "GAZON", "HAUTE", "CAGNES_LIGNE_DROITE"
            # Distances PSF
            if d in {1400, 1500, 1600, 1900, 2000, 2150, 2400, 2500}:
                return "PSF", "HAUTE", "CAGNES_PSF_HIVER"
        # Hors meeting d'hiver : defaut gazon
        return "GAZON", "MOYENNE", "CAGNES_DEFAUT"

    # --- PAU ---
    if "PAU" in h:
        # Obstacle : toujours gazon
        if "OBSTACLE" in discipline or "HAIES" in discipline or "STEEPLE" in discipline:
            return "GAZON", "HAUTE", "PAU_OBSTACLE"
        # Plat en hiver (janvier a mars) : PSF pour certaines distances
        if mois in {1, 2, 3} and d in {1500, 1600, 2000, 2200, 2300, 2400, 2500}:
            return "PSF", "HAUTE", "PAU_PLAT_PSF"
        # Defaut : gazon
        return "GAZON", "MOYENNE", "PAU_DEFAUT"

    # 3. HIPPODROMES 100% PSF
    if h in PSF:
        return "PSF", "HAUTE", "HIPPODROME_PSF"

    # 4. HIPPODROMES 100% GAZON
    if h in GRASS:
        return "GAZON", "HAUTE", "HIPPODROME_GAZON"

    # 5. FILET DE SECURITE : petits hippodromes non listes
    # En France, le plat en province est majoritairement sur gazon.
    # Confiance FAIBLE pour ne pas polluer l'IA avec des certitudes erronees.
    return "GAZON", "FAIBLE", "DEFAUT_PROVINCE"

# ============================================================
# HISTORIQUE SANS FUITE
# ============================================================
def construire_historiques(df, rank):
    """
    Calcule l'historique cheval AVANT chaque course.
    
    Point critique V2.6 :
    - On ne fait pas un simple shift ligne par ligne.
    - Les resultats de tous les chevaux d'une course sont integres
      seulement APRES avoir calcule les features de tous les partants.
    - Cela evite une fuite intra-course.
    """
    horse = (
        df["Nom"]
        .fillna("")
        .astype(str)
        .map(norm)
        .replace("", "INCONNU")
    )
    df["_horse_key"] = horse
    hist_courses = {}
    hist_wins = {}
    hist_podiums = {}
    out_courses = np.zeros(len(df), dtype=np.int32)
    out_wins = np.zeros(len(df), dtype=np.int32)
    out_podiums = np.zeros(len(df), dtype=np.int32)

    # Courses dans l'ordre chronologique
    race_ids = df["_race"].drop_duplicates().tolist()
    for race_id in race_ids:
        idx = df.index[df["_race"] == race_id].tolist()

        # FEATURES AVANT LA COURSE
        for i in idx:
            h = df.at[i, "_horse_key"]
            out_courses[df.index.get_loc(i)] = hist_courses.get(h, 0)
            out_wins[df.index.get_loc(i)] = hist_wins.get(h, 0)
            out_podiums[df.index.get_loc(i)] = hist_podiums.get(h, 0)

        # RESULTATS DE LA COURSE : mise a jour APRES tous les partants
        for i in idx:
            h = df.at[i, "_horse_key"]
            hist_courses[h] = hist_courses.get(h, 0) + 1
            r = rank.loc[i]
            if pd.notna(r) and r > 0:
                if r == 1:
                    hist_wins[h] = hist_wins.get(h, 0) + 1
                if r <= 3:
                    hist_podiums[h] = hist_podiums.get(h, 0) + 1

    df["Hist_Nb_Courses"] = out_courses
    df["Hist_Victoires"] = out_wins
    df["Hist_Podiums"] = out_podiums
    df["Hist_Taux_Victoire"] = np.where(
        df["Hist_Nb_Courses"] > 0,
        df["Hist_Victoires"] / df["Hist_Nb_Courses"],
        0.0,
    )
    df["Hist_Taux_Podium"] = np.where(
        df["Hist_Nb_Courses"] > 0,
        df["Hist_Podiums"] / df["Hist_Nb_Courses"],
        0.0,
    )
    return df

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🏇 GENERATION DU DATASET IA V2.6 ")
    print("🔒 MODE SANS FUITE TEMPORELLE ")
    print("🔒 MODE SANS FUITE A L'INTERIEUR D'UNE COURSE ")
    print("🚫 COTES EXCLUES DES FEATURES ")
    print("🚫 CLASSEMENTS VIDES = INCONNUS ")
    print("🌱 SURFACES : HIPPODROME + DISTANCE + PERIODE ")
    print("=" * 70)
    OUT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # CHARGEMENT
    # --------------------------------------------------------
    print("\n📥 Chargement du Google Sheets...")
    r = requests.get(
        URL_CSV,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
        allow_redirects=True,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Google Sheets a répondu HTTP {r.status_code}. "
            f"URL utilisée : {URL_CSV}"
        )

    # Verification : Google doit renvoyer un CSV, pas une page HTML
    sample = r.content[:500].lstrip().lower()
    if b"<html" in sample or b"<!doctype" in sample:
        raise RuntimeError(
            "Google Sheets a renvoyé une page HTML au lieu du CSV. "
            f"URL utilisée : {URL_CSV}"
        )

    df = pd.read_csv(
        io.BytesIO(r.content),
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )

    df.columns = [str(c).strip() for c in df.columns]
    print(f"📊 {len(df):,} lignes chargées.")

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------
    df["_date"] = dates(df["Date"])
    bad = int(df["_date"].isna().sum())
    df = df[df["_date"].notna()].copy()
    print(f"⚠️ {bad:,} lignes ont une date invalide et seront ignorées.")
    print(
        f"📅 Première date : {df['_date'].min():%Y-%m-%d}\n"
        f"📅 Dernière date : {df['_date'].max():%Y-%m-%d}"
    )

    # Tri chronologique explicite
    df["_ordre_source"] = np.arange(len(df), dtype=np.int64)
    df["_dist"] = num(df["Distance"])
    df = df.sort_values(
        ["_date", "_ordre_source"],
        kind="mergesort",
    ).reset_index(drop=True)
    df["_race"] = (
        df[["Date", "Reunion", "Course", "Hippodrome"]]
        .astype(str)
        .agg("|".join, axis=1)
    )
    print(f"🏇 {df['_race'].nunique():,} courses détectées.")

    # --------------------------------------------------------
    # VARIABLES DATE / DISTANCE
    # --------------------------------------------------------
    df["Mois_Course"] = df["_date"].dt.month.astype("int8")
    df["Trimestre_Course"] = df["_date"].dt.quarter.astype("int8")
    df["Saison_Course"] = df["Mois_Course"].map(saison)
    df["Distance_Classe"] = df["_dist"].map(distance_classe)

    # --------------------------------------------------------
    # SURFACE
    # --------------------------------------------------------
    print("\n🌱 Enrichissement des surfaces...")
    z = df.apply(surface, axis=1, result_type="expand")
    z.columns = [
        "Surface_Inferée",
        "Surface_Confiance",
        "Surface_Methode",
    ]
    df = pd.concat([df, z], axis=1)

    src = df["Nature_Piste"].fillna("").map(norm)
    df["Surface_Source"] = np.select(
        [
            src.str.contains("PSF|SABLE", regex=True),
            src.str.contains("GAZON|HERBE", regex=True),
        ],
        [
            "PSF",
            "GAZON",
        ],
        default="INCONNUE",
    )
    df["Surface_Final"] = np.where(
        df["Surface_Source"].isin(["PSF", "GAZON"]),
        df["Surface_Source"],
        df["Surface_Inferée"],
    )

    print("\n   Répartition Surface_Final :")
    print(df["Surface_Final"].value_counts(dropna=False).to_string())
    print("\n   Répartition Surface_Methode :")
    print(df["Surface_Methode"].value_counts(dropna=False).to_string())

    # Indice de fiabilite numerique
    df["Surface_Confiance_Code"] = (
        df["Surface_Confiance"]
        .map({"FAIBLE": 0, "MOYENNE": 1, "HAUTE": 2})
        .fillna(0)
        .astype("int8")
    )

    # --------------------------------------------------------
    # CIBLES
    # --------------------------------------------------------
    rank = num(df["Classement"])
    df["Classement"] = rank
    df["Target_Victoire"] = (
        (rank == 1) & rank.notna()
    ).astype("int8")
    df["Target_Podium"] = (
        (rank <= 3) & rank.notna()
    ).astype("int8")

    # --------------------------------------------------------
    # HISTORIQUES
    # --------------------------------------------------------
    print("\n🧠 Calcul des historiques sans fuite...")
    df = construire_historiques(df, rank)

    # --------------------------------------------------------
    # VARIABLES DERIVEES
    # --------------------------------------------------------
    df["Distance_km"] = df["_dist"] / 1000.0
    df["Distance_Courte"] = (df["_dist"] <= 1400).astype("int8")
    df["Distance_Moyenne"] = (
        (df["_dist"] > 1400) & (df["_dist"] <= 2000)
    ).astype("int8")
    df["Distance_Longue"] = (df["_dist"] > 2000).astype("int8")
    df["Surface_PSF"] = (
        df["Surface_Final"] == "PSF"
    ).astype("int8")
    df["Surface_Gazon"] = (
        df["Surface_Final"] == "GAZON"
    ).astype("int8")
    df["Surface_Inconnue"] = (
        df["Surface_Final"] == "INCONNUE"
    ).astype("int8")

    numeric_cols = [
        "Age", "Poids", "Place_Corde", "Nb_Courses", "Nb_Victoires",
        "Nb_Places", "Nb_Places_2e", "Nb_Places_3e",
        "Gains_Carriere", "Gains_Victoires", "Gains_Place",
        "Gains_Annee_En_Cours", "Gains_Annee_Precedente", "Supplement",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = num(df[c])

    categorical_cols = [
        "Hippodrome", "Discipline", "Corde_Piste", "Oeilleres", "Sexe",
        "Inedit", "Allure", "Surface_Final", "Surface_Source",
        "Surface_Confiance", "Surface_Methode", "Distance_Classe",
        "Saison_Course",
    ]
    for c in categorical_cols:
        if c in df.columns:
            df[c + "_Code"] = pd.factorize(
                df[c].fillna("INCONNU").astype(str),
                sort=True,
            )[0].astype("int16")

    # --------------------------------------------------------
    # VARIABLES INTERDITES
    # --------------------------------------------------------
    forbidden = {
        "Classement", "Cote_Direct", "Target_Victoire", "Target_Podium",
        "Nom", "Proprietaire", "Entraineur", "Driver_Jockey", "Musique",
        "_date", "_dist", "_race", "_ordre_source", "_horse_key",
    }
    features = [
        c for c in df.columns
        if c not in forbidden
    ]

    # --------------------------------------------------------
    # CONTROLES
    # --------------------------------------------------------
    valid_rank = (
        df["Classement"].notna()
        & (df["Classement"] > 0)
    )
    print("\n" + "=" * 70)
    print("🔎 CONTROLES FINAUX")
    print(f"   Observations        : {len(df):,}")
    print(f"   Colonnes            : {len(df.columns):,}")
    print(f"   Features autorisées : {len(features):,}")
    print(f"   Classements valides : {valid_rank.sum():,}")
    print(f"   Classements inconnus: {(~valid_rank).sum():,}")
    print(f"   Victoires           : {df['Target_Victoire'].sum():,}")
    print(f"   Podiums             : {df['Target_Podium'].sum():,}")

    print("\n🚫 VARIABLES INTERDITES AU MODELE")
    for c in sorted(
        {
            "Classement", "Cote_Direct", "Target_Podium", "Target_Victoire",
            "Nom", "Proprietaire", "Entraineur", "Driver_Jockey", "Musique",
        }
    ):
        print(f"   - {c}")

    # --------------------------------------------------------
    # ECRITURE
    # --------------------------------------------------------
    print("\n💾 Ecriture des fichiers...")
    # Date lisible pour audit
    df["Date_dt"] = df["_date"].dt.strftime("%Y-%m-%d")

    # Nettoyage des colonnes techniques
    df = df.drop(
        columns=[
            "_date", "_dist", "_race", "_ordre_source", "_horse_key",
        ],
        errors="ignore",
    )

    # Mettre Date_dt juste apres Date
    if "Date" in df.columns:
        cols = df.columns.tolist()
        cols.remove("Date_dt")
        pos = cols.index("Date") + 1
        cols.insert(pos, "Date_dt")
        df = df[cols]

    df.to_csv(
        CSV,
        index=False,
        encoding="utf-8-sig",
    )

    parquet_ok = False
    try:
        df.to_parquet(
            PARQUET,
            index=False,
        )
        parquet_ok = True
    except Exception as e:
        print(f"⚠️ Parquet non écrit : {e}")

    print("\n" + "=" * 70)
    print("✅ DATASET IA V2.6 TERMINE")
    print(f"📊 {len(df):,} observations")
    print(f"📊 {len(df.columns):,} colonnes")
    print(f"💾 CSV    : {CSV}")
    if parquet_ok:
        print(f"💾 Parquet: {PARQUET}")
    print("=" * 70)


if __name__ == "__main__":
    main()
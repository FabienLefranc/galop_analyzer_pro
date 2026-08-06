import pandas as pd
import os

masters_dir = "data/masters"
fichiers = ['master_chevaux.parquet', 'master_jockeys.parquet', 'master_entraineurs.parquet', 'master_couplages.parquet']

for f in fichiers:
    chemin = os.path.join(masters_dir, f)
    if os.path.exists(chemin):
        df = pd.read_parquet(chemin)
        print(f"✅ {f} : {len(df)} lignes, {len(df.columns)} colonnes")
        print(f"   Colonnes : {list(df.columns[:5])} ...\n")
    else:
        print(f"❌ {f} est introuvable dans {masters_dir}\n")
"""
================================================================================
SCORING RISQUE PAYS — MODÈLE ÉLARGI (15 INDICATEURS, 6 PILIERS)
================================================================================
Objectif : construire un score composite de risque souverain "non-orthodoxe"
intégrant des dimensions sociales, climatiques, énergétiques et sanitaires
souvent absentes des modèles standards des agences de notation.

ARCHITECTURE DU MODÈLE — 6 PILIERS, 15 INDICATEURS :

  Pilier 1 — ÉCONOMIQUE (poids 20%)
    · Croissance du PIB %          [API Banque Mondiale, NY.GDP.MKTP.KD.ZG]
    · Taux de chômage %            [API Banque Mondiale, SL.UEM.TOTL.ZS]

  Pilier 2 — FISCAL (poids 15%)
    · Solde budgétaire % PIB       [API Banque Mondiale, GC.NLD.TOTL.GD.ZS]
    · Réserves de change (mois)    [API Banque Mondiale, FI.RES.TOTL.MO]

  Pilier 3 — EXTERNE / MONÉTAIRE (poids 15%)
    · Compte courant % PIB         [API Banque Mondiale, BN.CAB.XOKA.GD.ZS]
    · Inflation %                  [API Banque Mondiale, FP.CPI.TOTL.ZG]

  Pilier 4 — SOCIAL & INÉGALITÉS (poids 20%)  ← non-orthodoxe
    · Coefficient de Gini          [API Banque Mondiale, SI.POV.GINI]
    · Taux de pauvreté <$2.15/j %  [API Banque Mondiale, SI.POV.DDAY]
    · Espérance de vie (années)    [API Banque Mondiale, SP.DYN.LE00.IN]

  Pilier 5 — CLIMAT & ÉNERGIE (poids 15%)     ← non-orthodoxe
    · Vulnérabilité climatique     [ND-GAIN Index — données embarquées 2022]
    · Import. énergie % conso.     [API Banque Mondiale, EG.IMP.CONS.ZS]
    · Accès à l'électricité %      [API Banque Mondiale, EG.ELC.ACCS.ZS]

  Pilier 6 — SANTÉ & CAPITAL HUMAIN (poids 15%) ← non-orthodoxe
    · Dépenses de santé % PIB      [API Banque Mondiale, SH.XPD.CHEX.GD.ZS]
    · Médecins / 1000 hab.         [API Banque Mondiale, SH.MED.PHYS.ZS]
    · INFORM Risk Score            [INFORM — données embarquées 2023]

SOURCES DE DONNÉES :
  · API Banque Mondiale (gratuite, sans clé) — 13 indicateurs
  · ND-GAIN Country Index (Univ. Notre-Dame) — vulnérabilité climatique
    https://gain.nd.edu/our-work/country-index/
  · INFORM Risk Index (UE/OCHA) — risque humanitaire multi-crises
    https://drmkc.jrc.ec.europa.eu/inform-index
  Les deux derniers sont des indices publics annuels encodés directement
  dans le script (pas d'API REST disponible) — mise à jour manuelle conseillée.

ORIGINALITÉ DU MODÈLE :
  Les piliers Social, Climatique et Santé représentent 50% du score composite.
  Un pays à finances publiques saines mais à forte exposition climatique et
  inégalités élevées sera pénalisé — contrairement aux modèles standards.

⚠️  Les données de repli sont ILLUSTRATIVES. Ne pas utiliser pour décision réelle.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

USE_LIVE_API = True  # passer à False pour forcer le jeu illustratif

# ==============================================================================
# 1. PANEL & INDICATEURS
# ==============================================================================

COUNTRIES = {
    "USA": "Etats-Unis",   "DEU": "Allemagne",    "FRA": "France",
    "BRA": "Bresil",       "TUR": "Turquie",       "ZAF": "Afrique du Sud",
    "EGY": "Egypte",       "IND": "Inde",          "IDN": "Indonesie",
    "ARG": "Argentine",    "NGA": "Nigeria",        "VNM": "Vietnam",
}

# Indicateurs récupérés via l'API Banque Mondiale
WB_INDICATORS = {
    # Pilier Économique
    "PIB_croissance":     "NY.GDP.MKTP.KD.ZG",
    "Chomage":            "SL.UEM.TOTL.ZS",
    # Pilier Fiscal
    # NB: GC.BAL.CASH.GD.ZS est quasi abandonne par la BM (plus de donnees
    # recentes pour la plupart des pays) -> remplace par l'indicateur actif
    # equivalent GC.NLD.TOTL.GD.ZS (solde budgetaire global, % PIB).
    "Solde_budgetaire":   "GC.NLD.TOTL.GD.ZS",
    # NB: "FI.RES.TOTM.MO" etait une coquille -> le vrai code est TOTL (pas TOTM).
    "Reserves_mois":      "FI.RES.TOTL.MO",
    # Pilier Externe / Monétaire
    "CompteCourant_PIB":  "BN.CAB.XOKA.GD.ZS",
    "Inflation":          "FP.CPI.TOTL.ZG",
    # Pilier Social
    "Gini":               "SI.POV.GINI",
    "Pauvrete_extreme":   "SI.POV.DDAY",
    "Esperance_vie":      "SP.DYN.LE00.IN",
    # Pilier Climat & Énergie
    "Import_energie":     "EG.IMP.CONS.ZS",
    "Acces_electricite":  "EG.ELC.ACCS.ZS",
    # Pilier Santé
    "Depenses_sante":     "SH.XPD.CHEX.GD.ZS",
    "Medecins":           "SH.MED.PHYS.ZS",
}

# ------------------------------------------------------------------------------
# Données embarquées — indices publics sans API REST
# ND-GAIN Vulnerability Score 0-1 (1 = très vulnérable) — année 2022
# https://gain.nd.edu/our-work/country-index/rankings/
ND_GAIN_VULNERABILITY = {
    "USA": 0.392, "DEU": 0.265, "FRA": 0.298, "BRA": 0.453,
    "TUR": 0.421, "ZAF": 0.511, "EGY": 0.538, "IND": 0.498,
    "IDN": 0.487, "ARG": 0.432, "NGA": 0.612, "VNM": 0.468,
}

# INFORM Risk Index 2023 (score 0-10, 10 = risque humanitaire max)
# https://drmkc.jrc.ec.europa.eu/inform-index/INFORM-Risk
INFORM_RISK = {
    "USA": 2.4,  "DEU": 1.7,  "FRA": 2.1,  "BRA": 4.8,
    "TUR": 4.2,  "ZAF": 5.3,  "EGY": 4.9,  "IND": 5.1,
    "IDN": 5.6,  "ARG": 4.1,  "NGA": 7.2,  "VNM": 4.3,
}

# ==============================================================================
# 2. RÉCUPÉRATION DES DONNÉES
# ==============================================================================

def fetch_indicator_live(iso3, code, date_range="2015:2023"):
    """Interroge l'API Banque Mondiale — remonte à 2015 pour maximiser
    la couverture des indicateurs sociaux/santé (souvent publiés avec retard)."""
    import requests
    url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/{code}"
    r = requests.get(url, params={"format": "json", "date": date_range,
                                   "per_page": 100}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if len(payload) < 2 or not payload[1]:
        return np.nan
    obs = [p for p in payload[1] if p["value"] is not None]
    if not obs:
        return np.nan
    obs.sort(key=lambda p: p["date"], reverse=True)
    return obs[0]["value"]


def fetch_live_dataset():
    """Construit le DataFrame complet via API BM + données embarquées."""
    rows = []
    total = len(COUNTRIES) * len(WB_INDICATORS)
    done = 0
    for iso3, name in COUNTRIES.items():
        row = {"Pays": name, "iso3": iso3}
        for label, code in WB_INDICATORS.items():
            row[label] = fetch_indicator_live(iso3, code)
            done += 1
            print(f"  [{done}/{total}] {name} — {label}          ", end="\r")
        row["ND_GAIN_vulnerabilite"] = ND_GAIN_VULNERABILITY.get(iso3, np.nan)
        row["INFORM_risk"]           = INFORM_RISK.get(iso3, np.nan)
        rows.append(row)
    print()
    df = pd.DataFrame(rows)
    if df.drop(columns=["Pays", "iso3"]).isna().all(axis=None):
        raise RuntimeError("Aucune donnée renvoyée par l'API")
    return df


def fallback_dataset():
    """Jeu illustratif si l'API est inaccessible — NE PAS utiliser pour analyse réelle."""
    data = [
      # Pays              iso3  PIB  Chô   Bud   Rés   CC    Inf   Gin  Pvr   EV    ImpE  ElecA  DS    Med   GAIN  INF
      ("Etats-Unis",     "USA",  2.5,  3.7, -5.5,  3.0, -3.0,  3.0, 41.0,  1.0, 78.9,  -7.0, 100.0, 17.3, 2.61, 0.392, 2.4),
      ("Allemagne",      "DEU",  0.2,  3.0, -2.5,  2.5,  6.0,  2.5, 31.7,  0.1, 81.3,  61.0, 100.0, 12.8, 4.28, 0.265, 1.7),
      ("France",         "FRA",  1.0,  7.3, -5.0,  3.0, -1.0,  2.5, 32.4,  0.1, 82.3,  47.0, 100.0, 12.1, 3.08, 0.298, 2.1),
      ("Bresil",         "BRA",  2.5, 13.0, -7.0,  8.0, -1.5,  4.5, 53.4, 11.0, 75.9,  10.0,  99.8,  9.9, 2.31, 0.453, 4.8),
      ("Turquie",        "TUR",  3.0, 10.5, -3.0,  5.0, -2.5, 45.0, 41.9,  1.5, 77.7,  28.0, 100.0,  4.6, 1.97, 0.421, 4.2),
      ("Afrique du Sud", "ZAF",  0.8, 31.9, -4.5,  6.0, -2.0,  5.0, 63.0, 23.0, 64.1,  22.0,  84.2,  8.1, 0.79, 0.511, 5.3),
      ("Egypte",         "EGY",  3.5,  7.3, -6.0,  4.0, -2.0, 30.0, 31.5, 22.0, 71.8,  90.0,  99.8,  5.4, 0.73, 0.538, 4.9),
      ("Inde",           "IND",  7.0,  8.0, -5.0,  9.0, -1.0,  5.0, 35.7, 22.5, 70.2,  42.0,  97.6,  3.5, 0.74, 0.498, 5.1),
      ("Indonesie",      "IDN",  5.0,  5.5, -2.5,  5.0, -0.5,  3.0, 38.2,  6.0, 71.4,  30.0,  98.5,  3.1, 0.47, 0.487, 5.6),
      ("Argentine",      "ARG", -1.5,  6.9,-10.0,  3.0,  0.5,150.0, 42.7,  4.0, 76.5,  10.0,  99.9, 10.1, 3.99, 0.432, 4.1),
      ("Nigeria",        "NGA",  3.0, 33.3, -4.0,  4.0,  0.5, 28.0, 35.1, 53.5, 55.2,  15.0,  57.5,  3.4, 0.38, 0.612, 7.2),
      ("Vietnam",        "VNM",  6.0,  2.3, -1.5,  3.5,  4.0,  3.5, 35.7,  5.0, 75.4,  35.0,  99.7,  5.6, 0.83, 0.468, 4.3),
    ]
    cols = ["Pays", "iso3", "PIB_croissance", "Chomage", "Solde_budgetaire",
            "Reserves_mois", "CompteCourant_PIB", "Inflation", "Gini",
            "Pauvrete_extreme", "Esperance_vie", "Import_energie",
            "Acces_electricite", "Depenses_sante", "Medecins",
            "ND_GAIN_vulnerabilite", "INFORM_risk"]
    return pd.DataFrame(data, columns=cols)


# --- Chargement ---
data_is_live = False
if USE_LIVE_API:
    try:
        print("Récupération des données (API Banque Mondiale)...")
        df = fetch_live_dataset()
        data_is_live = True
        print("Données récupérées en direct depuis l'API Banque Mondiale.")
    except Exception as e:
        print(f"API inaccessible ({e}). Repli sur le jeu de données illustratif.")
        df = fallback_dataset()
else:
    df = fallback_dataset()

if not data_is_live:
    print("\n*** ATTENTION : données ILLUSTRATIVES — ne pas utiliser pour analyse réelle ***\n")

# Injection des données embarquées (toujours depuis les dicts internes)
if data_is_live:
    df["ND_GAIN_vulnerabilite"] = df["iso3"].map(ND_GAIN_VULNERABILITY)
    df["INFORM_risk"]           = df["iso3"].map(INFORM_RISK)

INDICATEURS_NUM = [
    "PIB_croissance", "Chomage", "Solde_budgetaire", "Reserves_mois",
    "CompteCourant_PIB", "Inflation", "Gini", "Pauvrete_extreme",
    "Esperance_vie", "Import_energie", "Acces_electricite",
    "Depenses_sante", "Medecins", "ND_GAIN_vulnerabilite", "INFORM_risk",
]

nb_manquants = df[INDICATEURS_NUM].isna().sum().sum()
if nb_manquants > 0:
    manquants_detail = df[INDICATEURS_NUM].isna().sum()
    manquants_detail = manquants_detail[manquants_detail > 0]
    print(f"\n{nb_manquants} valeur(s) manquante(s) detectee(s) :")
    for col, n in manquants_detail.items():
        print(f"   - {col} : {n} pays impute(s) par la moyenne du panel")
    for col in INDICATEURS_NUM:
        moyenne_col = df[col].mean()
        if pd.isna(moyenne_col):
            # Colonne entierement vide (ex: indicateur API en panne/retire) :
            # impossible de calculer une moyenne -> on neutralise la colonne
            # (z-score = 0, ni bonus ni malus) plutot que de laisser du NaN
            # se propager jusqu'au score composite et au classement.
            print(f"   ! ATTENTION : '{col}' est vide sur TOUT le panel "
                  f"(indicateur API indisponible) -> colonne neutralisee (= 0), "
                  f"a corriger/verifier le code indicateur correspondant.")
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(moyenne_col)

print("\n" + "=" * 90)
print("DONNEES BRUTES DU PANEL")
print("=" * 90)
print(df[["Pays"] + INDICATEURS_NUM].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"\nSource : {'API Banque Mondiale (live) + ND-GAIN/INFORM (embarques)' if data_is_live else 'jeu illustratif local'}"
      f" — genere le {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ==============================================================================
# 3. STANDARDISATION ET CONSTRUCTION DES PILIERS
# ==============================================================================

def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)

# Convention : z-score élevé = situation favorable (moins risquée).

# Pilier 1 : Économique
df["z_economique"] = (zscore(df["PIB_croissance"]) + zscore(-df["Chomage"])) / 2

# Pilier 2 : Fiscal
df["z_fiscal"] = (zscore(df["Solde_budgetaire"]) + zscore(df["Reserves_mois"])) / 2

# Pilier 3 : Externe / Monétaire
df["z_externe"] = (zscore(df["CompteCourant_PIB"]) + zscore(-df["Inflation"])) / 2

# Pilier 4 : Social & Inégalités — Gini élevé = inégalités → inversé
df["z_social"] = (zscore(-df["Gini"])
                + zscore(-df["Pauvrete_extreme"])
                + zscore(df["Esperance_vie"])) / 3

# Pilier 5 : Climat & Énergie
df["z_climat"] = (zscore(-df["ND_GAIN_vulnerabilite"])
                + zscore(-df["Import_energie"])
                + zscore(df["Acces_electricite"])) / 3

# Pilier 6 : Santé & Capital humain
df["z_sante"] = (zscore(-df["INFORM_risk"])
               + zscore(df["Depenses_sante"])
               + zscore(df["Medecins"])) / 3

PILLAR_COLS = ["z_economique", "z_fiscal", "z_externe", "z_social", "z_climat", "z_sante"]
PILLAR_LABELS = {
    "z_economique": "Economique (croissance + emploi)",
    "z_fiscal":     "Fiscal (solde + reserves)",
    "z_externe":    "Externe / Monetaire",
    "z_social":     "Social & Inegalites",
    "z_climat":     "Climat & Energie",
    "z_sante":      "Sante & Cap. humain",
}
PILLAR_SHORT = {
    "z_economique": "Eco",
    "z_fiscal":     "Fiscal",
    "z_externe":    "Externe",
    "z_social":     "Social",
    "z_climat":     "Climat",
    "z_sante":      "Sante",
}

# Pondérations : social + climat + santé = 50% du score
WEIGHTS = {
    "z_economique": 0.20,
    "z_fiscal":     0.15,
    "z_externe":    0.15,
    "z_social":     0.20,
    "z_climat":     0.15,
    "z_sante":      0.15,
}

df["score_composite"] = sum(df[col] * w for col, w in WEIGHTS.items())
if df["score_composite"].isna().any():
    pays_pb = df.loc[df["score_composite"].isna(), "Pays"].tolist()
    raise RuntimeError(
        "score_composite contient des NaN pour : " + ", ".join(pays_pb) +
        " -> impossible d'etablir un classement. Verifiez les codes "
        "indicateurs WB_INDICATORS et la disponibilite des donnees."
    )
df["rang"] = df["score_composite"].rank(ascending=False, method="min").astype(int)
df = df.sort_values("score_composite", ascending=False).reset_index(drop=True)


def categorie_risque(score):
    if score >= 0.75:    return "Risque tres faible"
    elif score >= 0.25:  return "Risque faible"
    elif score >= -0.25: return "Risque modere"
    elif score >= -0.75: return "Risque eleve"
    else:                return "Risque tres eleve"


df["categorie_risque"] = df["score_composite"].apply(categorie_risque)

print("\n" + "=" * 90)
print("CLASSEMENT — MODELE ELARGI (15 INDICATEURS, 6 PILIERS)")
print("=" * 90)
cols_affich = ["rang", "Pays", "score_composite", "categorie_risque"] + PILLAR_COLS
print(df[cols_affich].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

df.to_csv("country_risk_scores.csv", index=False)
print("\n-> Resultats exportes dans country_risk_scores.csv")

# ==============================================================================
# 4. GÉNÉRATION AUTOMATIQUE DE NOTES D'ANALYSE
# ==============================================================================

FORCE_TEMPLATES = {
    "z_economique": "une dynamique economique solide ({pib:.1f}% de croissance, chomage a {cho:.1f}%)",
    "z_fiscal":     "des fondamentaux fiscaux sains (solde {bud:+.1f}% du PIB, reserves {res:.1f} mois)",
    "z_externe":    "une position exterieure favorable (CC {cc:+.1f}% du PIB, inflation {inf:.1f}%)",
    "z_social":     "une cohesion sociale relativement preservee (Gini {gin:.0f}, esp. vie {ev:.1f} ans)",
    "z_climat":     "une exposition climatique maitrisee (vulnerabilite ND-GAIN {gain:.2f}/1)",
    "z_sante":      "un systeme de sante solide ({med:.1f} medecins/1000 hab., sante {ds:.1f}% PIB)",
}
FRAGILITE_TEMPLATES = {
    "z_economique": "une dynamique economique fragile ({pib:.1f}% de croissance, chomage a {cho:.1f}%)",
    "z_fiscal":     "des fragilites fiscales (solde {bud:+.1f}% du PIB, reserves {res:.1f} mois)",
    "z_externe":    "des tensions externes significatives (CC {cc:+.1f}% du PIB, inflation {inf:.1f}%)",
    "z_social":     "des inegalites structurelles elevees (Gini {gin:.0f}, pauvrete extreme {pvr:.1f}%)",
    "z_climat":     "une forte exposition climatique (vulnerabilite {gain:.2f}/1, INFORM {inf_r:.1f}/10)",
    "z_sante":      "un systeme de sante sous-dimensionne ({med:.1f} medecins/1000 hab., INFORM {inf_r:.1f}/10)",
}


def _fmt(pilier, row, template_dict):
    kwargs = dict(
        pib=row["PIB_croissance"], cho=row["Chomage"],
        bud=row["Solde_budgetaire"], res=row["Reserves_mois"],
        cc=row["CompteCourant_PIB"], inf=row["Inflation"],
        gin=row["Gini"], pvr=row["Pauvrete_extreme"], ev=row["Esperance_vie"],
        gain=row["ND_GAIN_vulnerabilite"], imp=row["Import_energie"],
        elec=row["Acces_electricite"],
        ds=row["Depenses_sante"], med=row["Medecins"], inf_r=row["INFORM_risk"],
    )
    return template_dict[pilier].format(**kwargs)


def generer_note(row):
    piliers_tries = row[PILLAR_COLS].sort_values(ascending=False)
    pilier_force   = piliers_tries.index[0]
    pilier_fragile = piliers_tries.index[-1]

    note = (
        f"## {row['Pays']} - {row['categorie_risque']}\n\n"
        f"**Score composite : {row['score_composite']:+.2f}** "
        f"(rang {row['rang']}/{len(df)} sur le panel)\n\n"
        f"**Point fort :** {_fmt(pilier_force, row, FORCE_TEMPLATES)}\n\n"
        f"**Point de vigilance :** {_fmt(pilier_fragile, row, FRAGILITE_TEMPLATES)}\n\n"
    )

    if row["score_composite"] >= 0.25:
        note += ("Profil global favorable : atouts macro et structurels limitant la vulnerabilite systemique.")
    elif row["score_composite"] <= -0.25:
        note += ("Profil degrade sur plusieurs dimensions. Convergence de risques macro, sociaux "
                 "et/ou climatiques justifiant un suivi renforce.")
    else:
        note += ("Profil intermediaire : atouts sur certains piliers compenses par des fragilites sur d'autres.")

    note += "\n\n| Pilier | Score |\n|---|---|\n"
    for col in PILLAR_COLS:
        note += f"| {PILLAR_LABELS[col]} | {row[col]:+.2f} |\n"

    note += ("\n*Sources : API Banque Mondiale (live) + ND-GAIN 2022 + INFORM 2023. "
             "Modele illustratif - pas une notation souveraine officielle.*\n")
    return note


with open("notes_pays.md", "w", encoding="utf-8") as f:
    f.write("# Notes d'analyse risque pays - Modele elargi (15 indicateurs)\n\n")
    f.write(f"*{len(df)} pays analyses — "
            f"{'Donnees live BM + ND-GAIN/INFORM embarques' if data_is_live else 'Donnees illustratives'}*\n\n---\n\n")
    for _, row in df.iterrows():
        f.write(generer_note(row))
        f.write("\n---\n\n")

print("-> Notes d'analyse generees dans notes_pays.md")

# ==============================================================================
# 5. VISUALISATIONS
# ==============================================================================

RADAR_LABELS = [PILLAR_SHORT[c] for c in PILLAR_COLS]

# --- Graphique 1 : Classement composite ---
fig, ax = plt.subplots(figsize=(11, 7))
norm_vals = (df["score_composite"] - df["score_composite"].min()) / \
            (df["score_composite"].max() - df["score_composite"].min())
colors = plt.cm.RdYlGn(norm_vals)
bars = ax.barh(df["Pays"], df["score_composite"], color=colors, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.9)
for bar, (_, row) in zip(bars, df.iterrows()):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            row["categorie_risque"], va="center", fontsize=7.5, color="#444")
ax.set_xlabel("Score composite (z-score pondere, 6 piliers)")
ax.set_title("Classement risque pays - Modele elargi 15 indicateurs\n"
             "(social, climatique, sanitaire = 50% du score)", fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("classement_risque_pays.png", dpi=150)
plt.close()
print("-> classement_risque_pays.png")

# --- Graphique 2 : Radars par pays ---
n = len(df)
ncols = 4
nrows = int(np.ceil(n / ncols))
angles = np.linspace(0, 2 * np.pi, len(PILLAR_COLS), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                          subplot_kw=dict(polar=True))
axes = axes.flatten()
for i, (_, row) in enumerate(df.iterrows()):
    ax = axes[i]
    values = row[PILLAR_COLS].tolist() + [row[PILLAR_COLS[0]]]
    score_norm = np.clip((row["score_composite"] + 1.5) / 3, 0, 1)
    color = plt.cm.RdYlGn(score_norm)
    ax.plot(angles, values, color=color, linewidth=1.8)
    ax.fill(angles, values, color=color, alpha=0.28)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f"{row['Pays']}\n({row['score_composite']:+.2f})", fontsize=9, pad=10)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Profil par pilier - 6 dimensions de risque (z-scores)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("radar_par_pays.png", dpi=150)
plt.close()
print("-> radar_par_pays.png")

# --- Graphique 3 : Carte Social vs Climat ---
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(df["z_social"], df["z_climat"], s=350,
                      c=df["score_composite"], cmap="RdYlGn",
                      edgecolors="black", linewidths=0.7, zorder=3)
for _, row in df.iterrows():
    ax.annotate(row["Pays"], (row["z_social"], row["z_climat"]),
                textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=8.5)
ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
ax.text( 1.8,  1.8, "Cohesion forte\nClimat maitrise", fontsize=7.5,
         color="#2d6a2d", ha="center", alpha=0.7)
ax.text(-1.8,  1.8, "Inegalites elevees\nClimat maitrise", fontsize=7.5,
         color="#8a6d00", ha="center", alpha=0.7)
ax.text( 1.8, -1.8, "Cohesion forte\nExposition climatique", fontsize=7.5,
         color="#8a6d00", ha="center", alpha=0.7)
ax.text(-1.8, -1.8, "Double vulnerabilite\nsociale + climatique", fontsize=7.5,
         color="#8b1a1a", ha="center", alpha=0.7)
ax.set_xlabel("Pilier Social & Inegalites (z-score, + = favorable)")
ax.set_ylabel("Pilier Climat & Energie (z-score, + = favorable)")
ax.set_title("Carte de positionnement - Vulnerabilite sociale vs climatique\n"
             "(dimensions absentes des modeles standard)", fontsize=11)
plt.colorbar(scatter, ax=ax, label="Score composite global")
plt.tight_layout()
plt.savefig("carte_social_climat.png", dpi=150)
plt.close()
print("-> carte_social_climat.png")

# --- Graphique 4 : Heatmap 6 piliers ---
fig, ax = plt.subplots(figsize=(10, 8))
heat_data = df.set_index("Pays")[PILLAR_COLS].copy()
heat_data.columns = [PILLAR_LABELS[c] for c in PILLAR_COLS]
im = ax.imshow(heat_data.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
ax.set_xticks(range(len(heat_data.columns)))
ax.set_xticklabels(heat_data.columns, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(heat_data.index)))
ax.set_yticklabels(heat_data.index, fontsize=9)
for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        val = heat_data.values[i, j]
        ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8,
                color="black" if abs(val) < 1.2 else "white")
ax.set_title("Heatmap des 6 piliers de risque (z-scores)\nvert = favorable / rouge = risque", fontsize=11)
plt.colorbar(im, ax=ax, label="z-score (+ = favorable, - = risque)")
plt.tight_layout()
plt.savefig("heatmap_piliers.png", dpi=150)
plt.close()
print("-> heatmap_piliers.png")

# --- Graphique 5 : Décomposition stacked bar ---
fig, ax = plt.subplots(figsize=(12, 7))
pilier_colors = {
    "z_economique": "#2196F3",
    "z_fiscal":     "#9C27B0",
    "z_externe":    "#FF9800",
    "z_social":     "#F44336",
    "z_climat":     "#4CAF50",
    "z_sante":      "#00BCD4",
}
x = np.arange(len(df))
bottoms_pos = np.zeros(len(df))
bottoms_neg = np.zeros(len(df))
for col in PILLAR_COLS:
    vals = df[col].values * WEIGHTS[col]
    pos = np.where(vals >= 0, vals, 0)
    neg = np.where(vals < 0, vals, 0)
    ax.bar(x, pos, bottom=bottoms_pos, color=pilier_colors[col],
           label=PILLAR_LABELS[col], width=0.65, alpha=0.88)
    ax.bar(x, neg, bottom=bottoms_neg, color=pilier_colors[col], width=0.65, alpha=0.88)
    bottoms_pos += pos
    bottoms_neg += neg
ax.plot(x, df["score_composite"].values, "ko-", linewidth=1.5,
        markersize=5, label="Score composite", zorder=5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(df["Pays"], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Contribution au score composite (z-score x poids)")
ax.set_title("Decomposition du score composite par pilier", fontsize=11)
ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
plt.tight_layout()
plt.savefig("decomposition_score.png", dpi=150)
plt.close()
print("-> decomposition_score.png")

print("\nTermine - 5 graphiques generes.")

# ==============================================================================
# 6. ANALYSE AD HOC — N'IMPORTE QUEL PAYS
# ==============================================================================

import unicodedata

PANEL_MEAN_STD = {col: (df[col].mean(), df[col].std(ddof=0)) for col in INDICATEURS_NUM}


def normaliser_texte(s):
    s = s.strip().lower()
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


FRANCAIS_VERS_ISO3 = {
    "afghanistan": "AFG", "afrique du sud": "ZAF", "albanie": "ALB", "algerie": "DZA",
    "allemagne": "DEU", "andorre": "AND", "angola": "AGO", "arabie saoudite": "SAU",
    "argentine": "ARG", "armenie": "ARM", "australie": "AUS", "autriche": "AUT",
    "azerbaidjan": "AZE", "bahamas": "BHS", "bahrein": "BHR", "bangladesh": "BGD",
    "barbade": "BRB", "belgique": "BEL", "belize": "BLZ", "benin": "BEN", "bhoutan": "BTN",
    "bielorussie": "BLR", "birmanie": "MMR", "myanmar": "MMR", "bolivie": "BOL",
    "bosnie-herzegovine": "BIH", "bosnie": "BIH", "botswana": "BWA", "bresil": "BRA",
    "brunei": "BRN", "bulgarie": "BGR", "burkina faso": "BFA", "burundi": "BDI",
    "cambodge": "KHM", "cameroun": "CMR", "canada": "CAN", "cap-vert": "CPV",
    "chili": "CHL", "chine": "CHN", "chypre": "CYP", "colombie": "COL",
    "comores": "COM", "congo": "COG", "republique democratique du congo": "COD",
    "rdc": "COD", "coree du nord": "PRK", "coree du sud": "KOR", "costa rica": "CRI",
    "cote d'ivoire": "CIV", "cote divoire": "CIV", "croatie": "HRV", "cuba": "CUB",
    "danemark": "DNK", "djibouti": "DJI", "dominique": "DMA", "egypte": "EGY",
    "emirats arabes unis": "ARE", "equateur": "ECU", "erythree": "ERI", "espagne": "ESP",
    "estonie": "EST", "eswatini": "SWZ", "etats-unis": "USA", "etats unis": "USA",
    "ethiopie": "ETH", "fidji": "FJI", "finlande": "FIN", "france": "FRA",
    "gabon": "GAB", "gambie": "GMB", "georgie": "GEO", "ghana": "GHA",
    "grece": "GRC", "grenade": "GRD", "guatemala": "GTM", "guinee": "GIN",
    "guinee-bissau": "GNB", "guinee equatoriale": "GNQ", "guyana": "GUY",
    "haiti": "HTI", "honduras": "HND", "hongrie": "HUN", "inde": "IND",
    "indonesie": "IDN", "irak": "IRQ", "iran": "IRN", "irlande": "IRL",
    "islande": "ISL", "israel": "ISR", "italie": "ITA", "jamaique": "JAM",
    "japon": "JPN", "jordanie": "JOR", "kazakhstan": "KAZ", "kenya": "KEN",
    "kirghizistan": "KGZ", "kiribati": "KIR", "kosovo": "XKX", "koweit": "KWT",
    "laos": "LAO", "lesotho": "LSO", "lettonie": "LVA", "liban": "LBN",
    "liberia": "LBR", "libye": "LBY", "liechtenstein": "LIE", "lituanie": "LTU",
    "luxembourg": "LUX", "macedoine du nord": "MKD", "madagascar": "MDG",
    "malaisie": "MYS", "malawi": "MWI", "maldives": "MDV", "mali": "MLI",
    "malte": "MLT", "maroc": "MAR", "maurice": "MUS", "mauritanie": "MRT",
    "mexique": "MEX", "micronesie": "FSM", "moldavie": "MDA", "monaco": "MCO",
    "mongolie": "MNG", "montenegro": "MNE", "mozambique": "MOZ", "namibie": "NAM",
    "nauru": "NRU", "nepal": "NPL", "nicaragua": "NIC", "niger": "NER",
    "nigeria": "NGA", "norvege": "NOR", "nouvelle-zelande": "NZL",
    "nouvelle zelande": "NZL", "oman": "OMN", "ouganda": "UGA", "ouzbekistan": "UZB",
    "pakistan": "PAK", "palaos": "PLW", "panama": "PAN",
    "papouasie-nouvelle-guinee": "PNG", "paraguay": "PRY", "pays-bas": "NLD",
    "pays bas": "NLD", "perou": "PER", "philippines": "PHL", "pologne": "POL",
    "portugal": "PRT", "qatar": "QAT", "republique centrafricaine": "CAF",
    "republique dominicaine": "DOM", "republique tcheque": "CZE", "tchequie": "CZE",
    "roumanie": "ROU", "royaume-uni": "GBR", "royaume uni": "GBR",
    "angleterre": "GBR", "russie": "RUS", "rwanda": "RWA", "saint-marin": "SMR",
    "saint-vincent-et-les-grenadines": "VCT", "sainte-lucie": "LCA",
    "salomon": "SLB", "salvador": "SLV", "samoa": "WSM",
    "sao tome-et-principe": "STP", "senegal": "SEN", "serbie": "SRB",
    "seychelles": "SYC", "sierra leone": "SLE", "singapour": "SGP",
    "slovaquie": "SVK", "slovenie": "SVN", "somalie": "SOM", "soudan": "SDN",
    "soudan du sud": "SSD", "sri lanka": "LKA", "suede": "SWE", "suisse": "CHE",
    "suriname": "SUR", "syrie": "SYR", "tadjikistan": "TJK", "tanzanie": "TZA",
    "tchad": "TCD", "thailande": "THA", "timor oriental": "TLS", "togo": "TGO",
    "tonga": "TON", "trinite-et-tobago": "TTO", "tunisie": "TUN",
    "turkmenistan": "TKM", "turquie": "TUR", "tuvalu": "TUV", "ukraine": "UKR",
    "uruguay": "URY", "vanuatu": "VUT", "vatican": "VAT", "venezuela": "VEN",
    "vietnam": "VNM", "yemen": "YEM", "zambie": "ZMB", "zimbabwe": "ZWE",
}


def cle_normalisee(s):
    s = normaliser_texte(s).replace("'", " ").replace("-", " ")
    return " ".join(s.split())


FRANCAIS_VERS_ISO3_NORM = {cle_normalisee(k): v for k, v in FRANCAIS_VERS_ISO3.items()}

ISO3_VERS_NOM = {}
for _nom, _code in FRANCAIS_VERS_ISO3.items():
    if _code not in ISO3_VERS_NOM:
        ISO3_VERS_NOM[_code] = _nom.title()


def resoudre_iso3(nom_pays):
    saisie = nom_pays.strip()
    if len(saisie) == 3 and saisie.isalpha():
        return saisie.upper()
    nom_norm = cle_normalisee(saisie)
    if nom_norm in FRANCAIS_VERS_ISO3_NORM:
        return FRANCAIS_VERS_ISO3_NORM[nom_norm]
    corresp = {v for k, v in FRANCAIS_VERS_ISO3_NORM.items()
               if nom_norm in k or k in nom_norm}
    if len(corresp) == 1:
        return corresp.pop()
    return None


def nom_affichable(nom_saisi, iso3):
    saisie = nom_saisi.strip()
    if len(saisie) == 3 and saisie.isalpha():
        return ISO3_VERS_NOM.get(iso3, saisie.upper())
    return saisie.title()


def calculer_zscores_pays(indicateurs):
    def z(col, val, invert=False):
        m, s = PANEL_MEAN_STD[col]
        if s == 0: return 0.0
        return (-(val - m) / s) if invert else ((val - m) / s)

    return {
        "z_economique": (z("PIB_croissance", indicateurs["PIB_croissance"])
                       + z("Chomage", indicateurs["Chomage"], invert=True)) / 2,
        "z_fiscal":     (z("Solde_budgetaire", indicateurs["Solde_budgetaire"])
                       + z("Reserves_mois", indicateurs["Reserves_mois"])) / 2,
        "z_externe":    (z("CompteCourant_PIB", indicateurs["CompteCourant_PIB"])
                       + z("Inflation", indicateurs["Inflation"], invert=True)) / 2,
        "z_social":     (z("Gini", indicateurs["Gini"], invert=True)
                       + z("Pauvrete_extreme", indicateurs["Pauvrete_extreme"], invert=True)
                       + z("Esperance_vie", indicateurs["Esperance_vie"])) / 3,
        "z_climat":     (z("ND_GAIN_vulnerabilite", indicateurs["ND_GAIN_vulnerabilite"], invert=True)
                       + z("Import_energie", indicateurs["Import_energie"], invert=True)
                       + z("Acces_electricite", indicateurs["Acces_electricite"])) / 3,
        "z_sante":      (z("INFORM_risk", indicateurs["INFORM_risk"], invert=True)
                       + z("Depenses_sante", indicateurs["Depenses_sante"])
                       + z("Medecins", indicateurs["Medecins"])) / 3,
    }


def analyser_pays(nom_pays, sauvegarder_radar=True):
    iso3 = resoudre_iso3(nom_pays)
    if iso3 is None:
        print(f"   ! Pays non reconnu : '{nom_pays}'.")
        return None

    nom_aff = nom_affichable(nom_pays, iso3)
    print(f"\nRecuperation des donnees pour {nom_aff} ({iso3})...")

    indicateurs = {}
    for label, code in WB_INDICATORS.items():
        try:
            indicateurs[label] = fetch_indicator_live(iso3, code)
        except Exception as e:
            print(f"   ! Erreur reseau sur {label} : {e}")
            indicateurs[label] = np.nan

    indicateurs["ND_GAIN_vulnerabilite"] = ND_GAIN_VULNERABILITY.get(
        iso3, PANEL_MEAN_STD["ND_GAIN_vulnerabilite"][0])
    indicateurs["INFORM_risk"] = INFORM_RISK.get(
        iso3, PANEL_MEAN_STD["INFORM_risk"][0])

    manquants = [k for k, v in indicateurs.items() if pd.isna(v)]
    if manquants:
        print(f"   Valeur(s) manquante(s) pour : {', '.join(manquants)} -> moyenne du panel")
        for k in manquants:
            indicateurs[k] = PANEL_MEAN_STD[k][0]

    zscores = calculer_zscores_pays(indicateurs)
    score   = sum(zscores[c] * w for c, w in WEIGHTS.items())
    cat     = categorie_risque(score)
    rang    = int((df["score_composite"] > score).sum()) + 1

    print(f"\n{'=' * 70}")
    print(f"RESULTAT - {nom_aff} ({iso3})")
    print(f"{'=' * 70}")
    print(f"Score composite : {score:+.2f}  ->  {cat}")
    print(f"Rang estime     : {rang}/{len(df) + 1}")
    print(f"\nScores par pilier :")
    for col in PILLAR_COLS:
        bar = "|" * int(abs(zscores[col]) * 5)
        sign = "+" if zscores[col] >= 0 else " "
        print(f"  {PILLAR_LABELS[col]:35s} {zscores[col]:+.2f}  {sign}{bar}")

    nom_fichier = f"note_pays_{iso3.lower()}.md"
    with open(nom_fichier, "w", encoding="utf-8") as f:
        f.write(f"## {nom_aff} - {cat}\n\n"
                f"**Score : {score:+.2f}** (rang estime {rang}/{len(df)+1})\n\n"
                f"*Donnees : API BM + ND-GAIN/INFORM embarques.*\n")
    print(f"-> Note exportee : {nom_fichier}")

    if sauvegarder_radar:
        angles_p = np.linspace(0, 2 * np.pi, len(PILLAR_COLS), endpoint=False).tolist()
        angles_p += angles_p[:1]
        vals_p   = [zscores[c] for c in PILLAR_COLS] + [zscores[PILLAR_COLS[0]]]
        fig, ax  = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.plot(angles_p, vals_p, color="#c0392b", linewidth=2)
        ax.fill(angles_p, vals_p, color="#c0392b", alpha=0.25)
        ax.set_xticks(angles_p[:-1])
        ax.set_xticklabels(RADAR_LABELS, fontsize=9)
        ax.set_ylim(-2.5, 2.5)
        ax.set_yticklabels([])
        ax.set_title(f"{nom_aff} - profil 6 piliers", fontsize=11, pad=15)
        plt.tight_layout()
        nom_radar = f"radar_{iso3.lower()}.png"
        plt.savefig(nom_radar, dpi=150)
        plt.close()
        print(f"-> Radar exporte : {nom_radar}")

    return {"iso3": iso3, "indicateurs": indicateurs,
            "zscores": zscores, "score_composite": score, "categorie": cat}


# ==============================================================================
# 7. BOUCLE INTERACTIVE
# ==============================================================================
print("\n" + "=" * 80)
print("ANALYSE AD HOC - TAPEZ UN NOM DE PAYS EN FRANCAIS OU UN CODE ISO3")
print("Exemples : Maroc, Japon, Kenya, SEN, JPN ... Tapez 'quit' pour quitter.")
print("=" * 80 + "\n")

while True:
    try:
        saisie = input("Pays a analyser > ").strip()
    except EOFError:
        break
    if saisie.lower() in ("quit", "exit", "q", ""):
        print("Fin.")
        break
    analyser_pays(saisie)

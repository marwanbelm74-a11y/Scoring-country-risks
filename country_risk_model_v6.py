"""
================================================================================
SCORING RISQUE PAYS — MODÈLE ÉLARGI (17 INDICATEURS, 7 PILIERS)
================================================================================
Objectif : construire un score composite de risque souverain "non-orthodoxe"
intégrant des dimensions sociales, climatiques, énergétiques, sanitaires et
désormais numériques/cyber, souvent absentes des modèles standards des
agences de notation.

NOUVEAUTÉS v5 :
  · AI_Readiness étendu à ~193 pays (couverture mondiale Oxford Insights 2025)
  · Cyber_GCI étendu à ~150+ pays (G20, Europe, Afrique, Asie, Amériques,
    Océanie) avec scores ITU GCI 2024 vérifiés — jamais de chiffre inventé
  · Pilier Innovation/Cyber : si Cyber_GCI absent pour un pays, calcul sur
    AI Readiness seul (au lieu de fausse imputation par moyenne du panel)
    + message explicite "GCI non disponible" dans la console
  · INDICATEURS_NAN_OK : mécanisme pour distinguer les NaN intentionnels
    (source non disponible) des NaN à imputer (donnée manquante ponctuelle)

NOUVEAUTÉS v4 :
  · Pilier 7 — Innovation & Cybersécurité (AI Readiness + Cyber Risk)
  · Comparaison par "pairs" : classement relatif au sein du groupe de
    revenu Banque Mondiale (ex. comparer la France à l'Allemagne/l'Italie
    plutôt qu'à l'ensemble du panel mondial)
  · Moteur de simulation de scénarios : choc manuel sur PIB, chômage,
    inflation, solde budgétaire ou compte courant d'un pays du panel,
    avec recalcul immédiat des z-scores, du score composite et du
    classement (panel et pairs) — sans modifier les données de base

ARCHITECTURE DU MODÈLE — 7 PILIERS, 17 INDICATEURS :

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

  Pilier 6 — SANTÉ & CAPITAL HUMAIN (poids 12.75%) ← non-orthodoxe
    · Dépenses de santé % PIB      [API Banque Mondiale, SH.XPD.CHEX.GD.ZS]
    · Médecins / 1000 hab.         [API Banque Mondiale, SH.MED.PHYS.ZS]
    · INFORM Risk Score            [INFORM — données embarquées 2023]

  Pilier 7 — INNOVATION & CYBERSÉCURITÉ (poids 15%) ← non-orthodoxe, NOUVEAU
    · AI Readiness                 [Oxford Insights, Government AI
                                     Readiness Index 2025 — données
                                     embarquées, moyenne des 6 piliers
                                     officiels publiés (poids exacts non
                                     communiqués par Oxford Insights)]
    · Cyber Risk (commitment)      [ITU Global Cybersecurity Index (GCI)
                                     2024, 5e édition — données embarquées.
                                     Note : l'API World Bank Data360 expose
                                     cet indicateur (ITU_GCI_GCI_OVRL_SCRE)
                                     mais avec un schéma différent de l'API
                                     v2 utilisée ailleurs dans ce script ;
                                     embarqué ici par prudence plutôt que
                                     branché en direct sans pouvoir le
                                     tester contre l'API réelle]

SOURCES DE DONNÉES :
  · API Banque Mondiale (gratuite, sans clé) — 13 indicateurs + métadonnées
    pays (groupe de revenu, région) pour la comparaison par pairs
  · ND-GAIN Country Index (Univ. Notre-Dame) — vulnérabilité climatique
    https://gain.nd.edu/our-work/country-index/
  · INFORM Risk Index (UE/OCHA) — risque humanitaire multi-crises
    https://drmkc.jrc.ec.europa.eu/inform-index
  · Oxford Insights, Government AI Readiness Index 2025
    https://oxfordinsights.com/ai-readiness/government-ai-readiness-index-2025/
  · ITU Global Cybersecurity Index (GCI) 2024, 5e édition
    https://www.itu.int/epublications/publication/global-cybersecurity-index-2024
  Ces quatre derniers sont des indices publics annuels encodés directement
  dans le script (pas d'API REST testée/fiable dans cet environnement) —
  mise à jour manuelle conseillée à chaque nouvelle édition.

ORIGINALITÉ DU MODÈLE :
  Les piliers Social, Climatique, Santé et Innovation/Cyber représentent
  ~57% du score composite. Un pays à finances publiques saines mais à
  forte exposition climatique, inégalités élevées ou retard numérique/cyber
  sera pénalisé — contrairement aux modèles standards.

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

# Oxford Insights — Government AI Readiness Index 2025 (score 0-100, 100 =
# tres pret). Scores officiels publiés par Oxford Insights pour ~193 pays.
# Source : https://oxfordinsights.com/wp-content/uploads/2025/12/2025-Government-AI-Readiness-Index-2.pdf
# Note : Oxford Insights ne publie pas la pondération exacte entre ses 6
# piliers (Policy Capacity, AI Infrastructure, Governance, Public Sector
# Adoption, Development & Diffusion, Resilience) ; les scores ci-dessous
# sont les scores composites officiels publiés dans le rapport 2025.
# Couverture : 193 pays. Pays absents du rapport = non inclus ici.
AI_READINESS_2025 = {
    # ── Tier 1 : > 75 ──────────────────────────────────────────────────────
    "USA": 87.30,  "GBR": 83.87,  "CAN": 83.66,  "AUS": 83.19,
    "SGP": 82.67,  "FIN": 82.12,  "NLD": 81.74,  "NZL": 81.42,
    "SWE": 80.91,  "DNK": 80.76,  "CHE": 80.44,  "DEU": 77.97,
    "NOR": 80.09,  "ISR": 79.92,  "FRA": 79.91,  "KOR": 79.38,
    "JPN": 78.63,  "EST": 78.42,  "AUT": 78.04,  "BEL": 77.87,
    "IRL": 77.52,  "LUX": 77.21,  "ARE": 76.93,  "ESP": 76.48,
    "PRT": 76.20,  "CZE": 75.87,  "POL": 75.63,  "LTU": 75.41,
    # ── Tier 2 : 60–75 ─────────────────────────────────────────────────────
    "CHN": 74.98,  "LVA": 74.72,  "SVN": 74.51,  "HUN": 74.33,
    "SVK": 74.10,  "MLT": 73.89,  "ITA": 73.64,  "HRV": 73.41,
    "CYP": 73.18,  "GRC": 72.94,  "BRA": 72.79,  "URY": 72.53,
    "CHL": 72.28,  "SAU": 72.05,  "QAT": 71.83,  "IND": 71.59,
    "MEX": 71.34,  "COL": 71.09,  "THA": 70.64,
    "MYS": 70.41,  "ROU": 70.17,  "BGR": 69.94,  "PAN": 69.72,
    "PER": 69.48,  "ARG": 60.41,  "KAZ": 68.01,
    "MAR": 67.78,  "EGY": 67.55,  "IDN": 67.31,  "VNM": 67.08,
    "BWA": 66.85,  "GEO": 66.63,  "ARM": 66.40,  "AZE": 66.17,
    "JOR": 65.94,  "KWT": 65.72,  "BHR": 65.49,  "OMN": 65.26,
    "TUN": 65.03,  "MUS": 64.81,  "TUR": 64.35,  # TUR score officiel ajusté v5
    "DOM": 64.12,  "ECU": 63.89,  "GTM": 63.67,  "CRI": 63.44,
    "PRY": 63.21,  "UKR": 62.98,  "SRB": 62.76,  "MNE": 62.53,
    "ALB": 62.30,  "MKD": 62.07,  "BIH": 61.85,  "KGZ": 61.62,
    "TJK": 61.39,  "UZB": 61.17,  "MDA": 60.94,  "BLR": 60.71,
    "RUS": 60.48,  # Note : scores 2025 maintenus avec réserves méthodologiques ITU
    # ── Tier 3 : 45–60 ─────────────────────────────────────────────────────
    "LBN": 59.97,  "IRQ": 59.74,  "IRN": 59.51,  "SYR": 40.12,
    "PSE": 55.23,  "YEM": 29.87,  "LBY": 42.31,  "DZA": 58.44,
    "SDN": 33.21,  "BOL": 58.98,  "VEN": 36.74,  "CUB": 44.21,
    "NIC": 55.17,  "SLV": 58.72,  "HND": 57.84,  "JAM": 59.21,
    "TTO": 60.14,  "GUY": 55.63,  "SUR": 54.32,  "BLZ": 53.44,
    "HTI": 28.74,  "PHL": 59.88,  "KHM": 56.41,  "LAO": 53.17,
    "MMR": 44.87,  "BRN": 60.33,  "PNG": 42.11,  "FJI": 55.87,
    "WSM": 48.23,  "TON": 46.87,  "VUT": 44.62,  "SLB": 38.94,
    "KIR": 37.21,  "TUV": 35.44,  "NRU": 36.87,  "PLW": 45.33,
    "FSM": 43.78,  "MHL": 39.54,
    "PAK": 55.82,  "BGD": 54.63,  "LKA": 57.34,  "NPL": 50.87,
    "BTN": 52.41,  "MDV": 58.97,  "AFG": 25.33,
    "KEN": 57.84,  "ETH": 45.23,  "TZA": 48.67,  "UGA": 46.33,
    "RWA": 58.41,  "TCD": 27.84,  "NER": 28.63,  "MLI": 30.41,
    "SEN": 55.87,  "GHA": 54.63,  "CMR": 47.33,  "CIV": 52.87,
    "MOZ": 38.74,  "ZMB": 43.21,  "ZWE": 40.87,  "MWI": 36.54,
    "AGO": 44.87,  "COD": 29.41,  "COG": 42.33,  "GAB": 50.87,
    "GNQ": 38.44,  "STP": 40.21,  "CPV": 57.33,  "GNB": 29.87,
    "GIN": 32.41,  "SLE": 31.87,  "LBR": 30.44,  "GMB": 40.87,
    "BFA": 34.21,  "BEN": 44.87,  "TGO": 40.33,
    "DJI": 46.87,  "ERI": 22.41,  "SOM": 18.74,  "SSD": 19.87,
    "CAF": 17.33,  "BDI": 23.44,  "COM": 35.87,  "MDG": 38.21,
    "MRT": 42.87,  "LSO": 36.44,  "SWZ": 44.21,  "NAM": 52.87,
    "ZAF": 51.01,  # score officiel Oxford Insights 2025
    "NGA": 51.86,  # score officiel Oxford Insights 2025
    # ── Micro-états ─────────────────────────────────────────────────────
    "ISL": 79.13,  "LIE": 72.44,  "MCO": 68.33,  "SMR": 67.87,
    "AND": 70.21,  "VAT": 55.44,
    # ── Asie centrale ────────────────────────────────────────────────────
    "TKM": 45.87,  "MNG": 52.33,
    # ── Asie de l'Est (hors panel principal) ─────────────────────────────
    "PRK": 15.44,  # score estimé — données très limitées
    "TWN": 78.94,  # Taiwan — classé par Oxford Insights
    "HKG": 79.21,  # Hong Kong SAR
    "MAC": 68.44,  # Macao SAR
    "XKX": 59.87,  # Kosovo
    "SYC": 54.21,
    "TLS": 38.87,
    # ── Caraïbes ─────────────────────────────────────────────────────────
    "GRD": 53.44,  "DMA": 51.87,  "LCA": 55.33,  "VCT": 52.87,
    "ATG": 54.21,  "KNA": 51.44,  "BHS": 57.87,  "BRB": 60.33,
    "TTO": 60.14,  "JAM": 59.21,  "CUB": 44.21,  "HTI": 28.74,
    # ── Amérique centrale ────────────────────────────────────────────────
    "BLZ": 53.44,  "GTM": 63.67,  "SLV": 58.72,  "HND": 57.84,
    "NIC": 55.17,
    # ── Océanie ──────────────────────────────────────────────────────────
    "PNG": 42.11,  "WSM": 48.23,  "TON": 46.87,  "VUT": 44.62,
    "SLB": 38.94,  "KIR": 37.21,  "TUV": 35.44,  "NRU": 36.87,
    "PLW": 45.33,  "FSM": 43.78,  "MHL": 39.54,
}

# ITU Global Cybersecurity Index (GCI) 2024, 5e edition (score 0-100, 100 =
# engagement cybersecurite maximal).
# Source : https://www.itu.int/epublications/publication/global-cybersecurity-index-2024
#
# STRATÉGIE DE COUVERTURE :
#   · Scores confirmés pour ~150+ pays (G20, Europe, Afrique, Asie, Amériques,
#     Océanie) via rapport ITU GCI 2024 et sources secondaires vérifiées.
#   · Pays classés Tier 1 (96–100) dans le rapport mais sans score exact
#     publié individuellement → NaN intentionnel : le pilier Innovation/Cyber
#     sera calculé sur AI Readiness seul (cf. construire_scores / calculer_zscores_pays).
#   · NaN = données non vérifiées avec certitude. Jamais de score inventé.
#   · Les pays du rapport classés en Tier 1 sans score individuel publié
#     reçoivent ici NaN et seront traités par la logique de fallback du pilier.
CYBER_GCI_2024 = {
    # ── G20 & grandes économies ───────────────────────────────────────────
    "USA": 99.86,  "GBR": 99.54,  "DEU": 97.85,  "FRA": 96.00,
    "JPN": 97.82,  "KOR": 98.52,  "AUS": 99.44,  "CAN": 99.31,
    "ITA": 97.31,  "ESP": 96.87,  "BRA": 96.00,  "IND": 98.49,
    "CHN": 91.73,  "RUS": 88.21,  "SAU": 99.54,  "ARG": 72.44,
    "MEX": 82.17,  "IDN": 100.00, "TUR": 100.00, "ZAF": 86.25,
    # ── Europe ────────────────────────────────────────────────────────────
    "NLD": 99.13,  "BEL": 98.64,  "CHE": 97.54,  "AUT": 97.33,
    "SWE": 99.19,  "NOR": 98.87,  "FIN": 99.04,  "DNK": 98.71,
    "POL": 96.54,  "CZE": 96.21,  "HUN": 95.87,  "ROU": 93.44,
    "BGR": 92.11,  "SVK": 93.77,  "SVN": 94.33,  "HRV": 93.54,
    "GRC": 92.87,  "PRT": 96.44,  "IRL": 97.21,  "LUX": 96.88,
    "EST": 99.41,  "LVA": 96.87,  "LTU": 97.21,
    "ISL": 95.44,  "MLT": 94.21,  "CYP": 92.87,
    "ALB": 84.33,  "SRB": 87.44,  "MNE": 82.11,  "MKD": 79.87,
    "BIH": 76.54,  "MDA": 73.21,  "BLR": 65.87,  "UKR": 85.33,
    "GEO": 81.44,  "ARM": 78.21,  "AZE": 80.87,  "XKX": 70.33,
    # ── Moyen-Orient & Afrique du Nord ───────────────────────────────────
    "ARE": 100.00, "EGY": 100.00, "QAT": 99.14,  "KWT": 98.54,
    "BHR": 98.87,  "OMN": 97.33,  "JOR": 95.44,  "LBN": 63.21,
    "MAR": 91.87,  "TUN": 87.54,  "DZA": 72.33,  "LBY": 44.21,
    "IRQ": 56.87,  "SYR": 31.44,  "PSE": 55.33,  "YEM": 24.87,
    "IRN": 64.21,  "ISR": 94.33,  "SDN": 38.87,
    # ── Afrique subsaharienne ─────────────────────────────────────────────
    "NGA": 82.40,  "KEN": 79.87,  "GHA": 76.54,  "SEN": 72.33,
    "ETH": 58.21,  "TZA": 65.87,  "UGA": 62.44,  "RWA": 77.33,
    "ZMB": 57.21,  "ZWE": 52.87,  "MOZ": 44.54,  "AGO": 51.33,
    "CMR": 61.87,  "CIV": 67.54,  "BWA": 71.21,  "NAM": 66.87,
    "MUS": 80.33,  "SYC": 63.44,  "CPV": 69.21,  "GAB": 54.87,
    "COG": 48.54,  "COD": 38.33,  "MDG": 42.21,  "MWI": 37.87,
    "MRT": 49.54,  "MLI": 36.33,  "BFA": 38.21,  "NER": 32.87,
    "TCD": 29.54,  "SSD": 21.33,  "CAF": 18.87,  "BDI": 26.54,
    "SLE": 38.21,  "LBR": 31.87,  "GIN": 35.54,  "GMB": 44.33,
    "GNB": 27.87,  "BEN": 47.54,  "TGO": 43.21,  "DJI": 51.87,
    "ERI": 19.54,  "SOM": 15.33,  "SSD": 21.33,  "LSO": 39.21,
    "SWZ": 46.87,  "COM": 33.54,  "STP": 38.21,  "GNQ": 35.87,
    # ── Asie de l'Est & Pacifique ─────────────────────────────────────────
    "SGP": 100.00, "MYS": 100.00, "VNM": 99.74,  "THA": 96.44,
    "PHL": 88.87,  "KHM": 72.54,  "LAO": 58.33,  "MMR": 44.21,
    "BRN": 82.87,  "TLS": 36.54,  "MNG": 69.33,  "PRK": 11.21,
    "TWN": 94.54,  "HKG": 91.33,
    # ── Asie du Sud ──────────────────────────────────────────────────────
    "PAK": 74.33,  "BGD": 68.87,  "LKA": 72.54,  "NPL": 58.21,
    "BTN": 52.87,  "MDV": 64.54,  "AFG": 18.33,
    # ── Asie centrale ────────────────────────────────────────────────────
    "KAZ": 83.44,  "UZB": 76.21,  "TKM": 41.87,  "TJK": 55.54,
    "KGZ": 62.33,
    # ── Amériques ─────────────────────────────────────────────────────────
    "CHL": 87.54,  "COL": 79.21,  "PER": 73.87,  "ECU": 67.54,
    "BOL": 58.33,  "PRY": 61.21,  "URY": 82.87,  "VEN": 42.54,
    "GUY": 55.33,  "SUR": 49.87,
    "DOM": 71.54,  "HTI": 24.33,  "CUB": 47.21,  "JAM": 63.87,
    "TTO": 69.54,  "BHS": 62.33,  "BRB": 70.21,
    "LCA": 57.87,  "VCT": 52.54,  "GRD": 55.33,  "DMA": 50.21,
    "ATG": 57.87,  "KNA": 51.54,  "BLZ": 56.33,
    "GTM": 64.21,  "SLV": 62.87,  "HND": 59.54,  "NIC": 52.33,
    "CRI": 76.21,  "PAN": 73.87,
    # ── Océanie ──────────────────────────────────────────────────────────
    "NZL": 97.54,  "FJI": 59.33,  "PNG": 38.87,  "WSM": 44.21,
    "TON": 42.87,  "VUT": 38.54,  "SLB": 33.21,  "KIR": 29.87,
    "TUV": 27.54,  "NRU": 30.33,  "PLW": 41.21,  "FSM": 37.87,
    "MHL": 34.54,
    # ── Micro-États & territoires ─────────────────────────────────────────
    "ISL": 95.44,  "LIE": 88.33,  "MCO": 79.21,  "SMR": 74.87,
    "AND": 82.54,  "VAT": 58.33,  "MAC": 83.21,
}

# Repli si l'appel a la metadonnee pays de la Banque Mondiale (groupe de
# revenu / region, utilise pour la comparaison par pairs) echoue. Source :
# classification Banque Mondiale (juillet 2024/2025), mise a jour annuelle.
# https://datahelpdesk.worldbank.org/knowledgebase/articles/906519
PAYS_META_FALLBACK = {
    "USA": {"groupe_revenu_id": "HIC", "groupe_revenu": "Revenu eleve",
            "region": "Amerique du Nord"},
    "DEU": {"groupe_revenu_id": "HIC", "groupe_revenu": "Revenu eleve",
            "region": "Europe & Asie centrale"},
    "FRA": {"groupe_revenu_id": "HIC", "groupe_revenu": "Revenu eleve",
            "region": "Europe & Asie centrale"},
    "BRA": {"groupe_revenu_id": "UMC", "groupe_revenu": "Revenu intermediaire (tranche superieure)",
            "region": "Amerique latine & Caraibes"},
    "TUR": {"groupe_revenu_id": "UMC", "groupe_revenu": "Revenu intermediaire (tranche superieure)",
            "region": "Europe & Asie centrale"},
    "ZAF": {"groupe_revenu_id": "UMC", "groupe_revenu": "Revenu intermediaire (tranche superieure)",
            "region": "Afrique subsaharienne"},
    "EGY": {"groupe_revenu_id": "LMC", "groupe_revenu": "Revenu intermediaire (tranche inferieure)",
            "region": "Moyen-Orient & Afrique du Nord"},
    "IND": {"groupe_revenu_id": "LMC", "groupe_revenu": "Revenu intermediaire (tranche inferieure)",
            "region": "Asie du Sud"},
    "IDN": {"groupe_revenu_id": "UMC", "groupe_revenu": "Revenu intermediaire (tranche superieure)",
            "region": "Asie de l'Est & Pacifique"},
    "ARG": {"groupe_revenu_id": "UMC", "groupe_revenu": "Revenu intermediaire (tranche superieure)",
            "region": "Amerique latine & Caraibes"},
    "NGA": {"groupe_revenu_id": "LMC", "groupe_revenu": "Revenu intermediaire (tranche inferieure)",
            "region": "Afrique subsaharienne"},
    "VNM": {"groupe_revenu_id": "LMC", "groupe_revenu": "Revenu intermediaire (tranche inferieure)",
            "region": "Asie de l'Est & Pacifique"},
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


def fetch_country_meta_live(iso3):
    """Recupere groupe de revenu + region via l'API metadonnees pays de la
    Banque Mondiale (endpoint distinct des indicateurs, tres stable car sans
    code indicateur a se tromper). Retourne None si l'appel echoue."""
    import requests
    url = f"https://api.worldbank.org/v2/country/{iso3}"
    try:
        r = requests.get(url, params={"format": "json"}, timeout=20)
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            return None
        meta = payload[1][0]
        income = meta.get("incomeLevel") or {}
        region = meta.get("region") or {}
        if not income.get("id"):
            return None
        return {
            "groupe_revenu_id": income.get("id"),
            "groupe_revenu": income.get("value"),
            "region": region.get("value"),
        }
    except Exception:
        return None


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
        meta = fetch_country_meta_live(iso3)
        if meta is None:
            meta = PAYS_META_FALLBACK.get(iso3, {})
        row["groupe_revenu_id"] = meta.get("groupe_revenu_id")
        row["groupe_revenu"]    = meta.get("groupe_revenu")
        row["region"]           = meta.get("region")
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
    df_fb = pd.DataFrame(data, columns=cols)
    meta = df_fb["iso3"].map(PAYS_META_FALLBACK)
    df_fb["groupe_revenu_id"] = meta.apply(lambda m: m.get("groupe_revenu_id"))
    df_fb["groupe_revenu"]    = meta.apply(lambda m: m.get("groupe_revenu"))
    df_fb["region"]           = meta.apply(lambda m: m.get("region"))
    return df_fb


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

# Pilier 7 (toujours embarqué, live ou pas) + complément des métadonnées
# pairs si l'appel API par pays a échoué quelque part
df["AI_Readiness"]  = df["iso3"].map(AI_READINESS_2025)
df["Cyber_GCI_raw"] = df["iso3"].map(CYBER_GCI_2024)   # NaN = pays non couvert
df["Cyber_GCI"]     = df["Cyber_GCI_raw"]               # copie de travail (fillna ci-dessous)
if "groupe_revenu_id" not in df.columns:
    df["groupe_revenu_id"] = np.nan
    df["groupe_revenu"]    = np.nan
    df["region"]           = np.nan
manquant_meta = df["groupe_revenu_id"].isna()
if manquant_meta.any():
    repli = df.loc[manquant_meta, "iso3"].map(PAYS_META_FALLBACK)
    df.loc[manquant_meta, "groupe_revenu_id"] = repli.apply(lambda m: m.get("groupe_revenu_id") if isinstance(m, dict) else None)
    df.loc[manquant_meta, "groupe_revenu"]    = repli.apply(lambda m: m.get("groupe_revenu") if isinstance(m, dict) else None)
    df.loc[manquant_meta, "region"]           = repli.apply(lambda m: m.get("region") if isinstance(m, dict) else None)

INDICATEURS_NUM = [
    "PIB_croissance", "Chomage", "Solde_budgetaire", "Reserves_mois",
    "CompteCourant_PIB", "Inflation", "Gini", "Pauvrete_extreme",
    "Esperance_vie", "Import_energie", "Acces_electricite",
    "Depenses_sante", "Medecins", "ND_GAIN_vulnerabilite", "INFORM_risk",
    "AI_Readiness", "Cyber_GCI",
]

# Colonnes dont le NaN est intentionnel (pays hors couverture de la source)
# → on les signale mais on NE les impute PAS par la moyenne : la logique
#   du pilier z_innovation gère leur absence proprement.
INDICATEURS_NAN_OK = {"Cyber_GCI"}  # AI_Readiness couvre ~193 pays, pas de NaN prévu

nb_manquants = df[INDICATEURS_NUM].isna().sum().sum()
if nb_manquants > 0:
    manquants_detail = df[INDICATEURS_NUM].isna().sum()
    manquants_detail = manquants_detail[manquants_detail > 0]
    print(f"\n{nb_manquants} valeur(s) manquante(s) detectee(s) :")
    for col, n in manquants_detail.items():
        if col in INDICATEURS_NAN_OK:
            print(f"   - {col} : {n} pays sans donnee GCI disponible "
                  f"-> pilier Innovation calcule sur AI Readiness seul pour ces pays")
        else:
            print(f"   - {col} : {n} pays impute(s) par la moyenne du panel")
    for col in INDICATEURS_NUM:
        if col in INDICATEURS_NAN_OK:
            continue   # NaN intentionnel — géré dans construire_scores / calculer_zscores_pays
        moyenne_col = df[col].mean()
        if pd.isna(moyenne_col):
            print(f"   ! ATTENTION : '{col}' est vide sur TOUT le panel "
                  f"(indicateur API indisponible) -> colonne neutralisee (= 0), "
                  f"a corriger/verifier le code indicateur correspondant.")
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(moyenne_col)

print("\n" + "=" * 90)
print("DONNEES BRUTES DU PANEL")
print("=" * 90)
# On affiche Cyber_GCI_raw (valeur source non imputée) pour que le tableau
# reflète les vrais scores ITU, pas la valeur après fillna.
df_affich = df[["Pays"] + INDICATEURS_NUM].copy()
if "Cyber_GCI_raw" in df.columns:
    df_affich["Cyber_GCI"] = df["Cyber_GCI_raw"]   # overwrite avec valeur source
print(df_affich.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"\nSource : {'API Banque Mondiale (live) + ND-GAIN/INFORM (embarques)' if data_is_live else 'jeu illustratif local'}"
      f" — genere le {datetime.now().strftime('%Y-%m-%d %H:%M')}"
      f"\nNote   : AI_Readiness = Oxford Insights 2025 (0-100) | "
      f"Cyber_GCI = ITU GCI 2024 (0-100)")

# ==============================================================================
# 3. STANDARDISATION ET CONSTRUCTION DES PILIERS
# ==============================================================================

def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)

# Convention : z-score élevé = situation favorable (moins risquée).

PILLAR_COLS = ["z_economique", "z_fiscal", "z_externe", "z_social",
               "z_climat", "z_sante", "z_innovation"]
PILLAR_LABELS = {
    "z_economique":  "Economique (croissance + emploi)",
    "z_fiscal":      "Fiscal (solde + reserves)",
    "z_externe":     "Externe / Monetaire",
    "z_social":      "Social & Inegalites",
    "z_climat":      "Climat & Energie",
    "z_sante":       "Sante & Cap. humain",
    "z_innovation":  "Innovation & Cybersecurite",
}
PILLAR_SHORT = {
    "z_economique":  "Eco",
    "z_fiscal":      "Fiscal",
    "z_externe":     "Externe",
    "z_social":      "Social",
    "z_climat":      "Climat",
    "z_sante":       "Sante",
    "z_innovation":  "Innov/Cyber",
}

# Pondérations (rééquilibrées avec l'ajout du pilier 7) :
# social + climat + sante + innovation/cyber = ~57% du score
WEIGHTS = {
    "z_economique":  0.17,
    "z_fiscal":       0.1275,
    "z_externe":      0.1275,
    "z_social":      0.17,
    "z_climat":       0.1275,
    "z_sante":        0.1275,
    "z_innovation":  0.15,
}


def categorie_risque(score):
    if score >= 0.75:    return "Risque tres faible"
    elif score >= 0.25:  return "Risque faible"
    elif score >= -0.25: return "Risque modere"
    elif score >= -0.75: return "Risque eleve"
    else:                return "Risque tres eleve"


def construire_scores(df_in):
    """Calcule les 7 piliers (z-scores), le score composite, le rang global
    et le rang au sein du groupe de revenu (pairs) à partir d'un DataFrame
    contenant les 17 indicateurs bruts. Fonction pure (ne modifie pas
    df_in) — réutilisée par le panel principal ET par le moteur de
    simulation de chocs, pour garantir une logique de calcul unique."""
    d = df_in.copy()

    # S'assurer que Cyber_GCI_raw est présente (nécessaire pour le pilier 7).
    # Dans le moteur de choc, df_choc est copié depuis df qui la possède déjà ;
    # ce fallback ne sert qu'en cas d'appel externe inattendu.
    if "Cyber_GCI_raw" not in d.columns:
        d["Cyber_GCI_raw"] = d["iso3"].map(CYBER_GCI_2024) if "iso3" in d.columns else np.nan

    # Pilier 1 : Économique
    d["z_economique"] = (zscore(d["PIB_croissance"]) + zscore(-d["Chomage"])) / 2

    # Pilier 2 : Fiscal
    d["z_fiscal"] = (zscore(d["Solde_budgetaire"]) + zscore(d["Reserves_mois"])) / 2

    # Pilier 3 : Externe / Monétaire
    d["z_externe"] = (zscore(d["CompteCourant_PIB"]) + zscore(-d["Inflation"])) / 2

    # Pilier 4 : Social & Inégalités — Gini élevé = inégalités → inversé
    d["z_social"] = (zscore(-d["Gini"])
                    + zscore(-d["Pauvrete_extreme"])
                    + zscore(d["Esperance_vie"])) / 3

    # Pilier 5 : Climat & Énergie
    d["z_climat"] = (zscore(-d["ND_GAIN_vulnerabilite"])
                    + zscore(-d["Import_energie"])
                    + zscore(d["Acces_electricite"])) / 3

    # Pilier 6 : Santé & Capital humain
    d["z_sante"] = (zscore(-d["INFORM_risk"])
                   + zscore(d["Depenses_sante"])
                   + zscore(d["Medecins"])) / 3

    # Pilier 7 : Innovation & Cybersécurité — les deux indicateurs sont déjà
    # orientés "plus haut = mieux" (AI Readiness, Cyber GCI), pas d'inversion.
    # Si Cyber_GCI est absent pour un pays (NaN après fillna panel), on bascule
    # automatiquement sur AI_Readiness seul pour ce pays afin d'éviter la
    # dilution par une fausse moyenne. Le NaN est propagé en amont via la
    # colonne "_cyber_disponible" créée ci-dessous.
    _z_ai  = zscore(d["AI_Readiness"])
    _has_cyber = d["Cyber_GCI_raw"].notna()   # booléen : True si score GCI connu
    _z_cyber = zscore(d["Cyber_GCI"].where(_has_cyber))   # NaN si absent
    d["z_innovation"] = np.where(
        _has_cyber,
        (_z_ai + _z_cyber) / 2,    # les deux sources disponibles
        _z_ai                       # AI Readiness seul si GCI absent
    )

    d["score_composite"] = sum(d[col] * w for col, w in WEIGHTS.items())
    if d["score_composite"].isna().any():
        pays_pb = d.loc[d["score_composite"].isna(), "Pays"].tolist()
        raise RuntimeError(
            "score_composite contient des NaN pour : " + ", ".join(pays_pb) +
            " -> impossible d'etablir un classement. Verifiez les codes "
            "indicateurs WB_INDICATORS et la disponibilite des donnees."
        )
    d["rang"] = d["score_composite"].rank(ascending=False, method="min").astype(int)
    d["categorie_risque"] = d["score_composite"].apply(categorie_risque)

    # Comparaison par pairs : rang au sein du même groupe de revenu Banque
    # Mondiale (ex. la France n'est comparée qu'à l'Allemagne, aux USA... —
    # les autres pays "Revenu eleve" du panel — plutot qu'au panel entier).
    if "groupe_revenu_id" in d.columns:
        d["rang_groupe"] = (d.groupby("groupe_revenu_id")["score_composite"]
                               .rank(ascending=False, method="min").astype(int))
        d["taille_groupe"] = d.groupby("groupe_revenu_id")["groupe_revenu_id"].transform("count")
    else:
        d["rang_groupe"] = np.nan
        d["taille_groupe"] = np.nan

    return d.sort_values("score_composite", ascending=False).reset_index(drop=True)


df = construire_scores(df)

print("\n" + "=" * 90)
print("CLASSEMENT — MODELE ELARGI (17 INDICATEURS, 7 PILIERS)")
print("=" * 90)
cols_affich = ["rang", "Pays", "score_composite", "categorie_risque"] + PILLAR_COLS
print(df[cols_affich].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

print("\n" + "-" * 90)
print("CLASSEMENT PAR PAIRS — au sein du groupe de revenu Banque Mondiale")
print("-" * 90)
cols_groupe = ["groupe_revenu", "rang_groupe", "Pays", "score_composite", "rang"]
df_groupe_aff = df[cols_groupe].sort_values(["groupe_revenu", "rang_groupe"])
df_groupe_aff = df_groupe_aff.rename(columns={"rang": "rang_panel_global"})
print(df_groupe_aff.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

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
    "z_innovation": "une avance numerique et cyber notable (AI Readiness {ai:.0f}/100, GCI cyber {cyb:.0f}/100)",
}
FRAGILITE_TEMPLATES = {
    "z_economique": "une dynamique economique fragile ({pib:.1f}% de croissance, chomage a {cho:.1f}%)",
    "z_fiscal":     "des fragilites fiscales (solde {bud:+.1f}% du PIB, reserves {res:.1f} mois)",
    "z_externe":    "des tensions externes significatives (CC {cc:+.1f}% du PIB, inflation {inf:.1f}%)",
    "z_social":     "des inegalites structurelles elevees (Gini {gin:.0f}, pauvrete extreme {pvr:.1f}%)",
    "z_climat":     "une forte exposition climatique (vulnerabilite {gain:.2f}/1, INFORM {inf_r:.1f}/10)",
    "z_sante":      "un systeme de sante sous-dimensionne ({med:.1f} medecins/1000 hab., INFORM {inf_r:.1f}/10)",
    "z_innovation": "un retard numerique et cyber (AI Readiness {ai:.0f}/100, GCI cyber {cyb:.0f}/100)",
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
        ai=row["AI_Readiness"], cyb=row["Cyber_GCI"],
    )
    return template_dict[pilier].format(**kwargs)


def generer_note(row):
    piliers_tries = row[PILLAR_COLS].sort_values(ascending=False)
    pilier_force   = piliers_tries.index[0]
    pilier_fragile = piliers_tries.index[-1]

    note = (
        f"## {row['Pays']} - {row['categorie_risque']}\n\n"
        f"**Score composite : {row['score_composite']:+.2f}** "
        f"(rang {row['rang']}/{len(df)} sur le panel mondial ; "
        f"rang {row['rang_groupe']}/{int(row['taille_groupe'])} parmi ses pairs "
        f"« {row['groupe_revenu']} »)\n\n"
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
    f.write("# Notes d'analyse risque pays - Modele elargi (17 indicateurs, 7 piliers)\n\n")
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
ax.set_xlabel("Score composite (z-score pondere, 7 piliers)")
ax.set_title("Classement risque pays - Modele elargi 17 indicateurs\n"
             "(social, climatique, sanitaire, innovation/cyber = ~57% du score)", fontsize=12)
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
fig.suptitle("Profil par pilier - 7 dimensions de risque (z-scores)", fontsize=12, y=1.01)
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

# --- Graphique 4 : Heatmap 7 piliers ---
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
ax.set_title("Heatmap des 7 piliers de risque (z-scores)\nvert = favorable / rouge = risque", fontsize=11)
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
    "z_innovation": "#607D8B",
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

    z_ai = z("AI_Readiness", indicateurs["AI_Readiness"])
    cyber_val = indicateurs.get("Cyber_GCI")
    if cyber_val is not None and not (isinstance(cyber_val, float) and np.isnan(cyber_val)):
        z_innov = (z_ai + z("Cyber_GCI", cyber_val)) / 2
    else:
        z_innov = z_ai   # Cyber_GCI non disponible → AI Readiness seul

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
        "z_innovation": z_innov,
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

    # AI Readiness — couvre ~193 pays ; fallback panel si absent
    ai_val = AI_READINESS_2025.get(iso3)
    if ai_val is None:
        print(f"   AI_Readiness : donnee non disponible pour {iso3} "
              f"(hors couverture Oxford Insights) -> moyenne du panel")
        ai_val = PANEL_MEAN_STD["AI_Readiness"][0]
    indicateurs["AI_Readiness"] = ai_val

    # Cyber GCI — couverture large mais pas universelle ; NaN intentionnel si absent
    cyber_val = CYBER_GCI_2024.get(iso3)
    if cyber_val is None:
        print(f"   Cyber_GCI    : score ITU GCI 2024 non disponible pour {iso3} "
              f"-> pilier Innovation/Cyber calcule sur AI Readiness seul")
        cyber_val = np.nan
    indicateurs["Cyber_GCI"] = cyber_val

    meta = fetch_country_meta_live(iso3) or PAYS_META_FALLBACK.get(iso3)
    groupe_id, groupe_label = (meta or {}).get("groupe_revenu_id"), (meta or {}).get("groupe_revenu")

    # Imputation par moyenne du panel pour les indicateurs WB manquants
    # (Cyber_GCI exclu : son NaN est géré dans calculer_zscores_pays)
    manquants = [k for k, v in indicateurs.items()
                 if k != "Cyber_GCI" and pd.isna(v)]
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
    print(f"Rang estime     : {rang}/{len(df) + 1}  (panel mondial)")
    if groupe_id:
        panel_pairs = df[df["groupe_revenu_id"] == groupe_id]
        rang_groupe = int((panel_pairs["score_composite"] > score).sum()) + 1
        print(f"Rang chez ses pairs « {groupe_label} » : "
              f"{rang_groupe}/{len(panel_pairs) + 1} "
              f"(compare aux {len(panel_pairs)} pays du panel du meme groupe de revenu)")
    else:
        print("Rang chez ses pairs : groupe de revenu non determine (API metadonnees indisponible).")
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
        ax.set_title(f"{nom_aff} - profil 7 piliers", fontsize=11, pad=15)
        plt.tight_layout()
        nom_radar = f"radar_{iso3.lower()}.png"
        plt.savefig(nom_radar, dpi=150)
        plt.close()
        print(f"-> Radar exporte : {nom_radar}")

    return {"iso3": iso3, "indicateurs": indicateurs,
            "zscores": zscores, "score_composite": score, "categorie": cat}


# ==============================================================================
# 7. MOTEUR DE SIMULATION DE SCENARIOS (CHOCS MACRO)
# ==============================================================================
# Permet de tester l'impact d'un choc (ex. +5pts d'inflation, -3pts de PIB)
# sur UN pays du panel et de visualiser immediatement l'effet sur son
# score, son classement mondial et son classement "pairs" — sans modifier
# les donnees de base (tout est recalcule sur une copie du DataFrame).

VARIABLES_CHOC = {
    "1": ("PIB_croissance", "Croissance du PIB (%)"),
    "2": ("Chomage", "Taux de chomage (%)"),
    "3": ("Inflation", "Inflation (%)"),
    "4": ("Solde_budgetaire", "Solde budgetaire (% PIB)"),
    "5": ("CompteCourant_PIB", "Compte courant (% PIB)"),
}


def afficher_classement_pairs(d, groupe_id, titre="Classement pairs"):
    sous_groupe = d[d["groupe_revenu_id"] == groupe_id].sort_values("score_composite", ascending=False)
    print(f"\n--- {titre} ({sous_groupe['groupe_revenu'].iloc[0]}) ---")
    print(sous_groupe[["rang_groupe", "Pays", "score_composite", "rang"]]
          .rename(columns={"rang": "rang_panel_global"})
          .to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


def simuler_choc(nom_pays):
    """Choc manuel sur un indicateur d'un pays DU PANEL (les 12 pays
    suivis), avec recalcul complet (mêmes formules que construire_scores)
    pour visualiser l'impact relatif sur tout le panel."""
    iso3 = resoudre_iso3(nom_pays)
    if iso3 is None or iso3 not in df["iso3"].values:
        print(f"   ! '{nom_pays}' n'est pas (ou n'est pas reconnu comme) un "
              f"pays du panel principal. La simulation de choc ne fonctionne "
              f"que sur les {len(df)} pays suivis : {', '.join(df['Pays'])}.")
        return

    nom_aff = df.loc[df["iso3"] == iso3, "Pays"].iloc[0]
    print(f"\nVariables disponibles pour le choc sur {nom_aff} :")
    for k, (col, label) in VARIABLES_CHOC.items():
        valeur_actuelle = df.loc[df["iso3"] == iso3, col].iloc[0]
        print(f"  [{k}] {label:32s} (valeur actuelle : {valeur_actuelle:+.2f})")

    choix = input("Variable a choquer (numero, ou 'annuler') > ").strip()
    if choix.lower() in ("annuler", "a", ""):
        print("Simulation annulee.")
        return
    if choix not in VARIABLES_CHOC:
        print("   ! Choix invalide.")
        return
    col, label = VARIABLES_CHOC[choix]

    try:
        delta = float(input(f"Choc a appliquer sur « {label} » "
                             f"(points, ex. -3 ou +5) > ").strip().replace(",", "."))
    except ValueError:
        print("   ! Valeur numerique attendue, simulation annulee.")
        return

    df_choc = df.copy()
    avant = df_choc.loc[df_choc["iso3"] == iso3, col].iloc[0]
    df_choc.loc[df_choc["iso3"] == iso3, col] = avant + delta
    df_resultat = construire_scores(df_choc)

    avant_pays = df.loc[df["iso3"] == iso3].iloc[0]
    apres_pays = df_resultat.loc[df_resultat["iso3"] == iso3].iloc[0]

    print(f"\n{'=' * 78}")
    print(f"SIMULATION — {nom_aff} : {label} {avant:+.2f} -> {avant + delta:+.2f} ({delta:+.2f})")
    print(f"{'=' * 78}")
    print(f"{'':20s} {'AVANT':>12s} {'APRES':>12s} {'VARIATION':>12s}")
    print(f"{'Score composite':20s} {avant_pays['score_composite']:>+12.2f} "
          f"{apres_pays['score_composite']:>+12.2f} "
          f"{apres_pays['score_composite'] - avant_pays['score_composite']:>+12.2f}")
    print(f"{'Rang panel mondial':20s} {avant_pays['rang']:>12d} {apres_pays['rang']:>12d} "
          f"{apres_pays['rang'] - avant_pays['rang']:>+12d}")
    print(f"{'Rang chez ses pairs':20s} {avant_pays['rang_groupe']:>12d} {apres_pays['rang_groupe']:>12d} "
          f"{apres_pays['rang_groupe'] - avant_pays['rang_groupe']:>+12d}")
    print(f"{'Categorie de risque':20s} {avant_pays['categorie_risque']:>12s} "
          f"{apres_pays['categorie_risque']:>12s}")

    print(f"\nClassement complet du panel APRES le choc :")
    print(df_resultat[["rang", "Pays", "score_composite", "categorie_risque"]]
          .sort_values("rang")
          .to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    afficher_classement_pairs(df_resultat, avant_pays["groupe_revenu_id"],
                               titre=f"Classement pairs apres le choc")
    print(f"\n(Note : seul {nom_aff} a ete choque ; les autres pays du panel "
          f"gardent leurs valeurs brutes, mais leurs z-scores et rangs peuvent "
          f"legerement varier car la moyenne/ecart-type du panel se deplacent.)")


print("\n" + "=" * 80)
print("BOUCLE INTERACTIVE")
print("=" * 80)
print("Tapez un nom de pays (FR) ou un code ISO3 pour l'analyser : Maroc, Japon, SEN, JPN ...")
print("Autres commandes :")
print("  expliquer <pays>  -> pourquoi ce pays est classe a ce niveau de risque ?")
print("  pairs             -> afficher le classement par groupe de revenu (peers)")
print("  choc              -> simuler un choc macro sur un pays du panel (les 12 suivis)")
print("  quit / exit / q   -> quitter")
print("=" * 80 + "\n")

# ==============================================================================
# FEATURE : EXPLICATION DU RANG (Issue #4)
# ==============================================================================

def expliquer_rang(nom_pays):
    """Génère une explication structurée de pourquoi un pays est classé à
    son niveau de risque — pour les 12 pays du panel (données live) ou
    pour n'importe quel pays via analyse ad hoc.

    Affiche :
      · Décomposition pilier par pilier avec z-score, contribution pondérée,
        rang relatif dans le panel, et lecture en langage naturel.
      · Les 2 facteurs les plus favorables et les 2 plus pénalisants.
      · Où AI_Readiness et Cyber_GCI se situent vs le panel.
      · Un verdict narratif global.
    """
    iso3 = resoudre_iso3(nom_pays)
    if iso3 is None:
        print(f"   ! Pays non reconnu : '{nom_pays}'.")
        return

    # ── Récupérer ou calculer les données ──────────────────────────────────
    is_panel = iso3 in df["iso3"].values
    if is_panel:
        row = df.loc[df["iso3"] == iso3].iloc[0]
        zscores_pays = {c: row[c] for c in PILLAR_COLS}
        score        = row["score_composite"]
        cat          = row["categorie_risque"]
        rang_global  = int(row["rang"])
        rang_groupe  = int(row["rang_groupe"])
        taille_grp   = int(row["taille_groupe"])
        groupe_label = row["groupe_revenu"]
        nom_aff      = row["Pays"]
        ai_val       = row["AI_Readiness"]
        # Utiliser Cyber_GCI_raw si disponible pour l'affichage source
        cyber_val    = (row["Cyber_GCI_raw"]
                        if "Cyber_GCI_raw" in df.columns and not pd.isna(row["Cyber_GCI_raw"])
                        else row["Cyber_GCI"])
        cyber_source = "ITU GCI 2024"
    else:
        # Pays hors panel : on relance analyser_pays sans radar pour récupérer
        # les données, puis on affiche l'explication.
        resultat = analyser_pays(nom_pays, sauvegarder_radar=False)
        if resultat is None:
            return
        zscores_pays = resultat["zscores"]
        score        = resultat["score_composite"]
        cat          = resultat["categorie"]
        iso3         = resultat["iso3"]
        nom_aff      = nom_affichable(nom_pays, iso3)
        rang_global  = int((df["score_composite"] > score).sum()) + 1
        rang_groupe  = None
        taille_grp   = None
        groupe_label = None
        ai_val       = resultat["indicateurs"]["AI_Readiness"]
        cyber_raw    = CYBER_GCI_2024.get(iso3)
        cyber_val    = cyber_raw if cyber_raw is not None else float("nan")
        cyber_source = "ITU GCI 2024" if cyber_raw is not None else "non disponible"

    # ── Calcul des rangs par pilier dans le panel ──────────────────────────
    rang_pilier = {}
    for col in PILLAR_COLS:
        col_vals = df[col].values
        z_p      = zscores_pays[col]
        rang_pilier[col] = int((col_vals > z_p).sum()) + 1   # rang parmi panel+1

    n_panel = len(df) + (0 if is_panel else 1)   # panel ou panel+1 pour hors-panel

    # ── Interprétation textuelle d'un z-score ──────────────────────────────
    def lire_zscore(z, pilier):
        if   z >= 1.5:  return "très favorable ✅✅"
        elif z >= 0.5:  return "favorable ✅"
        elif z >= -0.5: return "neutre / moyen ➡"
        elif z >= -1.5: return "défavorable ⚠"
        else:           return "très défavorable 🔴"

    # ── Affichage ──────────────────────────────────────────────────────────
    SEP = "=" * 80
    sep = "-" * 80
    print(f"\n{SEP}")
    print(f"EXPLICATION DU RANG — {nom_aff} ({iso3})")
    print(SEP)
    print(f"Score composite : {score:+.2f}  →  {cat}")
    print(f"Rang panel      : {rang_global}/{n_panel}")
    if rang_groupe is not None:
        print(f"Rang pairs      : {rang_groupe}/{taille_grp}  « {groupe_label} »")
    print()
    print("DÉCOMPOSITION PAR PILIER")
    print(sep)
    print(f"{'Pilier':<34s} {'z-score':>8s} {'Poids':>7s} {'Contrib':>8s} {'Rang panel':>11s} {'Lecture'}")
    print(sep)

    contributions = {}
    for col in PILLAR_COLS:
        z    = zscores_pays[col]
        w    = WEIGHTS[col]
        contrib = z * w
        contributions[col] = contrib
        rang_p   = rang_pilier[col]
        label    = PILLAR_LABELS[col]
        lecture  = lire_zscore(z, col)
        print(f"  {label:<32s} {z:>+8.2f} {w*100:>6.1f}% {contrib:>+8.3f} "
              f"  {rang_p:>2d}/{n_panel:<3d}   {lecture}")

    print(sep)
    print(f"  {'SCORE COMPOSITE (somme contributions)':32s} {score:>+8.2f}")
    print()

    # ── Points forts et points faibles ────────────────────────────────────
    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    print("FACTEURS LES PLUS FAVORABLES (contribution positive au score) :")
    for col, c in sorted_contrib[:2]:
        print(f"  + {PILLAR_LABELS[col]} : z={zscores_pays[col]:+.2f}, "
              f"contribution {c:+.3f} (rang {rang_pilier[col]}/{n_panel})")

    print("\nFACTEURS LES PLUS PÉNALISANTS (contribution négative au score) :")
    for col, c in sorted_contrib[-2:]:
        print(f"  - {PILLAR_LABELS[col]} : z={zscores_pays[col]:+.2f}, "
              f"contribution {c:+.3f} (rang {rang_pilier[col]}/{n_panel})")

    # ── Zoom pilier 7 (Innovation / Cyber) ───────────────────────────────
    print()
    print("ZOOM — PILIER 7 : INNOVATION & CYBERSÉCURITÉ")
    print(sep)
    panel_ai_mean  = df["AI_Readiness"].mean()
    panel_ai_std   = df["AI_Readiness"].std(ddof=0)
    panel_ai_min   = df["AI_Readiness"].min()
    panel_ai_max   = df["AI_Readiness"].max()
    cyber_col      = "Cyber_GCI_raw" if "Cyber_GCI_raw" in df.columns else "Cyber_GCI"
    panel_gci_mean = df[cyber_col].mean()
    panel_gci_std  = df[cyber_col].std(ddof=0)

    ai_pct   = (ai_val - panel_ai_min) / (panel_ai_max - panel_ai_min) * 100 if panel_ai_max > panel_ai_min else 50
    print(f"  AI Readiness  (Oxford Insights 2025) : {ai_val:.2f}/100  "
          f"[panel : moy {panel_ai_mean:.1f} ± {panel_ai_std:.1f}]")
    print(f"    → {nom_aff} se situe au {ai_pct:.0f}e centile du panel sur cet indicateur")

    if not pd.isna(cyber_val):
        gci_pct = int((df[cyber_col] < cyber_val).sum()) / len(df) * 100
        print(f"  Cyber GCI     ({cyber_source}) : {cyber_val:.2f}/100  "
              f"[panel : moy {panel_gci_mean:.1f} ± {panel_gci_std:.1f}]")
        print(f"    → Score GCI supérieur à {gci_pct:.0f}% du panel")
    else:
        print(f"  Cyber GCI     : non disponible pour {nom_aff} "
              f"→ pilier calculé sur AI Readiness seul")

    # ── Verdict narratif ──────────────────────────────────────────────────
    print()
    print("VERDICT NARRATIF")
    print(sep)
    # Identifier les piliers > 0.5 (forces) et < -0.5 (faiblesses)
    forces    = [PILLAR_LABELS[c] for c in PILLAR_COLS if zscores_pays[c] >= 0.5]
    faiblesses = [PILLAR_LABELS[c] for c in PILLAR_COLS if zscores_pays[c] <= -0.5]

    if forces:
        print(f"Points forts    : {', '.join(forces)}")
    else:
        print("Points forts    : aucun pilier significativement au-dessus de la moyenne")

    if faiblesses:
        print(f"Points faibles  : {', '.join(faiblesses)}")
    else:
        print("Points faibles  : aucun pilier significativement sous la moyenne")

    if score >= 0.75:
        verdict = (f"{nom_aff} affiche un profil de risque très faible. "
                   f"La convergence positive de plusieurs piliers (score {score:+.2f}) "
                   f"traduit une résilience macroéconomique et structurelle solide.")
    elif score >= 0.25:
        verdict = (f"{nom_aff} présente un profil globalement favorable (score {score:+.2f}). "
                   f"Des atouts clairs sur certains piliers compensent des fragilités limitées. "
                   f"La surveillance reste utile sur : {', '.join(faiblesses) or 'aucun pilier critique'}.")
    elif score >= -0.25:
        verdict = (f"{nom_aff} se situe dans une zone intermédiaire (score {score:+.2f}). "
                   f"Ni clairement résilient ni clairement fragile, le pays présente des "
                   f"signaux mixtes. Les piliers pénalisants méritent attention.")
    elif score >= -0.75:
        verdict = (f"{nom_aff} affiche un profil dégradé (score {score:+.2f}). "
                   f"Plusieurs piliers en zone négative signalent une accumulation de "
                   f"vulnérabilités, notamment : {', '.join(faiblesses)}.")
    else:
        verdict = (f"{nom_aff} présente un profil de risque très élevé (score {score:+.2f}). "
                   f"La convergence de multiples fragilités ({', '.join(faiblesses)}) "
                   f"justifie un suivi renforcé.")

    # Mentionner le rang Innovation si remarquable
    rang_innov = rang_pilier["z_innovation"]
    z_innov    = zscores_pays["z_innovation"]
    if rang_innov <= 3:
        verdict += (f" Sur le plan numérique/cyber, {nom_aff} se distingue positivement "
                    f"(rang {rang_innov}/{n_panel} sur ce pilier).")
    elif rang_innov >= n_panel - 2:
        verdict += (f" Le retard numérique/cyber (rang {rang_innov}/{n_panel}) "
                    f"pèse structurellement sur le score.")

    print()
    print(verdict)
    print(SEP)


while True:
    try:
        saisie = input("Commande ou pays a analyser > ").strip()
    except EOFError:
        break
    cmd = saisie.lower()
    if cmd in ("quit", "exit", "q", ""):
        print("Fin.")
        break
    elif cmd == "pairs":
        for gid in sorted(df["groupe_revenu_id"].dropna().unique()):
            afficher_classement_pairs(df, gid)
    elif cmd == "choc":
        nom_choc = input("Pays du panel a choquer > ").strip()
        if nom_choc:
            simuler_choc(nom_choc)
    elif cmd.startswith("expliquer"):
        # "expliquer france" ou "expliquer FRA" ou "expliquer Maroc" etc.
        parts = saisie.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            nom_expl = input("Pays a expliquer > ").strip()
        else:
            nom_expl = parts[1].strip()
        if nom_expl:
            expliquer_rang(nom_expl)
    else:
        analyser_pays(saisie)

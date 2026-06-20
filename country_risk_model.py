"""
================================================================================
SCORING RISQUE PAYS & GENERATION AUTOMATISEE DE NOTES D'ANALYSE
================================================================================
Objectif : construire un score composite de risque souverain à partir
d'indicateurs macroéconomiques publics, classer un panel de pays, et générer
automatiquement une note d'analyse texte par pays (NLG simple à partir de
données structurées).

Méthodologie (inspirée, de façon très simplifiée, des piliers utilisés par
les agences de notation souveraines — sans reproduire leur méthodologie
propriétaire) :

  1. Récupération des données macro (API Banque Mondiale, gratuite, sans clé)
     -> avec repli automatique sur un jeu de données illustratif si l'accès
        réseau n'est pas disponible (cf. avertissement ci-dessous)
  2. Standardisation (z-scores) de chaque indicateur par rapport au panel
  3. Agrégation en 4 piliers : Économique / Fiscal / Externe / Monétaire
  4. Score composite + classement + catégorie de risque illustrative
  5. Génération automatique d'une note texte par pays (force/fragilité
     identifiées à partir des données, pas de texte rédigé à la main)
  6. Visualisations : classement, radar par pays, carte de positionnement,
     heatmap des piliers

NOTE DE VERSION : le pilier "institutionnel" (gouvernance, indicateurs WGI
CC.EST / RL.EST / GE.EST) a été retiré dans cette version, car ces codes
d'indicateur renvoient une erreur "indicator not found" sur l'API Banque
Mondiale actuelle (probablement déplacés/archivés côté Banque Mondiale). Le
modèle fonctionne donc ici sur 4 piliers. Piste d'extension : retrouver les
codes d'indicateur WGI à jour (catalogue API Banque Mondiale) pour
réintroduire un pilier institutionnel.

⚠️ AVERTISSEMENT IMPORTANT
Les données macro utilisées en repli (si l'API est inaccessible) sont des
VALEURS ILLUSTRATIVES APPROXIMATIVES (ordres de grandeur) et NE DOIVENT PAS
être utilisées pour une décision réelle.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from datetime import datetime

USE_LIVE_API = True  # passer à False pour forcer l'utilisation du jeu illustratif

# ==============================================================================
# 1. RECUPERATION DES DONNEES
# ==============================================================================

# Panel de pays étudié (code ISO3 -> nom)
COUNTRIES = {
    "USA": "Etats-Unis", "DEU": "Allemagne", "FRA": "France", "BRA": "Bresil",
    "TUR": "Turquie", "ZAF": "Afrique du Sud", "EGY": "Egypte", "IND": "Inde",
    "IDN": "Indonesie", "ARG": "Argentine", "NGA": "Nigeria", "VNM": "Vietnam",
}

# Indicateurs Banque Mondiale utilisés (gratuits, sans clé API)
WB_INDICATORS = {
    "PIB_croissance": "NY.GDP.MKTP.KD.ZG",      # croissance du PIB, % annuel
    "Inflation": "FP.CPI.TOTL.ZG",                # inflation, prix à la consommation, %
    "Dette_PIB": "GC.DOD.TOTL.GD.ZS",             # dette publique centrale, % du PIB
    "CompteCourant_PIB": "BN.CAB.XOKA.GD.ZS",     # solde du compte courant, % du PIB
}


def fetch_indicator_live(iso3, code, date_range="2018:2023"):
    """Interroge l'API Banque Mondiale (gratuite, sans clé) pour un pays et un
    indicateur donnés, renvoie la dernière valeur disponible."""
    import requests
    url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/{code}"
    r = requests.get(url, params={"format": "json", "date": date_range, "per_page": 100}, timeout=30)
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
    """Construit le DataFrame en interrogeant l'API Banque Mondiale pour
    chaque pays x indicateur. Lève une exception si l'API n'est pas
    accessible (réseau bloqué, etc.) -> le script bascule alors sur le jeu
    de données illustratif."""
    rows = []
    for iso3, name in COUNTRIES.items():
        row = {"Pays": name, "iso3": iso3}
        for label, code in WB_INDICATORS.items():
            row[label] = fetch_indicator_live(iso3, code)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.drop(columns=["Pays", "iso3"]).isna().all(axis=None):
        raise RuntimeError("Aucune donnée renvoyée par l'API")
    return df


def fallback_dataset():
    """Jeu de données illustratif (ordres de grandeur approximatifs, non
    sourcés en direct) utilisé uniquement si l'API Banque Mondiale n'est pas
    accessible."""
    data = [
        # Pays,            iso3, PIB%,  Inflation%, Dette/PIB, CC/PIB
        ("Etats-Unis",     "USA", 2.5,  3.0,   122,  -3.0),
        ("Allemagne",      "DEU", 0.2,  2.5,    64,   6.0),
        ("France",         "FRA", 1.0,  2.5,   110,  -1.0),
        ("Bresil",         "BRA", 2.5,  4.5,    85,  -1.5),
        ("Turquie",        "TUR", 3.0, 45.0,    30,  -2.5),
        ("Afrique du Sud", "ZAF", 0.8,  5.0,    75,  -2.0),
        ("Egypte",         "EGY", 3.5, 30.0,    90,  -2.0),
        ("Inde",           "IND", 7.0,  5.0,    82,  -1.0),
        ("Indonesie",      "IDN", 5.0,  3.0,    39,  -0.5),
        ("Argentine",      "ARG",-1.5,150.0,    90,   0.5),
        ("Nigeria",        "NGA", 3.0, 28.0,    45,   0.5),
        ("Vietnam",        "VNM", 6.0,  3.5,    37,   4.0),
    ]
    df = pd.DataFrame(data, columns=["Pays", "iso3", "PIB_croissance", "Inflation",
                                       "Dette_PIB", "CompteCourant_PIB"])
    return df


data_is_live = False
if USE_LIVE_API:
    try:
        df = fetch_live_dataset()
        data_is_live = True
        print("Données récupérées en direct depuis l'API Banque Mondiale.")
    except Exception as e:
        print(f"API Banque Mondiale inaccessible ({e}). Repli sur le jeu de données illustratif.")
        df = fallback_dataset()
else:
    df = fallback_dataset()

if not data_is_live:
    print("\n*** ATTENTION : données ILLUSTRATIVES (ordres de grandeur), pas de source ***")
    print("*** live confirmée — ne pas utiliser pour une analyse réelle.            ***\n")

# Sécurité : certains pays/années peuvent renvoyer des valeurs manquantes
# (NaN) même en mode live. On les remplace par la moyenne du panel pour cet
# indicateur, plutôt que de planter -> à mentionner explicitement si on
# présente ce projet (imputation simple, pas idéale, mais transparente).
indicateurs_num = ["PIB_croissance", "Inflation", "Dette_PIB", "CompteCourant_PIB"]
nb_manquants = df[indicateurs_num].isna().sum().sum()
if nb_manquants > 0:
    print(f"\n*** {nb_manquants} valeur(s) manquante(s) détectée(s) -> imputées par la moyenne du panel ***\n")
    for col in indicateurs_num:
        df[col] = df[col].fillna(df[col].mean())

print(df.to_string(index=False))
print(f"\nSource : {'API Banque Mondiale (live)' if data_is_live else 'jeu illustratif local'}"
      f" — généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ==============================================================================
# 2. STANDARDISATION (Z-SCORES) ET PILIERS
# ==============================================================================

def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)

# Chaque pilier est orienté de sorte qu'un score plus élevé = situation plus
# favorable (moins risquée). On inverse donc les variables où une valeur
# élevée est défavorable (dette, inflation).
df["z_economique"] = zscore(df["PIB_croissance"])
df["z_fiscal"] = zscore(-df["Dette_PIB"])
df["z_externe"] = zscore(df["CompteCourant_PIB"])
df["z_monetaire"] = zscore(-df["Inflation"])

PILLAR_COLS = ["z_economique", "z_fiscal", "z_externe", "z_monetaire"]
PILLAR_LABELS = {
    "z_economique": "Économique (croissance)",
    "z_fiscal": "Fiscal (endettement public)",
    "z_externe": "Externe (compte courant)",
    "z_monetaire": "Monétaire (inflation)",
}

# Pondérations des piliers dans le score composite (modifiable).
WEIGHTS = {col: 0.25 for col in PILLAR_COLS}

df["score_composite"] = sum(df[col] * w for col, w in WEIGHTS.items())
df["rang"] = df["score_composite"].rank(ascending=False, method="min").astype(int)
df = df.sort_values("score_composite", ascending=False).reset_index(drop=True)


def categorie_risque(score):
    """Catégorie de risque illustrative (échelle propre à ce modèle, ne
    reproduisant pas les échelles de notation des agences officielles)."""
    if score >= 0.75:
        return "Risque très faible"
    elif score >= 0.20:
        return "Risque faible"
    elif score >= -0.20:
        return "Risque modéré"
    elif score >= -0.75:
        return "Risque élevé"
    else:
        return "Risque très élevé"


df["categorie_risque"] = df["score_composite"].apply(categorie_risque)

print("\n" + "=" * 80)
print("CLASSEMENT RISQUE PAYS")
print("=" * 80)
print(df[["rang", "Pays", "score_composite", "categorie_risque"]].to_string(index=False))

df.to_csv("country_risk_scores.csv", index=False)
print("\n-> Résultats détaillés exportés dans country_risk_scores.csv")

# ==============================================================================
# 3. GENERATION AUTOMATIQUE DE NOTES D'ANALYSE (NLG A PARTIR DES DONNEES)
# ==============================================================================

FORCE_TEMPLATES = {
    "z_economique": "une dynamique de croissance favorable ({val:.1f}% de croissance du PIB)",
    "z_fiscal": "un niveau d'endettement public contenu ({val:.0f}% du PIB)",
    "z_externe": "une position extérieure solide (solde courant de {val:+.1f}% du PIB)",
    "z_monetaire": "une inflation maîtrisée ({val:.1f}%)",
}
FRAGILITE_TEMPLATES = {
    "z_economique": "une croissance économique atone ({val:.1f}%)",
    "z_fiscal": "un endettement public élevé ({val:.0f}% du PIB)",
    "z_externe": "une position extérieure fragile (solde courant de {val:+.1f}% du PIB)",
    "z_monetaire": "une inflation élevée ({val:.1f}%)",
}
VAL_COL = {
    "z_economique": "PIB_croissance", "z_fiscal": "Dette_PIB",
    "z_externe": "CompteCourant_PIB", "z_monetaire": "Inflation",
}


def generer_note(row):
    forces = row[PILLAR_COLS].sort_values(ascending=False)
    fragilites = row[PILLAR_COLS].sort_values()
    pilier_force = forces.index[0]
    pilier_fragile = fragilites.index[0]

    phrase_force = FORCE_TEMPLATES[pilier_force].format(val=row[VAL_COL[pilier_force]])
    phrase_fragile = FRAGILITE_TEMPLATES[pilier_fragile].format(val=row[VAL_COL[pilier_fragile]])

    note = (
        f"## {row['Pays']} — {row['categorie_risque']}\n\n"
        f"**Score composite : {row['score_composite']:+.2f}** (rang {row['rang']}/{len(df)} sur le panel étudié)\n\n"
        f"{row['Pays']} présente, sur la base des indicateurs retenus, {phrase_force}. "
        f"À l'inverse, le principal point de vigilance porte sur {phrase_fragile}. "
    )

    if row["score_composite"] >= 0.20:
        note += "Le profil de risque global ressort comme favorable au regard du panel de comparaison."
    elif row["score_composite"] <= -0.20:
        note += "Le profil de risque global ressort comme dégradé au regard du panel de comparaison, et justifie un suivi renforcé."
    else:
        note += "Le profil de risque global se situe dans une zone intermédiaire, sans signal de stress marqué ni de marge de sécurité confortable."

    note += (
        f"\n\n*Indicateurs : croissance du PIB {row['PIB_croissance']:.1f}% | "
        f"inflation {row['Inflation']:.1f}% | dette publique {row['Dette_PIB']:.0f}% du PIB | "
        f"compte courant {row['CompteCourant_PIB']:+.1f}% du PIB.*\n"
    )
    return note


with open("notes_pays.md", "w", encoding="utf-8") as f:
    f.write("# Notes d'analyse risque pays — générées automatiquement\n\n")
    f.write(f"*Panel de {len(df)} pays, source des données : "
            f"{'API Banque Mondiale (live)' if data_is_live else 'jeu illustratif local — à ne pas utiliser tel quel'}.*\n\n")
    f.write("---\n\n")
    for _, row in df.iterrows():
        f.write(generer_note(row))
        f.write("\n---\n\n")

print("-> Notes d'analyse générées automatiquement dans notes_pays.md")

# ==============================================================================
# 4. ANALYSE AD HOC D'UN PAYS AU CHOIX (N'IMPORTE LEQUEL DANS LE MONDE)
# ==============================================================================
# On garde en mémoire la moyenne/écart-type du panel de référence (les 12 pays
# ci-dessus) afin de pouvoir positionner N'IMPORTE QUEL AUTRE PAYS par rapport
# à cette même échelle, sans recalculer tout le panel à chaque fois.

import unicodedata

PANEL_MEAN_STD = {col: (df[col].mean(), df[col].std(ddof=0)) for col in indicateurs_num}


def normaliser_texte(s):
    """Retire les accents et met en minuscules, pour une recherche tolérante
    aux fautes de frappe/accents ('Sénégal', 'senegal', 'Senegal' -> identique)."""
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s


# Dictionnaire nom français (normalisé) -> code ISO3. Couvre la quasi-totalité
# des pays reconnus par l'ONU. Liste non exhaustive à 100% (territoires et cas
# limites) -> à compléter si besoin (cf. README, section limites).
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
    """Normalisation complète pour la recherche : sans accents, sans
    apostrophes/tirets (remplacés par des espaces), espaces multiples
    réduits à un seul. Permet de faire correspondre 'Côte d'Ivoire',
    'cote-d-ivoire' et 'Côte d Ivoire' à la même clé."""
    s = normaliser_texte(s)
    s = s.replace("'", " ").replace("-", " ")
    return " ".join(s.split())


def resoudre_iso3(nom_pays):
    """Convertit un nom de pays saisi en français (ou un code ISO3 à 3
    lettres) en code ISO3 standard. Renvoie None si le pays n'est pas
    reconnu."""
    saisie = nom_pays.strip()
    if len(saisie) == 3 and saisie.isalpha():
        return saisie.upper()
    nom_norm = cle_normalisee(saisie)
    if nom_norm in FRANCAIS_VERS_ISO3_NORM:
        return FRANCAIS_VERS_ISO3_NORM[nom_norm]
    # recherche partielle si la correspondance exacte échoue
    correspondances = {v for k, v in FRANCAIS_VERS_ISO3_NORM.items() if nom_norm in k or k in nom_norm}
    if len(correspondances) == 1:
        return correspondances.pop()
    return None


# Dictionnaire normalisé (sans accents/apostrophes/tirets) pour une recherche
# tolérante aux variantes de saisie ("Côte d'Ivoire", "cote-d-ivoire", etc.)
FRANCAIS_VERS_ISO3_NORM = {cle_normalisee(k): v for k, v in FRANCAIS_VERS_ISO3.items()}


# Table inverse ISO3 -> nom français affichable (pour un affichage propre
# même quand l'utilisateur saisit directement un code ISO3, ex: "JPN").
ISO3_VERS_NOM = {}
for _nom, _code in FRANCAIS_VERS_ISO3.items():
    if _code not in ISO3_VERS_NOM:
        ISO3_VERS_NOM[_code] = _nom.title()


def nom_affichable(nom_saisi, iso3):
    saisie = nom_saisi.strip()
    if len(saisie) == 3 and saisie.isalpha():
        return ISO3_VERS_NOM.get(iso3, saisie.upper())
    return saisie.title()


def calculer_zscores_pays(indicateurs):
    """Calcule les z-scores d'un pays par rapport au panel de référence
    (PANEL_MEAN_STD), avec les mêmes conventions de signe que pour le panel
    principal (un score élevé = situation favorable)."""
    m_pib, s_pib = PANEL_MEAN_STD["PIB_croissance"]
    m_dette, s_dette = PANEL_MEAN_STD["Dette_PIB"]
    m_cc, s_cc = PANEL_MEAN_STD["CompteCourant_PIB"]
    m_inf, s_inf = PANEL_MEAN_STD["Inflation"]
    return {
        "z_economique": (indicateurs["PIB_croissance"] - m_pib) / s_pib,
        "z_fiscal": -(indicateurs["Dette_PIB"] - m_dette) / s_dette,
        "z_externe": (indicateurs["CompteCourant_PIB"] - m_cc) / s_cc,
        "z_monetaire": -(indicateurs["Inflation"] - m_inf) / s_inf,
    }


def generer_note_generique(nom_affiche, indicateurs, zscores, score, categorie, rang_estime, taille_panel):
    forces = sorted(zscores.items(), key=lambda x: -x[1])
    fragilites = sorted(zscores.items(), key=lambda x: x[1])
    pilier_force, pilier_fragile = forces[0][0], fragilites[0][0]
    phrase_force = FORCE_TEMPLATES[pilier_force].format(val=indicateurs[VAL_COL[pilier_force]])
    phrase_fragile = FRAGILITE_TEMPLATES[pilier_fragile].format(val=indicateurs[VAL_COL[pilier_fragile]])

    texte = (
        f"## {nom_affiche} — {categorie}\n\n"
        f"**Score composite : {score:+.2f}** (positionnement estimé : rang {rang_estime}/{taille_panel + 1} "
        f"si inséré dans le panel de référence)\n\n"
        f"{nom_affiche} présente, sur la base des indicateurs retenus, {phrase_force}. "
        f"À l'inverse, le principal point de vigilance porte sur {phrase_fragile}. "
    )
    if score >= 0.20:
        texte += "Le profil de risque global ressort comme favorable au regard du panel de comparaison."
    elif score <= -0.20:
        texte += "Le profil de risque global ressort comme dégradé au regard du panel de comparaison, et justifie un suivi renforcé."
    else:
        texte += "Le profil de risque global se situe dans une zone intermédiaire."
    texte += (
        f"\n\n*Indicateurs : croissance du PIB {indicateurs['PIB_croissance']:.1f}% | "
        f"inflation {indicateurs['Inflation']:.1f}% | dette publique {indicateurs['Dette_PIB']:.0f}% du PIB | "
        f"compte courant {indicateurs['CompteCourant_PIB']:+.1f}% du PIB.*\n\n"
        f"*Score positionné par rapport au panel de référence ({taille_panel} pays) — "
        f"pas une mesure absolue de risque de défaut. cf. README, section limites.*\n"
    )
    return texte


def analyser_pays(nom_pays, sauvegarder_radar=True):
    """Fonction principale : récupère les données d'un pays quelconque,
    calcule son score, génère sa note, et (optionnellement) un radar
    individuel. C'est la fonction appelée par la boucle interactive en bas du
    script."""
    iso3 = resoudre_iso3(nom_pays)
    if iso3 is None:
        print(f"   ! Pays non reconnu : '{nom_pays}'. Essayez en français (ex: Maroc, Japon) "
              f"ou un code ISO3 à 3 lettres (ex: MAR, JPN).")
        return None

    nom_aff = nom_affichable(nom_pays, iso3)
    print(f"\nRécupération des données pour {nom_aff} ({iso3})...")
    indicateurs = {}
    for label, code in WB_INDICATORS.items():
        try:
            indicateurs[label] = fetch_indicator_live(iso3, code)
        except Exception as e:
            print(f"   ! Erreur réseau sur {label} ({e})")
            indicateurs[label] = np.nan

    manquants = [k for k, v in indicateurs.items() if pd.isna(v)]
    if manquants:
        print(f"   (valeur(s) manquante(s) pour : {', '.join(manquants)} "
              f"-> remplacées par la moyenne du panel de référence)")
        for k in manquants:
            indicateurs[k] = PANEL_MEAN_STD[k][0]

    zscores = calculer_zscores_pays(indicateurs)
    score = sum(zscores[c] * w for c, w in WEIGHTS.items())
    categorie = categorie_risque(score)
    rang_estime = int((df["score_composite"] > score).sum()) + 1

    print(f"\n{'=' * 70}")
    print(f"RESULTAT — {nom_aff} ({iso3})")
    print(f"{'=' * 70}")
    print(f"Score composite : {score:+.2f}  ->  {categorie}")
    print(f"Rang estimé      : {rang_estime}/{len(df) + 1} (panel de référence + ce pays)")
    print(f"Indicateurs      : PIB {indicateurs['PIB_croissance']:.1f}% | "
          f"inflation {indicateurs['Inflation']:.1f}% | dette {indicateurs['Dette_PIB']:.0f}% du PIB | "
          f"compte courant {indicateurs['CompteCourant_PIB']:+.1f}% du PIB")

    note = generer_note_generique(nom_aff, indicateurs, zscores, score, categorie,
                                    rang_estime, len(df))
    nom_fichier = f"note_pays_{iso3.lower()}.md"
    with open(nom_fichier, "w", encoding="utf-8") as f:
        f.write(note)
    print(f"-> Note exportée dans {nom_fichier}")

    if sauvegarder_radar:
        angles_p = np.linspace(0, 2 * np.pi, len(PILLAR_COLS), endpoint=False).tolist()
        angles_p += angles_p[:1]
        valeurs_p = [zscores[c] for c in PILLAR_COLS]
        valeurs_p += valeurs_p[:1]
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.plot(angles_p, valeurs_p, color='#c0392b', linewidth=2)
        ax.fill(angles_p, valeurs_p, color='#c0392b', alpha=0.25)
        ax.set_xticks(angles_p[:-1])
        ax.set_xticklabels(["Éco", "Fiscal", "Externe", "Monét."], fontsize=10)
        ax.set_ylim(-2, 2)
        ax.set_yticklabels([])
        ax.set_title(f"{nom_aff} — profil par pilier", fontsize=12, pad=15)
        plt.tight_layout()
        nom_radar = f"radar_{iso3.lower()}.png"
        plt.savefig(nom_radar, dpi=150)
        plt.close()
        print(f"-> Radar exporté dans {nom_radar}")

    return {"iso3": iso3, "indicateurs": indicateurs, "zscores": zscores,
            "score_composite": score, "categorie_risque": categorie}



# ==============================================================================
# 4. VISUALISATIONS
# ==============================================================================

# --- Graphique 1 : classement des pays par score composite ------------------
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn((df["score_composite"] - df["score_composite"].min()) /
                        (df["score_composite"].max() - df["score_composite"].min()))
ax.barh(df["Pays"], df["score_composite"], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Score composite (z-score moyen, 4 piliers)")
ax.set_title("Classement risque pays — score composite")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("classement_risque_pays.png", dpi=150)
plt.close()
print("-> Graphique exporté : classement_risque_pays.png")

# --- Graphique 2 : radar par pays (grille de sous-graphiques) ----------------
n = len(df)
ncols = 4
nrows = int(np.ceil(n / ncols))
angles = np.linspace(0, 2 * np.pi, len(PILLAR_COLS), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), subplot_kw=dict(polar=True))
axes = axes.flatten()
for i, (_, row) in enumerate(df.iterrows()):
    ax = axes[i]
    values = row[PILLAR_COLS].tolist()
    values += values[:1]
    ax.plot(angles, values, color='#1f77b4', linewidth=1.5)
    ax.fill(angles, values, color='#1f77b4', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Éco", "Fiscal", "Externe", "Monét."], fontsize=8)
    ax.set_yticklabels([])
    ax.set_ylim(-2, 2)
    ax.set_title(f"{row['Pays']}\n({row['score_composite']:+.2f})", fontsize=10, pad=12)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("Profil par pilier de risque, par pays (z-scores)", fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig("radar_par_pays.png", dpi=150)
plt.close()
print("-> Graphique exporté : radar_par_pays.png")

# --- Graphique 3 : carte de positionnement externe vs fiscal -----------------
fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(df["z_fiscal"], df["z_externe"], s=300,
                      c=df["score_composite"], cmap="RdYlGn", edgecolors='black', linewidths=0.8)
for _, row in df.iterrows():
    ax.annotate(row["Pays"], (row["z_fiscal"], row["z_externe"]),
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
ax.set_xlabel("Solidité fiscale (z-score, endettement inversé)")
ax.set_ylabel("Solidité externe (z-score, compte courant)")
ax.set_title("Carte de positionnement — risque fiscal vs risque externe")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Score composite")
plt.tight_layout()
plt.savefig("carte_positionnement.png", dpi=150)
plt.close()
print("-> Graphique exporté : carte_positionnement.png")

# --- Graphique 4 : heatmap des piliers ---------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
heat_data = df.set_index("Pays")[PILLAR_COLS]
heat_data.columns = [PILLAR_LABELS[c].split(" (")[0] for c in PILLAR_COLS]
im = ax.imshow(heat_data.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
ax.set_xticks(range(len(heat_data.columns)))
ax.set_xticklabels(heat_data.columns, rotation=30, ha='right')
ax.set_yticks(range(len(heat_data.index)))
ax.set_yticklabels(heat_data.index)
for i in range(heat_data.shape[0]):
    for j in range(heat_data.shape[1]):
        ax.text(j, i, f"{heat_data.values[i, j]:.1f}", ha='center', va='center', fontsize=8)
ax.set_title("Heatmap des piliers de risque par pays (z-scores)")
plt.colorbar(im, ax=ax, label="z-score (+ = favorable)")
plt.tight_layout()
plt.savefig("heatmap_piliers.png", dpi=150)
plt.close()
print("-> Graphique exporté : heatmap_piliers.png")

print("\nTerminé.")

# ==============================================================================
# 5. BOUCLE INTERACTIVE — ANALYSE D'UN PAYS AU CHOIX
# ==============================================================================
print("\n" + "=" * 80)
print("ANALYSE AD HOC D'UN PAYS AU CHOIX")
print("=" * 80)
print("Tapez le nom d'un pays en français (ex: Maroc, Japon, Sénégal) ou son code")
print("ISO3 (ex: MAR, JPN, SEN). Tapez 'quit' pour terminer.\n")

while True:
    try:
        saisie = input("Pays à analyser > ").strip()
    except EOFError:
        break
    if saisie.lower() in ("quit", "exit", "q", ""):
        print("Fin de l'analyse interactive.")
        break
    analyser_pays(saisie)

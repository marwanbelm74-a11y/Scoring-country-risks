# Scoring risque pays & génération automatisée de notes d'analyse

Modèle pédagogique de scoring de risque souverain : standardisation d'indicateurs
macro/institutionnels publics, agrégation en piliers, classement d'un panel de
pays, et **génération automatique de notes d'analyse texte** à partir des données
(et non rédigées à la main).

## 1. Pourquoi ce projet

Une cellule risque pays (banque, assureur-crédit type Coface, département
international d'une banque centrale) doit suivre en continu un grand nombre de
pays. Ce projet automatise les deux étapes les plus chronophages de ce travail :
1. la mise à jour et la standardisation des indicateurs macro/institutionnels,
2. la rédaction d'une première version de note pays — le rédacteur humain
   intervient ensuite pour enrichir, nuancer et challenger ce premier jet, pas
   pour repartir de zéro.

## 2. Pipeline du projet

```
API Banque Mondiale (ou jeu de repli) 
        │
        ▼
Standardisation (z-scores) par indicateur
        │
        ▼
Agrégation en 5 piliers : Économique / Fiscal / Externe / Institutionnel / Monétaire
        │
        ▼
Score composite + classement + catégorie de risque
        │
        ├──► Visualisations (classement, radar, carte de positionnement, heatmap)
        │
        └──► Génération automatique de notes texte (NLG simple, basé sur les données)
                    │
                    └──► Note pays au format Word (exemple : Turquie)
```

## 3. Sources de données

- **API Banque Mondiale** (`api.worldbank.org`), gratuite, sans clé. Indicateurs
  utilisés : croissance du PIB, inflation, dette publique, solde du compte
  courant, et indicateurs de gouvernance (Worldwide Governance Indicators :
  maîtrise de la corruption, état de droit, efficacité des pouvoirs publics).
- **Repli automatique** : si l'API n'est pas accessible (réseau restreint,
  pas de connexion), le script bascule sur un jeu de données illustratif
  intégré, clairement signalé comme tel à chaque exécution et dans tous les
  fichiers générés.

> ⚠️ **Dans cette démonstration, l'accès à l'API Banque Mondiale a été bloqué
> par les restrictions réseau de l'environnement d'exécution** (l'API n'est pas
> sur la liste des domaines autorisés). Les résultats présentés ici utilisent
> donc le **jeu de données illustratif** (ordres de grandeur approximatifs).
> Exécuté chez vous avec un accès internet complet, le script interroge
> automatiquement les données réelles (`USE_LIVE_API = True` en tête de fichier).

## 4. Méthodologie de scoring

### a. Standardisation
Chaque indicateur est converti en z-score par rapport au panel de pays étudié
(`(valeur - moyenne du panel) / écart-type du panel`). Les indicateurs où une
valeur élevée est défavorable (dette, inflation) sont inversés, pour que
"score élevé" signifie toujours "situation plus favorable".

### b. Cinq piliers
| Pilier | Indicateur(s) | Logique |
|---|---|---|
| Économique | Croissance du PIB | Une croissance plus forte soutient la soutenabilité de la dette |
| Fiscal | Dette publique / PIB (inversé) | Moins de dette = plus de marge de manœuvre budgétaire |
| Externe | Solde du compte courant / PIB | Une économie qui ne dépend pas des financements extérieurs est moins vulnérable à un choc de capitaux |
| Institutionnel | Indicateurs de gouvernance (WGI) | La qualité institutionnelle conditionne la capacité de réponse aux chocs |
| Monétaire | Inflation (inversée) | Une inflation maîtrisée signale une politique monétaire crédible |

### c. Score composite et catégorie
Moyenne pondérée des 5 piliers (poids égaux de 20 % par défaut, modifiables
dans le code — `WEIGHTS`). Le score est ensuite traduit en 5 catégories de
risque **propres à ce modèle** (très faible / faible / modéré / élevé / très
élevé) — cette échelle ne reproduit volontairement pas la méthodologie ni les
échelles de notation des agences officielles (S&P, Moody's, Fitch), qui sont
propriétaires et bien plus riches (facteurs qualitatifs, comité de notation,
horizon de notation, etc.).

### d. Génération automatique des notes
Pour chaque pays, le code identifie automatiquement le pilier le plus fort et
le plus faible (à partir des z-scores), puis assemble une phrase à partir d'un
gabarit pré-écrit et des valeurs réelles de l'indicateur concerné. C'est de la
génération de texte structurée à partir de données ("NLG" simple), pas un texte
écrit librement — chaque phrase est strictement déterminée par les chiffres.

## 5. Résultats obtenus (jeu illustratif, cf. `country_risk_scores.csv`)

| Rang | Pays | Score composite | Catégorie |
|---|---|---|---|
| 1 | Vietnam | +0,87 | Risque très faible |
| 2 | Allemagne | +0,80 | Risque très faible |
| 3 | Indonésie | +0,47 | Risque faible |
| ... | ... | ... | ... |
| 10 | États-Unis | -0,23 | Risque élevé |
| 11 | Égypte | -0,46 | Risque élevé |
| 12 | Argentine | -1,24 | Risque très élevé |

**Point méthodologique important** : les États-Unis se classent mal dans ce
panel précis (rang 10/12) malgré la taille et la solidité structurelle de leur
économie. Ce n'est pas une anomalie du modèle mais la conséquence directe de la
méthodologie choisie : le score est **relatif au panel étudié** (z-score), pas
absolu, et pénalise ici un déficit courant et un endettement public élevés par
rapport aux 11 autres pays du panel. **Un score composite relatif n'a de sens
qu'au sein d'un panel donné** — c'est une limite à expliciter systématiquement
à l'oral si ce projet est présenté.

## 6. Limites (à lire avant toute interprétation)

- Données illustratives dans cette démonstration — à remplacer par les
  données API en direct pour tout usage sérieux.
- Score **relatif au panel**, pas une mesure absolue de risque de défaut.
- Pas de facteurs qualitatifs (risque politique de court terme, calendrier
  électoral, relations diplomatiques, structure de la dette par maturité/devise).
- Pondération égale des 5 piliers — une vraie méthodologie pondérerait
  différemment selon le profil de pays (ex. poids plus fort du pilier externe
  pour un pays très dépendant des financements de marché).
- Pas de dimension temporelle : le modèle compare un instant T, sans capter les
  trajectoires (un pays qui s'améliore vite vs un pays qui se détériore vite).
- La génération de texte est volontairement simple (un seul point fort/un seul
  point faible) — un vrai rédacteur croiserait plusieurs piliers et nuancerait.

## 7. Pistes d'extension

- Ajouter une **dimension temporelle** : suivre l'évolution du score composite
  sur 5-10 ans par pays (alerte sur les trajectoires qui se dégradent vite).
- Pondérer les piliers différemment selon des **clusters de pays** (économies
  avancées vs émergentes vs frontière).
- Croiser avec des **données de marché** (spread de CDS souverain, spread
  EMBI, notation Bloomberg/Refinitiv) pour valider le score contre un signal
  de marché.
- Remplacer la génération de texte par gabarits avec un véritable modèle de
  langage (API Claude) pour produire une synthèse plus nuancée à partir des
  mêmes données structurées — en gardant les chiffres sourcés du modèle comme
  unique source de vérité.
- Construire un historique et calculer un score de **momentum** (vitesse de
  variation du score composite) en plus du niveau.

## 8. Fichiers du projet

- `country_risk_model.py` — pipeline complet (données, scoring, NLG, graphiques)
- `country_risk_scores.csv` — résultats détaillés (z-scores, score composite, rang)
- `notes_pays.md` — notes d'analyse générées automatiquement, un par pays
- `classement_risque_pays.png` — classement par score composite
- `radar_par_pays.png` — profil par pilier, grille des 12 pays
- `carte_positionnement.png` — positionnement fiscal vs externe
- `heatmap_piliers.png` — vue d'ensemble des 5 piliers x 12 pays
- `note_pays_turquie.docx` — exemple de note pays individuelle au format Word,
  générée à partir des mêmes données (titre, tableau de synthèse, radar,
  commentaire, méthodologie et limites)
- `generate_note_docx.js` — script Node.js de génération de la note Word

## 9. Comment l'exécuter

```bash
pip install pandas numpy matplotlib requests
python3 country_risk_model.py
```

Pour générer une note Word pour un autre pays : adapter les valeurs dans
`generate_note_docx.js` (ou, en extension, automatiser entièrement la
génération pour les 12 pays du panel en boucle).

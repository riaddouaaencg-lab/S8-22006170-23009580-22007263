# 🔍 Prédiction de la Fraude Interne en Entreprise par Machine Learning

> **Compte Rendu de Projet — Semestre 8**
> École Nationale de Commerce et de Gestion de Settat · Université Hassan 1er
> Année Universitaire 2024–2025

---

## 📋 Informations du projet

| Champ | Détail |
|-------|--------|
| **Module** | Audit, Contrôle Interne & Systèmes d'Information |
| **Filière** | [CAC / CI / PSCM — L3 S8] |
| **Réalisé par** | [Prénom NOM] |
| **Encadrant** | [Titre Prénom NOM] |
| **Période** | 2024 – 2025 |

---

## 📑 Sommaire

1. [Contexte et Problématique](#1-contexte-et-problématique)
2. [Fondement Théorique — Triangle de la Fraude](#2-fondement-théorique--triangle-de-la-fraude)
3. [Dataset et Variables](#3-dataset-et-variables)
4. [Analyse Exploratoire](#4-analyse-exploratoire)
5. [Modélisation Machine Learning](#5-modélisation-machine-learning)
6. [Résultats](#6-résultats)
7. [Segmentation des Profils à Risque](#7-segmentation-des-profils-à-risque)
8. [Recommandations de Contrôle Interne](#8-recommandations-de-contrôle-interne)
9. [Discussion et Limites](#9-discussion-et-limites)
10. [Conclusion](#10-conclusion)
11. [Technologies Utilisées](#11-technologies-utilisées)
12. [Structure du Projet](#12-structure-du-projet)
13. [Bibliographie](#13-bibliographie)

---

## 1. Contexte et Problématique

La fraude interne représente l'une des menaces les plus coûteuses pour les organisations. Selon l'**ACFE Report to the Nations 2024**, les entreprises perdent en moyenne **5 % de leur chiffre d'affaires annuel** à cause de la fraude occupationnelle, et la majorité des cas impliquent des employés internes disposant d'un accès privilégié aux systèmes et ressources de l'entreprise.

Dans ce contexte, ce projet pose la problématique suivante :

> **Dans quelle mesure les algorithmes d'apprentissage automatique permettent-ils de prédire le risque de fraude interne à partir des caractéristiques comportementales et organisationnelles des employés ?**

### Objectifs

- Opérationnaliser le **Triangle de la Fraude de Cressey (1953)** sous forme de variables quantitatives mesurables
- Construire et comparer **trois modèles de classification** supervisée pour la détection proactive de la fraude
- Fournir des **recommandations de contrôle interne** fondées sur les résultats analytiques et les standards ACFE

---

## 2. Fondement Théorique — Triangle de la Fraude

Le modèle de **Donald Cressey (1953)** établit que tout acte de fraude interne résulte de la convergence simultanée de trois facteurs :

```
              OPPORTUNITÉ
             (Accès Privilégié,
            Contrôles Faibles)
                   △
                  /|\
                 / | \
                /  |  \
               /FRAUDE \
              /INTERNE  \
             /___________\
PRESSION                  RATIONALISATION
(Stress Financier,        (Insatisfaction,
Objectifs Irréalistes)    Sentiment d'Injustice)
```

| Sommet | Définition | Variable proxy |
|--------|-----------|---------------|
| **Opportunité** | Accès aux ressources et contrôles insuffisants | `Acces_Privilegie` |
| **Pression** | Contraintes financières ou professionnelles | `Score_Pression_Financiere`, `Heures_Supp_Mois` |
| **Rationalisation** | Justification morale de l'acte frauduleux | `Satisfaction_Travail`, `Conges_Non_Pris` |

> Source : Cressey, D.R. (1953). *Other People's Money*. Free Press. Validé par l'ACFE dans ses rapports biennaux depuis 1988.

---

## 3. Dataset et Variables

### 3.1 Construction du dataset

Le dataset a été généré synthétiquement avec une **seed fixe (42)** pour garantir la reproductibilité totale des résultats.

```python
np.random.seed(42)
n_employes = 1000
```

**Répartition par département** (probabilités d'échantillonnage) :

| Département | Part dans l'effectif |
|------------|----------------------|
| Ventes | 25 % |
| Logistique | 20 % |
| IT | 15 % |
| Finance | 15 % |
| Achats | 15 % |
| RH | 10 % |

### 3.2 Dictionnaire des variables

| Variable | Type | Plage | Rôle |
|----------|------|-------|------|
| `Departement` | Catégorielle | 6 modalités | Prédicteur |
| `Anciennete_Annees` | Entier | [1, 24] | Prédicteur |
| `Score_Pression_Financiere` | Entier | [1, 10] | Prédicteur |
| `Satisfaction_Travail` | Entier | [1, 10] | Prédicteur |
| `Heures_Supp_Mois` | Entier | [0, 50] | Prédicteur |
| `Conges_Non_Pris` | Entier | [0, 30] | Prédicteur |
| `Acces_Privilegie` | Binaire | {0, 1} | Prédicteur clé |
| `Fraude_Interne` | Binaire | {0, 1} | **Variable cible** |

### 3.3 Construction de la variable cible

La variable cible est dérivée d'un **score de risque composite** inspiré du Triangle de Cressey :

```
Score_Risque = (Pression_Financière × 0.4)
             + (Accès_Privilégié    × 15.0)
             + ((10 − Satisfaction) × 0.3)
             + (Ancienneté          × 0.2)
```

**Seuil de fraude** : 95e percentile du score de risque = **22.30**

Les employés dont le score dépasse ce seuil sont étiquetés `Fraude_Interne = 1`.

### 3.4 Indicateurs clés du dataset

| KPI | Valeur |
|-----|--------|
| Effectif total | 1 000 employés |
| Cas de fraude | **50 (5,0 %)** |
| Employés avec accès privilégié | **19 %** |
| Score de pression moyen | 5.44 / 10 |
| Score de satisfaction moyen | 5.45 / 10 |

---

## 4. Analyse Exploratoire

### 4.1 Taux de fraude par département

| Département | Effectif | Cas de fraude | Taux de fraude |
|------------|---------|--------------|----------------|
| **RH** | 112 | 7 | **6.25 %** |
| **Ventes** | 240 | 15 | **6.25 %** |
| Finance | 153 | 8 | 5.23 % |
| Achats | 166 | 7 | 4.22 % |
| Logistique | 199 | 8 | 4.02 % |
| IT | 130 | 5 | 3.85 % |

> ⚠️ Les départements **RH** et **Ventes** présentent le taux de fraude le plus élevé (6.25 %), soit 1.6× la moyenne globale de 5 %.

### 4.2 Impact de l'accès privilégié

L'accès privilégié est le facteur le plus discriminant du dataset :

| Accès Privilégié | Taux Non-Fraude | Taux Fraude |
|-----------------|-----------------|-------------|
| **Non (0)** | 100.00 % | **0.00 %** |
| **Oui (1)** | 73.68 % | **26.32 %** |

> 🔑 **Aucun employé sans accès privilégié n'est fraudeur dans ce dataset.** La variable `Acces_Privilegie` constitue une condition nécessaire (mais non suffisante) à la fraude.

### 4.3 Taux de fraude par ancienneté

| Tranche d'ancienneté | Taux de fraude |
|---------------------|----------------|
| 1–5 ans | 0.00 % |
| 5–9 ans | 0.97 % |
| 9–13 ans | 6.06 % |
| 13–18 ans | 5.10 % |
| **18+ ans** | **13.47 %** |

> 📈 Le risque augmente fortement à partir de **9 ans d'ancienneté** et atteint son pic au-delà de 18 ans. Les employés les plus anciens combinent une connaissance approfondie des processus avec un potentiel sentiment d'impunité.

### 4.4 Matrice de corrélation — Observations principales

- `Acces_Privilegie` présente la corrélation la plus forte avec `Fraude_Interne`
- `Score_Pression_Financiere` et `Anciennete_Annees` sont les deux autres variables positivement corrélées
- `Satisfaction_Travail` est **négativement** corrélée à la fraude (plus la satisfaction est faible, plus le risque est élevé)
- Les heures supplémentaires et les congés non pris présentent une corrélation marginale

---

## 5. Modélisation Machine Learning

### 5.1 Protocole expérimental

```
Dataset (1 000 observations)
        │
        ├── Encodage : LabelEncoder (Département → Dept_encoded)
        ├── Split 70/30 stratifié (stratify=y, random_state=42)
        │       ├── Train : 700 observations (35 fraudes)
        │       └── Test  : 300 observations (15 fraudes)
        └── Normalisation StandardScaler (Régression Logistique uniquement)
```

**Validation croisée** : StratifiedKFold k=5, shuffle=True, random_state=42

### 5.2 Modèles entraînés

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight='balanced',
    min_samples_leaf=3,
    random_state=42
)
```

#### Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    random_state=42
)
```

#### Régression Logistique
```python
LogisticRegression(
    C=0.5,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
```

---

## 6. Résultats

### 6.1 Comparaison des performances AUC

| Modèle | AUC Test | AUC CV-5 | Average Precision |
|--------|----------|----------|-------------------|
| Random Forest | 0.9939 | 0.9951 | 0.9248 |
| Gradient Boosting | 0.9972 | 0.9952 | 0.9528 |
| **Régression Logistique** | **0.9998** | **0.9997** | **0.9958** |

> 🏆 **Meilleur modèle : Régression Logistique** avec AUC CV-5 = **0.9997** — ce résultat s'explique par la nature quasi-linéaire de la fonction de score de risque utilisée pour construire la variable cible.

### 6.2 Rapport de classification — Random Forest (sur test set)

```
              precision    recall  f1-score   support

  Non-Fraude       0.98      1.00      0.99       285
      Fraude       1.00      0.67      0.80        15

    accuracy                           0.98       300
   macro avg       0.99      0.83      0.90       300
weighted avg       0.98      0.98      0.98       300
```

### 6.3 Matrice de confusion — Random Forest

|  | Prédit : Non-Fraude | Prédit : Fraude |
|--|---------------------|-----------------|
| **Réel : Non-Fraude** | 285 (TN) | 0 (FP) |
| **Réel : Fraude** | 5 (FN) | 10 (TP) |

- **0 fausse alarme** : aucun employé non-fraudeur n'est incorrectement signalé
- **10/15 fraudeurs détectés** (recall = 67 %) sur le set de test
- **5 fraudeurs manqués** (FN) — profils à score légèrement sous le seuil

### 6.4 Importance des variables (Random Forest — Gini)

| Rang | Variable | Importance |
|------|----------|-----------|
| 🥇 1 | **Accès Privilégié** | **0.5241** |
| 🥈 2 | Ancienneté | 0.1471 |
| 🥉 3 | Pression Financière | 0.1423 |
| 4 | Satisfaction au Travail | 0.0880 |
| 5 | Heures Supplémentaires | 0.0477 |
| 6 | Congés Non Pris | 0.0313 |
| 7 | Département | 0.0194 |

> La variable `Acces_Privilegie` représente à elle seule **52.41 %** de l'importance totale du modèle Random Forest, confirmant le rôle central du sommet « Opportunité » dans le Triangle de Cressey.

---

## 7. Segmentation des Profils à Risque

### 7.1 Courbe de gain cumulé

| Population auditée | Fraudes détectées |
|--------------------|-------------------|
| Top 10 % | ~80 % |
| **Top 20 %** | **100 %** |
| Audit aléatoire 20 % | ~20 % |

> 📊 En ciblant uniquement les **20 % de profils les plus risqués** selon le modèle RF, on détecte **100 % des 50 cas de fraude** — soit un facteur multiplicatif de **5×** par rapport à un audit aléatoire.

### 7.2 Top 10 profils à risque maximal

| Département | Ancienneté | Accès Privilégié | Pression | Prob. RF | Fraude |
|------------|-----------|-----------------|---------|----------|--------|
| Ventes | 15 ans | ✅ | 10/10 | 0.976 | ✅ |
| Ventes | 23 ans | ✅ | 10/10 | 0.962 | ✅ |
| RH | 24 ans | ✅ | 9/10 | 0.959 | ✅ |
| IT | 21 ans | ✅ | 9/10 | 0.952 | ✅ |
| Achats | 21 ans | ✅ | 8/10 | 0.949 | ✅ |
| Finance | 15 ans | ✅ | 9/10 | 0.940 | ✅ |
| Ventes | 15 ans | ✅ | 7/10 | 0.934 | ✅ |
| Finance | 24 ans | ✅ | 8/10 | 0.933 | ✅ |
| Ventes | 20 ans | ✅ | 5/10 | 0.917 | ✅ |
| RH | 17 ans | ✅ | 10/10 | 0.914 | ✅ |

> Les 10 profils à plus haut risque partagent tous **accès privilégié = 1**, une **ancienneté ≥ 15 ans** et une **pression financière ≥ 7/10**.

### 7.3 Matrice de risque — Priorisation des audits

```
PRESSION FINANCIÈRE
      ↑
Élevée│  AUDIT            ⚠️  RISQUE
      │  PRIORITAIRE          CRITIQUE
      │  (Accès limité    (Fort accès +
      │   + Forte          Forte pression)
      │   pression)
      │─────────────────────────────────
Faible│  SURVEILLANCE     SURVEILLANCE
      │  ROUTINIÈRE           ACCRUE
      │  (Accès limité    (Fort accès +
      │   + Faible         Faible pression)
      │   pression)
      └──────────────────────────────────→
             Sans accès          Avec accès
                          ACCÈS PRIVILÉGIÉ
```

---

## 8. Recommandations de Contrôle Interne

Basées sur l'**ACFE Report to the Nations 2024** et les résultats du modèle.

### 🔐 Axe 1 — Gouvernance des accès

- Mettre en œuvre une **revue semestrielle des accès privilégiés** avec validation hiérarchique obligatoire
- Appliquer le **Principe du Moindre Privilège (PoLP)** — chaque employé n'accède qu'aux ressources strictement nécessaires à sa mission
- Déployer une **journalisation en temps réel** des actions sensibles avec système d'alertes automatiques
- Rendre obligatoire la **séparation des tâches (SoD)** sur tous les processus financiers critiques (engagement, validation, paiement)

### 🛡️ Axe 2 — Programme de prévention

- Installer une **hotline fraude anonyme** : selon l'ACFE 2024, les organisations dotées d'un tel dispositif détectent les fraudes **50 % plus rapidement** et subissent des pertes médianes **20 % inférieures**
- Rendre la **formation anti-fraude annuelle obligatoire** pour les employés disposant d'un accès privilégié
- Instaurer une **rotation des postes sensibles** (Finance, Achats) tous les 3 à 5 ans
- Conduire des **audits surprises trimestriels** ciblés sur les départements RH et Ventes (taux de fraude les plus élevés)

### 🔍 Axe 3 — Détection précoce par data analytics

- **Déployer le modèle RF en production** avec recalcul mensuel du score de risque pour chaque employé
- Créer un **tableau de bord RH** intégrant les signaux comportementaux (congés non pris, heures supplémentaires anormales)
- Soumettre les **5 % de profils les plus à risque** à une revue mensuelle par le comité d'audit interne
- Intégrer la **courbe de gain cumulé** comme outil de calibration du budget d'audit

---

## 9. Discussion et Limites

### Points forts

- **Reproductibilité totale** : seed fixée à 42, toutes les étapes sont documentées et exécutables
- **Framework théorique solide** : ancrage dans le Triangle de Cressey, validé par l'ACFE
- **Pipeline ML complet** : EDA → Feature engineering → Modélisation → Évaluation → Recommandations
- **Métriques adaptées au déséquilibre** : AUC-ROC, Average Precision, confusion matrix détaillée

### Limites identifiées

| Limite | Impact | Solution envisagée |
|--------|--------|-------------------|
| Données simulées | Généralisation limitée | Validation sur données réelles d'entreprise |
| Variable cible construite | Performance artificiellement élevée | Collecte de données de fraude avérée |
| Déséquilibre 95/5 | Sous-détection de la classe minoritaire | Techniques SMOTE / under-sampling |
| Pas de variables qualitatives | Modèle incomplet | Intégration des évaluations RH, signalements |
| Interprétabilité limitée | Acceptabilité légale faible | Ajout de SHAP values |

### Perspectives

- Tester des approches **SMOTE** (*Synthetic Minority Over-sampling Technique*) pour améliorer le recall sur la classe fraude
- Intégrer des méthodes d'explicabilité **SHAP** pour des explications individuelles par profil
- Étendre le modèle à des **données comportementales** (logs systèmes, patterns d'accès horaires)
- Envisager une **validation externe** sur des données historiques de fraude avérée

---

## 10. Conclusion

Ce projet a démontré la faisabilité d'une approche de **détection proactive de la fraude interne** fondée sur le Machine Learning, en combinant le cadre théorique du Triangle de Cressey avec des algorithmes de classification supervisée.

Les principaux enseignements sont :

1. **L'accès privilégié est le facteur dominant** (52 % de l'importance Gini) — ce qui valide empiriquement le sommet « Opportunité » comme levier d'action prioritaire
2. **Les trois modèles atteignent des AUC > 0.99**, confirmant la faisabilité technique d'un tel système de détection
3. **Le ciblage du top 20 %** selon le score RF permet de détecter **100 % des fraudes**, soit un gain d'efficience de 5× par rapport à un audit aléatoire
4. **Les départements RH et Ventes** requièrent une attention particulière avec un taux de fraude de 6.25 %

Sur le plan managérial, ces résultats fournissent une justification quantitative solide pour investir dans la **gouvernance des accès** et les **programmes de détection analytique**, en cohérence avec les recommandations de l'ACFE et les standards COSO de contrôle interne.

---

## 11. Technologies Utilisées

```python
# Environnement
Python 3.x
Jupyter Notebook / Google Colab

# Bibliothèques
pandas          # Manipulation des données
numpy           # Calcul numérique (seed=42)
matplotlib      # Visualisations (5 figures)
seaborn         # Heatmaps et graphiques statistiques
scikit-learn    # Modèles ML, métriques, preprocessing
```

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![pandas](https://img.shields.io/badge/pandas-Data-green?logo=pandas)

---

## 12. Structure du Projet

```
📦 fraude-interne-ml/
├── 📓 notebook.ipynb          # Code source principal (9 cellules)
├── 📄 README.md               # Ce compte rendu
├── 📊 figures/
│   ├── fig1_overview.png      # Vue d'ensemble & Triangle de Cressey
│   ├── fig2_eda.png           # Analyse exploratoire — Profils à risque
│   ├── fig3_models.png        # Performances des modèles ML
│   ├── fig4_risk.png          # Segmentation des profils à risque
│   └── fig5_recommendations.png  # Dashboard recommandations ACFE
└── 📋 requirements.txt        # Dépendances Python
```

### Lancer le projet

```bash
# Cloner le dépôt
git clone https://github.com/[username]/fraude-interne-ml.git
cd fraude-interne-ml

# Installer les dépendances
pip install -r requirements.txt

# Lancer le notebook
jupyter notebook notebook.ipynb
```

### `requirements.txt`

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

---

## 13. Bibliographie

- **ACFE** (2024). *Report to the Nations: Global Study on Occupational Fraud and Abuse*. Association of Certified Fraud Examiners. https://www.acfe.com/report-to-the-nations
- **Cressey, D.R.** (1953). *Other People's Money: A Study in the Social Psychology of Embezzlement*. Free Press, Glencoe.
- **COSO** (2013). *Internal Control — Integrated Framework*. Committee of Sponsoring Organizations of the Treadway Commission.
- **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- **Friedman, J.H.** (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.
- **Pedregosa, F. et al.** (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830. https://scikit-learn.org
- **Chawla, N.V. et al.** (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.

---

*Document rédigé conformément au format académique ENCG Settat — Semestre 8 — Année Universitaire 2024–2025*

*Toutes les valeurs numériques présentées dans ce document sont issues de l'exécution du notebook avec `np.random.seed(42)` et sont 100 % reproductibles.*

# PRÉDICTION DE LA FRAUDE INTERNE PAR DES MODÈLES DE CLASSIFICATION

## 1. AVANT-PROPOS

Dans un environnement économique de plus en plus incertain, hybride et interconnecté, le **contrôle interne** s'impose comme un pilier fondamental pour assurer la pérennité et l'intégrité des organisations. L'essor rapide de la digitalisation a certes fluidifié les processus d'entreprise, mais il a également entraîné une multiplication et une sophistication sans précédent des risques de fraude. Selon l'Association of Certified Fraud Examiners (ACFE), les entreprises perdent chaque année près de 5 % de leur chiffre d'affaires en raison de fraudes internes, soulignant ainsi un impact économique mondial massif.

Face aux limites des approches d'audit traditionnelles, l'intelligence artificielle (IA) et l'apprentissage automatique (Machine Learning) émergent comme des solutions incontournables. Ils transforment la détection des anomalies en un processus proactif, continu et de plus en plus précis. C'est dans ce contexte de transformation digitale et sécuritaire que s'inscrit ce travail, justifiant le choix du sujet : **« Prédiction de la fraude interne par des modèles de classification »**.

---

## 2. INTRODUCTION GÉNÉRALE

### Problématique
La numérisation des systèmes d'information génère des volumes immenses de données, rendant obsolètes les méthodes manuelles d'échantillonnage de contrôle interne. Dès lors, la question centrale de cette étude est : **Comment l'intelligence artificielle peut-elle améliorer la détection et la prévention de la fraude interne dans les entreprises ?**

### Objectifs
* **Comprendre** les mécanismes de fraude interne
* **Identifier** les variables explicatives (comportementales, financières, organisationnelles)
* **Construire** un modèle de prédiction fiable

### Hypothèses
* L'IA permet une détection plus rapide, précise et à plus grande échelle que les méthodes traditionnelles.
* L'intégration de variables comportementales et liées aux accès améliore significativement la performance des modèles.

### Méthodologie
* **Approche quantitative (Machine Learning)**
* **Données profilées** : Génération d'un Dataset métier réaliste (1000 employés) basé sur les typologies ACFE.

---

## 3. SOMMAIRE

### PARTIE I : CADRE THÉORIQUE
1. Contrôle interne à l’ère de l’IA
2. Fraude interne : concepts et modèles explicatifs
3. Intelligence artificielle et détection de fraude

### PARTIE II : ÉTUDE EMPIRIQUE
1. Présentation des données
2. Méthodologie de modélisation
3. Résultats et interprétation (Graphiques)

---

# 🧠 PARTIE I : CADRE THÉORIQUE

## 4. CONTEXTE (Transformation Digitale & Contrôle Interne)

### Le contrôle interne à l’ère de l’IA
Le contrôle interne désigne l’ensemble des dispositifs appliqués par la direction pour s’assurer de la fiabilité de l'information financière, de la conformité aux lois et de la protection des actifs. Historiquement, ces systèmes présentent des limites majeures :
* **Détection tardive :** La fraude dure en moyenne 12 à 18 mois avant d'être découverte.
* **Dépendance humaine :** Forte vulnérabilité aux biais subjectifs.
* **Faible capacité d’analyse massive :** Impossible d'auditer 100% des transactions dans un monde de Big Data.

Face à la complexité croissante des fraudes, l'intelligence artificielle permet une détection par **analyse prédictive** capable de traiter des téraoctets de données en temps réel pour réduire drastiquement les coûts de la fraude et améliorer la gouvernance d'entreprise.

## 5. FRAUDE INTERNE : DÉFINITION ET THÉORIES

La fraude interne, ou fraude occupationnelle, est définie comme un acte frauduleux commis par un employé (Détournement d’actifs, corruption, manipulation comptable).
Selon la typologie de l'Association of Certified Fraud Examiners (ACFE), le modèle conceptuel dominant pour l'expliquer est le **Triangle de la Fraude** (Donald Cressey, 1953) reposant sur trois dimensions :
1. **Pression :** Problèmes financiers, objectifs irréalistes.
2. **Opportunité :** Faiblesse du contrôle interne, accès aux ressources ou *Accès Privilégié* dans les SI.
3. **Rationalisation :** Justification morale, insatisfaction au travail.

---

# 🔬 PARTIE II : ÉTUDE EMPIRIQUE (ANALYSE DATA SCIENCE)

## 6. REFORMULATION, DONNÉES ET PIPELINE

### Reformulation du problème
L'objectif est de construire un modèle de **classification binaire** permettant de prédire le risque de fraude : (`Fraude interne = Oui / Non`).

### Présentation des Données et Variables
Nous modélisons une entreprise de 1000 employés à travers les lentilles du Triangle de Cressey :
* **Pression :** Score de pression financière.
* **Opportunité :** Accès Privilégié (Oui/Non).
* **Rationalisation :** Niveau de Satisfaction au Travail.
Une équation de risque a été développée et les **5% d'employés les plus à risque** ont été labellisés expérimentalement comme fraudeurs (`seuil du 95e centile`).

### Vue d'ensemble du DataSet
![Vue d'Ensemble & Triangle de Cressey](fig1_overview.png)
*(Figure 1 : Tableau de bord initial démontrant un taux de fraude isolé à 5.0%, la forte prédominance de la fraude dans certains départements et la distribution asymétrique du risque global.)*

---

## 7. ANALYSE EXPLORATOIRE (EDA) ET PROFILS À RISQUE

Avant d'entraîner l'intelligence artificielle, l'exploration des données valide nos hypothèses comportementales.

![Analyse Exploratoire](fig2_eda.png)
*(Figure 2 : Corrélations et Distributions des caractéristiques liées à la fraude.)*

**Interprétation des Variables :**
* *Pression vs Satisfaction :* On observe visuellement que les individus fraudeurs (points rouges) se concentrent dans le quadrant "Forte Pression / Faible Satisfaction".
* *Accès Privilégié :* Conformément à l'hypothèse d'opportunité, une large part des fraudes est rendue possible parce que l'employé détient un accès critique.
* *Comportements suspects :* Les fraudeurs présentent un volume moyen d'Heures Supplémentaires plus élevé et des Congés Non Pris importants, symptômes classiques pour masquer une fraude continue (théorie de la dissimulation).

---

## 8. ENTRAÎNEMENT, MODÈLES ET ÉVALUATION

### Choix des algorithmes
Face à des données non-linéaires et complexes, nous avons confronté trois modèles :
1. **Régression Logistique (LR)** (Baseline)
2. **Gradient Boosting (GB)**
3. **Random Forest (RF)** (Ensemble de sous-arbres, robuste face aux déséquilibres)

Le jeu de données (Train 70% / Test 30%) a subi un traitement StandardScaler pour la régression, et le déséquilibre natif (5% fraudeurs / 95% normaux) a été mitigé par l'hyperparamètre `class_weight='balanced'`.

![Performances des Modèles](fig3_models.png)
*(Figure 3 : Courbes ROC, Précision-Rappel, Matrice de Confusion et Feature Importance)*

### Interprétation des Résultats ML :
* **Modèle Prédictif Performant :** Le modèle Random Forest excelle avec une **AUC cross-validée parfaite (~1.00)**. La matrice de confusion prouve sa capacité à isoler les vrais fraudeurs avec précision (zéro faux positif), ce qui réduit drastiquement les coûts d'audit inutiles.
* **Importance des Variables (Feature Importance) :** Le graphique d'importance de Gini confirme catégoriquement la théorie de Cressey : **l'Accès Privilégié (Opportunité) et la Pression Financière dominent les règles de classification**.

---

## 9. APPLICATIONS PRATIQUES ET BONUS GOUVERNANCE

Un algorithme prédictif n'a de valeur que s'il est converti en indicateurs actionnables.

![Segmentation et Risque](fig4_risk.png)
*(Figure 4 : Suivi du risque par département et profil)*
Le modèle permet d'établir une **liste dynamique des "Top 50 Profils à Risque"**. Cela permet au pilotage RH et à la Direction de l'Audit Interne d'allouer leurs ressources avec une efficience maximale sur les employés combinant des accès critiques, une forte pression et une forte insatisfaction.

![Recommandations ACFE](fig5_recommendations.png)
*(Figure 5 : Matrice des recommandations basées sur les modèles d'IA et l'ACFE)*

### Le Pont entre le ML et l'ACFE 2024
La Matrice 2x2 de Gouvernance (Figure 5) démontre que les employés ayant un profil "Fort Accès + Forte Pression" relèvent du risque rouge ("Critique") justifiant un Audit immédiat.
En réponse, trois recommandations principales de contrôle interne s'imposent :
1. **Contrôle des Accès (PoLP) :** Retirer rapidement les accès informatiques orphelins.
2. **Programme de Prévention :** Mettre en place des mécanismes de *Hotline* de signalement (Whistleblowing), les statistiques ACFE prouvant leur forte efficacité.
3. **Analytique Continue :** Déployer ce modèle RandomForest de prédiction en production (Score de risque Live calculé par mois) transformant l'audit d'une approche réactive (post-mortem) à une posture proactive (prédictive).

---

## 10. CONCLUSION

L’intelligence artificielle offre un levier sans précédent pour le contrôle interne. L'intégration des données comportementales, couplées avec les théories criminologiques comme le Triangle de Cressey, prouve que l'on peut identifier la fraude interne avant qu'elle n'infuse durablement l'organisation. L'algorithme *Random Forest* s'est révélé être une méthode souveraine face aux données fortement déséquilibrées des fraudes occupationnelles. 

Toutefois, cette transition vers un contrôle prédictif se heurte à des **limites éthiques et légales** : le recueil de variables telles que la "Pression financière" peut heurter la réglementation sur les données privées (RGPD). Les futures initiatives devront équilibrer l'optimisation mathématique avec le strict respect de la vie privée des collaborateurs.

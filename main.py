from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time

# ==============================================================================
# 1. CHARGEMENT ET PREPARATION DES DONNEES (TRAIN)
# ==============================================================================
df_train = pd.read_csv("train.csv", index_col=0)
df_train = df_train.drop(columns=['id', 'Gender'], errors='ignore')

miles_limits = [0, 700, 3000, float('inf')]
flight_categories = ['Short-Haul', 'Medium-Haul', 'Long-Haul']
df_train['Flight_Category'] = pd.cut(df_train['Flight Distance'], bins=miles_limits, labels=flight_categories)

age_limits = [2, 12, 18, 65, 120]
age_categories = ['enfant', 'ado', 'adulte', 'senior']
df_train['Age_Category'] = pd.cut(df_train['Age'], bins=age_limits, labels=age_categories)

cols_numeriques = df_train.select_dtypes(include=[np.number]).columns.tolist()
cols_a_exclure = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
cols_notes = [col for col in cols_numeriques if col not in cols_a_exclure]

df_moyennes = df_train.copy()
df_moyennes[cols_notes] = df_moyennes[cols_notes].replace(0, np.nan)

# ==============================================================================
# 2. CALCUL DES MOYENNES DES NOTES (DESCRIPTIF)
# ==============================================================================
moyennes_par_age = df_moyennes.groupby('Age_Category', observed=False)[cols_notes].mean().T
moyennes_par_age['Moyenne Globale'] = moyennes_par_age.mean(axis=1)
moyennes_par_age = moyennes_par_age.round(2)

moyennes_vol_et_classe = df_moyennes.groupby(['Flight_Category', 'Class'], observed=False)[cols_notes].mean().T
moyennes_vol_et_classe['Moyenne Globale'] = moyennes_vol_et_classe.mean(axis=1)
moyennes_vol_et_classe = moyennes_vol_et_classe.round(2)

# ==============================================================================
# 3. FILTRAGE ET ENTRAINEMENT SUR LE DATASET TRAIN
# ==============================================================================
df_train_pred = df_train[df_train['Age'] >= 18].copy()
y_train = df_train_pred['satisfaction'].astype('category').cat.codes

cols_prediction = [
    'Customer Type',
    'Age', 
    'Class',
    'Flight_Category', 
    'Departure Delay in Minutes', 
    'Arrival Delay in Minutes'
]
X_train = df_train_pred[cols_prediction].copy()

X_train['Customer Type'] = X_train['Customer Type'].astype('category').cat.codes
X_train['Class'] = X_train['Class'].astype('category').cat.codes
X_train['Flight_Category'] = X_train['Flight_Category'].astype('category').cat.codes
X_train = X_train.fillna(X_train.median())

model_foret = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model_foret.fit(X_train, y_train)

# ==============================================================================
# 4. CHARGEMENT ET PREDICTIONS SUR LE DATASET TEST
# ==============================================================================
df_test = pd.read_csv("test.csv", index_col=0)
df_test = df_test.drop(columns=['id', 'Gender'], errors='ignore')

df_test['Flight_Category'] = pd.cut(df_test['Flight Distance'], bins=miles_limits, labels=flight_categories)
df_test_pred = df_test[df_test['Age'] >= 18].copy()

X_test = df_test_pred[cols_prediction].copy()
X_test['Customer Type'] = X_test['Customer Type'].astype('category').cat.codes
X_test['Class'] = X_test['Class'].astype('category').cat.codes
X_test['Flight_Category'] = X_test['Flight_Category'].astype('category').cat.codes
X_test = X_test.fillna(X_train.median())

y_test_real_text = df_test_pred['satisfaction'].astype(str)

probas_test = model_foret.predict_proba(X_test)[:, 0]
seuil_top_10 = np.percentile(probas_test, 90)
predictions_test_texte = np.where(probas_test >= seuil_top_10, "neutral or dissatisfied", "satisfied")

df_liste_predictions = pd.DataFrame({
    'Customer Type': df_test_pred['Customer Type'],
    'Age': df_test_pred['Age'],
    'Class': df_test_pred['Class'],
    'Flight_Category': df_test_pred['Flight_Category'],
    'Departure Delay': X_test['Departure Delay in Minutes'],
    'Arrival Delay': X_test['Arrival Delay in Minutes'],
    'PREDICTION DE L\'IA (Statut Prédit)': predictions_test_texte,
    'SATISFACTION RÉELLE (Vérité du fichier test)': y_test_real_text.values,
    'Action Équipe au Sol': np.where(predictions_test_texte == "neutral or dissatisfied", "⚠️ À ABORDER IMMÉDIATEMENT", "Ras")
})
df_liste_predictions = df_liste_predictions.sort_values(by='PREDICTION DE L\'IA (Statut Prédit)', ascending=False)

# ==============================================================================
# 5. CALCULS DES METRIQUES DE PERFORMANCE (ONGLET 4)
# ==============================================================================
df_flaggues = df_liste_predictions[df_liste_predictions['Action Équipe au Sol'] == "⚠️ À ABORDER IMMÉDIATEMENT"]
total_abordes = len(df_flaggues)
vrais_insatisfaits = np.sum(df_flaggues['SATISFACTION RÉELLE (Vérité du fichier test)'] == "neutral or dissatisfied")
taux_precision = (vrais_insatisfaits / total_abordes) * 100

df_perf_excel = pd.DataFrame({
    "Métrique de Contrôle Qualité": [
        "Nombre de passagers abordés à la passerelle (Top 10%)",
        "Nombre de vrais mécontents correctement interceptés",
        "Pourcentage de réussite du modèle (Précision terrain)"
    ],
    "Valeur": [
        f"{total_abordes}",
        f"{vrais_insatisfaits}",
        f"{taux_precision:.2f} %"
    ]
})

# ==============================================================================
# 6. EXPORT DU LIVRABLE EXCEL FINAL
# ==============================================================================
nom_fichier = f"LIVRABLE_FINAL_{int(time.time())}.xlsx"

with pd.ExcelWriter(nom_fichier, engine="openpyxl") as writer:
    moyennes_par_age.to_excel(writer, sheet_name="1. Moyennes par Age", index=True)
    moyennes_vol_et_classe.to_excel(writer, sheet_name="2. Moyennes Vol et Classe", index=True)
    df_liste_predictions.to_excel(writer, sheet_name="3. Predictions Fichier Test", index=False)
    df_perf_excel.to_excel(writer, sheet_name="4. Performance IA", index=False)

print(f"Export terminé avec succès : {nom_fichier}")

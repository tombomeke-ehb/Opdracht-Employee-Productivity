import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# 1. DATA LADEN & VOORBEREIDEN
# ==========================================
print("Bezig met data laden...")
df = pd.read_csv('Extended_Employee_Performance_and_Productivity_Data.csv')

# Data kopiëren voor ML bewerkingen
df_ml = df.copy()
le = LabelEncoder()

# Categorische kolommen omzetten naar getallen
categorical_cols = df_ml.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_ml[col] = le.fit_transform(df_ml[col])

# Zorg dat 'Resigned' 0 of 1 is
df_ml['Resigned'] = df_ml['Resigned'].astype(int)

# Algemene instellingen voor grafieken
sns.set_theme(style="whitegrid")


# ==========================================
# DEEL 1: DESCRIPTIEVE ANALYSE (Slide 3)
# ==========================================
print("Genereren: 1_Samenstelling_Personeelsbestand.png")
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

# 1a. Aantal per afdeling
sns.countplot(y='Department', data=df, ax=axes1[0], palette='viridis', 
              order=df['Department'].value_counts().index)
axes1[0].set_title('Aantal Medewerkers per Afdeling')

# 1b. Leeftijd
sns.histplot(df['Age'], bins=20, kde=True, ax=axes1[1], color='skyblue')
axes1[1].set_title('Leeftijdsverdeling')

# 1c. Salaris
sns.boxplot(x='Monthly_Salary', y='Department', data=df, ax=axes1[2], 
            palette='viridis', order=df['Department'].value_counts().index)
axes1[2].set_title('Salarisverdeling per Afdeling')

plt.tight_layout()
plt.savefig('1_Samenstelling_Personeelsbestand.png')
plt.close()


# ==========================================
# DEEL 2: INVLOED OP VERTREK (Slide 4)
# ==========================================
print("Genereren: 2_Invloed_op_Vertrek.png")
plt.figure(figsize=(12, 7))

# Alleen numerieke kolommen selecteren
numeric_df = df_ml.select_dtypes(include=['number'])

# Employee_ID verwijderen (is ruis)
if 'Employee_ID' in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=['Employee_ID'])

# Correlatie met 'Resigned' berekenen
corr_data = numeric_df.corr()['Resigned'].drop('Resigned').sort_values()

# Labels netjes maken (geen underscores, hoofdletters)
corr_data.index = corr_data.index.str.replace('_', ' ').str.title()

# Kleuren bepalen: Rood = Veroorzaakt vertrek, Groen = Houdt mensen vast
colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in corr_data.values]

sns.barplot(x=corr_data.values, y=corr_data.index, palette=colors)

plt.title('Wat drijft mensen tot vertrek? (Correlatie Analyse)', fontsize=16)
plt.xlabel('Correlatie Coëfficiënt (Links = Retentie, Rechts = Vertrekrisico)', fontsize=12)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.savefig('2_Invloed_op_Vertrek.png')
plt.close()


# ==========================================
# DEEL 3: ML - TEVREDENHEID (Slide 5)
# ==========================================
print("Genereren: 3_ML_Tevredenheid_Drivers.png")
features_sat = ['Work_Hours_Per_Week', 'Overtime_Hours', 'Remote_Work_Frequency', 
                'Team_Size', 'Training_Hours', 'Monthly_Salary']
X_sat = df_ml[features_sat]
y_sat = df_ml['Employee_Satisfaction_Score']

# BELANGRIJK: Schalen zodat coëfficiënten vergelijkbaar zijn
scaler = StandardScaler()
X_sat_scaled = scaler.fit_transform(X_sat)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sat_scaled, y_sat, test_size=0.2, random_state=42)

model_sat = LinearRegression()
model_sat.fit(X_train_s, y_train_s)

# Plotten
coef_df = pd.DataFrame({'Feature': features_sat, 'Coefficient': model_sat.coef_})
# Labels netjes maken
coef_df['Feature'] = coef_df['Feature'].str.replace('_', ' ').str.title()

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
plt.title('Impact factoren op Tevredenheid (Gestandaardiseerd)', fontsize=14)
plt.xlabel('Impact (Standaardafwijking)', fontsize=12)
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('3_ML_Tevredenheid_Drivers.png')
plt.close()


# ==========================================
# DEEL 4: ML - VERTREK VOORSPELLEN (Slide 6)
# ==========================================
print("Genereren: 4_ML_Verloop_Matrix.png & 5_ML_Verloop_Importance.png")
features_res = features_sat + ['Employee_Satisfaction_Score', 'Age']
X_res = df_ml[features_res]
y_res = df_ml['Resigned']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Random Forest gebruiken (krachtiger dan enkele boom)
model_res = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
model_res.fit(X_train_r, y_train_r)
y_pred_r = model_res.predict(X_test_r)

# 1. Confusion Matrix Plot
cm = confusion_matrix(y_test_r, y_pred_r)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: Random Forest Model', fontsize=14)
plt.xticks([0.5, 1.5], ['Blijft', 'Vertrekt'])
plt.yticks([0.5, 1.5], ['Blijft', 'Vertrekt'])
plt.ylabel('Werkelijkheid')
plt.xlabel('Voorspelling Model')
plt.tight_layout()
plt.savefig('4_ML_Verloop_Matrix.png')
plt.close()

# 2. Feature Importance Plot
imp_df = pd.DataFrame({'Feature': features_res, 'Importance': model_res.feature_importances_})
imp_df = imp_df.sort_values(by='Importance', ascending=False)
imp_df['Feature'] = imp_df['Feature'].str.replace('_', ' ').str.title()

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=imp_df, palette='magma')
plt.title('Belangrijkste voorspellers voor Vertrek', fontsize=14)
plt.tight_layout()
plt.savefig('5_ML_Verloop_Importance.png')
plt.close()


# ==========================================
# DEEL 5: DIEPTE ANALYSE - DENSITY PLOTS (Slide 7)
# ==========================================
print("Genereren: 6_Diepte_Analyse_Density.png")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

vars_to_plot = [
    ('Overtime_Hours', 'Overuren (Uur/Maand)'), 
    ('Employee_Satisfaction_Score', 'Tevredenheidsscore (1-5)'), 
    ('Training_Hours', 'Training (Uur/Jaar)')
]

# Hulpkolom voor labels
df['Status'] = df['Resigned'].map({True: 'Vertrokken', False: 'Gebleven'})

for i, (col, title) in enumerate(vars_to_plot):
    # Density plot (KDE) laat de verschuiving zien
    sns.kdeplot(data=df, x=col, hue='Status', fill=True, 
                common_norm=False, palette={'Vertrokken': '#e74c3c', 'Gebleven': '#2ecc71'}, 
                alpha=0.5, linewidth=2, ax=axes[i])
    
    axes[i].set_title(title, fontsize=14, fontweight='bold')
    axes[i].set_ylabel('Dichtheid')
    axes[i].set_xlabel('')

plt.suptitle('Deep Dive: De subtiele verschillen tussen Blijvers en Vertrekkers', fontsize=20, y=1.05)
plt.tight_layout()
plt.savefig('6_Diepte_Analyse_Density.png')
plt.close()

# ==========================================
# EXTRA SLIDE: VERLOOP PER AFDELING
# ==========================================
print("Genereren: Extra_Verloop_per_Afdeling.png")
plt.figure(figsize=(10, 6))

# 1. Data berekenen
attrition_per_dept = df.groupby('Department')['Resigned'].mean().sort_values(ascending=False) * 100

# 2. Kleuren: We gebruiken één basiskleur omdat er geen enorme uitschieters zijn.
# We geven de top 3 een iets donkerdere tint om de nuance te tonen.
colors = ['#34495e' if i < 3 else '#95a5a6' for i in range(len(attrition_per_dept))]

# 3. Plotten
ax = sns.barplot(x=attrition_per_dept.values, y=attrition_per_dept.index, palette=colors)

# 4. Styling
plt.title('Verloop is Bedrijfsbreed (Systemisch Probleem)', fontsize=16, fontweight='bold')
plt.xlabel('Verlooppercentage (%)', fontsize=12)

# Inzoomen op de relevante range om de kleine verschillen te tonen
plt.xlim(8, 11.5)

# Percentages toevoegen
for i, v in enumerate(attrition_per_dept.values):
    ax.text(v + 0.05, i, f"{v:.1f}%", va='center', fontweight='bold', color='black')

plt.tight_layout()
plt.savefig('Extra_Verloop_per_Afdeling.png')
plt.close()

print("Klaar! Alle 6 afbeeldingen zijn gegenereerd.")
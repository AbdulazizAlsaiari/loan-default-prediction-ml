import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. تحميل البيانات
df = pd.read_csv("data/Loan_default.csv")

# 2. حذف عمود LoanID
df = df.drop('LoanID', axis=1)

# 3. تحويل الأعمدة الثنائية من Yes/No إلى 1/0
df['HasMortgage'] = df['HasMortgage'].map({'Yes': 1, 'No': 0})
df['HasCoSigner'] = df['HasCoSigner'].map({'Yes': 1, 'No': 0})
df['HasDependents'] = df['HasDependents'].map({'Yes': 1, 'No': 0})

# 4. One-Hot Encoding
df = pd.get_dummies(df, columns=['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose'], drop_first=True)

# 5. فصل X عن y
X = df.drop('Default', axis=1)
y = df['Default']

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 7. Feature Scaling على الأعمدة الرقمية فقط
numeric_columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# 8. تدريب LogisticRegression
log_reg_balanced = LogisticRegression(random_state=42, class_weight='balanced')
log_reg_balanced.fit(X_train, y_train)

# 9. التقييم
y_pred_balanced = log_reg_balanced.predict(X_test)

accuracy_balanced = accuracy_score(y_test, y_pred_balanced)
print("Accuracy (Logistic Regression Balanced):", accuracy_balanced)

print("Classification Report (Balanced):")
print(classification_report(y_test, y_pred_balanced))

# 10. حفظ النموذج والـ scaler وأسماء الأعمدة
os.makedirs('models', exist_ok=True)

joblib.dump(log_reg_balanced, 'models/loan_default_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(list(X_train.columns), 'models/feature_columns.pkl')

print("تم حفظ النموذج والـ scaler وأسماء الأعمدة بنجاح!")

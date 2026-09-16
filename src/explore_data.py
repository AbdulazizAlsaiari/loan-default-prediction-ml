# نستورد مكتبة pandas ونعطيها اسم مختصر "pd" (اتفاقية شائعة بكل مشاريع بايثون)
import pandas as pd

# نقرأ ملف الـ CSV من مجلد data ونخزنه بمتغير اسمه df (اختصار DataFrame، وهو الشكل الجدولي اللي pandas يخزن فيه البيانات)
df = pd.read_csv("data/Loan_default.csv")

# نطبع أول 5 صفوف من الجدول عشان نشوف شكل البيانات بسرعة
print(df.head())

# نطبع معلومات عامة عن الأعمدة: عددها، أسماءها، نوع كل عمود (رقم/نص)، وهل فيها قيم مفقودة
print(df.info())

# نطبع عدد الصفوف والأعمدة (كم مقترض بالبيانات، وكم متغير عندنا)
print(df.shape)
# نطبع عدد كل قيمة بعمود Default (كم شخص تعثر وكم ما تعثر)
print(df['Default'].value_counts())

# نفس الشي بس كنسبة مئوية، عشان نشوف التوازن بشكل أوضح
print(df['Default'].value_counts(normalize=True))
# نطبع ملخص إحصائي (متوسط، أصغر، أكبر، إلخ) لكل الأعمدة الرقمية دفعة وحدة
print(df.describe())
# نشيل عمود LoanID لأنه مجرد معرف فريد، ما يفيد النموذج بالتنبؤ
# axis=1 يعني "احذف عمود" (لو axis=0 كان بيحذف صف)
df = df.drop('LoanID', axis=1)

# نتأكد إن العمود انشال فعلاً، نطبع أسماء الأعمدة المتبقية
print(df.columns)
# نحدد قائمة بأسماء الأعمدة النصية اللي نبي نفحصها
text_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'LoanPurpose', 'HasCoSigner', 'HasDependents']

# نلف على كل عمود بالقائمة، ونطبع اسمه والقيم الفريدة الموجودة فيه
for col in text_columns:
    print(col, ":", df[col].unique())
    # المرحلة الأولى: نحول الأعمدة الثنائية (Yes/No) لأرقام (1/0)
# map() تاخذ قاموس (Dictionary) يوضح كل قيمة نصية تتحول لأي رقم
df['HasMortgage'] = df['HasMortgage'].map({'Yes': 1, 'No': 0})
df['HasCoSigner'] = df['HasCoSigner'].map({'Yes': 1, 'No': 0})
df['HasDependents'] = df['HasDependents'].map({'Yes': 1, 'No': 0})

# المرحلة الثانية: نطبق One-Hot Encoding على الأعمدة اللي فيها أكثر من قيمتين
# pd.get_dummies() دالة جاهزة بـ pandas تسوي التحويل تلقائياً
# columns=[...] نحدد بالضبط أي أعمدة نبي نحولها
# drop_first=True نحذف عمود واحد من كل مجموعة (تفصيل تقني نشرحه تحت)
df = pd.get_dummies(df, columns=['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose'], drop_first=True)

# نطبع أسماء الأعمدة الجديدة بعد التحويل، نتأكد شكل الجدول تغير صح
print(df.columns)

# نطبع أول 5 صفوف نشوف شكل البيانات بعد كل التحويلات
print(df.head())
# نستورد الدالة الجاهزة اللي تسوي التقسيم (من مكتبة scikit-learn)
from sklearn.model_selection import train_test_split

# X = كل الأعمدة ما عدا Default (يعني كل المعلومات اللي النموذج بيستخدمها للتنبؤ)
# axis=1 يعني نحذف عمود (نفس الفكرة اللي استخدمناها قبل مع drop)
X = df.drop('Default', axis=1)

# y = عمود Default لحاله بس (اللي بنحاول نتنبأ فيه)
y = df['Default']

# نقسم البيانات: 80% تدريب، 20% اختبار
# test_size=0.2 يعني 20% للاختبار
# stratify=y يحافظ على نفس نسبة 88%/12% بكل مجموعة
# random_state=42 يثبت طريقة "الخلط العشوائي" عشان لو شغلنا الكود مرة ثانية نحصل نفس التقسيم بالضبط (رقم 42 اصطلاح شائع بين المبرمجين، أي رقم ثابت يشتغل)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# نطبع حجم كل مجموعة نتأكد التقسيم صار صح
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# نتأكد إن نسبة Default محفوظة بنفس الشكل بمجموعة التدريب
print("نسبة Default بمجموعة التدريب:")
print(y_train.value_counts(normalize=True))

# ونفس الشي بمجموعة الاختبار
print("نسبة Default بمجموعة الاختبار:")
print(y_test.value_counts(normalize=True))
# نستورد أداة التوحيد من scikit-learn
from sklearn.preprocessing import StandardScaler

# نحدد قائمة بأسماء الأعمدة الرقمية الأصلية اللي تحتاج Scaling
# (نستبعد أعمدة الـ One-Hot Encoding والأعمدة الثنائية لأنها أصلاً 0/1)
numeric_columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# ننشئ كائن (Object) من StandardScaler، فارغ لسا ما اتدرب على أي بيانات
scaler = StandardScaler()

# fit_transform(): يحسب المتوسط والانحراف المعياري من X_train، ثم يطبق التحويل عليها مباشرة
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])

# transform() بس (بدون fit): يطبق نفس الأرقام المحسوبة من X_train على X_test، بدون ما يعيد حسابها من جديد
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# نتأكد الشكل تغير صح، نطبع أول 5 صفوف من الأعمدة الرقمية بعد التوحيد
print(X_train[numeric_columns].head())

# نطبع وصف إحصائي سريع، نتأكد المتوسط قريب من صفر
print(X_train[numeric_columns].describe())
# نستورد نموذج Logistic Regression من scikit-learn
from sklearn.linear_model import LogisticRegression

# نستورد أدوات لقياس أداء النموذج (بالضبط زي اللي استخدمتها بمشروع Churn)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ننشئ النموذج (لسا فاضي، ما اتدرب على أي بيانات)
# random_state=42 لضمان نفس النتيجة لو شغلنا الكود مرة ثانية
log_reg = LogisticRegression(random_state=42)

# ندرب النموذج: يتعلم الأنماط من X_train وy_train (بيانات التدريب فقط)
log_reg.fit(X_train, y_train)

# نستخدم النموذج المدرب يتنبأ على X_test (بيانات ما شافها قبل إطلاقاً)
y_pred = log_reg.predict(X_test)

# نقيس دقة النموذج: نقارن تنبؤاته (y_pred) بالقيم الحقيقية (y_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (Logistic Regression Baseline):", accuracy)

# نطبع تقرير مفصل أكثر: Precision, Recall, F1-score لكل فئة (0 و1)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# نطبع مصفوفة الالتباس (Confusion Matrix): توضح بالتفصيل أين أخطأ النموذج
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# ننشئ نموذج Logistic Regression ثاني، بس هالمرة نضيف class_weight='balanced'
# هذا يخلي النموذج يعطي "وزن" أكبر لأخطاء فئة الأقلية (Default=1) أثناء التدريب
# بدل ما يتعامل مع الفئتين بنفس الأهمية رغم إن واحدة أكبر من الثانية بكثير
log_reg_balanced = LogisticRegression(random_state=42, class_weight='balanced')

# ندرب النموذج الثاني على نفس بيانات التدريب
log_reg_balanced.fit(X_train, y_train)

# نتنبأ على نفس بيانات الاختبار
y_pred_balanced = log_reg_balanced.predict(X_test)

# نقيس الأداء بنفس الطريقة عشان نقارن بسهولة
accuracy_balanced = accuracy_score(y_test, y_pred_balanced)
print("Accuracy (Logistic Regression Balanced):", accuracy_balanced)

print("Classification Report (Balanced):")
print(classification_report(y_test, y_pred_balanced))

print("Confusion Matrix (Balanced):")
print(confusion_matrix(y_test, y_pred_balanced))
# نستورد نموذج Random Forest من scikit-learn
from sklearn.ensemble import RandomForestClassifier

# ننشئ النموذج
# n_estimators=100 يعني نبني 100 شجرة قرار مختلفة (القيمة الافتراضية الشائعة)
# random_state=42 لضمان نفس النتيجة لو شغلنا الكود مرة ثانية
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# ندرب النموذج على بيانات التدريب
rf_model.fit(X_train, y_train)

# نتنبأ على بيانات الاختبار
y_pred_rf = rf_model.predict(X_test)

# نقيس الأداء بنفس المقاييس المعتادة
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy (Random Forest):", accuracy_rf)

print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))
# ننشئ نموذج Random Forest رابع، بس هالمرة مع class_weight='balanced'
rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# ندرب النموذج
rf_balanced.fit(X_train, y_train)

# نتنبأ على بيانات الاختبار
y_pred_rf_balanced = rf_balanced.predict(X_test)

# نقيس الأداء
accuracy_rf_balanced = accuracy_score(y_test, y_pred_rf_balanced)
print("Accuracy (Random Forest Balanced):", accuracy_rf_balanced)

print("Classification Report (Random Forest Balanced):")
print(classification_report(y_test, y_pred_rf_balanced))

print("Confusion Matrix (Random Forest Balanced):")
print(confusion_matrix(y_test, y_pred_rf_balanced))
# ننشئ جدول (DataFrame) يربط بين اسم كل عمود ومعامله (Coefficient) من نموذج LR Balanced
# log_reg_balanced.coef_[0] يرجع قائمة بالمعاملات بنفس ترتيب أعمدة X_train
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_reg_balanced.coef_[0]
})

# نرتب الجدول تنازلياً حسب القيمة المطلقة للمعامل (أكبر تأثير، سواء موجب أو سالب، يطلع فوق)
feature_importance['AbsCoefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values('AbsCoefficient', ascending=False)

# نطبع أهم 10 عوامل مؤثرة على القرار
print("أهم 10 عوامل مؤثرة على التنبؤ بالتعثر:")
print(feature_importance.head(10))
# نستورد النموذج الجديد من scikit-learn
from sklearn.ensemble import HistGradientBoostingClassifier

# ننشئ النموذج مع class_weight='balanced' مباشرة من البداية
gb_model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')

# ندرب النموذج
gb_model.fit(X_train, y_train)

# نتنبأ على بيانات الاختبار
y_pred_gb = gb_model.predict(X_test)

# نقيس الأداء بنفس المقاييس المعتادة
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Accuracy (Gradient Boosting Balanced):", accuracy_gb)

print("Classification Report (Gradient Boosting Balanced):")
print(classification_report(y_test, y_pred_gb))

print("Confusion Matrix (Gradient Boosting Balanced):")
print(confusion_matrix(y_test, y_pred_gb))
# نستورد مكتبة joblib لحفظ النموذج
import joblib

# نستورد مكتبة os للتعامل مع المجلدات
import os

# ننشئ مجلد اسمه "models" لو مو موجود أصلاً
os.makedirs('models', exist_ok=True)

# نحفظ النموذج المدرب (LR Balanced) كملف .pkl
joblib.dump(log_reg_balanced, 'models/loan_default_model.pkl')

# نحفظ الـ scaler كمان
joblib.dump(scaler, 'models/scaler.pkl')

# نحفظ قائمة أسماء الأعمدة بالترتيب الصحيح
joblib.dump(list(X_train.columns), 'models/feature_columns.pkl')

print("تم حفظ النموذج والـ scaler وأسماء الأعمدة بنجاح!")
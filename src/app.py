# نستورد Flask ودالة jsonify (تحول قاموس بايثون لصيغة JSON قبل الإرسال) و render_template لعرض صفحات HTML
from flask import Flask, jsonify, render_template

# ننشئ "تطبيق" Flask، وهو أساس أي API نبنيه
app = Flask(__name__)

# نعرّف "route" - يعني عنوان معين، لما حد يزوره، يشتغل الكود اللي تحته
# '/' يعني الصفحة الرئيسية (الأساسية) للـ API
# methods=['GET'] يعني هذا العنوان يستقبل طلبات من نوع "GET" (طلب قراءة/عرض بسيط)
@app.route('/', methods=['GET'])
def home():
    # نرجع رسالة بسيطة بصيغة JSON، نتأكد إن الـ API شغال
    return jsonify({"message": "API شغال بنجاح!"})



# نستورد المكتبات اللي بنحتاجها: joblib لتحميل النموذج، pandas لتنظيم البيانات، request لاستقبال بيانات الطلب
import joblib
import pandas as pd
from flask import request

# نحمّل النموذج والـ scaler وأسماء الأعمدة - نسويها مرة وحدة بس عند بدء تشغيل الخادم (مو مع كل طلب، توفيراً للوقت)
model = joblib.load('models/loan_default_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

# نحدد قائمة الأعمدة الرقمية اللي تحتاج Scaling (نفس القائمة اللي استخدمناها بالتدريب)
numeric_columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# نعرّف عنوان جديد '/predict'، يستقبل طلبات من نوع POST (طلب "إرسال بيانات"، مختلف عن GET)
@app.route('/predict', methods=['POST'])
def predict():
    # نستقبل البيانات المرسلة بصيغة JSON، ونحولها لقاموس بايثون
    data = request.get_json()
    
    # نحول القاموس لجدول (DataFrame) بصف واحد، بنفس شكل بياناتنا الأصلية
    input_df = pd.DataFrame([data])
    
    # نطبق One-Hot Encoding بنفس طريقة التدريب بالضبط
    input_df = pd.get_dummies(input_df, columns=['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose'])
    
    # نتأكد إن كل الأعمدة اللي درّبنا عليها النموذج موجودة، ولو ناقصة نضيفها بقيمة صفر
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # نرتب الأعمدة بنفس الترتيب بالضبط اللي درّبنا عليه النموذج
    input_df = input_df[feature_columns]
    
    # نطبق نفس الـ Scaling على الأعمدة الرقمية
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
    
    # نطلب من النموذج يتنبأ (0 أو 1)
    prediction = model.predict(input_df)[0]
    
    # نطلب من النموذج نسبة الاحتمال (كم بالمئة واثق من قراره)
    probability = model.predict_proba(input_df)[0][1]
    
    # نجهز الرد بصيغة JSON واضحة
    result = {
        "prediction": "Default" if prediction == 1 else "No Default",
        "probability": round(float(probability), 4)
    }
    
    return jsonify(result)


# عنوان جديد '/form' لعرض واجهة ويب بسيطة (نموذج HTML) للتنبؤ عن طريق المتصفح
@app.route('/form', methods=['GET'])
def form():
    return render_template('index.html')


# نشغّل الخادم، فقط لو شغّلنا هذا الملف مباشرة (مو لو تم استيراده من ملف ثاني)
# debug=True يعطينا رسائل خطأ مفصلة أثناء التطوير (نلغيها لاحقاً وقت الإنتاج الفعلي)
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
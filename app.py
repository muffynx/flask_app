from flask import Flask, render_template, request
import pickle
import numpy as np

# สร้างแอป Flask
app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)

    # ทำการทำนายโดยใช้โมเดล
    prediction = model.predict(features)

    # ส่งผลลัพธ์ไปยังหน้าเว็บ
    return render_template('index.html', prediction_text='Prediction: {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)

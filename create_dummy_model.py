import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# สร้างชุดข้อมูลจำลอง
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# สร้างโมเดล Logistic Regression จำลอง
model = LogisticRegression()
model.fit(X, y)

# บันทึกโมเดลลงไฟล์ model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model จำลองถูกสร้างและบันทึกเป็น model.pkl แล้ว!")

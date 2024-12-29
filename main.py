from flask import Flask, request, render_template, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os

# تحميل الموديل ResNet50 من torchvision
model = models.resnet50(pretrained=False, num_classes=9)  # تأكد من عدد الفئات 9
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))  # تحميل الأوزان من الملف
model.eval()  # وضع الموديل في وضع التقييم

# قائمة بأسماء الكلاسات
class_names = [
    '1', '10', '10 (new)', '100', '20', 
    '20 (new)', '200', '5', '50'
]

# تعريف التحويلات اللازمة للتنبؤ
predict_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[-0.0932, -0.0971, -0.1260], std=[0.5091, 0.4912, 0.4931])  # نفس القيم المستخدمة في التدريب
])

# إنشاء تطبيق Flask
app = Flask(__name__)

# دالة لتحميل الصورة والتنبؤ بالفئة
def predict_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # تطبيق التحويلات الثابتة على الصورة
    img_tensor = predict_transforms(img).unsqueeze(0)  # إضافة بعد إضافي ليكون [batch_size, C, H, W]
    
    # تنبؤ الفئة باستخدام الموديل
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
    
    # إرجاع اسم الكلاس بدلاً من الرقم
    predicted_class_name = class_names[predicted_class.item()]
    
    return predicted_class_name

# الصفحة الرئيسية لرفع الصور
@app.route('/')
def index():
    return render_template('index.html')

# استلام الصورة من المستخدم وعرض النتيجة
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    img_bytes = file.read()
    
    # التنبؤ بالفئة
    predicted_class = predict_image(img_bytes)
    
    # إعادة عرض النتيجة للمستخدم
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)

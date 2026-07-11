import json

# 1. حدد الأبعاد القديمة اللي كنت شغال عليها
OLD_WIDTH = 1240   # غير الرقم ده لعرض الصورة القديمة
OLD_HEIGHT = 796  # غير الرقم ده لطول الصورة القديمة

# الأبعاد الجديدة بتاعتك
NEW_WIDTH = 1000
NEW_HEIGHT = 645

# 2. حساب معاملات الضرب (Scale Factors)
factor_x = NEW_WIDTH / OLD_WIDTH
factor_y = NEW_HEIGHT / OLD_HEIGHT

print(f"معامل العرض (X) = {factor_x}")
print(f"معامل الطول (Y) = {factor_y}")

# 3. قراءة ملف الإعدادات القديم
# افترضنا إن اسمه config.json
with open('front_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 4. تحديث الإحداثيات
new_config = {}
for key, value in config.items():
    # بنفترض إن الإحداثيات متسجلة في شكل [x, y]
    if isinstance(value, list) and len(value) == 2:
        new_x = int(value[0] * factor_x)
        new_y = int(value[1] * factor_y)
        new_config[key] = [new_x, new_y]
    
    # لو عندك حجم الخط متسجل جوه الكونفيج مثلا كـ dictionary
    elif isinstance(value, dict) and "x" in value and "y" in value:
        new_config[key] = value.copy()
        new_config[key]["x"] = int(value["x"] * factor_x)
        new_config[key]["y"] = int(value["y"] * factor_y)
        
        # لو كاتب حجم الخط جوه الكونفيج، اضربه في معامل الطول
        if "size" in value:
            new_config[key]["size"] = int(value["size"] * factor_y)
    else:
        # أي داتا تانية سيبها زي ما هي
        new_config[key] = value

# 5. حفظ الملف الجديد
with open('config_new.json', 'w', encoding='utf-8') as f:
    json.dump(new_config, f, ensure_ascii=False, indent=4)

print("تم إنشاء ملف config_new.json بالإحداثيات المظبوطة على المقاس الجديد!")
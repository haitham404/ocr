import json
import random

# ==========================================
# 1. قوائم البيانات الأساسية
# ==========================================
MALE_NAMES = [
    "أحمد", "محمد", "محمود", "حسين", "مصطفى", "خالد", "عمر", "طارق", "كريم", "يوسف", 
    "علي", "حسن", "إبراهيم", "عادل", "سمير", "ياسين", "يحيى", "وليد", "ماجد", "تامر", 
    "وائل", "شريف", "حازم", "ياسر", "هشام", "بهاء", "أيمن", "أشرف", "عصام", "مروان", 
    "هاني", "سعد", "سعيد", "جمال", "كمال", "عبد الله", "عبد الرحمن", "عبد العزيز", 
    "عبد الفتاح", "طه", "رامي", "شادي", "أمير", "إيهاب", "باسم", "حسام", "سامح", "صلاح", 
    "عاطف", "علاء", "عماد", "مجدي", "ممدوح", "نادر", "هيثم", "يونس", "زين"
]

FEMALE_NAMES = [
    "فاطمة", "سارة", "مريم", "نورهان", "هند", "منى", "دعاء", "ياسمين", "ريهام", "سلمى", 
    "دينا", "شيماء", "آية", "إسراء", "هاجر", "عائشة", "خديجة", "زينب", "رقية", "هبة", 
    "نهى", "مها", "ريم", "ليلى", "ندى", "أميرة", "إيمان", "مروة", "شيرين", "حنان", 
    "رحاب", "أسماء", "أماني", "إيناس", "هالة", "رشا", "غادة", "مي", "يمنى", "شهد", 
    "جودي", "حلا", "فريدة", "يارا", "سهر", "سعاد", "عفاف", "فاتن", "نجلاء", "نرمين", 
    "نجوى", "هدى", "وفاء", "ياسمين", "بسنت", "تسنيم", "جيهان"
]

GOV_MAIN = [{"name": "القاهرة", "code": "01"}, {"name": "الإسكندرية", "code": "02"}, {"name": "الجيزة", "code": "21"}]
GOV_OTHER = [
    {"name": "بورسعيد", "code": "03"}, {"name": "السويس", "code": "04"}, {"name": "الدقهلية", "code": "12"},
    {"name": "الشرقية", "code": "13"}, {"name": "القليوبية", "code": "14"}, {"name": "كفر الشيخ", "code": "15"},
    {"name": "الغربية", "code": "16"}, {"name": "المنوفية", "code": "17"}, {"name": "البحيرة", "code": "18"},
    {"name": "الإسماعيلية", "code": "19"}, {"name": "بني سويف", "code": "22"}, {"name": "الفيوم", "code": "23"},
    {"name": "المنيا", "code": "24"}, {"name": "أسيوط", "code": "25"}, {"name": "سوهاج", "code": "26"},
    {"name": "قنا", "code": "27"}, {"name": "أسوان", "code": "28"}, {"name": "الأقصر", "code": "29"},
    {"name": "البحر الأحمر", "code": "31"}, {"name": "الوادي الجديد", "code": "32"}, {"name": "مطروح", "code": "33"},
    {"name": "شمال سيناء", "code": "34"}, {"name": "جنوب سيناء", "code": "35"}
]

REGIONS_MAP = {
    "القاهرة": ["وسط البلد", "جاردن سيتي", "الزمالك", "المعادي", "مدينة نصر", "مصر الجديدة", "النزهة", "التجمع الخامس", "الرحاب", "الشروق", "العبور", "شبرا", "روض الفرج", "السيدة زينب", "الحسين", "العباسية", "المطرية", "عين شمس", "حلوان", "المقطم", "دار السلام", "البساتين", "المرج", "الزيتون", "حدائق القبة"],
    "الجيزة": ["الدقي", "المهندسين", "العجوزة", "الهرم", "فيصل", "الوراق", "إمبابة", "بولاق الدكرور", "٦ أكتوبر", "الشيخ زايد", "حدائق الأهرام", "العمرانية", "الطالبية", "أرض اللواء"],
    "الإسكندرية": ["سموحة", "ميامي", "سيدي بشر", "العصافرة", "المندرة", "كامب شيزار", "كليوباترا", "رشدي", "جليم", "لوران", "سان ستيفانو", "محطة الرمل", "المنشية", "العطارين", "سبورتنج", "الإبراهيمية", "الشاطبي", "ستانلي", "سيدي جابر", "العجمي", "البيطاش"]
}

STREETS_NORMAL = [
    "شارع طلعت حرب", "شارع قصر النيل", "شارع رمسيس", "شارع الجلاء", "شارع شبرا", 
    "شارع بورسعيد", "شارع كورنيش النيل", "شارع الجمهورية", "شارع الجيش", "شارع الأزهر", 
    "شارع المعز", "شارع محمد محمود", "شارع الفلكي", "شارع عبد الخالق ثروت", "شارع شريف", 
    "شارع جامعة الدول العربية", "شارع البطل أحمد عبد العزيز", "شارع مصدق", "شارع التحرير", 
    "شارع السودان", "شارع النيل", "شارع الهرم", "شارع الملك فيصل", "شارع مراد", 
    "شارع عباس العقاد", "شارع مكرم عبيد", "شارع الطيران", "شارع الميرغني", "شارع صلاح سالم", 
    "شارع أبو قير", "شارع جمال عبد الناصر", "شارع سعد زغلول", "شارع جيهان", "شارع المحطة"
]

STREETS_NUMBERED = [
    "شارع ٢٦ يوليو", "شارع ١٥ مايو", "شارع ٦ أكتوبر", "الشارع العاشر", "الشارع الأول", 
    "الشارع الثاني", "الشارع الثالث", "شارع ٩", "شارع ٤٥", "شارع ١٠", "شارع ٢٠", 
    "شارع ٣٠", "شارع ٥٠", "شارع ٧٧", "شارع ١٠٥", "شارع ٢٣٣", "شارع التسعين الشمالي"
]

OCCUPATIONS = [
    "مهندس", "طبيب", "محاسب", "مدرس", "محام", "موظف", "باحث",
    "أعمال حرة", "سباك", "نجار", "كهربائي", "سائق",
    "طالب",
    "حاصل على بكالوريوس هندسة", "حاصل على بكالوريوس تجارة", "حاصل على ليسانس حقوق", "حاصل على ليسانس آداب", "حاصل على بكالوريوس الطب والجراحة",
    "حاصل على دبلوم المعهد الفني الصناعي", "حاصل على دبلوم معهد فني تجاري", "حاصل على دبلوم معهد فني صحي",
    "حاصل على دبلوم المدارس الثانوية الصناعية", "حاصل على دبلوم المدارس الثانوية التجارية", "حاصل على شهادة الثانوية العامة",
    "حاصل على الشهادة الإعدادية", "حاصل على الشهادة الابتدائية", "حاصل على شهادة محو الأمية",
    "حاصل على درجة الماجستير", "حاصل على درجة الدكتوراه"
]

# ==========================================
# 2. دوال المساعدة
# ==========================================
def to_arabic_nums(text):
    arabic_numbers = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
    return str(text).translate(arabic_numbers)

def generate_national_id(century, year, month, day, gov_code, is_male):
    century_digit = "2" if century == 19 else "3"
    yy = str(year)[-2:].zfill(2)
    mm = str(month).zfill(2)
    dd = str(day).zfill(2)
    seq = str(random.randint(1, 999)).zfill(3)
    gender_digit = random.choice([1, 3, 5, 7, 9]) if is_male else random.choice([2, 4, 6, 8])
    check_digit = str(random.randint(1, 9))
    
    id_str = f"{century_digit}{yy}{mm}{dd}{gov_code}{seq}{gender_digit}{check_digit}"
    return to_arabic_nums(id_str)

# ==========================================
# 3. بناء الـ 1000 بطاقة
# ==========================================
dataset = []
total_records = 500

gov_selections = GOV_OTHER + GOV_MAIN
remaining_count = total_records - len(gov_selections)
main_count = int(remaining_count * 0.70)
other_count = remaining_count - main_count 

gov_selections += random.choices(GOV_MAIN, k=main_count)
gov_selections += random.choices(GOV_OTHER, k=other_count)
random.shuffle(gov_selections)

for i in range(total_records):
    is_male = i % 2 == 0
    gender_str = "ذكر" if is_male else "أنثى"
    
    # الأسماء
    first_name = random.choice(MALE_NAMES) if is_male else random.choice(FEMALE_NAMES)
    rest_name = " ".join(random.choices(MALE_NAMES, k=4))
    
    # العنوان
    building_num = random.randint(1, 9999)
    street = random.choice(STREETS_NUMBERED) if random.random() < 0.20 else random.choice(STREETS_NORMAL)
    
    selected_gov = gov_selections[i]
    gov_name = selected_gov['name']
    
    if gov_name in REGIONS_MAP:
        region = random.choice(REGIONS_MAP[gov_name])
    else:
        prefixes = ["مركز", "أول", "ثان", "بندر"]
        region = f"{random.choice(prefixes)} {gov_name}"
    
    # إزالة المنطقة من السطر الأول لتفادي التكرار
    address_line1 = f"{building_num} {street}"
    address_line1 = to_arabic_nums(address_line1)
    
    address_line2 = f"{region} - {gov_name}"
    
    # تواريخ الميلاد والرقم القومي
    # تواريخ الميلاد (من 1940 لحد 2011 كحد أقصى)
    year = random.randint(1940, 2011)
    century = 19 if year < 2000 else 20
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    
    national_id = generate_national_id(century, year, month, day, selected_gov['code'], is_male)
    
    birth_date = to_arabic_nums(f"{str(day).zfill(2)} / {str(month).zfill(2)} / {year}")
    # تواريخ الإصدار والانتهاء
    issue_year = random.randint(2015, 2026)
    issue_month = random.randint(1, 12)
    expiry_year = issue_year + 7
    expiry_day = random.randint(1, 28)
    
    issue_date = to_arabic_nums(f"{str(issue_month).zfill(2)} / {issue_year}")
    expiry_date = to_arabic_nums(f"البطاقة سارية حتى {str(expiry_day).zfill(2)} / {str(issue_month).zfill(2)} / {expiry_year}")
    
    # الديانة والحالة الاجتماعية
    religion = random.choices(["مسلم", "مسيحي", "يهودي"], weights=[80, 10, 10])[0]
    marital = random.choices(["أعزب", "متزوج", "مطلق", "أرمل"], weights=[30, 30, 30, 10])[0]
    
    if not is_male:
        marital = marital.replace("أعزب", "عزباء").replace("متزوج", "متزوجة").replace("مطلق", "مطلقة").replace("أرمل", "أرملة")
        
    # الوظيفة
    occupation = random.choice(OCCUPATIONS)
    if not is_male and random.random() < 0.15:
        occupation = "ربة منزل"
        
    record = {
        "name_first": first_name,
        "name_rest": rest_name,
        "street": address_line1,
        "address_rest": address_line2,
        "national_id": national_id,
        "birth_date": birth_date,
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "occupation": occupation,
        "gender": gender_str,
        "religion": religion,
        "marital_status": marital
    }
    dataset.append(record)

# ==========================================
# 4. حفظ الملف
# ==========================================
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Done, {total_records} Records have been created!!")
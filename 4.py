import streamlit as st
import torch
import json
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plant_disease_classifier import PlantDiseaseModel, predict_image
import warnings
import cv2
import time
import base64

# Ogohlantirishlarni o‘chirish
warnings.filterwarnings("ignore")

# Sahifa sozlamalari
st.set_page_config(
    page_title="O'simlik Kasalliklari Klassifikatori",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fon rasmini base64 formatiga o‘tkazish funksiyasi
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Fon rasmini yuklash
bg_image = get_base64_image("images/background.png")
bg_image1 = get_base64_image("images/fon.png")

# CSS uslublari
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    .stApp {{
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, rgba(212, 228, 212, 0.5) 0%, rgba(240, 247, 240, 0.5) 100%), url(data:image/png;base64,{bg_image1});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #000000;
        min-height: 100vh;
    }}

    .upload-section, .results-section {{
        background: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url(data:image/png;base64,{bg_image});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid #ffffff;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        color: #000000;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.6);
    }}
    .upload-section.real-time {{
        margin-top: 130px; /* bu yerda kerakli masofani sinab ko‘rishingiz mumkin */
    }}

    .main-header {{
        font-size: 3rem;
        color: #000000;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }}

    .sub-header {{
        font-size: 1.8rem;
        color: #000000;
        margin: 0.5rem 0;
        font-weight: 600;
        display: inline-block;
        vertical-align: middle;
        text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.6);
    }}
    .sub-header::before {{
        content: '🍃';
        display: inline-block;
        margin-right: 0.5rem;
        font-size: 1.8rem;
        vertical-align: middle;
    }}

    .info-text {{
        font-size: 1.2rem;
        color: #000000;
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin: 1rem auto;
        max-width: 90%;
    }}

    .prediction-header {{
        font-size: 2.2rem;
        color: #000000;
        margin-top: 1rem;
        font-weight: 600;
    }}

    .confidence-text {{
        font-size: 1.4rem;
        color: #000000;
        margin-bottom: 1rem;
    }}

    .disease-info-header {{
        font-size: 1.6rem;
        color: #000000;
        margin-top: 1.5rem;
        font-weight: 600;
    }}

    .disease-info-subheader {{
        font-size: 1.3rem;
        color: #000000;
        font-weight: 600;
        margin-top: 0.8rem;
    }}

    .stButton>button {{
        background: #ffffff;
        color: #333333;
        border: 0.5px solid #4caf50;
        padding: 0.8rem 2rem;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }}
    .stButton>button:hover {{
        background: #ffffff;
        color: #333333;
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.3);
    }}
    .stButton>button:active {{
        background: #4caf50;
        color: #ffffff;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }}

    .sidebar .sidebar-content {{
        background: linear-gradient(to bottom, #c8e6c9, #e8f5e9);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }}
</style>
""", unsafe_allow_html=True)

# Tasvirni kichraytirish funksiyasi
def resize_image(image, max_width=300):
    """Tasvirni kichraytirish va eni-bo‘yini moslashtirish"""
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return image

# Kameradan tasvir olish funksiyasi
def capture_image(placeholder):
    """Kameradan tasvirni olib saqlash funksiyasi"""
    if not os.path.exists("real_time_detection"):
        os.makedirs("real_time_detection")
    st.session_state["kameradan_olingan"] = False
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Kamera ochilmadi. Iltimos, kamera ulanganligini tekshiring.")
        return
    
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            st.error("Kamera tasvirini o‘qib bo‘lmadi.")
            cap.release()
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        resized_image = resize_image(image, max_width=300)  # Real vaqtdagi tasvirni kichraytirish
        placeholder.image(resized_image, caption="Real Vaqtdagi Tasvir", use_container_width=True)
        time.sleep(0.1)
    
    ret, frame = cap.read()
    if not ret:
        st.error("Tasvirni olishda xatolik yuz berdi.")
        cap.release()
        return
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    image_path = f"real_time_detection/captured_image_{current_time}.jpg"
    cv2.imwrite(image_path, frame)
    
    cap.release()
    cv2.destroyAllWindows()
    #st.success(f"Tasvir {image_path} ga muvaffaqiyatli saqlandi.")
    st.session_state["kameradan_olingan"] = True
    st.session_state.bashorat_natijalari = None
    st.session_state.rasm_qayta_ishlandi = False
    st.session_state.yuklangan_fayl = image_path

# Model va resurslarni yuklash
@st.cache_resource
def load_model_resources():
    """Model va zarur fayllarni yuklash funksiyasi"""
    with open("models/model_config.json", "r") as f:
        config = json.load(f)
    with open(config["class_names_path"], "r") as f:
        class_names = json.load(f)
    with open(config["label_encoder_path"], "rb") as f:
        label_encoder = pickle.load(f)
    with open(config["transform_path"], "rb") as f:
        transform = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes=len(class_names))
    model.load_state_dict(torch.load(config["model_path"], map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, transform, label_encoder, class_names, device, config

# Bashorat qilish funksiyasi
def predict_disease(image_file, model, transform, label_encoder, device):
    """Rasm uchun kasallik bashoratini qaytaruvchi funksiya"""
    with open("temp_upload.jpg", "wb") as f:
        f.write(image_file.getvalue())
    class_name, confidence, all_probabilities = predict_image(model, "temp_upload.jpg", transform, device, label_encoder)
    class_indices = np.argsort(all_probabilities)[::-1][:5]
    top_classes = [label_encoder.inverse_transform([idx])[0] for idx in class_indices]
    formatted_top_classes = [format_class_name(class_name) for class_name in top_classes]
    formatted_main_class = format_class_name(class_name)
    top_confidences = [all_probabilities[idx] * 100 for idx in class_indices]
    os.remove("temp_upload.jpg")
    return class_name, formatted_main_class, confidence, top_classes, formatted_top_classes, top_confidences

# Sinf nomlarini formatlash
def format_class_name(name):
    """Sinf nomlarini formatlash funksiyasi"""
    return name.replace("_", " ").title().replace("  ", " ").replace("  ", " ")

# Bashoratni ko‘rsatish funksiyasi
def display_prediction(original_class, formatted_class, confidence, top_classes, formatted_top_classes, top_confidences):
    """Bashorat natijalarini ko‘rsatuvchi funksiya"""
    st.markdown(f"<h2 class='prediction-header'>Tashxis: {formatted_class}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p class='confidence-text'>Ishonch: {confidence:.2f}%</p>", unsafe_allow_html=True)
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    prediction_df = pd.DataFrame({"Kasallik": formatted_top_classes, "Ishonch": top_confidences})
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Greens(np.linspace(0.6, 0.95, len(prediction_df)))
    bars = ax.barh(prediction_df["Kasallik"], prediction_df["Ishonch"], color=colors)
    ax.set_xlabel("Ishonch (%)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Kasallik", fontsize=12, fontweight='bold')
    ax.set_title("Eng Yuqori 5 Bashorat", fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=11)
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{prediction_df['Ishonch'][i]:.2f}%", va='center', fontsize=10, fontweight='bold')
    fig.patch.set_facecolor('#f7faf7')
    ax.set_facecolor('#f7faf7')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Batafsil Bashoratlar:</h3>", unsafe_allow_html=True)
    styled_df = prediction_df.style.format({"Ishonch": "{:.2f}%"}).background_gradient(cmap='Greens', subset=['Ishonch'])
    st.table(styled_df)

# Kasallik haqida ma’lumot ko‘rsatish
def display_disease_info(class_name, class_names):
    """Kasallik haqida ma’lumotni ko‘rsatuvchi funksiya"""
    disease_info = {
        "Pepper__bell___Bacterial_spot": {
            "tavsif": "Bakterial dog‘ kasalligi barglar va mevalarda qoramtir, suv bilan to‘yingan yaralarni keltirib chiqaradi, bu esa barglarning tushishi va hosilning kamayishiga olib keladi.",
            "sabablar": "Xanthomonas campestris pv. vesicatoria bakteriyasi sabab bo‘lib, ko‘pincha ifloslangan urug‘lar va suv sachrashi orqali tarqaladi.",
            "davolash": "Mis asosidagi bakteritsidlarni qo‘llash va zararlangan o‘simlik qismlarini olib tashlash.",
            "oldini_olish": "Kasalliksiz urug‘lardan foydalanish, yuqoridan sug‘orishdan saqlanish va yaxshi havo aylanishini ta’minlash."
        },
        "Pepper__bell___healthy": {
            "tavsif": "Sog‘lom qalampir o‘simliklari mustahkam poyalar, quyuq yashil barglar va yaxshi rivojlangan mevalarga ega.",
            "sabablar": "To‘g‘ri oziqlantirish, sug‘orish va zararkunandalarga qarshi kurash bilan optimal o‘sish sharoitlari.",
            "davolash": "Kasalliklarning oldini olish uchun o‘simliklarga to‘g‘ri parvarish qilish va monitoring.",
            "oldini_olish": "Muntazam o‘g‘itlash, to‘g‘ri masofa va zararkunandalarni boshqarish."
        },
        "Potato___Early_blight": {
            "tavsif": "Erta blight barglarda jigarrang, konsentrik halqali yaralarni keltirib chiqaradi, bu barglarning tushishi va ildiz mevalar sifatining pasayishiga olib keladi.",
            "sabablar": "Alternaria solani sabab bo‘lib, issiq va nam sharoitlarda rivojlanadi.",
            "davolash": "Chlorothalonil yoki mancozeb kabi fungitsidlarni qo‘llash va zararlangan barglarni olib tashlash.",
            "oldini_olish": "Ekin almashlab ekish, to‘g‘ri masofani ta’minlash va chidamli kartoshka navlaridan foydalanish."
        },
        "Potato___Late_blight": {
            "tavsif": "Kech blight barglar va ildiz mevalarda qoramtir, suv bilan to‘yingan yaralarni keltirib chiqaradi, bu tez chirishga olib keladi.",
            "sabablar": "Phytophthora infestans sabab bo‘lib, salqin va nam sharoitlarda yaxshi rivojlanadi.",
            "davolash": "Metalaxyl kabi fungitsidlarni ishlatish va zararlangan o‘simliklarni zudlik bilan olib tashlash.",
            "oldini_olish": "Chidamli navlarni ekish, yaxshi drenajni ta’minlash va yuqoridan sug‘orishdan saqlanish."
        },
        "Potato___healthy": {
            "tavsif": "Sog‘lom kartoshka o‘simliklari yam-yashil barglar, mustahkam poyalar va yaxshi rivojlangan ildiz mevalarga ega.",
            "sabablar": "Tuproqni to‘g‘ri tayyorlash, sug‘orish va zararkunandalarni boshqarish.",
            "davolash": "Muntazam parvarish va kasalliklarni monitoring qilish.",
            "oldini_olish": "Ekin almashlab ekish, muvozanatli o‘g‘itlash va zararkunandalarni nazorat qilish."
        },
        "Tomato_Bacterial_spot": {
            "tavsif": "Bakterial dog‘ kasalligi barglar va mevalarda kichik, qoramtir yaralarni keltirib chiqaradi, bu hosil va sifatni pasaytiradi.",
            "sabablar": "Xanthomonas campestris pv. vesicatoria sabab bo‘lib, ifloslangan asboblar va suv orqali tarqaladi.",
            "davolash": "Mis asosidagi purkagichlarni ishlatish va zararlangan barglarni olib tashlash.",
            "oldini_olish": "Yuqoridan sug‘orishdan saqlanish, asboblarni dezinfeksiya qilish va kasalliksiz urug‘lardan foydalanish."
        },
        "Tomato_Early_blight": {
            "tavsif": "Erta blight barglar va poyalarda jigarrang, konsentrik halqali dog‘larni keltirib chiqaradi, o‘simlikni zaiflashtiradi.",
            "sabablar": "Alternaria solani sabab bo‘lib, issiq va nam sharoitlarda rivojlanadi.",
            "davolash": "Fungitsidlarni qo‘llash va zararlangan qismlarni olib tashlash.",
            "oldini_olish": "Ekin almashlab ekish, to‘g‘ri masofani ta’minlash va chidamli navlardan foydalanish."
        },
        "Tomato_Late_blight": {
            "tavsif": "Kech blight barglar va mevalarda suv bilan to‘yingan yaralarni keltirib chiqaradi, bu o‘simlikning tez pasayishiga olib keladi.",
            "sabablar": "Phytophthora infestans sabab bo‘lib, salqin va nam sharoitlarda tarqaladi.",
            "davolash": "Metalaxyl kabi fungitsidlarni ishlatish va zararlangan o‘simliklarni olib tashlash.",
            "oldini_olish": "Yuqoridan sug‘orishdan saqlanish, havo aylanishini oshirish va chidamli navlarni ekish."
        },
        "Tomato_Leaf_Mold": {
            "tavsif": "Barg chiriyotgani barglarda sariq dog‘lar sifatida paydo bo‘ladi, bu fotosintezni va hosilni kamaytiradi.",
            "sabablar": "Passalora fulva sabab bo‘lib, yuqori namlikda rivojlanadi.",
            "davolash": "Fungitsidlarni qo‘llash va havo aylanishini yaxshilash.",
            "oldini_olish": "To‘g‘ri masofani ta’minlash, ortiqcha barglarni kesish va yuqoridan sug‘orishdan saqlanish."
        },
        "Tomato_Septoria_leaf_spot": {
            "tavsif": "Septoria barg dog‘i kichik, sariq halqali qoramtir yaralarni keltirib chiqaradi, bu barglarning erta tushishiga olib keladi.",
            "sabablar": "Septoria lycopersici sabab bo‘lib, nam sharoitlarda rivojlanadi.",
            "davolash": "Fungitsidlarni ishlatish va zararlangan barglarni olib tashlash.",
            "oldini_olish": "Ekin almashlab ekish, yaxshi havo aylanishini ta’minlash va ildizdan sug‘orish."
        },
        "Tomato_Spider_mites_Two_spotted_spider_mite": {
            "tavsif": "O‘rgimchak oqadilar barglarda sariqlik va nuqtalarni keltirib chiqaradi, bu o‘simlikni zaiflashtiradi.",
            "sabablar": "Tetranychus urticae sabab bo‘lib, issiq va quruq sharoitlarda rivojlanadi.",
            "davolash": "Insektitsidli sovun yoki neem yog‘ini ishlatish.",
            "oldini_olish": "O‘simliklarni muntazam purkash, tabiiy yirtqichlarni (masalan, ladybug) jalb qilish va qurg‘oqchilik stressidan saqlanish."
        },
        "Tomato__Target_Spot": {
            "tavsif": "Maqsadli dog‘ barglar va poyalarda qoramtir, konsentrik yaralarni keltirib chiqaradi, o‘simlikni zaiflashtiradi.",
            "sabablar": "Corynespora cassiicola sabab bo‘lib, nam muhitda tarqaladi.",
            "davolash": "Fungitsidlarni qo‘llash va zararlangan qismlarni olib tashlash.",
            "oldini_olish": "To‘g‘ri masofani ta’minlash, ortiqcha barglarni kesish va barglarni quruq holda saqlash."
        },
        "Tomato__Tomato_YellowLeaf__Curl_Virus": {
            "tavsif": "Bu virus kasalligi barglarning sariqlashishi va burishishiga olib keladi, bu o‘sishning sekinlashishiga sabab bo‘ladi.",
            "sabablar": "Oq chivinlar orqali tarqaladi.",
            "davolash": "To‘g‘ridan-to‘g‘ri davosi yo‘q; oq chivinlarni insektitsidlar va chidamli navlar bilan boshqarish.",
            "oldini_olish": "Reflektiv mulchlardan foydalanish, tabiiy yirtqichlarni jalb qilish va zararlangan o‘simliklarni olib tashlash."
        },
        "Tomato__Tomato_mosaic_virus": {
            "tavsif": "Mozaik virusi barglarning rang-barang bo‘lib, deformatsiyaga uchraganligini va meva hosilining kamayishini keltirib chiqaradi.",
            "sabablar": "Ifloslangan urug‘lar, asboblar va inson qo‘llari orqali tarqaladi.",
            "davolash": "Davosi yo‘q; zararlangan o‘simliklarni olib tashlash va asboblarni dezinfeksiya qilish.",
            "oldini_olish": "Virusdan xoli urug‘lardan foydalanish, o‘simliklarga teginishdan oldin qo‘llarni yuvish va hasharot vektorlarini nazorat qilish."
        },
        "Tomato_healthy": {
            "tavsif": "Sog‘lom pomidor o‘simliklari jonli yashil barglar, mustahkam poyalar va normal meva rivojlanishiga ega.",
            "sabablar": "To‘g‘ri parvarish, yetarli sug‘orish, yaxshi quyosh nuri va muntazam o‘g‘itlash.",
            "davolash": "O‘simlik salomatligini saqlash uchun muntazam parvarish amaliyotlarini davom ettirish.",
            "oldini_olish": "Muntazam monitoring, muvozanatli oziqlantirish, mos sug‘orish va yaxshi havo aylanishi."
        }
    }
    if class_name in disease_info:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("<h3 class='disease-info-header'>Kasallik Haqida Ma'lumot</h3>", unsafe_allow_html=True)
        info = disease_info[class_name]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 class='disease-info-subheader'>Tavsif</h4>", unsafe_allow_html=True)
            st.write(info["tavsif"])
            st.markdown("<h4 class='disease-info-subheader'>Sabablar</h4>", unsafe_allow_html=True)
            st.write(info["sabablar"])
        with col2:
            st.markdown("<h4 class='disease-info-subheader'>Davolash</h4>", unsafe_allow_html=True)
            st.write(info["davolash"])
            st.markdown("<h4 class='disease-info-subheader'>Oldini Olish</h4>", unsafe_allow_html=True)
            st.write(info["oldini_olish"])
    else:
        st.info("Ushbu o‘simlik holati uchun batafsil ma’lumot mavjud emas. Iltimos, qishloq xo‘jaligi mutaxassisi bilan maslahatlashing.")

# Asosiy funksiya
def main():
    """Asosiy ilova funksiyasi"""
    try:
        model, transform, label_encoder, class_names, device, config = load_model_resources()
        model_loaded = True
    except Exception as e:
        st.error(f"Modelni yuklashda xatolik: {e}")
        model_loaded = False
        class_names = []

    st.markdown("<h1 class='main-header'>🌿 O'SIMLIK KASALLIKLARI AI TASHXISLOVCHI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-header'>Ushbu ilova chuqur o‘qitish yordamida o‘simlik barglaridagi kasalliklarni tashxis qiladi. O‘simlik bargining rasmini yuklang yoki kameradan tasvir oling.</p>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    # Sessiya holatini boshqarish
    if 'yuklangan_fayl' not in st.session_state:
        st.session_state.yuklangan_fayl = None
    if 'rasm_qayta_ishlandi' not in st.session_state:
        st.session_state.rasm_qayta_ishlandi = False
    if 'bashorat_natijalari' not in st.session_state:
        st.session_state.bashorat_natijalari = None

    with col1:
        st.markdown("<div class='upload-section'><h3 class='sub-header'>RASM YUKLASH BO'LIMI</h3></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Rasmni tanlang", type=["jpg", "jpeg", "png"], key="file_uploader", label_visibility="collapsed")
        if uploaded_file is not None:
            st.session_state.bashorat_natijalari = None
            st.session_state.rasm_qayta_ishlandi = False
            st.session_state.yuklangan_fayl = uploaded_file

        st.markdown("<div class='upload-section real-time'><h3 class='sub-header'>REAL VAQT REJIMI</h3></div>", unsafe_allow_html=True)
        st.button("Kamerani ishga tushirish", key="capture_button", use_container_width=True)

        with st.expander("Klassifikatsiya uchun Mavjud O'simlik Kasalliklari"):
            formatted_classes = [name.replace("_", " ").replace("__", " ").replace("___", " ").title() for name in class_names]
            classes_df = pd.DataFrame({"Mavjud Kasalliklar": formatted_classes})
            st.table(classes_df)

    with col2:
        st.markdown("<div class='results-section'><h3 class='sub-header'>TASHXISLANGAN RASM</h3></div>", unsafe_allow_html=True)
        placeholder = st.empty()  # Placeholder faqat col2 ichida aniqlanadi
        
        # Kamerani ishga tushirish tugmasi faqat col2 da tasvirni ko‘rsatadi
        if st.session_state.get("capture_button", False):
            capture_image(placeholder)

        if st.session_state.yuklangan_fayl is not None:
            if isinstance(st.session_state.yuklangan_fayl, str):
                image = Image.open(st.session_state.yuklangan_fayl)
                resized_image = resize_image(image, max_width=600)  # Tasvirni kichraytirish
                placeholder.image(resized_image, caption="Tanlangan Rasm", use_container_width=True)
                with open(st.session_state.yuklangan_fayl, "rb") as f:
                    file_content = f.read()
                    file_obj = type('obj', (object,), {'getvalue': lambda: file_content})
                st.session_state.bashorat_natijalari = predict_disease(file_obj, model, transform, label_encoder, device)
                st.session_state.rasm_qayta_ishlandi = True
            else:
                image = Image.open(st.session_state.yuklangan_fayl)
                resized_image = resize_image(image, max_width=600)  # Tasvirni kichraytirish
                placeholder.image(resized_image, caption="Yuklangan Rasm", use_container_width=True)
                st.session_state.bashorat_natijalari = predict_disease(st.session_state.yuklangan_fayl, model, transform, label_encoder, device)
                st.session_state.rasm_qayta_ishlandi = True

    if st.session_state.rasm_qayta_ishlandi and st.session_state.bashorat_natijalari is not None:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        class_name, formatted_class, confidence, top_classes, formatted_top_classes, top_confidences = st.session_state.bashorat_natijalari
        display_prediction(class_name, formatted_class, confidence, top_classes, formatted_top_classes, top_confidences)
        display_disease_info(class_name, class_names)

    with st.sidebar:
        st.markdown("<h2 style='color: #1a3c34; font-weight: 600;'>Model Haqida</h2>", unsafe_allow_html=True)
        st.write("Ushbu ilova o‘simlik barglari rasmlaridan kasalliklarni klassifikatsiya qilish uchun Konvolyutsion Neyron Tarmoqdan (CNN) foydalanadi.")
        st.markdown("<h3 style='color: #2e7d32; margin-top: 20px; font-weight: 600;'>Model Arxitekturasi:</h3>", unsafe_allow_html=True)
        if model_loaded:
            st.write("- **Model Turi:** 5 ta konvolyutsion blokli ilg‘or arxitekturalarga ega CNN")
            st.write(f"- **Sinf Soni:** {len(class_names)}")
            st.write("- **Sinov Aniqligi:** 96.5%")
        st.markdown("<h3 style='color: #2e7d32; margin-top: 20px; font-weight: 600;'>Ma'lumotlar To'plami Haqida:</h3>", unsafe_allow_html=True)
        st.write("- PlantVillage ma’lumotlar to‘plamida o‘qitilgan")
        st.write("- 15 ta o‘simlik kasalliklari va sog‘lom o‘simlik sinflarini o‘z ichiga oladi")
        st.write("- Sinflar pomidor, kartoshka va qalampirdagi kasalliklarni o‘z ichiga oladi")
        st.markdown("<h3 style='color: #2e7d32; margin-top: 20px; font-weight: 600;'>Foydalanish Yo'riqnomasi:</h3>", unsafe_allow_html=True)
        st.write("1. O‘simlik bargining rasmini yuklang")
        st.write("2. Tashxis va tavsiya etilgan harakatlarni ko‘ring")
        st.write("3. Kasallik haqida batafsil ma’lumotni tekshiring")
        st.markdown("<h3 style='color: #2e7d32; margin-top: 20px; font-weight: 600;'>Ishlab Chiquvchilar:</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Islomnur Ibragimov**
        """)
        st.markdown("---")
        st.caption("© 2025 O'simlik Kasalliklari Klassifikatori - Barcha Huquqlar Himoyalangan")

if __name__ == "__main__":
    main()
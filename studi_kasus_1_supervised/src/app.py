import gradio as gr
import joblib
import pandas as pd
import os

# Paths are resolved relative to this file so the app works regardless of cwd
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(ROOT_DIR, 'models')

# Load model and scaler
try:
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    supervised_model = joblib.load(os.path.join(models_dir, 'supervised_model.pkl'))
except Exception as e:
    print("Error loading models. Please ensure train_model.py has been run.")

# Define all 20 feature names in exact order
features = [
    'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
    'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
    'noise_level', 'living_conditions', 'safety', 'basic_needs',
    'academic_performance', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'social_support', 'peer_pressure',
    'extracurricular_activities', 'bullying'
]

# Default median values for hidden features to keep the model accurate
default_values = {
    'anxiety_level': 11, 'self_esteem': 19, 'mental_health_history': 0,
    'headache': 3, 'breathing_problem': 3, 'noise_level': 3,
    'living_conditions': 2, 'safety': 2, 'academic_performance': 2,
    'study_load': 2, 'future_career_concerns': 2, 'social_support': 2,
    'peer_pressure': 2
}

PLACEHOLDER_STRESS = "### Hasil akan muncul di sini"
PLACEHOLDER_DESC = "Belum ada hasil analisis. Isi data di sebelah kiri, lalu klik **Analisis Data Mahasiswa**."

def predict_stress(bp, sq, ea, bully, bn, tsr, dep):
    # Prepare the 20-feature array
    input_dict = default_values.copy()
    input_dict.update({
        'blood_pressure': bp,
        'sleep_quality': sq,
        'extracurricular_activities': ea,
        'bullying': bully,
        'basic_needs': bn,
        'teacher_student_relationship': tsr,
        'depression': dep
    })

    # Create ordered list for dataframe
    input_list = [input_dict[f] for f in features]
    input_data = pd.DataFrame([input_list], columns=features)

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Predict
    sup_pred = supervised_model.predict(scaled_data)[0]

    stress_mapping = {
        0: "🟢 Rendah (Low Stress)",
        1: "🟡 Sedang (Medium Stress)",
        2: "🔴 Tinggi (High Stress)"
    }

    res_stress = f"### {stress_mapping.get(sup_pred, 'Tidak Diketahui')}"

    # Keterangan tambahan berdasarkan tingkat stres
    desc = ""
    if sup_pred == 0:
        desc = "Mahasiswa dalam kondisi psikologis yang stabil. Tetap pertahankan pola tidur dan aktivitas dengan baik!"
    elif sup_pred == 1:
        desc = "Terdapat indikasi beban stres tingkat menengah. Mahasiswa disarankan untuk lebih banyak beristirahat dan mengelola beban akademiknya."
    else:
        desc = "Peringatan: Tingkat stres terdeteksi sangat tinggi! Mahasiswa ini disarankan untuk segera mencari dukungan sosial atau berkonsultasi dengan layanan konseling kampus."

    return res_stress, desc

custom_css = """
/* Make sliders look smoother */
input[type=range] {
    transition: all 0.2s ease-in-out;
}
input[type=range]:hover {
    cursor: grab;
    filter: brightness(1.1);
}
input[type=range]:active {
    cursor: grabbing;
}

/* Card-style panels for the input form and the results */
.panel-card {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    background: var(--background-fill-secondary) !important;
    box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.06);
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald"), css=custom_css) as demo:

    gr.Markdown("# 🧠 Deteksi Stres Mahasiswa")
    gr.Markdown("Aplikasi ini memprediksi indikasi tingkat stres berdasarkan 7 faktor utama (Machine Learning: Random Forest Classifier). Isi data di sebelah kiri, lalu klik **Analisis Data Mahasiswa**.")

    with gr.Accordion("📖 Panduan Parameter (Cara Membaca Skala)", open=False):
        gr.Markdown("""
        Berikut adalah penjelasan untuk setiap parameter dan makna dari skala (seperti 0-5). Nilai yang ideal atau tergolong **baik** ditandai dengan centang hijau (✅).

        * **Tekanan Darah (1-3):** 1 = Normal ✅ | 2 = Agak Tinggi | 3 = Tinggi
        * **Kualitas Tidur (0-5):** Semakin tinggi angkanya, semakin baik. (0 = Sangat Buruk ➔ 5 = Sangat Baik ✅)
        * **Aktivitas Ekstrakurikuler (0-5):** Menunjukkan tingkat kesibukan kegiatan. (0 = Tidak Aktif ➔ 5 = Sangat Aktif)
        * **Tingkat Bullying yang Dialami (0-5):** Semakin rendah angkanya, semakin baik. (0 = Tidak Pernah ✅ ➔ 5 = Sangat Sering)
        * **Pemenuhan Kebutuhan Dasar (0-5):** Semakin tinggi angkanya, semakin baik. (0 = Sangat Kurang ➔ 5 = Sangat Terpenuhi ✅)
        * **Hubungan Dosen-Mahasiswa (0-5):** Semakin tinggi angkanya, semakin baik. (0 = Sangat Buruk ➔ 5 = Sangat Baik ✅)
        * **Tingkat Depresi (0-27):** Menggunakan skala psikologis. Semakin rendah angkanya, semakin baik. (0 = Tidak Ada Gejala ✅ ➔ 27 = Gejala Sangat Berat)
        """)

    with gr.Row():
        with gr.Column(scale=4, elem_classes="panel-card"):
            gr.Markdown("### 📋 Input Data")

            bp = gr.Dropdown(
                choices=[("Normal", 1), ("Agak Tinggi", 2), ("Tinggi", 3)],
                label="Tekanan Darah",
                value=2,
                interactive=True
            )
            sq = gr.Slider(0, 5, step=1, label="Kualitas Tidur", value=3, interactive=True)
            ea = gr.Slider(0, 5, step=1, label="Aktivitas Ekstrakurikuler", value=2, interactive=True)
            bully = gr.Slider(0, 5, step=1, label="Tingkat Bullying yang Dialami", value=1, interactive=True)
            bn = gr.Slider(0, 5, step=1, label="Pemenuhan Kebutuhan Dasar", value=3, interactive=True)
            tsr = gr.Slider(0, 5, step=1, label="Hubungan Dosen-Mahasiswa", value=3, interactive=True)
            dep = gr.Slider(0, 27, step=1, label="Tingkat Depresi", value=10, interactive=True)

            inputs = [bp, sq, ea, bully, bn, tsr, dep]

            submit_btn = gr.Button("🔍 Analisis Data Mahasiswa", variant="primary", size="lg")

        with gr.Column(scale=3, elem_classes="panel-card"):
            gr.Markdown("### 📊 Hasil Analisis")
            out_stress = gr.Markdown(PLACEHOLDER_STRESS)
            out_desc = gr.Markdown(PLACEHOLDER_DESC)

    submit_btn.click(
        fn=predict_stress,
        inputs=inputs,
        outputs=[out_stress, out_desc]
    )

# Disable warnings by passing css directly to Blocks but avoiding deprecated args if possible.
# But Gradio still prefers it in Blocks on most versions. We will just pass it to both or ignore warning.
if __name__ == "__main__":
    demo.launch(inbrowser=True)

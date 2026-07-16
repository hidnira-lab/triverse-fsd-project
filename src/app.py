import gradio as gr
import joblib
import pandas as pd
import os

# Paths are resolved relative to this file so the app works regardless of cwd
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(ROOT_DIR, 'models')

# Load models and scaler
try:
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    supervised_model = joblib.load(os.path.join(models_dir, 'supervised_model.pkl'))
    unsupervised_model = joblib.load(os.path.join(models_dir, 'unsupervised_model.pkl'))
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
    unsup_pred = unsupervised_model.predict(scaled_data)[0]
    
    stress_mapping = {
        0: "🟢 Rendah (Low Stress)",
        1: "🟡 Sedang (Medium Stress)",
        2: "🔴 Tinggi (High Stress)"
    }
    
    res_stress = stress_mapping.get(sup_pred, "Tidak Diketahui")
    res_cluster = f"Cluster {unsup_pred}"
    
    # Keterangan tambahan berdasarkan tingkat stres
    desc = ""
    if sup_pred == 0:
        desc = "Mahasiswa dalam kondisi psikologis yang stabil. Tetap pertahankan pola tidur dan aktivitas dengan baik!"
    elif sup_pred == 1:
        desc = "Terdapat indikasi beban stres tingkat menengah. Mahasiswa disarankan untuk lebih banyak beristirahat dan mengelola beban akademiknya."
    else:
        desc = "Peringatan: Tingkat stres terdeteksi sangat tinggi! Mahasiswa ini disarankan untuk segera mencari dukungan sosial atau berkonsultasi dengan layanan konseling kampus."
        
    return gr.update(visible=True), res_stress, res_cluster, desc

def close_popup():
    return gr.update(visible=False)

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

/* Modal / Popup Styling */
#modal-background {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: rgba(0, 0, 0, 0.6);
    z-index: 9999;
    display: flex;
    justify-content: center;
    align-items: center;
    backdrop-filter: blur(4px);
}
#modal-box {
    background: var(--background-fill-primary);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 10px 40px rgba(0, 0, 0, 0.4);
    width: 90%;
    max-width: 500px;
    text-align: center;
    border: 2px solid var(--primary-500);
    animation: popIn 0.3s ease-out;
}
@keyframes popIn {
    0% { transform: scale(0.8); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald"), css=custom_css) as demo:
    
    # ------------------ POP-UP MODAL (HIDDEN BY DEFAULT) ------------------
    with gr.Column(visible=False, elem_id="modal-background") as modal_container:
        with gr.Column(elem_id="modal-box"):
            gr.Markdown("## 📋 Laporan Hasil Analisis")
            
            with gr.Group():
                out_stress = gr.Markdown("### Tingkat Stres: -")
                out_cluster = gr.Markdown("### Profil Kelompok: -")
                
            out_desc = gr.Markdown("Keterangan akan muncul di sini.", elem_classes="text-center")
            
            close_btn = gr.Button("Tutup Jendela Ini", variant="secondary", size="lg")
    # ----------------------------------------------------------------------
    
    # ------------------ MAIN INTERFACE ------------------
    with gr.Column():
        gr.Markdown("# 🧠 Deteksi Stres Mahasiswa")
        gr.Markdown("Aplikasi ini memprediksi tingkat stres berdasarkan 7 faktor utama (Machine Learning: Random Forest & K-Means Clustering). Geser slider di bawah, lalu klik **Analisis Data**.")
        
        # 7 Most Important Features
        bp = gr.Slider(1, 3, step=1, label="Tekanan Darah (Blood Pressure)", value=2, interactive=True)
        sq = gr.Slider(0, 5, step=1, label="Kualitas Tidur (Sleep Quality)", value=3, interactive=True)
        ea = gr.Slider(0, 5, step=1, label="Aktivitas Ekstrakurikuler", value=2, interactive=True)
        bully = gr.Slider(0, 5, step=1, label="Tingkat Bullying yang Dialami", value=1, interactive=True)
        bn = gr.Slider(0, 5, step=1, label="Pemenuhan Kebutuhan Dasar (Basic Needs)", value=3, interactive=True)
        tsr = gr.Slider(0, 5, step=1, label="Hubungan Dosen-Mahasiswa", value=3, interactive=True)
        dep = gr.Slider(0, 27, step=1, label="Tingkat Depresi (Depression Score)", value=10, interactive=True)
        
        inputs = [bp, sq, ea, bully, bn, tsr, dep]
        
        submit_btn = gr.Button("🔍 Analisis Data Mahasiswa", variant="primary", size="lg")

    # Action to open pop-up and show results
    submit_btn.click(
        fn=predict_stress,
        inputs=inputs,
        outputs=[modal_container, out_stress, out_cluster, out_desc]
    )
    
    # Action to close pop-up
    close_btn.click(
        fn=close_popup,
        outputs=[modal_container]
    )

# Disable warnings by passing css directly to Blocks but avoiding deprecated args if possible. 
# But Gradio still prefers it in Blocks on most versions. We will just pass it to both or ignore warning.
if __name__ == "__main__":
    demo.launch(inbrowser=True)

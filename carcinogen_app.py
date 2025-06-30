import streamlit as st
import pandas as pd
import sys
from PIL import Image
from predict_ensemble import predict_ensemble

st.set_page_config(page_title="Carcinogenicity Predictor", layout="centered")

# === Header & Intro ===
st.title("🧪 Carcinogenicity Predictior")
st.markdown("""
Predict the carcinogenic potential of a compound using an ensemble approach combining three machine learning models.

**Model:** Consensus framework integrating a BiLSTM model with MACCS fingerprints, a Random Forest model with EState descriptors, and a LightGBM model with RDKit fingerprints.  
**Developers:** Huynh Anh Duy<sup>1,2</sup>, Tarapong Srisongkram<sup>2</sup>  
**Affiliations:** <sup>1</sup>Can Tho University, Vietnam; <sup>2</sup>Khon Kaen University, Thailand
""", unsafe_allow_html=True)

# === Sidebar Instructions ===
with st.sidebar:
    st.header("🧾 Instructions")
    st.markdown("""
    1. Paste a SMILES string or upload a CSV file.
    2. Click **Predict** or **Run batch prediction**.
    3. Download the results if needed.
    """)

# ============ Input options ============
tab1, tab2 = st.tabs(["🧬 Single SMILES Input", "📄 Upload CSV File"])

# ==== Tab 1: Single SMILES input ====
with tab1:
    smiles_input = st.text_input("👉 Enter a SMILES string:", placeholder="e.g. CCOC(=O)c1ccccc1")

    if st.button("Predict from SMILES") and smiles_input.strip():
        with st.spinner("⏳ Running prediction..."):
            result = predict_ensemble(smiles_input)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("🔎 Prediction Result")
            st.write(f"**Canonical SMILES:** `{result['canonical_smiles']}`")
            st.metric("LightGBM (RDKit)", f"{result['prob_lightgbm']:.4f}")
            st.metric("Random Forest (EState)", f"{result['prob_rf']:.4f}")
            st.metric("BiLSTM (MACCS)", f"{result['prob_bilstm']:.4f}")
            st.markdown("---")
            st.metric("🎯 Average Probability from Consensus Framework", f"{result['average_probability']:.4f}")
            if result["predicted_label"] == "carcinogen":
                st.error("☣️ **Prediction: Carcinogen**")
            else:
                st.success("✅ **Prediction: Non-Carcinogen**")

# ==== Tab 2: Upload CSV ====
with tab2:
    uploaded_file = st.file_uploader("📤 Upload a CSV file with a column named 'SMILES'", type=["csv"])
    
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        if "SMILES" not in df_input.columns:
            st.error("⚠️ CSV file must contain a column named 'SMILES'.")
        else:
            st.success(f"✅ Successfully loaded {len(df_input)} SMILES.")
            if st.button("🔍 Run Batch Prediction"):
                results = []
                with st.spinner("⏳ Processing..."):
                    for smi in df_input["SMILES"]:
                        r = predict_ensemble(smi)
                        results.append({
                            "Input_SMILES": smi,
                            "Canonical_SMILES": r.get("canonical_smiles", None),
                            "Prob_LightGBM": r.get("prob_lightgbm", None),
                            "Prob_RF": r.get("prob_rf", None),
                            "Prob_BiLSTM": r.get("prob_bilstm", None),
                            "Average_Probability": r.get("average_probability", None),
                            "Prediction": r.get("predicted_label", r.get("error", "Error"))
                        })
                df_result = pd.DataFrame(results)
                st.dataframe(df_result)

                # Download result
                csv = df_result.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results as CSV", data=csv, file_name="carcinogen_predictions.csv", mime="text/csv")

# === Author Section ===
st.markdown("---")
st.subheader("👨‍🔬 About the Authors")

col1, col2 = st.columns(2)

with col1:
    image1 = Image.open("assets/duy.jpg")
    st.image(image1, caption="Huynh Anh Duy", width=160)
    st.markdown("""
    **Huynh Anh Duy**  
    Can Tho University, Vietnam  
    PhD Candidate, Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [huynhanhduy.h@kkumail.com](mailto:huynhanhduy.h@kkumail.com), [haduy@ctu.edu.vn](mailto:haduy@ctu.edu.vn)
    """)

with col2:
    image2 = Image.open("assets/tarasi.png")
    st.image(image2, caption="Tarapong Srisongkram", width=160)
    st.markdown("""
    **Asst Prof. Dr. Tarapong Srisongkram**  
    Faculty of Pharmaceutical Sciences  
    Khon Kaen University, Thailand  
    *Cheminformatics, QSAR Modeling, Computational Drug Discovery and Toxicity Prediction*  
    📧 [tarasri@kku.ac.th](mailto:tarasri@kku.ac.th)
    """)

# === Footer ===
st.markdown("---")
st.caption(f"🔧 Python version: {sys.version.split()[0]}")

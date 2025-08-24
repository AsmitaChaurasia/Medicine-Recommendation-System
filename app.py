# app.py
import streamlit as st
from recommend import recommend_top_medicines
import pandas as pd

st.set_page_config(page_title="💊 Medicine Recommender", layout="centered")
st.title("💊 Medicine Recommendation System")
st.markdown("Enter your symptoms and get the most relevant medicines using AI.")

symptoms = st.text_area("📝 Enter symptoms (e.g., fever, body pain, sore throat):")

if st.button("🔍 Get Recommendations"):
    if symptoms.strip() != "":
        results = recommend_top_medicines(symptoms, top_k=3)
        st.write("## 📋 Top 3 Recommended Medicines:")
        for idx, row in results.iterrows():
            st.markdown(f"**{idx+1}. {row['recommended_medicine']}**")
            st.markdown(f"🧪 Matched Symptoms: *{row['symptoms']}*")
            st.markdown(f"📊 Confidence Score: `{row['similarity_score']}`")
            st.markdown("---")

        # Export options
        st.download_button("📥 Download as CSV", results.to_csv(index=False), "recommendations.csv", "text/csv")
        
        import io
        from fpdf import FPDF

        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'Medicine Recommendation Report', ln=True, align='C')
            def chapter_body(self, df):
                self.set_font('Arial', '', 10)
                for i, row in df.iterrows():
                    self.multi_cell(0, 10, f"{i+1}. {row['recommended_medicine']} (Confidence: {row['similarity_score']})\nMatched Symptoms: {row['symptoms']}")
                    self.ln()

        if st.button("📄 Download PDF Report"):
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_body(results)
            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            st.download_button("⬇️ Download PDF", pdf_output.getvalue(), "recommendations.pdf", "application/pdf")
    else:
        st.warning("Please enter some symptoms.")

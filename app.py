# app.py

import os
import json
import textwrap
from typing import Dict, Any
from pathlib import Path

# --- Important Dependencies ---
# Note: For PDF export and Radar chart to work, you need to install these:
# pip install reportlab numpy matplotlib google-generativeai
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Load environment variables from .env file
load_dotenv()

# ------------------------------
# Page Config & Initial State
# ------------------------------
st.set_page_config(page_title="CodeGrade AI", page_icon="🎓", layout="centered")

# Initialize session state variables to manage the app's flow
if "evaluation_done" not in st.session_state:
    st.session_state.evaluation_done = False
if "result" not in st.session_state:
    st.session_state.result = None
if "submitted_code_text" not in st.session_state:
    st.session_state.submitted_code_text = ""
if "submitted_docs_text" not in st.session_state:
    st.session_state.submitted_docs_text = ""
if "submitted_code_filenames" not in st.session_state:
    st.session_state.submitted_code_filenames = []


# ------------------------------
# Styling (Hardcoded to Dark Theme)
# ------------------------------
def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {file_path}. Please ensure style.css exists.")

st.markdown(
    """<style>[data-testid="stSidebarNav"] {display: none;} [data-testid="stSidebar"] {display: none;} div[data-testid="collapsedControl"] {display: none;}</style>""",
    unsafe_allow_html=True
)
css_file = Path(__file__).parent / "style.css"
load_css(css_file)


# ------------------------------
# App Title & Header
# ------------------------------
st.title("🎓 CodeGrade AI — Static Code Review Assistant")
st.write("Paste requirements, upload files, and get a static code review.")

# --- Function to reset the state for a new evaluation ---
def reset_evaluation():
    # Reset all the result and submission-related state variables
    st.session_state.evaluation_done = False
    st.session_state.result = None
    st.session_state.submitted_code_text = ""
    st.session_state.submitted_docs_text = ""
    st.session_state.submitted_code_filenames = []
    
    # --- CHANGE: Clear the input widgets by deleting their keys from session state. ---
    # This forces Streamlit to re-initialize them as empty on the next script run.
    for key in ["requirements", "code_files", "docs_file"]:
        if key in st.session_state:
            del st.session_state[key]

# Show the 'Start New Evaluation' button only if an evaluation is done
if st.session_state.evaluation_done:
    st.button("↩️ Start New Evaluation", on_click=reset_evaluation, use_container_width=True)


# ------------------------------
# Constants & Schema
# ------------------------------
MAX_CODE_CHARS = 60_000
MAX_DOCS_CHARS = 30_000
EVAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "code_quality": {"type": "object","properties": {"score": {"type": "integer"}, "feedback": {"type": "string"}},"required": ["score", "feedback"]},
        "documentation": {"type": "object","properties": {"score": {"type": "integer"}, "feedback": {"type": "string"}},"required": ["score", "feedback"]},
        "adherence": {"type": "object","properties": {"score": {"type": "integer"}, "feedback": {"type": "string"}},"required": ["score", "feedback"]},
        "rubric": {"type": "object","properties": {"what_went_well": {"type": "array", "items": {"type": "string"}},"areas_for_improvement": {"type": "array", "items": {"type": "string"}},"suggested_next_steps": {"type": "array", "items": {"type": "string"}}},"required": ["what_went_well", "areas_for_improvement", "suggested_next_steps"]},
        "corrected_code": {"type": "string"},
    },
    "required": ["overall_score", "code_quality", "documentation", "adherence", "rubric", "corrected_code"],
}


# ------------------------------
# Helper functions
# ------------------------------
def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")

def _read_uploaded_text(uploader, limit: int) -> str:
    if uploader is None: return ""
    data = uploader.read()
    text = _decode_bytes(data)
    if len(text) > limit:
        return text[:limit] + f"\n\n[TRUNCATED to {limit} chars of {len(text)} total]"
    return text

def build_system_prompt() -> str:
    return textwrap.dedent("""
        You are an expert teaching assistant who performs static code reviews. You will receive: 1) natural-language project requirements, 2) one or more Python source files, and 3) a README or documentation file. IMPORTANT RULES: - Do NOT execute code; review by reading only. - Assess three categories: Code Quality, Documentation, Adherence to Requirements. - Use scores from 1-10 (10 is excellent). Be fair but strict. - Feedback must be concise, actionable, and specific to the submission. - If requirements are unclear or unmet, point that out explicitly. OUTPUT INSTRUCTIONS: - Return ONLY a JSON object that matches the provided schema. - The JSON must include a "corrected_code" field. - "corrected_code" must be the *full, corrected Python code* as a single string, incorporating fixes and improvements based on your feedback. - If multiple files were provided, combine them into a single corrected code block. - Do not wrap the code in backticks or markdown — just plain Python source text.
    """).strip()

def build_user_prompt(requirements: str, code_name: str, code_text: str, docs_name: str, docs_text: str) -> str:
    return textwrap.dedent(f"""
        [PROJECT REQUIREMENTS]
        {requirements.strip()}
        [STUDENT CODE: {code_name or "unnamed.py"}]
        {code_text}
        [DOCUMENTATION: {docs_name or "README.md"}]
        {docs_text or "(none provided)"}
    """).strip()


# ------------------------------
# Provider backends
# ------------------------------
def call_openai(messages: list, schema: Dict[str, Any], model: str) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("OpenAI SDK not installed. Please `pip install openai`")
    client = OpenAI()
    try:
        comp = client.chat.completions.create(model=model, messages=messages, response_format={"type": "json_object"})
        text = comp.choices[0].message.content
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")

def call_gemini(prompt_text: str, schema: Dict[str, Any], model: str) -> Dict[str, Any]:
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("Gemini SDK not installed. Please `pip install google-generativeai`")
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set in the environment.")
    genai.configure(api_key=api_key)
    
    model_obj = genai.GenerativeModel(model)
    try:
        resp = model_obj.generate_content([prompt_text], generation_config={"response_mime_type": "application/json", "response_schema": schema})
        return json.loads(resp.text)
    except Exception as e:
        # Fallback for models that might not fully support the new JSON schema feature
        system_line = "Return ONLY a JSON object matching this JSON Schema (no backticks, no extra text):\n" + json.dumps(schema)
        resp = model_obj.generate_content([system_line + "\n\n" + prompt_text])
        return json.loads(resp.text)


# ------------------------------
# Chart & PDF Functions
# ------------------------------
def export_to_pdf(result):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate("report.pdf")
    story = [Paragraph("CodeGrade AI Report", styles['Title']), Spacer(1, 12), Paragraph(f"Overall Score: {result.get('overall_score','-')}/10", styles['Normal'])]
    for key in ["code_quality", "documentation", "adherence"]:
        block = result.get(key, {})
        story.extend([Spacer(1, 12), Paragraph(f"<b>{key.replace('_',' ').title()} Score: {block.get('score','-')}/10</b>", styles['h3']), Paragraph(f"Feedback: {block.get('feedback','')}", styles['Normal'])])
    doc.build(story)
    return "report.pdf"

def radar_chart(scores):
    labels = np.array(["Code Quality", "Documentation", "Adherence"])
    num_scores = [s if isinstance(s, (int, float)) else 0 for s in scores]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    num_scores += num_scores[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, num_scores, color='#5e2b84', alpha=0.25)
    ax.plot(angles, num_scores, color='#5e2b84', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white")
    ax.set_yticks(np.arange(0, 11, 2))
    st.pyplot(fig)


# ------------------------------
# UI Form (with 'disabled' parameter)
# ------------------------------
with st.form("grade-form", clear_on_submit=False):
    # --- CHANGE: Added a 'key' to each input widget for programmatic clearing ---
    requirements = st.text_area(
        "Project requirements", 
        height=160, 
        placeholder="e.g., Create a Python script...", 
        disabled=st.session_state.evaluation_done,
        key="requirements"
    )
    
    code_files = st.file_uploader(
        "Upload Code Files (e.g., .py, .js, .html, .css)", 
        accept_multiple_files=True, 
        disabled=st.session_state.evaluation_done,
        key="code_files"
    )
    
    docs_file = st.file_uploader(
        "Documentation (README.md or .txt)", 
        type=["md", "txt"], 
        disabled=st.session_state.evaluation_done,
        key="docs_file"
    )
    
    st.markdown("**AI Engine**")
    engine = st.selectbox("Provider", ["OpenAI", "Google"], index=0, help="Choose which API to use for analysis.", disabled=st.session_state.evaluation_done)
    
    if engine == "OpenAI":
        model_name = "gpt-4o-mini"
    else:
        model_name = "gemini-1.5-pro-latest"
        
    st.text_input("Model", value=model_name, disabled=True)
    
    submitted = st.form_submit_button("🔍 Evaluate Project", use_container_width=True, disabled=st.session_state.evaluation_done)


# ------------------------------
# Main Logic
# ------------------------------
if submitted:
    if not requirements or not code_files:
        st.error("Please provide both project requirements and at least one code file.")
    else:
        st.session_state.submitted_code_text = ""
        st.session_state.submitted_code_filenames = [cf.name for cf in code_files]
        for cf in code_files:
            st.session_state.submitted_code_text += f"\n\n# --- File: {cf.name} ---\n" + _read_uploaded_text(cf, MAX_CODE_CHARS)
        st.session_state.submitted_code_text = st.session_state.submitted_code_text.strip()
        st.session_state.submitted_docs_text = _read_uploaded_text(docs_file, MAX_DOCS_CHARS) if docs_file else ""
        
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(
            requirements=requirements,
            code_name=", ".join(st.session_state.submitted_code_filenames),
            code_text=st.session_state.submitted_code_text,
            docs_name=docs_file.name if docs_file else "",
            docs_text=st.session_state.submitted_docs_text,
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        with st.spinner("Analyzing submission..."):
            try:
                if engine == "Google":
                    st.session_state.result = call_gemini(system_prompt + "\n\n" + user_prompt, EVAL_SCHEMA, model_name)
                else:
                    st.session_state.result = call_openai(messages, EVAL_SCHEMA, model_name)
                
                st.session_state.evaluation_done = True
                st.rerun()
            
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ------------------------------
# Report Rendering (Runs if evaluation is done and result exists)
# ------------------------------
if st.session_state.evaluation_done and st.session_state.result:
    result = st.session_state.result
    st.subheader("📋 Evaluation Report")
    
    overall = result.get("overall_score")
    if isinstance(overall, (int, float)):
        st.progress(max(0, min(100, int((overall / 10) * 100))), text=f"Overall Score: {overall}/10")
    
    cols = st.columns(3)
    scores = []
    for i, key in enumerate(["code_quality", "documentation", "adherence"]):
        with cols[i]:
            block = result.get(key, {})
            score = "—"
            if isinstance(block, dict):
                score = block.get("score", "—")
            elif isinstance(block, int):
                score = block
            scores.append(score)
            st.metric(label=key.replace("_", " ").title(), value=f"{score}/10" if isinstance(score, (int, float)) else "n/a")
    
    st.markdown("### Score Radar")
    radar_chart(scores)
    
    for key, title in [("code_quality", "Code Quality"), ("documentation", "Documentation"), ("adherence", "Adherence to Requirements")]:
        st.markdown(f"**{title} Feedback**")
        block = result.get(key, {})
        feedback = "No feedback provided."
        if isinstance(block, dict):
            feedback = block.get("feedback", feedback)
        st.write(feedback)
    
    with st.expander("Detailed Rubric"):
        rubric = result.get("rubric", {})
        for list_key, nice in [("what_went_well", "What went well"), ("areas_for_improvement", "Areas for improvement"), ("suggested_next_steps", "Suggested next steps")]:
            items = rubric.get(list_key, [])
            if items:
                st.markdown(f"**{nice}**")
                for it in items: st.write(f"- {it}")

    corrected_code = result.get("corrected_code")
    if corrected_code and isinstance(corrected_code, str):
        st.subheader("✅ Corrected Code")
        st.info("The AI has provided a corrected version of the code based on its feedback.")
        if st.session_state.submitted_code_filenames:
            original_filename = st.session_state.submitted_code_filenames[0]
            name, ext = os.path.splitext(original_filename)
            corrected_filename = f"{name}_corrected{ext}"
            st.download_button(
                label=f"⬇ Download {corrected_filename}", 
                data=corrected_code.encode("utf-8"), 
                file_name=corrected_filename, 
                mime="text/plain"
            )
        with st.expander("View Corrected Code"):
            st.code(corrected_code, language="python")
    
    st.divider()
    
    st.subheader("Submitted Files & Raw Data")
    
    with st.expander("View Submitted Files"):
        st.code(st.session_state.submitted_code_text, language=None, line_numbers=True)
        st.text_area("Docs Preview", value=st.session_state.submitted_docs_text, height=200, disabled=True)
        
    with st.expander("Raw JSON Response"):
        st.json(result)

    with st.expander("ℹ Notes & Environment Setup"):
        st.markdown("""
            - **Environment variables**: For OpenAI, set `OPENAI_API_KEY`; for Gemini, set `GOOGLE_API_KEY`.
            - **Install dependencies**: `pip install streamlit openai google-generativeai numpy matplotlib reportlab`
            - **Run locally**: `streamlit run app.py`
        """)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
<hr>
<div style="text-align: center; padding: 20px;">
    <p>Made with ❤️ for <b>NatWest Hack4Cause</b> by Team <b>HackX</b></p>
</div>
""", unsafe_allow_html=True)
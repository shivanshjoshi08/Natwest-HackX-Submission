import os
import json
import textwrap
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from pathlib import Path

# Function to load CSS file
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Hide the Streamlit sidebar toggle/hamburger
st.markdown(
    """
    <style>
    /* hide main menu and sidebar dragger */
    [data-testid="stSidebarNav"] {display: none;}
    [data-testid="stSidebar"] {display: none;}
    /* also hide the "≡" toggle button */
    div[data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

# Session state for theme
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default theme

def toggle_theme():
    st.session_state.theme = (
        "light" if st.session_state.theme == "dark" else "dark"
    )

# Theme toggle button (no text of current theme shown)
st.button(
    f"Switch Theme",
    on_click=toggle_theme
)

# Load correct css file according to theme
css_file = Path(__file__).parent / f"style_{st.session_state.theme}.css"
load_css(css_file)

# Your app content starts here
st.title("🎓 CodeGrade AI — Static Code Review Assistant")

st.write("Paste requirements, upload files, and get a static code review.")
MAX_CODE_CHARS = 60_000
MAX_DOCS_CHARS = 30_000

EVAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "minimum": 0, "maximum": 10},
        "code_quality": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "feedback": {"type": "string"}
            },
            "required": ["score", "feedback"],
            "additionalProperties": False
        },
        "documentation": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "feedback": {"type": "string"}
            },
            "required": ["score", "feedback"],
            "additionalProperties": False
        },
        "adherence": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "feedback": {"type": "string"}
            },
            "required": ["score", "feedback"],
            "additionalProperties": False
        },
        "rubric": {
            "type": "object",
            "properties": {
                "what_went_well": {"type": "array", "items": {"type": "string"}},
                "areas_for_improvement": {"type": "array", "items": {"type": "string"}},
                "suggested_next_steps": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["what_went_well", "areas_for_improvement", "suggested_next_steps"],
            "additionalProperties": False
        }
    },
    "required": ["overall_score", "code_quality", "documentation", "adherence"],
    "additionalProperties": False
}

def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")

def _read_uploaded_text(uploader, limit: int) -> str:
    if uploader is None:
        return ""
    data = uploader.read()
    text = _decode_bytes(data)
    if len(text) > limit:
        head = text[:limit]
        note = f"\n\n[TRUNCATED to {limit} chars of {len(text)} total]"
        return head + note
    return text

def build_system_prompt() -> str:
    return textwrap.dedent(
        """
        You are an expert teaching assistant who performs *static* code reviews.
        You will receive:
        1) natural-language project requirements,
        2) one Python source file,
        3) a README or documentation file.

        IMPORTANT RULES:
        - Do NOT execute code; review by reading only.
        - Assess three categories: Code Quality, Documentation, Adherence to Requirements.
        - Use scores from 1-10 (10 is excellent). Be fair but strict.
        - Feedback must be concise, actionable, and specific to the submission.
        - If requirements are unclear or unmet, point that out explicitly.
        - Return ONLY a JSON object that matches the provided schema. Do not include backticks or prose.
        """
    ).strip()

def build_user_prompt(requirements: str, code_name: str, code_text: str, docs_name: str, docs_text: str) -> str:
    return textwrap.dedent(
        f"""
        [PROJECT REQUIREMENTS]
        {requirements.strip()}

        [STUDENT CODE: {code_name or "unnamed.py"}]
        {code_text}

        [DOCUMENTATION: {docs_name or "README.md"}]
        {docs_text or "(none provided)"}
        """
    ).strip()

def call_openai(messages, schema: Dict[str, Any], model: str) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. pip install openai") from e

    client = OpenAI()
    try:
        r = client.responses.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "codegrade_eval_schema",
                    "schema": schema,
                    "strict": True
                }
            },
        )
        text = getattr(r, "output_text", None)
        if not text:
            if hasattr(r, "output") and r.output and len(r.output) and hasattr(r.output[0], "content"):
                text = r.output[0].content[0].text
            else:
                text = str(r)
        return json.loads(text)
    except Exception:
        try:
            comp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            text = comp.choices[0].message.content
            return json.loads(text)
        except Exception as e2:
            raise RuntimeError(f"OpenAI call failed: {e2}")

def call_gemini(prompt_text: str, schema: Dict[str, Any], model: str) -> Dict[str, Any]:
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("Gemini SDK not installed. pip install google-generativeai") from e

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set in the environment.")
    genai.configure(api_key=api_key)

    model_obj = genai.GenerativeModel(model)
    try:
        resp = model_obj.generate_content(
            [prompt_text],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        text = resp.text
        return json.loads(text)
    except Exception:
        system_line = "Return ONLY a JSON object matching this JSON Schema (no backticks, no extra text):\n" + json.dumps(schema)
        resp = model_obj.generate_content(
            [system_line + "\n\n" + prompt_text],
            generation_config={
                "response_mime_type": "application/json",
            },
        )
        text = resp.text
        return json.loads(text)

def export_to_pdf(result):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate("report.pdf")
    story = []
    story.append(Paragraph("CodeGrade AI Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Overall Score: {result.get('overall_score','-')}/10", styles['Normal']))
    for key in ["code_quality","documentation","adherence"]:
        block = result.get(key,{})
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"{key.title()} Score: {block.get('score','-')}/10", styles['Normal']))
        story.append(Paragraph(f"Feedback: {block.get('feedback','')}", styles['Normal']))
    doc.build(story)

def radar_chart(scores):
    labels = np.array(["Code Quality","Documentation","Adherence"])
    values = np.array(scores)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values = np.concatenate((values,[values[0]]))
    angles = np.concatenate((angles,[angles[0]]))

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_yticks(range(0,11,2))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    st.pyplot(fig)

# ---------- FORM ----------
with st.form("grade-form", clear_on_submit=False):
    requirements = st.text_area("Project requirements", height=160)
    code_files = st.file_uploader("Upload Python files (.py)", type=["py"], accept_multiple_files=True)
    docs_file = st.file_uploader("Documentation (README.md or .txt)", type=["md","txt"])

    st.markdown("**AI Engine**")
    engine = st.selectbox("Provider", ["OpenAI", "Gemini"], index=0)
    if engine == "OpenAI":
        model_name = st.text_input("OpenAI model", value="gpt-4o-mini")
    else:
        model_name = st.text_input("Gemini model", value="gemini-1.5-pro")

    submitted = st.form_submit_button("🔍 Evaluate Project", use_container_width=True)

if submitted:
    if not requirements or not code_files:
        st.error("Please provide both project requirements and at least one Python code file.")
        st.stop()

    # Combine multiple files
    code_text = ""
    for cf in code_files:
        code_text += f"\n\n# File: {cf.name}\n" + _read_uploaded_text(cf, MAX_CODE_CHARS)

    docs_text = _read_uploaded_text(docs_file, MAX_DOCS_CHARS) if docs_file else ""

    st.info("Your files are read-only. This tool **does not execute** any uploaded code.")

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        requirements=requirements,
        code_name=", ".join([cf.name for cf in code_files]),
        code_text=code_text,
        docs_name=docs_file.name if docs_file else "",
        docs_text=docs_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with st.spinner("Analyzing submission..."):
        try:
            if engine == "OpenAI":
                result = call_openai(messages, EVAL_SCHEMA, model=model_name)
            else:
                prompt_text = system_prompt + "\n\n" + user_prompt
                result = call_gemini(prompt_text, EVAL_SCHEMA, model=model_name)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    st.subheader("📋 Evaluation Report")

    overall = result.get("overall_score")
    if isinstance(overall,(int,float)):
        pct = max(0,min(100,int((overall/10)*100)))
        st.progress(pct, text=f"Overall Score: {overall}/10")
    else:
        st.write("Overall Score: (not provided)")

    cols = st.columns(3)
    scores=[]
    for i, key in enumerate(["code_quality","documentation","adherence"]):
        with cols[i]:
            block=result.get(key,{})
            score=block.get("score","—")
            scores.append(score if isinstance(score,(int,float)) else 0)
            st.metric(label=key.replace("_"," ").title(),value=f"{score}/10" if isinstance(score,(int,float)) else "n/a")

    # Radar Chart
    st.markdown("### Score Radar")
    radar_chart(scores)

    for key,title in [("code_quality","Code Quality"),("documentation","Documentation"),("adherence","Adherence to Requirements")]:
        block=result.get(key,{})
        feedback=block.get("feedback") or "No feedback provided."
        st.markdown(f"**{title} Feedback**")
        st.write(feedback)

    rubric=result.get("rubric",{})
    if rubric:
        with st.expander("Detailed Rubric"):
            for list_key,nice in [("what_went_well","What went well"),("areas_for_improvement","Areas for improvement"),("suggested_next_steps","Suggested next steps")]:
                items=rubric.get(list_key) or []
                if items:
                    st.markdown(f"**{nice}**")
                    for it in items:
                        st.write("- "+str(it))

    # Side-by-side view of uploaded files
    st.markdown("### Uploaded Files Preview")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Code Files")
        st.code(code_text,language="python")
    with c2:
        st.subheader("Documentation")
        st.text_area("Docs Preview",value=docs_text,height=400)

    st.divider()
    st.markdown("### Raw JSON")
    st.json(result,expanded=False)

    # Download JSON
    st.download_button("⬇️ Download report JSON",data=json.dumps(result,ensure_ascii=False,indent=2),file_name="codegrade_report.json",mime="application/json")

    # Export PDF button
    if st.button("⬇️ Export Report as PDF"):
        export_to_pdf(result)
        with open("report.pdf","rb") as f:
            st.download_button("Download PDF",f,file_name="codegrade_report.pdf")

st.markdown("""
<hr>
<div style="text-align: center; padding: 20px;">
    <p>Made with ❤️ for <b>NatWest Hack4Cause</b> by Team <b>HackX</b></p>
</div>
""",unsafe_allow_html=True)
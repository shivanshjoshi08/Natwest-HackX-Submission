
import os
import json
import textwrap
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="CodeGrade AI", page_icon="🎓", layout="centered")
st.title("🎓 CodeGrade AI — Static Code Review Assistant")
st.caption("Paste requirements, upload a .py file and README.md, then let AI create a structured evaluation. (No code execution.)")

# ------------------------------
# Constants
# ------------------------------
MAX_CODE_CHARS = 60_000
MAX_DOCS_CHARS = 30_000

# Structured output schema (shared across providers)
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

# ------------------------------
# Helper functions
# ------------------------------
def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    # As a last resort, replace errors
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

# ------------------------------
# Provider backends
# ------------------------------
def call_openai(messages, schema: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Try OpenAI Responses API w/ structured outputs, then fall back to Chat Completions JSON mode."""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. pip install openai") from e

    client = OpenAI()  # API key read from env OPENAI_API_KEY

    # 1) Try Responses API with structured outputs (json_schema)
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
        # new SDK returns a convenient output_text
        text = getattr(r, "output_text", None)
        if not text:
            # Fallback parse for older/newer SDK variations
            if hasattr(r, "output") and r.output and len(r.output) and hasattr(r.output[0], "content"):
                text = r.output[0].content[0].text
            else:
                text = str(r)
        return json.loads(text)
    except Exception:
        # 2) Fallback: Chat Completions w/ JSON mode
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
    """Gemini SDK with structured output, falling back to JSON MIME only."""
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("Gemini SDK not installed. pip install google-generativeai") from e

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set in the environment.")
    genai.configure(api_key=api_key)

    model_obj = genai.GenerativeModel(model)

    # Try structured output (response_schema) first
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
        # Fallback: Just enforce JSON mime type and put schema in the prompt
        system_line = "Return ONLY a JSON object matching this JSON Schema (no backticks, no extra text):\n" + json.dumps(schema)
        resp = model_obj.generate_content(
            [system_line + "\n\n" + prompt_text],
            generation_config={
                "response_mime_type": "application/json",
            },
        )
        text = resp.text
        return json.loads(text)

# ------------------------------
# UI Controls
# ------------------------------
with st.form("grade-form", clear_on_submit=False):
    requirements = st.text_area(
        "Project requirements",
        height=160,
        placeholder="e.g., Create a Python script that calculates Body Mass Index (BMI) from user input...",
    )
    code_file = st.file_uploader("Student Python code (.py)", type=["py"])
    docs_file = st.file_uploader("Documentation (README.md or .txt)", type=["md", "txt"])

    st.markdown("**AI Engine**")
    engine = st.selectbox("Provider", ["OpenAI", "Gemini"], index=0, help="Choose which API to use for analysis.")
    if engine == "OpenAI":
        model_name = st.text_input("OpenAI model", value="gpt-4o-mini", help="e.g., gpt-4o, gpt-4o-mini, o4-mini, etc.")
    else:
        model_name = st.text_input("Gemini model", value="gemini-1.5-pro", help="e.g., gemini-1.5-pro, gemini-1.5-flash")

    submitted = st.form_submit_button("🔍 Evaluate Project", use_container_width=True)

if submitted:
    # Validate inputs
    if not requirements or not code_file:
        st.error("Please provide both project requirements and a Python code file.")
        st.stop()

    code_text = _read_uploaded_text(code_file, MAX_CODE_CHARS)
    docs_text = _read_uploaded_text(docs_file, MAX_DOCS_CHARS) if docs_file else ""

    st.info("Your files are read-only. This tool **does not execute** any uploaded code.")

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        requirements=requirements,
        code_name=code_file.name if code_file else "",
        code_text=code_text,
        docs_name=docs_file.name if docs_file else "",
        docs_text=docs_text,
    )

    # Messages format compatible with both OpenAI Responses (with messages) and Chat Completions
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with st.spinner("Analyzing submission..."):
        try:
            if engine == "OpenAI":
                result = call_openai(messages, EVAL_SCHEMA, model=model_name)
            else:
                # Gemini takes a single combined prompt
                prompt_text = system_prompt + "\n\n" + user_prompt
                result = call_gemini(prompt_text, EVAL_SCHEMA, model=model_name)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    # ------------------------------
    # Render Report
    # ------------------------------
    st.subheader("📋 Evaluation Report")

    overall = result.get("overall_score")
    if isinstance(overall, (int, float)):
        pct = max(0, min(100, int((overall / 10) * 100)))
        st.progress(pct, text=f"Overall Score: {overall}/10")
    else:
        st.write("Overall Score: (not provided)")

    cols = st.columns(3)
    for i, key in enumerate(["code_quality", "documentation", "adherence"]):
        with cols[i]:
            block = result.get(key, {})
            score = block.get("score", "—")
            st.metric(label=key.replace("_", " ").title(), value=f"{score}/10" if isinstance(score, (int, float)) else "n/a")

    for key, title in [
        ("code_quality", "Code Quality"),
        ("documentation", "Documentation"),
        ("adherence", "Adherence to Requirements"),
    ]:
        block = result.get(key, {})
        feedback = block.get("feedback") or "No feedback provided."
        st.markdown(f"**{title} Feedback**")
        st.write(feedback)

    rubric = result.get("rubric", {})
    if rubric:
        with st.expander("Detailed Rubric"):
            for list_key, nice in [
                ("what_went_well", "What went well"),
                ("areas_for_improvement", "Areas for improvement"),
                ("suggested_next_steps", "Suggested next steps"),
            ]:
                items = rubric.get(list_key) or []
                if items:
                    st.markdown(f"**{nice}**")
                    for it in items:
                        st.write("- " + str(it))

    st.divider()
    st.markdown("### Raw JSON")
    st.json(result, expanded=False)

    # Download JSON
    st.download_button(
        label="⬇️ Download report JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="codegrade_report.json",
        mime="application/json",
    )

    # Helpful info
    with st.expander("ℹ️ Notes & Environment Setup"):
        st.markdown(
            """
            **Environment variables**

            - For OpenAI: set `OPENAI_API_KEY`
            - For Gemini: set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)

            **Install dependencies**
            ```bash
            pip install streamlit openai google-generativeai
            ```

            **Run locally**
            ```bash
            streamlit run app.py
            ```

            This app never executes uploaded code; it only reads files to perform a static review.
            Large files may be truncated to fit within model limits.
            """
        )

# 🎓 CodeGrade AI

[![Python Version][python-shield]][python-url]
[![Streamlit Version][streamlit-shield]][streamlit-url]
[![License: MIT][license-shield]][license-url]

An elegant AI-powered assistant for static code review. CodeGrade AI leverages large language models like OpenAI's GPT and Google's Gemini to provide structured, insightful, and automated feedback on code submissions.

This tool is designed for educators, teaching assistants, and developers who need to review code quickly and consistently. It analyzes code quality, documentation, and adherence to project requirements without ever executing the code, ensuring a safe and secure evaluation process.

---

### ✨ App Preview

![CodeGrade AI App Preview](https://i.imgur.com/LhB2O9V.png)
*(A preview of the sleek and intuitive user interface)*

---

## 🚀 Key Features

* **🤖 Dual AI Engine Support**: Seamlessly switch between **OpenAI (GPT models)** and **Google Gemini** for analysis.
* **📊 Structured Evaluation**: Receive a detailed report with scores (0-10) and qualitative feedback across three core areas:
    * **Code Quality**: Readability, structure, and best practices.
    * **Documentation**: Clarity and completeness of the README file.
    * **Adherence to Requirements**: How well the code meets the project goals.
* **📝 Detailed Rubric**: Get actionable insights on "What Went Well," "Areas for Improvement," and "Suggested Next Steps."
* **🔒 Secure by Design**: The app performs **static analysis only** and **never executes** uploaded code.
* **⬇️ Downloadable Reports**: Export the complete evaluation as a JSON file for your records.
* **🎨 Modern UI**: A clean, responsive, and user-friendly interface built with Streamlit.

---

## 🛠️ Tech Stack

* **Language**: Python
* **Framework**: Streamlit
* **AI Providers**: OpenAI API, Google Gemini API
* **Dependencies**: `python-dotenv`, `openai`, `google-generativeai`

---

## ⚙️ Getting Started

Follow these steps to set up and run CodeGrade AI on your local machine.

### 1. Prerequisites

* **Python 3.8+**
* **Git** for cloning the repository.

### 2. Installation Steps

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/codegrade-ai.git](https://github.com/your-username/codegrade-ai.git)
    cd codegrade-ai
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    # Create venv
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies**:
    ```bash
    pip install streamlit openai google-generativeai python-dotenv
    ```

4.  **Set up your API keys**:
    Create a file named `.env` in the root of your project directory and add your keys.
    ```env
    # Get from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
    OPENAI_API_KEY="sk-..."

    # Get from: [https://aistudio.google.com/](https://aistudio.google.com/)
    GOOGLE_API_KEY="AIza..."
    ```

### 3. Running the Application

Once the installation is complete, run the following command in your terminal:

```bash
streamlit run app.py
# 📬 NatWest Hack4aCause hackathon Project Submission

---

## 📄 Summary of Your Solution (under 150 words)

CodeGrade AI solves the problem of manual, time-consuming, and inconsistent code reviews for educators and developers. It works by using Large Language Models (like GPT-4o-mini and Gemini 1.5 Pro) to perform a secure, static analysis of the code. The tool evaluates code quality, documentation, and adherence to project requirements without ever running the code, ensuring safety. It provides a detailed, structured report with scores, a visual radar chart, and actionable feedback. The core technologies used are **Python**, **Streamlit**, and the **OpenAI & Google Gemini APIs**.

## 👥 Team Information

| Field            | Details                               |
| ---------------- | ------------------------------------- |
| Team Name        | HackX                                 |
| Title            | CodeGrade AI                          |
| Theme            | AI-Powered Project Evaluator Tool    |
| Contact Email    | shivanshjoshi2922@gmail.com           |
| Participants     | Mohit Bajpai(Leader), Shivansh Joshi, Hemant Kumar, Karan Singh |
| GitHub Usernames | @MohitBajpai78271, @shivanshjoshi08, @Hemantisbuilding, @SKaran872 |

---

## 🎥 Submission Video

Provide a video walkthrough/demo of your project. You can upload it to a platform like YouTube, Google Drive, Loom, etc.

- 📹 **Video Link**: https://drive.google.com/file/d/195W4vQRdz9E5clPptcJYv3hGq7jUJYIe/view?usp=sharing

---

## 🌐 Hosted App / Solution URL

If your solution is deployed, share the live link here.

- 🌍 **Deployed URL**: [https://codegradeai1.streamlit.app/]

---

## License

Copyright 2025 FINOS

Distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

SPDX-License-Identifier: [Apache-2.0](https://spdx.org/licenses/Apache-2.0)

---
<br>

# 🎓 CodeGrade AI (Original README)

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991.svg?style=for-the-badge&logo=openai)](https://openai.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini_1.5_Pro-8A2BE2.svg?style=for-the-badge&logo=google-gemini)](https://ai.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**A smart AI assistant that statically reviews your code and provides detailed feedback. Perfect for educators and developers!**

---

### ✨ App Preview

<details>
  <summary>🖼️ Click to see App Previews</summary>
  <br>
  
  <img src="https://github.com/user-attachments/assets/25395159-649a-4f6a-bf2a-eeb66e905b3c" alt="Preview 1">
  <br>
  <img src="https://github.com/user-attachments/assets/b6b79941-7a2a-4eec-a7ed-e394eb5b3e87" alt="Preview 2">
  <br>
  <img src="https://github.com/user-attachments/assets/59c268df-c4ec-46e6-95d9-3ae78e06b09a" alt="Preview 3">
  <br>
  <img src="https://github.com/user-attachments/assets/603f7ec8-4e70-4a5d-9cc2-02a9191b83a6" alt="Preview 4">
  <br>
  <img src="https://github.com/user-attachments/assets/0746a286-9823-4aea-9dbd-8199629c244b" alt="Preview 5">
  <br>
  <img src="https://github.com/user-attachments/assets/e5440b42-eb26-4048-8589-f5c8707afba3" alt="Preview 6">
  <br>
  <img src="https://github.com/user-attachments/assets/390c99f8-e90a-4633-8324-302cf820075b" alt="Preview 7">
  <br>
  <img src="https://github.com/user-attachments/assets/a73c40dc-b848-450e-8fa9-d266e384dfa5" alt="Preview 8">

</details>

---

## 📖 About The Project

CodeGrade AI is a powerful tool that uses Large Language Models (like OpenAI's GPT and Google's Gemini) to provide structured and automated feedback on code submissions.

This tool is specifically designed for educators, teaching assistants, and developers who need to review code quickly and consistently. It analyzes code quality, documentation, and project requirements, **without ever running the code**, which ensures a safe and secure evaluation process.

*This project was created by **Team HackX** for the **NatWest Hack4Cause**.*

---

## 🚀 Key Features

* **🤖 Dual AI Engine Support**: Seamlessly switch between **OpenAI (GPT models)** and **Google Gemini**.
* **📊 Structured Evaluation**: Get detailed reports with scores (0-10) in three core areas:
    * **Code Quality**: Readability, structure, and best practices.
    * **Documentation**: Clarity and completeness of the README file.
    * **Adherence to Requirements**: How well the code follows the project goals.
* **📝 Detailed Rubric**: Actionable insights on "What Went Well," "Areas for Improvement," and "Suggested Next Steps."
* **🎨 Visual Reports**: Visualize scores on a beautiful **Radar Chart**.
* **⬇️ Downloadable Reports**: Export the entire evaluation as a **JSON file**.
* **🔒 Secure by Design**: The app only performs **static analysis** and **never executes** the uploaded code.
* **🎨 Modern UI**: A clean, responsive, and user-friendly interface built with Streamlit.

---

## 🛠️ Tech Stack

This project is built using these technologies:

* **Language**: `Python`
* **Framework**: `Streamlit`
* **AI Providers**: `OpenAI API`, `Google Gemini API`
* **Core Libraries**: `python-dotenv`, `openai`, `google-generativeai`
* **Visualization & Reporting**: `numpy`, `matplotlib`, `reportlab`

---

## ⚙️ Getting Started

Follow these steps to set up and run CodeGrade AI on your local machine.

### 1. Prerequisites

* **Python 3.8+**
* **Git** (to clone the repository).

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
    The project has a `requirements.txt` file that lists all the necessary libraries. Use it:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API keys**:
    Create a file named `.env` in the project's root directory and add your API keys. You can start by copying `.env.example`.
    ```env
    # Get from: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
    OPENAI_API_KEY="sk-..."

    # Get from: [https://aistudio.google.com/](https://aistudio.google.com/)
    GOOGLE_API_KEY="AIza..."
    ```

### 3. Run the Application
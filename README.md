# 🚀 Project Setup Guide

This guide provides step-by-step instructions for setting up the project environment, installing dependencies, and configuring the required AI services.

## Prerequisites

Before getting started, ensure the following tools are installed on your system:

* Python 3.10+
* Conda (Anaconda or Miniconda)
* Git
* Visual Studio Code
* Internet connection for dependency installation

---

## 1. Create the Project Workspace

```bash
# Create a new project directory
mkdir <project_folder_name>

# Navigate to the project directory
cd <project_folder_name>

# Open the project in Visual Studio Code
code .
```

---

## 2. Create and Activate a Virtual Environment

```bash
# Create a Conda environment with Python 3.10
conda create -p <env_name> python=3.10 -y

# Activate the environment
conda activate <path_to_env>
```

Verify the Python installation:

```bash
python --version
```

---

## 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

Verify installed packages:

```bash
pip list
```

---

## 4. Git Version Control Setup

Initialize Git and commit your changes:

```bash
# Initialize Git repository
git init

# Stage all project files
git add .

# Create initial commit
git commit -m "Initial project setup"
```

---

## 5. Connect and Push to GitHub

```bash
# Add remote repository
git remote add origin <repository_url>

# Rename branch to main
git branch -M main

# Push project to GitHub
git push -u origin main
```

---

## 6. Clone the Repository

```bash
git clone <repository_url>
```

Example:

```bash
git clone https://github.com/Sayancode2026/DocAI.git
```

---

# AI Model Support

The project supports integration with multiple Large Language Models (LLMs), Embedding Models, and Vector Databases.

## Supported LLM Providers

| Provider     | Access Type          |
| ------------ | -------------------- |
| Groq         | Free                 |
| OpenAI       | Paid                 |
| Gemini       | Free Trial Available |
| Claude       | Paid                 |
| Hugging Face | Free                 |
| Ollama       | Local Deployment     |

---

## Supported Embedding Models

* OpenAI Embeddings
* Hugging Face Embeddings
* Gemini Embeddings

---

## Supported Vector Databases

* In-Memory Vector Stores
* On-Disk Vector Databases
* Cloud-Based Vector Databases

---

# API Configuration

To use external AI services, obtain the required API keys and configure them in your environment variables.

## Groq API

* Generate API Key from the Groq Console
* Refer to the official documentation for setup and usage instructions

## Gemini API

* Generate API Key from Google AI Studio
* Refer to the official Gemini API documentation for implementation details

---

# Environment Variables

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

---

# Running the Application

```bash
python app.py
```

or

```bash
streamlit run app.py
```

(depending on the project architecture)

---

# Contributing

Contributions, feature requests, and bug reports are welcome. Please create an issue or submit a pull request to contribute to the project.

---

# License

This project is licensed under the MIT License.

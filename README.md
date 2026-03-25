# 🔍 Bias Audit Framework for Generative AI (Visuals)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI CLIP](https://img.shields.io/badge/Model-OpenAI_CLIP-green)](#)
[![DeepFace](https://img.shields.io/badge/Computer_Vision-DeepFace-orange)](#)
[![Paper](https://img.shields.io/badge/Published_in-INTED_2026-red)](#)

> **Research & Development:** An automated framework for quantitative and qualitative analysis of representational bias in Generative AI models. This repository contains the dataset, evaluation metrics, and automation scripts (using Computer Vision) initiated during an academic exchange at Universidad Nacional de Colombia (UNAL) and published at INTED 2026.

## 🚀 The Project
Generative AI models often perpetuate societal biases. This project provides a structured auditing methodology to detect and quantify these biases, specifically in university visual communication scenarios. 

The next phase of this framework (currently in development) automates the qualitative manual evaluation using **Computer Vision** and **Semantic Similarity models**, optimizing costs via Cloud Computing processing.

### 🛠️ Tech Stack & Tools
- **Python & Pandas/SciPy:** For data processing and non-parametric statistical testing.
- **OpenAI CLIP:** Used to evaluate the semantic alignment between the generated image and the text prompt.
- **DeepFace:** Applied for automated facial attribute analysis (detecting demographic cues to scale the audit process).
- **Google Colab:** Primary environment for zero-cost cloud processing and model execution.
- **Stable Diffusion:** Currently being integrated for localized, cost-effective image generation testing.

---

## 📦 Dataset Summary

The initial baseline dataset consists of:
- **64 AI-generated images**, produced from 8 scenarios × 2 prompt variants (neutral/inclusive) × 4 models.
- **4 Text-to-Image Systems Examined**: *DALL·E, Midjourney v6, Whisk, NanoBanana Pro*.
- **Full metadata:** Model name, prompt ID, timestamp, seed, evaluator scores (R1 & R2), and aggregated medians.

---

## 🧪 Evaluation Framework

### 1. Quantitative (Rubric & Statistical Testing)
All images were evaluated using a 5-dimensional diversity rubric (Racial/ethnic, Gender, Cultural fit, Body representation, and Visual quality). 

Due to the non-normal distribution of the data, median values were utilized. All statistical comparisons between inclusive and neutral prompts were conducted using the **Mann–Whitney U test**.

### 2. Qualitative (Content Analysis & Automation)
Manual inductive coding was initially applied to identify patterns such as `eurocentric_faces`, `light_skin_dominance`, and `athletic_body_norm`. 
*Note: The current roadmap replaces manual coding with automated DeepFace extraction to scale the auditing process.*

---

## 📊 Reproducibility & Usage

All statistical analyses, boxplots, and metrics can be reproduced locally or via Google Colab.

```bash
# Clone the repository
git clone [https://github.com/Kauandugi/Bias-ai-university-visuals.git](https://github.com/Kauandugi/Bias-ai-university-visuals.git)

# Install dependencies
pip install -r requirements.txt

# Run the Mann-Whitney U analysis script
python scripts/analysis_mannwhitney.py




---

## 📦 Dataset Summary

The dataset consists of:

- **64 AI-generated images**, produced from  
  8 scenarios × 2 prompt variants (neutral/inclusive) × 4 models
- **4 text-to-image systems examined**:
  - **DALL·E**
  - **Midjourney v6**
  - **Whisk**
  - **NanoBanana Pro**
- **Full metadata**, including:
  - model name  
  - prompt ID and variant  
  - file path  
  - timestamp  
  - seed (when available)  
  - evaluator scores (R1 & R2)  
  - aggregated means & medians  
- **Prompt dataset** (neutral & inclusive versions)
- **Boxplots for all metrics and comparisons**
- **Statistical testing results (Mann–Whitney U)**

---

## 🧪 Evaluation Framework

### **Quantitative (Rubric)**
All images were evaluated using a 5-dimensional diversity rubric:

1. **Racial and ethnic diversity**  
2. **Gender representation**  
3. **Cultural/contextual fit**  
4. **Body representation**  
5. **Visual quality and coherence**

Each dimension was scored independently by two evaluators (R1 & R2).

Median values were used instead of means due to non-normal distributions.

All statistical comparisons (inclusive vs. neutral prompts) were conducted using:

> **Mann–Whitney U test (two-sided)**

### **Qualitative (Content Analysis)**  
Evaluator notes were inductively coded using categories such as:

- `eurocentric_faces`  
- `light_skin_dominance`  
- `lack_of_diversity`  
- `athletic_body_norm`  
- `gender_imbalance`  
- `context_mismatch`  
- `latin_american_cues`  
- `hyperrealism_style`  

---

## 📊 Reproducibility (Python Analysis Scripts)

All statistical analyses and boxplots can be reproduced by running:

```bash
python scripts/analysis_mannwhitney.py

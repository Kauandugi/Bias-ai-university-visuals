import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# =========================
# 1. Ler e limpar a planilha
# =========================

INPUT_CSV = "bia_study_images.csv"

df = pd.read_csv(INPUT_CSV, encoding="utf-8")

# Remove colunas sem nome (como aquela entre R1_Visual_Quality e R2_Racial_Diversity)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Remove linhas totalmente vazias / separadoras (sem Image_id)
df = df[~df["Image_id"].isna() & (df["Image_id"].astype(str).str.strip() != "")].copy()

# =========================
# 2. Garantir tipos numéricos
# =========================

r1_cols = [
    "R1_Racial_Diversity",
    "R1_Gender_Representation",
    "R1_Cultural_Context_Fit",
    "R1_Body_Representation",
    "R1_Visual_Quality",
]

r2_cols = [
    "R2_Racial_Diversity",
    "R2_Gender_Representation",
    "R2_Cultural_Context_Fit",
    "R2_Body_Representation",
    "R2_Visual_Quality",
]

for col in r1_cols + r2_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 3. Calcular as médias por imagem (R1/R2)
#    (a unidade de análise continua sendo a imagem)
# =========================

df["Mean_racial_diversity"] = df[["R1_Racial_Diversity", "R2_Racial_Diversity"]].mean(axis=1)
df["Mean_gender_representation"] = df[["R1_Gender_Representation", "R2_Gender_Representation"]].mean(axis=1)
df["Mean_cultural_context_fit"] = df[["R1_Cultural_Context_Fit", "R2_Cultural_Context_Fit"]].mean(axis=1)
df["Mean_body_representation"] = df[["R1_Body_Representation", "R2_Body_Representation"]].mean(axis=1)
df["Mean_visual_quality"] = df[["R1_Visual_Quality", "R2_Visual_Quality"]].mean(axis=1)

# =========================
# 4. Criar variáveis derivadas úteis
# =========================

# Normalizar nome de modelo (tira versões entre parênteses)
df["Model_clean"] = df["Model"].str.replace(r"\s*\(.*\)", "", regex=True).str.strip()

# Categoria de Prompt: Inclusive vs Neutral
df["Prompt_variant"] = df["Prompt_variant"].str.strip()

# Coluna auxiliar para boxplots por modelo + variante
df["ModelPrompt"] = df["Model_clean"] + " - " + df["Prompt_variant"]

# =========================
# 5. Estatísticas descritivas por modelo (MEDIANAS)
# =========================

metrics = [
    "Mean_racial_diversity",
    "Mean_gender_representation",
    "Mean_cultural_context_fit",
    "Mean_body_representation",
    "Mean_visual_quality",
]

# Aqui usamos median e count (em vez de mean/std)
desc_by_model = df.groupby("Model_clean")[metrics].agg(["median", "count"])
desc_by_model.to_csv("results_descriptive_by_model.csv")

# =========================
# 6. Estatísticas por modelo + tipo de prompt (MEDIANAS)
# =========================

desc_by_model_variant = df.groupby(["Model_clean", "Prompt_variant"])[metrics].agg(["median", "count"])
desc_by_model_variant.to_csv("results_descriptive_by_model_and_variant.csv")

# =========================
# 7. Testes estatísticos: Inclusive vs Neutral (MANN–WHITNEY)
#    dentro de cada modelo, por métrica
# =========================

rows = []
for model in df["Model_clean"].unique():
    sub = df[df["Model_clean"] == model]
    inc = sub[sub["Prompt_variant"] == "Inclusive"]
    neu = sub[sub["Prompt_variant"] == "Neutral"]

    for m in metrics:
        inc_values = inc[m].dropna()
        neu_values = neu[m].dropna()

        if len(inc_values) >= 3 and len(neu_values) >= 3:
            # Mann–Whitney U, two-sided
            U_stat, p_val = stats.mannwhitneyu(inc_values, neu_values, alternative="two-sided")
        else:
            U_stat, p_val = np.nan, np.nan

        rows.append({
            "Model": model,
            "Metric": m,
            "Inclusive_n": len(inc_values),
            "Neutral_n": len(neu_values),
            "Inclusive_median": inc_values.median() if len(inc_values) > 0 else np.nan,
            "Neutral_median": neu_values.median() if len(neu_values) > 0 else np.nan,
            "U_stat": U_stat,
            "p_value": p_val,
        })

mw_results = pd.DataFrame(rows)
mw_results.to_csv("results_mannwhitney_inclusive_vs_neutral.csv", index=False)

# =========================
# 8. Boxplots (por modelo e por modelo+variante)
# =========================

# Boxplots por modelo (para a seção "Model performance overview")
for m in metrics:
    plt.figure(figsize=(6, 4))
    df.boxplot(column=m, by="Model_clean")
    plt.title(f"{m} by model")
    plt.suptitle("")  # remove título extra padrão do pandas
    plt.xlabel("Model")
    plt.ylabel("Score (Likert)")
    plt.tight_layout()
    plt.savefig(f"boxplot_{m}_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()

# Boxplots por modelo + variante de prompt (opcional, útil para analisar efeito do prompt)
for m in metrics:
    plt.figure(figsize=(8, 4))
    df.boxplot(column=m, by="ModelPrompt")
    plt.title(f"{m} by model and prompt variant")
    plt.suptitle("")
    plt.xlabel("Model - Prompt variant")
    plt.ylabel("Score (Likert)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"boxplot_{m}_by_model_prompt.png", dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# 9. Exportar planilha enriquecida
# =========================

df.to_csv("bia_study_images_with_means.csv", index=False)

print("Análises concluídas:")
print("- results_descriptive_by_model.csv (medianas por modelo)")
print("- results_descriptive_by_model_and_variant.csv (medianas por modelo+prompt)")
print("- results_mannwhitney_inclusive_vs_neutral.csv (Mann–Whitney)")
print("- boxplot_*_by_model.png")
print("- boxplot_*_by_model_prompt.png")
print("- bia_study_images_with_means.csv")

"""Turn a model prediction into a short hazard report for an inspector / manager.

The ML model makes the prediction; the LLM only explains it. If GEMINI_API_KEY is not set
(or LangChain is not installed), a deterministic template report is returned instead.
"""
import os


def build_prompt(row, prob, top_features):
    return (
        "You are a food-safety inspection analyst. Using ONLY the facts below, write a short, "
        "actionable hazard report (max 5 sentences) for a health inspector. Do not invent facts.\n\n"
        f"Borough: {row['boro']}\nCuisine: {row['cuisine']}\n"
        f"Predicted probability of a high-risk inspection (score >= 28): {prob:.0%}\n"
        f"Previous inspection score: {row['prev_score']}\n"
        f"Average past score: {row['hist_mean_score']:.1f}\n"
        f"Critical violations at last inspection: {row['prev_n_critical']}\n"
        f"Days since last inspection: {row['days_since_prev']}\n"
        f"Past inspections on record: {row['n_prev_inspections']}\n"
        f"Top model risk factors: {', '.join(top_features)}\n"
    )


def template_report(row, prob, top_features):
    level = "HIGH" if prob >= 0.5 else "MODERATE" if prob >= 0.25 else "LOW"
    return (
        f"{level} risk ({prob:.0%} chance of a score of 28 or more). "
        f"{row['cuisine']} restaurant in {row['boro']}; last score {row['prev_score']:.0f} "
        f"with {row['prev_n_critical']:.0f} critical violation(s), average past score "
        f"{row['hist_mean_score']:.1f} over {row['n_prev_inspections']} prior inspection(s), "
        f"{row['days_since_prev']:.0f} days since the last visit. "
        f"Main model drivers: {', '.join(top_features)}. "
        f"Recommend {'prioritising an early inspection' if prob >= 0.5 else 'keeping to the normal inspection schedule'}."
    )


def generate_report(row, prob, top_features, use_llm=True):
    """Return (report_text, backend) where backend is 'gemini' or 'template'."""
    prompt = build_prompt(row, prob, top_features)
    if use_llm and os.environ.get("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                api_key=os.environ["GEMINI_API_KEY"],
            )
            return llm.invoke(prompt).content, "gemini"
        except Exception:
            pass  # fall through to template
    return template_report(row, prob, top_features), "template"

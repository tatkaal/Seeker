import pandas as pd
from core.annotation_pipeline import AnnotationEngine
from core.llm_integrations import LLMIntegrator

# --- 7. Real-time Ad Analyzer (Advertiser Assistance) ---
class RealTimeAdAnalyzer:
    def __init__(self, annotation_engine: AnnotationEngine, llm_integrator: LLMIntegrator, all_ads_df: pd.DataFrame = None):
        self.annotation_engine = annotation_engine
        self.llm_integrator = llm_integrator
        self.all_ads_df = all_ads_df # Potentially used for comparisons

        # Pre-compute some stats if all_ads_df is available
        self.average_skill_count = 10 # Default
        if self.all_ads_df is not None and 'annotations' in self.all_ads_df.columns:
             # Ensure annotations column exists and is not null
            valid_annotations = self.all_ads_df['annotations'].dropna().apply(lambda x: x.get('all_skills_normalized', []) if isinstance(x, dict) else [])
            if not valid_annotations.empty:
                self.average_skill_count = valid_annotations.apply(len).mean()


    def analyze_for_advertiser(self, current_ad_text: str, title: str = "", classification: str = "") -> dict:
        insights = {}
        # Annotate the current ad text first
        # For real-time, LLM enhancement might be too slow unless highly optimized or selective
        # Let's use a faster annotation for this, or a flag
        current_annotations = self.annotation_engine.annotate_ad(current_ad_text, use_llm_enhancement=False) # Faster

        # a. Skill Suggester & Skills Radar data
        extracted_skills = current_annotations.get("all_skills_normalized", [])
        insights["extracted_skills_radar"] = [{"skill": s, "count": 1} for s in extracted_skills] # Basic radar data
        
        # Suggest more skills (simple heuristic: if less than average for this classification)
        # More advanced: use classification and title to find similar ads and suggest common skills
        if len(extracted_skills) < self.average_skill_count / 2: # Arbitrary threshold
            # Use LLM to suggest skills based on title/text
            llm_skill_prompt = f"Given the job ad title '{title}' and the following text snippet, suggest 3-5 relevant skills that might be missing or important to highlight:\n\n{current_ad_text[:500]}"
            suggested_skills_llm = self.llm_integrator.query_openai(llm_skill_prompt, max_tokens=100) if self.llm_integrator.openai_api_key else "Could not suggest skills (OpenAI key missing)."
            insights["skill_suggestions"] = suggested_skills_llm

        # b. Responsibility Clarity Checker
        responsibilities = current_annotations.get("responsibilities_ner", []) + current_annotations.get("responsibilities_llm", [])
        clarity_issues = []
        for resp in responsibilities:
            if len(resp.split()) < 3 : clarity_issues.append(f"Too short: '{resp}'")
            # Add more linguistic checks (passive voice, vague verbs)
        if not responsibilities: clarity_issues.append("No clear responsibilities identified. Ensure they are listed clearly.")
        insights["responsibility_clarity"] = clarity_issues if clarity_issues else ["Responsibilities look reasonably clear."]
        
        llm_clarity_prompt = f"Analyze the following list of job responsibilities for clarity, action-orientation, and conciseness. Provide brief feedback.\n\nResponsibilities:\n{responsibilities}"
        # insights["responsibility_llm_feedback"] = self.llm_integrator.query_openai(llm_clarity_prompt) # Example with local LLM

        # c. Conciseness Score (Heuristic + LLM)
        word_count = len(current_ad_text.split())
        conciseness_score = 50 # Base
        if word_count > 800: conciseness_score -= 20
        elif word_count < 200: conciseness_score -=10 # Too short might also be bad
        else: conciseness_score +=10
        # Readability (e.g., Flesch reading ease using a library like 'textstat')
        try:
            import textstat # pip install textstat
            readability = textstat.flesch_reading_ease(current_ad_text)
            if readability > 70: conciseness_score +=15
            elif readability < 30: conciseness_score -=15
        except ImportError:
            readability = "textstat not installed"
            
        insights["conciseness_score_heuristic"] = max(0, min(100, conciseness_score))
        insights["readability_flesch"] = readability

        # d. Optimal Ad Comparison (Conceptual - needs definition of "optimal")
        # insights["optimal_ad_comparison"] = "Feature not fully implemented. Compare against top ads in same category."

        # e. Missing Information Prompts
        missing_info = []
        if not current_annotations.get("experience_years_regex") and not current_annotations.get("experience_llm"):
            missing_info.append("Consider specifying years of experience or experience level.")
        if not current_annotations.get("benefits_ner") and not current_annotations.get("benefits_llm"):
            missing_info.append("Highlighting benefits can attract more candidates.")
        # Check metadata fields (location, workType if available from UI)
        insights["missing_information_prompts"] = missing_info if missing_info else ["Basic information seems present."]

        # f. Structure Guidance
        # Check for presence of common section keywords or use LLM
        # insights["structure_guidance"] = "Ensure clear sections for Responsibilities, Skills/Qualifications, and Benefits."

        # g. Candidate Appeal Score (Highly Conceptual)
        appeal_score = 70 # Base
        if "great team" in current_ad_text.lower() or "growth opportunity" in current_ad_text.lower(): appeal_score += 5
        if insights["conciseness_score_heuristic"] < 50: appeal_score -=10
        if clarity_issues and "No clear responsibilities" in clarity_issues[0] : appeal_score -=10
        insights["candidate_appeal_score_conceptual"] = max(0, min(100, appeal_score))
        
        # h. Ad Completeness Checklist
        insights["ad_completeness_checklist"] = {
            "Clear Title": bool(title and len(title) > 5),
            "Key Responsibilities Listed": bool(responsibilities),
            "Skills/Qualifications Mentioned": bool(extracted_skills),
            "Experience Level Indicated": bool(current_annotations.get("experience_years_regex") or current_annotations.get("experience_llm")),
            "Location Specified": True, # Assume UI provides this
            "Work Type Specified": True, # Assume UI provides this
            "Benefits Mentioned": bool(current_annotations.get("benefits_ner") or current_annotations.get("benefits_llm"))
        }
        
        # i. Style and Tone Suggestions (LLM)
        llm_style_prompt = f"Analyze the style and tone of the following job ad. Is it engaging, professional, inclusive? Provide brief suggestions for improvement if any.\n\n{current_ad_text[:1000]}"
        insights["style_tone_suggestions_llm"] = self.llm_integrator.query_openai(llm_style_prompt, max_tokens=150) if self.llm_integrator.openai_api_key else "Style/Tone LLM check requires OpenAI key."
        
        return insights

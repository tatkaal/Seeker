# --- 5. Core Annotation Engine ---
import re

class AnnotationEngine:
    from core.text_processor import TextProcessor
    from core.skill_taxonomy import SkillTaxonomy
    from core.llm_integrations import LLMIntegrator
    
    def __init__(self, text_processor: TextProcessor, skill_taxonomy: SkillTaxonomy, llm_integrator: LLMIntegrator):
        self.text_processor = text_processor
        self.skill_taxonomy = skill_taxonomy
        self.llm_integrator = llm_integrator
        self.nlp_spacy = text_processor.nlp_spacy

    def extract_entities(self, text: str) -> dict:
        """Uses the DEFAULT NER model."""
        doc = self.nlp_spacy(text)
        # We can still use default entities like ORG, PRODUCT, LOC
        entities = {"SKILL": [], "RESPONSIBILITY_PHRASE": [], "EXPERIENCE_YEARS": [], 
                    "CERTIFICATION": [], "DEGREE": [], "BENEFIT": [],
                    "ORG": [], "LOC": [], "PRODUCT": []}
        for ent in doc.ents:
            if ent.label_ in entities: # spaCy default entities
                    entities[ent.label_].append(ent.text)
        return entities

    def extract_skills_keywords(self, text: str) -> list:
        extracted_skills = set()
        text_lower = text.lower()
        known_skills_sorted = sorted(self.skill_taxonomy.get_all_skills(), key=len, reverse=True) # Match longer skills first

        for skill in known_skills_sorted:
            # Regex to match whole words/phrases to avoid partial matches
            try:
                # Use word boundaries \b, ensure skill is properly escaped if it contains regex special chars
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    extracted_skills.add(skill)
                    # Try to remove the matched skill to avoid re-matching parts of it by shorter skills
                    # This is a simple approach; more sophisticated methods exist.
                    text_lower = text_lower.replace(skill, "", 1)
            except re.error:
                if skill in text_lower: # Fallback for complex skill names that break regex
                    extracted_skills.add(skill)
                    text_lower = text_lower.replace(skill, "", 1)
        
        # Add to dynamic taxonomy
        for skill_found in extracted_skills:
            self.skill_taxonomy.add_skill_candidate(skill_found)
            
        return sorted(list(extracted_skills))

    def extract_responsibilities(self, text: str, entities_from_ner: dict) -> list:
        """
        Combines NER responsibility phrases with list item extraction
        and potentially LLM-based structuring.
        """
        responsibilities = set(entities_from_ner.get("RESPONSIBILITY_PHRASE", []))
        
        # Heuristic: look for bullet points (actual HTML <li> would be better if parsed earlier)
        # Or sentences under "Responsibilities" type headers.
        # For this example, we'll just use the NER output.
        # An LLM could further refine these (e.g., ensure verb-driven phrases).
        
        # Simple verb phrase check if not using advanced parsing
        doc = self.nlp_spacy(text)
        COMMON_SECTION_KEYWORDS = {
            "responsibilities": ["responsibilities", "key tasks", "duties & responsibilities", "your role", "the role", "about the role", "key responsibilities"],
            "qualifications_skills": ["skills", "requirements", "qualifications", "criteria", "what you'll bring", "about you", "to succeed", "required skills", "knowledge, skills, and abilities"],
            "experience": ["experience", "years of experience"],
            "education": ["education", "degree"],
            "benefits": ["benefits", "what we offer", "perks"]
        }
        for sent in doc.sents:
            if any(kw in sent.text.lower() for kw_list in COMMON_SECTION_KEYWORDS["responsibilities"] for kw in kw_list):
                 # This sentence is likely part of a responsibility section header itself
                continue

            # A very basic heuristic: sentence starts with a verb and is of reasonable length
            if sent.root.pos_ == "VERB" and 5 < len(sent.text.split()) < 30:
                # Further checks needed to see if it's contextually a responsibility
                # For now, let's assume if our NER didn't pick it up, we add it if it sounds like one
                # responsibilities.add(sent.text.strip())
                pass # This needs more refinement to avoid noise

        return sorted(list(responsibilities))
        
    def extract_experience_years(self, text: str) -> list:
        """Extracts mentions of years of experience (e.g., "3+ years experience")."""
        patterns = [
            r'(\d+\s*-\s*\d+\s*years?)', # e.g., 3-5 years
            r'(\d+\s*to\s*\d+\s*years?)', # e.g., 3 to 5 years
            r'(\d+\+?\s*years?)\s*(?:of)?\s*(?:relevant\s*)?experience', # 3 years experience, 5+ years of relevant experience
            r'minimum\s*of?\s*(\d+\s*years?)', # minimum 2 years
            r'at\s*least\s*(\d+\s*years?)', # at least 3 years
            r'(\d+)\s*or\s*more\s*years', # 5 or more years
            r'over\s*(\d+)\s*years' # over 7 years
        ]
        found_experience = set()
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Find the group that actually captures the number/range
                for i in range(len(match.groups()) + 1):
                    if match.group(i) and any(char.isdigit() for char in match.group(i)):
                        exp_text = match.group(i).strip()
                        # Normalize: "3+ years" -> "3+ years"
                        # "3 years" -> "3 years"
                        # "minimum 2 years" -> "minimum 2 years" (the regex tries to capture core)
                        # We can refine normalization later.
                        found_experience.add(exp_text)
                        break
        return sorted(list(found_experience))


    def annotate_ad(self, job_ad_text: str, use_llm_enhancement=True) -> dict:
        """Main annotation function for a single job ad text."""
        annotations = {
            "skills_keyword": [], "skills_ner": [], "skills_llm": [], "all_skills_normalized": [],
            "responsibilities_ner": [], "responsibilities_llm": [],
            "experience_years_regex": [], "experience_llm": [],
            "qualifications_ner": [], "qualifications_llm": [], # Degrees, Certs
            "benefits_ner": [], "benefits_llm": [],
            "other_entities": {}
        }

        # 1. Preprocessing already assumed to be done (HTML cleaning)
        
        # 2. NER Extraction (spaCy default)
        ner_entities = self.extract_entities(job_ad_text)
        annotations["skills_ner"] = ner_entities.get("SKILL", [])
        annotations["responsibilities_ner"] = self.extract_responsibilities(job_ad_text, ner_entities) #ner_entities.get("RESPONSIBILITY_PHRASE", [])
        annotations["experience_years_regex"] = self.extract_experience_years(job_ad_text) # Regex is often good for this
        # NER could also identify degrees, certs, benefits if trained
        annotations["qualifications_ner"].extend(ner_entities.get("DEGREE", []))
        annotations["qualifications_ner"].extend(ner_entities.get("CERTIFICATION", []))
        annotations["benefits_ner"] = ner_entities.get("BENEFIT", [])
        
        # Store other potentially useful NER entities
        for k, v in ner_entities.items():
            if k not in ["SKILL", "RESPONSIBILITY_PHRASE", "DEGREE", "CERTIFICATION", "BENEFIT"] and v:
                annotations["other_entities"][k] = v
                
        # 3. Keyword-based Skill Extraction
        annotations["skills_keyword"] = self.extract_skills_keywords(job_ad_text)

        # 4. LLM Enhancement (if enabled)
        if use_llm_enhancement and self.llm_integrator:
            print("Applying LLM enhancement...")
            info_schema = {
                "skills": "list of specific technical and soft skills mentioned",
                "responsibilities": "list of key job responsibilities or tasks, phrased as actions if possible",
                "experience_required": "specific years of experience or experience level (e.g., '3-5 years', 'senior level')",
                "education_qualifications": "list of degrees, certifications, or educational requirements",
                "benefits_perks": "list of employee benefits or perks mentioned"
            }
            llm_extracted = self.llm_integrator.extract_structured_info_llm(
                job_ad_text, info_schema
            )
            if llm_extracted:
                annotations["skills_llm"] = [s.lower() for s in llm_extracted.get("skills", []) if isinstance(s, str)]
                annotations["responsibilities_llm"] = llm_extracted.get("responsibilities", [])
                annotations["experience_llm"] = llm_extracted.get("experience_required", []) # This might be a string or list
                annotations["qualifications_llm"] = llm_extracted.get("education_qualifications", [])
                annotations["benefits_llm"] = llm_extracted.get("benefits_perks", [])
                # Add LLM skills to taxonomy candidate list
                for skill in annotations["skills_llm"]:
                    self.skill_taxonomy.add_skill_candidate(skill)

        # 5. Consolidate and Normalize Skills
        all_found_skills = set(annotations["skills_keyword"]) | set(annotations["skills_ner"]) | set(annotations["skills_llm"])
        normalized_skills = {self.skill_taxonomy.get_normalized_skill(s) for s in all_found_skills if s}
        annotations["all_skills_normalized"] = sorted(list(normalized_skills))

        return annotations

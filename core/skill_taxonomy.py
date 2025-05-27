from collections import Counter

# --- 3. Skill Taxonomy (Dynamic Concept & Simplified Implementation) ---
class SkillTaxonomy:
    def __init__(self):
        # In a real system, this would be loaded from a database/knowledge graph
        # and regularly updated (e.g., from ESCO, LinkedIn skills, Seek's own data)
        self.base_skills = {
            # Technical
            "python", "java", "c++", "javascript", "sql", "aws", "azure", "docker", "kubernetes",
            "machine learning", "data analysis", "react", "angular", "node.js", "agile", "scrum",
            "cybersecurity", "project management", "tableau", "power bi", "sap", "salesforce",
            # Soft
            "communication", "teamwork", "problem solving", "leadership", "customer service",
            "attention to detail", "time management",
            # Domain-specific (from samples)
            "real estate", "crm systems", "forklift operation", "b737 type rating", 
            "quality assurance", "gmp", "haccp", "construction management", "contracts administration"
        }
        self.skill_variants = { # For normalization
            "ms office": "microsoft office suite",
            "excel": "microsoft excel",
            "project manager": "project management"
        }
        # For dynamic updates:
        self.learned_skills = set() # Store skills learned from ads (would need validation)
        self.skill_frequency = Counter() # Track frequency of seen skills

    def get_normalized_skill(self, skill_text: str) -> str:
        skill_text_lower = skill_text.lower().strip()
        if skill_text_lower in self.skill_variants:
            return self.skill_variants[skill_text_lower]
        return skill_text_lower

    def is_known_skill(self, skill_text: str) -> bool:
        return self.get_normalized_skill(skill_text) in self.base_skills or \
               self.get_normalized_skill(skill_text) in self.learned_skills
    
    def add_skill_candidate(self, skill_text: str, source="job_ad_extraction"):
        """
        Adds a potential new skill. In a real system, this would go into a
        validation pipeline (human-in-the-loop, frequency checks, etc.).
        """
        normalized_skill = self.get_normalized_skill(skill_text)
        if not self.is_known_skill(normalized_skill) and len(normalized_skill) > 2 : # Basic filter
            # Heuristic: If seen frequently, more likely a real skill
            self.skill_frequency[normalized_skill] +=1
            if self.skill_frequency[normalized_skill] > 5: # Arbitrary threshold
                 print(f"[Taxonomy] Learned new skill candidate: {normalized_skill} (seen {self.skill_frequency[normalized_skill]} times)")
                 self.learned_skills.add(normalized_skill)
                 self.base_skills.add(normalized_skill) # Add to base for this run

    def get_all_skills(self):
        return list(self.base_skills.union(self.learned_skills))

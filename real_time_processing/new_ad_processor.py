from core.annotation_pipeline import AnnotationEngine
from core.html_parser import HTMLParser
MIN_TEXT_LENGTH_FOR_ANALYSIS = 100  # Minimum length of text to consider for analysis

# --- 8. Real-time Processing of New Postings (Conceptual Backend) ---
class NewAdProcessor:
    def __init__(self, annotation_engine: AnnotationEngine):
        self.annotation_engine = annotation_engine
        # In a real system, this would interact with a message queue (Kafka, RabbitMQ)
        # and a database to store annotations.

    def process_new_ad(self, job_ad_data: dict) -> dict:
        """Simulates processing a new ad when it's posted."""
        print(f"Processing new ad ID: {job_ad_data.get('id', 'N/A')}")
        raw_text = HTMLParser.clean_html(job_ad_data.get("content", ""))
        
        if len(raw_text) < MIN_TEXT_LENGTH_FOR_ANALYSIS:
            print(f"Ad {job_ad_data.get('id')} has insufficient content. Skipping full annotation.")
            return {"id": job_ad_data.get('id'), "status": "skipped_short_content", "annotations": {}}

        # Use LLM for new ads if configured, as this is background processing
        annotations = self.annotation_engine.annotate_ad(raw_text, use_llm_enhancement=True)
        
        # Store annotations (e.g., in Elasticsearch or a relational DB alongside the ad)
        print(f"Annotations for ad {job_ad_data.get('id')}: {len(annotations['all_skills_normalized'])} skills found.")
        # Placeholder for saving
        # self.save_annotations_to_db(job_ad_data.get('id'), annotations)
        
        # Trigger downstream processes (e.g., update search index, alerts)

        return {"id": job_ad_data.get('id'), "status": "processed", "annotations": annotations}

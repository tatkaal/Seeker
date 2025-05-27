import json
import re
import os
from datetime import datetime
from collections import Counter

# Data Handling and EDA specific
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import spacy
spacy.require_gpu()

# Machine Learning & Embeddings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

import openai

#Local imports for modular components
from core.data_loader import DataLoader

from core.skill_taxonomy import SkillTaxonomy
from core.llm_integrations import LLMIntegrator
from eda.exploratory_analysis import JobAdEDA
from eda.eda_v2 import JobAdEDAAdvanced, EDAConfig
from core.annotation_pipeline import AnnotationEngine
from evaluation.performance_metrics import Evaluator
from advertiser_tools.real_time_analyzer import RealTimeAdAnalyzer
from real_time_processing.new_ad_processor import NewAdProcessor
from advanced_features.advanced_features import AdvancedFeatures
from core.html_parser import HTMLParser
from core.text_processor import TextProcessor

# from advertiser_tools.market_insights import MarketInsights
# from advertiser_tools.semantic_search import SemanticSearch
# from advertiser_tools.style_tone_suggestions import StyleToneSuggestions
# from advertiser_tools.conciseness_checker import ConcisenessChecker
# from advertiser_tools.responsibility_clarity_checker import ResponsibilityClarityChecker
# from advertiser_tools.skill_suggester import SkillSuggester
# from advertiser_tools.ad_completeness_checker import AdCompletenessChecker
# from advertiser_tools.candidate_appeal_analyzer import CandidateAppealAnalyzer
# from advertiser_tools.advertiser_insights import AdvertiserInsights
# from advertiser_tools.advertiser_tools import AdvertiserTools

# --- Configuration ---
DATA_FILE_PATH = "Data/original.json"

MIN_TEXT_LENGTH_FOR_ANALYSIS = 100

# Download NLTK resources if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')
try:
    nltk.pos_tag(["test"])
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    WordNetLemmatizer().lemmatize("tests")
except LookupError:
    nltk.download('wordnet')

# --- Main Execution / Orchestration ---
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Script started at {start_time}")

    # --- Setup ---
    data_loader = DataLoader(DATA_FILE_PATH)
    # Load a smaller sample for quick dev, or all for full run (e.g., 500 for dev, None for full)
    # For EDA of 50k, load all. For annotation pipeline testing, a sample is faster.
    # We'll simulate full EDA load, then process a smaller sample for annotation part of this script.
    print("Loading full dataset for EDA (can be slow for 50k)...")
    full_df = data_loader.load_data(sample_size=None) 
    print(f"Loaded {len(full_df)} job ads for EDA.")
    
    if full_df.empty:
        print("No data loaded. Exiting.")
        exit()

    text_processor_inst = TextProcessor()
    skill_taxonomy_inst = SkillTaxonomy()
    
    llm_integrator_inst = LLMIntegrator(openai_api_key=openai_key)

    # --- EDA ---
    # # To run EDA, uncomment:
    # eda_runner = JobAdEDA(full_df.head(1000), text_processor_inst) # Use a large sample for EDA if 50k is too slow for interactive visuals
    # eda_runner.run_full_eda()
    # print(f"EDA took: {datetime.now() - start_time}")

    # --- EDA v2 ---
    cfg = EDAConfig()
    eda_v2_runner = JobAdEDAAdvanced(full_df, cfg)
    eda_v2_runner.run_all()
    print(f"EDA v2 took: {datetime.now() - start_time}")

    import sys
    sys.exit(0)

    # --- Annotation Pipeline ---
    print("\n--- Setting up Annotation Pipeline ---")
    # Process a smaller sample for annotation demonstration to manage runtime
    annotation_sample_df = full_df.sample(n=min(20, len(full_df)), random_state=101)
    annotation_sample_df['cleaned_content'] = annotation_sample_df['content'].apply(HTMLParser.clean_html)
    
    annotation_engine_inst = AnnotationEngine(text_processor_inst, skill_taxonomy_inst, llm_integrator_inst)
    
    print(f"\n--- Annotating Sample of {len(annotation_sample_df)} Job Ads ---")
    annotated_ads_list = []
    for index, row in annotation_sample_df.iterrows():
        print(f"\nAnnotating Ad ID: {row.get('id', index)}, Title: {row.get('title', 'N/A')[:50]}...")
        if pd.isna(row['cleaned_content']) or len(row['cleaned_content']) < MIN_TEXT_LENGTH_FOR_ANALYSIS:
            print("Skipping due to short/missing content.")
            annotated_ads_list.append({"id": row.get('id', index), "annotations": {}, "error": "short_content"})
            continue
        
        current_annotations = annotation_engine_inst.annotate_ad(
            row['cleaned_content'], 
            use_llm_enhancement=True, # Set to False to speed up if LLMs are slow/costly for many ads
        )
        print(f"  Extracted Skills: {current_annotations.get('all_skills_normalized', [])[:5]}")
        print(f"  Responsibilities (LLM): {current_annotations.get('responsibilities_llm', [])[:3]}")
        print(f"  Responsibilities (NER): {current_annotations.get('responsibilities_ner', [])[:3]}")
        print(f" Experience Years (Regex): {current_annotations.get('experience_years_regex', 'N/A')}")
        print(f" Experience Years (LLM): {current_annotations.get('experience_llm', 'N/A')}")
        print(f" Qualifications (NER): {current_annotations.get('qualifications_ner', [])[:3]}")
        print(f" Qualifications (LLM): {current_annotations.get('qualifications_llm', 'N/A')}")
        print(f" Benefits (NER): {current_annotations.get('benefits_ner', [])[:3]}")
        print(f" Benefits (LLM): {current_annotations.get('benefits_llm', 'N/A')}")
        print(f" Other Entities: {current_annotations.get('other_entities', {})}")

        annotated_ads_list.append({"id": row.get('id', index), "title": row.get('title'), "annotations": current_annotations})

    # Merge annotations back to a DataFrame (conceptual for further use)
    # This creates a list of dicts, convert to DF or Series to merge with full_df for advanced features
    # For simplicity, we'll pass this list to evaluation and other components
    df_with_annotations_sample = pd.DataFrame(annotated_ads_list)

    # --- Evaluation (on the sample) ---
    print("\n--- Evaluating Annotations (Sample) ---")
    evaluator = Evaluator()
    evaluator.load_gold_standard(None) # Loads mock gold standard
    all_eval_results = []
    for _, ad_data in df_with_annotations_sample.iterrows():
        if 'error' not in ad_data['annotations']: # if ad was not skipped
            eval_res = evaluator.evaluate_annotations(ad_data['id'], ad_data['annotations'])
            if eval_res: all_eval_results.append(eval_res)
    
    if all_eval_results:
        avg_skill_f1 = np.mean([res['skills_f1'] for res in all_eval_results if 'skills_f1' in res])
        print(f"Average Skill F1 on evaluated sample: {avg_skill_f1:.2f}")


    # --- Real-time Ad Analyzer Demo (for one ad from the sample) ---
    print("\n--- Real-Time Ad Analyzer Demo ---")
    if not df_with_annotations_sample.empty:
        sample_ad_for_analyzer_text = annotation_sample_df.iloc[0]['cleaned_content']
        sample_ad_for_analyzer_title = annotation_sample_df.iloc[0]['title']
        
        # Pass the full_df with its annotations column if available for better comparisons
        # For this script, we'll use the small annotated sample for `all_ads_df` argument.
        # In a real system, `all_ads_df` would be the full 50k set *with their annotations pre-computed*.
        # We'll fake this by assigning the sample annotations to the `annotations` column.
        # This is a bit circular for THIS script's flow but illustrates how RealTimeAdAnalyzer would use it.
        # Create a temporary df that has an 'annotations' column for the analyzer's internal stats.
        # For this demo, it only uses average_skill_count.
        # We will use the small sample that was annotated.
        
        # Create a temporary df that looks like what RealTimeAdAnalyzer expects for `all_ads_df`
        temp_df_for_analyzer = full_df.copy().head(len(df_with_annotations_sample)) # Match size of annotated sample
        # Make sure 'annotations' exist and merge results for 'average_skill_count' calculation
        # This step is just to provide *some* data for the analyzer's internal stats
        temp_annotations_series = pd.Series([d.get('annotations',{}) for d in annotated_ads_list], index=temp_df_for_analyzer.index[:len(annotated_ads_list)])
        temp_df_for_analyzer['annotations'] = temp_annotations_series

        analyzer = RealTimeAdAnalyzer(annotation_engine_inst, llm_integrator_inst, all_ads_df=temp_df_for_analyzer)
        advertiser_insights = analyzer.analyze_for_advertiser(sample_ad_for_analyzer_text, sample_ad_for_analyzer_title)
        
        print(f"Insights for Advertiser (Ad: '{sample_ad_for_analyzer_title[:50]}...'):")
        print(f"  Extracted Skills Radar (sample): {advertiser_insights.get('extracted_skills_radar', [])[:3]}")
        print(f"  Skill Suggestions: {advertiser_insights.get('skill_suggestions', 'N/A')}")
        print(f"  Responsibility Clarity: {advertiser_insights.get('responsibility_clarity', ['N/A'])[:1]}")
        print(f"  Conciseness Score (Heuristic): {advertiser_insights.get('conciseness_score_heuristic', 'N/A')}")
        print(f"  Missing Info Prompts: {advertiser_insights.get('missing_information_prompts', [])}")
        print(f"  Ad Completeness (Title): {advertiser_insights.get('ad_completeness_checklist', {}).get('Clear Title')}")
        print(f"  Candidate Appeal (Conceptual): {advertiser_insights.get('candidate_appeal_score_conceptual', 'N/A')}")
        print(f"  Style/Tone LLM Suggestion: {advertiser_insights.get('style_tone_suggestions_llm', 'N/A')}")


    # --- Real-time Processing of New Ad (Demo) ---
    print("\n--- New Ad Processing Demo ---")
    new_ad_processor_inst = NewAdProcessor(annotation_engine_inst)
    if not annotation_sample_df.empty:
        # Take one of the sample ads as if it's "new"
        new_ad_payload = annotation_sample_df.iloc[0].to_dict() 
        processed_new_ad = new_ad_processor_inst.process_new_ad(new_ad_payload)
        # print(f"Processed new ad demo result: {processed_new_ad.get('status')}")


    # --- Advanced Features Demo ---
    print("\n--- Advanced Features Demo (Conceptual) ---")
    # For advanced features, we'd ideally use the full dataset with annotations.
    # We'll use our small annotated sample `df_with_annotations_sample` for this demo.
    # `df_with_annotations_sample` contains 'id', 'title', 'annotations'. We need to merge this info.
    
    # Prepare a dataframe similar to what AdvancedFeatures would expect
    # It needs a 'annotations' column with the dicts from annotation_engine_inst
    if not df_with_annotations_sample.empty and 'annotations' in df_with_annotations_sample.columns:
        advanced_features_df = df_with_annotations_sample # Use the sample as it is.
        
        adv_features_inst = AdvancedFeatures(advanced_features_df, skill_taxonomy_inst, llm_integrator_inst)
        
        # Knowledge Graph (NetworkX simulation)
        adv_features_inst.init_knowledge_graph_nx() # Will use the small sample
        if hasattr(adv_features_inst, 'kg_graph_nx') and adv_features_inst.kg_graph_nx.number_of_nodes() > 0:
            print("  KG: Example - Skills for first job title (if any edges exist):")
            first_title_node_example = advanced_features_df['title'].iloc[0]
            if adv_features_inst.kg_graph_nx.has_node(first_title_node_example):
                 neighbors = list(adv_features_inst.kg_graph_nx.neighbors(first_title_node_example))
                 skills_for_title = [n for n in neighbors if adv_features_inst.kg_graph_nx.nodes[n].get('type') == 'skill']
                 print(f"    Skills for '{first_title_node_example[:30]}...': {skills_for_title[:5]}")
        
        # Hyper-Personalized Feedback
        if not df_with_annotations_sample.empty:
            first_ad_row = annotation_sample_df.iloc[0]
            first_ad_text = first_ad_row['cleaned_content']
            first_ad_title = first_ad_row['title']
            first_ad_class = first_ad_row.get('metadata', {}).get('classification', {}).get('name', 'Unknown Classification') # Extract classification
            first_ad_annotations = df_with_annotations_sample[df_with_annotations_sample['id'] == first_ad_row['id']].iloc[0]['annotations']
            
            hyper_feedback = adv_features_inst.get_hyper_personalized_feedback(
                first_ad_text, first_ad_title, first_ad_class, first_ad_annotations
            )
            print(f"\n  Hyper-Personalized Feedback for Ad '{first_ad_title[:30]}...':\n    {hyper_feedback[:200]}...")

        # Market Insights
        market_trends = adv_features_inst.get_skill_trends(top_k=5)
        print(f"\n  Market Skill Trends (from sample): {market_trends}")

        # Semantic Search
        semantic_search_results = adv_features_inst.semantic_search_candidates(
            query="entry-level remote software engineer jobs with python"
        )
        print(f"\n  Semantic Search Demo: {semantic_search_results}")


    end_time = datetime.now()
    print(f"\n--- Script finished at {end_time}. Total duration: {end_time - start_time} ---")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk import word_tokenize
from wordcloud import WordCloud
from core.text_processor import TextProcessor
from core.html_parser import HTMLParser

class JobAdEDA:
    def __init__(self, df: pd.DataFrame, text_processor: TextProcessor):
        self.df = df.copy()
        self.text_processor = text_processor
        
        # Clean content once for EDA
        print("EDA: Cleaning HTML content for analysis...")
        self.df['cleaned_content'] = self.df['content'].apply(HTMLParser.clean_html)
        self.df['processed_content'] = self.df['cleaned_content'].apply(lambda x: self.text_processor.preprocess(x))

    def plot_text_length_distributions(self):
        print("\n--- EDA: Text Length Distributions ---")
        for col in ['title', 'abstract', 'cleaned_content']:
            if col in self.df:
                self.df[f'{col}_len'] = self.df[col].astype(str).apply(len)
                plt.figure(figsize=(10, 4))
                sns.histplot(self.df[f'{col}_len'], kde=True, bins=50)
                plt.title(f'Distribution of {col} Length')
                plt.xlabel('Length (characters)')
                plt.ylabel('Frequency')
                plt.show()
                print(f"Basic stats for {col} length:\n{self.df[f'{col}_len'].describe()}")

    def analyze_metadata_distributions(self):
        print("\n--- EDA: Metadata Distributions ---")
        metadata_cols = ['classification', 'subClassification', 'location', 'workType']
        for col_name in metadata_cols:
            # Values might be dicts like {'name': 'Actual Value'}
            def extract_name(val):
                if isinstance(val, dict) and 'name' in val:
                    return val['name']
                return val # if it's already a string or None

            if 'metadata' in self.df.columns:
                 # Create a temporary column for the specific metadata field if it doesn't directly exist
                temp_col_data = self.df['metadata'].apply(lambda x: x.get(col_name) if isinstance(x, dict) else None)
                extracted_data = temp_col_data.apply(extract_name)
                
                if not extracted_data.isnull().all():
                    plt.figure(figsize=(12, 6))
                    sns.countplot(y=extracted_data, order=pd.value_counts(extracted_data).iloc[:15].index) # Top 15
                    plt.title(f'Distribution of {col_name}')
                    plt.xlabel('Count')
                    plt.ylabel(col_name)
                    plt.tight_layout()
                    plt.show()
                    print(f"Top 5 {col_name}:\n{extracted_data.value_counts().nlargest(5)}")
                else:
                    print(f"No valid data found for metadata field: {col_name}")
            else:
                print(f"'metadata' column not found in DataFrame for {col_name} analysis.")


    def plot_word_clouds(self, text_column='processed_content', top_k=100):
        print(f"\n--- EDA: Word Cloud for {text_column} ---")
        if text_column not in self.df or self.df[text_column].isnull().all():
            print(f"Column {text_column} not suitable for word cloud.")
            return
            
        long_text = " ".join(self.df[text_column].dropna().astype(str))
        if not long_text.strip():
            print("Not enough text data to generate word cloud.")
            return
            
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=top_k).generate(long_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {text_column} (Top {top_k} words)')
        plt.show()

        #save the word cloud image
        wordcloud.to_file(f"{text_column}_wordcloud.png")

    def common_ngram_analysis(self, text_column='processed_content', n=2, top_k=20):
        print(f"\n--- EDA: Top {top_k} {n}-grams from {text_column} ---")
        if text_column not in self.df or self.df[text_column].isnull().all():
            print(f"Column {text_column} not suitable for n-gram analysis.")
            return
            
        all_ngrams = Counter()
        for text in self.df[text_column].dropna().astype(str):
            tokens = word_tokenize(text) # NLTK tokenizer
            if len(tokens) >=n:
                n_grams = nltk.ngrams(tokens, n)
                all_ngrams.update(n_grams)
        
        if not all_ngrams:
            print("No n-grams generated.")
            return

        print(f"Most common {n}-grams:")
        for ngram, count in all_ngrams.most_common(top_k):
            print(f"{' '.join(ngram)}: {count}")

    def identify_common_sections(self):
        COMMON_SECTION_KEYWORDS = {
            "responsibilities": ["responsibilities", "key tasks", "duties & responsibilities", "your role", "the role", "about the role", "key responsibilities"],
            "qualifications_skills": ["skills", "requirements", "qualifications", "criteria", "what you'll bring", "about you", "to succeed", "required skills", "knowledge, skills, and abilities"],
            "experience": ["experience", "years of experience"],
            "education": ["education", "degree"],
            "benefits": ["benefits", "what we offer", "perks"]
        }
        print("\n--- EDA: Common Section Analysis (Heuristic) ---")
        section_counts = {key: Counter() for key_list in COMMON_SECTION_KEYWORDS.values() for key in key_list}
        all_potential_headers = []

        for content in self.df['cleaned_content'].dropna():
            # Using spaCy for sentence segmentation
            doc = self.text_processor.nlp_spacy(content)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            for sent_text in sentences:
                if len(sent_text) > 100 or not sent_text.strip(): continue # Skip very long lines or empty lines as headers
                
                # A simple heuristic: lines with few words, ending with colon, or bolded in HTML
                # More robust would be to inspect HTML tags around these sentences.
                potential_header = sent_text.lower()
                all_potential_headers.append(potential_header)

                for section_type, keywords in COMMON_SECTION_KEYWORDS.items():
                    for keyword in keywords:
                        if keyword in potential_header:
                            section_counts[keyword][potential_header] += 1
                            break # Count first match
        
        print("Overall Potential Header Frequency (Top 20):")
        for header, count in Counter(all_potential_headers).most_common(20):
            print(f"- '{header}': {count}")
            
        print("\nFrequency of Section Keywords (Top 5 variants per keyword):")
        for keyword, variants_counter in section_counts.items():
            if variants_counter: # Only print if keyword was found
                print(f"  Keyword '{keyword}':")
                for variant, count in variants_counter.most_common(5):
                    print(f"    - '{variant}': {count}")

    def run_full_eda(self):
        print("Starting Full Exploratory Data Analysis...")
        self.plot_text_length_distributions()
        self.analyze_metadata_distributions()
        self.plot_word_clouds(text_column='title') # Word cloud for titles
        self.plot_word_clouds(text_column='processed_content') # Word cloud for main content
        self.common_ngram_analysis(text_column='processed_content', n=2, top_k=20)
        self.common_ngram_analysis(text_column='processed_content', n=3, top_k=15)
        self.identify_common_sections()
        print("EDA Complete.")

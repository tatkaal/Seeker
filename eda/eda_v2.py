# eda_v2.py
"""
Advanced Exploratory Data Analysis for a 50k-job-ad corpus
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

from core.html_parser import HTMLParser
from core.text_processor import TextProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EDAConfig:
    """
    Runtime configuration flags – adjust to trade speed vs. depth.
    """
    n_jobs: int = max(os.cpu_count() - 1, 1)
    sample_size_for_heavy: Optional[int] = None  # None => use full dataset
    output_dir: Path = Path("eda_outputs_gpu")
    min_df_tfidf: int = 5
    n_top_terms: int = 25
    n_topics: int = 12
    embedding_mode: str = "spacy"  # or "sbert"
    random_state: int = 42

# ---------------------------------------------------------------------------
# Helper patterns and functions
# ---------------------------------------------------------------------------
SECTION_PATTERNS = {
    "responsibilities": re.compile(r"responsibil|duties|key (tasks|responsibilities)", re.I),
    "skills": re.compile(r"skills|requirements|criteria|abilities", re.I),
    "experience": re.compile(r"\bexperience\b", re.I),
    "education": re.compile(r"education|degree", re.I),
    "benefits": re.compile(r"benefits|perks|what we offer", re.I),
}
BULLET_RE = re.compile(r"^[*\u2022\-]\s+(.*)")

def _spacy_pool(docs):
    """
    Flatten a sequence of spaCy Doc objects into lemmatized token strings.
    """
    for doc in docs:
        yield " ".join(
            t.lemma_ for t in doc
            if not (t.is_stop or t.is_punct or t.is_space)
        )

# ---------------------------------------------------------------------------
# Main EDA class
# ---------------------------------------------------------------------------
class JobAdEDAAdvanced:
    def __init__(self, df: pd.DataFrame, cfg: EDAConfig | None = None):
        self.cfg = cfg or EDAConfig()
        self.df_raw = df.copy()

        missing = {c for c in ["id","title","abstract","content","metadata"] if c not in self.df_raw}
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

        self.tp = TextProcessor()
        self.nlp = self.tp.nlp_spacy

        self._clean_and_tokenise()

    # -----------------------------------------------------------------------
    # Pre-processing
    # -----------------------------------------------------------------------
    def _clean_and_tokenise(self):
        print("Cleaning HTML and tokenising with spaCy …")
        content_clean = self.df_raw["content"].astype(str).map(HTMLParser.clean_html)

        cache_file = self.cfg.output_dir / "cleaned_content.parquet"
        if cache_file.exists():
            cleaned = pd.read_parquet(cache_file)["cleaned"]
            if len(cleaned) == len(content_clean):
                content_clean = cleaned
        else:
            content_clean.to_frame("cleaned")\
                         .to_parquet(cache_file, compression="snappy")

        self.df = self.df_raw.assign(cleaned_content=content_clean)

        # GPU-enabled spaCy pipe
        print("Creating spaCy docs on GPU …")
        pipe = self.nlp.pipe(
            content_clean.tolist(),
            batch_size=64,
            n_process=1  # GPU is single process
        )
        tokens = list(_spacy_pool(pipe))
        self.df["processed_content"] = tokens

    # -----------------------------------------------------------------------
    # Text length distributions
    # -----------------------------------------------------------------------
    def plot_text_lengths(self):
        for col in ["title","abstract","cleaned_content"]:
            self.df[f"{col}_len"] = self.df[col].astype(str).str.len()
            plt.figure(figsize=(9,4))
            plt.hist(self.df[f"{col}_len"], bins=60, alpha=0.7)
            plt.title(f"{col.title()} length distribution [{len(self.df)} ads]")
            plt.xlabel("Length (chars)"); plt.ylabel("Frequency")
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            path = self.cfg.output_dir / f"length_dist_{col}.png"
            plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
            print(f"Saved → {path}")

    # -----------------------------------------------------------------------
    # Metadata analysis
    # -----------------------------------------------------------------------
    def metadata_overview(self):
        print("Analysing categorical metadata …")
        cats = {k:[] for k in ["classification","subClassification","location","workType"]}
        for meta in self.df["metadata"]:
            if not isinstance(meta,dict): continue
            for k in cats:
                v = meta.get(k)
                if isinstance(v,dict): v = v.get("name")
                if v: cats[k].append(v)
        for k,vals in cats.items():
            if not vals: continue
            vc = pd.Series(vals).value_counts()
            vc.to_csv(self.cfg.output_dir / f"metadata_{k}_freq.csv")
            plt.figure(figsize=(10,6))
            vc.head(20).plot(kind="barh"); plt.title(f"Top {k}")
            plt.gca().invert_yaxis(); plt.tight_layout()
            path = self.cfg.output_dir / f"metadata_{k}_barh.png"
            plt.savefig(path, dpi=120); plt.close()
            print(f"{k}: saved → {path}")

    # -----------------------------------------------------------------------
    # TF-IDF keyword extraction
    # -----------------------------------------------------------------------
    def tfidf_keywords(self, by:Optional[str]=None, top_n:Optional[int]=None):
        top_n = top_n or self.cfg.n_top_terms
        docs = self.df["processed_content"].tolist()
        vec = TfidfVectorizer(min_df=self.cfg.min_df_tfidf)
        X = vec.fit_transform(docs)
        terms = np.array(vec.get_feature_names_out())
        scores = np.asarray(X.mean(axis=0)).ravel()
        idx = scores.argsort()[::-1][:top_n]
        pd.DataFrame(list(zip(terms[idx],scores[idx])),columns=["term","score"])\
          .to_csv(self.cfg.output_dir/"tfidf_top_terms.csv",index=False)
        print("Saved global TF-IDF top terms.")

        if by:
            kv = []
            for m in self.df["metadata"]:
                v = m.get(by) if isinstance(m,dict) else None
                if isinstance(v,dict): v = v.get("name")
                kv.append(v)
            groups=defaultdict(list)
            for i,g in enumerate(kv): groups[g].append(i)
            for g,idxs in groups.items():
                if g is None or len(idxs)<10: continue
                sub = X[idxs]; sc = np.asarray(sub.mean(axis=0)).ravel()
                top= sc.argsort()[::-1][:top_n]
                pd.DataFrame(list(zip(terms[top],sc[top])),columns=["term","score"])\
                  .to_csv(self.cfg.output_dir/f"tfidf_{by}={g}.csv",index=False)
            print(f"Saved per-{by} TF-IDF tables.")

    # -----------------------------------------------------------------------
    # Topic modelling (LDA / NMF)
    # -----------------------------------------------------------------------
    def topic_model(self, n_topics:Optional[int]=None, method:str="lda"):
        n_topics = n_topics or self.cfg.n_topics
        idxs = ( self.df.sample(self.cfg.sample_size_for_heavy,random_state=self.cfg.random_state).index
                 if self.cfg.sample_size_for_heavy else self.df.index )
        docs = self.df.loc[idxs,"processed_content"].tolist()
        vec = TfidfVectorizer(min_df=self.cfg.min_df_tfidf, max_df=0.95)
        X = vec.fit_transform(docs)
        print(f"Fitting {method.upper()} on {X.shape[0]} docs …")
        model = ( LatentDirichletAllocation(n_components=n_topics,learning_method="batch",random_state=self.cfg.random_state)
                  if method=="lda" else
                  NMF(n_components=n_topics,init="nndsvd",random_state=self.cfg.random_state) )
        model.fit(X)
        terms = np.array(vec.get_feature_names_out())
        rows=[]
        for i,comp in enumerate(model.components_):
            top=comp.argsort()[::-1][:10]
            rows.append((i,", ".join(terms[top])))
        pd.DataFrame(rows,columns=["topic","top_terms"])\
          .to_csv(self.cfg.output_dir/f"topics_{method}.csv",index=False)
        print(f"Saved topics_{method}.csv")
        for i,comp in enumerate(model.components_[:6]):
            top=comp.argsort()[::-1][:10]
            plt.figure(figsize=(8,4))
            plt.barh(terms[top][::-1],comp[top][::-1])
            plt.title(f"Topic {i}")
            plt.tight_layout()
            p=self.cfg.output_dir/f"topic_{method}_{i}.png"
            plt.savefig(p,dpi=120); plt.close()
        print("Topic plots saved.")

    # -----------------------------------------------------------------------
    # Skill / requirement extraction
    # -----------------------------------------------------------------------
    def extract_skills_requirements(self):
        bullets=[]
        for txt in self.df["cleaned_content"]:
            for ln in txt.split("\n"):
                m=BULLET_RE.match(ln.strip())
                if m: bullets.append(m.group(1).lower())
        proc=[self.tp.preprocess(b) for b in bullets]
        freq=Counter(proc)
        pd.DataFrame(freq.most_common(100),columns=["skill","count"])\
          .to_csv(self.cfg.output_dir/"skills_top100.csv",index=False)
        wc=WordCloud(800,400,background_color="white").generate_from_frequencies(freq)
        plt.figure(figsize=(10,5)); plt.imshow(wc,interpolation="bilinear"); plt.axis("off")
        plt.title("Skills word-cloud")
        p=self.cfg.output_dir/"skills_wordcloud.png"
        plt.savefig(p,dpi=120); plt.close()
        print("Skills extraction done.")

    # -----------------------------------------------------------------------
    # Clustering with embeddings
    # -----------------------------------------------------------------------
    def cluster_jobs(self,n_clusters:int=15):
        idxs = ( self.df.sample(self.cfg.sample_size_for_heavy,random_state=self.cfg.random_state).index
                 if self.cfg.sample_size_for_heavy else self.df.index )
        docs=self.df.loc[idxs,"processed_content"].tolist()
        print(f"Embedding {len(docs)} docs …")
        if self.cfg.embedding_mode=="spacy":
            vecs=np.vstack([self.nlp(d).vector for d in docs])
        else:
            from sentence_transformers import SentenceTransformer
            model=SentenceTransformer("all-MiniLM-L6-v2",device="cuda")
            vecs=model.encode(docs,batch_size=128,show_progress_bar=True)
        print("Clustering …")
        km=KMeans(n_clusters=n_clusters,random_state=self.cfg.random_state)
        labels=km.fit_predict(vecs)
        self.df.loc[idxs,"cluster"]=labels
        pd.Series(labels).value_counts().sort_index()\
          .to_csv(self.cfg.output_dir/"cluster_sizes.csv")
        print("Clustering done.")

    # -----------------------------------------------------------------------
    # Section heading detection
    # -----------------------------------------------------------------------
    def detect_common_sections(self):
        ctr=Counter()
        for txt in self.df["cleaned_content"]:
            for ln in txt.split("\n"):
                ln=ln.strip()
                if 0<len(ln.split())<=8 and ln.endswith(":"):
                    ctr[ln.rstrip(":").lower()]+=1
        pd.DataFrame(ctr.most_common(30),columns=["header","count"])\
          .to_csv(self.cfg.output_dir/"sections.csv",index=False)
        print("Section headers saved.")

    # -----------------------------------------------------------------------
    # Orchestrator
    # -----------------------------------------------------------------------
    def run_all(self):
        print("=== Advanced Job-Ad EDA start ===")
        self.plot_text_lengths()
        self.metadata_overview()
        self.tfidf_keywords(by="classification")
        self.topic_model(method="lda")
        self.extract_skills_requirements()
        self.cluster_jobs()
        self.detect_common_sections()
        print("EDA outputs in →", self.cfg.output_dir.resolve())

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--file",required=True)
    p.add_argument("--sample",type=int,default=None)
    p.add_argument("--heavy-sample",type=int,default=None)
    p.add_argument("--topics",type=int,default=None)
    args=p.parse_args()

    with open(args.file,"r",encoding="utf-8") as f:
        data=[json.loads(l) for l in f]
    df=pd.json_normalize(data)
    if args.sample: df=df.sample(args.sample,random_state=42)

    cfg=EDAConfig()
    if args.heavy_sample is not None: cfg.sample_size_for_heavy=args.heavy_sample
    if args.topics: cfg.n_topics=args.topics

    eda=JobAdEDAAdvanced(df,cfg)
    eda.run_all()
    sys.exit(0)

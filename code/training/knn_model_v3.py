import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class OrganizationCategorizerV3:
    """
    Data leakage-aware kNN organization categorizer.

    Key differences from v2:
    - Supports organizations without oc_id in training and evaluation
    - Uses group-level split (concept when present, normalized-name fallback when missing)
    - Uses weighted voting per category (Business/Group/Industry) instead of requiring concept map lookup
    - Adds explicit feature handling for common legal/membership words
    """

    def __init__(self, k=3, similarity_threshold=0.05):
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.word_vectorizer = None
        self.char_vectorizer = None
        self.word_train_vectors = None
        self.char_train_vectors = None

        self.train_df = None
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """ Text preprocessing for organization names """
        if pd.isna(text):
            return ""

        text = str(text).lower()

        legal_entities = {
            r'\binc\.?\b': 'incorporated',
            r'\bllc\.?\b': 'limited_liability_company',
            r'\bcorp\.?\b': 'corporation',
            r'\bltd\.?\b': 'limited',
            r'\bco\.?\b': 'company',
            r'\bplc\.?\b': 'public_limited_company',
            r'\bgmbh\.?\b': 'german_limited',
            r'\bsa\.?\b': 'sociedad_anonima',
            r'\bnv\.?\b': 'naamloze_vennootschap',
            r'\bag\.?\b': 'aktiengesellschaft',
            r'\bassoc\.?\b': 'association',
        }

        for pattern, replacement in legal_entities.items():
            text = re.sub(pattern, replacement, text)

        text = re.sub(r'[^a-z0-9\s\-_]', ' ', text)
        tokens = text.split()

        important_words = {
            'group', 'company', 'corporation', 'international',
            'systems', 'services', 'technologies', 'solutions',
            'incorporated', 'limited_liability_company', 'limited',
            'association', 'chapter', 'union'
        }

        tokens = [
            token for token in tokens
            if token not in self.stop_words or token in important_words
        ]

        abbrev_map = {
            'intl': 'international',
            'sys': 'systems',
            'tech': 'technology',
            'svcs': 'services',
            'mfg': 'manufacturing',
            'dist': 'distribution',
            'assoc': 'association',
            'dev': 'development',
        }

        tokens = [abbrev_map.get(token, token) for token in tokens]
        return ' '.join(tokens)

    def extract_features(self, text):
        """ Extract additional symbolic features from organization name """
        if pd.isna(text):
            return ""

        features = []
        text_lower = str(text).lower()

        industry_keywords = {
            'technology': ['tech', 'software', 'systems', 'data', 'digital', 'cyber', 'cloud', 'ai'],
            'healthcare': ['health', 'medical', 'pharma', 'bio', 'hospital', 'clinical', 'care'],
            'financial': ['bank', 'financial', 'capital', 'investment', 'insurance', 'fund'],
            'manufacturing': ['manufacturing', 'industrial', 'production', 'factory', 'materials'],
            'retail': ['retail', 'store', 'shop', 'market', 'consumer'],
            'energy': ['energy', 'power', 'oil', 'gas', 'electric', 'solar', 'utility'],
            'telecom': ['telecom', 'communication', 'wireless', 'network', 'mobile'],
            'education': ['education', 'university', 'college', 'school', 'academy', 'learning'],
            'real_estate': ['real estate', 'property', 'realty', 'housing', 'construction'],
            'transportation': ['transport', 'logistics', 'shipping', 'airline', 'aviation', 'freight'],
        }

        for industry, keywords in industry_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                features.append(f'industry_{industry}')

        if any(term in text_lower for term in ['inc', 'incorporated', 'corp', 'corporation']):
            features.append('type_corporation')
        if any(term in text_lower for term in ['llc', 'limited liability']):
            features.append('type_llc')
        if 'ltd' in text_lower or 'limited' in text_lower:
            features.append('type_limited')

        if any(term in text_lower for term in ['association', 'union', 'chapter']):
            features.append('org_membership_style')
        if 'chapter' in text_lower:
            features.append('org_chapter_structure')
        if 'union' in text_lower:
            features.append('org_union_structure')

        if any(term in text_lower for term in ['international', 'global', 'worldwide']):
            features.append('scope_international')

        return ' '.join(features)

    def fit(self, df, word_ngram=(1, 1), char_ngram = (3, 5), min_conf=0.0):
        """ Train model """
        print("Preprocessing training data...")

        df = df.copy()
        df = df.dropna(subset=['name_org', 'Business', 'Group', 'Industry'])
        df = df[df['Business'] != 'Uncoded']
        df = df[df['Group'] != 'Uncoded']
        df = df[df['Industry'] != 'Uncoded']

        df['processed_name'] = df['name_org'].apply(self.preprocess_text)
        df['features'] = df['name_org'].apply(self.extract_features)
        df['combined_text'] = df['processed_name'] + ' ' + df['features']

        self.train_df = df[['name_org', 'oc_id', 'combined_text', 'Business', 'Group', 'Industry', 'split_group_id']].copy()

        print(f"Training samples: {len(self.train_df)}")
        print(f"Unique split groups: {self.train_df['split_group_id'].nunique()}")

        print("Creating word TF-IDF vectors...")
        self.word_vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer="word",
            ngram_range=word_ngram,
            min_df=1,
            max_df=0.8,
            sublinear_tf=True,
            norm='l2'
        )

        self.word_train_vectors = self.word_vectorizer.fit_transform(self.train_df['combined_text'])

        print("Creating char TF-IDF vectors...")

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram,
            min_df=1,
            max_df=0.8,
            max_features=5000,
            sublinear_tf=True
        )

        self.char_train_vectors = self.char_vectorizer.fit_transform(self.train_df['combined_text'])

        print("Model training complete!")

    @staticmethod
    def _weighted_vote(neighbors, label_col):
        score_map = {}
        for _, row in neighbors.iterrows():
            label = row[label_col]
            score_map[label] = score_map.get(label, 0.0) + row['similarity']

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = ranked[0]
        total_score = sum(score for _, score in ranked)
        confidence = top_score / total_score if total_score > 0 else 0.0
        return top_label, confidence, ranked
    
    def predict_with_vectors(self, query_vector, train_vectors):
        similarities = cosine_similarity(query_vector, train_vectors)[0]

        candidate_idx = np.argsort(similarities)[::-1][:self.k]

        if len(candidate_idx) == 0:
            return None

        neighbor_df = self.train_df.iloc[candidate_idx].copy()
        neighbor_df['similarity'] = [float(similarities[i]) for i in candidate_idx]

        business, b_conf, _ = self._weighted_vote(neighbor_df, 'Business')
        group, g_conf, _ = self._weighted_vote(neighbor_df, 'Group')
        industry, i_conf, _ = self._weighted_vote(neighbor_df, 'Industry')

        return {
            'Business': business,
            'Group': group,
            'Industry': industry,
            'confidence': float(np.mean([b_conf, g_conf, i_conf])),
            'similar_orgs': [
                {
                    'name': row['name_org'],
                    'similarity': float(row['similarity'])
                }
                for _, row in neighbor_df.sort_values(
                    'similarity', ascending=False
                ).head(5).iterrows()
            ]
        }

    def predict(self, org_name, min_conf):
        """ Predict categories using weighted neighbor voting """

        processed_name = self.preprocess_text(org_name)
        features = self.extract_features(org_name)
        combined_text = processed_name + ' ' + features

        query_vector = self.word_vectorizer.transform([combined_text])

        result = self.predict_with_vectors(
            query_vector,
            self.word_train_vectors
        )

        if result['confidence'] <= min_conf:

            query_vector = self.char_vectorizer.transform([combined_text])

            result = self.predict_with_vectors(
                query_vector,
                self.char_train_vectors
            )

            result['model_used'] = "char_ngram"

        else:
            result['model_used'] = "word_ngram"

        return result

    def evaluate(self, test_df, min_conf):
        """ Evaluate model on held-out test data """
        print("\nEvaluating model...")

        predictions = {'Business': [], 'Group': [], 'Industry': []}
        true_labels = {'Business': [], 'Group': [], 'Industry': []}
        confidences = []

        test_df = test_df.reset_index(drop=True)
        for idx, row in test_df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(test_df)} samples...")

            pred = self.predict(row['name_org'], min_conf)

            predictions['Business'].append(pred['Business'])
            predictions['Group'].append(pred['Group'])
            predictions['Industry'].append(pred['Industry'])

            true_labels['Business'].append(row['Business'])
            true_labels['Group'].append(row['Group'])
            true_labels['Industry'].append(row['Industry'])

            confidences.append(pred['confidence'])

        metrics = {}
        for category in ['Business', 'Group', 'Industry']:
            acc = accuracy_score(true_labels[category], predictions[category])
            f1_macro = f1_score(true_labels[category], predictions[category], average='macro', zero_division=0)
            metrics[category] = {'accuracy': acc, 'f1_macro': f1_macro}
            print(f"\n{category} Classification:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Macro F1: {f1_macro:.4f}")

        overall_acc = np.mean([metrics[c]['accuracy'] for c in ['Business', 'Group', 'Industry']])
        metrics['overall_accuracy'] = float(overall_acc)
        metrics['mean_confidence'] = float(np.mean(confidences))

        print(f"\nOverall Accuracy: {overall_acc:.4f}")
        print(f"Mean Confidence: {np.mean(confidences):.4f}")

        return metrics

    def save_model(self, filepath):
        """ Save trained model """
        import os
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        model_data = {
            "word_vectorizer": self.word_vectorizer,
            "char_vectorizer": self.char_vectorizer,
            "word_train_vectors": self.word_train_vectors,
            "char_train_vectors": self.char_train_vectors,
            "train_df": self.train_df,
            "k": self.k,
            "similarity_threshold": self.similarity_threshold,
            "metrics": getattr(self, "metrics", None)
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """ Load trained model """

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.word_vectorizer = model_data["word_vectorizer"]
        self.char_vectorizer = model_data["char_vectorizer"]

        self.word_train_vectors = model_data["word_train_vectors"]
        self.char_train_vectors = model_data["char_train_vectors"]

        self.train_df = model_data["train_df"]
        self.k = model_data["k"]
        self.similarity_threshold = model_data["similarity_threshold"]

        print(f"Model loaded from {filepath}")


def _build_split_group_id(df, preprocessor):
    """
    Build leakage-resistant grouping key:
    - use concept id when available
    - fallback to normalized name when concept id is missing
    """
    df = df.copy()
    df['normalized_name_for_split'] = df['name_org'].apply(preprocessor)

    def resolve_group(row):
        if pd.notna(row['oc_id']):
            return f"concept::{int(row['oc_id'])}"
        return f"name::{row['normalized_name_for_split']}"

    df['split_group_id'] = df.apply(resolve_group, axis=1)
    return df


def leakage_aware_train_test_split(df, preprocessor, test_size=0.20, random_state=42):
    """
    Split by group ids to avoid train/test leakage across near-duplicate names
    Stratify at group-level by Industry
    """
    df = _build_split_group_id(df, preprocessor)

    group_df = df.groupby('split_group_id').first()[['Industry']].reset_index()

    industry_group_counts = group_df['Industry'].value_counts()
    valid_industries = industry_group_counts[industry_group_counts >= 2].index

    group_df = group_df[group_df['Industry'].isin(valid_industries)].copy()
    filtered_df = df[df['Industry'].isin(valid_industries)].copy()

    train_groups, test_groups = train_test_split(
        group_df['split_group_id'],
        test_size=test_size,
        random_state=random_state,
        stratify=group_df['Industry']
    )

    train_df = filtered_df[filtered_df['split_group_id'].isin(train_groups)].copy()
    test_df = filtered_df[filtered_df['split_group_id'].isin(test_groups)].copy()

    return train_df, test_df, len(industry_group_counts) - len(valid_industries)


def analyze_common_word_behavior(df, preprocessor):
    """
    Analysis of high-frequency words that can be weak separators
    """
    target_words = ['incorporated', 'corporation', 'group', 'association', 'chapter', 'union', 'company', 'the']

    names = df['name_org'].fillna('').astype(str).apply(preprocessor)
    total_docs = len(names)

    print("\n" + "=" * 60)
    print("Common Word Behavior Analysis")
    print("=" * 60)

    for word in target_words:
        mask = names.str.contains(rf'\\b{re.escape(word)}\\b', regex=True)
        doc_freq = int(mask.sum())
        pct = (doc_freq / total_docs * 100.0) if total_docs > 0 else 0.0

        subset = df[mask]
        if len(subset) == 0:
            print(f"{word:15s} -> docs: 0 (0.00%), distinct_industries: 0, top_industry_share: 0.00")
            continue

        industry_counts = subset['Industry'].value_counts(normalize=True)
        top_share = float(industry_counts.iloc[0]) if len(industry_counts) > 0 else 0.0

        print(
            f"{word:15s} -> docs: {doc_freq:6d} ({pct:5.2f}%), "
            f"distinct_industries: {subset['Industry'].nunique():3d}, "
            f"top_industry_share: {top_share:5.2f}"
        )


def main(k_val=3, word_gram=(1, 1), char_gram=(3,5), min_conf=0.0):
    print("Loading data...")
    df = pd.read_csv("../data/processed/OrganizationsFull.tsv", sep="\t", index_col=0)

    df = df.dropna(subset=['name_org', 'Business', 'Group', 'Industry'])
    df = df[df['Business'] != 'Uncoded']
    df = df[df['Group'] != 'Uncoded']
    df = df[df['Industry'] != 'Uncoded']

    print(f"Total valid organizations: {len(df)}")
    print(f"Orgs with concept id: {df['oc_id'].notna().sum()}")
    print(f"Orgs without concept id: {df['oc_id'].isna().sum()}")

    model = OrganizationCategorizerV3(k=k_val, similarity_threshold=0.00)

    train_df, test_df, removed_industries = leakage_aware_train_test_split(
        df,
        preprocessor=model.preprocess_text,
        test_size=0.20,
        random_state=42
    )
    print(test_df['name_org'].head(50))

    overlap = set(train_df['split_group_id']) & set(test_df['split_group_id'])

    print(f"After filtering rare industries by group count: {len(train_df) + len(test_df)} organizations")
    print(f"Industries removed: {removed_industries}")
    print(f"Training set: {len(train_df)} organizations, {train_df['split_group_id'].nunique()} groups")
    print(f"Test set: {len(test_df)} organizations, {test_df['split_group_id'].nunique()} groups")
    print(f"Leakage check - group overlap: {len(overlap)}")

    analyze_common_word_behavior(train_df, model.preprocess_text)

    print("\n" + "=" * 60)
    print(f"Training model with k={k_val}...")
    print("=" * 60)

    model.fit(train_df, word_gram, char_gram, min_conf)
    metrics = model.evaluate(test_df, min_conf)

    model.metrics = metrics
    model.save_model("../model/org_categorizer_knn_model_v3.pkl")

    print("\n" + "=" * 60)
    print("Test Example Predictions:")
    print("=" * 60)

    test_names = [
        "Apple Inc",
        "General Electric",
        "Microsoft Corporation"
    ]

    for name in test_names:
        pred = model.predict(name, min_conf)
        print(f"\nOrganization: {name}")
        print(f"  Business: {pred['Business']}")
        print(f"  Group: {pred['Group']}")
        print(f"  Industry: {pred['Industry']}")
        print(f"  Confidence: {pred['confidence']:.3f}")
        print(f"  Similar organizations: {[org['name'] for org in pred['similar_orgs'][:3]]}")

    return metrics


if __name__ == "__main__":
    main(3, (1,1), (3,5), 0.47)

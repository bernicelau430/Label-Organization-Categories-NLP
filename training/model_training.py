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

class OrganizationCategorizer:
    """
    kNN-based organization categorizer using concept-level prediction
    with text preprocessing and feature engineering
    """
    
    def __init__(self, k=5, similarity_threshold=0.1):
        """
        Initialize the categorizer
           k: Number of nearest neighbors to consider
           similarity_threshold: Minimum similarity score to consider a neighbor
        """
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.vectorizer = None
        self.train_vectors = None
        self.train_df = None
        self.concept_map = None
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Text preprocessing for organization names
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # replace common legal entity types with standardized tokens
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
            r'\bltd\.?\b': 'limited',
        }
        
        for pattern, replacement in legal_entities.items():
            text = re.sub(pattern, replacement, text)
        
        # remove special characters but keep spaces, hyphens, and underscores
        text = re.sub(r'[^a-z0-9\s\-_]', ' ', text)
        
        # tokenize
        tokens = text.split()
        
        # remove stopwords but keep important business words
        important_words = {'group', 'company', 'corporation', 'international', 
                          'systems', 'services', 'technologies', 'solutions',
                          'incorporated', 'limited_liability_company', 'limited'}
        
        tokens = [token for token in tokens 
                 if token not in self.stop_words or token in important_words]
        
        # handle common abbreviations
        abbrev_map = {
            'intl': 'international',
            'sys': 'systems',
            'tech': 'technology',
            'svcs': 'services',
            'mfg': 'manufacturing',
            'dist': 'distribution',
            'assoc': 'associates',
            'dev': 'development',
        }
        
        tokens = [abbrev_map.get(token, token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_features(self, text):
        """
        Extract additional features from organization name and
        return a string that can be concatenated to the preprocessed text
        """
        if pd.isna(text):
            return ""
        
        features = []
        text_lower = str(text).lower()
        
        # detect industry keywords
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
        
        # detect organization type
        if any(term in text_lower for term in ['inc', 'incorporated', 'corp', 'corporation']):
            features.append('type_corporation')
        if any(term in text_lower for term in ['llc', 'limited liability']):
            features.append('type_llc')
        if 'ltd' in text_lower or 'limited' in text_lower:
            features.append('type_limited')
        
        # detect geographic indicators
        if 'international' in text_lower or 'global' in text_lower or 'worldwide' in text_lower:
            features.append('scope_international')
        
        return ' '.join(features)
    
    def fit(self, df):
        """
        Train the model on the provided dataset
           df columns: ['name_org', 'oc_id', 'Business', 'Group', 'Industry']
        """
        print("Preprocessing training data...")
        
        # clean and filter data
        df = df.copy()
        df = df.dropna(subset=['name_org', 'oc_id'])
        df = df[df['Business'] != 'Uncoded']
        df = df[df['Group'] != 'Uncoded']
        df = df[df['Industry'] != 'Uncoded']
        
        # preprocess organization names
        df['processed_name'] = df['name_org'].apply(self.preprocess_text)
        df['features'] = df['name_org'].apply(self.extract_features)
        df['combined_text'] = df['processed_name'] + ' ' + df['features']
        
        # create concept mapping
        self.concept_map = df.groupby('oc_id').first()[['Business', 'Group', 'Industry']].to_dict('index')
        
        # store training data
        self.train_df = df[['name_org', 'oc_id', 'combined_text', 'Business', 'Group', 'Industry']].copy()
        
        print(f"Training samples: {len(self.train_df)}")
        print(f"Unique concepts: {len(self.concept_map)}")
        
        # create TF-IDF vectors with optimized parameters
        print("Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # increased from default
            ngram_range=(1, 3),  # unigrams, bigrams, and trigrams
            min_df=2,  # ignore terms that appear in fewer than 2 documents
            max_df=0.8,  # ignore terms that appear in more than 80% of documents
            sublinear_tf=True,  # use logarithmic term frequency
            norm='l2'  # L2 normalization
        )
        
        self.train_vectors = self.vectorizer.fit_transform(self.train_df['combined_text'])
        
        print("Model training complete!")
        
    def predict(self, org_name, return_alternatives=False):
        """
        Predict categories for a given organization name and 
        return a dictionary with predicted categories and confidence scores
           org_name: Name of the organization
           return_alternatives: If True, return top alternative predictions
        """
        # preprocess input
        processed_name = self.preprocess_text(org_name)
        features = self.extract_features(org_name)
        combined_text = processed_name + ' ' + features
        
        # vectorize input
        query_vector = self.vectorizer.transform([combined_text])
        
        # calculate similarities
        similarities = cosine_similarity(query_vector, self.train_vectors)[0]
        
        # get top k neighbors above threshold
        top_k_indices = np.argsort(similarities)[::-1][:self.k * 3]  # Get more candidates
        top_k_indices = [i for i in top_k_indices if similarities[i] >= self.similarity_threshold][:self.k]
        
        if len(top_k_indices) == 0:
            # no similar neighbors found, return most common concept
            most_common_concept = self.train_df['oc_id'].mode()[0]
            predicted_categories = self.concept_map[most_common_concept]
            return {
                'Business': predicted_categories['Business'],
                'Group': predicted_categories['Group'],
                'Industry': predicted_categories['Industry'],
                'confidence': 0.0,
                'similar_orgs': []
            }
        
        # weighted voting by concept
        concept_scores = {}
        neighbor_details = []
        
        for idx in top_k_indices:
            concept = self.train_df.iloc[idx]['oc_id']
            similarity = similarities[idx]
            
            if concept not in concept_scores:
                concept_scores[concept] = 0
            concept_scores[concept] += similarity
            
            neighbor_details.append({
                'name': self.train_df.iloc[idx]['name_org'],
                'similarity': float(similarity),
                'concept': int(concept)
            })
        
        # sort concepts by weighted score
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        
        # get predicted concept
        predicted_concept = sorted_concepts[0][0]
        total_score = sum(score for _, score in sorted_concepts)
        confidence = sorted_concepts[0][1] / total_score if total_score > 0 else 0
        
        # get categories from concept
        predicted_categories = self.concept_map[predicted_concept]
        
        result = {
            'Business': predicted_categories['Business'],
            'Group': predicted_categories['Group'],
            'Industry': predicted_categories['Industry'],
            'confidence': float(confidence),
            'similar_orgs': neighbor_details[:5]  # Top 5 similar organizations
        }
        
        if return_alternatives:
            result['alternatives'] = []
            for concept, score in sorted_concepts[1:4]:  # Top 3 alternatives
                alt_categories = self.concept_map[concept]
                result['alternatives'].append({
                    'Business': alt_categories['Business'],
                    'Group': alt_categories['Group'],
                    'Industry': alt_categories['Industry'],
                    'score': float(score / total_score)
                })
        
        return result
    
    def evaluate(self, test_df):
        """
        Evaluate the model on test data and
        return a dictionary with evaluation metrics
           test_df columns: ['name_org', 'Business', 'Group', 'Industry']
        """
        print("\nEvaluating model...")
        
        predictions = {
            'Business': [],
            'Group': [],
            'Industry': [],
            'oc_id': []
        }
        
        true_labels = {
            'Business': [],
            'Group': [],
            'Industry': [],
            'oc_id': []
        }
        
        confidences = []
        
        for idx, row in test_df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(test_df)} samples...")
            
            pred = self.predict(row['name_org'])
            
            predictions['Business'].append(pred['Business'])
            predictions['Group'].append(pred['Group'])
            predictions['Industry'].append(pred['Industry'])
            
            true_labels['Business'].append(row['Business'])
            true_labels['Group'].append(row['Group'])
            true_labels['Industry'].append(row['Industry'])
            
            confidences.append(pred['confidence'])
        
        # calculate metrics
        metrics = {}
        
        for category in ['Business', 'Group', 'Industry']:
            acc = accuracy_score(true_labels[category], predictions[category])
            f1_macro = f1_score(true_labels[category], predictions[category], average='macro', zero_division=0)
            
            metrics[category] = {
                'accuracy': acc,
                'f1_macro': f1_macro
            }
            
            print(f"\n{category} Classification:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Macro F1: {f1_macro:.4f}")
        
        # Overall metrics
        overall_acc = np.mean([metrics[cat]['accuracy'] for cat in ['Business', 'Group', 'Industry']])
        metrics['overall_accuracy'] = overall_acc
        metrics['mean_confidence'] = np.mean(confidences)
        
        print(f"\nOverall Accuracy: {overall_acc:.4f}")
        print(f"Mean Confidence: {np.mean(confidences):.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """ Save the trained model """
        # create model directory if it doesn't exist
        import os
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        model_data = {
            'vectorizer': self.vectorizer,
            'train_vectors': self.train_vectors,
            'train_df': self.train_df,
            'concept_map': self.concept_map,
            'k': self.k,
            'similarity_threshold': self.similarity_threshold
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """ Load a trained model """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.train_vectors = model_data['train_vectors']
        self.train_df = model_data['train_df']
        self.concept_map = model_data['concept_map']
        self.k = model_data['k']
        self.similarity_threshold = model_data['similarity_threshold']
        print(f"Model loaded from {filepath}")


def main():
    # load processed data
    print("Loading data...")
    df = pd.read_csv("../data/processed/OrganizationsFull.tsv", sep="\t", index_col=0)
    
    # filter out invalid/uncoded entries
    df = df.dropna(subset=['name_org', 'oc_id', 'Business', 'Group', 'Industry'])
    df = df[df['Business'] != 'Uncoded']
    df = df[df['Group'] != 'Uncoded']
    df = df[df['Industry'] != 'Uncoded']
    
    print(f"Total valid organizations: {len(df)}")
    print(f"Unique concepts: {df['oc_id'].nunique()}")
    print(f"Unique Business categories: {df['Business'].nunique()}")
    print(f"Unique Group categories: {df['Group'].nunique()}")
    print(f"Unique Industry categories: {df['Industry'].nunique()}")
    
    # remove industries with too few samples for stratified splitting
    industry_counts = df['Industry'].value_counts()
    valid_industries = industry_counts[industry_counts >= 2].index
    df_filtered = df[df['Industry'].isin(valid_industries)]
    
    print(f"After filtering rare categories: {len(df_filtered)} organizations")
    print(f"Industries removed: {len(industry_counts) - len(valid_industries)}")
    
    # split data with stratification (80-20 split)
    train_df, test_df = train_test_split(df_filtered, test_size=0.20, random_state=42, stratify=df_filtered['Industry'])
    
    print(f"\nTraining set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # train model with arbitrary k value
    print("\n" + "="*50)
    print("Training model with k=7...")
    print("="*50)
    
    model = OrganizationCategorizer(k=7, similarity_threshold=0.05)
    model.fit(train_df)
    
    # evaluate
    metrics = model.evaluate(test_df)
    
    # save model
    model.save_model("../model/org_categorizer_model.pkl")
    
    # test on some examples
    print("\n" + "="*50)
    print("Test Example Predictions:")
    print("="*50)
    
    test_names = [
        "Apple Inc",
        "General Electric",
        "Microsoft Corporation"
    ]
    
    for name in test_names:
        pred = model.predict(name, return_alternatives=True)
        print(f"\nOrganization: {name}")
        print(f"  Business: {pred['Business']}")
        print(f"  Group: {pred['Group']}")
        print(f"  Industry: {pred['Industry']}")
        print(f"  Similar organizations: {[org['name'] for org in pred['similar_orgs'][:3]]}")


if __name__ == "__main__":
    main()
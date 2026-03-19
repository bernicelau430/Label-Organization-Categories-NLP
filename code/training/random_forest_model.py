import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
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

class RandomForestCategorizer:
    """
    Machine Learning based organization categorizer using Random Forest
    Predicts Business, Group, and Industry categories directly from text features
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the categorizer
           n_estimators: number of trees for random forest
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.vectorizer = None
        self.models = {}  # separate model for each category
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """ Text preprocessing for organization names """
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
        }
        
        for pattern, replacement in legal_entities.items():
            text = re.sub(pattern, replacement, text)
        
        text = re.sub(r'[^a-z0-9\s\-_]', ' ', text)
        tokens = text.split()
        
        important_words = {'group', 'company', 'corporation', 'international', 
                          'systems', 'services', 'technologies', 'solutions',
                          'incorporated', 'limited_liability_company', 'limited'}
        
        tokens = [token for token in tokens 
                 if token not in self.stop_words or token in important_words]
        
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
        """ Extract additional features """
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
        
        if 'international' in text_lower or 'global' in text_lower or 'worldwide' in text_lower:
            features.append('scope_international')
        
        return ' '.join(features)
    
    def fit(self, df, analyzer="word", ngram_range=(1, 3), min_df=2, max_df=0.8, sublinear_tf=True):
        """ Train the model """
        print("Preprocessing training data for Random Forest model...")
        
        df = df.copy()
        df = df.dropna(subset=['name_org', 'Business', 'Group', 'Industry'])
        df = df[df['Business'] != 'Uncoded']
        df = df[df['Group'] != 'Uncoded']
        df = df[df['Industry'] != 'Uncoded']
        
        df['processed_name'] = df['name_org'].apply(self.preprocess_text)
        df['features'] = df['name_org'].apply(self.extract_features)
        df['combined_text'] = df['processed_name'] + ' ' + df['features']
        
        print(f"Training samples: {len(df)}")
        
        # create TF-IDF vectors
        print("Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm='l2'
        )
        
        X = self.vectorizer.fit_transform(df['combined_text'])
        
        # train separate model for each category
        categories = ['Business', 'Group', 'Industry']
        
        for category in categories:
            print(f"Training Random Forest model for {category}...")
            y = df[category]
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            
            model.fit(X, y)
            self.models[category] = model
        
        print("Random Forest model training complete!")
    
    def predict(self, org_name):
        """ Predict categories for a given organization name """
        processed_name = self.preprocess_text(org_name)
        features = self.extract_features(org_name)
        combined_text = processed_name + ' ' + features
        
        query_vector = self.vectorizer.transform([combined_text])
        
        predictions = {}
        for category, model in self.models.items():
            pred = model.predict(query_vector)[0]
            predictions[category] = pred
            
            # get confidence if available (probability)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(query_vector)[0]
                predictions[f'{category}_confidence'] = float(max(proba))
        
        return predictions
    
    def evaluate(self, test_df):
        """ Evaluate the model on test data """
        print("\nEvaluating Random Forest model...")
        
        predictions = {'Business': [], 'Group': [], 'Industry': []}
        true_labels = {'Business': [], 'Group': [], 'Industry': []}
        
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
        
        overall_acc = np.mean([metrics[cat]['accuracy'] for cat in ['Business', 'Group', 'Industry']])
        metrics['overall_accuracy'] = overall_acc
        
        print(f"\nOverall Accuracy: {overall_acc:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """ Save the trained model """
        import os
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """ Load a trained model """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.models = model_data['models']
        self.n_estimators = model_data['n_estimators']
        self.random_state = model_data.get('random_state', 42)
        print(f"Model loaded from {filepath}")


def main(analyzer="word", ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True):
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
    
    # filter industries that have at least 2 concepts (needed for stratified split by concept)
    concept_df = df.groupby('oc_id').first()[['Industry']].reset_index()
    industry_concept_counts = concept_df.groupby('Industry')['oc_id'].count()
    valid_industries = industry_concept_counts[industry_concept_counts >= 2].index
    
    # filter both concept_df and df to only include valid industries
    concept_df = concept_df[concept_df['Industry'].isin(valid_industries)]
    df_filtered = df[df['Industry'].isin(valid_industries)]
    
    print(f"After filtering industries with <2 concepts: {len(df_filtered)} organizations")
    print(f"Valid concepts: {len(concept_df)}")
    print(f"Industries removed: {len(industry_concept_counts) - len(valid_industries)}")
    
    # split concepts (80-20 split) with stratification by Industry to avoid data leakage
    train_concepts, test_concepts = train_test_split(
        concept_df['oc_id'], 
        test_size=0.20, 
        random_state=42, 
        stratify=concept_df['Industry']
    )
    
    # filter organizations by concept splits
    train_df = df_filtered[df_filtered['oc_id'].isin(train_concepts)].copy()
    test_df = df_filtered[df_filtered['oc_id'].isin(test_concepts)].copy()
    
    print(f"\nTraining set: {len(train_df)} organizations, {len(train_concepts)} concepts")
    print(f"Test set: {len(test_df)} organizations, {len(test_concepts)} concepts")
    print(f"Verify no concept overlap: {len(set(train_concepts) & set(test_concepts)) == 0}")
    
    # Train Random Forest Model
    print("\n" + "="*70)
    print("Training Random Forest Model")
    print("="*70)
    
    rf_model = RandomForestCategorizer(n_estimators=100, random_state=42)
    rf_model.fit(train_df, analyzer, ngram_range, min_df, max_df, sublinear_tf)
    rf_metrics = rf_model.evaluate(test_df)
    rf_model.save_model("../model/org_categorizer_randomforest_model.pkl")
    
    # test on examples
    print("\n" + "="*70)
    print("Example Predictions:")
    print("="*70)
    
    test_names = [
        "Apple Inc",
        "Goldman Sachs",
        "General Electric",
        "Microsoft Corporation"
    ]
    
    for name in test_names:
        pred = rf_model.predict(name)
        print(f"\nOrganization: {name}")
        print(f"  Business: {pred['Business']}")
        print(f"  Group: {pred['Group']}")
        print(f"  Industry: {pred['Industry']}")
        if 'Business_confidence' in pred:
            print(f"  Confidence: Business={pred['Business_confidence']:.3f}, Group={pred.get('Group_confidence', 0):.3f}, Industry={pred.get('Industry_confidence', 0):.3f}")
    
    return rf_metrics


if __name__ == "__main__":
    main(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
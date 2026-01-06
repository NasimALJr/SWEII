import pytest
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re

# Import functions from main.py
from main import preprocess_text, preprocess_query, get_doc_vector, recommend_materials_with_params

# Download required NLTK data for tests
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing"""
    data = {
        'course_code': ['CSE 1100', 'CSE 2102', 'CSE 2103'],
        'course_name': ['Introduction to Programming', 'Data Structures', 'Algorithms'],
        'material_type': ['slide', 'book', 'lab manual'],
        'material_title': ['Intro Slides', 'DS Book', 'Algo Lab Manual'],
        'material_link': ['link1.com', 'link2.com', 'link3.com'],
        'description': ['Basic programming slides', 'Data structures textbook', 'Algorithm lab exercises']
    }
    df = pd.DataFrame(data)
    df["combined_text"] = df.apply(
        lambda row: " ".join(str(x) for x in row.values if pd.notnull(x)),
        axis=1
    )
    return df

@pytest.fixture
def sample_model(sample_df):
    """Create a sample Doc2Vec model for testing"""
    tokenized_texts = [preprocess_text(text).split() for text in sample_df["combined_text"]]
    tagged_data = [TaggedDocument(words=text, tags=[str(i)]) for i, text in enumerate(tokenized_texts)]
    model = Doc2Vec(tagged_data, vector_size=50, window=3, min_count=1, workers=1, epochs=10)
    return model

@pytest.fixture
def sample_doc_vectors(sample_df, sample_model):
    """Create sample document vectors"""
    return np.array([sample_model.dv[str(i)] for i in range(len(sample_df))])

class TestPreprocessing:
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        text = "The quick brown fox jumps over the lazy dog"
        result = preprocess_text(text)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be lowercase and lemmatized
        assert "quick" in result or "jump" in result

    def test_preprocess_text_synonyms(self):
        """Test that synonyms are expanded"""
        text = "run"
        result = preprocess_text(text)
        # Should include synonyms like "running", "sprint", etc.
        assert "run" in result

    def test_preprocess_query(self):
        """Test query preprocessing"""
        query = "Find programming slides"
        result = preprocess_query(query)
        assert isinstance(result, str)
        assert "programming" in result
        assert "slide" in result

class TestDoc2Vec:
    def test_get_doc_vector(self, sample_model):
        """Test getting document vector for a query"""
        text = "programming algorithms"
        vector = get_doc_vector(text, sample_model)
        assert isinstance(vector, np.ndarray)
        assert vector.shape[0] == 50  # vector_size

class TestRecommendation:
    def test_recommend_materials_with_course_code(self, sample_df, sample_model, sample_doc_vectors):
        """Test recommendation with course code"""
        query = "CSE 1100 slides"
        result = recommend_materials_with_params(query, sample_df, sample_model, sample_doc_vectors)
        assert isinstance(result, str)
        assert "CSE 1100" in result or "Found Material" in result

    def test_recommend_materials_no_match(self, sample_df, sample_model, sample_doc_vectors):
        """Test recommendation when no materials match"""
        query = "nonexistent course"
        result = recommend_materials_with_params(query, sample_df, sample_model, sample_doc_vectors)
        assert "Sorry" in result or "couldn't find" in result

    def test_recommend_materials_with_material_type(self, sample_df, sample_model, sample_doc_vectors):
        """Test recommendation with specific material type"""
        query = "book for data structures"
        result = recommend_materials_with_params(query, sample_df, sample_model, sample_doc_vectors)
        assert isinstance(result, str)

class TestIntegration:
    def test_full_workflow(self, sample_df, sample_model, sample_doc_vectors):
        """Test the full recommendation workflow"""
        queries = [
            "CSE 1100 slides",
            "book",
            "lab manual"
        ]

        for query in queries:
            result = recommend_materials_with_params(query, sample_df, sample_model, sample_doc_vectors)
            assert isinstance(result, str)
            assert len(result) > 0

if __name__ == "__main__":
    pytest.main([__file__])
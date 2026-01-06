
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_sm")


lemmatizer = WordNetLemmatizer()

df = pd.read_csv("course_material.csv")


df["combined_text"] = df.apply(
    lambda row: " ".join(str(x) for x in row.values if pd.notnull(x)),
    axis=1
)

material_keywords = ["lab manual", "book", "slide", "note", "video", "manual", "sheet", "handout"]

material_synonyms = {
    "slide": ["slides", "class slides", "presentation", "ppt", "powerpoint"],
    "book": ["books", "textbook", "textbooks"],
    "lab manual": ["lab manuals", "manual", "lab guide", "lab sheet"],
    "note": ["notes", "handout", "handouts", "sheet", "lecture notes"],
    "video": ["videos", "video tutorial", "tutorials", "lecture video"]
}

def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    expanded = set(lemmas)
    for word in lemmas:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    
    return " ".join(expanded)

def preprocess_query(text):
    text = text.lower()
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(lemmas)

# Tokenize texts for Word2Vec
tokenized_texts = [preprocess_text(text).split() for text in df["combined_text"]]

# Prepare tagged data for Doc2Vec
tagged_data = [TaggedDocument(words=text, tags=[str(i)]) for i, text in enumerate(tokenized_texts)]

# Train Doc2Vec model
model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=1, workers=4)

# Precompute document vectors
doc_vectors = np.array([model.dv[str(i)] for i in range(len(df))])

# Function to get document vector for query
def get_doc_vector(text, model):
    words = text.split()
    return model.infer_vector(words)


def recommend_materials_wrapper(query, top_n=5, threshold=0.5):
    """Wrapper function that uses the global variables"""
    return recommend_materials_with_params(query, df, model, doc_vectors, top_n, threshold)


def recommend_materials_with_params(query, df, model, doc_vectors, top_n=5, threshold=0.5):
    
    # Extract material type from query
    query_lower = query.lower()
    requested_material_type = None
    
    for standard, synonyms in material_synonyms.items():
        for syn in synonyms + [standard]:
            if syn in query_lower:
                requested_material_type = standard
                break
        if requested_material_type:
            break
    
    # If course code is mentioned, filter by course code
    match = re.search(r'\b[A-Z]{3}[- ]?\d{4}\b', query)
    if match:
        course_code = match.group(0).replace(" ", "").replace("-", "")
        filtered_df = df[df["course_code"].str.replace(" ", "") == course_code]
        if filtered_df.empty:
            if requested_material_type:
                return "No " + requested_material_type + " found for course " + match.group(0) + "."
            else:
                return "No materials found for course " + match.group(0) + "."
        filtered_doc_vectors = doc_vectors[filtered_df.index]
    else:
        # Check for course name in query
        for course_name in df["course_name"].unique():
            if course_name.lower() in query_lower:
                filtered_df = df[df["course_name"] == course_name]
                if not filtered_df.empty:
                    filtered_doc_vectors = doc_vectors[filtered_df.index]
                break
    
    if 'filtered_df' not in locals():
        filtered_df = df
        filtered_doc_vectors = doc_vectors
    
    # If material type is specified in query, filter by material type
    if requested_material_type:
        temp_filtered = filtered_df[filtered_df["material_type"].str.lower().str.contains(requested_material_type, na=False)]
        if temp_filtered.empty:
            return "No " + requested_material_type + " found for this course."
        else:
            filtered_df = temp_filtered
            filtered_doc_vectors = doc_vectors[filtered_df.index]

    processed_query = preprocess_text(query)
    query_vec = get_doc_vector(processed_query, model)
    similarity = cosine_similarity([query_vec], filtered_doc_vectors).flatten()

    results = []

    for i, idx in enumerate(filtered_df.index):
        score = similarity[i]

        # Boost score if material type matches
        if requested_material_type:
            if requested_material_type in df.loc[idx, "material_type"].lower():
                score += 0.3

        if score > threshold:
            result_text = (
                f"**Found Material**\n\n"
                f"**Course:** {df.loc[idx, 'course_name']} ({df.loc[idx, 'course_code']})\n\n"
                f"**Type:** {df.loc[idx, 'material_type']}\n\n"
                f"**Title:** {df.loc[idx, 'material_title']}\n\n"
                f"**Link:** [{df.loc[idx, 'material_link']}]({df.loc[idx, 'material_link']})\n\n"
                f"**Description:** {df.loc[idx, 'description']}\n\n"
                f"---\n\n"
            )
            results.append((score, result_text))

    results = sorted(results, key=lambda x: x[0], reverse=True)

    # Fallback: show materials for that course if specific not found
    if not results and match:
        fallback = df[df["course_code"].str.replace(" ", "") == course_code]
        if not fallback.empty:
            if requested_material_type:
                fallback = fallback[fallback["material_type"].str.lower().str.contains(requested_material_type, na=False)]
                if fallback.empty:
                    return "No " + requested_material_type + " found for this course."
            return "\n".join([
                f"**Found Material**\n\n"
                f"**Course:** {row['course_name']} ({row['course_code']})\n\n"
                f"**Type:** {row['material_type']}\n\n"
                f"**Title:** {row['material_title']}\n\n"
                f"**Link:** [{row['material_link']}]({row['material_link']})\n\n"
                f"**Description:** {row['description']}\n\n"
                f"---\n\n"
                for _, row in fallback.iterrows()
            ])

    # Nothing at all found
    if not results:
        return "Sorry, I couldn't find any relevant material for your query."

    return "\n".join([r[1] for r in results[:top_n]])

   
    if not results:
        return "Sorry, I couldn't find any relevant material for your query."

    return "\n".join([r[1] for r in results[:top_n]])


queries = [
    "CSE 2103 lab manual",
    "CSE 2102 book",
    "slides for CSE 1100",
    "CSE 3105 manual",
    "CSE 3201 book"
]

if __name__ == "__main__":
    for q in queries:
        print(f"\nQuery: {q}")
        print(recommend_materials_wrapper(q)) 

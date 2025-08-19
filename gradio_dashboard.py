import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents,
                                 embedding=embeddings,)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    print(f"=== DEBUG: Query = '{query}' ===")

    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    print(f"Found {len(recs)} similarity results")

    books_list = []
    for i, (rec, score) in enumerate(recs[:5]):  # Show first 5 for debugging
        content = rec.page_content
        print(f"\nResult {i}:")
        print(f"  Score: {score:.4f}")
        print(f"  Raw content (first 100 chars): {content[:100]}")

        # Extract ISBN
        try:
            stripped = content.strip('"')
            first_word = stripped.split()[0]
            isbn = int(first_word)
            books_list.append(isbn)
            print(f"  Extracted ISBN: {isbn}")
        except (ValueError, IndexError) as e:
            print(f"  ERROR extracting ISBN: {e}")
            print(f"  First word was: '{stripped.split()[0] if stripped.split() else 'EMPTY'}'")

    print(f"\nTotal valid ISBNs extracted: {len(books_list)}")
    print(f"ISBNs: {books_list[:10]}")  # Show first 10

    # Check if ISBNs exist in the books dataframe
    book_recs = books[books["isbn13"].isin(books_list)]
    print(f"Books found in dataframe: {len(book_recs)}")

    if len(book_recs) == 0:
        print("WARNING: No books found! Checking ISBN format...")
        print(f"Sample ISBNs from books dataframe: {books['isbn13'].head().tolist()}")
        print(f"Sample extracted ISBNs: {books_list[:5]}")

    book_recs = book_recs.head(initial_top_k)

    # Apply category filter
    if category and category != "All":
        print(f"Filtering by category: {category}")
        before_filter = len(book_recs)
        book_recs = book_recs[book_recs["simple_categories"] == category]
        print(f"After category filter: {len(book_recs)} (was {before_filter})")
        book_recs = book_recs.head(final_top_k)

    # Apply tone sorting
    if tone and tone != "All":
        print(f"Sorting by tone: {tone}")
        if tone == "Happy":
            book_recs = book_recs.sort_values(by="joy", ascending=False)
        elif tone == "Surprising":
            book_recs = book_recs.sort_values(by="surprise", ascending=False)
        elif tone == "Angry":
            book_recs = book_recs.sort_values(by="anger", ascending=False)
        elif tone == "Suspenseful":
            book_recs = book_recs.sort_values(by="fear", ascending=False)
        elif tone == "Sad":
            book_recs = book_recs.sort_values(by="sadness", ascending=False)

    print(f"Final recommendations: {len(book_recs)}")
    if len(book_recs) > 0:
        print(f"First recommendation: {book_recs.iloc[0]['title']}")
    print("=== END DEBUG ===\n")

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    print(f"GRADIO DEBUG: Received query='{query}', category='{category}', tone='{tone}'")

    # Handle empty query
    if not query or query.strip() == '':
        print("WARNING: Empty query received!")
        return []

    recommendations = retrieve_semantic_recommendations(query.strip(), category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    print(f"GRADIO DEBUG: Returning {len(results)} results")
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()
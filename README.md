#  Book Recommender System

A data science portfolio project built to demonstrate **text analysis, recommendation algorithms, and interactive visualization** workflows.

This repository features a set of Jupyter notebooks exploring book datasets, text classification, sentiment tagging, vector search, and an interactive Gradio demo for recommending books based on descriptions and emotions.

---

##  Project Overview & Intent

I developed this Book Recommender System to showcase my skills in:
- **Text preprocessing & sentiment analysis**
- **Machine learning for classification and similarity-based recommendation**
- **Interactive UI development via Gradio**
- **End-to-end project structure—from raw data to deployable demo**

---

##  Repository Structure

book-recommender/
├── data/
│ ├── books_cleaned.csv
│ ├── books_with_categories.csv
│ └── books_with_emotions.csv
├── notebooks/
│ ├── data-exploration.ipynb
│ ├── sentiment-analysis.ipynb
│ ├── text_classification.ipynb
│ └── vector_search.ipynb
├── gradio_dashboard.py
├── tagged_description.txt
└── README.md


- **`data/`**: Cleaned datasets with additional metadata (categories, sentiment scores, etc.)
- **`notebooks/`**:
  - `data-exploration.ipynb`: Exploratory Data Analysis (EDA) visualizing book content and distributions.
  - `sentiment-analysis.ipynb`: Sentiment tagging of book descriptions.
  - `text_classification.ipynb`: Classification models for predicting book categories or genres.
  - `vector_search.ipynb`: Embeddings-based similarity and vector search for recommendation logic.
- **`gradio_dashboard.py`**: Python app to interactively test the recommender with a user-friendly UI.
- **`tagged_description.txt`**: Example custom description with sentiment tags.

---

##  Key Highlights & Skills Demonstrated

| Focus Area                         | Description |
|-----------------------------------|-------------|
| **Data Cleaning & Feature Engineering** | Consolidated and enriched book datasets for analysis. |
| **Exploratory Data Analysis**     | Used EDA techniques to understand data distributions and relationships. |
| **Text Processing & Sentiment**   | Applied sentiment analysis to assess user perception of book themes. |
| **Classification Modeling**       | Built models to predict book categories based on descriptions. |
| **Embedding & Vector Similarity** | Implemented vector search using text embeddings to recommend similar books. |
| **Interactive Presentation**      | Created a Gradio web interface for hands-on recommendation experience. |

---

##  How to Use This Repo

1. **Clone the repo:**
   ```bash
   git clone https://github.com/lukedev45/book-recommender.git
   cd book-recommender

2. Install dependencies (create a virtual environment if desired):
pip install pandas numpy scikit-learn gradio notebook

3. Run Jupyter Notebooks:
jupyter notebook notebooks/

4. Launch the interactive recommendation demo:
python gradio_dashboard.py

🧠 Topic Modelling using LDA

This project performs topic modeling on the NPR news dataset using TF-IDF Vectorization and NMF (Non-Negative Matrix Factorization) to uncover hidden topics from text data.

🚀 Steps

Load dataset (npr.csv)

Convert text to TF-IDF features

Apply NMF to extract topics

Display top words for each topic

Assign topic labels to articles

🧰 Libraries Used

pandas

scikit-learn

numpy

⚙️ How to Run
pip install -r requirements.txt
python topic_modelling_lda.py

📊 Output

Displays top words for each topic and assigns a Topic Label (e.g., Health, Politics, Education) to every article.

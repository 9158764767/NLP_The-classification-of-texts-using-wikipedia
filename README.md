# NLP_The-classification-of-texts-using-wikipedia
The goal of this project is to classify English texts into two categories—medical and non-medical—based on their content sourced from Wikipedia. The classification leverages natural language processing (NLP) techniques to pre-process and analyze text data.

Project Overview:

The goal of this project is to classify English texts into two categories—medical and non-medical—based on their content sourced from Wikipedia. The classification leverages natural language processing (NLP) techniques to pre-process and analyze text data, employing a combination of machine learning algorithms to accurately attribute the correct category.
Files contains two versions file name Assignment_1.ipynb contains multiple iteration’s I implemented during working on project implementation. File name GUItest.ipynb includes final implementation and testing of operational structure.

Operational Structure and Pipeline:

1.Data Retrieval: The project starts by fetching relevant articles from Wikipedia using the Wikipedia API. This includes articles that have predefined annotations, which can indicate medical or non-medical content.

2. Data Preprocessing:
  Tokenization: Splitting text into words or phrases to simplify further processing.
 Stop Word Removal: Eliminating common words that add no significant value to text analysis (e.g., "and", "the").
 Stemming and Lemmatization: Reducing words to their base or root form to improve the matching of terms during analysis.
  Vectorization: Transforming text data into numeric vectors using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) so that machine learning algorithms can process them.

3. Feature Extraction: Utilizing TF-IDF to turn preprocessed text into a matrix of TF-IDF features, which highlight the importance of words within documents relative to the corpus.

4. Model Training:
    - Naive Bayes Classifier: Employed for its effectiveness in text classification based on Bayes' theorem, with an assumption of independence among predictors.
    - Logistic Regression: Used as a robust classifier that estimates probabilities using a logistic function, known for its high accuracy in binary classification tasks.

5. Integration with GUI : Implementing a user interface using Tkinter to allow users to easily input text and receive classification results. This step involves setting up a backend to process user inputs through the trained models and display the results dynamically.

 Technologies Used:

Python: The primary programming language used for implementing the project.
NLTK (Natural Language Toolkit): Provides libraries for building Python programs to work with human language data.
Scikit-Learn: Utilized for machine learning, providing simple and efficient tools for data mining and data analysis.
Wikipedia API: For accessing and retrieving data directly from Wikipedia, facilitating the extraction of articles for classification.
Tkinter: For creating the graphical user interface to make the application user-friendly and accessible.

 Conclusion:
This project demonstrates the application of NLP techniques and machine learning to distinguish between medical and non-medical texts, showcasing the effectiveness of combining multiple NLP methods and machine learning for content-based classification tasks. The integration with a simple GUI ensures that the project is accessible to users without technical backgrounds, allowing for practical, real-world application.

Clone the repository, navigate to the project directory, and run the script:
```bash
git clone https://github.com/9158764767/NLP_The-classification-of-texts-using-wikipedia/.git
cd NLP_The-classification-of-texts-using-wikipedia
pip install -r requirements.txt
python guitest.py
or
python GUItest.ipynb


## Acknowledgements
This project was developed as part of the Advanced Text Classification Initiative at University of Verona.
Sincere thanks to all the contributors and maintainers of the Wikipedia API, NLTK, scikit-learn, and other open-source projects used in this work.
Special thanks to Professor Prof.Matteo Cristani  for their invaluable guidance and insights throughout the development of this project.

## Contact
For any queries regarding this project, please contact Abhishek Hirve at abhishek.hirve@studenti.univr.it




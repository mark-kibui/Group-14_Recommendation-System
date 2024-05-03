# Book Review Analysis and Recommendation System


## Business Understanding

The days of customers walking into a shop to buy what they need/want are long behind us and worse still if these are items are not basic needs. More and more clients prefer to make purchases from the comfort of their home. The goods that a retailer is able to market online is limitless however customers easily get tired of scrolling though an endless catalogue of items for sale.
Therefore rises the need for a recommendation system that will enable a client have a seamless buying experience. The reading culture is changing hence our choice of the amazon books dataset. A recommendation system will enable buyers get the most ideal and trending books to buy. The target audience would be both the retailers and the purchasers.

## Overview
The notebook is structured as follows:

    - Importing Libraries and Data
    - Data Preprocessing
    - Exploratory Data Analysis
    - Feature Engineering
    - Model Building
    - Evaluation
    - Clustering Analysis
    - Recommendation Generation
    - Conclusion
    - Recommendation

## Importing  Data
- The data has been obtained from https://amazon-reviews-2023.github.io/ and in jsonl format. An efficient format for storing data that is unstructured or produced over time.

- It contains a list of books sold in Amazon. The original dataset contains 4 million rows, from 1996 to 2023. We will trim it to the most recent 300k to make it easier to work with.

The data contains following features/columns in the dataset.
| Column Name | Description |
|---|---|
| rating | Rating of the product (from 1.0 to 5.0). |
| title_x | Title of the user review. |
| text | Text body of the user review. |
| images | Links to images (comma-separated if multiple). |
| asin(product key) | Unique identifier for the product. |
| parent_asin | Identifier for the parent product (applicable for variations). |
| user_id | Unique identifier for the reviewer. |
| timestamp | Date and time of the review. |
| helpful_vote | Number of helpful votes received by the review. |
| verified_purchase | Indicates whether the reviewer purchased the product (True/False). |
| main_Category | Main category (domain) to which the product belongs (e.g., Electronics, Clothing). |
| title_y | Name of the product as mentioned in the review. |
| price | Price of the product in US dollars. |

## Exploratory Data Analysis
- The dataset contains 300000 rows and 13 columns.
    - Book ratings were on a scale of 1 to 5. Below is a table showing the rating of books with the corresponding count.

| Rating | Count  |
|--------|--------|
| 1      | 18,798 |
| 2      | 9,954  |
| 3      | 14,855 |
| 4      | 32,038 |
| 5      | 224,355|

## Data Preprocessing
In this stage we dropped rows with duplicate book titles and also rows with null values. We renamed some of the columns: 'title_x' to 'title_rating', 'title_y' to 'title_book' and duplicate book title-rows dropped.
- We eventually dropped columns that were not necessary for the modelling and ended up having 156087 row and 6 columns.

      - rating
      - title_rating
      - text
      - user_id
      - verified_purchase
      - title_book
 
- The price columns was cleaned by filling null values with the mean price. Also converted to float data type.
- 'main_category' did not accurately give the book categorys thus being dropped. Below are the categories

| Main Category               | Count  |
|-----------------------------|--------|
| Books                       | 247,285|
| Buy a Kindle                | 44,247 |
| Audible Audiobooks          | 8,214  |
| Others                      | 204    |
| Toys & Games                | 29     |
| Amazon Home                 | 14     |
| Office Products             | 7      |
| Musical Instruments         | 6      |
| Arts, Crafts & Sewing       | 2      |
| AMAZON FASHION              | 1      |
| Tools & Home Improvement    | 1      |
| Industrial & Scientific     | 1      |

- 'title_x' column was renamed to 'title_rating' and 'title_y' to 'title_book'. The values were converted to lowercase and all punctuations removed. The values were tokenized and stopwords removed.
- The top 10 common words in 'tokenized_title_rating'are:

| Word      | Count  |
|-----------|--------|
| great     | 19,943 |
| read      | 10,661 |
| good      | 9,299  |
| love      | 6,119  |
| excellent | 3,931  |
| fun       | 3,845  |
| beautiful | 6,502  |
| amazing   | 2,875  |
| cute      | 2,577  |
| series    | 2,376  |


## Model Building
- We started with sentiment analysis. The goal was to understand the sentiment expressed in the review text. The sentiment analysis here included Multinomial Naive Bayes, Support Vector Machines (SVM) and Random Forest. We proceeded to building a hybrid recommendation system using the cosine similarity and Multinomial Naive Bayes

### Evaluation
- Below is assessment

Multinormial Naive Bayes

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.83      | 0.45   | 0.58     | 4926    |
| 1     | 0.90      | 0.98   | 0.94     | 26292   |
|       |           |        |          |         |
| **Accuracy** |           |        | **0.90**    | **31218**   |
| **Macro avg** | **0.87**      | **0.72**   | **0.76**    | **31218**   |
| **Weighted avg** | **0.89**      | **0.90**   | **0.89**    | **31218**   |


SVM

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.79      | 0.50   | 0.62     | 4926    |
| 1     | 0.91      | 0.98   | 0.94     | 26292   |
|       |           |        |          |         |
| **Accuracy** |           |        | **0.90**    | **31218**   |
| **Macro avg** | **0.85**      | **0.74**   | **0.78**    | **31218**   |
| **Weighted avg** | **0.89**      | **0.90**   | **0.89**    | **31218**   |

Random Forest

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.95      | 0.09   | 0.16     | 24700   |
| 1     | 0.85      | 1.00   | 0.92     | 131387  |
|       |           |        |          |         |
| **Accuracy** |           |        | **0.86**    | **156087**  |
| **Macro avg** | **0.90**      | **0.54**   | **0.54**    | **156087**  |
| **Weighted avg** | **0.87**      | **0.86**   | **0.80**    | **156087**  |

- Multinomial Naive Baye has a slight edge in identifying positive reviews with a recall of 0.98, but it struggles more with negative reviews  with a recall of 0.45.
- Multinomial Naive Bayes was better at capturing positive sentiment, while SVM and Random Forest offer a more balanced performance between positive and negative reviews.
- We compared the three classifiers with ROC curve but using One vs Rest (OvR) method. This method compares one class with others by reducing the multiclass classification to multiple binary classification.

![](https://github.com/mark-kibui/Group-14_Recommendation-System/blob/main/Images/ROC%20Curve.png)

- Naive Bayes: The curve for Naive Bayes has an AUC (Area Under the Curve) of 0.90, which indicates a high level of performance in distinguishing between the positive and negative classes.
- SVM has an AUC of 0.51, suggesting that it performs only slightly better than random guessing.
- RandomForest: The RandomForest curve has an AUC of 0.89, showing good performance, though not as high as Naive Bayes.

#### Content based filtering recommendation system
  - We created a TF-IDF matrix from the lemmatized text.
  - We then defined a function to get recommendations for a given book title.

This approach recommended books based on the lemmatized_title_rating text
  - Calculated TF-IDF vectors for "lemmatized_title_rating". It recommended books with the highest cosine similarity to a user's preferred book.
  
![](https://github.com/mark-kibui/Group-14_Recommendation-System/blob/main/Images/Content%20based%20recommendaiation.png)

#### Cosine similarity
- Recommendation of books based on **cosine similarity** between a given book title and other books in a TF-IDF matrix. It retrieves the 5-top most similar books and returns their titles.

####  Hybrid Recommendation System
- Multinomial Naive Bayes is better for short texts as in our case and there more suitable for the hybrid recommendation system. We will use it with the cosine similarity to have a more robust recommendation system for books.


# Conclusion
- We employed Multinomial Naive Bayes and cosine similarity to measure the closeness of books in the feature space. This combination allowed us to capitalize on the strengths of both methods, resulting robust recommendations.

# Recommendations
Some recommendations based on the  the finals result and some of the challenges encountered:
  - Consider refinin the model more for even better recommendations.
  - Enrich the data with more information about the books and even users' profiles
  - Implement a feedback loop where usres can also give feedback on the recommendations they get.

### Contributors
- Jacqueline Chepkwony
- Mark Kuria
- Peter Muthoma
- Nicholas Njubi
- Johnmark Kibui

### References
- Jawa, Vibhu. 2021. "Accelerating TF-IDF for Natural Language Processing with Dask and RAPIDS." RAPIDS AI. https://medium.com/rapids-ai/accelerating-tf-idf-for-natural-language-processing-with-dask-and-rapids-6f6e416429df. Accessed April 27, 2024.

- Scribendi Media. "How to Get Reviews for Your Book on Amazon." Scribendi Media, Dec. 2019. Image of screenshot from Amazon app with a 5-star review. Retrieved April 27, 2024, from https://scribemedia.com/wp-content/uploads/2019/12/How-To-Get-Reviews-For-Your-Book-On-Amazon-1024x594.jpg

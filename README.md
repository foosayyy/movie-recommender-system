# Movie Recommender System

This was my second Machine Learning project which I completed in my 3rd Year of College in **Graphic Era University**. First project being **Voice based e-mail sender**. OK now lets talk about this Recommender system, I have made a simple **Content-based recommender** where user can give a movie and movies those are closest to the user given movie are recommended.


# Working 

Basically all Content-based recommender systems use **K Nearest Neighbor** Algorithm. Even in this recommender system I have used K-NN Algorithm to find movies closest to the user provided movie.

## K-Nearest Neighbor(KNN) Algorithm

-   K-Nearest Neighbor is one of the simplest Machine Learning algorithms based on Supervised Learning technique.
-   K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.
-   K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.
-   K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.
-   K-NN is a  **non-parametric algorithm**, which means it does not make any assumption on underlying data.
-   It is also called a  **lazy learner algorithm**  because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.
-   KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.

## How does K-NN work in this Recommender System?

The K-NN working can be explained on the basis of the below algorithm:
-   **Step-0:**  Convert the movie data's to a vector.
-   **Step-1:**  Ask user to input a movie for which movies need to be recommended as.
-   **Step-2:**  Calculate the Euclidean distance of  **K number of neighbors**
-   **Step-3:**  Take the K nearest neighbors as per the calculated Euclidean distance.
-   **Step-4:**  Among these k neighbors, count the number of the data points in each category.
-   **Step-5:**  Assign the new data points to that category for which the number of the neighbor is maximum.
-   **Step-6:**  Our model is ready.

Suppose we have a new data point and we need to put it in the required category. Consider the below image:
![k-nearest-neighbor-algorithm-for-machine-learning5.jpg](https://github.com/foosayyy/movie-recommender-system/blob/main/README/k-nearest-neighbor-algorithm-for-machine-learning5.jpg?raw=true)
- We will calculate the  **Euclidean distance**  between the data points. The Euclidean distance is the distance between two points, which we have already studied in geometry. It can be calculated as:
![k-nearest-neighbor-algorithm-for-machine-learning4.jpg](https://github.com/foosayyy/movie-recommender-system/blob/main/README/k-nearest-neighbor-algorithm-for-machine-learning4.jpg?raw=true)
-   By calculating the Euclidean distance we got the nearest neighbors, as three nearest neighbors in category A and two nearest neighbors in category B. Consider the below image:
![k-nearest-neighbor-algorithm-for-machine-learning3.jpg](https://github.com/foosayyy/movie-recommender-system/blob/main/README/k-nearest-neighbor-algorithm-for-machine-learning3.jpg?raw=true)-   As we can see the 3 nearest neighbors are from category A, hence this new data point must belong to category A.

## Advantages of KNN Algorithm

-   It is simple to implement.
-   It is robust to the noisy training data
-   It can be more effective if the training data is large.



# Modules Used


## Pandas

-   Pandas is an open source library in Python. It provides ready to use high-performance data structures and data analysis tools.
-   Pandas module runs on top of  [NumPy](https://www.journaldev.com/15646/python-numpy-tutorial)  and it is popularly used for data science and data analytics.
-   NumPy is a low-level data structure that supports multi-dimensional arrays and a wide range of mathematical array operations. Pandas has a higher-level interface. It also provides streamlined alignment of tabular data and powerful time series functionality.
-   DataFrame is the key data structure in Pandas. It allows us to store and manipulate tabular data as a 2-D data structure.
-   Pandas provides a rich feature-set on the DataFrame. For example, data alignment, data statistics,  [slicing](https://www.journaldev.com/23139/python-slice), grouping, merging, concatenating data, etc.

## Numpy

- Python NumPy is the core library for scientific computing in Python. 
- NumPy provides a high-performance multidimensional array object and tools for working with these arrays.

## AST (Abstract Syntax tree)

- It allows us to interact with the Python code itself and can modify it.

> **Note:** The **Python interpreter** is responsible for running the Python code. It follows the pre-written instructions that translate the Python code into instructions that a machine can run.

## Scikit Learn

Scikit-learn is probably the most useful library for machine learning in Python. The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.

> **Note**: Sklearn is used to build machine learning models. It should not be used for reading the data, manipulating and summarizing it. There are better libraries for that (e.g. NumPy, Pandas etc.)

**Components of scikit-learn:**
Scikit-learn comes loaded with a lot of features. Here are a few of them to help you understand the spread:

-   **Supervised learning algorithms:**  Think of any supervised machine learning algorithm you might have heard about and there is a very high chance that it is part of scikit-learn. Starting from Generalized linear models (e.g Linear Regression), Support Vector Machines (SVM), Decision Trees to Bayesian methods – all of them are part of scikit-learn toolbox. The spread of machine learning algorithms is one of the big reasons for the high usage of scikit-learn. I started using scikit to solve supervised learning problems and would recommend that to people new to scikit / machine learning as well.
-   **Cross-validation:**  There are various methods to check the accuracy of supervised models on unseen data using sklearn.
-   **Unsupervised learning algorithms:**  Again there is a large spread of machine learning algorithms in the offering – starting from clustering, factor analysis, principal component analysis to unsupervised neural networks.
-   **Various toy datasets:**  This came in handy while learning scikit-learn. I had learned SAS using various academic datasets (e.g. IRIS dataset, Boston House prices dataset). Having them handy while learning a new library helped a lot.
-   **Feature extraction:**  Scikit-learn for extracting features from images and text (e.g. Bag of words)




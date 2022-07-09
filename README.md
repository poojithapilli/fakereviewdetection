# fakereviewdetection

##1. Collect the data of labelled reviews.

we were able to find a labelled dataset on 
kaggle provided by Amazon. The dataset contains a total of 21,000 reviews in which 
50% are fake reviews and 50% are genuine reviews.

##2. Preprocessing the dataset.
 
In real-life human writable text data 
contain various words with the wrong spelling, short words, special symbols, emojis, 
etc. we need to clean this kind of noisy text data before feeding it to the machine 
learning model.

##3. Feature extraction.

Variety of features that have been proposed and used separately by supervised 
approaches to identify fake reviews; which are review centric and reviewer centric 
features. In some cases review-centric features are considered separately. In other cases 
reviewer-centric features are taken into account. From the previous research proposed 
in we tried to pick the best features which would help identify the fake reviews. We 
have considered the following:

1. Rating: Users can rate the product from 1 to 5 stars representing
satisfaction/dissatisfaction about the product. This feature can be used to validate that 
the review written and the ratings given by the reviewer are intended only in one 
direction and do not contradict. Also, ratings corresponding to the fake reviews usually 
deviate from the average rating of the product. Thus, helping in classifying the fake 
reviews.
2. Verified purchase :The verified purchase means Amazon has verified that the 
person writing the review has purchased the product at Amazon and didn’t receive the 
product at deep discount. This feature helps to consider those reviews which are 
genuine as we get to know which purchaser has actually bought the product and used 
it.
3. Review length :The length of the review is also considered for training the model 
as suggests that reviews written by spammers are very short and defame/promote the 
product

##4. LabelEncoding.

It is used for converting labels into the numeric form from machine-readable form.
The algorithm will decide how the labels are operated. It is an important preprocessing 
step for the structured dataset.

##5. Count Vectorising the features.

The CountVectorizer function provided by sklearn in python is used to represent the 
corpus of words using a sparse matrix where each word acts as a column and the review 
as a row having the most frequent 1400 words from the corpus. This sparse matrix of 
1400 most frequent words is used as a feature vector to the model along with the verified 
purchase, rating and review length of the product.

##6. Training the logistic regression model and hyperparameter tuning.

Logistic regression model was trained on the train data for CountVectorizer using 
LogisticRegression in sklearn. Followed by this, hyperparameters of the model were 
tuned using GridSearch. Following is the list of optimal hyperparameters that result in 
best accuracy: C = 1, penalty = ‘l2’, solver = ‘newton-cg’.


##7. Evaluating the model on Test data.

Here the accuracy is upto 81% by using Logistic Regression Model for classification.


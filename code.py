import pandas as pd
import re
df=pd.read_csv("reviewslr.csv") #Load the data into python environment
df.head()


#Remove the punctuations which is noisy data.
df['new_reviews'] = df['REVIEW_TEXT'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['new_reviews'] = df['REVIEW_TEXT'].str.replace('[^\w\s]','')
df['new_reviews'].head()



#Removing unnecessary columns and extracting required columns.

from sklearn.preprocessing import LabelEncoder
df['labels'] = df['LABEL'].replace(['__label1__','__label2__'],['0','1'])
df.head(5)
df['len_of_reviews']=df['new_reviews'].str.len()
X=df[['RATING','VERIFIED_PURCHASE','len_of_reviews']]
def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df
s=Encoder(X)

#As the reviews are textual we use count vectorization inorder to convert to numerical data.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b",analyzer='word',max_features=1400, min_df=1,stop_words=None)
X_vectorizer = vectorizer.fit_transform(df['new_reviews'])
Y_vectorizer = vectorizer.transform(df['labels']) 
count_vect_df = pd.DataFrame(X_vectorizer.todense(), columns=vectorizer.get_feature_names())
k=pd.concat([s,count_vect_df],axis=1)

#Dividing the data into train and test in 80:20 ratio.
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

df1=df[['new_reviews','VERIFIED_PURCHASE','RATING']]
x_train,x_test,y_train,y_test=train_test_split(k,df['labels'],test_size=0.2, random_state=7)
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b",analyzer='word', min_df=1,stop_words=None)

#Scaling the values from 0 to 1.
scaler = MinMaxScaler()
model=scaler.fit(x_train)
scaled_data=model.transform(x_train)
scaled_data_test=model.transform(x_test)

#Training the Logistic Regression model and checking for accuracy and also tuning hyper parameters.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
clf = LogisticRegression(C=1,penalty='l2',solver='newton-cg').fit(scaled_data,y_train)
pred=clf.predict(scaled_data_test)
score=accuracy_score(y_test,pred)
print(f'Accuracy: {round(score*100,2)}%')
report=classification_report(y_test,pred,output_dict=True)
report_result = pd.DataFrame(report).transpose()
report_result


#Training the SVC model and checking for accuracy and also tuning hyper parameters.
from sklearn.svm import SVC
clf = SVC(C=10,gamma=0.0001,kernel='rbf').fit(scaled_data,y_train)
predct=clf.predict(scaled_data_test)
score=accuracy_score(y_test,predct)
print(f'Accuracy: {round(score*100,2)}%')
report1=classification_report(y_test,predct,output_dict=True)
report_result1 = pd.DataFrame(report).transpose()
report_result1

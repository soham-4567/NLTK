
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
df=pd.read_csv(r"C:\Users\Intern\Desktop\Myques.csv")
print(df)
x=input("Enter your question").lower()
x_list = word_tokenize(x)

s = stopwords.words('english')
x_set = {w for w in x_list if not w in s}
l1 =[];l2 =[]
similarity_ranking = []
questions=df['Questions'].tolist()
for y in questions:
    s = stopwords.words('english')
    y=y.lower()
    y_list = word_tokenize(y)
    y_set = {w for w in y_list if not w in s}
    rvector = x_set.union(y_set)
    for w in rvector:
    	if w in x_set: l1.append(1) # create a vector
    	else: l1.append(0)
    	if w in y_set: l2.append(1)
    	else: l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
    		c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    print("similarity: ", cosine)
    similarity_ranking.append(cosine)
similar_ques_idx=similarity_ranking.index(max(similarity_ranking))
relevant_answer=df["Answers"].iloc[similar_ques_idx]
print(f'The most relevant answer is {relevant_answer}')

# # Program to measure the similarity between
# # two sentences using cosine similarity.

#
# # X = input("Enter first string: ").lower()
# # Y = input("Enter second string: ").lower()
# X ="I love horror movies"
# Y ="Lights out is a horror movie"
#
# # tokenization
# X_list = word_tokenize(X)
# Y_list = word_tokenize(Y)
#
# # sw contains the list of stopwords
# sw = stopwords.words('english')
# l1 =[];l2 =[]
#
# # remove stop words from the string
# X_set = {w for w in X_list if not w in sw}
# Y_set = {w for w in Y_list if not w in sw}
#
# # form a set containing keywords of both strings
# rvector = X_set.union(Y_set)
# for w in rvector:
# 	if w in X_set: l1.append(1) # create a vector
# 	else: l1.append(0)
# 	if w in Y_set: l2.append(1)
# 	else: l2.append(0)
# c = 0
#
# # cosine formula
# for i in range(len(rvector)):
# 		c+= l1[i]*l2[i]
# cosine = c / float((sum(l1)*sum(l2))**0.5)
# print("similarity: ", cosine)


#!/usr/bin/env python
# coding: utf-8

# Sir Im new to GIT so dont know much about pull requests and all so uploaded .py file. Kindly excuse me sir!
# # DIABETES PREDICTION USING MACHINE LEARNING

# ##  IMMARAJU SAMUEL - 20011D0505

# #### IMPORT ALL NECESSARY LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image


# ## 1. DATA COLLECTION

# This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.This data has been directly collected from the patients of Sylhet Diabetes Hospital in Sylhet, Bangladesh and approved by the doctor.
# https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv
# 
# #### Features of the dataset
# The dataset consist of total **16** features and one target variable named class.
# 
# - **1. Age:** Age in years ranging from (16years to 90 years)
# - **2. Gender:** Male / Female
# - **3. Polyuria:** Yes / No
# - **4. Polydipsia:** Yes/ No
# - **5. Sudden weight loss:** Yes/ No 
# - **6. Weakness:** Yes/ No
# - **7. Polyphagia:** Yes/ No
# - **8. Genital Thrush:** Yes/ No
# - **9. Visual blurring:** Yes/ No
# - **10. Itching:** Yes/ No
# - **11. Irritability:** Yes/No
# - **12. Delayed healing:** Yes/ No
# - **13. Partial Paresis:** Yes/ No
# - **14. Muscle stiffness:** Yes/ No
# - **15. Alopecia:** Yes/ No
# - **16. Obesity:** Yes/ No
# 
# **Class:** Positive / Negative

# __LOAD THE DATA__

# In[5]:


a=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv', encoding = "utf-8")

'''
# __DATA DESCRIPTION__
# 

# In[4]:


print("CHECKING HOW MANY NULL VALUES EXIST")
a.isnull().sum()


# In[5]:


a.info()


# In[6]:


a.describe()


# ## DATA VISUALISATION

# In[45]:


plt.title("CLASS DISTRIBUTION")
sns.countplot(a['class'],data=a)


# ### Gender

# In[44]:


plt.title("GENDER DISTRIBUTION")
sns.countplot(a['Gender'], hue=a['class'], data=a)


# In[8]:


plot_criteria= ['Gender', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# ### Polyuria

# Polyuria is defined as the **frequent passage of large volumes of urine** – more than 3 litres a day compared to the normal daily urine output in adults of about 1 to 2 litres.
# #### Causes:
# The most common cause of polyuria in both adults and children is ***uncontrolled diabetes mellitus***, which causes **osmotic diuresis**, when glucose levels are so high that glucose is excreted in the urine. Water follows the glucose concentration passively, leading to abnormally high urine output. 
# 
# In the absence of diabetes mellitus, the most common causes are decreased secretion of aldosterone due to adrenal cortical tumor, primary **polydipsia** (excessive fluid drinking),
# 

# In[9]:


plot_criteria= ['Polyuria', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[43]:


plt.title("POLYURIA DISTRIBUTION")
sns.countplot(a['Polyuria'], hue=a['class'], data=a)


# ### Polydipsia

# Polydipsia is the term given to **excessive thirst** and is one of the initial symptoms of diabetes. It is also usually accompanied by temporary or prolonged dryness of the mouth.
# 
# However, if you feel thirsty all the time or your thirst is stronger than usual and continues even after you drink, it can be a sign that not all is well inside your body.
# 
# Excessive thirst can be caused by high blood sugar (hyperglycemia), and is also one of the ‘Big 3’ signs of diabetes mellitus i.e., 
# 
# **1. Polyuria**<br>
# **2. Polydipsia** <br>
# **3. Polyphagia** <br>
# 
# Generally, increased thirst (polydipsia) and an increased need to urinate (polyuria) will often come as a pair.

# In[11]:


plot_criteria= ['Polydipsia', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[42]:


plt.title("POLYDIPSIA DISTRIBUTION")
sns.countplot(a['Polydipsia'], hue=a['class'], data=a)


# ### Sudden weight loss

# In[13]:


plot_criteria= ['sudden weight loss', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[41]:


plt.title("SUDDEN WEIGHT LOSS DISTRIBUTION")
sns.countplot(a['sudden weight loss'], hue=a['class'], data=a)


# ### Weakness

# In[15]:


plot_criteria= ['weakness', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[40]:


plt.title("WEAKNESS DISTRIBUTION")
sns.countplot(a['weakness'], hue=a['class'], data=a)


# ### Polyphagia

# Polyphagia, also known as hyperphagia, is the medical term for **excessive or extreme hunger**. 
# 
# It's different than having an increased appetite after exercise or other physical activity. 
# 
# While your hunger level will return to normal after eating in those cases, polyphagia won't go away if you eat more food.

# In[17]:


plot_criteria= ['Polyphagia', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[39]:


plt.title("POLHAGIA DISTRIBUTION")
sns.countplot(a['Polyphagia'], hue=a['class'], data=a)


# ### Genital thrush
# 
# Thrush (or candidiasis) is a common condition caused by a type of yeast called Candida. It mainly affects the private parts and can be irritating and painful.

# In[19]:


plot_criteria= ['Genital thrush', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[38]:


plt.title("GENITAL THRUSH DISTRIBUTION")
sns.countplot(a['Genital thrush'], hue=a['class'], data=a)


# ### Visual Blurring

# In[21]:


plot_criteria= ['visual blurring', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[46]:


plt.title("VISUAL BLURRING DISTRIBUTION")
sns.countplot(a['visual blurring'], hue=a['class'], data=a)


# ### Itching

# In[23]:


plot_criteria= ['Itching', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[47]:


plt.title("ITCHING DISTRIBUTION")
sns.countplot(a['Itching'], hue=a['class'], data=a)


# ### Irritability

# In[25]:


plot_criteria= ['Irritability', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[49]:


plt.title("IRRITABILITY DISTRIBUTION")
sns.countplot(a['Irritability'], hue=a['class'], data=a)


# ### Delayed Healing

# It means **taking allot more time than normal to heal** when there is any injury particularly healing of skin when there is a cut 

# In[27]:


plot_criteria= ['delayed healing', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[50]:


plt.title("DELAYED HEALING DISTRIBUTION")
sns.countplot(a['delayed healing'], hue=a['class'], data=a)


# ### Partial Paresis

# Paresis involves the **weakening of a muscle or group of muscles**. It may also be referred to as partial or mild paralysis. Unlike paralysis, people with paresis can still move their muscles. These movements are just weaker than normal.

# In[29]:


plot_criteria= ['partial paresis', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[51]:


plt.title("PARTIAL PARESIS DISTRIBUTION")
sns.countplot(a['partial paresis'], hue=a['class'], data=a)


# ### Muscle Stiffness

# In[31]:


plot_criteria= ['muscle stiffness', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[52]:


plt.title("MUSCLE STIFFNESS DISTRIBUTION")
sns.countplot(a['muscle stiffness'], hue=a['class'], data=a)


# ### Alopecia

# **Sudden hair loss** that starts with **one or more circular bald patches** that may overlap.
# Alopecia areata occurs when the immune system attacks hair follicles and may be brought on by severe stress.
# The main symptom is hair loss.

# In[33]:


plot_criteria= ['Alopecia', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[53]:


plt.title("ALOPECIA DISTRIBUTION")
sns.countplot(a['Alopecia'], hue=a['class'], data=a)


# ### Obesity

# In[35]:


plot_criteria= ['Obesity', 'class']
round(pd.crosstab(a[plot_criteria[0]], a[plot_criteria[1]], normalize='columns')*100,2)


# In[54]:


plt.title("OBESITY DISTRIBUTION")
#sns.countplot(a['Obesity'], hue=a['class'], data=a)
sns.countplot('Obesity', hue='class', data=a)


# ## DATA VISUALTION USING DIMENSIONALITY REDUCTION

# In[14]:


c=a.copy()


# In[38]:


# CONVERTING THE FEATURE VALUES INTO BINARY VALUES
c['Gender'] = c['Gender'].apply(lambda x: 0 if x=='Female' else (1 if x=='Male' else x )) 
for i in c.columns[2:-1]:
    c[i] = c[i].apply(lambda x: 0 if x=='No' else (1 if x=='Yes' else x ))
    

# CONVERTING CLASS LABELS INTO BINARY VALUES
#c['class'] = c['class'].apply(lambda x: 0 if x=='Negative' else 1) 


# In[39]:


c_d=c.drop(['class'],axis=1)
label=c[['class']]

ss=StandardScaler()
c_std=ss.fit_transform(c_d)


# In[40]:


print(c_std.shape)
print(label.shape)
print(c.shape)


# In[56]:


k=label.replace('Negative', 0)
k=k.replace('Positive', 1)


# In[58]:


df=np.hstack((c_std,k))


# In[63]:


df


# In[64]:


df=pd.DataFrame(df, columns=c.columns)
dfc=df.corr()


# In[70]:


dfs=dfc.style.background_gradient()
dfs


# In[75]:


import dataframe_image as dfi
dfi.export(dfs, 'corr.png')

#UNCOMMENT HERE TO GET VISUALISATIONS FOR DIMENSIONALTY REDUCTION USING PCA AND TSNE

# ## PCA - PRINCIPAL COMPONENT ANALYSIS

# In[7]:


from sklearn import decomposition
pca = decomposition.PCA()
pca.n_components = 2
pca_c = pca.fit_transform(c_std)

pca_c = np.hstack((pca_c, label))
pca_c


# In[12]:


# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_c, columns=("2nd_principal", "1st_principal", "label"))
pca_df[["1st_principal", "2nd_principal", "label"]]


# In[13]:


sns.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()


# ## (T-SNE) T-DISTRIBUTED STOCHASTIC NEIGHBOURHOOD EMBEDDING

# ### COMBINATIONS OF VARIOUS PERPLEXITY, ITERATIONS

# In[65]:


from sklearn.manifold import TSNE

n_components=2

ts = TSNE(random_state=0)

ts_data = ts.fit_transform(c_std)

ts_c = np.hstack((ts_data, label))


# **PERPLEXITY:** No. of Points to which distance should be preserved
# <br>**ITERATIONS:** No. of times the algo should run

# In[68]:


tsne_df = pd.DataFrame(data=ts_c, columns=("Dim_1", "Dim_2", "label"))
tsne_df[["Dim_1", "Dim_2", "label"]]


# In[69]:


# Ploting the result of tsne
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


# In[70]:


ts = TSNE(n_components=2, random_state=0, perplexity=75,  n_iter=400)

ts_data = ts.fit_transform(c_std)

ts_c = np.hstack((ts_data, label))

tsne_df = pd.DataFrame(data=ts_c, columns=("Dim_1", "Dim_2", "label"))

#tsne_df


# In[71]:


# Ploting the result of tsne
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


# In[80]:


ts = TSNE(n_components=2, random_state=0, perplexity=100,  n_iter=400)

ts_data = ts.fit_transform(c_std)

ts_c = np.hstack((ts_data, label))

tsne_df = pd.DataFrame(data=ts_c, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
'''

# ## 2. DATA PREPROCESSING

# In[4]:


b=a.copy() #making a new copy of the entire dataframe


# The entire data consists of 'yes' or 'no', 'positive' or 'negative', 'male' or 'female'. so we convert this data into binary numbers to fit into the model 


# So we convert the above string values into bbinary values
# - YES - 1
# - NO - 0
# - MALE - 1
# - FEMALE - 0
# - POSITIVE - 1
# - NEGATIVE - 0
# positive and negative are class labels


# CONVERTING THE FEATURE VALUES INTO BINARY VALUES
b['Gender'] = b['Gender'].apply(lambda x: 0 if x=='Female' else (1 if x=='Male' else x )) 
for i in b.columns[2:-1]:
    b[i] = b[i].apply(lambda x: 0 if x=='No' else (1 if x=='Yes' else x ))
    

# CONVERTING CLASS LABELS INTO BINARY VALUES
b['class'] = b['class'].apply(lambda x: 0 if x=='Negative' else 1) 


# NORMALISING AGE
minmax = MinMaxScaler()
b[['Age']] = minmax.fit_transform(b[['Age']])

X=b.drop(['class'],axis=1)
y=b['class']

'''

#X.corrwith(y).plot.bar(figsize = (14, 6), title = "Correlation with Diabetes", fontsize = 12, rot = 80, grid = True)

'''
# ## 3. BUILDING A ML MODEL

# #### Splitting the data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify=y, random_state = 1234)
'''


## checking distribution of traget variable in train test split
print('Distribution of target variable in training set')
print(y_train.value_counts())

print('Distribution of target variable in test set')
print(y_test.value_counts())


# In[20]:


X_train.head()
'''

'''

#CROSS VALIDATION METRIC

kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'


# ## Logistic Regression


lr = LogisticRegression(random_state = 0, penalty = 'l2')
lr.fit(X_train, y_train)




cv_lr = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = kfold,scoring=scoring)
cv_lr.mean()



y_pred_lr = lr.predict(X_test)

acc_lr= accuracy_score(y_test, y_pred_lr)
roc_lr=roc_auc_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

lrdf = pd.DataFrame([['Logistic Regression',acc_lr, cv_lr.mean(), prec_lr, rec_lr, f1_lr,roc_lr]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
lrdf



cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.title('Confusion matrix of the Logistic classifier')
sns.heatmap(cm_lr,annot=True,fmt="d")
plt.show()


TP1 = cm_lr[1,1] # true positive 
TN1 = cm_lr[0,0] # true negatives
FP1 = cm_lr[0,1] # false positives
FN1 = cm_lr[1,0] # false negatives

print(round((TP1+TN1)/(TP1+FP1+TN1+FN1)*100,2),"%") #accuracy


# ## K Nearest Neighbour

knn = KNeighborsClassifier(n_neighbors=7,metric = 'euclidean',p = 2)
knn.fit(X_train,y_train)



cv_knn = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = kfold, scoring=scoring)
cv_knn.mean()



y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
roc_knn =roc_auc_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

knndf = pd.DataFrame([['K-Nearest Neighbour',acc_knn, cv_knn.mean(), prec_knn, rec_knn, f1_knn, roc_knn]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
knndf


cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.title('Confusion matrix of KNN Classifier')
sns.heatmap(cm_knn,annot=True,fmt="d")
plt.show()


TP2 = cm_knn[1,1] # true positive 
TN2 = cm_knn[0,0] # true negatives
FP2 = cm_knn[0,1] # false positives
FN2 = cm_knn[1,0] # false negatives

print(round((TP2+TN2)/(TP2+FP2+TN2+FN2)*100,2),"%") #accuracy


# ## Gaussian Naive Bayes

gnb = GaussianNB()
gnb.fit(X_train,y_train)


cv_gnb = cross_val_score(estimator = gnb, X = X_train, y = y_train, cv = kfold, scoring=scoring)
cv_gnb.mean()


y_pred_gnb = gnb.predict(X_test)

acc_gnb = accuracy_score(y_test, y_pred_gnb)
roc_gnb = roc_auc_score(y_test, y_pred_gnb)
prec_gnb = precision_score(y_test, y_pred_gnb)
rec_gnb = recall_score(y_test, y_pred_gnb)
f1_gnb = f1_score(y_test, y_pred_gnb)

gnbdf = pd.DataFrame([['Gaussian Naive Bayes',acc_gnb, cv_gnb.mean(), prec_gnb, rec_gnb, f1_gnb, roc_gnb]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
gnbdf


cm_gnb = confusion_matrix(y_test, y_pred_gnb)
plt.title('Confusion matrix of Naive Bayes Classifier')
sns.heatmap(cm_gnb,annot=True,fmt="d")
plt.show()


TP3 = cm_gnb[1,1] # true positive 
TN3 = cm_gnb[0,0] # true negatives
FP3 = cm_gnb[0,1] # false positives
FN3 = cm_gnb[1,0] # false negatives
print(round((TP3+TN3)/(TP3+FP3+TN3+FN3)*100,2),"%") #accuracy


# ## Decision Tree


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


cv_dt = cross_val_score(estimator = dt, X = X_train, y = y_train, cv = kfold, scoring=scoring)
cv_dt.mean()


y_pred_dt = dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
roc_dt = roc_auc_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

dtdf = pd.DataFrame([['Decision Tree',acc_dt, cv_dt.mean(), prec_dt, rec_dt, f1_dt, roc_dt]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
dtdf

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.title('Confusion matrix of Decision Tree Classifier')
sns.heatmap(cm_dt,annot=True,fmt="d")
plt.show()


TP4 = cm_dt[1,1] # true positive 
TN4 = cm_dt[0,0] # true negatives
FP4 = cm_dt[0,1] # false positives
FN4 = cm_dt[1,0] # false negatives
print(round((TP4+TN4)/(TP4+FP4+TN4+FN4)*100,2),"%") #accuracy
'''

# ## Random Forest

rf = RandomForestClassifier(criterion='gini',n_estimators=100)
rf.fit(X_train,y_train)
'''
cv_rf = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = kfold, scoring=scoring)
cv_rf.mean()
'''
#y_pred_rf = rf.predict(X_test)

'''
acc_rf = accuracy_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

dtrf = pd.DataFrame([['Random Forest', acc_rf, cv_rf.mean(), prec_rf, rec_rf, f1_rf, roc_rf]],
               columns = ['Model', 'Accuracy','Cross Val Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
dtrf

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.title('Confusion matrix of Random Forest')
sns.heatmap(cm_rf,annot=True,fmt="d")
plt.show()


TP5 = cm_rf[1,1] # true positive 
TN5 = cm_rf[0,0] # true negatives
FP5 = cm_rf[0,1] # false positives
FN5 = cm_rf[1,0] # false negatives
print(round((TP5+TN5)/(TP5+FP5+TN5+FN5)*100,2),"%") #accuracy


# ## 4. EVALUATION

results=pd.concat([dtrf,dtdf,lrdf,knndf,gnbdf], ignore_index=True)
results
'''

def diapred():
    z=[]
    z_new=[]
    minage=16
    maxage=90 #minage and maxage are taken from dataset

    print("ENTER ALL VALUES")
#taking inputs from user and converting them to appropriate form to fit into the data. the inputs are stored in a list
    print("What's your age")
    w=int(input())
    z.append(w)
    w=(w-minage)/float(maxage-minage)  #normalising age
    z_new.append(w)

    print("What's your Gender (M/F)")
    w=input()
    z.append(w)
    if w=='M' or w=='m':
        z_new.append(1)
    elif w=='F' or w=='f':
        z_new.append(0)
    else:
        print("WRONG ENTRY!!")

    for v in b.columns[2:-1]:
        print("Do you have",v,"(Y/N)")
        w=input()
        z.append(w)
        if w=='Y' or w=='y':
            z_new.append(1)
        elif w=='N' or w=='n':
            z_new.append(0)
        else:
            print("WRONG ENTRY!!")
            #sys.exit()
            
            
#the list is not suitable to give it as input to the model. So...            
    z_np=np.array(z_new)           #1st converting the list to a numpy array    
    z_np1=z_np.reshape(1,16)       #changing the shape of array to 1x16 as it was 16x1
    

#passing the numpy array (inputs) to the model and predicting diabetes
    y_pred = rf.predict(z_np1)
    
    if y_pred==0:
        print("Your Safe!! You don't have diabetes")
    else:
        print("Aww!! You might have Diabetes so start taking Precautions as soon as possible")


#This func PREDICTs THE DIABETES OF A PERSON.
diapred() 


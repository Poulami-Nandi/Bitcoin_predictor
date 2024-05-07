# Import necessary libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
 
import warnings
warnings.filterwarnings('ignore')

# Import dataframe 
df = pd.read_csv('Bitcoin Historical Data.csv', sep=',')
df.head()

# Change data type of Price Open High Low and Change columns
df['Price'].dtype
df['Price'] = df['Price'].str.replace(',','')
df['Price'] = df['Price'].astype(float)
df['Open'] = df['Open'].str.replace(',','')
df['Open'] = df['Open'].astype(float)
df['High'] = df['High'].str.replace(',','')
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].str.replace(',','')
df['Low'] = df['Low'].astype(float)
df.info()

# Modify data type of "Change %" from str to float
df['Change %'] = df['Change %'].str[:-1].astype(float)
# Add a new field "Close" = "Open" + "Change %" of "Open"
df['Close'] = df['Open'] + (df['Change %']/100) * df['Open']
df.head()

def replace_MKB_with_values(x):
  m = 0
  if x[-1] == "K": m = 1000
  elif x[-1] == "M": m = 1000000
  elif x[-1] == "B": m = 1000000000
  return float(x[:-1]) * m

# Convert string "65.5K" into float 65500.0 from "Vol." column 
df['Vol.'] = df['Vol.'].apply(replace_MKB_with_values)

# Plot bitcoint closing price over the year
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

# show the dataframe's information and characteristic
df.head()
df.info()
df.describe()

# Plot bitcoint closing price 
plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Bitcoin Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

# check for null entries
df.isnull().sum()

# Plot distribution graph of Open, High, Low and Close columns 
features = ['Open', 'High', 'Low', 'Close']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,2,i+1)
  sb.distplot(df[col])
plt.show()

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,2,i+1)
  sb.boxplot(df[col])
plt.show()

splitted = df['Date'].str.split('/', expand=True)

# Split the date column into Day Month and Year 
df['Month'] = splitted[0].astype('int')
df['Day'] = splitted[1].astype('int')
df['Year'] = splitted[2].astype('int')
df.drop(['Date'], axis=1) 
df.head(20)

# Drop the Date column, its redundant now
df = df.drop(['Date'], axis=1) 
df.info()

# Group 'Open', 'High', 'Low' and 'Close' columns by year and plot those
data_grouped = df.groupby('Year').mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()

# add a marker for records from quarter end months
df['is_quarter_end'] = np.where(df['Month']%3==0,1,0)
df.head(140)

# Add 'open-close' and 'low-high' column
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']

# Add \target\ column. This column denote whether the market is bulling or bearish as compared to the previous month
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Plat a pie chart to show quantity of bullish vs bearish market 
plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10, 10))
# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

# split the records between test set and validation set
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.2, random_state=55)
print(X_train.shape, X_valid.shape)

# Run the model and check accurecy 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  training_proba = models[i].predict_proba(X_train)[:,1]
  validation_proba = models[i].predict_proba(X_valid)[:,1]
  print('Training Accuracy : ', metrics.roc_auc_score(Y_train, training_proba))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, validation_proba))
  print()

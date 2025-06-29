# -HealthMate-AI-
**Your AI-Powered Health Companion for Precision Care"**

Flow Chart that explains the whole project


![WhatsApp Image 2024-04-24 at 11 23 12 PM](https://github.com/manyaj011/-HealthGuard-/assets/110671189/d38b1cab-fcc6-4368-ac89-29eac194758e)
![WhatsApp Image 2024-04-24 at 11 36 42 PM](https://github.com/manyaj011/-HealthGuard-/assets/110671189/ff35aa2e-4eea-4621-b49e-d7f70355e306)
![WhatsApp Image 2024-04-24 at 11 27 07 PM](https://github.com/manyaj011/-HealthGuard-/assets/110671189/0d21bad5-457e-47ef-9630-6f55ac8063fb)



## To Run the project, libraries used
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline 
```

## For Grid search and random search techniques used for hyperparameter tuning 
```bash
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import randint
```

## For MultiClassifier Approach and Multiple Binary Approach

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import itertools
import cv2
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```
**IF USER ENTERS SYMPTOMS-**
1.Decision Tree
2.Random Forest Classification
3.eXtreme Gradient Boosting

Grid search and random search are common techniques used for hyperparameter tuning in machine learning models are applied on all three models
XGBoost turned out to be best

<img width="368" alt="image" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/7de43367-1cea-49c7-b4e6-d0d0a37a6173">



**IF USER PROVIDES IMAGE-**
CNN IS APPLIED
1 Multi Classifier Approach
2.Multiple Binary Classfication Approach




**The Result of the multiclassifier approach, where a Convolutional Neural Network (CNN) is applied to classify all seven diseases simultaneously:**

<img width="482" alt="all7graph" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/21595570-2659-4951-8bc3-a6614f88826a">


**The result of the multiple binary classification approach(one disease  MEL is shown), where a Convolutional Neural Network (CNN) model is applied to each of the seven diseases independently:**

<img width="455" alt="MELgraph" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/1d27eb31-65cb-403b-aa68-151ccfecdebe">



**Comparison of Multi classifier Approach and Multiple Binary Classification Approach**

![Screenshot 2024-04-24 231959](https://github.com/manyaj011/-HealthGuard-/assets/110671189/95a640f6-1abf-49a8-a438-abfc1771d922)
![Screenshot 2024-04-24 232305](https://github.com/manyaj011/-HealthGuard-/assets/110671189/a5939a60-7a20-424a-9b11-840d5426ce81)
![Screenshot 2024-04-24 232343](https://github.com/manyaj011/-HealthGuard-/assets/110671189/30dd120b-9868-4b77-b2c5-cf8123912b1a)




**Doctors and Drug Recommendation(LLM ChatBot)**

<img width="416" alt="image" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/f92a9879-283a-4304-98d5-1b1a7bc6867a">



**Future Scope**

1.Contribution to the world by giving information about identfiable disease

demo-

<img width="305" alt="image" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/5f06891e-15aa-4a65-9e67-d59523f33ff8">



2.Real Time Mapping of the idetified disease

demo of Real Time Mapping-

<img width="305" alt="image" src="https://github.com/manyaj011/-HealthGuard-/assets/110671189/985b7f0e-8b60-4284-befc-8eb7de81c890">


## Technologies Used

### Machine-Learning Models used

Symptom-Based Disease Prediction: Decision Tree Classifier, Random Forest, and XGBoost.

Image-Based Disease Identification: Convolutional Neural Networks (CNN) for multi-class and multiple binary classifications.

Large Language Models (LLMs): For ChatBot 

### DSA Algorithms Used

Dijiktra's Algorithm

Hash Table(Priority Queue)

### Programming Technologies

Python


Tensor Flow/Keras


PyTorch
 

### dataset

disease , drug and doctor dataset is already provided  
for images, dataset used HAM10000

DGX Workstation is used

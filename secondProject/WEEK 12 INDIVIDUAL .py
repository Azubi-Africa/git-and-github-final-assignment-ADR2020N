{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "07aa2c86",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "from sklearn.datasets import fetch_openml\n",
    "\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7d05f472",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "sklearn.utils.Bunch"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mnist = fetch_openml('mnist_784', version=1)\n",
    "type(mnist)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c92ba619",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mnist.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "58ee7b09",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(pandas.core.frame.DataFrame, pandas.core.series.Series)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X, y = mnist[\"data\"], mnist[\"target\"]\n",
    "\n",
    "type(X), type(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a4193b1a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((70000, 784), (70000,))"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape, y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "21a47863",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pixel1</th>\n",
       "      <th>pixel2</th>\n",
       "      <th>pixel3</th>\n",
       "      <th>pixel4</th>\n",
       "      <th>pixel5</th>\n",
       "      <th>pixel6</th>\n",
       "      <th>pixel7</th>\n",
       "      <th>pixel8</th>\n",
       "      <th>pixel9</th>\n",
       "      <th>pixel10</th>\n",
       "      <th>...</th>\n",
       "      <th>pixel775</th>\n",
       "      <th>pixel776</th>\n",
       "      <th>pixel777</th>\n",
       "      <th>pixel778</th>\n",
       "      <th>pixel779</th>\n",
       "      <th>pixel780</th>\n",
       "      <th>pixel781</th>\n",
       "      <th>pixel782</th>\n",
       "      <th>pixel783</th>\n",
       "      <th>pixel784</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 784 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   pixel1  pixel2  pixel3  pixel4  pixel5  pixel6  pixel7  pixel8  pixel9  \\\n",
       "0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0   \n",
       "1     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0   \n",
       "2     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0   \n",
       "3     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0   \n",
       "4     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0   \n",
       "\n",
       "   pixel10  ...  pixel775  pixel776  pixel777  pixel778  pixel779  pixel780  \\\n",
       "0      0.0  ...       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "1      0.0  ...       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "2      0.0  ...       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "3      0.0  ...       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "4      0.0  ...       0.0       0.0       0.0       0.0       0.0       0.0   \n",
       "\n",
       "   pixel781  pixel782  pixel783  pixel784  \n",
       "0       0.0       0.0       0.0       0.0  \n",
       "1       0.0       0.0       0.0       0.0  \n",
       "2       0.0       0.0       0.0       0.0  \n",
       "3       0.0       0.0       0.0       0.0  \n",
       "4       0.0       0.0       0.0       0.0  \n",
       "\n",
       "[5 rows x 784 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1aa4ecbc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    5\n",
       "1    0\n",
       "2    4\n",
       "3    1\n",
       "4    9\n",
       "Name: class, dtype: category\n",
       "Categories (10, object): ['0', '1', '2', '3', ..., '6', '7', '8', '9']"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "28f53f62",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "some_digit = X.iloc[0,:].values\n",
    "some_digit_image = some_digit.reshape(28, 28)\n",
    "\n",
    "plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation=\"nearest\")\n",
    "plt.axis(\"on\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b33b9c8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "y = y.astype(np.uint8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "8f151bc4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "a8ce1911",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.isnull().sum().sum() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d13dac0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e6efa95b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but SGDClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([3], dtype=uint8)"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import SGDClassifier\n",
    "\n",
    "sgd_clf = SGDClassifier(random_state=42)\n",
    "#sgd_clf.fit(X_train, y_train_5)\n",
    "sgd_clf.fit(X_train, y_train) # y_train, not y_train_5\n",
    "sgd_clf.predict([some_digit])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "30af696c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.87365, 0.85835, 0.8689 ])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_predict\n",
    "\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring=\"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "79e98500",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.8983, 0.891 , 0.9018])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))\n",
    "cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring=\"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "6c30a8be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5577,    0,   22,    5,    8,   43,   36,    6,  225,    1],\n",
       "       [   0, 6400,   37,   24,    4,   44,    4,    7,  212,   10],\n",
       "       [  27,   27, 5220,   92,   73,   27,   67,   36,  378,   11],\n",
       "       [  22,   17,  117, 5227,    2,  203,   27,   40,  403,   73],\n",
       "       [  12,   14,   41,    9, 5182,   12,   34,   27,  347,  164],\n",
       "       [  27,   15,   30,  168,   53, 4444,   75,   14,  535,   60],\n",
       "       [  30,   15,   42,    3,   44,   97, 5552,    3,  131,    1],\n",
       "       [  21,   10,   51,   30,   49,   12,    3, 5684,  195,  210],\n",
       "       [  17,   63,   48,   86,    3,  126,   25,   10, 5429,   44],\n",
       "       [  25,   18,   30,   64,  118,   36,    1,  179,  371, 5107]],\n",
       "      dtype=int64)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "y_train_pred_sgd = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)\n",
    "conf_mx = confusion_matrix(y_train, y_train_pred_sgd)\n",
    "conf_mx "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "1f1cfdee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "     class 0       0.97      0.94      0.95      5923\n",
      "     class 1       0.97      0.95      0.96      6742\n",
      "     class 2       0.93      0.88      0.90      5958\n",
      "     class 3       0.92      0.85      0.88      6131\n",
      "     class 4       0.94      0.89      0.91      5842\n",
      "     class 5       0.88      0.82      0.85      5421\n",
      "     class 6       0.95      0.94      0.95      5918\n",
      "     class 7       0.95      0.91      0.93      6265\n",
      "     class 8       0.66      0.93      0.77      5851\n",
      "     class 9       0.90      0.86      0.88      5949\n",
      "\n",
      "    accuracy                           0.90     60000\n",
      "   macro avg       0.91      0.90      0.90     60000\n",
      "weighted avg       0.91      0.90      0.90     60000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "\n",
    "target_names = ['class 0','class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']\n",
    "\n",
    "print(classification_report(y_train, y_train_pred_sgd,target_names=target_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "9ba0eb4b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\lib\\site-packages\\sklearn\\base.py:450: UserWarning: X does not have valid feature names, but OneVsOneClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([5], dtype=uint8)"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.multiclass import OneVsOneClassifier\n",
    "\n",
    "ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))\n",
    "ovo_clf.fit(X_train, y_train)\n",
    "ovo_clf.predict([some_digit])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "a9848e29",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.91545, 0.9131 , 0.92045])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cross_val_score(ovo_clf, X_train, y_train, cv=3, scoring=\"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "7bf3a4ae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.91595, 0.9149 , 0.91845])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))\n",
    "cross_val_score(ovo_clf, X_train_scaled, y_train, cv=3, scoring=\"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "747a4114",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5645,    1,   37,   18,    8,  106,   46,    6,   44,   12],\n",
       "       [   2, 6514,   54,   30,   10,   22,    3,   13,   89,    5],\n",
       "       [  27,   55, 5351,  152,   84,   37,   61,   52,  124,   15],\n",
       "       [  13,   22,  122, 5455,    2,  248,    9,   39,  181,   40],\n",
       "       [   7,   16,   68,    5, 5375,   29,   42,   41,   61,  198],\n",
       "       [  34,   13,   29,  228,   29, 4820,   61,    7,  163,   37],\n",
       "       [  35,    9,   76,    9,   41,   85, 5601,    2,   60,    0],\n",
       "       [   8,   16,   78,   62,   48,   31,    1, 5800,   44,  177],\n",
       "       [  34,   77,   99,  168,   10,  169,   30,   15, 5196,   53],\n",
       "       [  13,   34,   37,   64,  180,   57,    1,  249,   85, 5229]],\n",
       "      dtype=int64)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train_pred_ovo = cross_val_predict(ovo_clf, X_train_scaled, y_train, cv=3)\n",
    "conf_mx = confusion_matrix(y_train, y_train_pred_ovo)\n",
    "conf_mx "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "04bc729b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "     class 0       0.97      0.95      0.96      5923\n",
      "     class 1       0.96      0.97      0.97      6742\n",
      "     class 2       0.90      0.90      0.90      5958\n",
      "     class 3       0.88      0.89      0.89      6131\n",
      "     class 4       0.93      0.92      0.92      5842\n",
      "     class 5       0.86      0.89      0.87      5421\n",
      "     class 6       0.96      0.95      0.95      5918\n",
      "     class 7       0.93      0.93      0.93      6265\n",
      "     class 8       0.86      0.89      0.87      5851\n",
      "     class 9       0.91      0.88      0.89      5949\n",
      "\n",
      "    accuracy                           0.92     60000\n",
      "   macro avg       0.92      0.92      0.92     60000\n",
      "weighted avg       0.92      0.92      0.92     60000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "\n",
    "target_names = ['class 0','class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']\n",
    "print(classification_report(y_train, y_train_pred_ovo,target_names=target_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5b14a12",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "2554eb0a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from scipy.ndimage import interpolation\n",
    "\n",
    "X_aug_down = interpolation.shift(np.array(X_train).reshape(60000,28,28), [0,1,0], cval=0)\n",
    "X_aug_down.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "69472b90",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_aug_up = interpolation.shift(np.array(X_train).reshape(60000,28,28), [0,-1,0], cval=0)\n",
    "X_aug_up.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "875af874",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_aug_right = interpolation.shift(np.array(X_train).reshape(60000,28,28), [0,0,1], cval=0)\n",
    "X_aug_right.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "ee859dc4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 28, 28)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_aug_left = interpolation.shift(np.array(X_train).reshape(60000,28,28), [0,0,-1], cval=0)\n",
    "X_aug_left.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "072f78d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(240000, 28, 28)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_temp = np.concatenate((X_aug_down, X_aug_up, X_aug_right, X_aug_left))\n",
    "X_temp.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "05a4cb09",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(300000, 784)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_aug = np.concatenate((X_train, X_temp.reshape(240000, 784)))\n",
    "X_aug.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "9ea27847",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(300000,)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_aug = np.concatenate((y_train, y_train, y_train, y_train, y_train))\n",
    "y_aug.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "71c214be",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split the train and test set\n",
    "X_train_shift, X_test_shift, y_train_shift, y_test_shift = X_aug[:300000], X_aug[300000:], y_aug[:300000], y_aug[300000:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "de75d836",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5], dtype=uint8)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import SGDClassifier\n",
    "\n",
    "sgd_clf_shift = SGDClassifier(random_state=42)\n",
    "#sgd_clf.fit(X_train, y_train_5)\n",
    "sgd_clf_shift.fit(X_train_shift, y_train_shift) # y_train, not y_train_5\n",
    "sgd_clf_shift.predict([some_digit])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "d070a75d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.84679, 0.79802, 0.81172])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "cross_val_score(sgd_clf_shift, X_train_shift, y_train_shift, cv=3, scoring=\"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "5df7b865",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_stochastic_gradient.py:696: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([0.84325, 0.77361, 0.79561])"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled_shift = scaler.fit_transform(X_train_shift.astype(np.float64))\n",
    "cross_val_score(sgd_clf_shift, X_train_scaled_shift, y_train_shift, cv=3, scoring=\"accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07df50cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.model_selection import cross_val_predict\n",
    "\n",
    "y_train_pred_sgd_shift = cross_val_predict(sgd_clf_shift, X_train_scaled_shift, y_train_shift, cv=3)\n",
    "conf_mx = confusion_matrix(y_train_shift, y_train_pred_sgd_shift)\n",
    "conf_mx "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3c2cda60",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "\n",
    "target_names = ['class 0','class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9']\n",
    "\n",
    "print(classification_report(y_train_shift, y_train_pred_sgd_shift,target_names=target_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88187814",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.multiclass import OneVsOneClassifier\n",
    "\n",
    "ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))\n",
    "ovo_clf.fit(X_train_shift, y_train_shift)\n",
    "ovo_clf.predict([some_digit])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

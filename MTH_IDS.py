import numpy as np
import streamlit as st
import time
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

from sklearn.cluster import MiniBatchKMeans
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def resample(X_train: np.ndarray, y_train: np.ndarray, train_benign: int, seed: int):
    target = 2000
    count = pd.Series(y_train).value_counts()
    n_attack = sum(count) - count[train_benign] #type: ignore
    n_benign = min(count[train_benign], 2*n_attack) #type: ignore
    small = {i: target for i in range(count.size) if i != train_benign and count[i]< target}

    pl = Pipeline([
        ("under", RandomUnderSampler(sampling_strategy={train_benign: n_benign}, random_state=seed)), #type:ignore
        ("smote", SMOTE(sampling_strategy=small, k_neighbors=5, random_state=seed)),#type:ignore
    ])

    result = pl.fit_resample(X_train, y_train)
    if result is None:
        raise RuntimeError("Lỗi dữ liệu rỗng")
    return result

class first_layer:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # vừa tạo đầu vào out-of-fold vừa tính f1-score trung bình của từng base model
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        base = [("XGBoost", xgb.XGBClassifier(random_state = self.seed)),
                ("Random Forest", RandomForestClassifier(random_state = self.seed)),
                ("Extra Trees", ExtraTreesClassifier(random_state = self.seed)),
                #("SVM", LinearSVC()),
                ("Logistic Regression", LogisticRegression(max_iter=2000))]
        w = compute_sample_weight("balanced", y_train)
        w_soft = np.sqrt(w)   

        oof = []
        f_list = []

        with st.status("Train base model", expanded=True, type="step"):
            for name, model in base:
                proba = cross_val_predict(model, X_train, y_train, cv=skf, method='predict_proba', n_jobs=-1, 
                                          params={"sample_weight": w_soft}) # type: ignore
                oof.append(proba)
                pred = proba.argmax(axis=1)
                f_score = round(float(f1_score(y_train, pred, average="macro")), 4)
                print(name, f_score)
                f_list.append(f_score)
                st.write("Xong " + name)

        X_meta = np.hstack(oof)
        best = np.argmax(np.array(f_list))
        meta = clone(base[best][1])

        # trên mô hình meta
        with st.status("Train meta model", type="step"):
            self.base = [model.fit(X_train, y_train, sample_weight=w_soft) for _, model in base]
            self.meta = meta.fit(X_meta, y_train, sample_weight = w_soft)

        st.status("Hoàn thành train", type="step", state='complete')

    def predict(self, X_test: np.ndarray):
        base_pred = [model.predict_proba(X_test) for model in self.base]
        X_meta_test = np.hstack(base_pred)
        return self.meta.predict(X_meta_test)

class second_layer:
    def __init__(self, seed) -> None:
        self.seed = seed

    def optimize(self, X_train, X_opt, y_train, y_opt):
        # ban đầu 2->50
        space  = [Integer(2, 100, name='n_clusters')]
        @use_named_args(space)
        def objective(**params):
            km_cluster = MiniBatchKMeans(batch_size=100, random_state=self.seed, **params)
            n=params['n_clusters']
                
            result = km_cluster.fit_predict(X_train)
            result2 = km_cluster.predict(X_opt)
    
            a=np.zeros(n)
            b=np.zeros(n)
            for v in range(0,n):
                for i in range(0,len(y_train)):
                    if result[i]==v:
                        if y_train[i]==1:
                            a[v]=a[v]+1
                        else:
                            b[v]=b[v]+1
            list1=[]
            list2=[]
            for v in range(0,n):
                if a[v]<=b[v]:
                    list1.append(v)
                else: 
                    list2.append(v)
            for v in range(0,len(y_opt)):
                if result2[v] in list1:
                    result2[v]=0
                elif result2[v] in list2:
                    result2[v]=1
                else:
                    print("-1")
            cm=accuracy_score(y_opt,result2)
            print(str(n)+" "+str(cm))
            return (1-cm)
        t1=time.time()
        res_gp = gp_minimize(objective, space, n_calls=20, random_state=self.seed)
        t2=time.time()
        if res_gp is None: return -1

        print(t2-t1)
        print("Best score=%.4f" % (1-res_gp.fun))
        print("""Best parameters: n_clusters=%d""" % (res_gp.x[0]))
        return res_gp.x[0]

    def fit(self, X_train, X_opt, y_train, y_opt, b=100):
        n = self.optimize(X_train, X_opt, y_train, y_opt)
        if n != -1:
            self.km_cluster = MiniBatchKMeans(n_clusters=n,batch_size=b, random_state=self.seed)
            result = self.km_cluster.fit_predict(X_train)
            a=np.zeros(n)
            b=np.zeros(n)
            for v in range(0,n):
                for i in range(0,len(y_train)):
                    if result[i]==v:
                        if y_train[i]==1:
                            a[v]=a[v]+1
                        else:
                            b[v]=b[v]+1
    
            # list1: cụm ko phải tấn công, list2: cụm tấn công
            self.list1=[]
            self.list2=[]
            for v in range(0,n):
                if a[v]<=b[v]:
                    self.list1.append(v)
                else: 
                    self.list2.append(v)

    def predict(self, X_test):
        result2 = self.km_cluster.predict(X_test)
        for v in range(0, len(X_test)):
            if result2[v] in self.list1:
                result2[v]=0
            elif result2[v] in self.list2:
                result2[v]=1
            else:
                print("-1")
        return result2
    
class MTH_IDS:
    def __init__(self, seed = 0, process_fl = False):
        self.seed = seed
        self.process_fl = process_fl

    def fit(self, X_train, X_opt, y_train, y_opt, train_benign):
        if (self.process_fl):
            X_train, y_train = resample(X_train, y_train, train_benign, self.seed)
        self.n_train = X_train.shape[0]

        start_fl = time.perf_counter()
        self.fl = first_layer(self.seed)
        self.fl.fit(X_train, y_train)
        end_fl = time.perf_counter()
        self.time_fl = round(end_fl - start_fl, 4)

        if (not self.process_fl):
            X_train, y_train = resample(X_train, y_train, train_benign, self.seed)

        y_train = (y_train != train_benign).astype(int)
        y_opt = (y_opt != train_benign).astype(int)

        start_sl = time.perf_counter()
        self.sl = second_layer(self.seed)
        self.sl.fit(X_train, X_opt, y_train, y_opt)
        end_sl = time.perf_counter()
        self.time_sl = round(end_sl - start_sl, 4)

    def predict_binary(self, X_test, train_benign):
        result = self.fl.predict(X_test)
        result = (result != train_benign).astype(int)

        suspect = X_test[result == 0]
        result[result == 0] = self.sl.predict(suspect)

        return result
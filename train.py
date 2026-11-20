import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn import metrics

import MTH_IDS

"""
sửa
- chia X_train, X_test rồi mới tiền xử lý
vì tiền xử lý trước cũng coi như lọt thông tin :)))
- thay phần cross_val_predict từ predict -> predict_proba
ý tưởng là có thêm xác suất xếp các lớp => mô hình meta quyết định tốt hơn
"""
def train_1_ds(df: pd.DataFrame, label_ind: str, benign_label: str, train_rate: float):
    df = df.drop_duplicates()

    y = df[label_ind]
    X = df.drop(columns=label_ind)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # tiền xử lý
    """
    for col_name in X.columns:
        col = X[col_name]
        if col.dtypes == 'str': continue
        X[col_name] = (col - col.mean())/col.std()
    X = X.fillna(0)
    """

    lb = LabelEncoder()
    lb.fit(y)
    y = lb.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y, train_size=train_rate, stratify=y)
    X_opt, X_val, y_opt, y_val = train_test_split(X_test, y_test, train_size=0.5, stratify=y_test)

    mean = X_train.mean(axis = 0)
    std = X_train.std(axis = 0)
    std[std == 0] = 1

    def normalize(X):
        return (X-mean)/std
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    X_opt = normalize(X_opt)
    X_val = normalize(X_val)

    # lớp 1: dự đoán tấn công đã biết
    st.title("Lớp 1: Dự đoán tấn công đã biết")

    benign_le = lb.transform([benign_label])[0] #type: ignore
    
    target = 2000
    count = pd.Series(y_train).value_counts()
    n_attack = sum(count) - count[benign_le] #type: ignore
    n_benign = min(count[benign_le], 2*n_attack) #type: ignore
    small = {i: target for i in range(count.size) if i != benign_le and count[i]< target}

    pl = Pipeline([
        ("under", RandomUnderSampler(sampling_strategy={benign_le: n_benign}, random_state=42)),
        ("smote", SMOTE(sampling_strategy=small, k_neighbors=5, random_state=42)),
    ])
    X_rsp, y_rsp = pl.fit_resample(X_train, y_train)

    f1 = MTH_IDS.first_layer()
    f1.fit(X_rsp, y_rsp)
    y_pred = f1.predict(X_test)
    ac_score=accuracy_score(y_test, y_pred)
    y_true=y_test

    def to_pc(x):
        return round(x*100, 4)
    st.write('Accuracy of meta: '+ str(to_pc(ac_score)) + '%')
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_pred, average='weighted') 
    st.write('Precision of meta: '+(str(round(float(precision), 4))))
    st.write('Recall of meta: '+(str(round(float(recall), 4))))
    st.write('F1-score of meta: '+(str(round(float(fscore), 4))))
    st.code(classification_report(lb.inverse_transform(y_true),lb.inverse_transform(y_pred)))
    cm=confusion_matrix(lb.inverse_transform(y_true),lb.inverse_transform(y_pred))
    f,ax=plt.subplots(figsize=(10, 10))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax, xticklabels=lb.classes_.tolist(), yticklabels=lb.classes_.tolist())
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    st.pyplot(f)

    # lớp 2: dự đoán tấn công chưa biết
    st.title("Dự đoán tấn công chưa biết")
    

    def change_label(y):
        y = np.asarray(y)
        return (y != benign_le).astype(int)

    y_rsp = change_label(y_rsp)
    y_opt_bin = change_label(y_opt)
    y_val_bin = change_label(y_val)

    f2 = MTH_IDS.second_layer()
    f2.fit(X_rsp, X_opt, y_rsp, y_opt_bin)
    result2 = f2.predict(X_val)
    st.code(classification_report(y_val_bin, result2))
    cm=confusion_matrix(y_val_bin,result2)
    acc=metrics.accuracy_score(y_val_bin,result2)
    st.write(str(acc))
    st.code(cm)

    results = []
    st.write('Kiểm tra phát hiện tấn công chưa biết bằng cách bỏ từng loại tấn công')
    for lbi in range(lb.classes_.size):
        if lbi == benign_le: continue
        with st.status("Bỏ kiểu " + lb.classes_[lbi], type="step"):
            t_ind = (y_train != lbi)
            o_ind = (y_opt != lbi)
            X_rmv, y_rmv, X_opt_rmv, y_opt_rmv = X_train[t_ind], y_train[t_ind], X_opt[o_ind], y_opt[o_ind]

            target = 2000
            count = pd.Series(y_rmv).value_counts()
            n_attack = sum(count) - count[benign_le] #type: ignore
            n_benign = min(count[benign_le], 2*n_attack) #type: ignore
            small = {i: target for i in range(lb.classes_.size) if i != benign_le and i != lbi and count[i]< target}
        
            pl = Pipeline([
                ("under", RandomUnderSampler(sampling_strategy={benign_le: n_benign}, random_state=42)),
                ("smote", SMOTE(sampling_strategy=small, k_neighbors=5, random_state=42)),
            ])
            X_rsp, y_rsp = pl.fit_resample(X_rmv, y_rmv)
            y_opt_rmv = change_label(y_opt_rmv)
            y_rsp = change_label(y_rsp)
            f2 = MTH_IDS.second_layer()
            f2.fit(X_rsp, X_opt_rmv, y_rsp, y_opt_rmv)
            result2 = f2.predict(X_val)
            results.append({
                "Lớp bị loại (zero-day)": lb.classes_[lbi],
                "Recall": np.count_nonzero(result2[y_val == lbi]) / np.count_nonzero(y_val == lbi),
                "FAR": np.count_nonzero(result2[y_val == benign_le]) / np.count_nonzero(y_val == benign_le),
            })
    df_result = pd.DataFrame(results).set_index("Lớp bị loại (zero-day)")
    st.dataframe(df_result.style.format("{:.4f}"), use_container_width=True)

def slog(x):  # log1p giữ dấu, vì CIC có giá trị âm
    return np.sign(x) * np.log1p(np.abs(x))

def train_2_ds(train_df: pd.DataFrame, test_df: pd.DataFrame, train_label: str, train_benign, 
               test_label: str, test_benign):
    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()
    
    y = train_df[train_label]
    X = train_df.drop(columns=train_label)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    lb_train = LabelEncoder()
    y = lb_train.fit_transform(y)
    train_benign = lb_train.transform([train_benign])[0] #type: ignore

    X_train, X_opt, y_train, y_opt = train_test_split(X.to_numpy(), y, train_size= 0.95, stratify=y)
    #X_train, X_opt = slog(X_train), slog(X_opt)

    mean = X_train.mean(axis = 0)
    std = X_train.std(axis = 0)
    std[std == 0] = 1
    X_train = (X_train-mean)/std
    X_opt = (X_opt-mean)/std

    y_test = test_df[test_label].to_numpy()
    X_test = test_df.drop(columns=test_label)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
    y_test = (y_test != test_benign).astype(int)
    #X_test = slog(X_test)

    mean = X_test.mean(axis = 0)
    std = X_test.std(axis = 0)
    std[std == 0] = 1
    X_test = (X_test-mean)/std

    target = 2000
    count = pd.Series(y_train).value_counts()
    n_attack = sum(count) - count[train_benign] #type: ignore
    n_benign = min(count[train_benign], 2*n_attack) #type: ignore
    small = {i: target for i in range(count.size) if i != train_benign and count[i]< target}

    pl = Pipeline([
        ("under", RandomUnderSampler(sampling_strategy={train_benign: n_benign}, random_state=42)),
        ("smote", SMOTE(sampling_strategy=small, k_neighbors=5, random_state=42)),
    ])
    X_train, y_train = pl.fit_resample(X_train, y_train)

    model = MTH_IDS.MTH_IDS()
    model.fit(X_train, X_opt, y_train, y_opt, train_benign)
    result = model.predict_binary(X_test, train_benign)
    st.code(classification_report(y_test, result))
    cm=confusion_matrix(y_test,result)
    acc=metrics.accuracy_score(y_test,result)
    st.write('Accuracy : '+ str(round(float(acc)*100, 4)) + '%')
    st.code(cm)


    
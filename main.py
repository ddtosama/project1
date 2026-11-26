import streamlit as st
import pandas as pd
import train as md
from datetime import datetime
import random

ss = st.session_state
if "setting_bool" not in ss:
    ss.setting_bool = False
ss.train_rate = 80 if "train_rate" not in ss else ss.train_rate
ss.seed = None if "seed" not in ss else ss.seed
ss.process_fl = False if "process_fl" not in ss else ss.process_fl
if 'train_df' not in ss:
    ss.train_df = None
if 'test_df' not in ss:
    ss.test_df = None

st.title("Mô hình Demo")

# Chọn 1/2 tập dataset
option_map = {
    True: 'Cùng dataset',
    False: 'Khác dataset'
}

def change_dataset_mode():
    ss.test_df = None
st.segmented_control("Chọn tập huấn luyện/kiểm thử", option_map.keys(), selection_mode='single', default=True, required=True,
                     format_func= lambda x: option_map[x], key="dataset_same", on_change=change_dataset_mode)

# Chọn file
def read_file(mode):
    if mode == "train":
        if ss.train_file is None:
            ss.train_df = None
        else:
            ss.train_df = pd.read_csv(ss.train_file)
    else:
        if ss.test_file is None:
            ss.test_df = None
        else:
            ss.test_df = pd.read_csv(ss.test_file)
if ss.dataset_same:
    st.file_uploader("Dataset", ".csv", on_change=read_file, key="train_file", args=['train'])
else:
    col1, col2 = st.columns(2)
    col1.file_uploader("Tập huấn luyện", ".csv", on_change=read_file, key="train_file", args=['train'])
    col2.file_uploader("Tập kiểm thử", ".csv", on_change=read_file, key="test_file", args=['test'])

st.divider()   

# Nút cài đặt
def change_setting():
    ss.setting_bool = not ss.setting_bool
col1, col2 = st.columns(2)
if ss.train_df is not None:
    col1.selectbox("Chọn cột nhãn tập huấn luyện", ss.train_df.columns, index=ss.train_df.columns.size-1, key="train_label")
    col1.selectbox("Chọn tên nhãn thường", ss.train_df[ss.train_label].unique(), index=0, key="train_benign_label")
if ss.test_df is not None:
    col2.selectbox("Chọn cột nhãn tập kiểm thử", ss.test_df.columns, index=ss.test_df.columns.size-1, key="test_label")
    col2.selectbox("Chọn tên nhãn thường", ss.test_df[ss.test_label].unique(), index=0, key="test_benign_label")
if ss.train_df is not None: 
    st.button("Setting ︾", on_click=change_setting)
if (ss.setting_bool):
    st.slider("Tỉ lệ tập huấn luyện", min_value=0, max_value=100, key="train_rate")
    st.number_input("Seed", min_value=0, value=None, step=1, key="seed")
    st.checkbox("SMOTE + giảm benign từ lớp 1", value=False, key="process_fl")

# Nút nộp
def close_setting():
    ss.setting_bool = False

# nút train
_, col_center, _ = st.columns([4, 1, 4])
if col_center.button("Train", on_click=close_setting, use_container_width=True):
    seed = ss.seed
    if seed is None:
        seed = random.randint(0, 2**32-1)
    if ss.dataset_same:
        if ss.train_df is None:
            st.text("Hãy up file lên")
        else:
            md.train_1_ds(ss.train_df, ss.train_label, ss.train_benign_label, ss.train_rate/100.0)
    else:
        if ss.train_df is None or ss.test_df is None:
            st.text("Hãy up file lên")
        elif ss.train_df.shape[1] != ss.test_df.shape[1]:
            st.text("2 dataset không cùng số đặc trưng")
            st.text(str(ss.train_df.shape[1])+ " "+ str(ss.test_df.shape[1]))
        else:
            record = {
                "Thời điểm": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                "Seed": seed,
                "Tập train": ss.train_file.name,
                "Số hàng tập train": 0,
                "Tập test": ss.test_file.name
            }
            md.train_2_ds(ss.train_df, ss.test_df, ss.train_label, ss.train_benign_label, ss.test_label, 
                          ss.test_benign_label, record, seed, ss.process_fl)
import pandas as pd
import os
import glob

def save_first_5_lines(datasets_path='datasets', sample_path='datasets_sample'):
    os.makedirs(sample_path, exist_ok=True)
    csv_files = glob.glob(os.path.join(datasets_path, '*.csv'))

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df_temp = pd.read_csv(file_path, nrows=5)
            sample_file = os.path.join(sample_path, filename)
            df_temp.to_csv(sample_file, index=False)
            print(f"{filename} 已儲存前5行到 {sample_file}")
        except Exception as e:
            print(f"{filename} 讀取錯誤: {e}")

save_first_5_lines()
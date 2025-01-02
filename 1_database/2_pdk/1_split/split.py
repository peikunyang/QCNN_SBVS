import pandas as pd
import numpy as np

data = pd.read_csv("../0_pkd/pkd", delim_whitespace=True, header=None)
data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
total_rows = len(data_shuffled)
split_point = int(total_rows * 2 / 3)
part1 = data_shuffled.iloc[:split_point]
part2 = data_shuffled.iloc[split_point:]

def save_with_alignment(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.iterrows():
            formatted_row = "{:4} {:5.2f} {:6.2f}".format(row[0], row[1], row[2])
            f.write(formatted_row + '\n')

save_with_alignment(part1, "pkd_train")
save_with_alignment(part2, "pkd_test")



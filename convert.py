import pandas as pd

def load_dat_file(file_path, delimiter=","):
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Save as .csv file
def save_to_csv(df, output_file_path):
    try:
        df.to_csv(output_file_path, index=False)
        print(f"File successfully saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


def convert_dat_to_csv(input_file_path, output_file_path, delimiter=","):
    df = load_dat_file(input_file_path, delimiter)
    if df is not None:
        save_to_csv(df, output_file_path)


for i in range(22):
    if i < 10:
        input_dat_file = f'./data/d0{i}.dat'
        output_csv_file = f'./data/d0{i}.csv'
    else:
        input_dat_file = f'./data/d{i}.dat'
        output_csv_file = f'./data/d{i}.csv'
    convert_dat_to_csv(input_dat_file, output_csv_file, delimiter=',')

for i in range(22):
    if i < 10:
        input_dat_file = f'./data/d0{i}_te.dat'
        output_csv_file = f'./data/d0{i}_te.csv'
    else:
        input_dat_file = f'./data/d{i}_te.dat'
        output_csv_file = f'./data/d{i}_te.csv'
    convert_dat_to_csv(input_dat_file, output_csv_file, delimiter=',')
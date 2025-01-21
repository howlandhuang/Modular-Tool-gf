import os, re
import numpy as np
import pandas as pd
from func.ulti import split_string, ProcessingConfig

class StackProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.basic_info_line_num = 5
        self.type_of_noise = 4

    def pre_process_df(self, single_file):
        df = pd.read_excel(single_file)

        device_info = os.path.basename(single_file)
        device_name, wafer_id, bias_id = split_string(device_info)

        return df, device_name, wafer_id, bias_id

    def check_df_columns(self, df):
        pattern = r"^Die\d+_Sid$"
        num_dies = sum(bool(re.match(pattern, col)) for col in df.columns)
        if int((num_dies + 1) * self.type_of_noise + 1) != df.shape[1]:
            raise ValueError("Check dataframe: noise columns mismatch")
        return num_dies

    def init_chck(self, file_paths):

        dataframes = []
        die_num = None

        for file_path in file_paths:
            df, device_name, wafer_id, bias_id = self.pre_process_df(file_path)
            if die_num is None:
                die_num = self.check_df_columns(df)
            elif die_num != self.check_df_columns(df):
                raise ValueError("All files must have the same number of columns.")

            dataframes.append((device_name, wafer_id, bias_id, df))
        return dataframes, die_num

    def from_selection(self, save_file):
        dataframes, die_num = self.init_chck(self.config.base_path)
        part1 = None
        part2 = None

        for device_name, wafer_id, bias_id, df in dataframes:
            df.insert(0,'Wafer', [f"{wafer_id}"] * df.shape[0])
            df.insert(0,'Device', [f"{device_name}"] * df.shape[0])
            columns_to_remove = ['Wafer', 'Device']
            df.loc[self.basic_info_line_num-1, columns_to_remove] = np.nan

            if part1 is None:
                part1 = df.iloc[:self.basic_info_line_num]
            else:
                part1 = pd.concat([part1, df.iloc[:self.basic_info_line_num]])

            if part2 is None:
                part2 = df.iloc[self.basic_info_line_num:]
                blank_row = pd.DataFrame('', columns=df.columns, index=[0])
            else:
                part2 = pd.concat([part2, blank_row, df.iloc[self.basic_info_line_num+1:]])
            result = pd.concat([part1, part2])

        # output_file = os.path.join(self.config.output_path, f"Stacked_tabel.xlsx")
        output = os.path.join(self.config.output_path, save_file + '.xlsx')
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            result.to_excel(writer, sheet_name='Stacked', index=False, header=True)
            workbook = writer.book
            worksheet = writer.sheets['Stacked']
            header_format = workbook.add_format({'bold': False, 'border': 0})
            for col_num, value in enumerate(result.columns):
                worksheet.write(0, col_num, value, header_format)
            for col_num in range(result.shape[1]):
                worksheet.set_column(col_num + 1, col_num + 1, 13)  # Adding 1 to skip index colum

    def start_from_extract(self):
        pass

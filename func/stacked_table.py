import os, re
import numpy as np
import pandas as pd
from func.ulti import ProcessingConfig, split_wafer_file_name

class StackProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.type_of_noise = 4

    def run_stacking(self, save_file):
        self.dataframes = []
        self.freq = None
        self.die_num = None
        if not self.init_process():
            raise ValueError("Failed to pass initial check") # Would never reach here

        self._stacking_files(save_file)

    def init_process(self):
        for file_path in self.config.base_path:
            self.get_dataframes(file_path)

        self.check_column_match()
        return True

    def get_dataframes(self, single_file):
        df = pd.read_excel(single_file)
        device_name, wafer_id, bias_id = split_wafer_file_name(os.path.basename(single_file))
        self.dataframes.append((device_name, wafer_id, bias_id, df))

    def check_column_match(self):
        shape = None
        for device_name, wafer_id, bias_id, df in self.dataframes:

            # Check column number
            # We don't need to check frequency because we only stack them vertically
            if shape is None:
                shape = df.shape[1]
            elif df.shape[1] != shape:
                raise ValueError("All files must have the same number of columns.")



    def _stacking_files(self, save_file):
        part1 = None
        part2 = None

        for device_name, wafer_id, bias_id, df in self.dataframes:
            df.insert(0,'Wafer', [f"{wafer_id}"] * df.shape[0])
            df.insert(0,'Device', [f"{device_name}"] * df.shape[0])
            columns_to_remove = ['Wafer', 'Device']
            df.loc[self.config.basic_info_line_num-1, columns_to_remove] = np.nan

            if part1 is None:
                part1 = df.iloc[:self.config.basic_info_line_num]
            else:
                part1 = pd.concat([part1, df.iloc[:self.config.basic_info_line_num]])

            if part2 is None:
                part2 = df.iloc[self.config.basic_info_line_num:]
                blank_row = pd.DataFrame('', columns=df.columns, index=[0])
            else:
                part2 = pd.concat([part2, blank_row, df.iloc[self.config.basic_info_line_num+1:]])
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


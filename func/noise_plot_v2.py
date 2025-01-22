import os, re
import pandas as pd
import matplotlib.pyplot as plt
from func.ulti import split_string, ProcessingConfig, remove_outliers
import numpy as np

class PlotProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.basic_info_line_num = 5 # This is the line number where the frequency table starts


    def pre_process_df(self, single_file):
        df = pd.read_excel(single_file)
        df = df.iloc[self.basic_info_line_num:].reset_index(drop=True)
        # Check if we need to remove outliers
        if self.config.filter_outliers_flag:
            df = remove_outliers(df, self.config.filter_threshold, self.config.filter_tolerance)
        device_info = os.path.basename(single_file)
        device_name, wafer_id, bias_id = split_string(device_info)

        return df, device_name, wafer_id, bias_id

    def check_df_columns(self, df):
        '''Check if the number of columns in the dataframe is correct'''
        pattern = r"^Die\d+_Sid$"
        num_dies = sum(bool(re.match(pattern, col)) for col in df.columns)
        if int((num_dies+1)*4+1) != df.shape[1]:
            raise ValueError("Check dataframe: noise columns mismatch")
        return num_dies

    def init_chck(self):
        dataframes = []
        freq = None
        die_num = None
        for file_path in self.config.base_path:
            df, device_name, wafer_id, bias_id = self.pre_process_df(file_path)
            if die_num is None:
                die_num = self.check_df_columns(df)
            elif die_num != self.check_df_columns(df):
                raise ValueError("All files must have the same number of columns.")
            if freq is None:
                freq = df["Frequency"]
            elif (freq != df["Frequency"]).any():
                raise ValueError("All files must have the same frequency range.")
            dataframes.append((device_name, wafer_id, bias_id, df))
        return dataframes, freq, die_num

    def figure_format(self, title):
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.grid(which='both', color='gray', linestyle='-.', linewidth=0.1)
        plt.title(title)
        plt.legend()

    def run_by_site(self, plot_type_list, save_name):
        dataframes, freq, die_num = self.init_chck()

        for plot_type in plot_type_list:
            plt.figure(figsize=(12, 8))
            colors1 = [(r, g, b, 0.2) for r, g, b, _ in plt.cm.tab10(range(10))]  # Semi-transparent
            colors2 = [(r, g, b, 1) for r, g, b, _ in plt.cm.tab10(range(10))]  # Use Tab10 for normal
            for idx, (device_name, wafer_id, bias_id, df) in enumerate(dataframes):
                for die in range(die_num):
                    plt.plot(freq, df[f"Die{die+1}_{plot_type}"], color=colors1[idx], label=f"{device_name}, {wafer_id}, {bias_id}" if die == 0 else "")
                plt.plot(freq, df[f"{plot_type}_med"], color=colors2[idx], label=f"{device_name}, {wafer_id}, {bias_id}, median")
            title = f"{plot_type} by site"
            self.figure_format(title)

            if not self.config.debug_flag:
                plt.savefig(f'{self.config.output_path}/{save_name}_{title.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
                plt.close()
        if self.config.debug_flag:
            plt.show()

    def run_med_only(self, plot_type_list, save_name):
        dataframes, freq, die_num = self.init_chck()
        for plot_type in plot_type_list:
            plt.figure(figsize=(12, 8))
            colors = [plt.cm.tab10(i / 10) for i in range(10)]  # Use Tab10 for normal
            for idx, (device_name, wafer_id, bias_id, df) in enumerate(dataframes):
                plt.plot(freq, df[f"{plot_type}_med"], color=colors[idx], label=f"{device_name}, {wafer_id}, {bias_id}, median")
            title = f"{plot_type} median only"
            self.figure_format(title)
            if not self.config.debug_flag:
                plt.savefig(f'{self.config.output_path}/{save_name}_{title.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
                plt.close()

        if self.config.debug_flag:
            plt.show()

    def save_filtered_result(self):
        for file_path in self.config.base_path:
            df = pd.read_excel(file_path)

            header = df.iloc[:self.basic_info_line_num]  # Preserve the first few lines
            data = df.iloc[self.basic_info_line_num:]   # Data to modify
            data = remove_outliers(data, self.config.filter_threshold, self.config.filter_tolerance)
            modified_df = pd.concat([header, data], ignore_index=True)
            device_info = os.path.basename(file_path)
            device_name, wafer_id, bias_id = split_string(device_info)
            output_file = os.path.join(self.config.output_path, f'{os.path.basename(file_path[:-5])}_filtered_threshold{self.config.filter_threshold}_tolerance{self.config.filter_tolerance}.xlsx')

            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                modified_df.to_excel(writer, sheet_name=bias_id, index=False, header=True)
                workbook = writer.book
                worksheet = writer.sheets[bias_id]
                header_format = workbook.add_format({'bold': False, 'border': 0})
                for col_num, value in enumerate(modified_df.columns):
                    worksheet.write(0, col_num, value, header_format)
                for col_num in range(modified_df.shape[1]):
                    worksheet.set_column(col_num + 1, col_num + 1, 12)  # Adding 1 to skip index column

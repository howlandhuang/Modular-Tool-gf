import os, re
import pandas as pd
import matplotlib.pyplot as plt
from func.ulti import split_wafer_file_name, ProcessingConfig, remove_outliers

class PlotProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def run_plots(self, noise_type_list, fig_type, save_name):
        '''
        noise_type_list: list of plot types to plot
        - ['Sid', 'Sid/id^2', 'Svg', 'Sid*f']

        fig_type:
        - 0: plot by site
        - 1: plot median only
        - 2: plot max only
        - 3: plot min only

        '''
        self.dataframes = []
        self.freq = None
        self.die_num = None
        if not self.init_process(noise_type_list, fig_type):
            raise ValueError("Failed to pass initial check") # Would never reach here

        for noise_type in noise_type_list:
            self._plot_data(noise_type, fig_type, save_name)

    def init_process(self, noise_type_list, fig_type):
        for file_path in self.config.base_path:
            self.get_dataframes(file_path)

        for noise_type in noise_type_list:
            self.check_column_match(noise_type, fig_type)

        return True

    def get_dataframes(self, single_file):
        df = pd.read_excel(single_file)
        df = df.iloc[self.config.basic_info_line_num:].reset_index(drop=True)
        device_name, wafer_id, bias_id = split_wafer_file_name(os.path.basename(single_file))
        if self.config.filter_outliers_flag:
            df = remove_outliers(df, self.config.filter_threshold, self.config.filter_tolerance)
        self.dataframes.append((device_name, wafer_id, bias_id, df))

    def check_column_match(self, noise_type, fig_type):
        shape = None
        for device_name, wafer_id, bias_id, df in self.dataframes:
            # Check frequency
            
            if self.freq is None:
                self.freq = df["Frequency"]
            elif (df["Frequency"] != self.freq).any():
                raise ValueError("All files must have the same frequency range.")

            # Check column number
            current_columns = len([col for col in df.columns if col.endswith(f"_{noise_type}")])
            self.die_num= current_columns
            current_columns += (3 if fig_type in {2, 3} else 1)
            if shape is None:
                shape = current_columns
            elif current_columns != shape:
                raise ValueError("All files must have the same number of columns.")

    def figure_format(self, plt, title):
        # Apply formatting
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.grid(which='both', color='gray', linestyle='-.', linewidth=0.1)
        plt.title(title)
        plt.legend()

    def _plot_data(self, noise_type, fig_type, save_name):
        # Create figure and canvas
        plt.figure(figsize=(12, 8))

        colors = plt.cm.tab10(range(10))

        for idx, (device_name, wafer_id, bias_id, df) in enumerate(self.dataframes):
            if fig_type == 0:
                for die in range(self.die_num):
                    plt.plot(self.freq, df[f"Die{die+1}_{noise_type}"],
                          color=(*colors[idx][:3], 0.1), # Reduced opacity
                          label=f"{device_name}, {wafer_id}, {bias_id}" if die == 0 else "")
                plt.plot(self.freq, df[f"{noise_type}_med"],
                    color=colors[idx],
                    label=f"{device_name}, {wafer_id}, {bias_id}, median")
            elif fig_type == 1:
                plt.plot(self.freq, df[f"{noise_type}_med"],
                    color=colors[idx],
                    label=f"{device_name}, {wafer_id}, {bias_id}, median")
            elif fig_type == 2:
                plt.plot(self.freq, df[f"{noise_type}_min"],
                    color=colors[idx],
                    label=f"{device_name}, {wafer_id}, {bias_id}, min")
            elif fig_type == 3:
                plt.plot(self.freq, df[f"{noise_type}_max"],
                    color=colors[idx],
                    label=f"{device_name}, {wafer_id}, {bias_id}, max")
            else:
                raise ValueError("Invalid fig_type") # Would never reach here

        title = f"{noise_type} {'median only' if fig_type else 'by site'}"
        self.figure_format(plt, title)

        if not self.config.debug_flag:
            plt.savefig(f'{self.config.output_path}/{save_name}_{title.replace("/", "_")}.png',
                      dpi=300, bbox_inches='tight')
            plt.close()
        elif self.config.debug_flag:
            plt.show()


    def save_filtered_result(self):
        for file_path in self.config.base_path:
            df = pd.read_excel(file_path)

            header = df.iloc[:self.config.basic_info_line_num]  # Preserve the first few lines
            data = df.iloc[self.config.basic_info_line_num:]   # Data to modify
            data = remove_outliers(data, self.config.filter_threshold, self.config.filter_tolerance)
            modified_df = pd.concat([header, data], ignore_index=True)
            device_info = os.path.basename(file_path)
            device_name, wafer_id, bias_id = split_wafer_file_name(device_info)
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

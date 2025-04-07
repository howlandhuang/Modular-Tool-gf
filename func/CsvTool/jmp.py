import pandas as pd
import numpy as np
from typing import Dict, Any

def get_std(x: np.ndarray) -> float:
    """Calculate standard deviation with ddof=1."""
    return np.std(x, ddof=1)

def process_array(arr: np.ndarray, decimals: int = 7) -> np.ndarray:
    """Process array with consistent rounding."""
    return np.round(np.array(arr), decimals)

def create_info_dict(file_path: str) -> Dict[str, Any]:
    """Creates a dictionary containing useful information from the multi-level DataFrame."""
    df = pd.read_excel(file_path, sheet_name='Prediction', header=[0,1,2], index_col=0)
    info_dict = {}

    for bias in df.columns.levels[0]:
        # Get bias parameters
        params = df[bias]['Parameters']
        Vd = params['Vd (V)'].iloc[0]
        Vg = params['Vg (V)'].iloc[0]
        new_bias = f"G{Vg}_D{Vd}"
        info_dict[new_bias] = {}

        # Process Id and Gm lists once per bias
        Id_list = process_array(params['Id (A)'])
        Gm_list = process_array(params['gm (S)'])

        for freq in [f for f in df[bias].columns.levels[0] if f != 'Parameters']:
            freq_data = df[bias][freq]

            # Process main measurements
            Sid_list = np.array(freq_data['Raw'])
            Sid_fit_list = np.array(freq_data['Prediction'])

            # Calculate derived values
            Sid_Id_squared_list = Sid_list / (Id_list ** 2)
            Sid_fit_Id_squared_list = Sid_fit_list / (Id_list ** 2)
            Svg_list = Sid_list / (Gm_list ** 2)
            Svg_fit_list = Sid_fit_list / (Gm_list ** 2)

            info_dict[new_bias][freq] = {
                'Vgs': Vg,
                'Vds': Vd,
                'Vgs_Vds': new_bias,
                'Id (A)': np.median(Id_list),
                'Gm (A/V)': np.median(Gm_list),
                'Sid (A^2/Hz)': np.median(Sid_list),
                'Sid_fit (A^2/Hz)': np.median(Sid_fit_list),
                'Sid/Id^2 (1/Hz)': np.median(Sid_Id_squared_list),
                'Id/gm (V)': np.median(Id_list / Gm_list),
                'Svg (V^2/Hz)': np.median(Svg_list),
                'sqrt(Svg) (V/sqrt(Hz))': np.median(np.sqrt(Svg_list)),
                'Sid_fit/Id^2 (1/Hz)': np.median(Sid_fit_Id_squared_list),
                'Svg_fit (V^2/Hz)': np.median(Svg_fit_list),
                'sqrt(Svg_fit) (V/sqrt(Hz))': np.median(np.sqrt(Svg_fit_list)),
                'Std_Id': get_std(Id_list),
                'Std_Gm': get_std(Gm_list),
                'Std_Sid': get_std(Sid_list),
                'Std_Sid_fit': get_std(Sid_fit_list),
                'Std_Sid/Id^2': get_std(Sid_Id_squared_list),
                'Std_Id_Gm': get_std(Id_list / Gm_list),
                'Std_Svg': get_std(Svg_list),
                'Std_sqrt(Svg)': get_std(np.sqrt(Svg_list)),
                'Std_Sid_fit/Id^2': get_std(Sid_fit_Id_squared_list),
                'Std_Svg_fit': get_std(Svg_fit_list),
                'Std_sqrt(Svg_fit)': get_std(np.sqrt(Svg_fit_list)),
                'gm/Id (1/V)': np.median(Gm_list / Id_list)
            }

    return info_dict

def save_info_dict(info_dict: Dict[str, Any], output_file: str ) -> None:
    """Saves the info dictionary data into a single CSV file."""
    first_bias = next(iter(info_dict))
    first_freq = next(iter(info_dict[first_bias]))

    rows = [
        [freq] + [info_dict[bias][freq][calc]
                 for calc in info_dict[first_bias][first_freq].keys()]
        for bias in info_dict
        for freq in info_dict[bias]
    ]

    headers = ['Frequency'] + list(info_dict[first_bias][first_freq].keys())
    df_output = pd.DataFrame(rows, columns=headers)
    df_output.sort_values(['Frequency', 'Vgs'], inplace=True)
    df_output.to_csv(output_file, index=False)



if __name__ == "__main__":
    # Example usage
    file_path = r"C:\Users\hhuang10\Documents\Local_Data\7XYX48621_W10L038\Extract\0_Prediction_NTK1_W10L0.38_TRIAL_NMOS_M_W=10_L=0.38_W#7.xlsx"
    info_dict = create_info_dict(file_path)
    save_info_dict(info_dict, "processed_data_new.csv")
    pass
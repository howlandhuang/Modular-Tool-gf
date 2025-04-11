"""
Noise Plot module for data visualization.
Provides functionality to plot various types of noise measurements and analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
import matplotlib.image as mpimg
import numpy as np
import matplotlib.ticker as ticker

from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from func.ulti import parse_device_info, ProcessingConfig, remove_outliers
from func.NoiseTool.base_processor import BaseProcessor


# Initialize module logger
logger = logging.getLogger(__name__)

class PlotProcessor(BaseProcessor):
    """
    Processor class for creating noise analysis plots.
    Handles different types of noise plots including site-specific and statistical plots.
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize plot processor with configuration.

        Args:
            config: ProcessingConfig object containing processing parameters
        """
        logger.info("Initializing PlotProcessor")
        super().__init__(config)

        # Preload watermark if it exists
        self.watermark = None
        watermark_path = Path(__file__).parent.parent.parent / 'ui' / 'frcmos device.jpg'
        if watermark_path.exists():
            try:
                self.watermark = mpimg.imread(str(watermark_path))
                logger.debug("Watermark loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load watermark: {str(e)}")
                self.watermark = None
        else:
            logger.warning(f"Watermark image not found at {watermark_path}")

        logger.debug("Plot processor initialized successfully")

    def _figure_format(self, plt, title):
        """
        Apply standard formatting to plot figure.

        Args:
            plt: matplotlib.pyplot instance
            title: Plot title
        """
        logger.debug(f"Applying formatting to plot: {title}")

        # Get current axis
        ax = plt.gca()

        # Configure logarithmic scale
        plt.xscale('log')
        plt.yscale('log')

        # Ensure minor ticks are visible
        ax.minorticks_on()

        # Set specific locators for log scale
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        # Configure tick parameters
        ax.tick_params(which='minor', length=4, color='k', width=1.0)
        ax.tick_params(which='major', length=7, color='k', width=1.5)

        ax.grid(False) # Clear any existing grid
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.8)
        ax.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.4, alpha=0.8)

        plt.title(title)
        # Place legend outside the figure at the upper right corner with smaller font
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0,
                  fontsize='small', framealpha=0.9)
        logger.debug("Plot formatting applied")

    def _add_watermark(self, ax):
        """
        Add watermark to the plot if available.

        Args:
            ax: matplotlib axis to add watermark to
        """
        if self.watermark is None:
            return

        try:
            # Create offset image with transparency
            imagebox = OffsetImage(self.watermark, zoom=0.3, alpha=0.5)

            # Position in data coordinates
            xlimits = ax.get_xlim()
            ylimits = ax.get_ylim()

            # Position watermark at bottom-left
            x_pos = xlimits[0] * 10
            y_pos = ylimits[0] * 3

            # Create and add annotation box
            ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False, pad=0)
            ax.add_artist(ab)
            logger.debug("Watermark added to plot")
        except Exception as e:
            logger.warning(f"Failed to add watermark to plot: {str(e)}")

    def _plot_data(self, noise_type, freq_type, fig_type, save_name):
        """
        Create and save a single plot.

        Args:
            noise_type: Type of noise to plot
            freq_type: Frequency type
            fig_type: Plot type
            save_name: Base name for saving plot
        """
        logger.info(f"Creating plot for {noise_type} with {fig_type}")
        try:
            # Create figure with wider width to accommodate legend
            fig = plt.figure(figsize=(14, 8))
            colors = plt.cm.tab10(range(10))
            logger.debug("Figure created")

            # Check if we need to multiply by frequency
            original_noise_type = noise_type
            multiply_by_freq = False

            if noise_type == 'Svg*f':
                noise_type = 'Svg'
                multiply_by_freq = True
                logger.debug("Will multiply Svg data by frequency")

            # Plot data for each file
            for idx, (info, df) in enumerate(self.dataframes):
                logger.debug(f"Plotting data for {info}")
                freq = df["Frequency"]
                if freq_type == '1/f':
                    freq = 1 / freq
                # Get columns for current noise type
                current_columns = len([col for col in df.columns if col.endswith(f"_{noise_type}")])

                # Format label on two lines to save horizontal space
                label_base = f"{info['device_name']}_W{info['width']}xL{info['length']},\n{info['lot_id']}W{info['wafer_id']}_{info['bias_id']}"

                if fig_type == 'by_site':
                    # Plot individual die data with reduced opacity
                    logger.debug("Creating site plot with individual dies")
                    for die in range(current_columns):
                        die_data = df[f"Die{die+1}_{noise_type}"]
                        if multiply_by_freq:
                            die_data = die_data * df["Frequency"]
                        plt.plot(freq, die_data,
                              color=(*colors[idx][:3], 0.1),
                              label=label_base if die == 0 else "")
                    # Plot median with full opacity
                    median_data = df[f"{noise_type}_med"]
                    if multiply_by_freq:
                        median_data = median_data * df["Frequency"]
                    plt.plot(freq, median_data,
                        color=colors[idx],
                        label=f"{label_base}, median")
                elif fig_type == 'median_only':
                    # Plot median only
                    logger.debug("Creating median-only plot")
                    median_data = df[f"{noise_type}_med"]
                    if multiply_by_freq:
                        median_data = median_data * df["Frequency"]
                    plt.plot(freq, median_data,
                        color=colors[idx],
                        label=f"{label_base}, median")
                elif fig_type == 'min_only':
                    # Plot minimum only
                    logger.debug("Creating minimum-only plot")
                    min_data = df[f"{noise_type}_min"]
                    if multiply_by_freq:
                        min_data = min_data * df["Frequency"]
                    plt.plot(freq, min_data,
                        color=colors[idx],
                        label=f"{label_base}, min")
                elif fig_type == 'max_only':
                    # Plot maximum only
                    logger.debug("Creating maximum-only plot")
                    max_data = df[f"{noise_type}_max"]
                    if multiply_by_freq:
                        max_data = max_data * df["Frequency"]
                    plt.plot(freq, max_data,
                        color=colors[idx],
                        label=f"{label_base}, max")
                else:
                    logger.error(f"Invalid figure type: {fig_type}")
                    raise ValueError("Invalid fig_type")

            # Format and save plot
            title = original_noise_type + "_" + fig_type
            self._figure_format(plt, title)

            # Adjust layout to make room for the legend
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)  # Adjust right margin to make space for legend
            # Add watermark
            self._add_watermark(plt.gca())

            if not self.config.debug_flag:
                output_path = f'{self.config.output_path}/{save_name}_{title.replace("/", "_").replace("*", "x").replace("^", "_")}_0001.png'
                logger.info(f"Saving plot to: {output_path}")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.debug("Plot saved and closed")
            else:
                logger.debug("Showing plot in debug mode")
                plt.show()

            logger.info("Plot creation completed successfully")
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            raise

    def run_plots(self, noise_type_list, freq_type, fig_type, save_name):
        """
        Create and save noise analysis plots.

        Args:
            noise_type_list: List of noise types to plot ['Sid', 'Sid/id^2', 'Svg', 'Sid*f', 'Svg_norm', 'Svg*f']
            freq_type: Frequency type ['f', '1/f']
            fig_type: Plot type
            save_name: Base name for saving plot files
        """
        logger.info(f"Starting plot generation for noise types: {noise_type_list}, freq type: {freq_type}")
        logger.debug(f"Plot type: {fig_type}, Save name: {save_name}")

        # Load and prepare data
        if not self.load_data(for_plot=True, noise_type=noise_type_list):
            logger.error("Failed to load data for plot generation")
            raise ValueError("Failed to load data for plotting")

        # Create plots for each noise type
        for noise_type in noise_type_list:
            logger.info(f"Creating plot for noise type: {noise_type}")
            self._plot_data(noise_type, freq_type, fig_type, save_name)

        logger.info("Plot generation completed successfully")

    def save_filtered_result(self, noise_type_list):
        """Save filtered data results to Excel files."""
        logger.info("Starting filtered result export")
        try:
            for file_path in self.config.base_path:
                logger.debug(f"Processing file: {file_path}")

                # Read original file (we need both header and data)
                df = pd.read_excel(file_path)
                logger.debug(f"Read file with shape: {df.shape}")

                # Preserve header and apply filtering to data
                header = df.iloc[:self.config.basic_info_line_num]
                data = df.iloc[self.config.basic_info_line_num:]
                logger.debug("Applying outlier filtering")
                data, removed_data = remove_outliers(data, self.config.filter_threshold, self.config.filter_tolerance, noise_type_list)
                modified_df = pd.concat([header, data], ignore_index=True)
                logger.debug(f"Final DataFrame shape: {modified_df.shape}")

                # Generate output filename
                device_info = os.path.basename(file_path)
                result = parse_device_info(device_info)
                output_file = os.path.join(
                    self.config.output_path,
                    f'{os.path.basename(file_path[:-5])}_filtered_threshold{self.config.filter_threshold}_tolerance{self.config.filter_tolerance}.xlsx'
                )
                logger.info(f"Saving filtered result to: {output_file}")

                # Save using the base class method
                with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                    modified_df.to_excel(writer, sheet_name=result['bias_id'], index=False)
                    removed_data.to_excel(writer, sheet_name='filtered', index=False)

                logger.debug(f"Completed processing file: {file_path}")

            logger.info("Filtered result export completed successfully")
        except Exception as e:
            logger.error(f"Error saving filtered results: {str(e)}")
            raise

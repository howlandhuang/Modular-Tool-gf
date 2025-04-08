"""
Noise Plot module for data visualization.
Provides functionality to plot various types of noise measurements and analysis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
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
        logger.debug("Plot processor initialized successfully")

    def _figure_format(self, plt, title):
        """
        Apply standard formatting to plot figure.

        Args:
            plt: matplotlib.pyplot instance
            title: Plot title
        """
        logger.debug(f"Applying formatting to plot: {title}")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.grid(which='both', color='gray', linestyle='-.', linewidth=0.1)
        plt.title(title)
        # Place legend outside the figure at the upper right corner with smaller font
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0,
                  fontsize='small', framealpha=0.9)
        logger.debug("Plot formatting applied")

    def _plot_data(self, noise_type, fig_type, save_name):
        """
        Create and save a single plot.

        Args:
            noise_type: Type of noise to plot
            fig_type: Plot type indicator
            save_name: Base name for saving plot
        """
        logger.info(f"Creating plot for {noise_type} with type {fig_type}")
        try:
            # Create figure with wider width to accommodate legend
            fig = plt.figure(figsize=(14, 8))
            colors = plt.cm.tab10(range(10))
            logger.debug("Figure created")

            # Plot data for each file
            for idx, (device_name, lot_id, wafer_id, bias_id, df) in enumerate(self.dataframes):
                logger.debug(f"Plotting data for {device_name} - {lot_id} - {wafer_id} - {bias_id}")
                freq = df["Frequency"]

                # Get columns for current noise type
                current_columns = len([col for col in df.columns if col.endswith(f"_{noise_type}")])

                # Format label on two lines to save horizontal space
                label_base = f"{device_name}, {lot_id}\n{wafer_id}, {bias_id}"

                if fig_type == 0:
                    # Plot individual die data with reduced opacity
                    logger.debug("Creating site plot with individual dies")
                    for die in range(current_columns):
                        plt.plot(freq, df[f"Die{die+1}_{noise_type}"],
                              color=(*colors[idx][:3], 0.1),
                              label=label_base if die == 0 else "")
                    # Plot median with full opacity
                    plt.plot(freq, df[f"{noise_type}_med"],
                        color=colors[idx],
                        label=f"{label_base}, median")
                elif fig_type == 1:
                    # Plot median only
                    logger.debug("Creating median-only plot")
                    plt.plot(freq, df[f"{noise_type}_med"],
                        color=colors[idx],
                        label=f"{label_base}, median")
                elif fig_type == 2:
                    # Plot minimum only
                    logger.debug("Creating minimum-only plot")
                    plt.plot(freq, df[f"{noise_type}_min"],
                        color=colors[idx],
                        label=f"{label_base}, min")
                elif fig_type == 3:
                    # Plot maximum only
                    logger.debug("Creating maximum-only plot")
                    plt.plot(freq, df[f"{noise_type}_max"],
                        color=colors[idx],
                        label=f"{label_base}, max")
                else:
                    logger.error(f"Invalid figure type: {fig_type}")
                    raise ValueError("Invalid fig_type")

            # Format and save plot
            title = f"{noise_type} {'median only' if fig_type else 'by site'}"
            self._figure_format(plt, title)

            # Adjust layout to make room for the legend
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)  # Adjust right margin to make space for legend

            # Add watermark to bottom-left corner
            try:
                from pathlib import Path
                import matplotlib.image as mpimg
                from matplotlib.offsetbox import OffsetImage, AnnotationBbox

                # Get the watermark path
                watermark_path = Path(__file__).parent.parent.parent / 'ui' / 'frcmos device.jpg'
                logger.debug(f"Loading watermark from: {watermark_path}")

                if watermark_path.exists():
                    # Load the image
                    watermark = mpimg.imread(str(watermark_path))

                    # Create offset image with transparency
                    imagebox = OffsetImage(watermark, zoom=0.3, alpha=0.5)  # Adjust zoom and alpha as needed

                    # Position in data coordinates (use ax.get_xlim() and ax.get_ylim())
                    ax = plt.gca()
                    xlimits = ax.get_xlim()
                    ylimits = ax.get_ylim()

                    # Position watermark at bottom-left (10% from edge)
                    x_pos = xlimits[0] + 10
                    y_pos = ylimits[0] * 10

                    # Create annotation box
                    ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False, pad=0)
                    ax.add_artist(ab)

                    logger.debug("Watermark added to plot")
                else:
                    logger.warning(f"Watermark image not found at {watermark_path}")
            except Exception as e:
                logger.warning(f"Failed to add watermark to plot: {str(e)}")

            if not self.config.debug_flag:
                output_path = f'{self.config.output_path}/{save_name}_{title.replace("/", "_").replace("*", "x")}.png'
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

    def run_plots(self, noise_type_list, fig_type, save_name):
        """
        Create and save noise analysis plots.

        Args:
            noise_type_list: List of noise types to plot ['Sid', 'Sid/id^2', 'Svg', 'Sid*f']
            fig_type: Plot type indicator
                     0: plot by site
                     1: plot median only
                     2: plot min only
                     3: plot max only
            save_name: Base name for saving plot files
        """
        logger.info(f"Starting plot generation for noise types: {noise_type_list}")
        logger.debug(f"Plot type: {fig_type}, Save name: {save_name}")

        # Load and prepare data
        if not self.load_data(for_plot=True, noise_type=noise_type_list):
            logger.error("Failed to load data for plot generation")
            raise ValueError("Failed to load data for plotting")

        # Create plots for each noise type
        for noise_type in noise_type_list:
            logger.info(f"Creating plot for noise type: {noise_type}")
            self._plot_data(noise_type, fig_type, save_name)

        logger.info("Plot generation completed successfully")

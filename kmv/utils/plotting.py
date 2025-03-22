import dearpygui.dearpygui as dpg
import random
import numpy as np


class Plotter:
    def __init__(self, window_title="Plotter", window_width=1000, window_height=600):
        self.plots = {}   # Dictionary to track plot data
        self.plot_count = 0  # Counter for dynamic plots
        self.visible_points = 200  # Number of recent points to show when auto-fitting
        self.plot_groups = {}  # Dictionary to track plot groups
        self.group_index_mappings = {}
        
        dpg.create_context()
        # Create a main window to hold all plots
        with dpg.window(label=window_title, tag="main_window"):
            # Create a horizontal group for columns (plot groups)
            dpg.add_group(tag="plot_columns", horizontal=True)
            
        dpg.create_viewport(title=window_title, width=window_width, height=window_height)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)

    def add_plot_group(self, group_name, index_mapping=None, y_axis_min=None, y_axis_max=None):
        """Add a new group (column) for plots."""
        if group_name in self.plot_groups:
            return  # Group already exists
            
        # Create a vertical group (column) for this group's plots
        with dpg.group(parent="plot_columns", tag=f"group_column_{group_name}"):
            # Add a title for the group
            
            dpg.add_text(f"== {group_name.upper()} ==")
            
        self.plot_groups[group_name] = []
        if index_mapping is not None:
            self.group_index_mappings[group_name] = index_mapping
            for idx, plot_name in index_mapping.items():
                self.add_plot(plot_name, group=group_name, y_axis_min=y_axis_min, y_axis_max=y_axis_max)
        return group_name

    def add_plot(self, plot_name, x_label="Total Sim Time", y_label="y", y_axis_min=None, y_axis_max=None, group=None):
        """Add a new plot, optionally to a specific group."""
        # If no group specified, use default
        if group is None:
            group = "default"
            
        # Create the group if it doesn't exist
        if group not in self.plot_groups:
            self.add_plot_group(group)
            
        # Add the plot to the specified group
        parent_tag = f"group_column_{group}"
        
        # Add a new plot inside the group column
        with dpg.group(parent=parent_tag, tag=f"plot_container_{plot_name}"):
            with dpg.plot(label=plot_name, width=500, height=200, tag=f"plot_{plot_name}"):
                dpg.add_plot_legend()
                # Create the axes with unique tags.
                dpg.add_plot_axis(dpg.mvXAxis, label=x_label, tag=f"x_axis_{plot_name}")
                dpg.add_plot_axis(dpg.mvYAxis, label=y_label, tag=f"y_axis_{plot_name}")
                if y_axis_min is not None and y_axis_max is not None:
                    dpg.set_axis_limits(f"y_axis_{plot_name}", y_axis_min, y_axis_max)
                else:
                    dpg.set_axis_limits_auto(f"y_axis_{plot_name}")
                # Create an empty line series to be updated later.
                dpg.add_line_series([], [], label=plot_name, parent=f"y_axis_{plot_name}", tag=f"series_{plot_name}")
                
            dpg.add_checkbox(label="Auto-fit x-axis limits", tag=f"auto_fit_checkbox_x_axis_{plot_name}", default_value=True)

        # Initialize empty data lists for the new plot.
        self.plots[plot_name] = {"x_data": [], "y_data": [], "series_tag": f"series_{plot_name}", "group": group}
        self.plot_groups[group].append(plot_name)
        print(f"Added plot: {plot_name} to group: {group}")

    def update_plot(self, plot_name, x, y):
        if plot_name in self.plots:
            self.plots[plot_name]["x_data"].append(x)
            self.plots[plot_name]["y_data"].append(y)
            # Update the corresponding line series.
            dpg.set_value(
                self.plots[plot_name]["series_tag"],
                [self.plots[plot_name]["x_data"], self.plots[plot_name]["y_data"]]
            )
            if dpg.get_value(f"auto_fit_checkbox_x_axis_{plot_name}"):
                x_data = self.plots[plot_name]["x_data"]
                # If we have fewer points than the window size, show all points
                if len(x_data) <= self.visible_points:
                    dpg.fit_axis_data(f"x_axis_{plot_name}")
                else:
                    # Otherwise, show only the most recent points
                    x_min = x_data[-self.visible_points]
                    x_max = x_data[-1]
                    # Add a small margin (5% of range)
                    margin = (x_max - x_min) * 0.05
                    dpg.set_axis_limits(f"x_axis_{plot_name}", x_min - margin, x_max + margin)
            else:
                dpg.set_axis_limits_auto(f"x_axis_{plot_name}")
            # dpg.set_axis_limits_auto(f"y_axis_{plot_name}")
        else:
            print(f"Plot '{plot_name}' not found!")

    def update_plot_group(self, group_name, x_value, y_values):
        if group_name in self.group_index_mappings:
            index_mapping = self.group_index_mappings[group_name]
            for i, y_value in enumerate(y_values):
                # breakpoint()
                plot_name = index_mapping[i]
                self.update_plot(plot_name, x_value, y_value)
                
    def render_frame(self):
        dpg.render_dearpygui_frame()
    
    def start(self):
        dpg.show_viewport()
        
    def close(self):
        dpg.destroy_context()



if __name__ == "__main__":
    # Example usage:
    plotter = Plotter("Robot Simulation", 1200, 800)
    
    # Add plots in different groups
    plotter.add_plot("joint1_position", y_label="Position", group="Joints")
    plotter.add_plot("joint1_velocity", y_label="Velocity", group="Joints")
    
    plotter.add_plot("reward_total", y_label="Total Reward", group="Rewards")
    plotter.add_plot("reward_height", y_label="Height Reward", group="Rewards")
    
    plotter.add_plot("torso_height", y_label="Height", group="Body")
    plotter.add_plot("head_height", y_label="Height", group="Body")
    
    # Simulate data updates
    for i in range(100):
        time_val = i * 0.1
        plotter.update_plot("joint1_position", time_val, np.sin(time_val))
        plotter.update_plot("joint1_velocity", time_val, np.cos(time_val))
        plotter.update_plot("reward_total", time_val, 0.8 + 0.2 * np.sin(time_val))
        plotter.update_plot("reward_height", time_val, 0.5 + 0.5 * np.sin(time_val))
        plotter.update_plot("torso_height", time_val, 1.0 + 0.1 * np.sin(time_val))
        plotter.update_plot("head_height", time_val, 1.5 + 0.1 * np.sin(time_val))
        plotter.render_frame()
        
    plotter.start()

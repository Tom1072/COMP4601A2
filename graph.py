import csv
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import defaultdict
# from collections.abc import Iterable

col_names = ["mae", "neighborhood-size", "sim-threshold", "absolute-sim-threshold", "time"]

class GraphController:
    def __init__(self):
        self.graphs = defaultdict(list)

    def add_graph(self, dimension, support_axis_types, data):
        """
        Add a graph that contains 2 data sets to print 
        
        Keyword arguments:
        x_type -- the data name/type of the x axis
        y_type -- the data name/type of the y axis
        data   -- the full data set
        """
        graph = Graph(dimension, support_axis_types)
        min_mae = float("inf")
        min_list = []

        for row in data:
            data_list = []

            for t in support_axis_types:
                data_list.append(row[t])

            if row["mae"] < min_mae:
                min_mae = row["mae"]
                min_list = [[row[t] for t in support_axis_types]]
            elif row["mae"] < min_mae:
                min_list.append([row[t] for t in support_axis_types])

            graph.add_data(data_list)
        
        for m in min_list:
            graph.add_min_mae_data(m)
        self.graphs[dimension].append(graph)


    def display_graph(self, window_title=None, grid=True):
        """
        Plot a graph to matplotlib figure
        
        Keyword arguments:
        x_right_limit -- The right limit of the x axis on the plot
        grid          -- Indicate whether to show grid
        """
        fig = plt.figure()
        for i, g in enumerate(self.graphs[2]):
            sp = fig.add_subplot(len(self.graphs[2]), 1, i+1)
            sp.set_xlabel(g.labels[0])
            sp.set_ylabel(g.labels[1])
            # sp.set_xlim(left=0)
            # sp.draw()
            sp.plot(*g.axes)
            sp.plot(*g.min_list, "ro")
            sp.grid(grid)
            fig.tight_layout(pad=2)

        fig.canvas.manager.set_window_title(window_title + " 2D")

        for i, g in enumerate(self.graphs[3]):
            fig = plt.figure()
            sp = fig.add_subplot(111, projection='3d')
            # X, Y, Z = np.asarray(g.axes[0]), np.asarray(g.axes[1]), np.asarray(g.axes[2])
            sp.set_xlabel(g.labels[0])
            sp.set_ylabel(g.labels[1])
            sp.set_zlabel(g.labels[2])
            sp.plot_trisurf(*g.axes)
            sp.scatter(*g.min_list, color="r")
            fig.canvas.manager.set_window_title(window_title + " 3D")


        # fig.set_size_inches(20, 10)
        # plt.subplots_adjust(hspace=0.5)

        fig.show()


class Graph:
    def __init__(self, dimension, labels):
        self.labels = labels
        self.dimension = dimension
        self.axes = [[] for _ in range(dimension)]
        self.min_list = [[] for _ in range(dimension)]

    def add_data(self, data_list):
        for i in range(len(data_list)):
            self.axes[i].append(data_list[i])
    
    def add_min_mae_data(self, min_list):
        for i in range(len(min_list)):
            self.min_list[i].append(min_list[i])
        # self.min_list = min_list
        # self.ax.set_title(self.title)
        # self.ax.plot(self.x_axis, self.y_axis, color="C0")


class JSONParser:
    def __init__(self, filename, is_user_based):
        self.filename = filename
        self.index_map = {n: i for i, n in enumerate(col_names)}
        self.graphs = []
        self.graph_controller = GraphController()
        self.data = self.parse_json(is_user_based)


    def parse_json(self, is_user_based):
        """
        Parse the CSV file specified in the class's filename variable
        """
        try:
            with open(self.filename, 'r') as json_file:
                return json.load(json_file)["user-based" if is_user_based else "item-based"]
        except FileNotFoundError:
            print("Error: Input file doesn't exist")
            return {}
    

    def print_data(self):
        print(self.data)
    
    
    def plot_data(self, window_title):
        self.graph_controller.display_graph(window_title)
    

    def add_graph(self, types, data_key):
        '''
        Takes a pair of data type to add a graph on x and y axis.
        Default of x axis will be "interval".
        E.g. y_axis = "gas" and x_axis = "interval" 
        '''
        self.graph_controller.add_graph(len(types), types, self.data[data_key])

    
def plot_data(window_title, filename, is_user_based, data_names_2d, data_names_3d=[]):
    """
    Plot the data in the filename with a list of subplot data names
    All names: ["interval", "gas", "brake","speed", "object", "object_speed","distance", "skid"]
    """
    json_reader = JSONParser(filename, is_user_based)
    if not json_reader.data:
        return

    for n in data_names_2d:
        json_reader.add_graph((n, "mae"), n)

    for k, n1, n2 in data_names_3d:
        json_reader.add_graph((data_names_2d[n1], data_names_2d[n2], "mae"), k)

    json_reader.plot_data(window_title)


if __name__ == "__main__":
    # Put all graphs want to plot here
    # plot_data("csv_files/ACC_Manual_preemption.csv", ["gas", "brake", "speed"])
    plot_data(
        "Item-based",
        "./assignment2-results-Thang.json",
        False,
        ["neighborhood-size", "sim-threshold", "absolute-sim-threshold"],
        [("size-and-threshold", 0, 1), ("size-and-absolute-threshold", 0, 2)], # comment/uncomment to add/remove 3D graphs
    )
    plot_data(
        "User-based",
        "./assignment2-results-Thang.json",
        True,
        ["neighborhood-size", "sim-threshold", "absolute-sim-threshold"],
        [("size-and-threshold", 0, 1), ("size-and-absolute-threshold", 0, 2)], # comment/uncomment to add/remove 3D graphs
    )

    plt.show() # Don't modify this and please do only one show
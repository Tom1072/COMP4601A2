import csv
import json
import matplotlib.pyplot as plt
from collections.abc import Iterable

col_names = ["mae", "neighborhood-size", "sim-threshold", "absolute-sim-threshold", "time"]
col_types = [int, int, int, float, lambda x: bool(int(x)), float, lambda d: float(d), lambda x: bool(int(x))]
INTERVAL = 50

class GraphController:
    def __init__(self):
        self.graphs = []

    def add_graph(self, x_type, y_type, data):
        """
        Add a graph that contains 2 data sets to print 
        
        Keyword arguments:
        x_type -- the data name/type of the x axis
        y_type -- the data name/type of the y axis
        data   -- the full data set
        """
        graph = Graph2D(x_type, y_type)

        print(x_type, y_type)
        for row in data[x_type]:
            x, y = row[x_type], row[y_type]
            graph.add_data(x, y)

        self.graphs.append(graph)


    def display_graph(self, grid=False):
        """
        Plot a graph to matplotlib figure
        
        Keyword arguments:
        x_right_limit -- The right limit of the x axis on the plot
        grid          -- Indicate whether to show grid
        """
        fig, subplots = plt.subplots(len(self.graphs))

        if not isinstance(subplots, Iterable):
            subplots = [subplots] 
            
        for i, sp in enumerate(subplots):
            g = self.graphs[i]
            sp.plot(g.x_axis, g.y_axis)
            sp.grid(grid)
            sp.set_xlabel(g.x_label)
            sp.set_ylabel(g.y_label)
            sp.set_xlim(left=0)
        fig.tight_layout(pad=2)
        # fig.set_size_inches(20, 10)
        fig.show()
        # plt.subplots_adjust(hspace=0.1)

class Graph2D:
    def __init__(self, x_label, y_label):
        self.x_label = x_label
        self.y_label = y_label
        self.x_axis, self.y_axis = [], []

    def add_data(self, x, y):
        """
        Add data to the graph
        
        Keyword arguments:
        x -- the x coordinate of the data point
        y -- the y coordinate of the data point
        """
        
        if y >= 9999:
            return

        self.x_axis.append(x)
        self.y_axis.append(y)
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
    

    def str_row_to_types(self, row):
        """
        Convert a parsed str array to corresponding types
        """
        if len(row) > len(col_names):
            return []

        return tuple(col_types[i](val) for i, val in enumerate(row))
    

    def print_data(self):
        print(self.data)
    
    
    def plot_data(self):
        self.graph_controller.display_graph()
    

    def add_graph(self, x_type: str, y_type="mae"):
        '''
        Takes a pair of data type to add a graph on x and y axis.
        Default of x axis will be "interval".
        E.g. y_axis = "gas" and x_axis = "interval" 
        '''
        self.graph_controller.add_graph(x_type, y_type, self.data)

    
def plot_data(filename, data_names: list[str]):
    """
    Plot the data in the filename with a list of subplot data names
    All names: ["interval", "gas", "brake","speed", "object", "object_speed","distance", "skid"]
    """
    json_reader = JSONParser(filename, data_names)
    if not json_reader.data:
        return

    for n in data_names:
        json_reader.add_graph(n)

    json_reader.plot_data()


if __name__ == "__main__":
    # Put all graphs want to plot here
    # plot_data("csv_files/ACC_Manual_preemption.csv", ["gas", "brake", "speed"])
    plot_data("./assignment2-results-Thang.json", ["neighborhood-size", "sim-threshold", "absolute-sim-threshold"])

    plt.show() # Don't modify this and please do only one show
import numpy as np
import matplotlib.pyplot as plt
#from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
import pandas as pd
from statistics import mean

class Simulation():
    """
            A class to represent a simple simulator of an epidemic

    """

    def __init__(self, m=40, n=25, r=6, k=20,
                 alpha_infected=0.01, alpha_recovered=0,
                 beta_recover=0.05, beta_death=0.005,
                 gamma=0.075, N=100):
        self.m = m
        self.n = n
        self.r = r
        self.k = k
        self.alpha_infected = alpha_infected
        self.alpha_recovered = alpha_recovered
        self.beta_recover = beta_recover
        self.beta_death = beta_death
        self.gamma = gamma
        self.N = N
        self.grid_points = self._create_grid_points()
        self.individuals = self._individuals_class()
        self.connections = self._initial_connections()
        self.results_df = self._record_results()

    def __repr__(self):
        result = f"Simulation with values: " \
                 f"     {(self.m * self.n)} people " \
                 f"     {int((self.m * self.n * self.k) / 2)} connections " \
                 f"     {self.N} days "\
                 f"& Probabilities: " \
                 f"     infected={self.alpha_infected} , " \
                 f"     spread={self.gamma}, " \
                 f"     recovery= {self.beta_recover}, " \
                 f"     death, {self.beta_death}"
        return result

    def __str__(self):
        return f"Simulation of {(self.m * self.n)} people over {self.N} days"

    def _create_grid_points(self):
        #inverting rows and columns for plot
        return np.array([(x,y) for x in range(self.m) for y in range(self.n)])

    # create a lis of objects relating to the individuals in the simulation
    def _individuals_class(self):
        return [Individual(i, coords, self.alpha_infected,
                            self.alpha_recovered, self.beta_recover, self.beta_death)
                    for i, coords in enumerate(self.grid_points)]

    def _record_results(self):
        # separate co-ordinates for plot function
        df = pd.DataFrame(data=self.grid_points, columns=["x_coord", "y_coord"])
        df["state_0"] = [self.individuals[i].state for i in range(len(self.grid_points))]
        return df

    def _initial_connections(self):
        #method below more efficient than randomly chosing points to loop through

        #KDTree provides an index of k dimensional points which can be queried  for the pairs that are within a maximum distance
        #changed to this method from scipy pdist to avoid duplicate pairs (i.e. 0,1 and 1,0)
        points_tree = KDTree(self.grid_points)
        points_in_radius = np.array(list(points_tree.query_pairs(self.r)))
        #points in radius contains the index of all possible connections within r distance
        #then randomly choose pairs of points for the required number of connections
        conn_elements = np.random.choice(len(points_in_radius), int((self.m * self.n * self.k) / 2), replace=False)
        # ^random indicies of connections
        #use the chosen indices to find inidicies of pairs
        pairs_indices = np.take(points_in_radius, conn_elements, 0)
        ##^^first column is the index of point1 and second column is index of point 2
        #create list containing pairs of points for plot
        connections = [list(pairs_indices[i, :]) for i in range(len(pairs_indices))]

        ##update the connections in the Individual objects
        for i in range(len(connections)):
            #object relating to point 1 & 2
            p1 = self.individuals[connections[i][0]]
            p2 = self.individuals[connections[i][1]]
            #update instance of each individual class here
            p1._add_connection(p2.grid_coords)
            p2._add_connection(p1.grid_coords)
            #also want to add index so easier to update states
            p1._add_connection_index(p2.id)
            p2._add_connection_index(p1.id)

        return connections

    @classmethod
    def averaged_chart(cls, Nsim, settings={}):
        # Needs to call Simulation Nsim times and average the curves from the simulations
        # store each simulation in a list
        simulations = [Simulation(**settings) for i in range(Nsim)]
        # for each simulation object run model
        [s.run() for s in simulations]
        # now can average results
        pdList = [simulations[i].summary_time for i in range(Nsim)]
        all_results = pd.concat(pdList)

        # group by index to produce means across each day
        averages = all_results.groupby(all_results.index).mean()

        # plot average chart
        #reference first instance of class in order to access chart/max/peak methods
        #Not sure this is the best way, is there a better way to call the chart function here?
        simulations[0].chart(averages)

        # also return:  (not from all simulations)
        #max_infected = simulations[0].max_infected(averages)
        #peak_infected = simulations[0].peak_infected(averages)

        #return average value of all max&peaks
        max_infected = mean([s.max_infected() for s in simulations])
        peak_infected = mean([s.peak_infected() for s in simulations])

        return f"max: {max_infected}, peak: {peak_infected}"


    def plot_state(self, time=0):
        #obtain state vars at required time
        x = list(self.results_df.columns)
        plot_var = [var for var in x if var.split("_")[1] == str(time)][0]
        #state = self.results_df[plot_var]
        # Manually map states to colours
        colours = self.results_df[plot_var].map(
            {"S": "green",
             "I": "red",
             "R": "blue",
             "D": "black"})

        # add line's for connections
        for i in range(len(self.connections)):
            #from the indices of connections find the co-ordinates for the points
            p1 = self.grid_points[self.connections[i][0]]
            p2 = self.grid_points[self.connections[i][1]]
            xs = [p1[0], p2[0]]
            ys = [p1[1], p2[1]]
            plt.plot(xs, ys, c="grey", alpha=0.5)

        plt.scatter(self.results_df["x_coord"], self.results_df["y_coord"],
                    s=5,
                    c=colours)
        plt.axis("off")
        plt.show()


    def run(self):
        for i in range(self.N):
            #for each day set state to infectced for suseptable individuals in contact with an infected person
            ##index of infected people
            infected = [inf for inf in range(len(self.grid_points)) if self.individuals[inf].state == "I"]

            # now create a list of the connections of each infected person
            day_susceptable = []
            for inf in infected:
                day_susceptable.extend(self.individuals[inf].connections_index)

                # also for each infected, recover/die with probabilities...
                self.individuals[inf].state = self.individuals[inf].set_state(
                                                    (1 - self.beta_recover - self.beta_death) #remains infected
                                                    ,self.beta_recover #recovers
                                                    ,0 #once infected cannot be susceptible again
                                                    ,self.beta_death #dies
                                                    )

            # for each of these set status to infected with probability gamma
            inf_connected = set(day_susceptable)   #need to create a distinct list so dont evaluate same person twice
                                #A person can only get infected with probability gamma regardless of the number of infected people around them.
            #if wanted to evaluate each individual at ever connection then dont use set above
            #this gives higher peak akin to project spec, but advised not to worry about peak value and stick to simpler method above
            #inf_connected = day_susceptable

            for inf_con in inf_connected:
                if self.individuals[inf_con].state == "S":
                    self.individuals[inf_con].state = self.individuals[inf_con].set_state(
                                                            self.gamma, 0, (1 - self.gamma), 0)

            #record state
            state_var = "state_" + str(i+1)
            self.results_df[state_var] = [self.individuals[i].state for i in range(self.m*self.n)]

        #record summary results
        #create empty df to record summary results
        summary_df = pd.DataFrame(index = ["I", "S", "R", "D"])
        plot_var = [var for var in list(self.results_df.columns) if var.split("_")[1].isdigit()]

        for i in plot_var:
            #create a df of the counts of each state at each daily iteration
            summary_df[i] = pd.DataFrame(self.results_df[i].value_counts())

        #need to transpose df
        summary_melted = summary_df.reset_index().melt(id_vars="index", var_name="Time", value_name="Stat")
        summary_melted['Time'] = summary_melted['Time'].apply( lambda x: int(x.split("_")[1]))
        #need in wide format but time as rows
        summary_time = summary_melted.pivot( index="Time", columns="index", values="Stat")
        # store summary table
        self.summary_time = summary_time

    def chart(self, df=None):
        if df is None:
            df = self.summary_time

        df.plot.line( color={"S": "green",
             "I": "red", "R": "blue","D": "black"})
        plt.show()

    def max_infected(self):
        """function to calculate max value of infected"""
        return self.summary_time["I"].max()

    def peak_infected(self):
        """function to identify day reached max infected (found by index of summary_time df)"""
        return self.summary_time["I"].idxmax()


class Individual(Simulation):
    """
        A class to represent an individual person in the simulation.

        ...

        Attributes
        ----------
        ID : integer
        State : str
            state of individual which changes throughout simulation

        Methods
        -------
        set_state:
            determines initial state and updates state throughout simulation
        _add_connection:
            Add's a connection to another individual as determined by the Simulation Class
        _add_connection_index:
            adds the index for the connection similar to relating to _add_connection

        """

    def __init__(self, id, grid_coords,
                 alpha_infected,
                 alpha_recovered,
                 beta_recover,
                 beta_death):
        self.id = id
        self.grid_coords = grid_coords
        self.alpha_infected = alpha_infected
        self.alpha_recovered = alpha_recovered
        self.beta_recover = beta_recover
        self.beta_death = beta_death
        self.state = self.set_state(self.alpha_infected, self.alpha_recovered,
                                         (1 - self.alpha_infected - self.alpha_recovered), 0)      #set the state based on initial parameters
        self.connections = []
        self.connections_index = []

    def __repr__(self):
        result = f"Individual {self.id} from Simulation with Probabilities: " \
                 f"infected={self.alpha_infected}, recovered= {self.beta_recover} death, {self.beta_death}"
        return result

    def __str__(self):
        return f"Individual {self.id} from Simulation class"

    def set_state(self, infected,
                    recovered,
                    susceptible,
                    deceased ):
        #default parameters set for initial state calculation
       state = np.random.choice(["I", "R", "S", "D"], 1, True,
                                [infected, recovered,
                                susceptible, deceased])[0]
       return state

    def _add_connection(self, connection):
        self.connections.append(connection)

    def _add_connection_index(self, connection_id):
        self.connections_index.append(connection_id)

#dictionary of settings
settings = {"m": 40,    #rows of individuals
            "n": 25,    #columns of individuals
            "r": 2,     #distance to neighbours
            "k": 4,
            "alpha_infected": 0.01,
            "alpha_recovered": 0,
            "beta_recover": 0.05,
            "beta_death": 0.005,
            "gamma": 0.075,
            "N": 100
            }
#Example Implementation
s = Simulation(*settings.values())
s.run()
s.plot_state(100)
plt.show()
s.chart()
plt.show()

Simulation.averaged_chart(100, settings)
plt.show()

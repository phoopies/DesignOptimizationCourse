"""
Solving the Two-bar truss optimization problem with
DESDEO framework as defined in 
http://apmonitor.com/me575/uploads/Main/twobar.pdf
But in a multiobjective fashion:
Instead of just minimizing one objective,
try to minimize weight, stress, buckling stress and deflection simultaneously 


Take the time to convert everything to metric?
Currently imperial
H height (in)
d diameter (in)
t thickness (in)
B seperation distance (in)
E modulus of elasticity (1000lbs/in^2)
p density (lbs/in^3)
P load (1000 lbs)
"""

from desdeo_problem.Constraint import ScalarConstraint
from desdeo_problem.Problem import MOProblem
from desdeo_problem.Objective import _ScalarObjective
from desdeo_problem.Variable import variable_builder
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII
import numpy as np

# Maybe the load should be a constant assigned here and then change the objective funcions a little
load = 66

# Defining the objective functions
def weight(xs: np.ndarray) -> np.ndarray:
    xs = np.atleast_2d(xs)
    H, d, t, B, E, p, P = xs[0]  # Assign the values to named variables for clarity
    return p * 2 * np.pi * d * t * np.sqrt(np.square(B / 2) + np.square(H))


def stress(xs: np.ndarray) -> np.ndarray:
    xs = np.atleast_2d(xs)
    H, d, t, B, E, p, P = xs[0]
    numerator = P * np.sqrt(np.square(B / 2) + np.square(H))
    denominator = 2 * t * np.pi * d * H
    return numerator / denominator


def buckling_stress(xs: np.ndarray) -> np.ndarray:
    xs = np.atleast_2d(xs)
    H, d, t, B, E, p, P = xs[0]
    numerator = np.square(np.pi) * E * (np.square(d) + np.square(t))
    denominator = 8 * (np.square(B / 2) + np.square(H))
    return numerator / denominator


def deflection(xs: np.ndarray) -> np.ndarray:
    xs = np.atleast_2d(xs)
    H, d, t, B, E, p, P = xs[0]
    numerator = P * np.power(np.square(B / 2) + np.square(H), (3/2))
    denominator = 2 * t * np.pi * d * np.square(H) * E
    return numerator / denominator


# Defining (decision) variables
# Height, diameter, thickness, Seperation distance, modulus of elasticity, density, load
var_names = ["H", "d", "t", "B", "E", "p", "P"] 
var_count = len(var_names)

initial_values = np.array([30.0, 3.0, 0.15, 60.0, 30000., 0.3, 66.0])

# set lower bounds for each variable
lower_bounds = np.array([20.0, 0.5, 0.01, 20.0, 25000., 0.01, 60.0])

# set upper bounds for each variable
upper_bounds = np.array([60.0, 5.0, 1.0, 100.0, 40000., 0.5, 70.0])
# TODO If one wishes to fix a variable to a certain value (constant value)

# Trying to minimize everything so no need to define minimize array

# Create a list of Variables for MOProblem class
variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

# Define objectives
obj1 = _ScalarObjective("Weight", weight)
obj2 = _ScalarObjective("Stress", stress)
obj3 = _ScalarObjective("Buckling stress", buckling_stress)
obj4 = _ScalarObjective("Deflection", deflection)

# In this example we are minimizing all the objectives subject to buckling stress < 250 and deflection < 1

# List of objectives for MOProblem class
objectives = [obj1, obj2, obj3, obj4]
objectives_count = len(objectives)

# Define maximun values for objective functions
weight_max = 24
stress_max = 100
b_stress_max = 250
deflection_max = 1

# Define constraint functions
"""
From desdeo_problem documentation:
The constraint should be defined so, that when evaluated, it should return a positive value, 
if the constraint is adhered to, and a negative, if the constraint is breached.
"""
const_weight = lambda xs, ys: weight_max - weight(xs)
const_stress = lambda xs, ys: stress_max - stress(xs)
const_buckling_stress = lambda xs, ys: b_stress_max - buckling_stress(xs)
const_deflection = lambda xs, ys: deflection_max - deflection(xs)

# Define DESDEO ScalarConstraints
con1 = ScalarConstraint("Weight", var_count, objectives_count, const_weight)
con2 = ScalarConstraint("Stress", var_count, objectives_count, const_stress)
con3 = ScalarConstraint(
    "Buckling stress", var_count, objectives_count, const_buckling_stress
)
con4 = ScalarConstraint("Deflection", var_count, objectives_count, const_deflection)
"""
A constraint of form obj1 - obj2 < C can be declared as
const = lambda xs, ys: C - (o1(xs) - o2(xs))
con = ScalarConstraint("c", var_count, objectives_count, const)
Also if a constraint of form  A < obj1 < C is needed:
One can declare 2 constraints for obj1, i guess
"""

constraints = [con3, con4]  # List of constraints for MOProblem class

# Create the problem
# This problem object can be passed to various different methods defined in DESDEO
prob = MOProblem(objectives=objectives, variables=variables)
print(prob.variables)
print(prob.evaluate(initial_values))

# NSGAIII, Uncomment below to see solve with NSGAIII
# evolver = NSGAIII(prob, n_iterations=10, n_gen_per_iter=20, population_size=100)

# plot, pref = evolver.requests()


# while evolver.continue_evolution():
#     evolver.iterate()

# individuals = evolver.population.individuals

# # Some 2D-plot
# import plotly.graph_objects as go
# layout = go.Layout(
#     title="A graph of decision variables",
#     xaxis=dict(
#         title="Height"
#     ),
#     yaxis=dict(
#         title="Diameter"
#     ) ) 

# fig1 = go.Figure(
#     layout=layout,
#     data=go.Scatter(x=individuals[:,0], y=individuals[:,1], mode="markers")
# )
# fig1.show()



# RVEA
# from desdeo_emo.othertools.plotlyanimate import animate_init_, animate_next_

# evolver = RVEA(prob, interact=True, n_iterations=5, n_gen_per_iter=100)
# figure = animate_init_(evolver.population.objectives, filename="river.html")
# plot, pref = evolver.requests()

# print(plot.content["dimensions_data"])

# while evolver.continue_evolution():
#     print(f"Current iteration {evolver._iteration_counter}")
#     plot, pref = evolver.iterate()
#     figure = animate_next_(
#         plot.content['data'].values,
#         figure,
#         filename="river.html",
#         generation=evolver._iteration_counter,
#     )

# print(plot.content['data'])

# individuals = evolver.population.individuals

# print(f"Some individual: {individuals[0]}")
# print(f"Objective values with i.e individual:\n"
#         f"Weight = {weight(individuals[0])}\n"
#         f"Stress = {stress(individuals[0])}\n"
#         f"Buckling stress = {buckling_stress(individuals[0])}\n"
#         f"Deflection = {deflection(individuals[0])}\n")



# # RPM
# from desdeo_mcdm.interactive.ReferencePointMethod import ReferencePointMethod
# # from desdeo_mcdm.utilities.solvers import payoff_table_method

# # ideal, nadir = payoff_table_method(prob)

# # print(f"Nadir: {nadir}\nIdeal: {ideal}")

# method = ReferencePointMethod(problem=prob)

# req = method.start()
# rp = np.array([23.99, 49.52, 185.3, 0.099])[:objectives_count]
# req.response = {
#     "reference_point": rp,
# }

# req = method.iterate(req)
# step = 1
# print("\nStep number: ", method._h)
# print("Reference point: ", rp)
# print("Pareto optimal solution: ", req.content["current_solution"])
# print("Additional solutions: ", req.content["additional_solutions"])

# while step < 4:
#         step += 1
#         rp = np.array([np.random.uniform(i, n) for i, n in zip(ideal, nadir)])
#         req.response = {"reference_point": rp, "satisfied": False}
#         req = method.iterate(req)
#         print("\nStep number: ", method._h)
#         print("Reference point: ", rp)
#         print("Pareto optimal solution: ", req.content["current_solution"])
#         print("Additional solutions: ", req.content["additional_solutions"])

# req.response = {"satisfied": True, "solution_index": 0}
# req = method.iterate(req)

# msg, final_sol, obj_vector = req.content.values()
# print(msg)

# print(f"Objective vector = {obj_vector}")
# print(f"Is the deflection constraint breached: {'YES' if (deflection_max - obj_vector[3] < 0) else 'NO'}")

#NIMBUS
from desdeo_mcdm.interactive.NIMBUS import NIMBUS

method = NIMBUS(prob) # 
# Start solving
classification_request, plot_request = method.start()

print(classification_request.content["message"])
print(classification_request.content)

#Give preference info
response = {"classifications": [""]}
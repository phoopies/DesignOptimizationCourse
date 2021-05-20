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
from desdeo_mcdm.interactive.ReferencePointMethod import ReferencePointMethod
import numpy as np



# Defining the objective functions
def weight(xs: np.ndarray) -> np.ndarray:
    xs = np.atleast_2d(xs)
    H, d, t, B, E, p, P = xs[0] # Assign the values to named variables for clarity
    return np.array([p * 2 * np.pi * d * t * np.sqrt(np.square(B/2) + np.square(H))])

def stress(xs: np.ndarray) -> np.ndarray:
    xs = np.atleast_2d(xs)
    H, d, t, B, E, p, P = xs[0] 
    numerator = P * np.sqrt(np.square(B/2) + np.square(H))
    denominator = 2 * t * np.pi * d * H 
    return np.array([numerator / denominator])

def buckling_stress(xs: np.ndarray) -> np.ndarray:
    xs = np.atleast_2d(xs)
    H, d, t, B, E, p, P = xs[0] 
    numerator = np.pi * E*(np.square(d) + np.square(t))
    denominator = 8 * (np.square(B/2) + np.square(H))
    return np.array([numerator / denominator])

def deflection(xs: np.ndarray) -> np.ndarray:
    xs = np.atleast_2d(xs)
    H, d, t, B, E, p, P = xs[0] 
    numerator = P * np.float_power(np.square(B/2) + np.square(H), 1.5)
    denominator = 2 * t * np.pi * d * np.square(H) * E
    return np.array([numerator / denominator])

# Defining variables
var_names = ["H", "d", "t", "B", "E", "p", "P"] # Meanings above 
var_count = len(var_names)

initial_values = np.array([30., 3., 0.15, 60., 30., 0.3, 66.])

# set lower bounds for each variable
lower_bounds = np.array([20., 0.5, 0.01, 20., 10., 0.01, 50])

# set upper bounds for each variable
upper_bounds = np.array([60., 5., 1., 100., 50., 0.5, 100])
# If one wished to fix a variable to a certain value (constant value)
# then set lower_bound = upper_found for that variable

# Trying to minimize everything so no need to define minimize array

# Create a list of Variables for MOProblem class
variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds) 


# Define objectives
obj1 = _ScalarObjective("Weight", weight)
obj2 = _ScalarObjective("Stress", stress)
obj3 = _ScalarObjective("Buckling stress", buckling_stress)
obj4 = _ScalarObjective("Deflection", deflection)

# In this example we are minimizing Weight and stress subject to buckling stress and deflection

# List of objectives for MOProblem class
objectives=[obj1, obj2]
objectives_count = len(objectives)

# Define maximun values for constrains
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
con1 = ScalarConstraint("Weight", var_count, objectives_count , const_weight)
con2 = ScalarConstraint("Stress", var_count, objectives_count, const_stress)
con3 = ScalarConstraint("Buckling stress", var_count, objectives_count, const_buckling_stress)
con4 = ScalarConstraint("Deflection", var_count, objectives_count, const_deflection)
# A constraint of form obj1 - obj2 < C can be declared as
# const = lambda xs, ys: C - (o1(xs) - o2(xs))
# con = ScalarConstraint("c", var_count, objectives_count, const)

constraints = [con1 ,con3, con4] # List of constraints for MOProblem class


# Create the problem
# This problem object can be passed to various different methods defined in DESDEO
# In this example it will be passed to the reference point method at least for now
prob = MOProblem(objectives=objectives, variables=variables, constraints=constraints)


# Solving the problem with the reference point method

# Make sure the size of ideal and nadir vectors are equal to the count of objectives
# Is there a method which calculates ideal and nadir points
ideal = np.zeros(objectives_count)
nadir = np.array([50., 80., 250., 1.,])[:objectives_count]

method = ReferencePointMethod(problem = prob, ideal = ideal, nadir = nadir)

req = method.start()
rp = np.array([23.99, 49.52, 185.3, 0.099])[:objectives_count]
req.response = {
    "reference_point": rp,
}

req = method.iterate(req)
step = 1
print("\nStep number: ", method._h)
print("Reference point: ", rp)
print("Pareto optimal solution: ", req.content["current_solution"])
print("Additional solutions: ", req.content["additional_solutions"])

while step < 4:
    step += 1
    rp = np.array([np.random.uniform(i, n) for i, n in zip(ideal, nadir)])
    req.response = {"reference_point": rp, "satisfied": False}
    req = method.iterate(req)
    print("\nStep number: ", method._h)
    print("Reference point: ", rp)
    print("Pareto optimal solution: ", req.content["current_solution"])
    print("Additional solutions: ", req.content["additional_solutions"])

req.response = {"satisfied": True, "solution_index": 0}
req = method.iterate(req)

msg, solution, obj_values = req.content.values()
print(f"\n{msg}")

H, d, t, B, E, p, P = solution

print("\nFinal solution variables:\n"
    f"Height: {H}\nDiameter: {d}\nThickness: {t}\n"
    f"Seperation Distance: {B}\nModulus of elasticity: {E}\n"
    f"Density: {p}\nLoad: {P}\n"
)

# print("With objective values of:")
# for obj_value_index in range(len(obj_values)):
#     print(f"{objectives[obj_value_index].name}: {obj_values[obj_value_index]}")

print("Objective function values:\n")
print(f"Weight: {weight(solution)}")
print(f"Stress: {stress(solution)}")
print(f"Buckling stress: {buckling_stress(solution)}")
print(f"Deflection: {deflection(solution)}")

# Plotting the solution ? 
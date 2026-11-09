using JSON

params_loaded = JSON.parsefile("params.json")

MTBF_list = params_loaded["MTBF"]
n_opt_list = params_loaded["n_opt"]
years_list = params_loaded["years"]
Q_list = params_loaded["Q"]
d_list = params_loaded["d"]
nb_test_per_data = params_loaded["nb_test_per_data"]



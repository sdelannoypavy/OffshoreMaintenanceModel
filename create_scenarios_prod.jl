load_factor_list = [40.1,44.65,28.7,24.6,34.85,42.95]
w = 100/maximum(load_factor_list)

pr = [w*load_factor_list[mod(t-1,6)+1] for t in 1:180, s in 1:62]
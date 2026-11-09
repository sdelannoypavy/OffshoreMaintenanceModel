using CSV, DataFrames, Statistics, Plots

# Lire le fichier CSV
df = CSV.read("results.csv", DataFrame)

# Préparer les moyennes par valeur de S
grouped = combine(groupby(df, :S), 
    :cost => mean => :mean_cost, 
    :simulated_cost => mean => :mean_simulated_cost)

# Tracer
scatter(df.S, df.simulated_cost, color=:red, markerstrokecolor=:transparent, markerstrokewidth=0, label="Simulation cost", legend=:topleft)
scatter!(df.S, df.cost, color=:blue, markerstrokecolor=:transparent, markerstrokewidth=0, label="Optimization cost")

# Tracer les moyennes comme des lignes
plot!(grouped.S, grouped.mean_cost, color=:blue, lw=2, label="Mean optimization cost")
plot!(grouped.S, grouped.mean_simulated_cost, color=:red, lw=2, label="Mean simulation cost")

xlabel!("number of scenarios in optimization")
ylabel!("Cost")

savefig("scenario_sensitivity.pdf")

# Afficher le graphique
display(current())

using CSV
using DataFrames
using StatsPlots
using StatsBase
using Statistics

function BuildPlot()

    df1 = CSV.read("cost.csv", DataFrame)
    println("mean cost for fluid: ", mean(df1[:, 2]))
    df2 = CSV.read("cost_benchmark_1.csv", DataFrame)
    println("mean cost for operational: ", mean(df2[:, 2]))
    df2[:, 1] = fill("operational", nrow(df2))
    rename!(df1, ["method", "cost"])
    rename!(df2, ["method", "cost"])
    df = vcat(df1, df2)

    #df.method = CategoricalArray(df.method, ordered=true, levels=["FA-w", "heuristics", "FA"])
    
    means = combine(groupby(df, :method), :cost => mean => :mean_cost)

    @df filtered_df boxplot(:method, (:cost);
    legend=false,
    fillalpha=0.7,
    linewidth=1.5,
    group=:method,
    color=[:blue :red],
    xtickfont = font(20),   # taille des noms de méthodes (axe x)
    ytickfont = font(20),   # taille des numéros sur l'axe y
    ylims=(0, 500))

    savefig("boxplot.pdf") 


end


function BuildPlot_sensitivity()

    # first boxplot

    df0101 = CSV.read("cost0101m.csv", DataFrame)
    rename_dict = Dict("fluid" => "θ = 0.1")
    df0101.method = map(x -> rename_dict[x], df0101.method)

    df9901 = CSV.read("cost9901m.csv", DataFrame)
    rename_dict = Dict("fluid" => "θ = 0.99")
    df9901.method = map(x -> rename_dict[x], df9901.method)

    
    df_combined01 = vcat(df0101, df9901)

    plot()

    @df df_combined01 boxplot(:method, :cost;
    legend=false,
    fillalpha=0.7,
    linewidth=1.5,
    group=:method,
    color=[:cyan :lime],
    xtickfont = font(15),   # taille des noms de méthodes (axe x)
    ytickfont = font(15),  
    guidefont = font(15),  # taille des numéros sur l'axe y
    ylims=(0, 2000),
    xlabel = "Model in optimization",
    ylabel = "Cost simulated with θ = 0.1")

    savefig("boxplot01.pdf") 

        
    
    # second boxplot

    df0199 = CSV.read("cost0199m.csv", DataFrame)
    rename_dict = Dict("fluid" => "θ = 0.1")
    df0199.method = map(x -> rename_dict[x], df0199.method)

    df9999 = CSV.read("cost9999m.csv", DataFrame)
    rename_dict = Dict("fluid" => "θ = 0.99")
    df9999.method = map(x -> rename_dict[x], df9999.method)

    
    df_combined99 = vcat(df0199, df9999)

    plot()

    @df df_combined99 boxplot(:method, :cost;
    legend=false,
    fillalpha=0.7,
    linewidth=1.5,
    group=:method,
    color=[:cyan :lime],
    xtickfont = font(15),   # taille des noms de méthodes (axe x)
    ytickfont = font(15),  
    guidefont = font(15),  # taille des numéros sur l'axe y
    ylims=(0, 2000),
    xlabel = "Model in optimization",
    ylabel = "Cost simulated with θ = 0.99")

    savefig("boxplot99.pdf") 


end

function plot_by_comp()
    
    simulation_data = CSV.read("simulation.csv", DataFrame)

    results = DataFrame(comp = String[], cost = Float64[])

    n = nrow(simulation_data)
    rows = simulation_data[1:n, :]

    years = n/180

    cost_co = 0
    cost_wa = 0
    cost_fa = 0
    cost_pu = 0
    cost_tr = 0
    cost_bu = 0
    cost_coo = 0

    for row in eachrow(rows)

        if (row[1] > 1) || (row[10] > 1)
            cost_co += row[18]
        end

        if (row[2] > 1) || (row[11] > 1)
            cost_wa += row[18]
        end

        if (row[3] > 1) || (row[12] > 1)
            cost_fa += row[18]
        end

        if (row[4] > 1) || (row[13] > 1)
            cost_pu += row[18]
        end

        if (row[5] > 1) || (row[14] > 1)
            cost_tr += row[18]
        end

        if (row[6] > 1) || (row[15] > 1)
            cost_coo += row[18]
        end

        if (row[7] > 1) || (row[16] > 1)
            cost_bu += row[18]
        end

    end

    cost_co = cost_co/years
    cost_wa = cost_wa/years
    cost_fa = cost_fa/years
    cost_pu = cost_pu/years
    cost_fa = cost_fa/years
    cost_coo = cost_coo/years
    cost_bu = cost_bu/years

        

    push!(results, (comp = "conv.", cost = cost_co))
    push!(results, (comp = "water", cost = cost_wa))
    push!(results, (comp = "fan", cost = cost_fa))
    push!(results, (comp = "pump", cost = cost_pu))
    push!(results, (comp = "trans.", cost = cost_tr))
    push!(results, (comp = "cool.", cost = cost_coo))
    push!(results, (comp = "bus.", cost = cost_bu))


    # Sauvegarder dans le fichier cost_by_comp.csv
    CSV.write("cost_by_comp.csv", results)

    bar(
    results.comp,      # noms des composants sur l'axe des x
    results.cost,      # hauteur des barres
    ylabel = "Mean cost",
    lw = 0.5,
    alpha = 0.8,
    color = :gray,
    legend = false,
    guidefont=font(15),
    xtickfont = font(15),  # taille de police des labels x
    rotation = 45,
    size = (900, 250),        # rotation si les noms sont longs
    left_margin = 15Plots.mm,
    right_margin = 15Plots.mm,
    top_margin = 15Plots.mm,
    bottom_margin = 20Plots.mm,
    yticks=[0,50],
)

    savefig("boxplot_comp.pdf") 

end

function compute_statistics()
  
    df = CSV.read("cost.csv", DataFrame)

    mean_val =  mean(df[:, 2])
    println("mean = $mean_val")

    α = 0.95  # niveau de confiance
    VaR = quantile(df[:, 2], α)
    println("VaR at 95 % = $VaR")

    freq_pos = (length(df[:, 2]) - count(x -> x == 0, df[:, 2])) / length(df[:, 2])
    println("probability of strictly positive cost = $freq_pos")

end
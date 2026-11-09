using CSV
using DataFrames
using Statistics
using Plots


# maintenance frequency depending on the season


df = CSV.read("simulation_AO6.csv", DataFrame)

df.sum = [sum(Vector(row)) > 0 ? 1 : 0 for row in eachrow(df[:, 13:21])]
df.season = mod.(df[:, 12], 6)


result = combine(
    groupby(df, :season),
    :sum => (x -> mean(x) * 100) => :pourcentage_B_1
)

labels = Dict(
    1 => "Jan–Feb",
    2 => "Mar–Apr",
    3 => "May–Jun",
    4 => "Jul–Aug",
    5 => "Sep–Oct",
    0 => "Nov–Dec"
)

result.label = [labels[s] for s in result.season]

order = Dict(1=>1, 2=>2, 3=>3, 4=>4, 5=>5, 0=>6)

sort!(result, :season, by = s -> order[s])

bar(
    result.label,
    result.pourcentage_B_1,
    xlabel = "",
    ylabel = "Probability (%)",
    legend = false
)

savefig("Frequency of maintenance depending on the season.pdf")

#maintenance frequency depending on the year

df.year = (df[:, 12] .÷ 6) .+ 1

result_year = combine(
    groupby(df, :year),
    :sum => (x -> (1 - prod(1 .- x))*100) => :pourcentage_year
)

sort!(result_year, :year)

bar(
    result_year[1:30,1],
    result_year[1:30,2],
    xlabel = "year",
    ylabel = "Probability (%)",
    legend = false
)

savefig("Frequency of maintenance depending on the year.pdf")


# maintenance frequency

df = CSV.read("simulation_AO6.csv", DataFrame)

for i in 0:7
    sum = (1/6)*1/mean([x > 0 ? 1 : 0 for x in df[:, 10 + i]])
    println("component ", i+1, " : average time between two maintenances = ", sum, "years")
end

# cost by component

simulation_data = CSV.read("simulation_AO6.csv", DataFrame)

results = DataFrame(comp = String[], cost_cur = Float64[], cost_prev = Float64[], cost_fail = Float64[])

n = nrow(simulation_data)
rows = simulation_data[1:n, :]

years = n/180

global cost_co_cur = 0
global cost_wa_cur = 0
global cost_fa_cur = 0
global cost_pu_cur = 0
global cost_tr_cur = 0
global cost_coo_cur = 0
global cost_bu_cur = 0

global cost_co_prev = 0
global cost_wa_prev = 0
global cost_fa_prev = 0
global cost_pu_prev = 0
global cost_tr_prev = 0
global cost_coo_prev = 0
global cost_bu_prev = 0

global cost_co_fail = 0
global cost_wa_fail = 0
global cost_fa_fail = 0
global cost_pu_fail = 0
global cost_tr_fail = 0
global cost_coo_fail = 0
global cost_bu_fail = 0

for row in eachrow(rows)

    if (row[1] == 12) && (row[10] > 0)
        global cost_co_cur += row[18]
    end

    if (row[2] == 3) && (row[11] > 0)
        global cost_wa_cur += row[18]
    end

    if (row[3] == 8) && (row[12] > 0)
        global cost_fa_cur += row[18]
    end

    if (row[4] == 3) && (row[13] > 0)
        global cost_pu_cur += row[18]
    end

    if (row[5] == 3) && (row[14] > 0)
        global cost_tr_cur += row[18]
    end

    if (row[6] == 3) && (row[15] > 0)
        global cost_coo_cur     += row[18]
    end

    if (row[7] == 3) && (row[16] > 0)
        global cost_bu_cur += row[18]
    end

    if (row[1] == 12) && (row[10] == 0)
        global cost_co_fail += row[18]
    end

    if (row[2] == 3) && (row[11] == 0)
        global cost_wa_fail += row[18]
    end

    if (row[3] == 8) && (row[12] == 0)
        global cost_fa_fail += row[18]
    end

    if (row[4] == 3) && (row[13] == 0)
        global cost_pu_fail += row[18]
    end

    if (row[5] == 3) && (row[14] == 0)
        global cost_tr_fail += row[18]
    end

    if (row[6] == 3) && (row[15] == 0)
        global cost_coo_fail += row[18]
    end

    if (row[7] == 3) && (row[16] == 0)
        global cost_bu_fail += row[18]
    end

    if (row[1] <= 11) && (row[10] > 0)
        global cost_co_prev += row[18]
    end

    if (row[2] <= 2) && (row[11] > 0)
        global cost_wa_prev += row[18]
    end

    if (row[3] <= 7) && (row[12] > 0)
        global cost_fa_prev += row[18]
    end

    if (row[4] <= 2) && (row[13] > 0)
        global cost_pu_prev += row[18]
    end

    if (row[5] <= 2) && (row[14] > 0)
        global cost_tr_prev += row[18]
    end

    if (row[6] <= 2) && (row[15] > 0)
        global cost_coo_prev += row[18]
    end

    if (row[7] <= 2) && (row[16] > 0)
        global cost_bu_prev += row[18]
    end


end

cost_co_cur = cost_co_cur/years
cost_wa_cur = cost_wa_cur/years
cost_fa_cur = cost_fa_cur/years
cost_pu_cur = cost_pu_cur/years
cost_tr_cur = cost_tr_cur/years
cost_coo_cur = cost_coo_cur/years
cost_bu_cur = cost_bu_cur/years

cost_co_prev = cost_co_prev/years
cost_wa_prev = cost_wa_prev/years
cost_fa_prev = cost_fa_prev/years
cost_pu_prev = cost_pu_prev/years
cost_tr_prev = cost_tr_prev/years
cost_coo_prev = cost_coo_prev/years
cost_bu_prev = cost_bu_prev/years

cost_co_fail = cost_co_fail/years
cost_wa_fail = cost_wa_fail/years
cost_fa_fail = cost_fa_fail/years
cost_pu_fail = cost_pu_fail/years
cost_tr_fail = cost_tr_fail/years
cost_coo_fail = cost_coo_fail/years
cost_bu_fail = cost_bu_fail/years
    

push!(results, (comp = "conv.", cost_cur = cost_co_cur, cost_prev = cost_co_prev, cost_fail = cost_co_fail))
push!(results, (comp = "water", cost_cur = cost_wa_cur, cost_prev = cost_wa_prev, cost_fail = cost_wa_fail))
push!(results, (comp = "fan", cost_cur = cost_fa_cur, cost_prev = cost_fa_prev, cost_fail = cost_fa_fail))
push!(results, (comp = "pump", cost_cur = cost_pu_cur, cost_prev = cost_pu_prev, cost_fail = cost_pu_fail))
push!(results, (comp = "trans.", cost_cur = cost_tr_cur, cost_prev = cost_tr_prev,  cost_fail = 0)) # No failure data for transformer
push!(results, (comp = "cool.", cost_cur = cost_coo_cur,  cost_prev = 0, cost_fail = cost_coo_fail)) # No preventive data for cooler
push!(results, (comp = "bus.", cost_cur = 0, cost_prev = 0, cost_fail = cost_bu_fail))


# Sauvegarder dans le fichier cost_by_comp.csv
CSV.write("cost_by_comp.csv", results)

bar(
results.comp,      # noms des composants sur l'axe des x
results.cost_cur,      # hauteur des barres
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

savefig("boxplot_comp_cur.pdf") 

bar(
results.comp,      # noms des composants sur l'axe des x
results.cost_prev,      # hauteur des barres
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

savefig("boxplot_comp_prev.pdf") 

bar(
results.comp,      # noms des composants sur l'axe des x
results.cost_fail,      # hauteur des barres
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

savefig("boxplot_comp_fail.pdf")    


#corrective vs preventive maintenance 
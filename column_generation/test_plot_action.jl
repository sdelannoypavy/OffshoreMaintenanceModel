# test_plot_action.jl
include("../create_simple_pole.jl")   # d, C
include("column_generation.jl")       # MaintenancePlan, MaintenanceColumn
include("plot_action.jl")             # plot_and_save_action

# Crée une action fictive
plan = MaintenancePlan(
    [1, 3, 5],        # composants maintenus : Converter (d=6), Fans (d=1), Transformer (d=1)
    [5, 20, 35]       # jours de début
)
col = MaintenanceColumn(plan, [10, 11, 12, 40, 41])  # jours libres

T_per = 55

plot_and_save_action(col, d, C, T_per, "test", "results_test")
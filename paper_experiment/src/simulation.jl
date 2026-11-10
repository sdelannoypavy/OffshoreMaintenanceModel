# simulation.jl
#
# Monte Carlo simulation of a policy over the whole concession (Section "Setup", paragraph "Policy
# evaluation"). A policy is a function (x, k, t) -> (Column, computing time), where x is the vector of
# the degradation states of the components, k the vector of the remaining quotas and t ∈ 1:120 the
# period. The only randomness is the degradation of the components. Scenario s uses the same random
# draws for every method (common random numbers).

using Printf, Random, Statistics

"Append `row` to the csv file `file`, writing `header` first if the file is new."
function append_csv(file::String, header::String, row::String)
    mkpath(dirname(file))
    (!isfile(file) || filesize(file) == 0) && open(io -> println(io, header), file, "w")
    open(io -> println(io, row), file, "a")
end

"Random draws of scenario `scen`: U[t, τ, m] uniform on [0, 1] (only U[t, 1, m] is used, once per period)."
scenario_draws(fd::FluidData, scen::Int) =
    rand(MersenneTwister(1000 + scen), fd.inst.n_years * N_PERIODS_PER_YEAR, fd.inst.T_period, fd.M)

"Next state of a Markov chain of transition matrix P[to, from], from state x, with the uniform draw u."
function sample_next(P::Matrix{Float64}, x::Int, u::Float64)
    cum = 0.0
    for xp in 1:size(P, 1)
        cum += P[xp, x]
        u <= cum && return xp
    end
    return x
end

"""
Simulate one period from the states x with the joint action `col` and the draws U[τ, m]. A maintained
component returns to the new state, and every other component transitions once, at the end of the
period, with P_period (no degradation during the period, as in the MDP). Returns (penalty, replacement
cost, new states).
"""
function simulate_period(fd::FluidData, x::Vector{Int}, col::Column, U::Matrix{Float64})
    penalty, replacement = period_cost(fd, x, col)
    xnext = [col.start[m] > 0 ? 1 : sample_next(fd.inst.poles[fd.pole_of[m]].P_period[fd.comp_of[m]], x[m], U[1, m])
             for m in 1:fd.M]
    return penalty, replacement, xnext
end

"Check that a joint action satisfies the technician capacity on every day and the quota of every substation."
function check_action(fd::FluidData, col::Column, k::Vector{Int})
    for τ in 1:fd.inst.T_period
        demand = sum((fd.rho[m] for m in 1:fd.M if under_repair(fd, col.start, m, τ)); init = 0.0)
        demand <= fd.inst.R + 1e-9 || error("technician capacity exceeded on day $τ")
    end
    for p in 1:fd.inst.n_poles
        length(col.free_days[p]) <= k[p] || error("quota exceeded on substation $p")
    end
end

const PERIODS_HEADER = "method,instance,scenario,period,states,quota,maintained,start_days,n_free_days,penalty,replacement_cost,solve_time"

"""
Simulate scenario `scen` over the whole concession, starting with every component new and every
quota full. Writes one row per period in `periods_file` (if not `nothing`). Returns the total cost, its
penalty and replacement parts, the number of maintenances and the time spent computing new decisions.
"""
function simulate_scenario(fd::FluidData, policy::Function, scen::Int;
                           method::String, instance::String, periods_file::Union{String,Nothing})
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    U = scenario_draws(fd, scen)
    x = ones(Int, fd.M); k = copy(fd.K0)
    penalty_total = 0.0; replacement_total = 0.0; n_maint = 0; solve_time = 0.0
    for t in 1:T_total
        (t - 1) % N_PERIODS_PER_YEAR == 0 && (k = copy(fd.K0))   # annual reset of the quota
        col, dt = policy(x, k, t)
        check_action(fd, col, k)
        solve_time += dt
        penalty, replacement, x_next = simulate_period(fd, x, col, U[t, :, :])
        penalty_total += penalty; replacement_total += replacement
        n_maint += count(>(0), col.start)
        if !isnothing(periods_file)
            append_csv(periods_file, PERIODS_HEADER,
                @sprintf("%s,%s,%d,%d,%s,%s,%s,%s,%s,%.4f,%.4f,%.4f", method, instance, scen, t,
                    join(x, "-"), join(k, "-"),
                    join([m for m in 1:fd.M if col.start[m] > 0], "-"),
                    join([col.start[m] for m in 1:fd.M if col.start[m] > 0], "-"),
                    join(length.(col.free_days), "-"), penalty, replacement, dt))
        end
        k = k .- length.(col.free_days)
        x = x_next
    end
    return (total = penalty_total + replacement_total, penalty = penalty_total, replacement = replacement_total,
            n_maintenances = n_maint, solve_time = solve_time)
end

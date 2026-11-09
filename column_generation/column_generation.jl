using Gurobi
using JuMP
using StatsBase
using Printf
using JLD2

include("../parameters_weakly_coupled.jl")

const GRB_ENV_CG = Gurobi.Env()
optimizer_cg = () -> Gurobi.Optimizer(GRB_ENV_CG)

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

# A plan specifies, for each maintained component, a start day.
struct MaintenancePlan
    comps      :: Vector{Int}   # maintained components (sorted)
    start_days :: Vector{Int}   # start_days[i] = start day of comps[i]
end

# A column = plan + specific subset of free days (quota).
# On a free day, no capacity penalty is paid even if the substation is
# offline for maintenance.
struct MaintenanceColumn
    plan      :: MaintenancePlan
    free_days :: Vector{Int}    # calendar days τ declared free (quota)
end

Base.:(==)(a::MaintenancePlan, b::MaintenancePlan) =
    a.comps == b.comps && a.start_days == b.start_days
Base.hash(p::MaintenancePlan, h::UInt) = hash((p.comps, p.start_days), h)

Base.:(==)(a::MaintenanceColumn, b::MaintenanceColumn) =
    a.plan == b.plan && a.free_days == b.free_days
Base.hash(c::MaintenanceColumn, h::UInt) = hash((c.plan, c.free_days), h)


# Cyclic position (1..6) of LP stage t, given that stage 1 is at t0_cyc.
stage_tcyc(t::Int, t0_cyc::Int) = (t0_cyc - 1 + t - 1) % 6 + 1

# Calendar year index (1-based) for LP stage t.
# The quota resets at the start of each cyclic year (t_cyc = 1).
# Year 1 contains the (7 - t0_cyc) remaining stages of the current year.
function stage_year(t::Int, t0_cyc::Int)
    stages_in_year1 = 7 - t0_cyc
    t <= stages_in_year1 && return 1
    return 1 + ceil(Int, (t - stages_in_year1) / 6)
end

# ─────────────────────────────────────────────────────────────────────────────
# Precomputed data for the master LP
#
# p_simple[c, x_from, x_to, m, t_cyc]  (m=1: normal, m=2: maintenance)
# δ_simple[c, x, t_cyc]                 failure cost (no free days)
# ─────────────────────────────────────────────────────────────────────────────

function compute_p_simple(C, n, n_max, P_period; nb_t_cyc=6)
    p = zeros(C, n_max, n_max, 2, nb_t_cyc)
    for c in 1:C, x in 1:n[c], xp in 1:n[c], t in 1:nb_t_cyc
        p[c, x, xp, 1, t] = P_period[c][xp, x]   # normal mode
    end
    for c in 1:C, x in 1:n[c], t in 1:nb_t_cyc
        p[c, x, 1, 2, t] = 1.0                    # maintenance → reset to state 1
    end
    return p
end

function compute_δ_simple(C, h_optim, pr, n, d, P_daily, P_evac; nb_t_cyc=6)
    n_max_loc = maximum(n)
    δ = zeros(C, n_max_loc, nb_t_cyc)
    for c in 1:C, x in 1:n[c], t in 1:nb_t_cyc
        T_per    = 62 - count(==(-1), h_optim[t, 1, :])
        pol_m    = create_schedule_m(0, 1, T_per)
        pol_κ    = create_schedule_κ(0, 1, T_per)
        nb_scen  = size(h_optim, 2)
        cost     = 0.0
        for scen in 1:nb_scen
            cost += average_period_cost(T_per, pol_m, pol_κ,
                        x, h_optim[t, scen, :], pr[t, :],
                        n[c], d[c], P_daily[c], P_evac[c, :]) / nb_scen
        end
        δ[c, x, t] = cost
    end
    return δ
end

# ─────────────────────────────────────────────────────────────────────────────
# Joint (conv+fan) helpers
# ─────────────────────────────────────────────────────────────────────────────

joint_flat(xi::Int, xj::Int, n_fan::Int) = (xi - 1) * n_fan + xj
joint_unfold(k::Int, n_fan::Int)         = ((k - 1) ÷ n_fan + 1, (k - 1) % n_fan + 1)
conv_mode(mj::Int) = mj in (2, 4) ? 2 : 1
fan_mode(mj::Int)  = mj in (3, 4) ? 2 : 1

# δ_joint[xi, xj, t] — expected failure cost when converter is at state xi,
# fan at state xj, no maintenance on either, all other components at full capacity.
# Computed analytically by evolving marginal distributions forward with P_daily.
function compute_δ_joint(h_optim, pr, n_conv::Int, n_fan::Int,
                          P_daily_conv::Matrix, P_daily_fan::Matrix,
                          P_evac::Matrix;
                          conv_comp::Int=1, fan_comp::Int=3, nb_t_cyc::Int=6)
    δ = zeros(n_conv, n_fan, nb_t_cyc)
    for xi in 1:n_conv, xj in 1:n_fan, t in 1:nb_t_cyc
        T_per = 62 - count(==(-1), h_optim[t, 1, :])
        pr_t  = pr[t, 1:T_per]
        p_c = zeros(n_conv); p_c[xi] = 1.0
        p_f = zeros(n_fan);  p_f[xj] = 1.0
        cost = 0.0
        for τ in 1:T_per
            for xi_τ in 1:n_conv, xj_τ in 1:n_fan
                p = p_c[xi_τ] * p_f[xj_τ]
                p < 1e-12 && continue
                cap = min(P_evac[conv_comp, xi_τ], P_evac[fan_comp, xj_τ])
                cost += p * max(pr_t[τ] - cap, 0.0)
            end
            p_c = P_daily_conv * p_c
            p_f = P_daily_fan  * p_f
        end
        δ[xi, xj, t] = cost
    end
    return δ
end

# Δ_pairs[(c,c2)][xc,xc2,t] = δ_joint(c,c2,xc,xc2,t) − δ_simple[c,xc,t] − δ_simple[c2,xc2,t]
# Pairwise interaction correction: additional cost due to having BOTH components degraded
# simultaneously (always ≤ 0 because min(cap_c,cap_c') ≥ max(cap_c,cap_c') component-wise).
function compute_Δ_pairs(δ_simple, h_optim, pr, n::Vector{Int}, C::Int,
                          P_daily, P_evac::Matrix; nb_t_cyc::Int=6)
    Δ_pairs = Dict{Tuple{Int,Int}, Array{Float64,3}}()
    for c in 1:C, c2 in c+1:C
        δ_jt = compute_δ_joint(h_optim, pr, n[c], n[c2],
                                P_daily[c], P_daily[c2], P_evac;
                                conv_comp=c, fan_comp=c2, nb_t_cyc=nb_t_cyc)
        Δ = zeros(n[c], n[c2], nb_t_cyc)
        for xc in 1:n[c], xc2 in 1:n[c2], t in 1:nb_t_cyc
            Δ[xc, xc2, t] = δ_jt[xc, xc2, t] - δ_simple[c, xc, t] - δ_simple[c2, xc2, t]
        end
        Δ_pairs[(c, c2)] = Δ
    end
    return Δ_pairs
end

# ─────────────────────────────────────────────────────────────────────────────
# Joint column cost η_{col}(t)
#
# Expected capacity penalty when plan col.plan is executed with free days
# col.free_days, at cyclic stage t_cyc.
# Days τ ∈ free_days do not contribute to the penalty.
# ─────────────────────────────────────────────────────────────────────────────

function joint_column_cost(col::MaintenanceColumn, t_cyc::Int,
                            h_optim, pr, d::Vector{Int})
    plan = col.plan
    isempty(plan.comps) && return 0.0

    nb_scen  = size(h_optim, 2)
    T_per    = 62 - count(==(-1), h_optim[t_cyc, 1, :])
    free_set = Set(col.free_days)

    total = 0.0
    for scen in 1:nb_scen
        h_t    = h_optim[t_cyc, scen, :]
        pr_t   = pr[t_cyc, :]
        q_c    = zeros(Int, length(plan.comps))
        cost_s = 0.0

        for τ in 1:T_per
            for (i, c) in enumerate(plan.comps)
                plan.start_days[i] == τ && (q_c[i] += d[c])
            end

            any_active = any(q_c[i] > 0 for i in eachindex(plan.comps))

            if any_active && !(τ in free_set)
                cost_s += pr_t[τ]
            end

            if h_t[τ] == 1
                for i in eachindex(plan.comps)
                    q_c[i] = max(0, q_c[i] - 1)
                end
            end
        end

        total += cost_s / nb_scen
    end
    return total
end

# Global η cache: (column, t0_cyc) → cost vector over all stages.
# Persists across all calls to column_generation_solve.
const η_cache_global = Dict{Tuple{MaintenanceColumn,Int}, Vector{Float64}}()

"""
Compute and cache η[col][t] for t = 1..H_full.
Checks η_cache_global (persistent) before computing.
"""
function get_η!(η_cache::Dict{MaintenanceColumn, Vector{Float64}},
                col::MaintenanceColumn, H_full::Int, t0_cyc::Int,
                h_optim, pr, d::Vector{Int})
    if !haskey(η_cache, col)
        global_key = (col, t0_cyc)
        if haskey(η_cache_global, global_key)
            η_cache[col] = η_cache_global[global_key]
        else
            vec = zeros(H_full)
            for t in 1:H_full
                vec[t] = joint_column_cost(col, stage_tcyc(t, t0_cyc), h_optim, pr, d)
            end
            η_cache[col]               = vec
            η_cache_global[global_key] = vec
            @printf("  η computed: comps=%s  starts=%s  free=%s\n",
                    string(col.plan.comps), string(col.plan.start_days),
                    string(col.free_days))
        end
    end
    return η_cache[col]
end

# ─────────────────────────────────────────────────────────────────────────────
# Incremental column addition to the master LP
#
# Creates variable B[col,t] for each stage t, and updates:
#   - objective coefficient : η[col][t]
#   - maint_prob[c, κ_col, t] : -1 for each c ∈ col.plan.comps
#     where κ_col = |col.free_days|
#
# A column found useful at some (t₀,K₀) is added for ALL stages t.
# The master LP decides which stages actually use it (others keep B=0).
# ─────────────────────────────────────────────────────────────────────────────

function add_column!(model, col::MaintenanceColumn,
                     η_vec::Vector{Float64},
                     H_full::Int,
                     comp_cost::Vector{Float64},
                     B_vars::Vector{Vector{VariableRef}})
    κ_col      = length(col.free_days)
    maint_prob = model[:maint_prob]
    col_vars   = Vector{VariableRef}(undef, H_full)
    cost_col   = sum(comp_cost[c] for c in col.plan.comps; init = 0.0)
    is_dict    = maint_prob isa Dict
    for t in 1:H_full
        x = @variable(model, lower_bound = 0)
        set_objective_coefficient(model, x, η_vec[t] + cost_col)
        for c in col.plan.comps
            con = is_dict ? maint_prob[(c, κ_col, t)] : maint_prob[c, κ_col, t]
            set_normalized_coefficient(con, x, -1.0)
        end
        col_vars[t] = x
    end
    push!(B_vars, col_vars)
end

# ─────────────────────────────────────────────────────────────────────────────
# Master LP
#
# State variables q[c, x, m, i, t]:
#   c     : component index
#   x     : degradation state (1..n[c])
#   m     : mode (1=normal, 2=maintenance)
#   i     : index into the list of valid (k, κ) pairs, where
#             k  = remaining free-day quota at the START of period t
#             κ  = free days used IN period t  (κ ≤ k by construction)
#   t     : LP stage (1..H_full)
#
# Objective:
#   Σ δ_simple[c,x,t_cyc] * q[c,x,1,i,t]         failure costs
# + Σ η[col][t] * B[col,t]                          joint maintenance costs
# + Σ comp_cost[c] * Σ_{x,i} q[c,x,2,i,t]          fixed replacement costs
#
# Constraints:
#   init      : initial state (x=x0, k=k0)
#   flow      : Markov dynamics on x + quota dynamics on k
#   coupling  : maint_prob[c, κ, t] links q to column B variables
#               (no separate quota constraint — embedded in flow)
# ─────────────────────────────────────────────────────────────────────────────

function build_master_LP(cols::Vector{MaintenanceColumn},
                          η_cache::Dict{MaintenanceColumn, Vector{Float64}},
                          x0::Vector{Int}, H_full::Int,
                          k0::Int, K0::Int, t0_cyc::Int,
                          δ_simple, p_simple,
                          n::Vector{Int}, C::Int,
                          comp_cost::Vector{Float64}, n_max::Int,
                          optimizer;
                          enhanced::Bool=false, δ_joint=nothing,
                          conv_comp::Int=1, fan_comp::Int=3,
                          full_enhanced::Bool=false, Δ_pairs=nothing)

    year_resets = [stage_year(t+1, t0_cyc) > stage_year(t, t0_cyc) for t in 1:H_full-1]
    valid_kκ    = [(k, κ) for k in 0:K0 for κ in 0:k]

    model = Model(optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "Method", 1)

    # Components modelled individually (all, or all except conv+fan in enhanced mode)
    indep_comps = enhanced ? [c for c in 1:C if c ∉ (conv_comp, fan_comp)] : collect(1:C)

    # ── State-action variables ─────────────────────────────────────────────────
    @variable(model, q[c=indep_comps, x=1:n[c], m=1:2, (k,κ)=valid_kκ, t=1:H_full] >= 0)
    for c in indep_comps, x in 1:n[c], m in 1:2, (k, κ) in valid_kκ
        k != k0 && fix(q[c, x, m, (k, κ), 1], 0.0; force=true)
    end

    # ── Joint variable for enhanced mode ──────────────────────────────────────
    # q_joint[(xi,xj), mj, (k,κ), t] : conv at xi, fan at xj, joint mode mj∈1..4
    if enhanced
        n_conv   = n[conv_comp]
        n_fan    = n[fan_comp]
        N_joint  = n_conv * n_fan
        joint_states = [(xi, xj) for xi in 1:n_conv for xj in 1:n_fan]
        @variable(model, q_joint[(xi,xj)=joint_states, mj=1:4, (k,κ)=valid_kκ, t=1:H_full] >= 0)
        for (xi, xj) in joint_states, mj in 1:4, (k, κ) in valid_kκ
            k != k0 && fix(q_joint[(xi,xj), mj, (k,κ), 1], 0.0; force=true)
        end
    end

    # ── Pairwise variables for full_enhanced mode ─────────────────────────────
    # p_pair[(c,c2)][xc, xc2, (k,κ), t] : joint probability of c at xc, c2 at xc2,
    # both in normal mode (m=1), with quota (k,κ) at stage t.
    # Used only to improve the objective via pairwise δ corrections.
    p_pair_dict = Dict{Tuple{Int,Int}, Any}()
    if full_enhanced
        for c in 1:C, c2 in c+1:C
            nc_val, nc2_val = n[c], n[c2]
            p_pair_dict[(c, c2)] = @variable(model,
                [xc=1:nc_val, xc2=1:nc2_val, (k,κ)=valid_kκ, t=1:H_full],
                lower_bound=0, upper_bound=1)
        end
    end

    # ── Objective ──────────────────────────────────────────────────────────────
    if enhanced
        n_conv = n[conv_comp]; n_fan = n[fan_comp]
        joint_states = [(xi, xj) for xi in 1:n_conv for xj in 1:n_fan]
        @objective(model, Min,
            sum(δ_simple[c, x, stage_tcyc(t, t0_cyc)] * q[c, x, 1, (k, κ), t]
                for c in indep_comps, x in 1:n[c], (k, κ) in valid_kκ, t in 1:H_full)
            + sum(δ_joint[xi, xj, stage_tcyc(t, t0_cyc)] *
                  q_joint[(xi,xj), 1, (k,κ), t]   # mj=1: no maintenance on either
                  for (xi,xj) in joint_states, (k,κ) in valid_kκ, t in 1:H_full))
    elseif full_enhanced
        # Individual δ_simple terms (weakly-coupled baseline) + pairwise Δ corrections.
        # Δ[(c,c2)] ≤ 0 always, so the optimizer will saturate the p_pair variables
        # against their marginal constraints, tightening the relaxation.
        @objective(model, Min,
            sum(δ_simple[c, x, stage_tcyc(t, t0_cyc)] * q[c, x, 1, (k, κ), t]
                for c in 1:C, x in 1:n[c], (k, κ) in valid_kκ, t in 1:H_full)
            + sum(Δ_pairs[(c, c2)][xc, xc2, stage_tcyc(t, t0_cyc)] *
                  p_pair_dict[(c, c2)][xc, xc2, (k, κ), t]
                  for c in 1:C, c2 in c+1:C,
                      xc in 1:n[c], xc2 in 1:n[c2],
                      (k, κ) in valid_kκ, t in 1:H_full))
    else
        @objective(model, Min,
            sum(δ_simple[c, x, stage_tcyc(t, t0_cyc)] * q[c, x, 1, (k, κ), t]
                for c in 1:C, x in 1:n[c], (k, κ) in valid_kκ, t in 1:H_full))
    end

    # ── Initial state ──────────────────────────────────────────────────────────
    α = zeros(C, n_max)
    for c in 1:C; α[c, x0[c]] = 1.0; end

    @constraint(model, init[c=indep_comps, x=1:n[c]],
        sum(q[c, x, m, (k0, κ), 1] for m in 1:2, κ in 0:k0) == α[c, x])

    if enhanced
        n_conv = n[conv_comp]; n_fan = n[fan_comp]
        joint_states = [(xi, xj) for xi in 1:n_conv for xj in 1:n_fan]
        @constraint(model, init_joint[(xi,xj)=joint_states],
            sum(q_joint[(xi,xj), mj, (k0,κ), 1] for mj in 1:4, κ in 0:k0)
            == α[conv_comp, xi] * α[fan_comp, xj])
    end

    # ── Flow: independent components ──────────────────────────────────────────
    for t in 2:H_full, c in indep_comps, x in 1:n[c], k_new in 0:K0
        tc_prev = stage_tcyc(t-1, t0_cyc)
        resets  = year_resets[t-1]
        lhs = sum(q[c, x, m, (k_new, κ), t] for m in 1:2, κ in 0:k_new)
        if resets
            k_new != K0 && continue
            rhs = sum(p_simple[c, xp, x, m, tc_prev] * q[c, xp, m, (k, κ), t-1]
                      for xp in 1:n[c], m in 1:2, (k, κ) in valid_kκ)
        else
            rhs = sum(p_simple[c, xp, x, m, tc_prev] *
                      q[c, xp, m, (k_new + κ_prev, κ_prev), t-1]
                      for xp in 1:n[c], m in 1:2, κ_prev in 0:K0-k_new)
        end
        @constraint(model, lhs >= rhs)
    end

    # ── Flow: joint component ─────────────────────────────────────────────────
    # For mj=1 (no maint): both conv and fan evolve under their normal dynamics.
    # For mj=2 (conv only): conv resets to 1, fan evolves normally.
    # For mj=3 (fan only): fan resets to 1, conv evolves normally.
    # For mj=4 (both): both reset to 1.
    # Transitions use p_simple[conv_comp,:,:,conv_mode(mj),:] etc.
    if enhanced
        n_conv = n[conv_comp]; n_fan = n[fan_comp]
        joint_states_next = [(xi, xj) for xi in 1:n_conv for xj in 1:n_fan]
        for t in 2:H_full, xi in 1:n_conv, xj in 1:n_fan, k_new in 0:K0
            tc_prev = stage_tcyc(t-1, t0_cyc)
            resets  = year_resets[t-1]
            lhs = sum(q_joint[(xi,xj), mj, (k_new,κ), t]
                      for mj in 1:4, κ in 0:k_new)
            if resets
                k_new != K0 && continue
                rhs = sum(
                    p_simple[conv_comp, xi_p, xi, conv_mode(mj), tc_prev] *
                    p_simple[fan_comp,  xj_p, xj, fan_mode(mj),  tc_prev] *
                    q_joint[(xi_p,xj_p), mj, (k,κ), t-1]
                    for xi_p in 1:n_conv, xj_p in 1:n_fan,
                        mj in 1:4, (k,κ) in valid_kκ)
            else
                rhs = sum(
                    p_simple[conv_comp, xi_p, xi, conv_mode(mj), tc_prev] *
                    p_simple[fan_comp,  xj_p, xj, fan_mode(mj),  tc_prev] *
                    q_joint[(xi_p,xj_p), mj, (k_new+κ_prev, κ_prev), t-1]
                    for xi_p in 1:n_conv, xj_p in 1:n_fan,
                        mj in 1:4, κ_prev in 0:K0-k_new)
            end
            @constraint(model, lhs >= rhs)
        end
    end

    # ── Coupling: q ↔ column B variables ──────────────────────────────────────
    # For enhanced mode, use a Dict to allow indexing by (c, κ, t).
    # Modes 2,3,4 for the joint variable correspond to maintenance on conv (c=conv_comp),
    # fan (c=fan_comp), or both respectively.
    if enhanced
        n_conv = n[conv_comp]; n_fan = n[fan_comp]
        joint_states = [(xi, xj) for xi in 1:n_conv for xj in 1:n_fan]
        maint_prob_dict = Dict{Tuple{Int,Int,Int}, ConstraintRef}()
        for κ in 0:K0, t in 1:H_full
            for c in indep_comps
                maint_prob_dict[(c, κ, t)] = @constraint(model,
                    sum(q[c, x, 2, (k,κ), t] for x in 1:n[c], k in κ:K0) <= 0)
            end
            # conv_comp: mj ∈ {2,4}
            maint_prob_dict[(conv_comp, κ, t)] = @constraint(model,
                sum(q_joint[(xi,xj), mj, (k,κ), t]
                    for (xi,xj) in joint_states, mj in (2,4), k in κ:K0) <= 0)
            # fan_comp: mj ∈ {3,4}
            maint_prob_dict[(fan_comp, κ, t)] = @constraint(model,
                sum(q_joint[(xi,xj), mj, (k,κ), t]
                    for (xi,xj) in joint_states, mj in (3,4), k in κ:K0) <= 0)
        end
        model[:maint_prob] = maint_prob_dict
    else
        @constraint(model, maint_prob[c=1:C, κ=0:K0, t=1:H_full],
            sum(q[c, x, 2, (k, κ), t] for x in 1:n[c], k in κ:K0) <= 0)
    end

    # ── Consistency: pairwise marginals ≤ individual normal-mode marginals ───────
    # Σ_{xc2} p_pair[c,c2,xc,xc2,(k,κ),t] ≤ q[c,xc,1,(k,κ),t]  (c's marginal)
    # Σ_{xc}  p_pair[c,c2,xc,xc2,(k,κ),t] ≤ q[c2,xc2,1,(k,κ),t] (c2's marginal)
    # Inequality (not equality) avoids infeasibility when the two mode-1 masses differ.
    if full_enhanced
        for c in 1:C, c2 in c+1:C
            pvars = p_pair_dict[(c, c2)]
            for xc in 1:n[c], (k, κ) in valid_kκ, t in 1:H_full
                @constraint(model,
                    sum(pvars[xc, xc2, (k, κ), t] for xc2 in 1:n[c2]) <= q[c, xc, 1, (k, κ), t])
            end
            for xc2 in 1:n[c2], (k, κ) in valid_kκ, t in 1:H_full
                @constraint(model,
                    sum(pvars[xc, xc2, (k, κ), t] for xc in 1:n[c]) <= q[c2, xc2, 1, (k, κ), t])
            end
        end
    end

    # ── Initial columns ────────────────────────────────────────────────────────
    B_vars = Vector{Vector{VariableRef}}()
    for col in cols
        add_column!(model, col, η_cache[col], H_full, comp_cost, B_vars)
    end

    return model, B_vars
end

# ─────────────────────────────────────────────────────────────────────────────
# Solve the master LP and extract duals
# ─────────────────────────────────────────────────────────────────────────────

function solve_and_get_duals(model, C::Int, H_full::Int, K0::Int)
    optimize!(model)
    status = termination_status(model)
    obj    = status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) ? objective_value(model) : Inf

    # duals_maint[c, κ+1, t] = dual of maint_prob[c, κ, t]
    # (κ is 0-indexed, array is 1-indexed in second dimension)
    duals_maint = zeros(C, K0+1, H_full)

    if status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        maint_prob = model[:maint_prob]
        is_dict    = maint_prob isa Dict
        for c in 1:C, κ in 0:K0, t in 1:H_full
            con = is_dict ? maint_prob[(c, κ, t)] : maint_prob[c, κ, t]
            duals_maint[c, κ+1, t] = dual(con)
        end
    end

    return status, obj, duals_maint
end

# ─────────────────────────────────────────────────────────────────────────────
# Pricing MILP
#
# For a fixed stage t, finds the column (plan, free_days) minimising the
# reduced cost:
#   rc = η_{col,t} + Σ_{c∈plan} duals_t[c, κ_col+1]
#
# where κ_col = |col.free_days| is a DECISION of the MILP.
#
# Since the dual λ[c, κ, t] depends on κ (a decision), we introduce:
#   κ_sel[κ] ∈ {0,1}  : 1 iff exactly κ free days are used  (SOS-1 on 0..K0)
#   y[c, κ]  ∈ {0,1}  : linearisation of z[c] ∧ κ_sel[κ]
#
# The dual term in the objective becomes Σ_{c,κ} λ[c,κ+1] * y[c,κ], which
# is linear in the y variables.
# ─────────────────────────────────────────────────────────────────────────────

function solve_pricing_MILP(t::Int, K0::Int,
                              duals_t::Matrix{Float64},   # [c, κ+1], dual of maint_prob[c,κ,t]
                              h_optim, pr, d::Vector{Int},
                              r::Vector{Int}, R::Int, C::Int,
                              comp_cost::Vector{Float64};
                              S_price::Int=15,
                              exact_pricing::Bool=true,
                              fixed_start_day::Union{Int,Nothing}=15,
                              margin::Int=10,
                              time_limit::Float64=30.0,
                              optimizer=optimizer_cg)

    t_cyc       = (t - 1) % 6 + 1
    nb_scen_all = size(h_optim, 2)
    S           = exact_pricing ? nb_scen_all : min(S_price, nb_scen_all)
    scen_idx    = exact_pricing ? (1:nb_scen_all) : sort(sample(1:nb_scen_all, S; replace=false))

    T_per = 62 - count(==(-1), h_optim[t_cyc, 1, :])
    T_end = T_per - margin
    T_end < 1 && return nothing, Inf

    h    = [h_optim[t_cyc, scen_idx[s], τ] for s in 1:S, τ in 1:T_per]
    pr_t = pr[t_cyc, 1:T_per]
    M_c  = d

    model = Model(optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    set_optimizer_attribute(model, "TimeLimit",  time_limit)

    # ── Scheduling variables ───────────────────────────────────────────────────
    @variable(model, z[c=1:C], Bin)
    @variable(model, m_start[c=1:C, τ=1:T_end], Bin)
    @variable(model, ong_m[c=1:C, s=1:S, τ=1:T_per], Bin)
    @variable(model, q_var[c=1:C, s=1:S, τ=1:T_per] >= 0)
    @variable(model, any_maint[s=1:S, τ=1:T_per], Bin)
    @variable(model, free[τ=1:T_per], Bin)
    @variable(model, w[s=1:S, τ=1:T_per], Bin)

    # ── κ selection variables ──────────────────────────────────────────────────
    # κ_sel[κ] = 1 iff |free_days| = κ  (exactly one active)
    @variable(model, κ_sel[κ=0:K0], Bin)
    # y[c,κ] = z[c] ∧ κ_sel[κ]  (linearisation of the bilinear dual term)
    @variable(model, y[c=1:C, κ=0:K0], Bin)

    # ── Scheduling constraints ─────────────────────────────────────────────────
    @constraint(model, [c=1:C],
        sum(m_start[c, τ] for τ in 1:T_end) == z[c])

    if !isnothing(fixed_start_day)
        @constraint(model, [c=1:C, τ=1:T_end; τ != fixed_start_day],
            m_start[c, τ] == 0)
    end

    for c in 1:C, s in 1:S
        @constraint(model, q_var[c, s, 1] == d[c] * m_start[c, 1])
        for τ in 1:T_per-1
            rhs = q_var[c, s, τ] - h[s, τ] * ong_m[c, s, τ]
            τ+1 <= T_end && (rhs = rhs + d[c] * m_start[c, τ+1])
            @constraint(model, q_var[c, s, τ+1] == rhs)
        end
    end

    @constraint(model, [c=1:C, s=1:S, τ=1:T_per], ong_m[c,s,τ] <= q_var[c,s,τ])
    @constraint(model, [c=1:C, s=1:S, τ=1:T_per],
        ong_m[c,s,τ] >= (1.0 / M_c[c]) * q_var[c,s,τ])
    @constraint(model, [s=1:S, τ=1:T_per],
        sum(r[c] * ong_m[c,s,τ] for c in 1:C) <= R)
    @constraint(model, [c=1:C, s=1:S, τ=1:T_per],
        any_maint[s,τ] >= ong_m[c,s,τ])

    # ── w = free ∧ any_maint ──────────────────────────────────────────────────
    @constraint(model, [s=1:S, τ=1:T_per], w[s,τ] <= free[τ])
    @constraint(model, [s=1:S, τ=1:T_per], w[s,τ] <= any_maint[s,τ])
    @constraint(model, [s=1:S, τ=1:T_per], w[s,τ] >= free[τ] + any_maint[s,τ] - 1)

    # ── κ selection: link free days count to κ_sel ─────────────────────────────
    @constraint(model, sum(κ_sel[κ] for κ in 0:K0) == 1)
    @constraint(model, sum(free[τ] for τ in 1:T_per) == sum(κ * κ_sel[κ] for κ in 0:K0))

    # ── y = z ∧ κ_sel  (McCormick linearisation for binary × binary) ───────────
    @constraint(model, [c=1:C, κ=0:K0], y[c,κ] <= z[c])
    @constraint(model, [c=1:C, κ=0:K0], y[c,κ] <= κ_sel[κ])
    @constraint(model, [c=1:C, κ=0:K0], y[c,κ] >= z[c] + κ_sel[κ] - 1)

    # ── Objective: reduced cost ────────────────────────────────────────────────
    # rc = η_{col,t}  +  Σ_{c,κ} λ[c, κ+1, t] · y[c,κ]
    # η_{col,t} = (1/S) Σ pr_t[τ] (any_maint - w)   [capacity penalty]
    @objective(model, Min,
        (1.0 / S) * sum(pr_t[τ] * (any_maint[s,τ] - w[s,τ])
                         for s in 1:S, τ in 1:T_per)
        + sum(comp_cost[c] * z[c] for c in 1:C)
        + sum(duals_t[c, κ+1] * y[c,κ] for c in 1:C, κ in 0:K0))

    optimize!(model)

    status = termination_status(model)
    if !(status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.TIME_LIMIT))
        return nothing, Inf
    end

    obj_val = objective_value(model)

    # ── Extract column ─────────────────────────────────────────────────────────
    comps_found      = Int[]
    start_days_found = Int[]
    for c in 1:C
        if value(z[c]) > 0.5
            push!(comps_found, c)
            for τ in 1:T_end
                if value(m_start[c, τ]) > 0.5
                    push!(start_days_found, τ)
                    break
                end
            end
        end
    end

    length(comps_found) != length(start_days_found) && return nothing, Inf

    free_days_found = sort([τ for τ in 1:T_per if value(free[τ]) > 0.5])

    plan = MaintenancePlan(comps_found, start_days_found)
    col  = MaintenanceColumn(plan, free_days_found)
    return col, obj_val
end

# ─────────────────────────────────────────────────────────────────────────────
# Pricing step: solve one pricing MILP per stage t.
#
# The MILP jointly chooses which components to maintain AND how many free days
# to use (κ), using the full dual matrix duals_maint[c, κ+1, t].
# ─────────────────────────────────────────────────────────────────────────────

function pricing_step(duals_maint::Array{Float64,3},   # [c, κ+1, t]
                       cols::Vector{MaintenanceColumn},
                       H_full::Int, K0::Int, t0_cyc::Int,
                       h_optim, pr, d::Vector{Int},
                       r::Vector{Int}, R::Int, C::Int,
                       comp_cost::Vector{Float64};
                       tol_rc::Float64=-1.0,
                       S_price::Int=15,
                       exact_pricing::Bool=true,
                       fixed_start_day::Union{Int,Nothing}=15,
                       time_limit::Float64=30.0,
                       verbose::Bool=true)

    new_cols = MaintenanceColumn[]
    best_rc  = Inf

    for t in 1:H_full
        col, rc = solve_pricing_MILP(t, K0, duals_maint[:, :, t],
                                      h_optim, pr, d, r, R, C, comp_cost;
                                      S_price=S_price, exact_pricing=exact_pricing,
                                      fixed_start_day=fixed_start_day,
                                      time_limit=time_limit)
        isnothing(col) && continue

        κ_col = length(col.free_days)
        verbose && @printf("    t=%d  κ=%d  rc=%.6f  comps=%s  starts=%s  free=%s\n",
                           t, κ_col, rc, string(col.plan.comps),
                           string(col.plan.start_days), string(col.free_days))

        best_rc = min(best_rc, rc)

        if rc < tol_rc && !(col in cols) && !(col in new_cols)
            push!(new_cols, col)
        end
    end

    return new_cols, best_rc
end

# ─────────────────────────────────────────────────────────────────────────────
# Column generation main loop
# ─────────────────────────────────────────────────────────────────────────────

function column_generation_solve(x0::Vector{Int}, k0::Int, K0::Int,
                                   t0::Int,
                                   H_full::Int,
                                   h_optim, pr,
                                   n::Vector{Int}, d::Vector{Int},
                                   C::Int,
                                   comp_cost::Vector{Float64}, n_max::Int,
                                   δ_simple, p_simple;
                                   r::Vector{Int}=ones(Int, C),
                                   R::Int=C,
                                   max_iter::Int=30,
                                   S_price::Int=15,
                                   exact_pricing::Bool=true,
                                   fixed_start_day::Union{Int,Nothing}=15,
                                   time_limit_pricing::Float64=30.0,
                                   tol_rc::Float64=-1.0,
                                   warm_start_cols::Vector{MaintenanceColumn}=MaintenanceColumn[],
                                   verbose::Bool=true,
                                   enhanced::Bool=false, P_daily=nothing, P_evac=nothing,
                                   conv_comp::Int=1, fan_comp::Int=3,
                                   full_enhanced::Bool=false)

    t0_cyc = (t0 - 1) % 6 + 1

    # ── Initial column set ────────────────────────────────────────────────────
    cols    = MaintenanceColumn[]
    η_cache = Dict{MaintenanceColumn, Vector{Float64}}()

    push!(cols, MaintenanceColumn(MaintenancePlan(Int[], Int[]), Int[]))
    for c in 1:C
        push!(cols, MaintenanceColumn(MaintenancePlan([c], [1]), Int[]))
    end

    for col in warm_start_cols
        col ∉ cols && push!(cols, col)
    end

    for col in cols
        get_η!(η_cache, col, H_full, t0_cyc, h_optim, pr, d)
    end

    # ── Build master LP ───────────────────────────────────────────────────────
    δ_joint_val  = nothing
    Δ_pairs_val  = nothing
    if enhanced && !isnothing(P_daily) && !isnothing(P_evac)
        δ_joint_val = compute_δ_joint(h_optim, pr, n[conv_comp], n[fan_comp],
                                       P_daily[conv_comp], P_daily[fan_comp],
                                       P_evac; conv_comp=conv_comp, fan_comp=fan_comp)
    end
    if full_enhanced && !isnothing(P_daily) && !isnothing(P_evac)
        Δ_pairs_val = compute_Δ_pairs(δ_simple, h_optim, pr, n, C, P_daily, P_evac)
    end
    model, B_vars =
        build_master_LP(cols, η_cache, x0, H_full, k0, K0, t0_cyc,
                        δ_simple, p_simple, n, C, comp_cost, n_max, optimizer_cg;
                        enhanced=enhanced, δ_joint=δ_joint_val,
                        conv_comp=conv_comp, fan_comp=fan_comp,
                        full_enhanced=full_enhanced, Δ_pairs=Δ_pairs_val)

    obj_hist = Float64[]

    for iter in 1:max_iter
        verbose && @printf("\n══ Iteration %d  (%d columns) ══\n", iter, length(cols))

        status, obj, duals_maint = solve_and_get_duals(model, C, H_full, K0)

        if !(status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED))
            @printf("  Master LP status: %s — stopping\n", status)
            break
        end

        verbose && @printf("  LP value: %.6f\n", obj)
        push!(obj_hist, obj)

        new_cols, best_rc = pricing_step(
            duals_maint, cols, H_full, K0, t0_cyc,
            h_optim, pr, d, r, R, C, comp_cost;
            tol_rc=tol_rc, S_price=S_price, exact_pricing=exact_pricing,
            fixed_start_day=fixed_start_day,
            time_limit=time_limit_pricing, verbose=verbose)

        verbose && @printf("  Best reduced cost: %.8f  (%d new columns)\n",
                           best_rc, length(new_cols))

        if isempty(new_cols)
            verbose && @printf("  → Optimality reached.\n")
            break
        end

        for new_col in new_cols
            push!(cols, new_col)
            get_η!(η_cache, new_col, H_full, t0_cyc, h_optim, pr, d)
            add_column!(model, new_col, η_cache[new_col], H_full, comp_cost, B_vars)
        end

        verbose && @printf("  → %d columns added. Total: %d\n",
                           length(new_cols), length(cols))
    end

    status_f, obj_f, _ = solve_and_get_duals(model, C, H_full, K0)

    @printf("\n%s\n", "═"^60)
    @printf("Column generation complete\n")
    @printf("  Columns retained : %d\n", length(cols))
    @printf("  Optimal value    : %.6f\n", obj_f)
    @printf("  Status           : %s\n", status_f)

    return model, B_vars, cols, obj_f, obj_hist
end

# ─────────────────────────────────────────────────────────────────────────────
# Policy extraction at the first stage
# ─────────────────────────────────────────────────────────────────────────────

function extract_policy_cg(cols::Vector{MaintenanceColumn},
                            B_vars::Vector{Vector{VariableRef}}, C::Int)
    mass = zeros(length(cols))
    for col_id in eachindex(cols)
        mass[col_id] = value(B_vars[col_id][1])
    end

    col_star = cols[argmax(mass)]
    m_opt    = zeros(Int, C)
    for c in col_star.plan.comps
        m_opt[c] = 1
    end

    return m_opt, col_star.free_days, col_star
end

# ─────────────────────────────────────────────────────────────────────────────
# Policy cache and π_cg interface
# ─────────────────────────────────────────────────────────────────────────────

cache_cg = Dict{Tuple{Tuple{Vararg{Int64}}, Int64, Int64},
                Tuple{Vector{Int64}, MaintenanceColumn, Vector{Int64}}}()

# Last columns generated — used for warm starting the next solve.
last_cols_cg = MaintenanceColumn[]

function load_cache_cg!(file::String)
    !isfile(file) && return
    local saved
    @load file saved
    merge!(cache_cg, saved)
    @printf("  CG cache loaded: %d policies from %s\n", length(cache_cg), file)
end

function save_cache_cg(file::String)
    saved = cache_cg
    @save file saved
end

function π_cg(x0, k0, K0, t0,
               h_optim, pr, n, d, C,
               comp_cost, n_max, δ_simple, p_simple;
               H_full=6, r=ones(Int,C), R=C,
               max_iter=20, S_price=15,
               tol_rc=-1.0,
               exact_pricing::Bool=true,
               fixed_start_day::Union{Int,Nothing}=15,
               warm_start::Bool=false,
               cache_file::Union{String,Nothing}=nothing,
               enhanced::Bool=false, P_daily=nothing, P_evac=nothing,
               conv_comp::Int=1, fan_comp::Int=3,
               full_enhanced::Bool=false)

    cle = (Tuple(x0), k0, t0)

    if !haskey(cache_cg, cle)
        _, B_vars, cols, _, _ = column_generation_solve(
            x0, k0, K0, t0, H_full, h_optim, pr, n, d, C,
            comp_cost, n_max, δ_simple, p_simple;
            r=r, R=R, max_iter=max_iter, S_price=S_price, verbose=false,
            tol_rc=tol_rc, exact_pricing=exact_pricing,
            fixed_start_day=fixed_start_day,
            warm_start_cols=warm_start ? last_cols_cg : MaintenanceColumn[],
            enhanced=enhanced, P_daily=P_daily, P_evac=P_evac,
            conv_comp=conv_comp, fan_comp=fan_comp,
            full_enhanced=full_enhanced)

        global last_cols_cg = cols

        m_opt, free_days, col_star = extract_policy_cg(cols, B_vars, C)
        cache_cg[cle] = (m_opt, col_star, free_days)

        !isnothing(cache_file) && save_cache_cg(cache_file)
    end

    return cache_cg[cle]   # (m_opt, col_star, free_days)
end

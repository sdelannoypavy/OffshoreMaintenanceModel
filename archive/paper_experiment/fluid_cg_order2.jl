# fluid_cg_order2.jl
#
# Order-2 higher-order formulation (Section 4.2 of the paper) built on top of the Full
# Fluid LP solved by column generation (fluid_lp_cg.jl). Ported from the "full_enhanced"
# mode of column_generation/column_generation.jl.
#
# Why pairwise pseudomarginals help here. The failure cost δ_m used in the order-1 Fluid
# LP (fd.δ, see instances.jl / build_fluid_data) is computed component by component,
# assuming every other component of the pole is at full capacity (see the docstring of
# failure_cost in instances.jl). This is exactly the source of looseness described in
# Section 4.1: the true joint cost of a pole depends on the minimum capacity over its
# components (pole_penalty in instances.jl), not on the sum of their individual failure
# costs, so the order-1 relaxation under-estimates the cost of two components of the same
# pole being simultaneously degraded. Cross-pole pairs never interact this way (the
# penalty decomposes additively across poles, see instance_penalty): for those, summing
# the two components' own δ is EXACT, so Δ ≡ 0 identically -- tracking such a pair in the
# master LP would be correct but entirely pointless (see the note in compute_Δ_pairs and
# build_master_order2 below on how this is exploited to skip them cheaply).
#
# Rather than introducing full joint dynamics for a pair (which would multiply the state
# space of the master LP and require re-deriving the pricing MILP, as in the "enhanced"
# mode of column_generation.jl), we use the lighter route already validated there
# ("full_enhanced"): introduce pairwise occupation variables p_pair[m1,m2][x1,x2,k,κ,t]
# for "both m1 and m2 not maintained" (mode 1), tied to the existing order-1 marginals by
# INEQUALITY constraints (so no new pricing / no new consistency-with-columns constraint,
# hence no change needed to solve_pricing or add_column!), and add a correction term
#   Δ_pairs[m1,m2][x1,x2] = δ_joint(m1,m2 at x1,x2) − δ_1(x1) − δ_2(x2) <= 0
# to the objective. The optimizer saturates p_pair against its marginal bounds, which
# tightens the relaxation (Proposition 9-style argument, applied pairwise rather than
# through new joint dynamics).
#
# Requires fluid_lp_cg.jl to be loaded first (provides FluidData, Column, MasterLP,
# build_master, add_column!, solve_master!, consistency_duals, solve_pricing, stage_cyc,
# GRB_ENV, ...).

using JuMP, Gurobi, Printf

# ─────────────────────────────────────────────────────────────────────────────
# δ_joint and Δ_pairs: static pairwise correction to the failure-cost objective
# ─────────────────────────────────────────────────────────────────────────────

"""
Expected penalty of the pole over one period if neither m1 nor m2 is maintained and they
respectively start the period in states x1, x2, all other components of their (common)
pole assumed at full capacity. Weather-free: the two marginal distributions are evolved
forward day by day with their own daily transition matrices (independent evolution given
no maintenance), as in compute_δ_joint of column_generation.jl. Only meaningful for m1, m2
of the SAME pole (compute_Δ_pairs never calls this for a cross-pole pair, see below: it
sets Δ ≡ 0 for those directly, without needing a "joint capacity" that would not make
physical sense across two unrelated poles).
"""
function δ_joint_pair(fd::FluidData, m1::Int, m2::Int, x1::Int, x2::Int)
    p = fd.pole_of[m1]; pole = fd.inst.poles[p]; T = fd.inst.T_period
    c1 = fd.comp_of[m1]; c2 = fd.comp_of[m2]
    v1 = zeros(fd.n_states[m1]); v1[x1] = 1.0
    v2 = zeros(fd.n_states[m2]); v2[x2] = 1.0
    P1 = pole.P_daily[c1]; P2 = pole.P_daily[c2]
    cost = 0.0
    for τ in 1:T
        cap = 0.0
        for s1 in 1:fd.n_states[m1], s2 in 1:fd.n_states[m2]
            pr = v1[s1] * v2[s2]
            pr < 1e-12 && continue
            cap += pr * min(pole.P_evac[c1, s1], pole.P_evac[c2, s2])
        end
        cost += pole.daily_penalty[τ] * (100.0 - cap) / 100
        v1 = P1 * v1; v2 = P2 * v2
    end
    return cost
end

"""
Δ_pairs[(m1,m2)][x1,x2] = δ_joint_pair(m1,m2,x1,x2) − δ[m1][x1] − δ[m2][x2] <= 0, for
components m1, m2 of the SAME pole. For a pair of DIFFERENT poles, the true joint cost is
EXACTLY δ[m1][x1] + δ[m2][x2] (the penalty decomposes additively across poles, see
instance_penalty), so Δ ≡ 0 identically -- this is computed directly, without calling
δ_joint_pair (which has no meaning across two unrelated poles' capacities).

Cross-pole pairs are legitimate to request (e.g. via `pairs`, or by leaving every filter
at its default) and are never rejected; they are simply guaranteed to contribute nothing
to the objective, so build_master_order2 skips building their (expensive) LP structure
once it sees an all-zero Δ matrix.

By default, EVERY pair of distinct components is tracked (same-pole AND cross-pole): all
C(M,2) of them. Two OPTIONAL restriction mechanisms are available (neither applied unless
explicitly given), in the same spirit as Section 4.3 ("adding only a few well chosen
cuts") for when tracking every pair is not tractable:
  - `pairs`, if given, is used as the exact candidate list (no size/count filtering);
  - otherwise, if `max_pair_states` is given, pairs with n[m1]*n[m2] > max_pair_states are
    discarded first (they are the most expensive to track);
  - and if `max_pairs` is given, only the `max_pairs` remaining pairs with the largest
    impact max(|Δ|) are kept. Since cross-pole pairs always have impact 0, this ranking
    naturally prioritises same-pole pairs first whenever the pair count is restricted.
"""
function compute_Δ_pairs(fd::FluidData; pairs::Union{Vector{Tuple{Int,Int}},Nothing} = nothing,
                         max_pair_states::Union{Int,Nothing} = nothing,
                         max_pairs::Union{Int,Nothing} = nothing)
    fd.inst.simplified &&
        println("  order-2: simplified instance, additive cost is exact: every Δ ≡ 0 (no objective " *
                "correction), but the order-2 pseudomarginals p_pair are still built for same-pole " *
                "pairs, to tighten the relaxation of the marginals themselves (build_master_order2)")
    candidates = Tuple{Int,Int}[]
    if isnothing(pairs)
        skipped = Tuple{Int,Int}[]
        for m1 in 1:fd.M, m2 in (m1 + 1):fd.M
            if isnothing(max_pair_states) || fd.n_states[m1] * fd.n_states[m2] <= max_pair_states
                push!(candidates, (m1, m2))
            else
                push!(skipped, (m1, m2))
            end
        end
        isempty(skipped) || @printf("  order-2: skipping %d oversized pair(s) (n1*n2 > %d): %s\n",
                                    length(skipped), max_pair_states, string(skipped))
    else
        for (m1, m2) in pairs
            m1 != m2 || error("pair ($m1, $m2) is not a pair of distinct components")
            push!(candidates, (m1, m2))
        end
    end

    Δ_all = Dict{Tuple{Int,Int},Matrix{Float64}}()
    for (m1, m2) in candidates
        if fd.pole_of[m1] == fd.pole_of[m2] && !fd.inst.simplified
            # δ_joint_pair propagates day by day (pole.P_daily): only meaningful against the
            # ORIGINAL model's own day-by-day δ (fd.δ = failure_cost). On a simplified instance
            # the true joint cost is additive by construction (Section "Simplified problem" of
            # implemented_formulations.tex), so Δ ≡ 0 there too, without calling δ_joint_pair at
            # all (it would otherwise be comparing against the wrong cost model, local_cost /
            # failure_cost_simplified, and could spuriously come out non-zero).
            Δm = zeros(fd.n_states[m1], fd.n_states[m2])
            for x1 in 1:fd.n_states[m1], x2 in 1:fd.n_states[m2]
                Δm[x1, x2] = δ_joint_pair(fd, m1, m2, x1, x2) - fd.δ[m1][x1] - fd.δ[m2][x2]
            end
            Δ_all[(m1, m2)] = Δm
        else
            Δ_all[(m1, m2)] = zeros(fd.n_states[m1], fd.n_states[m2])   # cross-pole, or simplified: exactly 0
        end
    end

    if isnothing(pairs) && !isnothing(max_pairs) && length(Δ_all) > max_pairs
        # Cross-pole pairs are skipped outright by build_master_order2 regardless of Δ (see
        # its comment), so they must never displace a same-pole pair here: rank same-pole
        # pairs first, by decreasing impact max(|Δ|) as a tiebreak. On the original model
        # that tiebreak alone already put every same-pole pair before every cross-pole one
        # (cross-pole impact is always 0), but on a simplified instance every same-pole pair
        # ALSO has impact 0 (Section "Simplified problem" of implemented_formulations.tex),
        # so the explicit same-pole-first key is what keeps this correct there too.
        ranked = sort(collect(keys(Δ_all));
                      by = key -> (fd.pole_of[key[1]] != fd.pole_of[key[2]], -maximum(abs, Δ_all[key])))
        kept, dropped = ranked[1:max_pairs], ranked[(max_pairs + 1):end]
        @printf("  order-2: keeping the %d highest-impact pair(s) out of %d: %s\n",
                max_pairs, length(Δ_all), string(kept))
        @printf("  order-2: dropped (smaller impact): %s\n", string(dropped))
        Δ_all = Dict(k => Δ_all[k] for k in kept)
    end
    return Δ_all
end

# ─────────────────────────────────────────────────────────────────────────────
# Master LP augmented with pairwise (order-2) occupation variables
# ─────────────────────────────────────────────────────────────────────────────

"""
Build the order-1 master LP (fluid_lp_cg.build_master) and add, for every pair in
`Δ_pairs`, genuine order-2 pseudomarginals
  p_pair[m1,m2][x1,x2,a1,a2,k,κ,t] >= 0,  a1,a2 in {1,2} (both modes, not just "not
maintained"), with:
  - their OWN initial condition and flow constraints, mirroring exactly those of q in
    build_master, using the PRODUCT transition kernel
    p_{m1}^{a1}(x1|x1') * p_{m2}^{a2}(x2|x2') (the two components evolve independently
    given their own local actions -- Definition 1 of the paper); this pins down the
    correlation structure of the pair via genuine Markov propagation, instead of leaving
    it an unconstrained degree of freedom;
  - explicit marginal-consistency constraints tying p_pair back to q at EVERY stage t
    (not only t = 1):
      Σ_{x2,a2} p_pair[x1,x2,a1,a2,k,κ,t] = q[m1,x1,a1,k,κ,t]   ∀ x1, a1
      Σ_{x1,a1} p_pair[x1,x2,a1,a2,k,κ,t] = q[m2,x2,a2,k,κ,t]   ∀ x2, a2
    matching the general definition of a *consistent* order-k pseudomarginal (Section 4.2:
    "marginalising a higher-order pseudomarginal down to a lower order recovers it") and
    constraint (9c) of Proposition 9's LP (Appendix A), generalised to a pair of distinct
    local actions (a1, a2) instead of one shared action.

An earlier version of this function relied on Init + Flow ALONE to imply marginal
consistency by induction, omitting this second set of constraints. That argument is
wrong: q's own Flow constraint (build_master) only pins the sum over BOTH mode and κ at
every stage, never an individual mode -- so nothing stops the mode split of p_pair from
drifting away from q's own (column-determined) mode split at any t > 1, not just t = 1.
Without the explicit tie, the solver is free to route as much mass as it likes into the
a1 = a2 = 1 slice (rewarded by Δ <= 0) regardless of what q is actually doing, which is
unsound: it let the "order-2" bound fall BELOW the order-1 one in testing, the reverse of
a valid tightening.

Objective correction (only the "neither maintained" slice a1=a2=1 carries a Δ term, since
that is the only mode combination for which δ_joint_pair was computed):
  Σ_{k,κ,t} Σ_{x1,x2} Δ_pairs[m1,m2][x1,x2] * p_pair[m1,m2][x1,x2,1,1,k,κ,t]

Cross-pole pairs (m1, m2 of DIFFERENT poles) are handled by a separate branch below, WITHOUT
a (k, κ) index: the two poles have independent quotas, so there is no single shared (k, κ)
to index a cross-pole p_pair against the way a same-pole pair shares its one pole's (k, κ).
Δ ≡ 0 always for a cross-pole pair (the penalty decomposes additively across poles), so there
is no objective correction to make there either way -- but the pairwise Flow constraint is
still a genuine, useful tightening even with no Δ to correct: it enforces that the JOINT
(x1, x2) trajectory is consistent with the two components evolving via the *product* of
their own transition kernels (Definition 1: a subsystem's dynamics depend only on its own
local state and action), which q's own per-component Flow constraints do not enforce on
their own (they only pin each component's OWN marginal, never the correlation between two
components' trajectories). The cross-pole p_pair is tied to q's marginal OVER (k, κ) (summed
out), since which specific (k, κ) each pole is in has nothing to do with why this coupling is
useful here.
"""
mutable struct MasterLPOrder2
    master::MasterLP
    pp_same::Dict{Tuple{Int,Int},Dict{NTuple{7,Int},VariableRef}}    # (m1,m2) same-pole -> p_pair
    pp_cross::Dict{Tuple{Int,Int},Dict{NTuple{5,Int},VariableRef}}   # (m1,m2) cross-pole -> p_pair
    ic_same::Dict{NTuple{4,Int},ConstraintRef}     # (m1, m2, x1, x2) -> initial-condition constraint
    ic_cross::Dict{NTuple{4,Int},ConstraintRef}    # (m1, m2, x1, x2) -> initial-condition constraint
    active_k::Dict{Tuple{Int,Int},Int}             # (m1,m2) same-pole -> active k, -1 if none yet
end

"""
Builds `MasterLPOrder2`, generic in (x0, k0), for a given `(t0_cyc, H)`: every p_pair
variable at t=1 is created and, at first, FIXED to 0, and every pairwise initial-condition
constraint's right-hand side is 0. `configure_master_order2!` must be called once (with the
actual x0, k0) before the model is solved -- mirroring exactly how `build_master`/
`configure_master!` handle q's own t=1 variables and initial condition.
"""
function build_master_order2(fd::FluidData, t0_cyc::Int, H::Int,
                             Δ_pairs::Dict{Tuple{Int,Int},Matrix{Float64}})
    master = build_master(fd, t0_cyc, H)
    model = master.model
    q = master.q

    pp_same  = Dict{Tuple{Int,Int},Dict{NTuple{7,Int},VariableRef}}()
    pp_cross = Dict{Tuple{Int,Int},Dict{NTuple{5,Int},VariableRef}}()
    ic_same  = Dict{NTuple{4,Int},ConstraintRef}()
    ic_cross = Dict{NTuple{4,Int},ConstraintRef}()
    active_k = Dict{Tuple{Int,Int},Int}()

    obj_correction = AffExpr(0.0)
    for ((m1, m2), Δ) in Δ_pairs
        if fd.pole_of[m1] != fd.pole_of[m2]
            n1 = fd.n_states[m1]; n2 = fd.n_states[m2]
            P1 = fd.P[m1]; P2 = fd.P[m2]
            K01 = fd.K0[fd.pole_of[m1]]; K02 = fd.K0[fd.pole_of[m2]]

            # ── Variables p_pair[x1,x2,a1,a2,t] (no k,κ: the two poles' quotas are independent) ─
            pp = Dict{NTuple{5,Int},VariableRef}()
            for t in 1:H, a1 in 1:2, a2 in 1:2, x1 in 1:n1, x2 in 1:n2
                v = @variable(model, lower_bound = 0)
                pp[(x1, x2, a1, a2, t)] = v
                t == 1 && fix(v, 0.0; force = true)   # configure_master_order2! unfixes via the ic rhs
            end
            pp_cross[(m1, m2)] = pp

            # ── Initial condition: rhs 0 for now, set by configure_master_order2! ───────
            for x1 in 1:n1, x2 in 1:n2
                ic_cross[(m1, m2, x1, x2)] = @constraint(model,
                    sum(pp[(x1, x2, a1, a2, 1)] for a1 in 1:2, a2 in 1:2) == 0)
            end

            # ── Flow, using the product kernel p_{m1}^{a1}(x1|x1') * p_{m2}^{a2}(x2|x2') ─
            for t in 2:H
                for x1 in 1:n1, x2 in 1:n2
                    lhs = sum(pp[(x1, x2, a1, a2, t)] for a1 in 1:2, a2 in 1:2)
                    rhs = AffExpr(0.0)
                    for x1p in 1:n1, a1 in 1:2
                        prob1 = a1 == 1 ? P1[x1, x1p] : (x1 == 1 ? 1.0 : 0.0)
                        prob1 < 1e-12 && continue
                        for x2p in 1:n2, a2 in 1:2
                            prob2 = a2 == 1 ? P2[x2, x2p] : (x2 == 1 ? 1.0 : 0.0)
                            prob2 < 1e-12 && continue
                            add_to_expression!(rhs, prob1 * prob2, pp[(x1p, x2p, a1, a2, t - 1)])
                        end
                    end
                    @constraint(model, lhs == rhs)
                end
            end

            # ── Marginal consistency with q's OWN marginal over (k,κ), at every stage ───
            for t in 1:H
                for x1 in 1:n1, a1 in 1:2
                    @constraint(model,
                        sum(pp[(x1, x2, a1, a2, t)] for x2 in 1:n2, a2 in 1:2) ==
                        sum(q[(m1, x1, a1, k, κ, t)] for k in qrange(K01), κ in qrange(k)))
                end
                for x2 in 1:n2, a2 in 1:2
                    @constraint(model,
                        sum(pp[(x1, x2, a1, a2, t)] for x1 in 1:n1, a1 in 1:2) ==
                        sum(q[(m2, x2, a2, k, κ, t)] for k in qrange(K02), κ in qrange(k)))
                end
            end

            # ── Objective correction (always exactly 0 here, Δ ≡ 0 for a cross-pole pair) ─
            for t in 1:H, x1 in 1:n1, x2 in 1:n2
                add_to_expression!(obj_correction, Δ[x1, x2], pp[(x1, x2, 1, 1, t)])
            end
            continue
        end

        p = fd.pole_of[m1]; K0 = fd.K0[p]
        n1 = fd.n_states[m1]; n2 = fd.n_states[m2]
        P1 = fd.P[m1]; P2 = fd.P[m2]

        # ── Variables p_pair[x1,x2,a1,a2,k,κ,t] ────────────────────────────────────
        pp = Dict{NTuple{7,Int},VariableRef}()
        for t in 1:H, a1 in 1:2, a2 in 1:2, k in qrange(K0), κ in qrange(k), x1 in 1:n1, x2 in 1:n2
            v = @variable(model, lower_bound = 0)
            pp[(x1, x2, a1, a2, k, κ, t)] = v
            t == 1 && fix(v, 0.0; force = true)   # every k fixed at first; configure_master_order2! unfixes one
        end
        pp_same[(m1, m2)] = pp
        active_k[(m1, m2)] = -1

        # ── Initial condition: rhs 0 for now, set by configure_master_order2! ───────────
        for x1 in 1:n1, x2 in 1:n2
            ic_same[(m1, m2, x1, x2)] = @constraint(model,
                sum(pp[(x1, x2, a1, a2, k, κ, 1)] for a1 in 1:2, a2 in 1:2, k in qrange(K0), κ in qrange(k)) == 0)
        end

        # ── Flow, using the product kernel p_{m1}^{a1}(x1|x1') * p_{m2}^{a2}(x2|x2') ─
        for t in 2:H
            reset = stage_cyc(t - 1, t0_cyc) == N_PERIODS_PER_YEAR
            for x1 in 1:n1, x2 in 1:n2, kn in qrange(K0)
                lhs = sum(pp[(x1, x2, a1, a2, kn, κ, t)] for a1 in 1:2, a2 in 1:2, κ in qrange(kn))
                if reset && kn != K0
                    @constraint(model, lhs == 0)
                    continue
                end
                rhs = AffExpr(0.0)
                for x1p in 1:n1, a1 in 1:2
                    prob1 = a1 == 1 ? P1[x1, x1p] : (x1 == 1 ? 1.0 : 0.0)
                    prob1 < 1e-12 && continue
                    for x2p in 1:n2, a2 in 1:2
                        prob2 = a2 == 1 ? P2[x2, x2p] : (x2 == 1 ? 1.0 : 0.0)
                        prob2 < 1e-12 && continue
                        prob = prob1 * prob2
                        if reset
                            for k in qrange(K0), κ in qrange(k)
                                add_to_expression!(rhs, prob, pp[(x1p, x2p, a1, a2, k, κ, t - 1)])
                            end
                        else
                            for κp in 0:QUOTA_STEP:(K0 - kn)
                                add_to_expression!(rhs, prob, pp[(x1p, x2p, a1, a2, kn + κp, κp, t - 1)])
                            end
                        end
                    end
                end
                @constraint(model, lhs == rhs)
            end
        end

        # ── Marginal consistency with q, at EVERY stage (not just t = 1) ────────────
        # Σ_{x2,a2} p_pair[x1,x2,a1,a2,k,κ,t] = q[m1,x1,a1,k,κ,t], and symmetrically for
        # m2 -- this is what ties p_pair's mode split to q's actual (column-determined)
        # mode split; Flow alone does not (see the docstring above).
        for t in 1:H, k in qrange(K0), κ in qrange(k)
            for x1 in 1:n1, a1 in 1:2
                @constraint(model,
                    sum(pp[(x1, x2, a1, a2, k, κ, t)] for x2 in 1:n2, a2 in 1:2) ==
                    q[(m1, x1, a1, k, κ, t)])
            end
            for x2 in 1:n2, a2 in 1:2
                @constraint(model,
                    sum(pp[(x1, x2, a1, a2, k, κ, t)] for x1 in 1:n1, a1 in 1:2) ==
                    q[(m2, x2, a2, k, κ, t)])
            end
        end

        # ── Objective correction (mode a1=a2=1 only) ────────────────────────────────
        for t in 1:H, k in qrange(K0), κ in qrange(k), x1 in 1:n1, x2 in 1:n2
            add_to_expression!(obj_correction, Δ[x1, x2], pp[(x1, x2, 1, 1, k, κ, t)])
        end
    end
    set_objective_function(model, objective_function(model) + obj_correction)

    return MasterLPOrder2(master, pp_same, pp_cross, ic_same, ic_cross, active_k)
end

"""
    configure_master_order2!(fd, mo, x0, k0)

Points `mo` (built once by `build_master_order2`) at the state `(x0, k0)`: delegates to
`configure_master!` for the order-1 part, then updates every pairwise initial-condition
constraint's right-hand side, and re-fixes the t=1 p_pair variables of each same-pole pair so
that only its pole's `k0[p]` slice is free (mirroring `configure_master!`, applied to each
pairwise block instead of to q).
"""
function configure_master_order2!(fd::FluidData, mo::MasterLPOrder2, x0::Vector{Int}, k0::Vector{Int})
    configure_master!(fd, mo.master, x0, k0)

    for ((m1, m2, x1, x2), con) in mo.ic_cross
        set_normalized_rhs(con, (x1 == x0[m1] ? 1.0 : 0.0) * (x2 == x0[m2] ? 1.0 : 0.0))
    end
    for ((m1, m2, x1, x2), con) in mo.ic_same
        set_normalized_rhs(con, (x1 == x0[m1] ? 1.0 : 0.0) * (x2 == x0[m2] ? 1.0 : 0.0))
    end

    for ((m1, m2), pp) in mo.pp_same
        p = fd.pole_of[m1]
        k0[p] == mo.active_k[(m1, m2)] && continue
        n1 = fd.n_states[m1]; n2 = fd.n_states[m2]
        old_k = mo.active_k[(m1, m2)]
        if old_k != -1
            for a1 in 1:2, a2 in 1:2, κ in qrange(old_k), x1 in 1:n1, x2 in 1:n2
                fix(pp[(x1, x2, a1, a2, old_k, κ, 1)], 0.0; force = true)
            end
        end
        for a1 in 1:2, a2 in 1:2, κ in qrange(k0[p]), x1 in 1:n1, x2 in 1:n2
            unfix(pp[(x1, x2, a1, a2, k0[p], κ, 1)])
            set_lower_bound(pp[(x1, x2, a1, a2, k0[p], κ, 1)], 0.0)
        end
        mo.active_k[(m1, m2)] = k0[p]
    end
    return mo
end

# ─────────────────────────────────────────────────────────────────────────────
# Column generation main loop (identical to fluid_lp_cg.solve_fluid_cg, but building the
# master with the order-2 pairwise corrections; pricing and duals are unchanged since the
# pairwise variables/constraints do not involve the columns μ).
# ─────────────────────────────────────────────────────────────────────────────

"""
Solve the order-2 fluid LP by column generation, on an already-built and
`configure_master_order2!`-ed `mo::MasterLPOrder2`. Same interface as
`fluid_lp_cg.solve_fluid_cg`, operating on `mo.master.cols` so that columns (and the basis)
persist across states when `mo` itself is reused (see `make_policy_fluid_cg_order2`).
"""
function solve_fluid_cg_order2(fd::FluidData, mo::MasterLPOrder2, x0::Vector{Int}, k0::Vector{Int}, H::Int;
                               max_iter::Int = 50, tol::Float64 = 1e-6,
                               time_limit_pricing::Float64 = 60.0,
                               heuristic_extraction::Bool = true)
    t_begin = time()
    begin
        master = mo.master
        function push_col!(col)
            col in master.cols && return
            push!(master.cols, col); add_column!(fd, master, col)
        end
        if isempty(master.cols)
            push_col!(empty_column(fd))
            for m in 1:fd.M; push_col!(singleton_column(fd, m, 1)); end
        end

        iterations = 0
        converged = false
        for iter in 1:max_iter
            iterations = iter
            solve_master!(master)
            λ = consistency_duals(fd, master)
            n_new = 0
            for t in 1:H
                col, rc = solve_pricing(fd, λ[:, :, :, t]; time_limit = time_limit_pricing)
                (isnothing(col) || rc >= -tol || col in master.cols) && continue
                push_col!(col); n_new += 1
            end
            if n_new == 0
                converged = true
                break
            end
        end
        solve_master!(master)

        weights = [value(master.μ[i][1]) for i in eachindex(master.cols)]
        best = master.cols[argmax(weights)]
        if heuristic_extraction
            mode_mass(m, mode) = sum(value(master.q[(m, x0[m], mode, k0[fd.pole_of[m]], κ, 1)])
                                      for κ in qrange(k0[fd.pole_of[m]]))
            kappa_mass(p, κ) = sum(value(master.q[(fd.comps_of_pole[p][1], x0[fd.comps_of_pole[p][1]],
                                                     mode, k0[p], κ, 1)]) for mode in 1:2)
            best = extract_policy_heuristic(fd, x0, k0, mode_mass, kappa_mass)
        end
        lp_value = objective_value(master.model)
    end
    elapsed = time() - t_begin
    return best, (time = elapsed, lp_value = lp_value, iterations = iterations, converged = converged,
                  n_columns = length(master.cols), cols = master.cols)
end

# ─────────────────────────────────────────────────────────────────────────────
# Fluid policy with cache (same interface as make_policy_fluid_cg)
# ─────────────────────────────────────────────────────────────────────────────

"""
Returns a policy (x, k, t_global) -> (Column, solve_time), based on the order-2 fluid LP
(Section 4.2) solved by column generation. Same caching / logging interface as
make_policy_fluid_cg. By default only the `max_pairs = 3` highest-impact pairs (largest
max(|Δ|), see compute_Δ_pairs) are tracked, to keep the LP tractable -- the full pairwise
flow network of every mode combination (see compute_Δ_pairs' docstring and the module
header) makes tracking every C(M,2) pair prohibitively expensive on poles with several
components. Pass `pairs` (an explicit list), `max_pair_states`, or a different `max_pairs`
(e.g. `nothing` to track every pair) to override this default.

When `warm_start` is true (default), one `MasterLPOrder2` template is built per
`(t0_cyc, H_eff)` and reused/reconfigured (`configure_master_order2!`) across every state
sharing it, instead of being rebuilt from scratch (same warm-start mechanism as
`make_policy_fluid_cg`, applied to the order-1 part AND the pairwise p_pair variables alike).
`warm_start = false` rebuilds a fresh, column-less master for every state instead.
"""
function make_policy_fluid_cg_order2(fd::FluidData; H::Int = 6, warm_start::Bool = true,
                                     cache_file::Union{String,Nothing} = nothing,
                                     log_solve::Function = info -> nothing,
                                     max_iter::Int = 50, time_limit_pricing::Float64 = 60.0,
                                     pairs::Union{Vector{Tuple{Int,Int}},Nothing} = nothing,
                                     max_pair_states::Union{Int,Nothing} = nothing,
                                     max_pairs::Union{Int,Nothing} = 3,
                                     heuristic_extraction::Bool = true)
    T_total = fd.inst.n_years * N_PERIODS_PER_YEAR
    Δ_pairs = compute_Δ_pairs(fd; pairs = pairs, max_pair_states = max_pair_states, max_pairs = max_pairs)
    @printf("  order-2 formulation: %d pairwise corrections (poles: %d)\n",
            length(Δ_pairs), fd.inst.n_poles)

    cache, next_cache_id = load_cache(cache_file)
    masters = Dict{Tuple{Int,Int},MasterLPOrder2}()   # (t0_cyc, H_eff) -> template, reused iff warm_start

    function policy(x::Vector{Int}, k::Vector{Int}, t_global::Int)
        t0_cyc = (t_global - 1) % N_PERIODS_PER_YEAR + 1
        H_eff = min(H, T_total - t_global + 1)
        key = (Tuple(x), Tuple(k), t0_cyc, H_eff)
        if !haskey(cache, key)
            mo = warm_start ? get!(() -> build_master_order2(fd, t0_cyc, H_eff, Δ_pairs), masters, (t0_cyc, H_eff)) :
                               build_master_order2(fd, t0_cyc, H_eff, Δ_pairs)
            configure_master_order2!(fd, mo, x, k)
            col, info = solve_fluid_cg_order2(fd, mo, x, k, H_eff; max_iter = max_iter,
                                              time_limit_pricing = time_limit_pricing,
                                              heuristic_extraction = heuristic_extraction)
            cache[key] = (col.start, col.free_days, info.time, info.lp_value,
                          info.iterations, info.n_columns, info.converged)
            log_solve((key = key, time = info.time, lp_value = info.lp_value,
                       iterations = info.iterations, n_columns = info.n_columns,
                       converged = info.converged))
            append_cache_entry!(cache_file, next_cache_id, key, cache[key])
            return col, info.time
        end
        start, free_days = cache[key][1], cache[key][2]
        return Column(start, free_days), 0.0
    end
    return policy
end

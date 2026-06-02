# ADVECT_FLOW_IMPLICIT  Advance the flow s by one implicit step (steady driver).
#
#   s = ADVECT_FLOW_IMPLICIT(s, chemistry)
#
#   Dispatches to one of two implicit solvers, selected by the field
#   s["time_integration"]["implicit_solver"]:
#
#     "Newton"      (default) - Pseudo-transient continuation: one damped
#                               Newton step per call on the steady residual,
#                               using the sparse Jacobian L built by
#                               LINEARIZE_EQUATIONS_NO_DISCONTINUITY. The
#                               pseudo-time step (CFL) is ramped up as the
#                               residual decreases following the Switched
#                               Evolution Relaxation rule of Mulder & van Leer
#                               (1985), recovering Newton's method as CFL->inf
#                               (pseudo-transient continuation, Kelley & Keyes
#                               1998). Recommended for steady base flows.
#
#     "Relaxation"            - Legacy backward Euler solved by a relaxed
#                               fixed-point (Picard) iteration with adaptive
#                               under-relaxation. Kept for compatibility.
#
# Part of: Hypersonics Stability Julia Solver - Time Marching Module

using SparseArrays
using LinearAlgebra

# Module-level cache for the Newton Jacobian (mirrors the MATLAB `persistent`).
const _NEWTON_JAC_L   = Ref{Any}(nothing)
const _NEWTON_JAC_AGE = Ref{Int}(0)
const _NEWTON_JAC_SZ  = Ref{Int}(0)

function ADVECT_FLOW_IMPLICIT(s::Dict{String,Any}, chemistry::Dict{String,Any})

    solver = haskey(s["time_integration"], "implicit_solver") ?
             s["time_integration"]["implicit_solver"] : "Newton"

    if solver == "Newton"
        s = ADVECT_FLOW_IMPLICIT_NEWTON(s, chemistry)
    elseif solver == "Relaxation"
        s = ADVECT_FLOW_IMPLICIT_RELAXATION(s, chemistry)
    else
        error("Unknown s[\"time_integration\"][\"implicit_solver\"] = '$(solver)' " *
              "(use \"Newton\" or \"Relaxation\").")
    end

    return s
end


# =========================================================================
#  Newton / pseudo-transient-continuation solver
# =========================================================================
#
#   Solves, for the increment dq of the interior conservative variables,
#
#       (I/dtau - L) dq = f(q^k),     q^{k+1} = q^k + alpha*dq
#
#   where f(q) = NON_LINEAR_DYNAMICS_NO_DISCONTINUITY (the volume-normalised
#   flux residual, masked by the flow cells when shock fitting is active) and
#   L = df/dq is the sparse Jacobian from LINEARIZE_EQUATIONS_NO_DISCONTINUITY.
#   The pseudo-time step dtau is the explicit CFL-stable step scaled by the
#   ramped CFL number; alpha is a positivity safeguard (<=1). At the fixed
#   point f(q)=0 (steady state) and, as dtau->inf, this is Newton on f(q)=0.
function ADVECT_FLOW_IMPLICIT_NEWTON(s::Dict{String,Any}, chemistry::Dict{String,Any})

    Nchi = s["mesh"]["Nchi"]
    Neta = s["mesh"]["Neta"]
    n    = Nchi * Neta
    n4   = 4 * n

    get_opt(d, k, default) = haskey(d, k) ? d[k] : default
    rmsf(x) = isempty(x) ? 0.0 : sqrt(sum(abs2, x) / length(x))

    ## Solver / ramp options (defaults applied if absent from the input file)
    ti0       = s["time_integration"]
    CFL_0     = ti0["CFL"]
    CFL_max   = get_opt(ti0, "CFL_max",            1e5)
    p_ramp    = get_opt(ti0, "CFL_ramp_exponent",  1.0)
    growth    = get_opt(ti0, "CFL_growth",         2.0)
    jac_every = get_opt(ti0, "jacobian_refresh",   1)
    rho_frac  = get_opt(ti0, "newton_min_rho_fraction", 0.2)

    if !s["chemistry"]["chemical_equilibrium"]
        @warn "Newton implicit solver linearises only (rho, rho_u, rho_v, rho_E); " *
              "non-equilibrium fields (gamma_star, cv_star) are held frozen. Use " *
              "Explicit_RK4 (or implicit_solver=\"Relaxation\") for non-equilibrium chemistry." maxlog=1
    end

    ## 1) Flow cells + nonlinear steady residual f(q^k)
    s = UPDATE_FLOW_CELLS(s, chemistry)

    s["linearize"] = false                                # full physics in the residual
    s = NON_LINEAR_DYNAMICS_NO_DISCONTINUITY(s, chemistry)   # sets s["flux"], thermo, shock speed

    if s["shock"]["enabled"]
        fc = s["shock"]["flow_cells"] .> 0                # BitMatrix Nchi x Neta
    else
        fc = trues(Nchi, Neta)
    end

    flux = s["flux"]
    fr   = flux["rho"]::Matrix{Float64}
    fru  = flux["rho_u"]::Matrix{Float64}
    frv  = flux["rho_v"]::Matrix{Float64}
    frE  = flux["rho_E"]::Matrix{Float64}

    R_vec = vcat(vec(fr .* fc), vec(fru .* fc), vec(frv .* fc), vec(frE .* fc))

    ## Interior conservative state (copies) for residual norms + positivity
    var    = s["var"]
    rho    = var["rho"]::Matrix{Float64}
    rho_u  = var["rho_u"]::Matrix{Float64}
    rho_v  = var["rho_v"]::Matrix{Float64}
    rho_E  = var["rho_E"]::Matrix{Float64}
    rho_i  = rho[2:end-1, 2:end-1]
    rhou_i = rho_u[2:end-1, 2:end-1]
    rhov_i = rho_v[2:end-1, 2:end-1]
    rhoE_i = rho_E[2:end-1, 2:end-1]

    residual = Dict{String,Any}(
        "rho"   => rmsf(fr[fc])  / max(rmsf(rho_i[fc]),  eps()),
        "rho_u" => rmsf(fru[fc]) / max(rmsf(rhou_i[fc]), eps()),
        "rho_v" => rmsf(frv[fc]) / max(rmsf(rhov_i[fc]), eps()),
        "rho_E" => rmsf(frE[fc]) / max(rmsf(rhoE_i[fc]), eps()))

    ## 2) Explicit-stable time step + SER ramp of the pseudo-time step
    s = CFL_TIMESTEP(s)                                   # dt at the base CFL
    ti = s["time_integration"]                           # re-fetch (s may have been rebound)
    dt_explicit = ti["dt"]

    Rn = norm(R_vec) / sqrt(n4)
    if !haskey(ti, "residual_ref") || !isfinite(ti["residual_ref"]) || ti["residual_ref"] <= 0
        ti["residual_ref"] = Rn
        ti["CFL_current"]  = CFL_0
    end
    R0       = ti["residual_ref"]
    CFL_prev = ti["CFL_current"]

    CFL_cur = CFL_0 * (R0 / max(Rn, floatmin(Float64)))^p_ramp   # Switched Evolution Relaxation
    CFL_cur = min(CFL_cur, CFL_max)                              # ceiling
    CFL_cur = min(CFL_cur, growth * CFL_prev)                    # limit growth per step
    CFL_cur = max(CFL_cur, CFL_0)                                # never below base
    ti["CFL_current"] = CFL_cur

    dt_pstc = dt_explicit * (CFL_cur / CFL_0)                    # ramped pseudo-time step

    ## 3) Sparse Jacobian L = df/dq (rebuilt every jac_every steps)
    L = NEWTON_JACOBIAN(s, chemistry, jac_every, ti["iter"] == 0)

    ## 4) Solve (I/dtau - L) dq = f(q^k) with SuiteSparse sparse LU (UMFPACK)
    M  = spdiagm(0 => fill(1.0 / dt_pstc, n4)) - L
    dq = M \ R_vec

    if !all(isfinite, dq)
        @warn "Newton linear solve produced non-finite values; skipping update " *
              "this step (reduce CFL_max or CFL_growth)." maxlog=5
        dq = zeros(n4)
    end

    ## 5) Positivity-safeguarded update of the interior conservative variables
    d_rho   = reshape(dq[0*n+1:1*n], Nchi, Neta) .* fc
    d_rho_u = reshape(dq[1*n+1:2*n], Nchi, Neta) .* fc
    d_rho_v = reshape(dq[2*n+1:3*n], Nchi, Neta) .* fc
    d_rho_E = reshape(dq[3*n+1:4*n], Nchi, Neta) .* fc

    alpha = 1.0
    neg   = d_rho .< 0
    if any(neg)
        bound = (rho_frac - 1.0) .* rho_i[neg] ./ d_rho[neg]     # keep rho >= rho_frac*rho_old
        alpha = clamp(minimum(bound), 0.0, 1.0)
    end

    rho[2:end-1, 2:end-1]   .= rho_i  .+ alpha .* d_rho
    rho_u[2:end-1, 2:end-1] .= rhou_i .+ alpha .* d_rho_u
    rho_v[2:end-1, 2:end-1] .= rhov_i .+ alpha .* d_rho_v
    rho_E[2:end-1, 2:end-1] .= rhoE_i .+ alpha .* d_rho_E

    ## 6) Advance the shock explicitly on the (un-ramped) stable time scale
    if s["shock"]["enabled"]
        dt_shock = dt_explicit * s["shock"]["relaxation"]
        s["shock"]["points_x"] = s["shock"]["points_x"] .+ s["shock"]["speed_x"] .* dt_shock
        s["shock"]["points_y"] = s["shock"]["points_y"] .+ s["shock"]["speed_y"] .* dt_shock
    end

    ## 7) Post-step: boundary conditions, thermodynamics, density limiter
    s = UPDATE_THERMODYNAMIC_PROPERTIES(s, chemistry)
    if s["shock"]["enabled"]
        s = UPDATE_SHOCK_BC(s, chemistry)
    end
    s = APPLY_BOUNDARY_CONDITIONS(s, chemistry)
    s = MIN_RHO(s)

    ## 8) Advance counters and store diagnostics (re-fetch: post-step may rebind s)
    ti = s["time_integration"]
    ti["dt"]       = dt_pstc
    ti["t"]        = ti["t"] + dt_pstc
    ti["iter"]     = ti["iter"] + 1
    ti["residual"] = residual
    s["count_implicit_iterations"] = 1     # one Newton step per call

    return s
end


# NEWTON_JACOBIAN  Cached sparse flow Jacobian L = df/dq (A11 block only).
#
#   Rebuilds L (via LINEARIZE_EQUATIONS_NO_DISCONTINUITY, with the shock DOFs
#   disabled and chemistry frozen during the finite differences) every
#   jac_every calls, on a grid-size change, or when reset is true. Between
#   rebuilds the cached matrix is reused (modified Newton).
function NEWTON_JACOBIAN(s::Dict{String,Any}, chemistry::Dict{String,Any}, jac_every, reset::Bool)

    n4 = 4 * s["mesh"]["Nchi"] * s["mesh"]["Neta"]

    if reset
        _NEWTON_JAC_L[]   = nothing
        _NEWTON_JAC_AGE[] = 0
        _NEWTON_JAC_SZ[]  = 0
    end

    rebuild = _NEWTON_JAC_L[] === nothing || _NEWTON_JAC_SZ[] != n4 ||
              mod(_NEWTON_JAC_AGE[], max(1, jac_every)) == 0

    if rebuild
        s_lin = deepcopy(s)
        s_lin["linearize"] = true                        # frozen-chemistry Jacobian
        if !haskey(s_lin, "stability_analysis") || !isa(s_lin["stability_analysis"], Dict)
            s_lin["stability_analysis"] = Dict{String,Any}()
        end
        s_lin["stability_analysis"]["perturb_shock"] = false   # flow-flow (A11) block only
        if !haskey(s_lin["stability_analysis"], "perturbation_magnitude")
            s_lin["stability_analysis"]["perturbation_magnitude"] = 1e-6
        end
        _NEWTON_JAC_L[]   = LINEARIZE_EQUATIONS_NO_DISCONTINUITY(s_lin, chemistry)
        _NEWTON_JAC_SZ[]  = n4
        _NEWTON_JAC_AGE[] = 0
    end

    _NEWTON_JAC_AGE[] += 1
    return _NEWTON_JAC_L[]
end


# =========================================================================
#  Legacy relaxed fixed-point (Picard) backward-Euler solver
# =========================================================================
function ADVECT_FLOW_IMPLICIT_RELAXATION(s::Dict{String,Any}, chemistry::Dict{String,Any})

    ## Initialise implicit iteration
    residual          = 100.0
    residual_previous = 0.0

    s = UPDATE_FLOW_CELLS(s, chemistry)
    s = CFL_TIMESTEP(s) # Recompute CFL-based time step for the next iteration
    solution_temp     = deepcopy(s)
    count_implicit_iterations = 0

    ## Implicit iteration loop
    while residual > s["time_integration"]["tolerance"] && count_implicit_iterations < s["time_integration"]["max_iter_implicit"]
        s = ADAPT_RELAXATION(residual, residual_previous, s)

        # Compute PDE right-hand side
        solution_temp = PDE(solution_temp, chemistry)

        # Implicit time integration and residual evaluation
        solution_temp, residual = INTEGRATION_IMPLICIT(s, solution_temp)
        count_implicit_iterations = count_implicit_iterations + 1
        residual_previous = residual
    end
    s = deepcopy(solution_temp)

    ## Update boundary conditions and thermodynamic properties
    s = UPDATE_THERMODYNAMIC_PROPERTIES(s, chemistry)
    if s["shock"]["enabled"]
        s = UPDATE_SHOCK_BC(s, chemistry)
    end
    s = APPLY_BOUNDARY_CONDITIONS(s, chemistry)

    ## Enforce minimum density limiter
    s = MIN_RHO(s)

    ## Advance time and iteration counters
    s["time_integration"]["t"]    = s["time_integration"]["t"] + s["time_integration"]["dt"]
    s["time_integration"]["iter"] = s["time_integration"]["iter"] + 1
    s["count_implicit_iterations"] = count_implicit_iterations

    return s
end


# ADAPT_RELAXATION  Adjust the variable relaxation factor based on residual trend.
#
#   Uses a sigmoid mapping of the residual change to adapt the relaxation
#   factor between 0.5 and 0.99 for the implicit iteration.
function ADAPT_RELAXATION(residual::Float64, residual_previous::Float64, s::Dict{String,Any})

    C    = 0.5
    a    = C * (residual - residual_previous)
    temp = 1.7 * s["time_integration"]["relax_factor"] / (1 + exp(-a))
    m    = min(0.99, temp)
    s["relax_factor_variable"] = max(0.5, m)

    return s
end

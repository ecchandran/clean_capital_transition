# Clean Capital Transition Repository

This repository contains code used for experiments in my senior thesis at Princeton University, _Investment Allocation and Political Economy in the Climate Transition: A Physics-Informed Macroeconomic Model using Stochastic Optimal Control and Deep Learning_, advised by Prof. Jonathan Payne. The PDF of the thesis can be found [here](thesis/Chandran_Senior_Thesis_2025.pdf).

I completed all work for this thesis by April 28, 2025. This README replaces the original `README.txt` I committed prior to the thesis's submission and cleans up some of its typos; I have not edited the repository's code since submitting the thesis.

## Code Structure

Below is a description of each file in this repository, including example commands for executables. Each `--tag` names an output directory: `model_3.py`, `cp_dl.py`, and `cp_fd.py` write to `<tag>_M3DL`, `<tag>_CPDL`, and `<tag>_CPFD` respectively, and the plotting scripts take the same bare tag.

```
environment.py : defines parameters and settings for all models.

model_3.py : primary technical result, global solution algorithm for decentralized equilibrium
    requires the following command line arguments:
    --tag <tag used to name the output directory>
    --regime <1 or 2>
    example:
    python model_3.py --regime 1 --tag solution_dir_regime_1

plot_solo.py : plotting script for individual results
    requires the following command line arguments:
    --tag <tag of existing results>
    example:
    python plot_solo.py --tag solution_dir_regime_1

plot_compare.py : plotting script to compare two solutions
    requires the following command line arguments:
    --tag_a <tag of results from model a> --tag_b <tag of results from model b>
    example:
    python plot_compare.py --tag_a solution_dir_regime_1 --tag_b solution_dir_regime_2

cp_dl.py : deep learning global solution algorithm for central-planner problem
    requires the following command line arguments:
    --tag <tag used to name the output directory>
    example:
    python cp_dl.py --tag solution_dir_cp_dl

cp_fd.py : finite difference global solution algorithm for central-planner problem
    requires the following command line arguments:
    --tag <tag used to name the output directory>
    example:
    python cp_fd.py --tag solution_dir_cp_fd

plot_cp.py : original plotting script for central planner models only [Discontinued]
    requires the following command line arguments:
    --tag <tag of existing results>
    example:
    python plot_cp.py --tag solution_dir_cp_fd

lecture_replication_and_old_model/planner_problem_implementation.ipynb : algorithm and plotting for original one-state-variable problem in Appendix F

lecture_replication_and_old_model/eco529_L4_replication.ipynb : replication of lecture note model in Appendix C (see link below)
```

The lecture note model replicated in `eco529_L4_replication.ipynb` is [Chapter 4 of the PrincetonECO529 lecture notes](https://github.com/goutham-fin/PrincetonECO529/blob/main/Lecture%20note%20chapters/Chapter%204.ipynb).

Note that `environment.py` contains the calibrated parameters used in Sections 4.2–4.5 of the thesis. Reproducing the central-planner comparison in Section 4.1.1 requires setting the "dummy" parameters listed there.

## Known Errata

I found the following issues after submission with the help of AI review. I have left the code unchanged so that it matches the version used for the thesis.

1. **τ sampling flag defaults to on (`model_3.py`, lines 40 and 82).** `--ERGODIC_SAMPLE_TAU` is declared with `type=bool` and `default=env.alpha` (0.25, which is truthy), and line 82 copies it into `env.ERGODIC_SAMPLE_TAU`. As a result, `model_3.py` samples τ from the ergodic O-U distribution by default, while `environment.py` sets `ERGODIC_SAMPLE_TAU = False` and the thesis (Algorithm 4, Appendix D.2) describes uniform sampling of τ. Because of `type=bool`, passing `--ERGODIC_SAMPLE_TAU False` also evaluates to `True`. Running the code as-is uses the configuration that produced the thesis's results; to run the method as described in the thesis, set `env.ERGODIC_SAMPLE_TAU = False` after line 82 (or replace line 82). When σ_τ = 0 (constant tax), the ergodic branch returns τ₀ for every default sample, so W away from τ₀ is learned only from actively sampled points. With α = σ_τ = 0 the HJBE has no τ-derivatives, so values at τ = τ₀ (the values reported in the thesis) are in principle unaffected. Relatedly, the parenthetical note after Algorithm 4 should give the ergodic distribution as N(τ₀, σ_τ²/2α).

2. **Analytic steady-state benchmark (Section 4.1.1).** The thesis reports the analytic value W(η=1, H=0) = −36.7. This was computed with an earlier version of `W_m1H0` in `environment.py` that omitted the −1 in ι₁ = (Aβ − 1)/(φ + β). Eq. (200) in Appendix D.1.1 is correct; evaluated with the Section 4.1.1 parameters, it gives **W(1,0) = −34.47**. Against this value, the deep-learning solution (−34.5) has relative error of about 0.001 and normalized difference below 0.001, and the finite-difference solution (−33.3) has relative error of about 0.03. The thesis's reported relative errors of 0.09 and 0.06, and the 4% normalized difference, are therefore not correct. These figures are repeated in the abstract ("4% normalized difference from an analytic boundary value") and in the conclusion of Section 5 ("6% relative error").

3. **Consumption floor in the finite-difference solver (`cp_fd.py`, lines 88–89, 157, and 242).** The finite-difference utility uses `log(clip(c, FD_LOG_ARG_CLIP))` with `FD_LOG_ARG_CLIP = 0.1`. (The 10⁻⁴ log-argument clip stated in Appendix D.2.10 applies only to the deep-learning code.) At the steady state, consumption per unit capital is A₁ − ι₁ ≈ 0.093, so the floor binds there; with the Section 4.1.1 parameters, consumption per unit capital is below 0.1 even at H = 0 for both capital types, so the floor likely binds across nearly all of the grid. With the floor applied, the analytic steady-state value is −33.09, which matches the finite-difference value (−33.3). The floor likely accounts for a substantial part of the 14% mean relative difference between the finite-difference and deep-learning solutions in Section 4.1.1, although I have not quantified this.

4. **Diffusion term in the decentralized HJBE (`model_3.py`, line 1194; thesis Eq. (∗87) and Appendix D.2.9).** The code uses `-(sigma_n_a**2 + sigma_n_b**2) / rho`, matching the HJBE as printed in Eq. (∗87) and Appendix D.2.9. However, the HJBE for V earlier in Section 3.3.7 contains ½n²((σⁿ_a)² + (σⁿ_b)²)∂ₙₙV with V = log(n)/ρ + W, which evaluates to −((σⁿ_a)² + (σⁿ_b)²)/(2ρ). Both the printed equation and the code therefore omit a factor of ½. The first-order condition for θ (Eqs. (88) and (∗89)) does include the ½. Because σⁿ comes from the trained equilibrium network, this error affects only the value function W, not the equilibrium objects or the simulations. It adds a negative bias to W that grows with wealth volatility, which matters most under a stochastic tax, so it may overstate the reduction in agent value from the constant tax to the stochastic tax. I have not quantified the effect.

5. **Regime 2 at the η = 0 boundary (`model_3.py`, lines 355–361 and 686).** 1% of default training samples (in both equilibrium and value-function training, lines 1426 and 1730) are placed exactly at η = 0, where `iota_g = tau * safe_divide(1-eta, eta) * A_k2_val` evaluates to roughly τ · 10⁹ · A_k2. This gives Φ(ι_g) ≈ 9, so θ clamps to 1 and μⁿ/ρ ≈ 220 at those points. After epoch 5, when active sampling adds another 1,000 points to each mini-batch, these η = 0 points make up 0.5% of the batch. A back-of-envelope estimate of their contribution to the batch MSE is then about 2.6 × 10², close to the minimum regime-2 HJBE loss of roughly 2.5 × 10² reported in Section 4.4.1, so these points likely account for most of that loss.

6. **Gradient clipping.** Appendix D.2.10 states that gradients are scaled to norm at most 10.0. `model_3.py` uses `max_norm=10.0` only in value-function pretraining (line 1280) but `max_norm=1.0` in the main equilibrium and value-function training loops (lines 1519 and 1760). (`cp_dl.py` uses `max_norm=10.0` throughout.)

7. **Abstract vs. body (thesis text).** The abstract reports an 8% normalized reduction in agent value from the optimal constant tax to the stochastic tax. The body (discussion of Figure 9) reports 7.3%, which is the computed value.

8. **η boundary conditions in levels (`model_3.py`, lines 255–282).** Appendix D.2.6 gives the boundary conditions as q·μ_q = … and q·σ_q = …, but `boundary_forward` assigns these right-hand sides directly to μ_q and σ_q. The consistency conditions (lines 896–913) treat μ_q and σ_q as relative drift and volatility, so the boundary targets are too large in magnitude by a factor of q (about 1.13–1.15 at H = 0 with the calibrated parameters, approaching 1 at H = 8). I expect this effect to be small.

9. **Regime 2 τ-volatility boundary condition (`model_3.py`, lines 273–276; Appendix D.2.6).** At η = 0 in regime 2, the boundary target is σ_q2_b = −φA_k2/(1 + φρ). The consistency condition (line 913) instead implies q₂σ_q2_b = σ_τ ∂q₂/∂τ = −φA_k2σ_τ/(1 + φρ), so the target omits the factor σ_τ (and the division by q₂). Appendix D.2.6 prints this condition as q₂σ_q2_b = −φA_k2/(1 + φρ), so the thesis text also omits σ_τ. With the thesis's stochastic-tax parameters, the target is about 4 times too large (q₂/σ_τ ≈ 3.7–4.6), and it is nonzero even when σ_τ = 0. The μ_q2 boundary target in both the code and Appendix D.2.6 likewise omits the term (∂q₂/∂τ) · α(τ₀ − τ). This affects regime 2 only and may contribute to its poor equilibrium fit; I have not quantified the effect.

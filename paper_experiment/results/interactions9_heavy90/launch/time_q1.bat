@echo off
set GUROBI_THREADS=2
set QUOTA_STEP=1
set PARAMSET=heavy_pumps90
cd /d C:\Users\sodel\MultiHorizonMaintenanceModel
"C:\Users\sodel\.julia\juliaup\julia-1.12.7+0.x64.w64.mingw32\bin\julia.exe" --project=. paper_experiment/time9.jl 1 > "C:\Users\sodel\MultiHorizonMaintenanceModel\paper_experiment\results\interactions9_heavy90\timing\time_S1_q1.log" 2>&1 < NUL

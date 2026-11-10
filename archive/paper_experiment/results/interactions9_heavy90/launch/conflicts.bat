@echo off
set GUROBI_THREADS=2
set QUOTA_STEP=15
set PARAMSET=heavy_pumps90
cd /d C:\Users\sodel\MultiHorizonMaintenanceModel
"C:\Users\sodel\.julia\juliaup\julia-1.12.7+0.x64.w64.mingw32\bin\julia.exe" --project=. paper_experiment/conflicts9.jl 100 paper_experiment/results/interactions9_heavy90 2 2 > "C:\Users\sodel\MultiHorizonMaintenanceModel\paper_experiment\results\interactions9_heavy90\conflicts.out.log" 2> "C:\Users\sodel\MultiHorizonMaintenanceModel\paper_experiment\results\interactions9_heavy90\conflicts.err.log" < NUL

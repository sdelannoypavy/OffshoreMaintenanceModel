@echo off
set GUROBI_THREADS=3
set QUOTA_STEP=15
set MTBF_ALL=20
cd /d C:\Users\sodel\MultiHorizonMaintenanceModel
"C:\Users\sodel\.julia\juliaup\julia-1.12.7+0.x64.w64.mingw32\bin\julia.exe" --project=. paper_experiment/interactions9.jl 3 10 10 paper_experiment/results/interactions9_mtbf20 > "C:\Users\sodel\MultiHorizonMaintenanceModel\paper_experiment\results\interactions9_mtbf20\S3_10.out.log" 2> "C:\Users\sodel\MultiHorizonMaintenanceModel\paper_experiment\results\interactions9_mtbf20\S3_10.err.log" < NUL

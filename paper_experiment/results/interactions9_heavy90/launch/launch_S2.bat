@echo off
rem Relaunch the two-substation Full fluid LP + CG runs (PARAMSET=heavy_pumps90), each resumes from its cache
start "S2_14" /min cmd /c "%~dp0S2_14.bat"
start "S2_20" /min cmd /c "%~dp0S2_20.bat"
start "S2_24" /min cmd /c "%~dp0S2_24.bat"
start "S2_25" /min cmd /c "%~dp0S2_25.bat"
start "S2_34" /min cmd /c "%~dp0S2_34.bat"
start "S2_37" /min cmd /c "%~dp0S2_37.bat"

@echo off
rem Relaunch all the MTBF=20 interaction tests (each resumes from its files), in minimized windows
start "conflicts3" /min cmd /c "%~dp0conflicts3.bat"
start "S3_1" /min cmd /c "%~dp0S3_1.bat"
start "S3_2" /min cmd /c "%~dp0S3_2.bat"
start "S3_3" /min cmd /c "%~dp0S3_3.bat"
start "S3_5" /min cmd /c "%~dp0S3_5.bat"
start "S3_7" /min cmd /c "%~dp0S3_7.bat"
start "S3_10" /min cmd /c "%~dp0S3_10.bat"
start "S3_11" /min cmd /c "%~dp0S3_11.bat"
start "S2_7" /min cmd /c "%~dp0S2_7.bat"
start "S2_10" /min cmd /c "%~dp0S2_10.bat"
start "S2_25" /min cmd /c "%~dp0S2_25.bat"
start "S2_41" /min cmd /c "%~dp0S2_41.bat"

@echo off
rem Relaunch the S=3 Full fluid LP + CG runs (MTBF=20), each resumes from its cache
start "S3_1" /min cmd /c "%~dp0S3_1.bat"
start "S3_2" /min cmd /c "%~dp0S3_2.bat"
start "S3_3" /min cmd /c "%~dp0S3_3.bat"
start "S3_5" /min cmd /c "%~dp0S3_5.bat"
start "S3_7" /min cmd /c "%~dp0S3_7.bat"
start "S3_10" /min cmd /c "%~dp0S3_10.bat"
start "S3_11" /min cmd /c "%~dp0S3_11.bat"

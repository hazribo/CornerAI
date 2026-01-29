@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

call .\venv\Scripts\activate.bat
REM Set the raw fastf1 path to target:
if not exist "..\data\raw\historical\2025" mkdir ..\data\raw\historical\2025
SET target_dir= "..\data\raw\historical\2025"

:loop
REM Count number of subfolders:
SET folder_count=0
FOR /D %%d IN ("%target_dir%\*") DO (
    SET /A folder_count+=1
)

echo Found !folder_count! folders.

REM Check for 24 folders in 2025 - if true, completed:
IF !folder_count! GEQ 24 (
    echo Reached 24 folders in 2025 folder. Stopping.
    GOTO end
)

REM Run get_fastf1.py until timeout:
echo Running get_fastf1.py...
python ..\src\data_collection\get_fastf1.py

GOTO loop

:end
echo Done.
pause
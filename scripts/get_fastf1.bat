@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Set the path to check
SET target_dir=..\data\raw\historical\2025

:loop
REM Count number of subfolders
SET folder_count=0
FOR /D %%d IN ("%target_dir%\*") DO (
    SET /A folder_count+=1
)

echo Found !folder_count! folders.

REM Check if we have 24 folders (23 FOR NOW AS ABU DHABI HAS NOT BEEN COMPLETED)
IF !folder_count! GEQ 23 (
    echo Reached 23 folders in 2025 folder. Stopping.
    GOTO end
)

REM Run your Python script
echo Running get_fastf1.py...
python ..\src\data_collection\get_fastf1.py

REM Optional: wait a few seconds before checking again
timeout /t 2 /nobreak >nul

GOTO loop

:end
echo Done.
pause
@echo off
setlocal

if "%1"=="clean" (
    python ..\tools\build_docs.py --clean
) else if "%1"=="doxygen" (
    python ..\tools\build_docs.py --doxygen-only
) else (
    python ..\tools\build_docs.py
)

exit /b %errorlevel%

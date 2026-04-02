@echo off
echo Starting Chatbot Server...
echo.

REM Activate environment
call venv\Scripts\activate

REM Start server
python server.py

pause

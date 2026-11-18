@echo off
cd /d "%~dp0"
python manage.py test api ui -v 2
exit /b %ERRORLEVEL%

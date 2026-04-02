Write-Host "=== Installing Python 3.11 environment for ChatBot ==="

# Check for Python 3.11
$python = py -3.11 --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Python 3.11 is not installed. Please install Python 3.11.7 from python.org and run this script again."
    exit
}

Write-Host "Creating virtual environment..."
py -3.11 -m venv venv

Write-Host "Activating environment..."
& .\venv\Scripts\activate

Write-Host "Upgrading pip..."
pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

Write-Host "=== Setup Complete ==="
Write-Host "Run your server using:"
Write-Host "   .\venv\Scripts\activate"
Write-Host "   python flask_server.py"

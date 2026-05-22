import os
import sys
import webbrowser
from threading import Timer
import streamlit.web.cli as stcli

def open_browser():
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    print("Launching...")
    
    sys.argv = [
        "streamlit", 
        "run", 
        "core/main.py", 
        "--global.developmentMode=false", 
        "--logger.level=error"
    ]        
    
    Timer(1.5, open_browser).start()
    sys.exit(stcli.main())

import os
import sys
import webbrowser
from threading import Timer
import streamlit.web.cli as stcli
from rich.console import Console
import time

console = Console()

def open_browser():
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        os.chdir(sys._MEIPASS)

    console.print(f"\n[bold]Craton Studio[/bold]")
    with console.status("[dim]Launching Craton Studio server...[/dim]", spinner="dots"):
        sys.argv = [
            "streamlit", 
            "run", 
            "core/main.py", 
            "--global.developmentMode=false", 
            "--logger.level=error"
        ]        
        time.sleep(1)
    console.print("[green]✔[/green] [dim]Server active[/dim]\n")
    Timer(1.5, open_browser).start()
    sys.exit(stcli.main())

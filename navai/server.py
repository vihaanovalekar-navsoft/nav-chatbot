"""Start the NavAI server and open the browser."""

import os
import threading
import time
import webbrowser

import psutil
import uvicorn

PORT = 8000
URL = f"http://127.0.0.1:{PORT}"


def clear_port(port):
    """Stop processes listening on `port` so the server can bind."""
    pids = {
        connection.pid
        for connection in psutil.net_connections(kind="inet")
        if connection.laddr
        and connection.laddr.port == port
        and connection.status == psutil.CONN_LISTEN
        and connection.pid
        and connection.pid != os.getpid()
    }
    for pid in pids:
        try:
            process = psutil.Process(pid)
            process.terminate()
            process.wait(timeout=3)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            print(f"Cannot stop PID {pid} on port {port}; run as administrator.")
            continue
        except psutil.TimeoutExpired:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
            continue
        print(f"Stopped PID {pid} on port {port}.")


def open_browser_when_started(server):
    """Open the page once uvicorn has bound the socket."""
    while not server.started and not server.should_exit:
        time.sleep(0.05)
    if server.started:
        print(f"NavAI Assistant running at {URL}")
        webbrowser.open(URL)


def main(app):
    """Run `app` on port 8000 and open the browser after startup."""
    clear_port(PORT)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT))
    threading.Thread(
        target=open_browser_when_started, args=(server,), daemon=True
    ).start()
    server.run()

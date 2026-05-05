"""
Map Server – WebSocket live map streaming
==========================================
Minimal HTTP + WebSocket server for streaming the SLAM occupancy grid
to a browser-based viewer.  Uses only Python stdlib (no pip packages).

Implements RFC 6455 WebSocket handshake + framing from scratch.
"""

import base64
import hashlib
import http.server
import json
import os
import socket
import struct
import threading
import time
import webbrowser


class MapServer:
    """
    Background server that:
      1. Serves map_viewer.html on HTTP (port 8080)
      2. Accepts WebSocket connections (port 8765)
      3. Pushes map updates to all connected clients
    """

    def __init__(self, http_port=8080, ws_port=8765, auto_open=True):
        self.http_port = http_port
        self.ws_port = ws_port
        self.auto_open = auto_open

        self._ws_clients = []          # list of connected sockets
        self._ws_clients_lock = threading.Lock()
        self._running = False

        # Data to be sent (set by the controller)
        self._pending_data = None
        self._data_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API (called from the Webots controller thread)
    # ------------------------------------------------------------------

    def start(self):
        """Start HTTP and WebSocket servers in background threads."""
        self._running = True

        # HTTP server thread
        t_http = threading.Thread(target=self._run_http, daemon=True)
        t_http.start()

        # WebSocket accept thread
        t_ws = threading.Thread(target=self._run_ws_accept, daemon=True)
        t_ws.start()

        # WebSocket broadcast thread
        t_bc = threading.Thread(target=self._run_broadcast, daemon=True)
        t_bc.start()

        # Open browser after a short delay
        if self.auto_open:
            def _open():
                time.sleep(1.5)
                webbrowser.open(f"http://localhost:{self.http_port}")
            threading.Thread(target=_open, daemon=True).start()

        print(f"[MapServer] HTTP  → http://localhost:{self.http_port}")
        print(f"[MapServer] WS    → ws://localhost:{self.ws_port}")

    def stop(self):
        self._running = False

    def push_update(self, map_png_bytes, state_dict):
        """
        Queue a map update for broadcast.

        Parameters
        ----------
        map_png_bytes : bytes
            PNG-encoded occupancy grid image.
        state_dict : dict
            Metadata (robot position, trajectory, lidar points, etc.)
        """
        with self._data_lock:
            self._pending_data = (map_png_bytes, state_dict)

    # ------------------------------------------------------------------
    # HTTP Server
    # ------------------------------------------------------------------

    def _run_http(self):
        """Serve map_viewer.html from the controller directory."""
        server_dir = os.path.dirname(os.path.abspath(__file__))

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=server_dir, **kwargs)

            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.path = "/map_viewer.html"
                return super().do_GET()

            def log_message(self, format, *args):
                pass  # silence HTTP logs

        try:
            httpd = http.server.HTTPServer(("0.0.0.0", self.http_port), Handler)
            httpd.timeout = 1
            while self._running:
                httpd.handle_request()
        except Exception as e:
            print(f"[MapServer] HTTP error: {e}")

    # ------------------------------------------------------------------
    # WebSocket Server (RFC 6455 – pure stdlib)
    # ------------------------------------------------------------------

    def _run_ws_accept(self):
        """Accept incoming WebSocket connections."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)
        srv.bind(("0.0.0.0", self.ws_port))
        srv.listen(5)

        while self._running:
            try:
                client, addr = srv.accept()
                client.settimeout(5.0)
                if self._ws_handshake(client):
                    client.settimeout(None)
                    with self._ws_clients_lock:
                        self._ws_clients.append(client)
                else:
                    client.close()
            except socket.timeout:
                continue
            except Exception:
                continue

        srv.close()

    def _ws_handshake(self, client):
        """Perform the WebSocket opening handshake."""
        try:
            data = client.recv(4096).decode("utf-8", errors="ignore")
            if "Upgrade: websocket" not in data and "upgrade: websocket" not in data:
                return False

            # Extract Sec-WebSocket-Key
            key = None
            for line in data.split("\r\n"):
                if line.lower().startswith("sec-websocket-key"):
                    key = line.split(":", 1)[1].strip()
                    break

            if not key:
                return False

            # Compute accept key
            GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
            accept = base64.b64encode(
                hashlib.sha1((key + GUID).encode()).digest()
            ).decode()

            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            )
            client.sendall(response.encode())
            return True
        except Exception:
            return False

    def _ws_send(self, client, data):
        """Send a WebSocket text frame."""
        payload = data.encode("utf-8") if isinstance(data, str) else data
        length = len(payload)
        header = bytearray()
        header.append(0x81)  # FIN + text opcode

        if length < 126:
            header.append(length)
        elif length < 65536:
            header.append(126)
            header.extend(struct.pack(">H", length))
        else:
            header.append(127)
            header.extend(struct.pack(">Q", length))

        client.sendall(bytes(header) + payload)

    def _run_broadcast(self):
        """Broadcast map updates to all connected WebSocket clients."""
        while self._running:
            time.sleep(0.3)  # ~3 updates per second

            with self._data_lock:
                data = self._pending_data
                self._pending_data = None

            if data is None:
                continue

            png_bytes, state = data

            # Build JSON message with base64-encoded PNG
            msg = json.dumps({
                "map": base64.b64encode(png_bytes).decode("ascii"),
                "robot_gx": state.get("robot_gx", 0),
                "robot_gy": state.get("robot_gy", 0),
                "robot_yaw": state.get("robot_yaw", 0),
                "trajectory": state.get("trajectory", []),
                "lidar_points": state.get("lidar_points", []),
                "grid_size": state.get("grid_size", 300),
                "occupied_cells": state.get("occupied_cells", 0),
                # Section & lap tracking
                "section_name": state.get("section_name", "?"),
                "section_type": state.get("section_type", "?"),
                "section_idx": state.get("section_idx", -1),
                "lap_count": state.get("lap_count", 0),
                "sections": state.get("sections", []),
                "timestamp": time.time(),
            })

            dead = []
            with self._ws_clients_lock:
                for client in self._ws_clients:
                    try:
                        self._ws_send(client, msg)
                    except Exception:
                        dead.append(client)

                for d in dead:
                    try:
                        d.close()
                    except Exception:
                        pass
                    self._ws_clients.remove(d)

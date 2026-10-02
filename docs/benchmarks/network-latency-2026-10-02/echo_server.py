"""Temporary synthetic transport probe; run only inside a SLURM allocation."""
import os
import socket
import socketserver
import struct
import time

assert os.environ.get("SLURM_JOB_ID"), "Use a compute allocation"

class Echo(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(15)
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        def receive(n):
            result = bytearray()
            while len(result) < n:
                part = self.request.recv(n - len(result))
                if not part:
                    raise EOFError
                result.extend(part)
            return result
        try:
            while True:
                upload, download = struct.unpack("!II", receive(8))
                if max(upload, download) > 4 * 1024 * 1024:
                    return
                data = receive(upload)
                if download > len(data):
                    data.extend(bytes(download - len(data)))
                self.request.sendall(memoryview(data)[:download])
        except (EOFError, OSError):
            pass

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

print(f"READY host={socket.gethostname()} port=18765 job={os.environ['SLURM_JOB_ID']}", flush=True)
with Server(("0.0.0.0", 18765), Echo) as server:
    server.serve_forever()

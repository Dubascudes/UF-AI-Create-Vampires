# Phone to HiPerGator transport latency — 2026-10-02

## Finding

The measured campus Wi-Fi relay route does not consistently fit a 200 ms camera-to-display budget. With 128 KiB uploaded and 128 KiB returned, transport alone took 246.5 ms median and 862.0 ms p95. The USB reference took 29.0 ms median and 68.9 ms p95. These measurements do not rule out a faster route using a wired relay or an authenticated tunnel originating on the phone; neither was tested.

## Routes

- Wi-Fi: Galaxy S25 Ultra on eduroam → computer on eduroam → SSH tunnel through hpg.rc.ufl.edu → CPU compute node c0709a-s30.
- USB: phone → ADB reverse over USB → same computer and SSH tunnel → same compute node.
- Computer reference: computer → same SSH tunnel → same compute node.
- The phone was awake and externally powered during the Wi-Fi run.
- Direct phone-to-compute-node TCP access timed out on eduroam. On ufguest, the relay and direct compute connections were refused. The SSH gateway itself was reachable on ufguest; its single connection sample is not a latency distribution.

## Measurements

Values are milliseconds. Payload size is sent in EACH direction; 128 KiB means 256 KiB total plus protocol headers.

| Payload each way | Wi-Fi median | Wi-Fi p95 | USB median | USB p95 | Computer median | Computer p95 |
|---|---:|---:|---:|---:|---:|---:|
| 64 B | 66.3 | 219.6 | 10.0 | 23.8 | 10.4 | 23.6 |
| 64 KiB | 174.5 | 419.1 | 49.7 | 89.7 | 11.4 | 26.3 |
| 128 KiB | 246.5 | 862.0 | 29.0 | 68.9 | 19.1 | 41.7 |
| 256 KiB | 473.3 | 1099.0 | 51.6 | 86.8 | 38.5 | 56.5 |
| 512 KiB | 1010.7 | 2390.2 | 75.7 | 109.6 | 68.7 | 94.3 |

## Method and limits

Persistent TCP sockets with TCP_NODELAY. Server receives the complete synthetic payload and returns the same number of bytes. Elapsed time is measured on the originating device around socket write/flush/readFully using System.nanoTime. Every response is verified against the sent bytes outside the timed interval. Five warmup requests per size are discarded. Wi-Fi has 60 recorded samples per size; USB and computer have 40. P95 uses nearest rank. Requests are sequential with a 30 ms gap, so this is a request/response test, not a pipelined video throughput test.

No model, camera, JPEG encoder, decoder, or display is involved. The payloads are deterministic pseudorandom bytes, not measured frame sizes from the app. SSH setup/authentication is excluded because the tunnel already exists. Initial TCP connection time is stored separately in summary.json. Wi-Fi and USB runs were sequential, so their differences are not a controlled decomposition of each network hop. The non-monotonic USB results also show time-varying delay. A separate small direct-connect reachability check ran during the Wi-Fi trial.

Large Wi-Fi transfers varied substantially: the 512 KiB-each-way maximum was 2888.4 ms. This short test describes this session and route, not all campus locations or a production service.

## Reproduction and artifacts

- `NetworkProbe.java`: standalone Java/Android shell timing client; compiled with javac and Android SDK d8. No app installation.
- `echo_server.py` and `probe.sbatch`: synthetic echo service in SLURM job 44469721, hpg-dev, 1 CPU, 1 GiB RAM, 20-minute cap; no GPU. All remote job files under /blue/rewetz/will.english/absent-network-probe.
- `phone-wifi.csv`, `phone-usb.csv`, `computer-ssh.csv`: successful raw measurements.
- `metadata.json`: route, device, timing boundary, and run details.
- `summary.json`: statistics; regenerate with `python3 summarize.py`.
- Failed-route diagnostics remain in the original local run directory; they contain no successful latency samples.

The camera app and its model files were unchanged. The source is preserved by tag `v0.1-baseline-20261002`; the APK and model bundle are attached to that GitHub release.


Cleanup: the temporary job was cancelled (last observed state COMPLETING); both test SSH forwards, the ADB reverse, and the phone probe file were removed successfully.

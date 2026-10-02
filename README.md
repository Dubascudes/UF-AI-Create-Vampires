# Absent — hybrid live mirror

## Android app and laptop setup

The current competition app is the native Android project in [`android/`](android/):
ARCore camera → YOLO person masks → LaMa-Dilated on the phone NPU/GPU, or LaMa on a
computer over USB. Each removal result is displayed over its matching camera frame.

**Start with [the Android setup guide](android/README.md)** to clone onto another
computer, restore the exact model files, build, or install the preserved APK.
The working baseline is tagged `v0.1-baseline-20261002`; its APK and model bundle
are distributed as GitHub release assets rather than Git blobs.

[October 2 network measurements](docs/benchmarks/network-latency-2026-10-02/RESULTS.md)
compare the phone's USB and campus Wi-Fi routes to HiPerGator. ProPainter and
LGTSM are candidates for future experiments; neither is integrated or benchmarked
in this app yet.

The sections below document the browser prototypes, which remain available.

Absent is a webcam demonstration for the [AI² Gator Create Competition](https://ai.ufl.edu/ai-university/ai-days/gator-create/). The browser detects people with a selectable mask model—local GPU YOLO26n-seg by default, or the earlier MediaPipe Selfie Segmenter—learns the room as it becomes visible, and sends [JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit) a frame with the person area prefilled from that room model. When JoyAI returns a frame, the browser shows the live camera outside the person mask, learned room pixels inside known parts, and generated pixels inside unseen parts only if a second person check finds no person there. A detected generated face is replaced by approximate room fill.

JoyAI's current streaming API has no spatial mask input, so it still computes a full frame. The browser restricts which parts of that frame appear in the mirror. The learned room requires a fixed camera; use **Reset learned room** after moving it. An empty-room capture is unnecessary.

## Start the browser

Requires Node.js 18 or newer and a webcam. For the default YOLO option, this machine also needs an NVIDIA GPU and an existing CUDA-enabled system Python/PyTorch installation. The setup script makes a project-local environment and downloads the official checkpoint into `models/`; it does not replace system PyTorch.

```bash
npm install
npm run assets
npm run segment:setup
npm run segment:serve
```

In another terminal:

```bash
npm run dev
```

Open the URL printed by Vite (usually `http://127.0.0.1:5173`). The B200 endpoint defaults to `ws://127.0.0.1:8083/ws`; the RTX Pro 6000 profile defaults to port 8084. Edit the endpoint if your forwarding uses another port. The mirror remains blank until model output arrives. The footer shows input, model, and displayed frame rates, capture-to-display delay, and recent browser work times. Camera frames are sent to the configured JoyAI endpoint; reduced raw-pixel frames are also sent over a persistent local WebSocket to the YOLO service when that option is selected. Select MediaPipe in **Person segmentation** to compare without the YOLO service.

For a mask comparison that does not require JoyAI, open `http://127.0.0.1:5173/mask-preview.html`. Green pixels show the selected model's person mask over the webcam image. Switch between YOLO and MediaPipe while the camera is running; the page reports model time, full local mask time, and mask coverage. The mirror links to this page from its **Person segmentation** control.

To make a held object float, click **Choose object** and then its visible surface in the mirror. The selected object's tracked pixels stay visible above the person replacement. Tracking can fail when the object turns or changes appearance. Segmentation can also miss parts of a person; the generated-output check and expanded temporal mask reduce this risk but do not guarantee perfect removal.

## Local 2D mirror

`local.html` removes people using only this computer: webcam → local YOLO person mask → learned room → composite, with LaMa filling parts of the room that have never been seen. Each displayed frame is the exact frame that was segmented, so the mask does not lag the image. It needs a fixed camera.

```bash
npm run dev:all      # YOLO (8001), LaMa (8000), and Vite
```

Open `http://127.0.0.1:5173/local.html`. **View** switches between the mirror, the mirror with masks (red = person, blue = room never seen), the learned room, and the raw camera. The footer shows displayed frames per second and per-stage times (capture / YOLO round trip / mask cleanup / room model / compose). The YOLO service runs with WebSocket compression off; with it on, compressing raw frames added about 50 ms per round trip.

## Phone AR check

The final piece targets moving-camera AR on an ARCore Android phone (Galaxy S25 Ultra) in Chrome. `xr.html` is a feasibility page: it reports which WebXR features the phone grants (camera access, depth sensing, plane detection), frame rate, camera readback time, MediaPipe mask time, and depth coverage, and it can record camera frames with pose and depth.

```bash
npm run dev          # on the laptop
npm run xr:phone     # with the phone on USB: forwards phone localhost:5173 to the laptop
```

Open `http://localhost:5173/xr.html` in Chrome on the phone. Because the page is served from `localhost`, WebXR works without an HTTPS certificate. In the session, **View** cycles through camera, depth, and person-mask overlays; **Mask** loads MediaPipe (GPU, falling back to CPU); **Record** saves about 5 frames per second. Recordings go to `recordings/<id>/` as `NNNNNN.jpg`, `NNNNNN.json` (camera-to-world and projection matrices, column-major; depth dimensions, format, `rawValueToMeters`, and `normDepthBufferFromNormView`), and `NNNNNN.depth.bin` (raw row-major depth). Each session's summary is saved to `recordings/reports/`.

**Start front camera check** uses the selfie camera through `getUserMedia`. ARCore provides tracking and depth only for the rear camera, so these recordings have `source: "front-camera"`, no `cameraToWorld`, `projection`, or depth, and an `orientation` field instead: a `RelativeOrientationSensor` quaternion, or `deviceorientation` angles if the sensor is unavailable. The preview is mirrored; recorded JPEGs are not. Use `chrome://inspect` on the laptop to see the phone's console.

## Run JoyAI on HiPerGator

Setup and service jobs are in [`hpg/`](hpg/). They use the `rewetz` allocation and keep source, caches, weights, and logs under `/blue/rewetz/will.english/absent-joyai`. Setup and model loading run only in compute jobs, not on a login node. The serving jobs request either one B200 or one RTX Pro 6000. [`hpg/run-joyai-no-record.sh`](hpg/run-joyai-no-record.sh) omits upstream automatic MP4 recording. The shared CUDA extension must contain both `sm_100a` and `sm_120a` code; [`hpg/rebuild-joyai-ops.sbatch`](hpg/rebuild-joyai-ops.sbatch) rebuilds it on a compute node.

Find a running service node and forward its port to the local port selected in the browser:

```bash
ssh hpg.rc.ufl.edu 'squeue -u will.english -o "%i %j %T %N"'
ssh -N -L 8083:<node>:8080 hpg.rc.ufl.edu
```

For the RTX profile, forward to local port 8084 instead. JoyAI offers MJPEG and H.264 transports; this integration uses MJPEG. The browser disables JoyAI's face/person presence gate so it requests edited output even when no person is detected.

## Validation

```bash
npm test
npm run build
```

The [September 24 results](RESULTS_2026-09-24.md) include the prerecorded JoyAI probe, local YOLO mask timing, and a live hybrid browser check. The hybrid YOLO mirror displayed roughly 9–12 fps with 0.8–1.1 seconds of delay in those brief checks; it has not yet recovered the earlier pipeline's frame rate. Walk into and out of the shot, raise and lower your hands, move aside to reveal the room, and compare YOLO with MediaPipe for face removal, mask edges, held objects, output frame rate, and delay.

The earlier MediaPipe/LaMa masked prototype and full-frame JoyAI page remain in the source tree for comparison; the hybrid page is active.

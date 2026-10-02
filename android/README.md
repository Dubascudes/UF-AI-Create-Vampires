# Android app: restore the working baseline

The app uses ARCore, YOLO26n segmentation through LiteRT, and LaMa-Dilated with
Places2 weights. The Fill button selects phone NPU, phone GPU, or laptop LaMa.
The default is phone NPU. These instructions preserve the current model choices;
ProPainter and LGTSM have not been integrated.

## 1. Clone and restore model files

```bash
git clone https://github.com/Dubascudes/UF-AI-Create-Vampires.git
cd UF-AI-Create-Vampires
```

Download `android-models.zip`, `app-release.apk`, and `SHA256SUMS` from the
[baseline release](https://github.com/Dubascudes/UF-AI-Create-Vampires/releases/tag/v0.1-baseline-20261002).
With GitHub CLI installed:

```bash
gh release download v0.1-baseline-20261002 --repo Dubascudes/UF-AI-Create-Vampires --pattern android-models.zip --pattern app-release.apk --pattern SHA256SUMS --dir release-assets
```

On Linux/macOS, verify and extract from the repository root:

```bash
(cd release-assets && sha256sum -c SHA256SUMS)
unzip -o release-assets/android-models.zip
```

On Windows PowerShell, compare each SHA-256 against `release-assets/SHA256SUMS`, then extract:

```powershell
Get-FileHash release-assets/android-models.zip -Algorithm SHA256
Get-FileHash release-assets/app-release.apk -Algorithm SHA256
Expand-Archive release-assets/android-models.zip -DestinationPath . -Force
```

The bundle restores three YOLO files under `android/app/src/main/assets/` and
`models/lama_dilated_places.tflite`. The exact file hashes, sizes, and provenance
are recorded in [`model-manifest.json`](model-manifest.json). Model files are
ignored by Git. Do not substitute a face-trained LaMa checkpoint: it can generate
faces inside the removed person region.

## 2. Install the preserved APK

Install Android SDK Platform Tools, enable USB debugging on the phone, connect a
data-capable USB cable, and accept the debugging prompt. From the repository root:

```bash
adb devices
adb install -r release-assets/app-release.apk
```

Open Absent once, grant camera permission, and install/update Google Play Services
for AR if prompted. Then copy the external fill model:

```bash
adb shell mkdir -p /sdcard/Android/data/edu.ufl.absent/files
adb push models/lama_dilated_places.tflite /sdcard/Android/data/edu.ufl.absent/files/lama_dilated_places.tflite
```

Restart the app. The first NPU load may take time to compile its model cache.
The current app targets an ARCore-capable Android 11+ device; the working phone
used for the network measurements is a Galaxy S25 Ultra. NPU compatibility on
other devices has not been established.

The APK is the existing October 2 artifact, preserved before laptop migration.
Its historical source-to-build provenance was not independently verified at
capture time. The source build is checked separately; installing this APK is
the way to reproduce the preserved binary.

## 3. Build on the laptop

Install Android Studio, Android SDK Platform 36, and JDK 17 or newer compatible
with the checked-in Gradle/Android plugin versions. Open the `android` directory
in Android Studio, let it locate your SDK, and restore the models before building.
`local.properties` is machine-specific and is intentionally excluded from Git.

Linux/macOS:

```bash
cd android
./gradlew assembleRelease
```

Windows PowerShell:

```powershell
cd android
.\gradlew.bat assembleRelease
```

Output: `android/app/build/outputs/apk/release/app-release.apk` from the repository
root. This prototype signs release builds with the local Android debug key.
Another computer normally has a different debug key, so its APK cannot update an
existing installation signed here. Keep the preserved app installed: for model
experiments, use a separate application ID and copy the LaMa model into that
app's external files directory. Signing keys are not included in the repository.

## 4. Use laptop LaMa over USB

The existing server/setup scripts use Linux paths. On Windows, run that backend
in WSL2 with working NVIDIA CUDA support and use Windows ADB for the USB phone.
Verify the service is reachable at Windows `http://localhost:8000/docs` before
forwarding it to the phone. Windows/WSL2 operation has not been tested in this
checkout.

With Node.js and Conda available, from the repository root:

```bash
npm ci
npm run inpaint:setup
npm run inpaint:serve
```

In a separate terminal (Windows terminal when using WSL2):

```bash
adb reverse tcp:8000 tcp:8000
```

Select **Fill: laptop LaMa** in the app. It sends masked image crops to
`http://127.0.0.1:8000/api/inpaint-frame` through USB. The server downloads its own
ONNX LaMa model during setup; it does not use the Android TFLite model. Laptop
GPU speed and USB-only latency have not yet been measured. The October 2 USB
numbers include an additional network trip to HiPerGator.

## Validation

```bash
npm test
npm run build
python3 -m unittest discover -s tests
```

Python tests require the segmentation service's imports (including NumPy,
Pillow, and FastAPI). Use a suitable existing environment rather than installing
into system Python. The native app's build check is `assembleRelease` above;
browser/Python tests do not establish Android camera quality or device speed.

// WebXR Raw Camera Access (https://immersive-web.github.io/raw-camera-access/), not yet in @types/webxr.
interface XRCamera {
  readonly width: number;
  readonly height: number;
}

interface XRView {
  readonly camera?: XRCamera | null;
}

interface XRWebGLBinding {
  getCameraImage(camera: XRCamera): WebGLTexture | null;
}

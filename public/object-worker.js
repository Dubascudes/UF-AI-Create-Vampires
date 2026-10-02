importScripts('/vision_bundle.js');

let segmenter = null;

self.onmessage = async event => {
  const request = event.data;
  try {
    if (request.type === 'init') {
      const vision = await Vision.FilesetResolver.forVisionTasks('/wasm');
      segmenter = await Vision.InteractiveSegmenterLegacy.createFromOptions(vision, {
        baseOptions: { modelAssetPath: '/models/magic_touch.tflite', delegate: 'CPU' },
        outputConfidenceMasks: true,
        outputCategoryMask: false,
      });
      self.postMessage({ type: 'ready' });
      return;
    }
    if (!segmenter) throw new Error('Object segmenter is not ready');
    const { id, bitmap, x, y } = request;
    try {
      segmenter.segment(bitmap, { keypoint: { x, y } }, result => {
        const mask = result.confidenceMasks?.[0];
        if (!mask) throw new Error('No object mask returned');
        const values = mask.getAsFloat32Array().slice();
        self.postMessage({ type: 'selected', id, width: mask.width, height: mask.height, values }, [values.buffer]);
      });
    } finally {
      bitmap.close();
    }
  } catch (error) {
    self.postMessage({ type: 'error', id: request.type === 'select' ? request.id : undefined, message: String(error) });
  }
};

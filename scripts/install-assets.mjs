import { copyFileSync, createWriteStream, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import https from 'node:https';

const source = 'node_modules/@mediapipe/tasks-vision/wasm';
const destination = 'public/wasm';
mkdirSync(destination, { recursive: true });
for (const file of readdirSync(source)) {
copyFileSync(join(source, file), join(destination, file));
}
copyFileSync('node_modules/@mediapipe/tasks-vision/vision_bundle.js', 'public/vision_bundle.js');
console.log(`Copied MediaPipe WASM files to ${destination}`);

const modelDirectory = 'public/models';
mkdirSync(modelDirectory, { recursive: true });
const models = [
  ['selfie_segmenter.tflite', 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite'],
  ['magic_touch.tflite', 'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite'],
];
for (const [name, url] of models) {
  await new Promise((resolve, reject) => {
    https.get(url, response => {
      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`${name} download failed: HTTP ${response.statusCode}`));
        return;
      }
      const file = createWriteStream(join(modelDirectory, name));
      response.pipe(file);
      file.on('finish', () => { file.close(); resolve(); });
      file.on('error', reject);
    }).on('error', reject);
  });
  console.log(`Downloaded ${name}`);
}

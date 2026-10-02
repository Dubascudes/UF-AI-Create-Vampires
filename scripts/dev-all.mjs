import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';

if (!existsSync('env/bin/python') || !existsSync('models/lama_fp32.onnx') ||
    !existsSync('yolo-env/bin/python') || !existsSync('models/yolo26n-seg.pt')) {
  console.error('Local models are missing. Run npm run inpaint:setup and npm run segment:setup first.');
  process.exit(1);
}

const backend = spawn('env/bin/python', ['-m', 'uvicorn', 'server.inpaint:app', '--host', '127.0.0.1', '--port', '8000'], { stdio: 'inherit' });
const segmenter = spawn('yolo-env/bin/python', ['-m', 'uvicorn', 'server.person_segment:app', '--host', '127.0.0.1', '--port', '8001', '--no-access-log', '--ws-per-message-deflate', 'false'], { stdio: 'inherit' });
const frontend = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1'], { stdio: 'inherit' });
let stopping = false;

function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  backend.kill('SIGTERM');
  segmenter.kill('SIGTERM');
  frontend.kill('SIGTERM');
  process.exitCode = code;
}
process.on('SIGINT', () => stop(0));
process.on('SIGTERM', () => stop(0));
backend.on('error', error => { console.error(error); stop(1); });
segmenter.on('error', error => { console.error(error); stop(1); });
frontend.on('error', error => { console.error(error); stop(1); });
backend.on('exit', code => { if (!stopping) stop(code || 1); });
segmenter.on('exit', code => { if (!stopping) stop(code || 1); });
frontend.on('exit', code => { if (!stopping) stop(code || 1); });

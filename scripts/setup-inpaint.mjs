import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream, existsSync, mkdirSync, renameSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';

const model = 'models/lama_fp32.onnx';
const expected = '1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6';
const url = 'https://huggingface.co/Carve/LaMa-ONNX/resolve/a3ee2fc/lama_fp32.onnx';

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: 'inherit' });
    child.on('error', reject);
    child.on('exit', code => code === 0 ? resolve() : reject(new Error(`${command} exited with ${code}`)));
  });
}

function environmentReady() {
  if (!existsSync('env/bin/python')) return false;
  return new Promise(resolve => {
    const child = spawn('env/bin/python', ['-c', 'import fastapi, numpy, onnxruntime, PIL, uvicorn'], { stdio: 'ignore' });
    child.on('error', () => resolve(false));
    child.on('exit', code => resolve(code === 0));
  });
}

async function digest(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

if (!await environmentReady()) {
  const environmentCommand = existsSync('env/bin/python') ? 'update' : 'create';
  await run('conda', ['env', environmentCommand, '-p', 'env', '-f', 'environment.yml']);
}

if (!existsSync(model) || await digest(model) !== expected) {
  mkdirSync('models', { recursive: true });
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Model download failed: HTTP ${response.status}`);
  const temporary = `${model}.part`;
  await pipeline(Readable.fromWeb(response.body), createWriteStream(temporary));
  if (await digest(temporary) !== expected) throw new Error('Downloaded model checksum does not match');
  renameSync(temporary, model);
}
console.log('Local CUDA inpainter is ready. Run npm run dev:all.');

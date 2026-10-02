import { defineConfig, type Plugin } from 'vite';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { decodeFramePacket, isSafeRecordingId } from './src/xr-recording';

const MAX_PACKET_BYTES = 16 * 1024 * 1024;

/** Dev-only endpoint: the phone POSTs recorded AR frames to /xr-record/<id>; files land in recordings/<id>/. */
function xrRecorder(): Plugin {
  return {
    name: 'absent-xr-recorder',
    configureServer(server) {
      server.middlewares.use('/xr-report', (request, response) => {
        if (request.method !== 'POST') { response.statusCode = 405; response.end(); return; }
        const chunks: Buffer[] = [];
        request.on('data', (chunk: Buffer) => { if (chunks.length < 64) chunks.push(chunk); });
        request.on('end', async () => {
          try {
            const report = JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
            const directory = resolve(__dirname, 'recordings', 'reports');
            await mkdir(directory, { recursive: true });
            const name = new Date().toISOString().replace(/[:.]/g, '-');
            await writeFile(resolve(directory, `${name}.json`), JSON.stringify(report, null, 2));
            response.statusCode = 204;
            response.end();
          } catch (error) {
            response.statusCode = 400;
            response.end(String(error));
          }
        });
      });
      server.middlewares.use('/xr-record/', (request, response) => {
        const id = decodeURIComponent((request.url ?? '').replace(/^\//, '').split('?')[0]);
        if (request.method !== 'POST' || !isSafeRecordingId(id)) {
          response.statusCode = 400;
          response.end('Expected POST /xr-record/<recording id>');
          return;
        }
        const chunks: Buffer[] = [];
        let size = 0;
        request.on('data', (chunk: Buffer) => {
          size += chunk.length;
          if (size > MAX_PACKET_BYTES) request.destroy();
          else chunks.push(chunk);
        });
        request.on('end', async () => {
          try {
            const { meta, image, depth } = decodeFramePacket(new Uint8Array(Buffer.concat(chunks)));
            const directory = resolve(__dirname, 'recordings', id);
            const name = String(meta.index).padStart(6, '0');
            await mkdir(directory, { recursive: true });
            await Promise.all([
              writeFile(resolve(directory, `${name}.json`), JSON.stringify(meta)),
              writeFile(resolve(directory, `${name}.jpg`), image),
              depth ? writeFile(resolve(directory, `${name}.depth.bin`), depth) : Promise.resolve(),
            ]);
            response.statusCode = 204;
            response.end();
          } catch (error) {
            response.statusCode = 400;
            response.end(String(error));
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [xrRecorder()],
  build: { rollupOptions: { input: {
    mirror: resolve(__dirname, 'index.html'),
    masks: resolve(__dirname, 'mask-preview.html'),
    xr: resolve(__dirname, 'xr.html'),
    local: resolve(__dirname, 'local.html'),
  } } },
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000', '/seg': 'http://127.0.0.1:8001' },
  },
});

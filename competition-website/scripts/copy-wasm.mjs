import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const dest = join(root, 'public', 'wasm');

if (!existsSync(src)) {
  console.warn('onnxruntime-web not installed — skip copy-wasm');
  process.exit(0);
}

mkdirSync(dest, { recursive: true });

const keep = (name) =>
  name.startsWith('ort') &&
  (name.endsWith('.wasm') ||
    name.endsWith('.mjs') ||
    name.endsWith('.js') ||
    name.endsWith('.map'));

let n = 0;
for (const name of readdirSync(src)) {
  if (!keep(name)) continue;
  cpSync(join(src, name), join(dest, name));
  n++;
}
console.log(`Copied ${n} onnxruntime-web assets → public/wasm/`);

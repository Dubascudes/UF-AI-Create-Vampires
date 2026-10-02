export interface ConservativeMask { mask: Uint8Array; temporal: Float32Array }

export function conservativePersonMask(
  confidence: Float32Array, width: number, height: number, previous?: Float32Array,
): ConservativeMask {
  if (confidence.length !== width * height) throw new Error('Person confidence dimensions do not match');
  const temporal = new Float32Array(confidence.length);
  const mask = new Uint8Array(confidence.length);
  const horizontal = new Uint8Array(confidence.length);
  for (let index = 0; index < confidence.length; index++) {
    temporal[index] = Math.max(confidence[index], (previous?.[index] ?? 0) * 0.7);
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let dx = -4; dx <= 4; dx++) {
        const nx = x + dx;
        if (nx >= 0 && nx < width && temporal[y * width + nx] >= 0.2) {
          horizontal[y * width + x] = 255;
          break;
        }
      }
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let dy = -4; dy <= 4; dy++) {
        const ny = y + dy;
        if (ny >= 0 && ny < height && horizontal[ny * width + x]) {
          mask[y * width + x] = 255;
          break;
        }
      }
    }
  }
  return { mask, temporal };
}

export function erasePersonForModel(
  raw: Uint8ClampedArray, room: Uint8ClampedArray, person: Uint8Array, object?: Uint8Array,
): Uint8ClampedArray {
  validate(raw, room, person, object);
  const output = raw.slice();
  for (let index = 0; index < person.length; index++) {
    if (object?.[index]) continue;
    const opacity = person[index] / 255;
    const offset = index * 4;
    for (let channel = 0; channel < 3; channel++) {
      output[offset + channel] = raw[offset + channel] * (1 - opacity) + room[offset + channel] * opacity;
    }
  }
  return output;
}

export function composeHybrid(
  raw: Uint8ClampedArray, room: Uint8ClampedArray, generated: Uint8ClampedArray,
  masks: { person: Uint8Array; known: Uint8Array; generatedPerson: Uint8Array | null; object?: Uint8Array },
): Uint8ClampedArray {
  validate(raw, room, masks.person, masks.object);
  if (generated.length !== raw.length || masks.known.length !== masks.person.length ||
      (masks.generatedPerson && masks.generatedPerson.length !== masks.person.length)) {
    throw new Error('Generated frame or masks have different dimensions');
  }
  const output = raw.slice();
  for (let index = 0; index < masks.person.length; index++) {
    if (masks.object?.[index]) continue;
    const opacity = masks.person[index] / 255;
    if (!opacity) continue;
    const replacement = masks.known[index] || !masks.generatedPerson || masks.generatedPerson[index] >= 128
      ? room : generated;
    const offset = index * 4;
    for (let channel = 0; channel < 3; channel++) {
      output[offset + channel] = raw[offset + channel] * (1 - opacity) + replacement[offset + channel] * opacity;
    }
  }
  return output;
}

function validate(raw: Uint8ClampedArray, room: Uint8ClampedArray, person: Uint8Array, object?: Uint8Array): void {
  if (raw.length !== room.length || raw.length !== person.length * 4 ||
      (object && object.length !== person.length)) {
    throw new Error('Input frame and masks have different dimensions');
  }
}

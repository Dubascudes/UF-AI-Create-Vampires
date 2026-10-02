// Fullscreen-triangle shaders for the rear-camera WebXR check.
const FULLSCREEN_VERTEX = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const COPY_FRAGMENT = `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uTexture;
out vec4 color;
void main() { color = texture(uTexture, vUv); }`;

// Output is premultiplied alpha. Depth and mask rows are uploaded top-down, so they are sampled in
// normalized view coordinates (origin top-left, y down).
export const OVERLAY_FRAGMENT = `#version 300 es
precision highp float;
in vec2 vUv;
uniform int uMode;
uniform sampler2D uTexture;
uniform mat4 uDepthUv;
out vec4 color;
void main() {
  vec2 view = vec2(vUv.x, 1.0 - vUv.y);
  if (uMode == 1) {
    vec4 depth = texture(uTexture, (uDepthUv * vec4(view, 0.0, 1.0)).xy);
    float alpha = depth.a * 0.65;
    color = vec4(depth.rgb * alpha, alpha);
  } else {
    float alpha = texture(uTexture, view).r * 0.6;
    color = vec4(vec3(1.0, 0.15, 0.3) * alpha, alpha);
  }
}`;

export function compile(gl: WebGL2RenderingContext, fragment: string): WebGLProgram {
  const program = gl.createProgram()!;
  for (const [type, source] of [[gl.VERTEX_SHADER, FULLSCREEN_VERTEX], [gl.FRAGMENT_SHADER, fragment]] as const) {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? 'Shader failed');
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? 'Program failed');
  return program;
}

export function dataTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const texture = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return texture;
}


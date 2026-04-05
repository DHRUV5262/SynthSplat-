#version 460 core
// Fullscreen triangle (no VBO) — vertical gradient like a simple Unity skybox
out float vSkyT;

void main()
{
    vec2 uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vec2 ndc = uv * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.999, 1.0);
    vSkyT = uv.y;
}

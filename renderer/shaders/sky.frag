#version 460 core
out vec4 FragColor;

in float vSkyT;

uniform uint uFrame;

void main()
{
    float t = pow(clamp(vSkyT, 0.0, 1.0), 0.78);

    vec3 horizon = vec3(0.58, 0.72, 0.90);
    vec3 zenith  = vec3(0.15, 0.35, 0.72);
    vec3 lin = mix(horizon, zenith, t);

    vec2 g = gl_FragCoord.xy + vec2(float(uFrame) * 13.7, float(uFrame) * 9.1);
    float n = fract(sin(dot(g, vec2(12.9898, 78.233))) * 43758.5453);
    lin += (n - 0.5) * 0.018;

    vec3 color = pow(lin, vec3(1.0 / 2.2));
    FragColor = vec4(color, 1.0);
}

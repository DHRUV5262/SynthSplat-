#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
// Tangent is needed for normal mapping
layout (location = 3) in vec4 aTangent; 

out vec2 TexCoords;
out vec3 WorldPos;
out vec3 Normal;
out mat3 TBN;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    TexCoords = aTexCoords;
    WorldPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;

    // Calculate TBN matrix for normal mapping
    vec3 T = normalize(vec3(model * vec4(aTangent.xyz, 0.0)));
    vec3 N = normalize(vec3(model * vec4(aNormal, 0.0)));
    // Re-orthogonalize T with respect to N
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T) * aTangent.w; // Use w component for handedness
    TBN = mat3(T, B, N);

    gl_Position = projection * view * vec4(WorldPos, 1.0);
}

#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 normalMatrix;
    vec4 parameters;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inTangent;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec3 TangentViewPos;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 WorldPos;
layout(location = 3) out vec3 TangentFragPos;
layout(location = 4) out vec4 parameters;


void main() {
    const vec3 campos = vec3(0,0,3);
    
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    WorldPos = vec3(ubo.model * vec4(inPosition.xyz, 1.0));
    
    mat3 normalMatrix = mat3(1.0);

    vec3 T = normalize(normalMatrix * inTangent);
    vec3 N = normalize(normalMatrix * inNormal);
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);

    mat3 TBN = transpose(mat3(T, B, N));
    TangentViewPos  = TBN * campos;
    TangentFragPos  = TBN * WorldPos.xyz;
    
    parameters = ubo.parameters;
}

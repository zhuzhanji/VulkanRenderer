#version 450
layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec2 uv;

layout(location = 2) in vec4 offset;
layout(location = 3) in vec4 inColor;


layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 texcoord;
void main() {

    gl_Position = ubo.proj * ubo.view * ubo.model * vec4((inPosition).xyz + offset.xyz, 1.0);
    fragColor = vec4(vec3(uv, 1) * inColor.xyz, 1);
    texcoord = uv;
}

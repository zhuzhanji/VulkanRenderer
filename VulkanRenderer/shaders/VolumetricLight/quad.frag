#version 450

layout (binding = 0) uniform sampler2D sceneSampler;
layout (binding = 1) uniform sampler2D lightSampler;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;


void main() 
{
	vec3 color = texture(sceneSampler, inUV).rgb;
    vec3 light = texture(lightSampler, inUV).rgb;
	outFragColor = vec4(color + light, 1.0);
}

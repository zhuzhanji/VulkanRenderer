#version 450

layout (location = 0) in vec3 inPos;

layout (binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 view;
	mat4 model;
    vec4 lightPos;
    vec3 lightAxis;
} ubo;

layout (location = 0) out vec4 wpos;
layout (location = 1) out vec2 uv;
layout (location = 2) out float atten;
layout (location = 3) out vec4 lightPos;
layout (location = 4) out vec3 lightAxis;

vec2 ComputeScreenPos(vec4 pos)
{
    return pos.xy / pos.w * 0.5 + 0.5;
}



void main() 
{
    wpos = ubo.model * vec4(inPos.xyz, 1.0);
    
	gl_Position = ubo.projection * ubo.view * wpos;
    atten = inPos.z;
    uv = ComputeScreenPos(gl_Position);
    lightAxis = ubo.lightAxis;
    lightPos = ubo.lightPos;

}


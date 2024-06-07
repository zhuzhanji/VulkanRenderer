#version 450

layout (binding = 0) uniform sampler2D samplerColorMap;
layout (binding = 1) uniform sampler2D samplerGradientRamp;

layout (location = 0) in float inGradientPos;

layout (location = 0) out vec4 outFragColor;

void main () 
{
	outFragColor.rgba = texture(samplerColorMap, gl_PointCoord).rgba;
}

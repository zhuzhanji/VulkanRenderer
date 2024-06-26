#version 450

layout (binding = 1) uniform sampler2D shadowMap;
layout (binding = 2) uniform sampler2D baseColor;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inViewVec;
layout (location = 2) in vec3 inLightVec;
layout (location = 3) in vec4 inShadowCoord;
layout (location = 4) in vec2 uv;
layout (location = 5) in vec3 spotLightDir;
layout (location = 6) in vec2 cutoff;

layout (location = 0) out vec4 outFragColor;

#define ambient 0.1
#define OFFSET 0.001

float textureProj(vec4 shadowCoord, vec2 off)
{
    vec3 N = normalize(inNormal);
    vec3 L = normalize(inLightVec);
    float bias = max(0.05 * (1.0 - dot(N, L)), 0.025);
    
	float shadow = 0;
	if ( shadowCoord.z > 0.0 && shadowCoord.z < 1.0 && shadowCoord.w > 0)
	{
		float dist = texture( shadowMap, shadowCoord.xy + off ).r;
		if ( shadowCoord.z < dist + bias)
		{
			shadow = 1.0;
		}
	}
	return shadow;
}

float filterPCF(vec4 sc)
{
	ivec2 texDim = textureSize(shadowMap, 0);
	float scale = 1.5;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;
	
	for (int x = -range; x <= range; x++)
	{
		for (int y = -range; y <= range; y++)
		{
			shadowFactor += textureProj(sc, vec2(dx*x, dy*y));
			count++;
		}
	
	}
	return shadowFactor / count;
}

void main() 
{
    vec3 inColor = texture( baseColor, uv ).rgb;
    vec3 N = normalize(inNormal);
    vec3 L = normalize(inLightVec);
    vec3 V = normalize(inViewVec);
    vec3 R = normalize(-reflect(L, N));
    
    float theta     = dot(spotLightDir, -L);
    float epsilon   = cutoff.x - cutoff.y;
    float intensity = clamp((theta - cutoff.y) / epsilon, 0.0, 1.0);
    
    if(theta < cutoff.y){
        outFragColor = vec4(ambient * inColor, gl_FragCoord.z);
        return;
    }
	//float shadow = filterPCF(inShadowCoord / inShadowCoord.w);
    
	vec3 diffuse = max(dot(N, L), 0.0) * inColor;
 
    //vec4 shadowCoord = inShadowCoord;
    //shadowCoord.xyz /= shadowCoord.w;
    //shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;
    
    //float shadowAtten = textureProj( inShadowCoord, vec2(0.0, 0.0) ).r;
    float shadowAtten = filterPCF( inShadowCoord ).r;
	outFragColor = vec4(ambient * inColor + intensity * shadowAtten * diffuse, gl_FragCoord.z);

}

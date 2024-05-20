#version 450

layout(binding = 1) uniform sampler2D albedoSampler;
layout(binding = 2) uniform sampler2D normalSampler;

layout(location = 0) in vec3 TangentViewPos;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 WorldPos;
layout(location = 3) in vec3 TangentFragPos;
layout(location = 4) in vec4 parameters;

layout(location = 0) out vec4 outColor;

//const vec3 camPos = vec3(0,0,3);


vec3 getNormalFromMap()
{
    vec3 tangentNormal = normalize(texture(normalSampler, fragTexCoord).xyz * 2.0 - 1.0);
    return vec3(0,0,1);
    return normalize(tangentNormal);
}


void main() {
    const float PI = 3.14159265359;
    
    const float RimExp = parameters.x;
    const float RimScale = parameters.y;
    const float InnerExp = parameters.z;
    const float InnerScale = parameters.w;
    
    vec3 V = normalize(TangentViewPos - TangentFragPos);
    vec3 N = getNormalFromMap();
    float VdN = clamp(dot(V, N), 0, 1);
    float Rim = RimScale * pow(1 - VdN, RimExp);
    float Inner = InnerScale * pow(VdN, InnerExp);
    float multiplier = Inner + Rim;
    
    outColor = texture(albedoSampler, fragTexCoord * 4);
    outColor.rgb *= (multiplier);
}

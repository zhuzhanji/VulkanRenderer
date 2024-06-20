#version 450

layout (binding = 1) uniform sampler2D shadowMap;
layout (binding = 2) uniform sampler2D cameraDepth;

layout (binding = 3) uniform UBO
{
    mat4 lightSpace;
    //mat4 lightMVP;
    // x = 1-far/near
       // y = far/near
       // z = x/far
       // w = y/far
    vec4 ZBufferParams;
    vec4 mieG;
    vec4 lightColor;
    vec4 worldSpaceCameraPos;
    vec4 cameraForward;
    // x: scattering coef, y: extinction coef
    vec4 volumetricLight;
    //cos(35), cos(45)
    //vec2 cutoff;
    //x: planeD;
    vec4 params;

    // x: scale, y: intensity, z: intensity offset, w: _Time
    vec4 _NoiseData;
    // x: x velocity, y: z velocity
    vec4 _NoiseVelocity;
    
} ubo;

layout (binding = 4) uniform sampler3D perlinNoise;

layout (location = 0) in vec4 wpos;
layout (location = 1) in vec2 uv;
layout (location = 2) in float attenz;
layout (location = 3) in vec4 lightPos;
layout (location = 4) in vec3 lightAxis;

layout (location = 0) out vec4 outFragColor;

float LinearEyeDepth( float z )
{
    return 1.0 / (ubo.ZBufferParams.z * z + ubo.ZBufferParams.w);
}

//-----------------------------------------------------------------------------------------
// MieScattering
//-----------------------------------------------------------------------------------------
float MieScattering(float cosAngle, vec4 g)
{
    return g.w * (g.x / (pow(g.y - g.z * cosAngle, 1.5)));
}
//-----------------------------------------------------------------------------------------
// GetLightAttenuation
//
//-----------------------------------------------------------------------------------------
float GetLightAttenuation(vec3 position){
    
    float shadow = 0;
    
    vec4 shadowCoord = ubo.lightSpace * vec4(wpos.xyz/wpos.w, 1.0);
    shadowCoord.xyz /= shadowCoord.w;
    shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;
    
    if ( shadowCoord.z > 0.0 && shadowCoord.z < 1.0 && shadowCoord.w > 0)
    {
        float dist = texture( shadowMap, shadowCoord.xy ).r;
        if ( shadowCoord.z < dist + 0.002)
        {
            shadow = 1;
        }
    }
    
    vec3 tolight = lightPos.xyz - position;
    vec3 lightDir = normalize(tolight);
    
    float theta     = dot(lightAxis, -lightDir);
    float epsilon   = ubo.volumetricLight.z - ubo.volumetricLight.w;
    float intensity = clamp((theta - ubo.volumetricLight.w) / epsilon, 0.0, 1.0);
    
    float att = max(1.0 - dot(tolight, tolight) * lightPos.w, 0.0);
    return att * att * att * intensity * shadow;
}
//-----------------------------------------------------------------------------------------
// GetDensity
//-----------------------------------------------------------------------------------------
float GetDensity(vec3 wpos)
{
    float density = 1;
    float noise = texture(perlinNoise, fract(wpos * ubo._NoiseData.x + vec3(ubo._NoiseData.w * ubo._NoiseVelocity.x, 0, ubo._NoiseData.w * ubo._NoiseVelocity.y))).r;
    noise = clamp(noise - ubo._NoiseData.z, 0.0, 1.0) * ubo._NoiseData.y;
    density = clamp(noise, 0.0, 1.0);

    return density;
}

//-----------------------------------------------------------------------------------------
// RayMarch
//-----------------------------------------------------------------------------------------
vec4 RayMarch(vec3 rayStart, vec3 rayDir, float rayLength)
{
    /*
#ifdef DITHER_4_4
    vec2 interleavedPos = (fmod(floor(screenPos.xy), 4.0));
    float offset = tex2D(_DitherTexture, interleavedPos / 4.0 + vec2(0.5 / 4.0, 0.5 / 4.0)).w;
#else
    vec2 interleavedPos = (fmod(floor(screenPos.xy), 8.0));
    float offset = tex2D(_DitherTexture, interleavedPos / 8.0 + vec2(0.5 / 8.0, 0.5 / 8.0)).w;
#endif
     */
    const int sampleCount = 16;
    
    int stepCount = sampleCount;

    float stepSize = rayLength / stepCount;
    vec3 seg = rayDir * stepSize;

    //vec3 currentPosition = rayStart + seg * offset;
    vec3 currentPosition = rayStart;

    vec4 vlight = vec4(0.0f);

    float cosAngle;

    // we don't know about density between camera and light's volume, assume 0.5
    float extinction = length(ubo.worldSpaceCameraPos.xyz - currentPosition) * ubo.volumetricLight.y * 0.5;

    for (int i = 0; i < stepCount; ++i)
    {
        float atten = GetLightAttenuation(currentPosition);
        float density = GetDensity(currentPosition);

        float scattering = ubo.volumetricLight.x * stepSize * density;
        extinction += ubo.volumetricLight.y * stepSize * density;// +scattering;
        
        vec4 light = vec4(atten * scattering * exp(-extinction));

        // phase functino for spot and point lights
        vec3 tolight = normalize(currentPosition - lightPos.xyz);
        cosAngle = dot(tolight, -rayDir);
        light *= MieScattering(cosAngle, ubo.mieG);

        vlight += light;

        currentPosition += seg;
    }

    // apply light's color
    vlight *= ubo.lightColor;

    vlight = max(vec4(0.0), vlight);

    vlight.w = 1.0;

    return vlight;
}

//-----------------------------------------------------------------------------------------
// RayConeIntersect
//-----------------------------------------------------------------------------------------
vec2 RayConeIntersect(in vec3 f3ConeApex, in vec3 f3ConeAxis, in float fCosAngle, in vec3 f3RayStart, in vec3 f3RayDir)
{
    float inf = 10000;
    f3RayStart -= f3ConeApex;
    float a = dot(f3RayDir, f3ConeAxis);
    float b = dot(f3RayDir, f3RayDir);
    float c = dot(f3RayStart, f3ConeAxis);
    float d = dot(f3RayStart, f3RayDir);
    float e = dot(f3RayStart, f3RayStart);
    fCosAngle *= fCosAngle;
    float A = a*a - b*fCosAngle;
    float B = 2 * (c*a - d*fCosAngle);
    float C = c*c - e*fCosAngle;
    float D = B*B - 4 * A*C;

    if (D > 0)
    {
        D = sqrt(D);
        vec2 t = (-B + sign(A)*vec2(-D, +D)) / (2 * A);
        int b2IsCorrect_x = int(c + a * t.x > 0 && t.x > 0);
        int b2IsCorrect_y = int(c + a * t.y > 0 && t.y > 0);
        t.x = t.x * b2IsCorrect_x + (1 - b2IsCorrect_x) * (inf);
        t.y = t.y * b2IsCorrect_y + (1 - b2IsCorrect_y) * (inf);
        return t;
    }
    else // no intersection
        return vec2(inf, inf);
}

//-----------------------------------------------------------------------------------------
// RayPlaneIntersect
//-----------------------------------------------------------------------------------------
float RayPlaneIntersect(vec3 planeNormal, float planeD, vec3 rayOrigin, vec3 rayDir)
{
    float NdotD = dot(planeNormal, rayDir);
    float NdotO = dot(planeNormal, rayOrigin);

    float t = -(NdotO + planeD) / NdotD;
    if (t < 0)
        t = 100000;
    return t;
}

void main() 
{
    
    // read depth and reconstruct world position
    float depth = texture(cameraDepth, uv).r;

    vec3 rayStart = ubo.worldSpaceCameraPos.xyz;
    vec3 rayEnd = wpos.xyz;

    vec3 rayDir = (rayEnd - rayStart);
    float rayLength = length(rayDir);

    rayDir /= rayLength;

    // inside cone
    vec3 r1 = rayEnd + rayDir * 0.0001;

    // plane intersection
    float planeCoord = RayPlaneIntersect(lightAxis, ubo.params.x, r1, rayDir);
    // ray cone intersection
    vec2 lineCoords = RayConeIntersect(lightPos.xyz, lightAxis, ubo.volumetricLight.w, r1, rayDir);

    float linearDepth = LinearEyeDepth(depth);
    float projectedDepth = linearDepth / dot(ubo.cameraForward.xyz, rayDir);

    float z = (projectedDepth - rayLength);
    rayLength = min(planeCoord, min(lineCoords.x, lineCoords.y));
    rayLength = min(rayLength, z);
    
    outFragColor = mix(vec4(vec3(0.0), 1.0), RayMarch(rayEnd, rayDir, rayLength) , float(rayLength > 0.0));

}

#version 450

layout (binding = 0) uniform sampler2D mainTexSampler;
layout (binding = 1) uniform sampler2D cameraDepthSampler;

layout (binding = 2) uniform UBO{
    vec2 direction;
    vec2 texSize;
    vec4 ZBufferParams;
} ubo;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

#define HALF_RES_BLUR_KERNEL_SIZE 5
#define GAUSS_BLUR_DEVIATION 1.5
#define PI 3.1415927f
#define BLUR_DEPTH_FACTOR 0.5

float LinearEyeDepth( float z )
{
    return 1.0 / (ubo.ZBufferParams.z * z + ubo.ZBufferParams.w);
}

//-----------------------------------------------------------------------------------------
// GaussianWeight
//-----------------------------------------------------------------------------------------
float GaussianWeight(float offset, float deviation)
{
    float weight = 1.0f / sqrt(2.0f * PI * deviation * deviation);
    weight *= exp(-(offset * offset) / (2.0f * deviation * deviation));
    return weight;
}

vec4 BilateralBlur()
{
    const vec2 direction = ubo.direction;
    const int kernelRadius = HALF_RES_BLUR_KERNEL_SIZE;
    
    //const float deviation = kernelRadius / 2.5;
    const float deviation = kernelRadius / GAUSS_BLUR_DEVIATION; // make it really strong

    vec2 uv = inUV;
    vec4 centerColor = texture(mainTexSampler, uv);
    vec3 color = centerColor.xyz;
    //return float4(color, 1);
    float centerDepth = (LinearEyeDepth(texture(cameraDepthSampler, uv).r));

    float weightSum = 0;

    // gaussian weight is computed from constants only -> will be computed in compile time
    float weight = GaussianWeight(0, deviation);
    color *= weight;
    weightSum += weight;
                
    for (int i = -kernelRadius; i < 0; i += 1)
    {
        vec2 offset = (direction * i) / ubo.texSize;
        vec3 sampleColor = texture(mainTexSampler, uv + offset).rgb;
        float sampleDepth = LinearEyeDepth(texture(cameraDepthSampler, uv + offset).r);

        float depthDiff = abs(centerDepth - sampleDepth);
        float dFactor = depthDiff * BLUR_DEPTH_FACTOR;
        float w = exp(-(dFactor * dFactor));

        // gaussian weight is computed from constants only -> will be computed in compile time
        weight = GaussianWeight(i, deviation) * w;

        color += weight * sampleColor;
        weightSum += weight;
    }

    for (int i = 1; i <= kernelRadius; i += 1)
    {
        vec2 offset = (direction * i) / ubo.texSize;
        vec3 sampleColor = texture(mainTexSampler, uv + offset).rgb;
        float sampleDepth = LinearEyeDepth(texture(cameraDepthSampler, uv + offset).r);

        float depthDiff = abs(centerDepth - sampleDepth);
        float dFactor = depthDiff * BLUR_DEPTH_FACTOR;
        float w = exp(-(dFactor * dFactor));
        
        // gaussian weight is computed from constants only -> will be computed in compile time
        weight = GaussianWeight(i, deviation) * w;

        color += weight * sampleColor;
        weightSum += weight;
    }

    color /= weightSum;
    return vec4(color, centerColor.w);
}

void main() 
{
	outFragColor = vec4(BilateralBlur());
}

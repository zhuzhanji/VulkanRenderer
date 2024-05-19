#version 450

/*
Generalized Kuwahara filter, based on work of Acerola
https://github.com/GarrettGunnell/Post-Processing/tree/main/Assets/Kuwahara%20Filter
*/

layout(binding = 0) uniform UniformBufferObject {
    vec2 resolution;
    int radius;
} ubo;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

const int N = 4;
const float alpha = 0.5;
const float q = 18.;
const float PI = 3.14159265358979323846;
const float _Hardness = 100.;

void main() {
    vec2 uv = fragTexCoord;
    int kernelRadius = ubo.radius;
    float zeta = 2.0f / float(kernelRadius);
    float zeroCross = 2.;
    float sinZeroCross = sin(zeroCross);
    float eta = 0.;
    int k;
    vec4 m[8];
    vec3 s[8];

    for (k = 0; k < N; ++k) {
        m[k] = vec4(0.0f);
        s[k] = vec3(0.0f);
    }
    
    for (int y = -kernelRadius; y <= kernelRadius; ++y) {
        for (int x = -kernelRadius; x <= kernelRadius; ++x) {
            vec2 v = vec2(x, y) / float(kernelRadius);
            vec3 c = texture ( texSampler , uv + vec2(x, y) * 1.0/ubo.resolution.xy). xyz;
            c = clamp(c,0.0,1.0) ;
            float sum = 0.;
            float w[8];
            float z, vxx, vyy;

            /* Calculate Polynomial Weights */
            vxx = zeta - eta * v.x * v.x;
            vyy = zeta - eta * v.y * v.y;
            z = max(0., v.y + vxx);
            w[0] = z * z;
            sum += w[0];
            z = max(0., -v.x + vyy);


            w[2] = z * z;
            sum += w[2];
            z = max(0., -v.y + vxx);
            w[4] = z * z;
            sum += w[4];
            z = max(0., v.x + vyy);
            w[6] = z * z;
            sum += w[6];
            v = sqrt(2.0f) / 2.0f * vec2(v.x - v.y, v.x + v.y);
            vxx = zeta - eta * v.x * v.x;
            vyy = zeta - eta * v.y * v.y;
            z = max(0., v.y + vxx);
            w[1] = z * z;
            sum += w[1];
            z = max(0., -v.x + vyy);
            w[3] = z * z;
            sum += w[3];
            z = max(0., -v.y + vxx);
            w[5] = z * z;
            sum += w[5];
            z = max(0., v.x + vyy);
            w[7] = z * z;
            sum += w[7];


            float g = exp(-3.125f * dot(v,v)) / sum;

            for (int k = 0; k < 8; ++k) {
                float wk = w[k] * g;
                m[k] += vec4(c * wk, wk);
                s[k] += c * c * wk;
            }
        }

    }

    vec4 ou = vec4(0.);
    for (k = 0; k < N; ++k) {
        m[k].rgb /= m[k].w;
        s[k] = abs(s[k] / m[k].w - m[k].rgb * m[k].rgb);

        float sigma2 = s[k].r + s[k].g + s[k].b;
        float w = 1.0f / (1.0f + pow(_Hardness * 1000.0f * sigma2, 0.5f * q));

        ou += vec4(m[k].rgb * w, w);
    }

    outColor = clamp((ou / ou.w),0.0,1.0);
}





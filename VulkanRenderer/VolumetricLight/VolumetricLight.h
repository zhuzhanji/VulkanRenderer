//
//  VolumetricLight.h
//  VulkanRenderer
//
//  Created by Yang on 2024/6/21.
//

#ifndef VolumetricLight_h
#define VolumetricLight_h

#include "../base/ApplicationBase.hpp"

namespace vks {
class VolumetricLight : public ApplicationBase {
public:
    void run();
    
    VolumetricLight(){
        
    }
    ~VolumetricLight(){
        cleanup();
    }
private:
    bool displayShadowMap = false;
    bool filterPCF = true;
    
    // Keep depth range as small as possible
    // for better shadow map precision
    float zNear = 1.0f;
    float zFar = 96.0f;
    
    // Depth bias (and slope) are used to avoid shadowing artifacts
    // Constant depth bias factor (always applied)
    float depthBiasConstant = 3.0f;
    // Slope depth bias factor, applied depending on polygon's slope
    float depthBiasSlope = 2.5f;
    
    glm::vec3 lightPos = glm::vec3();
    float lightFOV = 45.0f;
    float lightOuterFOV = 48.0;
    
    struct UniformDataScene {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
        glm::mat4 depthBiasMVP;
        glm::vec4 lightPos;
        glm::vec4 lightParam;
        glm::vec2 cutoff;
    } uniformDataScene;
    
    struct UniformDataOffscreen {
        glm::mat4 depthMVP;
    } uniformDataOffscreen;
    
    struct UniformDataPrelight{
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
        glm::vec4 lightPos;
        glm::vec3 lightdir;
    } uniformDataPrelight;
    
    struct UniformDataPrelightFrag
    {
        glm::mat4 lightSpace;
        // x = 1-far/near
        // y = far/near
        // z = x/far
        // w = y/far
        glm::vec4 ZBufferParams;
        //w: 1/(range * range)
        glm::vec4 mieG;
        glm::vec4 lightColor;
        glm::vec4 worldSpaceCameraPos;
        glm::vec4 cameraForward;
        // x: scattering coef, y: extinction coef
        glm::vec4 volumetricLight;
        //cos(35), cos(45)
        //glm::vec4 cutoff;
        //float planeD;
        glm::vec4 params;
        // x: scale, y: intensity, z: intensity offset, w: _Time
        glm::vec4 _NoiseData;
        // x: x velocity, y: z velocity
        glm::vec4 _NoiseVelocity;
    } uniformDataPrelightFrag;
    
    struct UniformDataGaussian{
        glm::vec2 direction;
        glm::vec2 texSize;
        glm::vec4 ZBufferParams;
    } uniformDataGaussian;
    
    struct {
        vks::Buffer scene;
        vks::Buffer offscreen;
        vks::Buffer prelight;
        vks::Buffer prelightFrag;
        vks::Buffer hblur;
        vks::Buffer vblur;
    } uniformBuffers;
    
    struct {
        VkPipeline offscreen{ VK_NULL_HANDLE };
        // Pipeline with percentage close filtering (PCF) of the shadow map
        VkPipeline sceneShadowPCF{ VK_NULL_HANDLE };
        VkPipeline blit{ VK_NULL_HANDLE };
        VkPipeline prelight{ VK_NULL_HANDLE };
        VkPipeline gaussian{ VK_NULL_HANDLE };
    } pipelines;
    
    struct {
        VkPipelineLayout offscreen{ VK_NULL_HANDLE };            // Layout of the graphics pipeline
        VkPipelineLayout scene{ VK_NULL_HANDLE };
        VkPipelineLayout blit{ VK_NULL_HANDLE };
        VkPipelineLayout prelight{ VK_NULL_HANDLE };
        VkPipelineLayout gaussian{ VK_NULL_HANDLE };
    }pipelineLayouts;
    
    
    struct {
        VkDescriptorSet offscreen{ VK_NULL_HANDLE };
        VkDescriptorSet scene{ VK_NULL_HANDLE };
        VkDescriptorSet blit{ VK_NULL_HANDLE };
        VkDescriptorSet prelight{ VK_NULL_HANDLE };
        VkDescriptorSet hblur{ VK_NULL_HANDLE };
        VkDescriptorSet vblur{ VK_NULL_HANDLE };
    } descriptorSets;
    
    struct{
        VkDescriptorSetLayout offscreen{ VK_NULL_HANDLE };
        VkDescriptorSetLayout scene{ VK_NULL_HANDLE };
        VkDescriptorSetLayout blit{ VK_NULL_HANDLE };
        VkDescriptorSetLayout prelight{ VK_NULL_HANDLE };
        VkDescriptorSetLayout gaussian{ VK_NULL_HANDLE };
    }descriptorSetLayouts;
    
    struct OffscreenPass {
        int32_t width, height;
        VkFramebuffer frameBuffer;
        vks::Texture depth;
        VkRenderPass renderPass;
    } offscreenPass{};
    
    
    
    struct RenderPass{
        VkRenderPass renderpass{ VK_NULL_HANDLE };
        VkFramebuffer fbo{ VK_NULL_HANDLE };
        Texture color;
        Texture depth;
    } scenePass, preLightPass, blurPass;
    
    Texture resolveColor;
    //Texture resolveDepth;
    
    VkFramebuffer vblurPassFbo{ VK_NULL_HANDLE };
    
    struct Model{
        Buffer indicesBuffer;
        Buffer verticesBuffer;
        unsigned int size;
    } scene, coneModel;
    
    vks::Texture baseColor;
    vks::Texture perlinNoiseTex;
    
    // 16 bits of depth is enough for such a small scene
    const VkFormat offscreenDepthFormat{ VK_FORMAT_D16_UNORM };
    const uint32_t shadowMapize{ 1024 };
    
    //PFN_vkCreateRenderPass2KHR vkCreateRenderPass2KHR {VK_NULL_HANDLE};
public:
    void initVulkan() override;
    
    void mainLoop();
    
    void cleanup();
    
    void loadAssets();
    
    void CreateSpotLightMesh();
    
    // Prepare all Vulkan resources for the 3D texture (including descriptors)
    // Does not fill the texture with data
    void PrepareNoiseTexture(uint32_t width, uint32_t height, uint32_t depth);
    
    void createUniformBuffers();
    
    void updateNoiseTexture(uint32_t width, uint32_t height, uint32_t depth);
    
    void createRenderPasses();
    
    
    void createFrameBuffers();
    
    void createDescriptorsets();
    
    void createRenderPipelines();
    
    
    
    void addComputeToComputeBarriers(VkCommandBuffer commandBuffer)
    {
    }
    
    void recordCommandBuffer();
    
    void updateUniformBuffer();
    
    void drawFrame();
    
};
}
#endif /* VolumetricLight_h */

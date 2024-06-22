//
//  Volumetric light.cpp
//  VulkanRenderer
//
//  Created by Yang on 2024/6/14.
//

/*
 Volumetric light:
 1. camera depth, render to texture
 2. spot light mesh: cone generation
 3. down sample camera depth texture
 4. render cone
 5. horizontal blur
 6. vertical blur
 7. upscale
 8. blit add, render to screen
 9. light attenuation
 10. Dither texture
 11.3D Noise texture
 */


#define TINYOBJLOADER_IMPLEMENTATION
#include "../base/tiny_obj_loader.h"

#include "VolumetricLight.h"

#include "PerlinNoise.h"
struct Vertex{
    glm::vec3 pos;
    glm::vec2 uv;
    glm::vec3 normal;
    bool operator==(const Vertex& other) const {
        return pos == other.pos && uv == other.uv && normal == other.normal;
    }
};

struct ConeVertex{
    glm::vec3 pos;
} coneVertex;

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.uv) << 1);
        }
    };
}

namespace vks {



    void VolumetricLight::run()
    {
        initWindow();
        initVulkan();
        mainLoop();
    }

    void VolumetricLight::initVulkan() {

        
        ApplicationBase::initVulkan();
        
        //vkCreateRenderPass2KHR = (PFN_vkCreateRenderPass2KHR)vkGetDeviceProcAddr(device, "vkCreateRenderPass2KHR");
        //if(!vkCreateRenderPass2KHR){
        //    cleanup();
        //    std::cout<<"Could not get a valid function vkCreateRenderPass2KHR" << std::endl;
        //    return;
        //}
        loadAssets();
        createUniformBuffers();
        
        createRenderPasses();
        createFrameBuffers();
        createDescriptorsets();
        createRenderPipelines();
        
        recordCommandBuffer();
        
    
    }
    
    void VolumetricLight::mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            auto tStart = std::chrono::high_resolution_clock::now();
            
            glfwPollEvents();
            updateUniformBuffer();
            drawFrame();
            // We want to animate the particle system using the last frames time to get smooth, frame-rate independent animation
            
            auto tEnd = std::chrono::high_resolution_clock::now();
            auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            lastFrameTime = tDiff / 1000.0f;
            
            timer += lastFrameTime;
        }
        
        vkDeviceWaitIdle(device);
    }
    
    void VolumetricLight::cleanup() {

        if (device) {

            
            baseColor.destroy(device);
            perlinNoiseTex.destroy(device);
            
            vkDestroyRenderPass(device, scenePass.renderpass, nullptr);
            vkDestroyRenderPass(device, preLightPass.renderpass, nullptr);
            vkDestroyRenderPass(device, blurPass.renderpass, nullptr);
            vkDestroyRenderPass(device, offscreenPass.renderPass, nullptr);
            
            vkDestroyFramebuffer(device, offscreenPass.frameBuffer, nullptr);
            vkDestroyFramebuffer(device, scenePass.fbo, nullptr);
            vkDestroyFramebuffer(device, preLightPass.fbo, nullptr);
            vkDestroyFramebuffer(device, blurPass.fbo, nullptr);
            vkDestroyFramebuffer(device, vblurPassFbo, nullptr);
            
            
            offscreenPass.depth.destroy(device);
            scenePass.color.destroy(device);
            scenePass.depth.destroy(device);
            blurPass.color.destroy(device);
            resolveColor.destroy(device);
            
            
            
            vkDestroyPipeline(device, pipelines.blit, nullptr);
            vkDestroyPipeline(device, pipelines.offscreen, nullptr);
            vkDestroyPipeline(device, pipelines.sceneShadowPCF, nullptr);
            vkDestroyPipeline(device, pipelines.prelight, nullptr);
            vkDestroyPipeline(device, pipelines.gaussian, nullptr);
            
            vkDestroyPipelineLayout(device, pipelineLayouts.offscreen, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayouts.scene, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayouts.blit, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayouts.prelight, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayouts.gaussian, nullptr);
            
            vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.offscreen, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.scene, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.blit, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.prelight, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.gaussian, nullptr);
            
            // Uniform buffers
            uniformBuffers.offscreen.destroy(device);
            uniformBuffers.scene.destroy(device);
            uniformBuffers.prelight.destroy(device);
            uniformBuffers.prelightFrag.destroy(device);
            uniformBuffers.hblur.destroy(device);
            uniformBuffers.vblur.destroy(device);
        }
        
        ApplicationBase::cleanup();
    }
    
    void VolumetricLight::loadAssets(){
        baseColor = createTextureImageFromFile("textures/viking_room.png", VK_FORMAT_R8G8B8A8_UNORM);
        
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;
        std::string MODEL_PATH = "models/viking_room.obj";
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.uv = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2]
                };

                if (uniqueVertices.count(vertex) == 0)
                {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }
        {
            uint32_t vertexBufferSize = static_cast<uint32_t>(vertices.size()) * sizeof(vertices[0]);

            Buffer stagingBuffer = createBuffer(vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            vkMapMemory(device, stagingBuffer.bufferMemory, 0, vertexBufferSize, 0, &stagingBuffer.bufferMapped);
            memcpy(stagingBuffer.bufferMapped, vertices.data(), (size_t)vertexBufferSize);
            vkUnmapMemory(device, stagingBuffer.bufferMemory);
            
            scene.verticesBuffer = createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            copyBuffer(stagingBuffer.buffer, scene.verticesBuffer.buffer, vertexBufferSize);

            stagingBuffer.destroy(device);
        }
        {
            uint32_t indexBufferSize = static_cast<uint32_t>(indices.size()) * sizeof(indices[0]);

            Buffer stagingBuffer = createBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            vkMapMemory(device, stagingBuffer.bufferMemory, 0, indexBufferSize, 0, &stagingBuffer.bufferMapped);
            memcpy(stagingBuffer.bufferMapped, indices.data(), (size_t)indexBufferSize);
            vkUnmapMemory(device, stagingBuffer.bufferMemory);
            
            scene.indicesBuffer = createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            copyBuffer(stagingBuffer.buffer, scene.indicesBuffer.buffer, indexBufferSize);

            stagingBuffer.destroy(device);
        }
        scene.size = indices.size();
        
        CreateSpotLightMesh();
        
        PrepareNoiseTexture(64, 64, 64);
    }
    
    void VolumetricLight::CreateSpotLightMesh(){
        const int segmentCount = 16;
        
        std::vector<glm::vec3> vertices(2 + segmentCount * 3);

        vertices[0] = {0, 0, 0};
        vertices[1] = {0, 0, 0};

        float angle = 0;
        float step = glm::two_pi<float>() / segmentCount;
        float ratio = 0.9f;

        for (int i = 0; i < segmentCount; ++i)
        {
            vertices[i + 2] = {-cos(angle) * ratio, sin(angle) * ratio, ratio};
            vertices[i + 2 + segmentCount] = {-cos(angle), sin(angle), 1};
            vertices[i + 2 + segmentCount * 2] = {-cos(angle) * ratio, sin(angle) * ratio, 1};
            angle += step;
        }
        coneModel.size = segmentCount * 3 * 2 + segmentCount * 6 * 2;
        
        std::vector<uint32_t> indices(coneModel.size);
        int index = 0;

        for (int i = 2; i < segmentCount + 1; ++i)
        {
            indices[index++] = 0;
            indices[index++] = i;
            indices[index++] = i + 1;
        }

        indices[index++] = 0;
        indices[index++] = segmentCount + 1;
        indices[index++] = 2;

        for (int i = 2; i < segmentCount + 1; ++i)
        {
            indices[index++] = i;
            indices[index++] = i + segmentCount;
            indices[index++] = i + 1;

            indices[index++] = i + 1;
            indices[index++] = i + segmentCount;
            indices[index++] = i + segmentCount + 1;
        }

        indices[index++] = 2;
        indices[index++] = 1 + segmentCount;
        indices[index++] = 2 + segmentCount;

        indices[index++] = 2 + segmentCount;
        indices[index++] = 1 + segmentCount;
        indices[index++] = 1 + segmentCount + segmentCount;

        //------------
        for (int i = 2 + segmentCount; i < segmentCount + 1 + segmentCount; ++i)
        {
            indices[index++] = i;
            indices[index++] = i + segmentCount;
            indices[index++] = i + 1;

            indices[index++] = i + 1;
            indices[index++] = i + segmentCount;
            indices[index++] = i + segmentCount + 1;
        }

        indices[index++] = 2 + segmentCount;
        indices[index++] = 1 + segmentCount * 2;
        indices[index++] = 2 + segmentCount * 2;

        indices[index++] = 2 + segmentCount * 2;
        indices[index++] = 1 + segmentCount * 2;
        indices[index++] = 1 + segmentCount * 3;

        ////-------------------------------------
        for (int i = 2 + segmentCount * 2; i < segmentCount * 3 + 1; ++i)
        {
            indices[index++] = 1;
            indices[index++] = i + 1;
            indices[index++] = i;
        }

        indices[index++] = 1;
        indices[index++] = 2 + segmentCount * 2;
        indices[index++] = segmentCount * 3 + 1;
        
        {
            uint32_t vertexBufferSize = static_cast<uint32_t>(vertices.size()) * sizeof(vertices[0]);

            Buffer stagingBuffer = createBuffer(vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            vkMapMemory(device, stagingBuffer.bufferMemory, 0, vertexBufferSize, 0, &stagingBuffer.bufferMapped);
            memcpy(stagingBuffer.bufferMapped, vertices.data(), (size_t)vertexBufferSize);
            vkUnmapMemory(device, stagingBuffer.bufferMemory);
            
            coneModel.verticesBuffer = createBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            copyBuffer(stagingBuffer.buffer, coneModel.verticesBuffer.buffer, vertexBufferSize);

            stagingBuffer.destroy(device);
        }
        {
            uint32_t indexBufferSize = static_cast<uint32_t>(indices.size()) * sizeof(indices[0]);

            Buffer stagingBuffer = createBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            vkMapMemory(device, stagingBuffer.bufferMemory, 0, indexBufferSize, 0, &stagingBuffer.bufferMapped);
            memcpy(stagingBuffer.bufferMapped, indices.data(), (size_t)indexBufferSize);
            vkUnmapMemory(device, stagingBuffer.bufferMemory);
            
            coneModel.indicesBuffer = createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            copyBuffer(stagingBuffer.buffer, coneModel.indicesBuffer.buffer, indexBufferSize);

            stagingBuffer.destroy(device);
        }
    }
    
    // Prepare all Vulkan resources for the 3D texture (including descriptors)
    // Does not fill the texture with data
    void VolumetricLight::PrepareNoiseTexture(uint32_t width, uint32_t height, uint32_t depth)
    {
        // A 3D texture is described as width x height x depth

        // Format support check
        // 3D texture support in Vulkan is mandatory (in contrast to OpenGL) so no need to check if it's supported
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8_UNORM, &formatProperties);
        // Check if format supports transfer
        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT))
        {
            std::cout << "Error: Device does not support flag TRANSFER_DST for selected texture format!" << std::endl;
            return;
        }
        // Check if GPU supports requested 3D texture dimensions
        uint32_t maxImageDimension3D(properties.limits.maxImageDimension3D);
        if (width > maxImageDimension3D || height > maxImageDimension3D || depth > maxImageDimension3D)
        {
            std::cout << "Error: Requested texture dimensions is greater than supported 3D texture dimension!" << std::endl;
            return;
        }

        // Create optimal tiled target image
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
        imageCreateInfo.format = VK_FORMAT_R8_UNORM;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.extent.width = width;
        imageCreateInfo.extent.height = height;
        imageCreateInfo.extent.depth = depth;
        // Set initial layout of the image to undefined
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &perlinNoiseTex.image));

        // Device local memory to back up image
        VkMemoryRequirements memRequirements = {};
        vkGetImageMemoryRequirements(device, perlinNoiseTex.image, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &perlinNoiseTex.imageMemory));
        VK_CHECK_RESULT(vkBindImageMemory(device, perlinNoiseTex.image, perlinNoiseTex.imageMemory, 0));

        // Create sampler
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.maxAnisotropy = 1.0;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VK_CHECK_RESULT(vkCreateSampler(device, &samplerInfo, nullptr, &perlinNoiseTex.sampler));

        // Create image view
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = perlinNoiseTex.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = VK_FORMAT_R8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        viewInfo.subresourceRange.levelCount = 1;
        VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &perlinNoiseTex.imageView));

        // Fill image descriptor image info to be used descriptor set setup
        perlinNoiseTex.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        perlinNoiseTex.imageInfo.imageView = perlinNoiseTex.imageView;
        perlinNoiseTex.imageInfo.sampler = perlinNoiseTex.sampler;

        updateNoiseTexture(width, height, depth);
    }

    void VolumetricLight::updateNoiseTexture(uint32_t width, uint32_t height, uint32_t depth){
        const uint32_t texMemSize = width * height * depth;

        uint8_t *data = new uint8_t[texMemSize];
        memset(data, 0, texMemSize);

        // Generate perlin based noise
        std::cout << "Generating " << width << " x " << height << " x " << depth << " noise texture..." << std::endl;

        auto tStart = std::chrono::high_resolution_clock::now();

        PerlinNoise<float> perlinNoise(true);
        FractalNoise<float> fractalNoise(perlinNoise);

        const float noiseScale = static_cast<float>(rand() % 10) + 4.0f;

#pragma omp parallel for
        for (int32_t z = 0; z < static_cast<int32_t>(depth); z++)
        {
            for (int32_t y = 0; y < static_cast<int32_t>(height); y++)
            {
                for (int32_t x = 0; x < static_cast<int32_t>(width); x++)
                {
                    float nx = (float)x / (float)width;
                    float ny = (float)y / (float)height;
                    float nz = (float)z / (float)depth;
                    float n = fractalNoise.noise(nx * noiseScale, ny * noiseScale, nz * noiseScale);
                    n = n - floor(n);
                    data[x + y * width + z * width * height] = static_cast<uint8_t>(floor(n * 255));
                }
            }
        }

        auto tEnd = std::chrono::high_resolution_clock::now();
        auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        std::cout << "Done in " << tDiff << "ms" << std::endl;

        // Create a host-visible staging buffer that contains the raw image data
        VkDeviceMemory stagingMemory;

        // Buffer object
        Buffer stagingBuffer = createBuffer(texMemSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(device, stagingBuffer.bufferMemory, 0, texMemSize, 0, &stagingBuffer.bufferMapped);
        memcpy(stagingBuffer.bufferMapped, data, (size_t)texMemSize);
        vkUnmapMemory(device, stagingBuffer.bufferMemory);

        transitionImageLayout(perlinNoiseTex.image, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        
        VkCommandBuffer copyCmd = beginSingleTimeCommands();
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {
            width,
            height,
            depth
        };
        
        vkCmdCopyBufferToImage(copyCmd, stagingBuffer.buffer, perlinNoiseTex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        endSingleTimeCommands(copyCmd);

        // Change texture image layout to shader read after all mip levels have been copied
        transitionImageLayout(perlinNoiseTex.image, VK_FORMAT_R8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


        // Clean up staging resources
        delete[] data;
        stagingBuffer.destroy(device);
    }
    
    void VolumetricLight::createUniformBuffers(){
        //update every frame, so it's host visible to allow memcpy
        uniformBuffers.offscreen = createBuffer(sizeof(uniformDataOffscreen), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, uniformBuffers.offscreen.bufferMemory, 0, sizeof(uniformDataOffscreen), 0, &uniformBuffers.offscreen.bufferMapped);
        
        uniformBuffers.scene = createBuffer(sizeof(uniformDataScene), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, uniformBuffers.scene.bufferMemory, 0, sizeof(uniformDataScene), 0, &uniformBuffers.scene.bufferMapped);
        
        uniformBuffers.prelight = createBuffer(sizeof(uniformDataPrelight), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, uniformBuffers.prelight.bufferMemory, 0, sizeof(uniformDataPrelight), 0, &uniformBuffers.prelight.bufferMapped);
        
        uniformBuffers.prelightFrag = createBuffer(sizeof(uniformDataPrelightFrag), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, uniformBuffers.prelightFrag.bufferMemory, 0, sizeof(uniformDataPrelightFrag), 0, &uniformBuffers.prelightFrag.bufferMapped);
        
        uniformBuffers.hblur = createBuffer(sizeof(uniformDataGaussian), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, uniformBuffers.hblur.bufferMemory, 0, sizeof(uniformDataGaussian), 0, &uniformBuffers.hblur.bufferMapped);
        
        uniformBuffers.vblur = createBuffer(sizeof(uniformDataGaussian), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, uniformBuffers.vblur.bufferMemory, 0, sizeof(uniformDataGaussian), 0, &uniformBuffers.vblur.bufferMapped);
        
    }
    
    void VolumetricLight::createRenderPasses(){
        //offscreenPass.renderPass, shadowmap pass
        {
            VkRenderPassCreateInfo renderPassCreateInfo = initializers::renderPassCreateInfo();
            //typedef struct VkRenderPassCreateInfo {
            //    VkStructureType                   sType;
            //    const void*                       pNext;
            //    VkRenderPassCreateFlags           flags;
            //    uint32_t                          attachmentCount;
            //    const VkAttachmentDescription*    pAttachments;
            //    uint32_t                          subpassCount;
            //    const VkSubpassDescription*       pSubpasses;
            //    uint32_t                          dependencyCount;
            //    const VkSubpassDependency*        pDependencies;
            //} VkRenderPassCreateInfo;
            VkAttachmentDescription attachmentDescription{};
            //typedef struct VkAttachmentDescription {
            //    VkAttachmentDescriptionFlags    flags;
            //    VkFormat                        format;
            //    VkSampleCountFlagBits           samples;
            //    VkAttachmentLoadOp              loadOp;
            //    VkAttachmentStoreOp             storeOp;
            //    VkAttachmentLoadOp              stencilLoadOp;
            //    VkAttachmentStoreOp             stencilStoreOp;
            //    VkImageLayout                   initialLayout;
            //    VkImageLayout                   finalLayout;
            //} VkAttachmentDescription;
            
            attachmentDescription.format = offscreenDepthFormat;
            //not multi-sampled
            attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
            attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            //finalLayout is the layout the attachment image subresource will be transitioned to when a render pass instance ends.
            attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            
            VkSubpassDescription subPassDescription{};
            /*
             VkSubpassDescriptionFlags       flags;
             VkPipelineBindPoint             pipelineBindPoint;
             uint32_t                        inputAttachmentCount;
             const VkAttachmentReference*    pInputAttachments;
             uint32_t                        colorAttachmentCount;
             const VkAttachmentReference*    pColorAttachments;
             const VkAttachmentReference*    pResolveAttachments;
             const VkAttachmentReference*    pDepthStencilAttachment;
             uint32_t                        preserveAttachmentCount;
             const uint32_t*                 pPreserveAttachments;
             */
            
            VkAttachmentReference depthReference = {};
            //attachment is either an integer value identifying an attachment at the corresponding index in VkRenderPassCreateInfo::pAttachments
            depthReference.attachment = 0;
            depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            
            subPassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subPassDescription.colorAttachmentCount = 0;
            subPassDescription.pDepthStencilAttachment = &depthReference;
            
            VkSubpassDependency dependencies[2];
            dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass = 0;
            dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            dependencies[1].srcSubpass = 0;
            dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            renderPassCreateInfo.attachmentCount = 1;
            renderPassCreateInfo.pAttachments = &attachmentDescription;
            renderPassCreateInfo.subpassCount = 1;
            renderPassCreateInfo.pSubpasses = &subPassDescription;
            renderPassCreateInfo.dependencyCount = 2;
            renderPassCreateInfo.pDependencies = dependencies;
            
            VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &offscreenPass.renderPass));
        }
        //scene renderpass
        {
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = VK_FORMAT_R16G16B16A16_UNORM;
            colorAttachment.samples = VK_SAMPLE_COUNT_4_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            
            VkAttachmentDescription depthAttachment{};
            depthAttachment.format = findDepthFormat();
            depthAttachment.samples = VK_SAMPLE_COUNT_4_BIT;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            
            VkAttachmentDescription colorAttachmentResolve{};
                colorAttachmentResolve.format = VK_FORMAT_R16G16B16A16_UNORM;
                colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            
            
            VkAttachmentReference colorAttachmentRef{};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            
            VkAttachmentReference depthAttachmentRef{};
            depthAttachmentRef.attachment = 1;
            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            
            VkAttachmentReference colorAttachmentResolveRef{};
            colorAttachmentResolveRef.attachment = 2;
            colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        
            
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;
            subpass.pDepthStencilAttachment = &depthAttachmentRef;
            subpass.pResolveAttachments = &colorAttachmentResolveRef;
            
            
            VkSubpassDependency dependency[2];
            dependency[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency[0].dstSubpass = 0;
            //wait for this of the src subpass
            dependency[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dependency[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            //wait on this of our pass
            dependency[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dependency[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            dependency[1].srcSubpass = 0;
            dependency[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependency[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            dependency[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dependency[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dependency[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
            VkRenderPassCreateInfo renderPassCreateInfo{};
            renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            renderPassCreateInfo.pAttachments = attachments.data();
            renderPassCreateInfo.subpassCount = 1;
            renderPassCreateInfo.pSubpasses = &subpass;
            renderPassCreateInfo.dependencyCount = 2;
            renderPassCreateInfo.pDependencies = dependency;
            
            VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &scenePass.renderpass));
            
        }
        //down sampling renderpass
        /*
        {
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = VK_FORMAT_R16G16B16A16_UNORM;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            
            VkAttachmentReference colorAttachmentRef{};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;

            VkSubpassDependency dependency[2];
            dependency[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency[0].dstSubpass = 0;
            //wait for this of the src subpass
            dependency[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            //wait on this of our pass
            dependency[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            dependency[1].srcSubpass = 0;
            dependency[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependency[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dependency[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            std::array<VkAttachmentDescription, 1> attachments = {colorAttachment};
            VkRenderPassCreateInfo renderPassCreateInfo{};
            renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            renderPassCreateInfo.pAttachments = attachments.data();
            renderPassCreateInfo.subpassCount = 1;
            renderPassCreateInfo.pSubpasses = &subpass;
            renderPassCreateInfo.dependencyCount = 2;
            renderPassCreateInfo.pDependencies = dependency;
            
            VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &downSamplingPass.renderpass));
        }
         */
        //spot light pass
        {
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = VK_FORMAT_R16G16B16A16_UNORM;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            
            VkAttachmentReference colorAttachmentRef{};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;

            VkSubpassDependency dependency[2];
            dependency[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency[0].dstSubpass = 0;
            //wait for this of the src subpass
            dependency[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            //wait on this of our pass
            dependency[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            dependency[1].srcSubpass = 0;
            dependency[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependency[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dependency[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            std::array<VkAttachmentDescription, 1> attachments = {colorAttachment};
            VkRenderPassCreateInfo renderPassCreateInfo{};
            renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            renderPassCreateInfo.pAttachments = attachments.data();
            renderPassCreateInfo.subpassCount = 1;
            renderPassCreateInfo.pSubpasses = &subpass;
            renderPassCreateInfo.dependencyCount = 2;
            renderPassCreateInfo.pDependencies = dependency;
            
            VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &preLightPass.renderpass));
        }
        //gaussian blur, horizontal and vertical pass
        {
            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = VK_FORMAT_R16G16B16A16_UNORM;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            
            VkAttachmentReference colorAttachmentRef{};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;

            VkSubpassDependency dependency[2];
            dependency[0].srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency[0].dstSubpass = 0;
            //wait for this of the src subpass
            dependency[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            //wait on this of our pass
            dependency[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            dependency[1].srcSubpass = 0;
            dependency[1].dstSubpass = VK_SUBPASS_EXTERNAL;
            dependency[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            dependency[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependency[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dependency[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
            
            std::array<VkAttachmentDescription, 1> attachments = {colorAttachment};
            VkRenderPassCreateInfo renderPassCreateInfo{};
            renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            renderPassCreateInfo.pAttachments = attachments.data();
            renderPassCreateInfo.subpassCount = 1;
            renderPassCreateInfo.pSubpasses = &subpass;
            renderPassCreateInfo.dependencyCount = 2;
            renderPassCreateInfo.pDependencies = dependency;
            
            VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &blurPass.renderpass));
        }
        
    }

    
    void VolumetricLight::createFrameBuffers(){
        {
            //shadow pass fbo, depth only
            offscreenPass.width = shadowMapize;
            offscreenPass.height = shadowMapize;
            
            VkFormat depthFormat = offscreenDepthFormat;
            createImage(shadowMapize, shadowMapize,
                        depthFormat,
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        offscreenPass.depth.image, offscreenPass.depth.imageMemory);
            offscreenPass.depth.imageView = createImageView(offscreenPass.depth.image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
            
            VkSamplerCreateInfo samplerCreateInfo{};
            samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
            samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
            samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerCreateInfo.mipLodBias = 0.0f;
            samplerCreateInfo.maxAnisotropy = 1.0f;
            samplerCreateInfo.minLod = 0.0f;
            samplerCreateInfo.maxLod = 1.0f;
            samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            VK_CHECK_RESULT(vkCreateSampler(device, &samplerCreateInfo, nullptr, &offscreenPass.depth.sampler));
            
            offscreenPass.depth.imageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            offscreenPass.depth.imageInfo.imageView = offscreenPass.depth.imageView;
            offscreenPass.depth.imageInfo.sampler = offscreenPass.depth.sampler;
            
            
            VkFramebufferCreateInfo framebufferCreateInfo = initializers::framebufferCreateInfo();
            /*
             VkRenderPass                renderPass;
             uint32_t                    attachmentCount;
             const VkImageView*          pAttachments;
             uint32_t                    width;
             uint32_t                    height;
             uint32_t                    layers;
             */
            framebufferCreateInfo.renderPass = offscreenPass.renderPass;
            framebufferCreateInfo.attachmentCount = 1;
            framebufferCreateInfo.pAttachments = &offscreenPass.depth.imageView;
            framebufferCreateInfo.width = shadowMapize;
            framebufferCreateInfo.height = shadowMapize;
            framebufferCreateInfo.layers = 1;
            VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &offscreenPass.frameBuffer));
        }
        
        {
            createImage(swapChainExtent.width, swapChainExtent.height, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scenePass.color.image, scenePass.color.imageMemory,
                        VK_SAMPLE_COUNT_4_BIT);
            scenePass.color.imageView = createImageView(scenePass.color.image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
            scenePass.color.sampler = createTextureSampler();
            
            
            VkFormat depthFormat = findDepthFormat();
            createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                scenePass.depth.image, scenePass.depth.imageMemory,
                        VK_SAMPLE_COUNT_4_BIT);
            scenePass.depth.imageView = createImageView(scenePass.depth.image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
            scenePass.depth.sampler = createTextureSampler();
            
            
            
            {
                createImage(swapChainExtent.width, swapChainExtent.height, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, resolveColor.image, resolveColor.imageMemory);
                resolveColor.imageView = createImageView(resolveColor.image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
                resolveColor.sampler = createTextureSampler();
                
                resolveColor.imageInfo = {
                    resolveColor.sampler,
                    resolveColor.imageView,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };
                
            }
        
            
            std::array<VkImageView, 3> attachments = {scenePass.color.imageView, scenePass.depth.imageView, resolveColor.imageView};
            VkFramebufferCreateInfo framebufferCreateInfo = initializers::framebufferCreateInfo();
            framebufferCreateInfo.renderPass = scenePass.renderpass;
            framebufferCreateInfo.attachmentCount = attachments.size();
            framebufferCreateInfo.pAttachments = attachments.data();
            framebufferCreateInfo.width = swapChainExtent.width;
            framebufferCreateInfo.height = swapChainExtent.height;
            framebufferCreateInfo.layers = 1;
            VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &scenePass.fbo));
            
            
            /*
            {
                createImage(swapChainExtent.width/2, swapChainExtent.height/2, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fbo.halfResolutionColor.image, fbo.halfResolutionColor.imageMemory);
                fbo.halfResolutionColor.imageView = createImageView(fbo.halfResolutionColor.image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
                fbo.halfResolutionColor.sampler = createTextureSampler();
                
                fbo.halfResolutionColor.imageInfo = {
                    fbo.halfResolutionColor.sampler,
                    fbo.halfResolutionColor.imageView,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };
            }
            {
                createImage(swapChainExtent.width/2, swapChainExtent.height/2, depthFormat, VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    fbo.halfResolutionDepth.image, fbo.halfResolutionDepth.imageMemory);
                fbo.halfResolutionDepth.imageView = createImageView(fbo.halfResolutionDepth.image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
                fbo.halfResolutionDepth.sampler = createTextureSampler();
                
                fbo.halfResolutionDepth.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                fbo.halfResolutionDepth.imageInfo.imageView = fbo.halfResolutionDepth.imageView;
                fbo.halfResolutionDepth.imageInfo.sampler =   fbo.halfResolutionDepth.sampler;
                
                framebufferCreateInfo.renderPass = downSamplingPass.renderpass;
                framebufferCreateInfo.attachmentCount = 1;
                framebufferCreateInfo.pAttachments = &fbo.halfResolutionDepth.imageView;
                framebufferCreateInfo.width = swapChainExtent.width / 2;
                framebufferCreateInfo.height = swapChainExtent.height / 2;
                VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &downSamplingPass.fbo));
            }
            */
        }
        
        {
            createImage(swapChainExtent.width / 2, swapChainExtent.height / 2, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, preLightPass.color.image, preLightPass.color.imageMemory);
            preLightPass.color.imageView = createImageView(preLightPass.color.image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
            preLightPass.color.sampler = createTextureSampler();
            
            preLightPass.color.imageInfo = {
                preLightPass.color.sampler,
                preLightPass.color.imageView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };
        
            
            std::array<VkImageView, 1> attachments = {preLightPass.color.imageView};
            VkFramebufferCreateInfo framebufferCreateInfo = initializers::framebufferCreateInfo();
            framebufferCreateInfo.renderPass = preLightPass.renderpass;
            framebufferCreateInfo.attachmentCount = 1;
            framebufferCreateInfo.pAttachments = attachments.data();
            framebufferCreateInfo.width = swapChainExtent.width / 2;
            framebufferCreateInfo.height = swapChainExtent.height / 2;
            framebufferCreateInfo.layers = 1;
            VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &preLightPass.fbo));
 
        }
        //fbo for blur
        {
            createImage(swapChainExtent.width / 2, swapChainExtent.height / 2, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blurPass.color.image, blurPass.color.imageMemory);
            blurPass.color.imageView = createImageView(blurPass.color.image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
            blurPass.color.sampler = createTextureSampler();
            
            blurPass.color.imageInfo = {
                blurPass.color.sampler,
                blurPass.color.imageView,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };
        
            std::array<VkImageView, 1> attachments = {blurPass.color.imageView};
            VkFramebufferCreateInfo framebufferCreateInfo = initializers::framebufferCreateInfo();
            framebufferCreateInfo.renderPass = blurPass.renderpass;
            framebufferCreateInfo.attachmentCount = 1;
            framebufferCreateInfo.pAttachments = attachments.data();
            framebufferCreateInfo.width = swapChainExtent.width / 2;
            framebufferCreateInfo.height = swapChainExtent.height / 2;
            framebufferCreateInfo.layers = 1;
            VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &blurPass.fbo));
            
            framebufferCreateInfo.pAttachments = &preLightPass.color.imageView;
            VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &vblurPassFbo));
 
        }
    }
    
    void VolumetricLight::createDescriptorsets(){
        std::vector<VkDescriptorPoolSize> poolSize = {
            initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6),
            initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 12)
        };
        VkDescriptorPoolCreateInfo poolCreateInfo = initializers::descriptorPoolCreateInfo(poolSize, 7);
        vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool);
        {
            // Layout
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                // Binding 0 : Vertex shader uniform buffer
                vks::initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
            };
        
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.offscreen));
        
            VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, 1, &descriptorSetLayouts.offscreen);
        
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.offscreen));
            std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                // Binding 0 : Vertex shader uniform buffer
                vks::initializers::writeDescriptorSet(descriptorSets.offscreen, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uniformBuffers.offscreen.bufferInfo),
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
        }
        {
            // Layout
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                // Binding 0 : Vertex shader uniform buffer
                vks::initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT),
                // Binding 1 : Fragment shader sampler (shadow map)
                vks::initializers::descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
                // Binding 2: Fragment shader sampler ( baseCColor)
                vks::initializers::descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            };
        
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.scene));
        
            VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, 1, &descriptorSetLayouts.scene);
            
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.scene));
            std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                // Binding 0 : Vertex shader uniform buffer
                vks::initializers::writeDescriptorSet(descriptorSets.scene, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uniformBuffers.scene.bufferInfo),
                vks::initializers::writeDescriptorSet(descriptorSets.scene, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &offscreenPass.depth.imageInfo, nullptr),
                vks::initializers::writeDescriptorSet(descriptorSets.scene, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &baseColor.imageInfo, nullptr)
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
        }

        //prelight
        {
            // Layout
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                // Binding 0: Vertex shader uniform
                vks::initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT),
                vks::initializers::descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
                vks::initializers::descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
                vks::initializers::descriptorSetLayoutBinding(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT),
                vks::initializers::descriptorSetLayoutBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            };
        
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.prelight));
        
            VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, 1, &descriptorSetLayouts.prelight);
            
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.prelight));
            std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                vks::initializers::writeDescriptorSet(descriptorSets.prelight, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uniformBuffers.prelight.bufferInfo),
                vks::initializers::writeDescriptorSet(descriptorSets.prelight, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &offscreenPass.depth.imageInfo, nullptr),
                vks::initializers::writeDescriptorSet(descriptorSets.prelight, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &resolveColor.imageInfo, nullptr),
                vks::initializers::writeDescriptorSet(descriptorSets.prelight, 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uniformBuffers.prelightFrag.bufferInfo),
                vks::initializers::writeDescriptorSet(descriptorSets.prelight, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &perlinNoiseTex.imageInfo, nullptr)
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
        }
        //blur
        {
            // Layout
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                vks::initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
                vks::initializers::descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
                vks::initializers::descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT),
            };
        
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.gaussian));
        
            VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, 1, &descriptorSetLayouts.gaussian);
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.hblur));
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.vblur));
            
            std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                vks::initializers::writeDescriptorSet(descriptorSets.hblur, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &preLightPass.color.imageInfo, nullptr),
                vks::initializers::writeDescriptorSet(descriptorSets.hblur, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &resolveColor.imageInfo, nullptr),
                vks::initializers::writeDescriptorSet(descriptorSets.hblur, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uniformBuffers.hblur.bufferInfo),
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
            
            writeDescriptorSets = {
                vks::initializers::writeDescriptorSet(descriptorSets.vblur, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &blurPass.color.imageInfo, nullptr),
                vks::initializers::writeDescriptorSet(descriptorSets.vblur, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &resolveColor.imageInfo, nullptr),
                vks::initializers::writeDescriptorSet(descriptorSets.vblur, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uniformBuffers.vblur.bufferInfo),
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
        }
        
        //downsampling
        /*
        {
            // Layout
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                // Binding 0: Fragment shader sampler ( full scale depth)
                vks::initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            };
        
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.downsampling));
        
            VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, 1, &descriptorSetLayouts.downsampling);
            
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.downsampling));
            std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                vks::initializers::writeDescriptorSet(descriptorSets.downsampling, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &fbo.halfResolutionDepth.imageInfo, nullptr)
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
        }
        */
        
        //blit to screen
        {
            // Layout
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                // Binding 0: Fragment shader sampler ( blit source 1)
                vks::initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
                // Binding 1: Fragment shader sampler ( blit source 2)
                vks::initializers::descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
            };
        
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayouts.blit));
        
            VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, 1, &descriptorSetLayouts.blit);
            
            VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets.blit));
            std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
                vks::initializers::writeDescriptorSet(descriptorSets.blit, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &resolveColor.imageInfo, nullptr),
                vks::initializers::writeDescriptorSet(descriptorSets.blit, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &preLightPass.color.imageInfo, nullptr)
            };
            vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
        }
    }
    
    void VolumetricLight::createRenderPipelines(){
        {
            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = initializers::pipelineLayoutCreateInfo(1, &descriptorSetLayouts.offscreen);
            /*
             uint32_t                        setLayoutCount;
             const VkDescriptorSetLayout*    pSetLayouts;
             uint32_t                        pushConstantRangeCount;
             const VkPushConstantRange*      pPushConstantRanges;
             */
            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.offscreen));
        }
        {
            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = initializers::pipelineLayoutCreateInfo(1, &descriptorSetLayouts.scene);
            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.scene));
        }
        {
            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = initializers::pipelineLayoutCreateInfo(1, &descriptorSetLayouts.blit);
            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.blit));
        }
        {
            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = initializers::pipelineLayoutCreateInfo(1, &descriptorSetLayouts.prelight);
            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.prelight));
        }
        {
            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = initializers::pipelineLayoutCreateInfo(1, &descriptorSetLayouts.gaussian);
            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.gaussian));
        }
        /*
        {
            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = initializers::pipelineLayoutCreateInfo(1, &descriptorSetLayouts.downsampling);
            VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.downsampling));
        }
         */
        //pipelines.offscreen
        VkGraphicsPipelineCreateInfo pipelineCreateInfo = initializers::pipelineCreateInfo();
        /*
         uint32_t                                         stageCount;
         const VkPipelineShaderStageCreateInfo*           pStages;
         const VkPipelineVertexInputStateCreateInfo*      pVertexInputState;
         const VkPipelineInputAssemblyStateCreateInfo*    pInputAssemblyState;
         const VkPipelineTessellationStateCreateInfo*     pTessellationState;
         const VkPipelineViewportStateCreateInfo*         pViewportState;
         const VkPipelineRasterizationStateCreateInfo*    pRasterizationState;
         const VkPipelineMultisampleStateCreateInfo*      pMultisampleState;
         const VkPipelineDepthStencilStateCreateInfo*     pDepthStencilState;
         const VkPipelineColorBlendStateCreateInfo*       pColorBlendState;
         const VkPipelineDynamicStateCreateInfo*          pDynamicState;
         VkPipelineLayout                                 layout;
         VkRenderPass                                     renderPass;
         uint32_t                                         subpass;
         VkPipeline                                       basePipelineHandle;
         int32_t                                          basePipelineIndex;
         */
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
            loadShader("shaders/VolumetricLight/offscreen.vert.spv", VK_SHADER_STAGE_VERTEX_BIT)
        };
        VkPipelineVertexInputStateCreateInfo vertexInput = initializers::pipelineVertexInputStateCreateInfo();
        /*
         uint32_t                                    vertexBindingDescriptionCount;
         const VkVertexInputBindingDescription*      pVertexBindingDescriptions;
         uint32_t                                    vertexAttributeDescriptionCount;
         const VkVertexInputAttributeDescription*    pVertexAttributeDescriptions;
         */
        
        VkVertexInputBindingDescription vertexInputBinding = initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX);

        std::vector<VkVertexInputAttributeDescription> vertexInputAttrib = {
            initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
            initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)),
            initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
        };
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &vertexInputBinding;
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttrib.size());
        vertexInput.pVertexAttributeDescriptions = vertexInputAttrib.data();
      
        
        VkPipelineInputAssemblyStateCreateInfo assemblyState = initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, false);
        
        VkPipelineRasterizationStateCreateInfo rasterizationState = initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
        rasterizationState.depthBiasEnable = VK_TRUE;
        VkPipelineColorBlendAttachmentState blendAttachmentState = initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo colorBlendState = initializers::pipelineColorBlendStateCreateInfo(0, &blendAttachmentState);
        VkPipelineDepthStencilStateCreateInfo depthStencilState = initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
        VkPipelineViewportStateCreateInfo viewportState = initializers::pipelineViewportStateCreateInfo(1, 1, 0);
        VkPipelineMultisampleStateCreateInfo multisampleState = initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
        
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, //VK_DYNAMIC_STATE_DEPTH_BIAS
            
        };
        VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

        
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();
        pipelineCreateInfo.pVertexInputState = &vertexInput;
        pipelineCreateInfo.pInputAssemblyState = &assemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.layout = pipelineLayouts.offscreen;
        pipelineCreateInfo.renderPass = offscreenPass.renderPass;
        
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.offscreen));

        //create scene render pipeline
 
        //pipelines.offscreen

        shaderStages = {
            loadShader("shaders/VolumetricLight/scene.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader("shaders/VolumetricLight/scene.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
        };
        
        rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
        dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
        colorBlendState = initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
        rasterizationState.depthBiasEnable = VK_FALSE;
        
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.layout = pipelineLayouts.scene;
        pipelineCreateInfo.renderPass = scenePass.renderpass;

        multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_4_BIT;
        
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.sceneShadowPCF));
  
        multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        shaderStages = {
            loadShader("shaders/VolumetricLight/quad.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader("shaders/VolumetricLight/quad.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
        };
        pipelineCreateInfo.pStages = shaderStages.data();
        pipelineCreateInfo.stageCount = 2;
        rasterizationState.cullMode = VK_CULL_MODE_NONE;
        VkPipelineVertexInputStateCreateInfo emptyVertex = initializers::pipelineVertexInputStateCreateInfo();
        pipelineCreateInfo.pVertexInputState = &emptyVertex;
        pipelineCreateInfo.layout = pipelineLayouts.blit;
        pipelineCreateInfo.renderPass = renderPass;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.blit));
  
        pipelineCreateInfo.layout = pipelineLayouts.gaussian;
        pipelineCreateInfo.renderPass = blurPass.renderpass;
        shaderStages[1] = loadShader("shaders/VolumetricLight/gaussian.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.gaussian));
        
        shaderStages = {
            loadShader("shaders/VolumetricLight/spotlight.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader("shaders/VolumetricLight/spotlight.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
        };
        rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
        pipelineCreateInfo.pVertexInputState = &vertexInput;
        vertexInputBinding = initializers::vertexInputBindingDescription(0, sizeof(ConeVertex), VK_VERTEX_INPUT_RATE_VERTEX);
        vertexInputAttrib = {
            initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ConeVertex, pos))
        };
        vertexInput.vertexAttributeDescriptionCount = 1;
        pipelineCreateInfo.layout = pipelineLayouts.prelight;
        pipelineCreateInfo.renderPass = preLightPass.renderpass;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.prelight));
        
        /*
        shaderStages = {
            loadShader("shaders/VolumetricLight/downsampling.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader("shaders/VolumetricLight/downsampling.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
        };
        pipelineCreateInfo.layout = pipelineLayouts.downsampling;
        pipelineCreateInfo.renderPass = downSamplingPass.renderpass;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.downsampling));
  */
    }
    
    
    void VolumetricLight::recordCommandBuffer() {
        VkCommandBufferBeginInfo commandBufferbeginInfo = initializers::commandBufferBeginInfo();
        for(uint i = 0; i < graphicsCommandBuffer.size(); i++)
        {
            VK_CHECK_RESULT(vkBeginCommandBuffer(graphicsCommandBuffer[i], &commandBufferbeginInfo));
            //shadow pass
            {
                VkClearValue clearValues;
                clearValues.depthStencil.depth = 1.0;
                clearValues.depthStencil.stencil = 0.0;
                VkViewport viewPort = initializers::viewport(offscreenPass.width, offscreenPass.height, 0, 1);
                VkRect2D rect2D = initializers::rect2D(offscreenPass.width, offscreenPass.height, 0, 0);
                //vkCmdBeginRenderPass
                VkRenderPassBeginInfo renderPassBeginInfo= initializers::renderPassBeginInfo();
                //VkRenderPass           renderPass;
                //VkFramebuffer          framebuffer;
                //VkRect2D               renderArea;
                //uint32_t               clearValueCount;
                //const VkClearValue*    pClearValues;
                renderPassBeginInfo.renderPass = offscreenPass.renderPass;
                renderPassBeginInfo.framebuffer = offscreenPass.frameBuffer;
                renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
                renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
                renderPassBeginInfo.clearValueCount = 1;
                renderPassBeginInfo.pClearValues = &clearValues;
                vkCmdBeginRenderPass(graphicsCommandBuffer[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
                //vkCmdSetViewport
                vkCmdSetViewport(graphicsCommandBuffer[i], 0, 1, &viewPort);
                //vkCmdSetScissor
                vkCmdSetScissor(graphicsCommandBuffer[i], 0, 1, &rect2D);
                //vkCmdSetDepthBias
                //vkCmdSetDepthBias(graphicsCommandBuffer[i],depthBiasConstant,0.0f,depthBiasSlope);
                //vkCmdBindPipeline
                vkCmdBindPipeline(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.offscreen);
                //vkCmdBindDescriptorSets
                vkCmdBindDescriptorSets(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.offscreen, 0, 1, &descriptorSets.offscreen, 0, nullptr);
                //vertex
                VkDeviceSize offsets[] = {0};
                vkCmdBindVertexBuffers(graphicsCommandBuffer[i], 0, 1, &scene.verticesBuffer.buffer, offsets);
                //index
                vkCmdBindIndexBuffer(graphicsCommandBuffer[i], scene.indicesBuffer.buffer, offsets[0], VK_INDEX_TYPE_UINT32);
                //draw
                vkCmdDrawIndexed(graphicsCommandBuffer[i], (scene.size), 1, 0, 0, 0);
                //vkCmdEndRenderPass
                //vkCmdDraw
                vkCmdEndRenderPass(graphicsCommandBuffer[i]);
            }
            //scene pass
            {
                VkClearValue clearValues[2];
                clearValues[1].depthStencil.depth = 1.0;
                clearValues[1].depthStencil.stencil = 0.0;
                clearValues[0].color = {0, 0, 0, 1};
                
                VkViewport viewPort = initializers::viewport(swapChainExtent.width, swapChainExtent.height, 0, 1);
                VkRect2D rect2D = initializers::rect2D(swapChainExtent.width, swapChainExtent.height, 0, 0);
                //vkCmdBeginRenderPass
                VkRenderPassBeginInfo renderPassBeginInfo= initializers::renderPassBeginInfo();
                //VkRenderPass           renderPass;
                //VkFramebuffer          framebuffer;
                //VkRect2D               renderArea;
                //uint32_t               clearValueCount;
                //const VkClearValue*    pClearValues;
                renderPassBeginInfo.renderPass = scenePass.renderpass;
                renderPassBeginInfo.framebuffer = scenePass.fbo;
                renderPassBeginInfo.renderArea.extent.width = swapChainExtent.width;
                renderPassBeginInfo.renderArea.extent.height = swapChainExtent.height;
                renderPassBeginInfo.clearValueCount = 2;
                renderPassBeginInfo.pClearValues = clearValues;
                vkCmdBeginRenderPass(graphicsCommandBuffer[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
                //vkCmdSetViewport
                vkCmdSetViewport(graphicsCommandBuffer[i], 0, 1, &viewPort);
                //vkCmdSetScissor
                vkCmdSetScissor(graphicsCommandBuffer[i], 0, 1, &rect2D);
                //vkCmdBindPipeline
                vkCmdBindPipeline(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.sceneShadowPCF);
                //vkCmdBindDescriptorSets
                vkCmdBindDescriptorSets(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.scene, 0, 1, &descriptorSets.scene, 0, nullptr);
                //vertex
                VkDeviceSize offsets[] = {0};
                vkCmdBindVertexBuffers(graphicsCommandBuffer[i], 0, 1, &scene.verticesBuffer.buffer, offsets);
                //index
                vkCmdBindIndexBuffer(graphicsCommandBuffer[i], scene.indicesBuffer.buffer, offsets[0], VK_INDEX_TYPE_UINT32);
                //draw
                vkCmdDrawIndexed(graphicsCommandBuffer[i], (scene.size), 1, 0, 0, 0);
                //vkCmdEndRenderPass
                
                vkCmdEndRenderPass(graphicsCommandBuffer[i]);
            }
            //prelight pass
            {
                VkClearValue clearValues[1];
                clearValues[0].color = {0, 0, 0};
                
                VkViewport viewPort = initializers::viewport(swapChainExtent.width / 2, swapChainExtent.height / 2, 0, 1);
                VkRect2D rect2D = initializers::rect2D(swapChainExtent.width / 2, swapChainExtent.height / 2, 0, 0);
                //vkCmdBeginRenderPass
                VkRenderPassBeginInfo renderPassBeginInfo= initializers::renderPassBeginInfo();
                renderPassBeginInfo.renderPass = preLightPass.renderpass;
                renderPassBeginInfo.framebuffer = preLightPass.fbo;
                renderPassBeginInfo.renderArea.extent.width = swapChainExtent.width / 2;
                renderPassBeginInfo.renderArea.extent.height = swapChainExtent.height / 2;
                renderPassBeginInfo.clearValueCount = 1;
                renderPassBeginInfo.pClearValues = clearValues;
                vkCmdBeginRenderPass(graphicsCommandBuffer[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
                //vkCmdSetViewport
                vkCmdSetViewport(graphicsCommandBuffer[i], 0, 1, &viewPort);
                //vkCmdSetScissor
                vkCmdSetScissor(graphicsCommandBuffer[i], 0, 1, &rect2D);
                //vkCmdBindPipeline
                vkCmdBindPipeline(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.prelight);
                //vkCmdBindDescriptorSets
                vkCmdBindDescriptorSets(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.prelight, 0, 1, &descriptorSets.prelight, 0, nullptr);
                //vertex
                VkDeviceSize offsets[] = {0};
                vkCmdBindVertexBuffers(graphicsCommandBuffer[i], 0, 1, &coneModel.verticesBuffer.buffer, offsets);
                //index
                vkCmdBindIndexBuffer(graphicsCommandBuffer[i], coneModel.indicesBuffer.buffer, offsets[0], VK_INDEX_TYPE_UINT32);
                //draw
                vkCmdDrawIndexed(graphicsCommandBuffer[i], (coneModel.size), 1, 0, 0, 0);
                
                vkCmdEndRenderPass(graphicsCommandBuffer[i]);
            }
            //hblur pass
            {
                VkClearValue clearValues[1];
                clearValues[0].color = {0, 0, 0};
                
                VkViewport viewPort = initializers::viewport(swapChainExtent.width / 2, swapChainExtent.height / 2, 0, 1);
                VkRect2D rect2D = initializers::rect2D(swapChainExtent.width / 2, swapChainExtent.height / 2, 0, 0);
                //vkCmdBeginRenderPass
                VkRenderPassBeginInfo renderPassBeginInfo= initializers::renderPassBeginInfo();
                renderPassBeginInfo.renderPass = blurPass.renderpass;
                renderPassBeginInfo.framebuffer = blurPass.fbo;
                renderPassBeginInfo.renderArea.extent.width = swapChainExtent.width / 2;
                renderPassBeginInfo.renderArea.extent.height = swapChainExtent.height / 2;
                renderPassBeginInfo.clearValueCount = 1;
                renderPassBeginInfo.pClearValues = clearValues;
                vkCmdBeginRenderPass(graphicsCommandBuffer[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
                //vkCmdSetViewport
                vkCmdSetViewport(graphicsCommandBuffer[i], 0, 1, &viewPort);
                //vkCmdSetScissor
                vkCmdSetScissor(graphicsCommandBuffer[i], 0, 1, &rect2D);
                //vkCmdBindPipeline
                vkCmdBindPipeline(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.gaussian);
                //vkCmdBindDescriptorSets
                vkCmdBindDescriptorSets(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.gaussian, 0, 1, &descriptorSets.hblur, 0, nullptr);

                vkCmdDraw(graphicsCommandBuffer[i], 3, 1, 0, 0);
                
                vkCmdEndRenderPass(graphicsCommandBuffer[i]);
            }
            //vblur pass
            {
                VkClearValue clearValues[1];
                clearValues[0].color = {0, 0, 0};
                
                VkViewport viewPort = initializers::viewport(swapChainExtent.width / 2, swapChainExtent.height / 2, 0, 1);
                VkRect2D rect2D = initializers::rect2D(swapChainExtent.width / 2, swapChainExtent.height / 2, 0, 0);
                //vkCmdBeginRenderPass
                VkRenderPassBeginInfo renderPassBeginInfo= initializers::renderPassBeginInfo();
                renderPassBeginInfo.renderPass = blurPass.renderpass;
                renderPassBeginInfo.framebuffer = vblurPassFbo;
                renderPassBeginInfo.renderArea.extent.width = swapChainExtent.width / 2;
                renderPassBeginInfo.renderArea.extent.height = swapChainExtent.height / 2;
                renderPassBeginInfo.clearValueCount = 1;
                renderPassBeginInfo.pClearValues = clearValues;
                vkCmdBeginRenderPass(graphicsCommandBuffer[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
                //vkCmdSetViewport
                vkCmdSetViewport(graphicsCommandBuffer[i], 0, 1, &viewPort);
                //vkCmdSetScissor
                vkCmdSetScissor(graphicsCommandBuffer[i], 0, 1, &rect2D);
                //vkCmdBindPipeline
                vkCmdBindPipeline(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.gaussian);
                //vkCmdBindDescriptorSets
                vkCmdBindDescriptorSets(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.gaussian, 0, 1, &descriptorSets.vblur, 0, nullptr);

                vkCmdDraw(graphicsCommandBuffer[i], 3, 1, 0, 0);
                
                vkCmdEndRenderPass(graphicsCommandBuffer[i]);
            }
            //blit pass
            {
                VkClearValue clearValues[2];
                clearValues[1].depthStencil.depth = 1.0;
                clearValues[1].depthStencil.stencil = 0.0;
                clearValues[0].color = {0, 0, 0};
                
                VkViewport viewPort = initializers::viewport(swapChainExtent.width, swapChainExtent.height, 0, 1);
                VkRect2D rect2D = initializers::rect2D(swapChainExtent.width, swapChainExtent.height, 0, 0);
                //vkCmdBeginRenderPass
                VkRenderPassBeginInfo renderPassBeginInfo= initializers::renderPassBeginInfo();
                renderPassBeginInfo.renderPass = renderPass;
                renderPassBeginInfo.framebuffer = swapChainFramebuffers[i];
                renderPassBeginInfo.renderArea.extent.width = swapChainExtent.width;
                renderPassBeginInfo.renderArea.extent.height = swapChainExtent.height;
                renderPassBeginInfo.clearValueCount = 2;
                renderPassBeginInfo.pClearValues = clearValues;
                vkCmdBeginRenderPass(graphicsCommandBuffer[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
                //vkCmdSetViewport
                vkCmdSetViewport(graphicsCommandBuffer[i], 0, 1, &viewPort);
                //vkCmdSetScissor
                vkCmdSetScissor(graphicsCommandBuffer[i], 0, 1, &rect2D);
                //vkCmdBindPipeline
                vkCmdBindPipeline(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.blit);
                //vkCmdBindDescriptorSets
                vkCmdBindDescriptorSets(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.blit, 0, 1, &descriptorSets.blit, 0, nullptr);

                vkCmdDraw(graphicsCommandBuffer[i], 3, 1, 0, 0);
                
                vkCmdEndRenderPass(graphicsCommandBuffer[i]);
            }

            
            vkEndCommandBuffer(graphicsCommandBuffer[i]);
        }
    }
    
    void VolumetricLight::updateUniformBuffer() {
        // Animate the light source
        lightPos.x = 0.f;
        lightPos.y = 0.f;
        lightPos.z = 1.f;
        
        // Matrix from light's point of view
        glm::mat4 depthProjectionMatrix = glm::perspective(lightFOV, 1.0f, 0.1f, 3.0f);
        //depthProjectionMatrix = glm::ortho(-2.0f, 2.f, -2.f, 2.f, 0.1f, 2.f);
        depthProjectionMatrix[1][1] *= -1;
        
        glm::vec3 center = {float(cos(timer)) * 1.2 + lightPos.x, lightPos.y - 0.2, 0};
        glm::mat4 depthViewMatrix = glm::lookAt(lightPos, center, glm::vec3(0,1,0));
        
        glm::mat4 depthModelMatrix = glm::mat4(1.0f);
        uniformDataOffscreen.depthMVP = depthProjectionMatrix * depthViewMatrix * depthModelMatrix;
        memcpy(uniformBuffers.offscreen.bufferMapped, &uniformDataOffscreen, sizeof(uniformDataOffscreen));
        
        float cameraZnear = 0.1f;
        float cameraZfar = 20.0;
        glm::vec3 worldCameraPos = glm::vec3(1.0f, 3.0f, 1.0f);
        glm::vec3 cameraObject = glm::vec3(0.0f, 0.0f, 0.0f);
        uniformDataScene.projection = glm::perspective(glm::radians(60.0f), swapChainExtent.width / (float) swapChainExtent.height, cameraZnear, cameraZfar);
        uniformDataScene.projection[1][1] *= -1;
        uniformDataScene.view = glm::lookAt(worldCameraPos, cameraObject, glm::vec3(0.0f, 0.0f, 1.0f));
        
        uniformDataScene.model = glm::mat4(1.0f);
        uniformDataScene.lightPos = glm::vec4(lightPos, 1.0f);
        glm::vec3 lightDir = glm::normalize(center - lightPos);
        uniformDataScene.lightParam = glm::vec4(lightDir, cos(glm::radians(lightFOV / 2.0)));
        uniformDataScene.cutoff = {cos(glm::radians(lightFOV / 2.0)), cos(glm::radians(lightOuterFOV / 2.0))};
        uniformDataScene.depthBiasMVP = uniformDataOffscreen.depthMVP;
        memcpy(uniformBuffers.scene.bufferMapped, &uniformDataScene, sizeof(uniformDataScene));
        
        /*
         glm::mat4 projection;
         glm::mat4 view;
         glm::mat4 model;
         glm::vec4 lightPos;
         glm::vec3 lightDir;
         */
        float lightrange = 1.5;
        float angleScale = tan(glm::radians((lightFOV + 1) * 0.5f)) * lightrange;
        
        uniformDataPrelight.projection = uniformDataScene.projection;
        uniformDataPrelight.view = uniformDataScene.view;
        
        glm::mat4 scale = glm::mat4(glm::vec4(angleScale,0,0,0), glm::vec4(0,angleScale,0,0), glm::vec4(0,0,lightrange,0), glm::vec4(0,0,0,1));
        
        float angle = acos(lightDir.z);
        glm::vec3 axis = glm::vec3(0, 1, 0);
        if(abs(lightDir.z) <  0.9999)
            axis = glm::cross(glm::vec3(0, 0, 1), lightDir);
        uniformDataPrelight.model = glm::translate(glm::mat4(1.0), lightPos) * glm::rotate(glm::mat4(1.0), angle, axis) * scale;
        uniformDataPrelight.lightdir = lightDir;
        uniformDataPrelight.lightPos = glm::vec4(lightPos, 1.0/(lightrange * lightrange));
        memcpy(uniformBuffers.prelight.bufferMapped, &uniformDataPrelight, sizeof(uniformDataPrelight));
        
        
        /*
         struct UniformDataPrelightFrag
         {
            mat4 lightSpace;
             // x = 1-far/near
                // y = far/near
                // z = x/far
                // w = y/far
             glm::vec4 ZBufferParams;
             glm::vec4 mieG;
             glm::vec3 lightColor;
             glm::vec3 worldSpaceCameraPos;
             glm::vec3 cameraForward;
             // x: scattering coef, y: extinction coef
             glm::vec2 volumetricLight;
             //cos(35), cos(45)
             glm::vec2 cutoff;
             float planeD;
         } uniformDataPrelightFrag;
         */
        uniformDataPrelightFrag.lightSpace = uniformDataOffscreen.depthMVP;
        uniformDataPrelightFrag.ZBufferParams = {
            1.0 - cameraZfar/cameraZnear,
            cameraZfar / cameraZnear,
            1/cameraZfar - 1/cameraZnear,
            1/cameraZnear};
        
        float MieG = 0.5;
        uniformDataPrelightFrag.mieG = {
            1 - (MieG * MieG), 1 + (MieG * MieG), 2 * MieG, 1.0f / (4.0f * glm::pi<float>())
        };
        uniformDataPrelightFrag.lightColor = {1, 1, 1, 1};
        uniformDataPrelightFrag.worldSpaceCameraPos = glm::vec4(worldCameraPos, 1.0);
        uniformDataPrelightFrag.cameraForward = {glm::normalize(cameraObject - worldCameraPos), 1.0};
        
        glm::vec2 cutoff = {cos(glm::radians(lightFOV / 2.0 - 5.0)), cos(glm::radians(lightFOV / 2.0))};
        uniformDataPrelightFrag.volumetricLight = {400, 0.5, cutoff};
        glm::vec3 planeCenter = glm::vec3(lightPos) + lightDir * lightrange;
        //dot((x, y ,z) - center, axis) = 0
        float d = -glm::dot(planeCenter, lightDir);
        uniformDataPrelightFrag.params = {d, 0, 0, 0};
        float NoiseScale = 0.8f;
        float NoiseIntensity = 1.0f;
        float NoiseIntensityOffset = 0.3f;
        uniformDataPrelightFrag._NoiseData = glm::vec4(NoiseScale, NoiseIntensity, NoiseIntensityOffset, timer);
        uniformDataPrelightFrag._NoiseVelocity = glm::vec4(0.01, 0.02, 0.0, 0.0);
        memcpy(uniformBuffers.prelightFrag.bufferMapped, &uniformDataPrelightFrag, sizeof(uniformDataPrelightFrag));
        
        uniformDataGaussian.ZBufferParams = uniformDataPrelightFrag.ZBufferParams;
        uniformDataGaussian.direction = glm::vec2(1, 0);
        uniformDataGaussian.texSize = glm::vec2(swapChainExtent.width / 2, swapChainExtent.height / 2);
        memcpy(uniformBuffers.hblur.bufferMapped, &uniformDataGaussian, sizeof(uniformDataGaussian));
        
        uniformDataGaussian.direction = glm::vec2(0, 1);
        memcpy(uniformBuffers.vblur.bufferMapped, &uniformDataGaussian, sizeof(UniformDataGaussian));
    }
    
    void VolumetricLight::drawFrame() {
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        
        
        //signal imageAvailableSemaphore
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            recordCommandBuffer();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        
        vkResetFences(device, 1, &inFlightFence);
        
        VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        std::vector<VkSemaphore> graphicsWaitSemaphores = { imageAvailableSemaphore };
        std::vector<VkSemaphore> graphicsSignalSemaphores = { renderFinishedSemaphore };

        // Submit graphics commands
        VkSubmitInfo graphicsSubmitInfo = initializers::submitInfo();
        graphicsSubmitInfo.commandBufferCount = 1;
        graphicsSubmitInfo.pCommandBuffers = &graphicsCommandBuffer[imageIndex];
        graphicsSubmitInfo.waitSemaphoreCount = graphicsWaitSemaphores.size();
        graphicsSubmitInfo.pWaitSemaphores = graphicsWaitSemaphores.data();
        //pWaitSemaphores signal 取得之前不执行 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        graphicsSubmitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
        graphicsSubmitInfo.signalSemaphoreCount = graphicsSignalSemaphores.size();
        graphicsSubmitInfo.pSignalSemaphores = graphicsSignalSemaphores.data();
        VK_CHECK_RESULT(vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, inFlightFence));

        
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
        
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        
        presentInfo.pImageIndices = &imageIndex;
        
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            //recreateSwapChain();
            //recordCommandBuffer();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        
        
    }


}




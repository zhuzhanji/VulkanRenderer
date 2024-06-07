//
//  CSParticlesMorpher.cpp
//  VulkanRenderer
//
//

#include "base/ApplicationBase.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "base/tiny_obj_loader.h"

namespace vks{

struct UniformBufferObjectVert {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct UniformBufferObjectComp {
    float time;
    int particle_count;
};

struct ParticleSource {
    glm::vec4 positionSrc;
    glm::vec4 positionDst;
    glm::vec4 color;
    glm::vec4 parameter;
};

struct ParticleCurrent {
    glm::vec4 position;
    glm::vec4 color;
};

struct Vertex {
    glm::vec4 position;
    glm::vec2 uv;
};

struct Stone {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
};

static std::array<VkVertexInputBindingDescription, 2> getBindingDescription() {
    VkVertexInputBindingDescription bindingDescriptionVert{};
    bindingDescriptionVert.binding = 0;
    bindingDescriptionVert.stride = sizeof(Vertex);
    bindingDescriptionVert.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    
    VkVertexInputBindingDescription bindingDescriptionInstance{};
    bindingDescriptionInstance.binding = 1;
    bindingDescriptionInstance.stride = sizeof(ParticleCurrent);
    bindingDescriptionInstance.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    
    return {bindingDescriptionVert, bindingDescriptionInstance};
}

static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
    
    
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, uv);
    
    attributeDescriptions[2].binding = 1;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(ParticleCurrent, position);
    attributeDescriptions[3].binding = 1;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(ParticleCurrent, color);
    
    return attributeDescriptions;
}


class DancingParticles : public ApplicationBase {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
    
private:
    
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;
    
    
    VkBuffer shaderStorageBufferSrc;
    VkDeviceMemory shaderStorageBufferMemorySrc;
    
    VkBuffer shaderStorageBufferDst;
    VkDeviceMemory shaderStorageBufferMemoryDst;
    
    VkBuffer uniformBufferComp;
    VkDeviceMemory uniformBufferMemoryComp;
    void* uniformBufferMappedComp;
    
    VkBuffer uniformBufferVert;
    VkDeviceMemory uniformBufferMemoryVert;
    void* uniformBufferMappedVert;
    
    
    VkDescriptorSet descriptorSet;
    VkDescriptorSet computeDescriptorSet;
    
    VkCommandBuffer computeCommandBuffer;
    
    std::string MODEL_PATH_1 = "models/bunny.obj";
    std::string MODEL_PATH_2 = "models/teapot.obj";
    std::string MODEL_PATH_3 = "models/rock.obj";
    
    int particle_count;
    
    Stone stone;
    
    void initVulkan() {
        ApplicationBase::initVulkan();
 
        loadModel();
        createShaderStorageBuffers();
        createUniformBuffers();
        
        createDescriptorPool();
        createDescriptorSetLayout();
        createDescriptorSets();
        createGraphicsPipeline();
        recordCommandBuffer();
        
        createComputeDescriptorSetLayout();
        createComputeDescriptorSets();
        createComputePipeline();
        createComputeCommandBuffer();
        recordComputeCommandBuffer();
        
    }
    
    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            auto tStart = std::chrono::high_resolution_clock::now();
            
            glfwPollEvents();
            drawFrame();
            
            auto tEnd = std::chrono::high_resolution_clock::now();
            auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            lastFrameTime = tDiff / 1000.0f;
        }
        
        vkDeviceWaitIdle(device);
    }
    
    void cleanup() {
        
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        
        vkDestroyPipeline(device, computePipeline, nullptr);
        vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);
        
        vkDestroyBuffer(device, uniformBufferComp, nullptr);
        vkFreeMemory(device, uniformBufferMemoryComp, nullptr);
        vkDestroyBuffer(device, uniformBufferVert, nullptr);
        vkFreeMemory(device, uniformBufferMemoryVert, nullptr);
        
        
        vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        
        vkDestroyBuffer(device, shaderStorageBufferDst, nullptr);
        vkFreeMemory(device, shaderStorageBufferMemoryDst, nullptr);
        
        vkDestroyBuffer(device, stone.indexBuffer, nullptr);
        vkFreeMemory(device, stone.indexBufferMemory, nullptr);
        
        vkDestroyBuffer(device, stone.vertexBuffer, nullptr);
        vkFreeMemory(device, stone.vertexBufferMemory, nullptr);
        
        vkDestroyBuffer(device, shaderStorageBufferSrc, nullptr);
        vkFreeMemory(device, shaderStorageBufferMemorySrc, nullptr);
        
        
        ApplicationBase::cleanup();
    }
    
    void createComputeDescriptorSetLayout() {
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings =
        {
            initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT),
            initializers::descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT),
            initializers::descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        };
        
        
        VkDescriptorSetLayoutCreateInfo layoutInfo = initializers::descriptorSetLayoutCreateInfo(layoutBindings);
        
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) );
    }
    
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        std::array<VkDescriptorSetLayoutBinding, 1> bindings = {uboLayoutBinding};
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }
    
    void createGraphicsPipeline() {
        
        VkPipelineShaderStageCreateInfo shaderStages[] = {
            loadShader("shaders/particle_morph_vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader("shaders/dancing_particle_frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT),
        };
        
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        
        auto bindingDescription = getBindingDescription();
        auto attributeDescriptions = getAttributeDescriptions();
        
        vertexInputInfo.vertexBindingDescriptionCount = bindingDescription.size();
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = bindingDescription.data();
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
        
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
        
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;
        
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;
        
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
        
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
        
        vkDestroyShaderModule(device, shaderStages[0].module, nullptr);
        vkDestroyShaderModule(device, shaderStages[1].module, nullptr);
    }
    
    void createComputePipeline() {
        auto computeShaderCode = readFile("shaders/particle_morph_comp.spv");
        
        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);
        
        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
        
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }
        
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;
        
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }
        
        vkDestroyShaderModule(device, computeShaderModule, nullptr);
    }
    
    
    void loadModel(){
        
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH_3.c_str())) {
            throw std::runtime_error(warn + err);
        }
        
        std::unordered_map<glm::vec3, uint32_t> uniqueVertices{};
        
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};
                vertex.position = {
                    attrib.vertices[3 * index.vertex_index + 0] * 0.005,
                    attrib.vertices[3 * index.vertex_index + 1] * 0.005,
                    attrib.vertices[3 * index.vertex_index + 2] * 0.005, 1.0
                };
                vertex.uv = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };
                
                if (uniqueVertices.count(vertex.position) == 0) {
                    uniqueVertices[vertex.position] = static_cast<uint32_t>(stone.vertices.size());
                    stone.vertices.push_back(vertex);
                }
                stone.indices.push_back(uniqueVertices[vertex.position]);
            }
        }
        //create vertex buffer
        {
            VkDeviceSize bufferSize = sizeof(stone.vertices[0]) * stone.vertices.size();
            
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, stone.vertices.data(), (size_t) bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, stone.vertexBuffer, stone.vertexBufferMemory);
            copyBuffer(stagingBuffer, stone.vertexBuffer, bufferSize);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }
        //create index buffer
        {
            VkDeviceSize bufferSize = sizeof(stone.indices[0]) * stone. indices.size();
            
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            
            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, stone.indices.data(), (size_t) bufferSize);
            vkUnmapMemory(device, stagingBufferMemory);
            
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, stone.indexBuffer, stone.indexBufferMemory);
            
            copyBuffer(stagingBuffer, stone.indexBuffer, bufferSize);
            
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }
    }
    
    void createShaderStorageBuffers() {
        std::vector<ParticleSource> particles;
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;
        
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH_1.c_str())) {
            throw std::runtime_error(warn + err);
        }
        
        std::default_random_engine rndEngine((unsigned)time(nullptr));
        std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
        std::unordered_map<int, bool> uniqueVertices;
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                ParticleSource p;
                if(uniqueVertices.find(index.vertex_index) != uniqueVertices.end())
                    continue;
                uniqueVertices[index.vertex_index] = true;
                
                p.positionSrc = glm::vec4(attrib.vertices[3 * index.vertex_index + 0] ,
                                          attrib.vertices[3 * index.vertex_index + 1] ,
                                          attrib.vertices[3 * index.vertex_index + 2], 1.0);
                glm::vec3 normal = glm::vec4(attrib.normals[3 * index.vertex_index + 0] ,
                                             attrib.normals[3 * index.vertex_index + 1] ,
                                             attrib.normals[3 * index.vertex_index + 2], 0.0);
                
                if(glm::dot(normal, normal) > 0.01)
                    particles.push_back(p);
            }
        }
        //to add dst
        std::cout<<"particle number " << particles.size() << std::endl;
        auto particle_count_1 = particles.size();
        
        shapes.clear();
        materials.clear();
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH_2.c_str())) {
            throw std::runtime_error(warn + err);
        }
        uniqueVertices.clear();
        uint count = 0;
        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                
                if(uniqueVertices.find(index.vertex_index) != uniqueVertices.end())
                    continue;
                uniqueVertices[index.vertex_index] = true;
                
                glm::vec4 position = glm::vec4(attrib.vertices[3 * index.vertex_index + 0] * 0.5,
                                               attrib.vertices[3 * index.vertex_index + 1] * 0.5,
                                               attrib.vertices[3 * index.vertex_index + 2] * 0.5, 1.0);
                glm::vec4 normal = glm::vec4(attrib.normals[3 * index.vertex_index + 0] ,
                                             attrib.normals[3 * index.vertex_index + 1] ,
                                             attrib.normals[3 * index.vertex_index + 2], 0.0);
                
                if(glm::dot(normal, normal) < 0.01)
                {
                    continue;
                }
                if(count < particle_count_1)
                {
                    particles[count].positionDst = position;
                }
                else{
                    ParticleSource p;
                    p.positionSrc = glm::vec4(0,0,0,1);
                    p.positionDst = position;
                }
                count ++;
            }
        }
        std::cout<<"particle count " << count << std::endl;
        
        particle_count = particles.size();
        for(int i = 0; i < particle_count; i++)
        {
            particles[i].color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0);
            particles[i].parameter = glm::vec4(0, rndDist(rndEngine) * 0.02, 1.5, 0);
        }
        
        for(int i = count; i < particle_count; i++){
            particles[i].positionDst = glm::vec4(0,0,0,1);
        }
        
        
        {
            VkDeviceSize bufferSizeSrc = sizeof(ParticleSource) * particle_count;
            
            // Create a staging buffer used to upload data to the gpu
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(bufferSizeSrc, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
            
            void* data;
            vkMapMemory(device, stagingBufferMemory, 0, bufferSizeSrc, 0, &data);
            memcpy(data, particles.data(), (size_t)bufferSizeSrc);
            vkUnmapMemory(device, stagingBufferMemory);
            
            
            // Copy initial particle data to all storage buffers
            createBuffer(bufferSizeSrc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shaderStorageBufferSrc, shaderStorageBufferMemorySrc);
            copyBuffer(stagingBuffer, shaderStorageBufferSrc, bufferSizeSrc);
            
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingBufferMemory, nullptr);
        }
        {
            
            VkDeviceSize bufferSizeDst = sizeof(ParticleCurrent) * particle_count;
            
            // Copy initial particle data to all storage buffers
            createBuffer(bufferSizeDst, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shaderStorageBufferDst, shaderStorageBufferMemoryDst);
        }
        
    }
    
    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObjectVert);
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBufferVert, uniformBufferMemoryVert);
        
        vkMapMemory(device, uniformBufferMemoryVert, 0, bufferSize, 0, &uniformBufferMappedVert);
        
        bufferSize = sizeof(UniformBufferObjectComp);
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBufferComp, uniformBufferMemoryComp);
        
        vkMapMemory(device, uniformBufferMemoryComp, 0, bufferSize, 0, &uniformBufferMappedComp);
    }
    
    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 2;
        
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 2;
        
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 2;
        
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }
    
    void createComputeDescriptorSets() {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &computeDescriptorSetLayout;
        
        if (vkAllocateDescriptorSets(device, &allocInfo, &computeDescriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
        
        VkDescriptorBufferInfo uniformBufferInfo{};
        uniformBufferInfo.buffer = uniformBufferComp;
        uniformBufferInfo.offset = 0;
        uniformBufferInfo.range = sizeof(UniformBufferObjectComp);
        std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = computeDescriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &uniformBufferInfo;
        
        VkDescriptorBufferInfo storageBufferInfoSource{};
        storageBufferInfoSource.buffer = shaderStorageBufferSrc;
        storageBufferInfoSource.offset = 0;
        storageBufferInfoSource.range = sizeof(ParticleSource) * particle_count;
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeDescriptorSet;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &storageBufferInfoSource;
        
        VkDescriptorBufferInfo storageBufferInfoCurrentFrame{};
        storageBufferInfoCurrentFrame.buffer = shaderStorageBufferDst;
        storageBufferInfoCurrentFrame.offset = 0;
        storageBufferInfoCurrentFrame.range = sizeof(ParticleCurrent) * particle_count;
        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = computeDescriptorSet;
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &storageBufferInfoCurrentFrame;
        vkUpdateDescriptorSets(device, 3, descriptorWrites.data(), 0, nullptr);
    }
    
    void createDescriptorSets() {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;
        
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
        
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBufferVert;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObjectVert);
        std::array<VkWriteDescriptorSet, 1> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
    
    
    void createComputeCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        
        if (vkAllocateCommandBuffers(device, &allocInfo, &computeCommandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate compute command buffers!");
        }
    }
    
    void recordCommandBuffer() {
        for(int imageIndex = 0; imageIndex < swapChainImages.size(); imageIndex++){
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            
            if (vkBeginCommandBuffer(graphicsCommandBuffer[imageIndex], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }
            
            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapChainExtent;
            
            std::array<VkClearValue, 2> clearValues{};
            clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            clearValues[1].depthStencil = {1.0f, 0};
            
            renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
            renderPassInfo.pClearValues = clearValues.data();
            
            
            vkCmdBeginRenderPass(graphicsCommandBuffer[imageIndex], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            
            vkCmdBindPipeline(graphicsCommandBuffer[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float) swapChainExtent.width;
            viewport.height = (float) swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(graphicsCommandBuffer[imageIndex], 0, 1, &viewport);
            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(graphicsCommandBuffer[imageIndex], 0, 1, &scissor);
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(graphicsCommandBuffer[imageIndex], 0, 1, &stone.vertexBuffer, offsets);
            
            vkCmdBindVertexBuffers(graphicsCommandBuffer[imageIndex], 1, 1, &shaderStorageBufferDst, offsets);
            vkCmdBindIndexBuffer(graphicsCommandBuffer[imageIndex], stone.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(graphicsCommandBuffer[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
            
            
            vkCmdDrawIndexed(graphicsCommandBuffer[imageIndex], uint32_t(stone.indices.size()), particle_count, 0, 0, 0);
            
            vkCmdEndRenderPass(graphicsCommandBuffer[imageIndex]);
            
            if (vkEndCommandBuffer(graphicsCommandBuffer[imageIndex]) != VK_SUCCESS) {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }
    
    void recordComputeCommandBuffer() {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        if (vkBeginCommandBuffer(computeCommandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording compute command buffer!");
        }
        
        vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        
        vkCmdBindDescriptorSets(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSet, 0, nullptr);
        
        vkCmdDispatch(computeCommandBuffer, ceil(particle_count / 256.0), 1, 1);
        
        if (vkEndCommandBuffer(computeCommandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record compute command buffer!");
        }
        
    }
    
    
    void updateUniformBuffer() {
        static auto startTime = std::chrono::high_resolution_clock::now();
        
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        
        UniformBufferObjectVert ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        ubo.model = glm::rotate(ubo.model, time * glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        
        ubo.view = glm::lookAt(glm::vec3(4.0f, 4.0f, 4.0f), glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        memcpy(uniformBufferMappedVert, &ubo, sizeof(ubo));
        
        UniformBufferObjectComp uboComp{};
        uboComp.time = time;
        uboComp.particle_count = particle_count;
        memcpy(uniformBufferMappedComp, &uboComp, sizeof(uboComp));
    }
    
    void drawFrame() {
        
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        
        updateUniformBuffer();
        
        VkSubmitInfo computeSubmitInfo{};
        computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        
        // Wait for rendering finished
        VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &computeCommandBuffer;
        //computeSubmitInfo.waitSemaphoreCount = 1;
        //computeSubmitInfo.pWaitSemaphores = &renderFinishedSemaphore;
        computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
        computeSubmitInfo.signalSemaphoreCount = 1;
        computeSubmitInfo.pSignalSemaphores = &computeFinishedSemaphore;
        
        //vkQueueWaitIdle(computeQueue);
        
        if (vkQueueSubmit(computeQueue, 1, &computeSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit compute command buffer!");
        };
        
        // Graphics submission
        
        uint32_t imageIndex;
        //signal imageAvailableSemaphore
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
        
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            recordComputeCommandBuffer();
            recordCommandBuffer();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        
        vkResetFences(device, 1, &inFlightFence);
        
        VkSemaphore waitSemaphores[] = { computeFinishedSemaphore, imageAvailableSemaphore };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        
        VkSubmitInfo graphicsSubmitInfo{};
        graphicsSubmitInfo = {};
        graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        
        graphicsSubmitInfo.waitSemaphoreCount = 2;
        graphicsSubmitInfo.pWaitSemaphores = waitSemaphores;
        graphicsSubmitInfo.pWaitDstStageMask = waitStages;
        graphicsSubmitInfo.commandBufferCount = 1;
        graphicsSubmitInfo.pCommandBuffers = &graphicsCommandBuffer[imageIndex];
        graphicsSubmitInfo.signalSemaphoreCount = 1;
        graphicsSubmitInfo.pSignalSemaphores = &renderFinishedSemaphore;
        
        if (vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        
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
            recreateSwapChain();
            recordComputeCommandBuffer();
            recordCommandBuffer();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        
        //vkQueueWaitIdle(graphicsQueue);
        //vkQueueWaitIdle(presentQueue);
        //vkQueueWaitIdle(computeQueue);
        
    }
    
};

 
}

/*
int main() {
    vks::DancingParticles app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

*/

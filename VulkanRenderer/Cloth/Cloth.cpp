//
//  Cloth.cpp
//  VulkanRenderer
//
//

#include "../base/ApplicationBase.hpp"

namespace vks {


class ComputeCloth : public ApplicationBase {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
    
private:
    uint32_t indexCount{ 0 };
    bool simulateWind{ false };
    
    Texture textureCloth;
    
    // Particle Definition
    struct Particle {
        glm::vec4 pos;
        glm::vec4 vel;
        glm::vec4 uv;
        glm::vec4 normal;
    };
    
    // Cloth definition parameters
    struct Cloth {
        glm::uvec2 gridsize{ 60, 60 };
        glm::vec2 size{ 5.0f, 5.0f };
    } cloth;

    struct StorageBuffers {
            vks::Buffer input;
            vks::Buffer output;
    } storageBuffers;
    
    uint32_t numParticles{ 0 };
    
    
    // Resources for the graphics part of the example
    struct Graphics {
        VkDescriptorSetLayout descriptorSetLayout;    // Particle system rendering shader binding layout
        VkDescriptorSet descriptorSet;                // Particle system rendering shader bindings
        VkPipelineLayout pipelineLayout;            // Layout of the graphics pipeline
        struct Pipelines {
            VkPipeline cloth{ VK_NULL_HANDLE };
            VkPipeline sphere{ VK_NULL_HANDLE };
        } pipelines;
        VkSemaphore semaphore;                      // Execution dependency between compute & graphic submission
        Buffer indices;
        
        struct UniformData {
            glm::mat4 projection;
            glm::mat4 view;
            glm::vec2 screenDim;
        } uniformData;
        
        Buffer uniformBuffer;
        
        void destroy(VkDevice device){
            uniformBuffer.destroy((device));
            indices.destroy(device);
            
            vkDestroyPipeline(device, pipelines.cloth, nullptr);
            vkDestroyPipeline(device, pipelines.sphere, nullptr);
            
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            vkDestroySemaphore(device, semaphore, nullptr);
        }
    } graphics;
    
    // Resources for the compute part of the example
    struct Compute {
        VkCommandPool commandPool;                    // Use a separate command pool (queue family may differ fromthe one used for graphics)
        VkCommandBuffer commandBuffer;                // Command buffer storing the dispatch commands and barriers
        VkSemaphore semaphore;
        VkDescriptorSetLayout descriptorSetLayout;    // Compute shader binding layout
        std::array<VkDescriptorSet, 2> descriptorSets{ VK_NULL_HANDLE };                // Compute shader bindings
        VkPipelineLayout pipelineLayout;            // Layout of the compute pipeline
        VkPipeline pipeline;
        struct UniformData {                        // Compute shader uniform block object
            float deltaT{ 0.0f };                    // Frame delta time
            float particleMass{ 0.1f };
            float springStiffness{ 2000.0f };
            float damping{ 0.25f };
            float restDistH{ 0 };
            float restDistV{ 0 };
            float restDistD{ 0 };
            float sphereRadius{ 1.0f };
            glm::vec4 spherePos{ 0.0f, 0.0f, 0.0f, 0.0f };
            glm::vec4 gravity{ 0.0f, 9.8f, 0.0f, 0.0f };
            glm::ivec2 particleCount{ 0 };
        } uniformData;
        
        Buffer uniformBuffer;
        
        void destroy(VkDevice device){
            uniformBuffer.destroy((device));
            
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            vkDestroySemaphore(device, semaphore, nullptr);
            vkDestroyCommandPool(device, commandPool, nullptr);
        }
    } compute;

    
public:
    void initVulkan() {
        ApplicationBase::initVulkan();
        
        loadAssets();
        createShaderStorageBuffers();
        
        prepareGraphics();
        prepareCompute();

    }
    
    void mainLoop() {
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
    
    void cleanup() {

        if (device) {
            // Graphics
            graphics.destroy(device);
            // Compute
            compute.destroy(device);
            
            storageBuffers.input.destroy(device);
            storageBuffers.output.destroy(device);
            
            textureCloth.destroy(device);
        }
        
        ApplicationBase::cleanup();
    }
    
    
    void prepareGraphics(){
        // Uniform buffer for passing data to the vertex shader
        // Vertex shader uniform buffer block
        VkDeviceSize bufferSize = sizeof(Graphics::uniformData);
        graphics.uniformBuffer = createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(device, graphics.uniformBuffer.bufferMemory, 0, bufferSize, 0, &graphics.uniformBuffer.bufferMapped);
        
        // Descriptor pool
        std::vector<VkDescriptorPoolSize> poolSizes = {
            initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3),
            initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4),
            initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2)
        };
        VkDescriptorPoolCreateInfo descriptorPoolInfo = initializers::descriptorPoolCreateInfo(poolSizes, 3);
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

        // Descriptor layout
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT),
            initializers::descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
        };
        VkDescriptorSetLayoutCreateInfo descriptorLayout = initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
            VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &graphics.descriptorSetLayout));

        // Decscriptor set
        VkDescriptorSetAllocateInfo allocInfo = initializers::descriptorSetAllocateInfo(descriptorPool, 1, &graphics.descriptorSetLayout);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSet));
        std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
            initializers::writeDescriptorSet(graphics.descriptorSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &graphics.uniformBuffer.bufferInfo),
            initializers::writeDescriptorSet(graphics.descriptorSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &textureCloth.imageInfo, nullptr)
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);

        // Layout
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(1, &graphics.descriptorSetLayout);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &graphics.pipelineLayout));

        // Pipeline
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, 0, VK_TRUE);
        VkPipelineRasterizationStateCreateInfo rasterizationState = initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
        VkPipelineColorBlendAttachmentState blendAttachmentState = initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo colorBlendState = initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
        VkPipelineDepthStencilStateCreateInfo depthStencilState = initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
        VkPipelineViewportStateCreateInfo viewportState = initializers::pipelineViewportStateCreateInfo(1, 1, 0);
        VkPipelineMultisampleStateCreateInfo multisampleState = initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

        // Rendering pipeline
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
            loadShader("shaders/computecloth/cloth.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader("shaders/computecloth/cloth.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
        };

        VkGraphicsPipelineCreateInfo pipelineCreateInfo = initializers::pipelineCreateInfo(graphics.pipelineLayout, renderPass);

        // Vertex Input
        std::vector<VkVertexInputBindingDescription> inputBindings = {
            initializers::vertexInputBindingDescription(0, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX)
        };
        // Attribute descriptions based on the particles of the cloth
        std::vector<VkVertexInputAttributeDescription> inputAttributes = {
            initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Particle, pos)),
            initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Particle, uv)),
            initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Particle, normal))
        };

        // Assign to vertex buffer
        VkPipelineVertexInputStateCreateInfo inputState = initializers::pipelineVertexInputStateCreateInfo();
        inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(inputBindings.size());
        inputState.pVertexBindingDescriptions = inputBindings.data();
        inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
        inputState.pVertexAttributeDescriptions = inputAttributes.data();

        pipelineCreateInfo.pVertexInputState = &inputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();
        pipelineCreateInfo.renderPass = renderPass;
        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &graphics.pipelines.cloth));

        // Sphere rendering pipeline
        //    pipelineCreateInfo.pVertexInputState = vkglTF::Vertex::getPipelineVertexInputState({ //vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV, vkglTF::VertexComponent::Normal });
        //    inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
        //    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        //    inputAssemblyState.primitiveRestartEnable = VK_FALSE;
        //    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
        //    shaderStages[0] = loadShader(getShadersPath() + "computecloth/sphere.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        //    shaderStages[1] = loadShader(getShadersPath() + "computecloth/sphere.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
        //    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, //&graphics.pipelines.sphere));
                     
        // We use a semaphore to synchronize compute and graphics
        VkSemaphoreCreateInfo semaphoreCreateInfo = initializers::semaphoreCreateInfo();
        VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &graphics.semaphore));
        
        recordCommandBuffer();

        
    }
    
    void prepareCompute(){
        compute.uniformBuffer = createBuffer(sizeof(Compute::UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, compute.uniformBuffer.bufferMemory, 0, sizeof(Compute::UniformData), 0, &compute.uniformBuffer.bufferMapped);
        
        // Set some initial values
        float dx = cloth.size.x / (cloth.gridsize.x - 1);
        float dy = cloth.size.y / (cloth.gridsize.y - 1);

        compute.uniformData.restDistH = dx;
        compute.uniformData.restDistV = dy;
        compute.uniformData.restDistD = sqrtf(dx * dx + dy * dy);
        compute.uniformData.particleCount = cloth.gridsize;

        // Create compute pipeline
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT),
            initializers::descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT),
            initializers::descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT)
        };

        VkDescriptorSetLayoutCreateInfo descriptorLayout = initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = initializers::pipelineLayoutCreateInfo(1, &compute.descriptorSetLayout);
        // Push constants used to pass some parameters
        VkPushConstantRange pushConstantRange = initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t));
        pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));

        VkDescriptorSetAllocateInfo allocInfo = initializers::descriptorSetAllocateInfo(descriptorPool, 1, &compute.descriptorSetLayout);

        // Create two descriptor sets with input and output buffers switched
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSets[0]));
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSets[1]));

        std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
            initializers::writeDescriptorSet(compute.descriptorSets[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &storageBuffers.input.bufferInfo),
            initializers::writeDescriptorSet(compute.descriptorSets[0], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &storageBuffers.output.bufferInfo),
            initializers::writeDescriptorSet(compute.descriptorSets[0], 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &compute.uniformBuffer.bufferInfo),

            initializers::writeDescriptorSet(compute.descriptorSets[1], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &storageBuffers.output.bufferInfo),
            initializers::writeDescriptorSet(compute.descriptorSets[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &storageBuffers.input.bufferInfo),
            initializers::writeDescriptorSet(compute.descriptorSets[1], 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &compute.uniformBuffer.bufferInfo)
        };

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

        // Create pipeline
        VkComputePipelineCreateInfo computePipelineCreateInfo = initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
        computePipelineCreateInfo.stage = loadShader("shaders/computecloth/cloth.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
        VK_CHECK_RESULT(vkCreateComputePipelines(device, nullptr, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline));

        // Separate command pool as queue family for compute may be different than graphics
        VkCommandPoolCreateInfo cmdPoolInfo = {};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

        // Create a command buffer for compute operations

        VkCommandBufferAllocateInfo cmdBufAllocateInfo{};
        cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdBufAllocateInfo.commandPool = compute.commandPool;
        cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdBufAllocateInfo.commandBufferCount = 1;
        
        // Create a command buffer for compute operations
        createCommandBuffers(compute.commandPool, 1, &compute.commandBuffer);
        
        // We use a semaphore to synchronize compute and graphics
        VkSemaphoreCreateInfo semaphoreCreateInfo = initializers::semaphoreCreateInfo();
        VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore));
        
        // Build a single command buffer containing the compute dispatch commands
        recordComputeCommandBuffer();
        
    }
       
    
    void loadAssets(){
        textureCloth = createTextureImageFromFile("textures/cotton_d.tga", VK_FORMAT_R8G8B8A8_UNORM);
    }
    
    void addComputeToComputeBarriers(VkCommandBuffer commandBuffer)
    {
        VkBufferMemoryBarrier bufferBarrier = initializers::bufferMemoryBarrier();
        bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufferBarrier.size = VK_WHOLE_SIZE;
        std::vector<VkBufferMemoryBarrier> bufferBarriers;
        bufferBarrier.buffer = storageBuffers.input.buffer;
        bufferBarriers.push_back(bufferBarrier);
        bufferBarrier.buffer = storageBuffers.output.buffer;
        bufferBarriers.push_back(bufferBarrier);
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE,
            0, nullptr,
            static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
            0, nullptr);
    }

    void createShaderStorageBuffers() {
        std::vector<Particle> particleBuffer(cloth.gridsize.x * cloth.gridsize.y);

        float dx = cloth.size.x / (cloth.gridsize.x - 1);
        float dy = cloth.size.y / (cloth.gridsize.y - 1);
        float du = 1.0f / (cloth.gridsize.x - 1);
        float dv = 1.0f / (cloth.gridsize.y - 1);

        // Set up a flat cloth that falls onto sphere
        glm::mat4 transM = glm::translate(glm::mat4(1.0f), glm::vec3(-cloth.size.x / 2.0f, -2.0f, -cloth.size.y / 2.0f));
        for (uint32_t i = 0; i < cloth.gridsize.y; i++) {
            for (uint32_t j = 0; j < cloth.gridsize.x; j++) {
                particleBuffer[i + j * cloth.gridsize.y].pos = transM * glm::vec4(dx * j, 0.0f, dy * i, 1.0f);
                float anchor_damping = (i == 0 && (j ==0 || j == cloth.gridsize.x - 1)) ? 0.0 : 1.0;
                particleBuffer[i + j * cloth.gridsize.y].vel = glm::vec4(0.0f, 0.0f, 0.0f, anchor_damping);
                particleBuffer[i + j * cloth.gridsize.y].uv = glm::vec4(1.0f - du * i, dv * j, 0.0f, 0.0f);
            }
        }

        VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

        // Staging
        // SSBO won't be changed on the host after upload so copy to device local memory

        // Create a staging buffer used to upload data to the gpu
        Buffer stagingBuffer = createBuffer(storageBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        vkMapMemory(device, stagingBuffer.bufferMemory, 0, storageBufferSize, 0, &stagingBuffer.bufferMapped);
        memcpy(stagingBuffer.bufferMapped, particleBuffer.data(), (size_t)storageBufferSize);
        vkUnmapMemory(device, stagingBuffer.bufferMemory);
        
        storageBuffers.input = createBuffer(storageBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        storageBuffers.output = createBuffer(storageBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        copyBuffer(stagingBuffer.buffer, storageBuffers.input.buffer, storageBufferSize);
        copyBuffer(stagingBuffer.buffer, storageBuffers.output.buffer, storageBufferSize);
        


        stagingBuffer.destroy(device);

        // Indices
        std::vector<uint32_t> indices;
        for (uint32_t y = 0; y < cloth.gridsize.y - 1; y++) {
            for (uint32_t x = 0; x < cloth.gridsize.x; x++) {
                indices.push_back((y + 1) * cloth.gridsize.x + x);
                indices.push_back((y)*cloth.gridsize.x + x);
            }
            // Primitive restart (signaled by special value 0xFFFFFFFF)
            indices.push_back(0xFFFFFFFF);
        }
        uint32_t indexBufferSize = static_cast<uint32_t>(indices.size()) * sizeof(uint32_t);
        indexCount = static_cast<uint32_t>(indices.size());

        stagingBuffer = createBuffer(indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(device, stagingBuffer.bufferMemory, 0, indexBufferSize, 0, &stagingBuffer.bufferMapped);
        memcpy(stagingBuffer.bufferMapped, indices.data(), (size_t)indexBufferSize);
        vkUnmapMemory(device, stagingBuffer.bufferMemory);
        
        graphics.indices = createBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        copyBuffer(stagingBuffer.buffer, graphics.indices.buffer, indexBufferSize);


        stagingBuffer.destroy(device);
    }
     
    
    void recordCommandBuffer() {
        VkCommandBufferBeginInfo cmdBufInfo = initializers::commandBufferBeginInfo();

        VkClearValue clearValues[2];
        clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassBeginInfo = initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent.width = swapChainExtent.width;
        renderPassBeginInfo.renderArea.extent.height = swapChainExtent.height;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        for (int32_t i = 0; i < graphicsCommandBuffer.size(); ++i)
        {
            // Set target frame buffer
            renderPassBeginInfo.framebuffer = swapChainFramebuffers[i];

            VK_CHECK_RESULT(vkBeginCommandBuffer(graphicsCommandBuffer[i], &cmdBufInfo));

            // Draw the particle system using the update vertex buffer

            vkCmdBeginRenderPass(graphicsCommandBuffer[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport = initializers::viewport((float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);
            vkCmdSetViewport(graphicsCommandBuffer[i], 0, 1, &viewport);

            VkRect2D scissor = initializers::rect2D((float)swapChainExtent.width, (float)swapChainExtent.height, 0, 0);
            vkCmdSetScissor(graphicsCommandBuffer[i], 0, 1, &scissor);

            VkDeviceSize offsets[1] = { 0 };

            // Render sphere
            //vkCmdBindPipeline(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelines.sphere);
            //vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSet, 0, NULL);
            //modelSphere.draw(drawCmdBuffers[i]);

            // Render cloth
            vkCmdBindPipeline(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelines.cloth);
            vkCmdBindDescriptorSets(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSet, 0, NULL);
            vkCmdBindIndexBuffer(graphicsCommandBuffer[i], graphics.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindVertexBuffers(graphicsCommandBuffer[i], 0, 1, &storageBuffers.output.buffer, offsets);
            vkCmdDrawIndexed(graphicsCommandBuffer[i], indexCount, 1, 0, 0, 0);

            vkCmdEndRenderPass(graphicsCommandBuffer[i]);

            VK_CHECK_RESULT(vkEndCommandBuffer(graphicsCommandBuffer[i]));
        }
    }
    
    void recordComputeCommandBuffer() {
        VkCommandBufferBeginInfo cmdBufInfo = initializers::commandBufferBeginInfo();
        cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;


        VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));

        vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);

        uint32_t calculateNormals = 0;
        vkCmdPushConstants(compute.commandBuffer, compute.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &calculateNormals);

        // Dispatch the compute job
        const uint32_t iterations = 64;
        int index = 1;
        for (uint32_t j = 0; j < iterations; j++) {
            vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSets[index], 0, 0);

            if (j == iterations - 1) {
                calculateNormals = 1;
                vkCmdPushConstants(compute.commandBuffer, compute.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &calculateNormals);
            }

            vkCmdDispatch(compute.commandBuffer, cloth.gridsize.x / 10, cloth.gridsize.y / 10, 1);

            // Don't add a barrier on the last iteration of the loop
            if (j != iterations - 1) {
                addComputeToComputeBarriers(compute.commandBuffer);
            }
            index = 1 - index;

        }

        VK_CHECK_RESULT(vkEndCommandBuffer(compute.commandBuffer));
    }
    
    
    void updateUniformBuffer() {
   
        graphics.uniformData.projection = glm::perspective(glm::radians(60.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 512.0f);
        graphics.uniformData.view = glm::lookAt(glm::vec3(0.0f, -4.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        
        memcpy(graphics.uniformBuffer.bufferMapped, &graphics.uniformData, sizeof(Graphics::UniformData));
        
        compute.uniformData.deltaT = fmin(lastFrameTime, 0.02f) * 0.0025f;
        
        std::default_random_engine rndEngine((unsigned)time(nullptr));
        std::uniform_real_distribution<float> rd(1.0f, 12.0f);
        compute.uniformData.gravity.x = cos(glm::radians(-timer * 360.0f)) * (rd(rndEngine) - rd(rndEngine));
        compute.uniformData.gravity.z = sin(glm::radians(timer * 360.0f)) * (rd(rndEngine) - rd(rndEngine));
        
        memcpy(compute.uniformBuffer.bufferMapped, &compute.uniformData, sizeof(Compute::UniformData));
    }
    
    void drawFrame() {
        
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        
        // Wait for rendering finished
        VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        // Submit compute commands
        VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
        computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
        computeSubmitInfo.signalSemaphoreCount = 1;
        computeSubmitInfo.pSignalSemaphores = &compute.semaphore;
        VK_CHECK_RESULT(vkQueueSubmit(computeQueue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
        
        
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
        
        VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSemaphore graphicsWaitSemaphores[] = { compute.semaphore, imageAvailableSemaphore };
        VkSemaphore graphicsSignalSemaphores[] = { graphics.semaphore };

        // Submit graphics commands
        VkSubmitInfo graphicsSubmitInfo = initializers::submitInfo();
        graphicsSubmitInfo.commandBufferCount = 1;
        graphicsSubmitInfo.pCommandBuffers = &graphicsCommandBuffer[imageIndex];
        graphicsSubmitInfo.waitSemaphoreCount = 2;
        graphicsSubmitInfo.pWaitSemaphores = graphicsWaitSemaphores;
        //pWaitSemaphores signal 取得之前不执行 pWaitDstStageMask
        graphicsSubmitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
        graphicsSubmitInfo.signalSemaphoreCount = 1;
        graphicsSubmitInfo.pSignalSemaphores = graphicsSignalSemaphores;
        VK_CHECK_RESULT(vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, inFlightFence));

        
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &graphics.semaphore;
        
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

//int main() {
//    vks::ComputeCloth app;
//
//    try {
//
//
//        app.run();
//    } catch (const std::exception& e) {
//        std::cerr << e.what() << std::endl;
//        return EXIT_FAILURE;
//    }
//
//    return EXIT_SUCCESS;
//}

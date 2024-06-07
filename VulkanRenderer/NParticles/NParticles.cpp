//
//  CSParticles.cpp
//  VulkanRenderer
//
//

#include "../base/ApplicationBase.hpp"

namespace vks {

#define PARTICLES_PER_ATTRACTOR 256

class NParticles : public ApplicationBase {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
    
private:
    
    struct Textures {
        Texture particle;
        Texture gradient;
    } textures;
    
    // Particle Definition
    struct Particle {
        glm::vec4 pos;                                // xyz = position, w = mass
        glm::vec4 vel;                                // xyz = velocity, w = gradient texture position
    };
    
    uint32_t numParticles{ 0 };
    
    VkBuffer storageBuffer;
    VkDeviceMemory storageBufferMemory;
    VkDescriptorBufferInfo storageBufferInfo;
    
    // Resources for the graphics part of the example
    struct Graphics {
        VkDescriptorSetLayout descriptorSetLayout;    // Particle system rendering shader binding layout
        VkDescriptorSet descriptorSet;                // Particle system rendering shader bindings
        VkPipelineLayout pipelineLayout;            // Layout of the graphics pipeline
        VkPipeline pipeline;                        // Particle rendering pipeline
        VkSemaphore semaphore;                      // Execution dependency between compute & graphic submission
        struct UniformData {
            glm::mat4 projection;
            glm::mat4 view;
            glm::vec2 screenDim;
        } uniformData;
        
        Buffer uniformBuffer;
        
        void destroy(VkDevice device){
            uniformBuffer.destroy((device));
            
            vkDestroyPipeline(device, pipeline, nullptr);
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
        VkDescriptorSet descriptorSet;                // Compute shader bindings
        VkPipelineLayout pipelineLayout;            // Layout of the compute pipeline
        VkPipeline pipelineCalculate;                // Compute pipeline for N-Body velocity calculation (1st pass)
        VkPipeline pipelineIntegrate;                // Compute pipeline for euler integration (2nd pass)
        struct UniformData {                        // Compute shader uniform block object
            float deltaT{ 0.0f };                    // Frame delta time
            int32_t particleCount{ 0 };
            // Parameters used to control the behaviour of the particle system
            float gravity{ 0.002f };
            float power{ 0.75f };
            float soften{ 0.05f };
        } uniformData;
        
        Buffer uniformBuffer;
        
        void destroy(VkDevice device){
            uniformBuffer.destroy((device));
            
            vkDestroyPipeline(device, pipelineCalculate, nullptr);
            vkDestroyPipeline(device, pipelineIntegrate, nullptr);
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
            
            vkDestroyBuffer(device, storageBuffer, nullptr);
            vkFreeMemory(device, storageBufferMemory, nullptr);
            
            textures.particle.destroy(device);
            textures.gradient.destroy(device);
        }
        
        ApplicationBase::cleanup();
    }
    
    
    void prepareGraphics(){
        // Vertex shader uniform buffer block
        VkDeviceSize bufferSize = sizeof(Graphics::uniformData);
        graphics.uniformBuffer = createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, graphics.uniformBuffer.bufferMemory, 0, bufferSize, 0, &graphics.uniformBuffer.bufferMapped);
        
        // Descriptor pool
        std::vector<VkDescriptorPoolSize> poolSizes = {
            initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
            initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
            initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2)
        };
        VkDescriptorPoolCreateInfo poolInfo = initializers::descriptorPoolCreateInfo(poolSizes, 2);
        
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool))

        // Descriptor layout
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings = {
            initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
            initializers::descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
            initializers::descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
        };
        
        VkDescriptorSetLayoutCreateInfo layoutInfo = initializers::descriptorSetLayoutCreateInfo(layoutBindings);
        
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &graphics.descriptorSetLayout))
        
        // Descriptor set
        VkDescriptorSetAllocateInfo allocInfo = initializers::descriptorSetAllocateInfo(descriptorPool, 1, &graphics.descriptorSetLayout);
        
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSet));
        
        
        std::vector<VkWriteDescriptorSet> descriptorWrites = {
            initializers::writeDescriptorSet(graphics.descriptorSet, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &textures.particle.imageInfo, nullptr),
            initializers::writeDescriptorSet(graphics.descriptorSet, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &textures.gradient.imageInfo, nullptr),
            initializers::writeDescriptorSet(graphics.descriptorSet, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &graphics.uniformBuffer.bufferInfo)
        };

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        
        
        // Pipeline layout
        VkPipelineLayoutCreateInfo createInfo = initializers::pipelineLayoutCreateInfo(1, &graphics.descriptorSetLayout);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &createInfo, nullptr, &graphics.pipelineLayout));

        // Pipeline
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);
        
        VkPipelineRasterizationStateCreateInfo rasterizationState = initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
        
        VkPipelineColorBlendAttachmentState blendAttachmentState = initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        // Additive blending
        blendAttachmentState.colorWriteMask = 0xF;
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
        blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
        
        VkPipelineColorBlendStateCreateInfo colorBlendState = initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
        
        VkPipelineDepthStencilStateCreateInfo depthStencilState = initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);
        
        VkPipelineViewportStateCreateInfo viewportState = initializers::pipelineViewportStateCreateInfo(1, 1, 0);
        
        VkPipelineMultisampleStateCreateInfo multisampleState = initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
        
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState = initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
    

        // Vertex Input state
        std::vector<VkVertexInputBindingDescription> inputBindings = {
            initializers::vertexInputBindingDescription(0, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX)
        };
        
        std::vector<VkVertexInputAttributeDescription> inputAttributes = {
            // Location 0 : Position
            initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Particle, pos)),
            // Location 1 : Velocity (used for color gradient lookup)
            initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Particle, vel)),
        };
        
        VkPipelineVertexInputStateCreateInfo vertexInputState = initializers::pipelineVertexInputStateCreateInfo();
        vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(inputBindings.size());
        vertexInputState.pVertexBindingDescriptions = inputBindings.data();
        vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
        vertexInputState.pVertexAttributeDescriptions = inputAttributes.data();

        // Shaders
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {
            loadShader("shaders/NParticles/particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
            loadShader("shaders/NParticles/particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
        };
        
        VkGraphicsPipelineCreateInfo pipelineCreateInfo = initializers::pipelineCreateInfo(graphics.pipelineLayout, renderPass, 0);
        pipelineCreateInfo.pVertexInputState = &vertexInputState;
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

        VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &graphics.pipeline));

        // We use a semaphore to synchronize compute and graphics
        VkSemaphoreCreateInfo semaphoreCreateInfo = initializers::semaphoreCreateInfo();
        VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &graphics.semaphore));

        // Signal the semaphore for the first run
        //VkSubmitInfo submitInfo = initializers::submitInfo();
        //submitInfo.signalSemaphoreCount = 1;
        //submitInfo.pSignalSemaphores = &graphics.semaphore;
        //VK_CHECK_RESULT(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
        //VK_CHECK_RESULT(vkQueueWaitIdle(graphicsQueue));
                                                                                
        recordCommandBuffer();

        
    }
    
    void prepareCompute(){

        // Compute shader uniform buffer block
        VkDeviceSize bufferSize = sizeof(Compute::UniformData);
        compute.uniformBuffer = createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkMapMemory(device, compute.uniformBuffer.bufferMemory, 0, bufferSize, 0, &compute.uniformBuffer.bufferMapped);
        
        // Create compute pipeline
        // Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)

        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Particle position storage buffer
            initializers::descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT),
            // Binding 1 : Uniform buffer
            initializers::descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT),
        };

        VkDescriptorSetLayoutCreateInfo descriptorLayout = initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

        VkDescriptorSetAllocateInfo allocInfo = initializers::descriptorSetAllocateInfo(descriptorPool,1, &compute.descriptorSetLayout);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));

        
        std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
            // Binding 0 : Particle position storage buffer
            initializers::writeDescriptorSet(compute.descriptorSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &storageBufferInfo),
            // Binding 1 : Uniform buffer
            initializers::writeDescriptorSet(compute.descriptorSet, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &compute.uniformBuffer.bufferInfo)
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, nullptr);

        // Create pipelines
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = initializers::pipelineLayoutCreateInfo(1, &compute.descriptorSetLayout);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));

        VkComputePipelineCreateInfo computePipelineCreateInfo = initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);

        // 1st pass
        computePipelineCreateInfo.stage = loadShader("shaders/NParticles/particle_calculate.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

        // We want to use as much shared memory for the compute shader invocations as available, so we calculate it based on the device limits and pass it to the shader via specialization constants
        uint32_t sharedDataSize = std::min((uint32_t)256, (uint32_t)(properties.limits.maxComputeSharedMemorySize / sizeof(glm::vec4)));
        std::cout<<"sharedDataSize:" <<sharedDataSize<<std::endl;
        VkSpecializationMapEntry specializationMapEntry = initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
        VkSpecializationInfo specializationInfo = initializers::specializationInfo(1, &specializationMapEntry, sizeof(int32_t), &sharedDataSize);
        computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;

        VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &compute.pipelineCalculate));

        // 2nd pass
        computePipelineCreateInfo.stage = loadShader( "shaders/NParticles/particle_integrate.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
        VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &compute.pipelineIntegrate));

        // Separate command pool as queue family for compute may be different than graphics
        VkCommandPoolCreateInfo cmdPoolInfo = {};
        cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmdPoolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();
        cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

        // Create a command buffer for compute operations
        createCommandBuffers(compute.commandPool, 1, &compute.commandBuffer);

        // Semaphore for compute & graphics sync
        VkSemaphoreCreateInfo semaphoreCreateInfo = initializers::semaphoreCreateInfo();
        VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore));

        // Build a single command buffer containing the compute dispatch commands
        recordComputeCommandBuffer();
        
    }
       
    
    void loadAssets(){
        textures.particle = createTextureImageFromFile("textures/Star-Png-172.png", VK_FORMAT_R8G8B8A8_UNORM);
        textures.gradient = createTextureImageFromFile("textures/rock.png", VK_FORMAT_R8G8B8A8_UNORM);
    }
    
    void createShaderStorageBuffers() {
        // We mark a few particles as attractors that move along a given path, these will pull in the other particles
        std::vector<glm::vec3> attractors = {
            glm::vec3(5.0f, 0.0f, 0.0f),
            glm::vec3(-5.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 5.0f),
            glm::vec3(0.0f, 0.0f, -5.0f),
        };

        numParticles = static_cast<uint32_t>(attractors.size()) * PARTICLES_PER_ATTRACTOR;

        // Initial particle positions
        std::vector<Particle> particleBuffer(numParticles);
    
        
        std::default_random_engine rndEngine(0);
        std::normal_distribution<float> rndDist(0.0f, 1.0f);

        for (uint32_t i = 0; i < static_cast<uint32_t>(attractors.size()); i++)
        {
            for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; j++)
            {
                Particle& particle = particleBuffer[i * PARTICLES_PER_ATTRACTOR + j];

                // First particle in group as heavy center of gravity
                if (j == 0)
                {
                    particle.pos = glm::vec4(attractors[i] * 1.5f, 90000.0f);
                    particle.vel = glm::vec4(glm::vec4(0.0f));
                }
                else
                {
                    // Position
                    glm::vec3 position(attractors[i] + glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine)) * 0.75f);
                    float len = glm::length(glm::normalize(position - attractors[i]));
                    position.y *= 2.0f - (len * len);

                    // Velocity
                    glm::vec3 angular = glm::vec3(0.5f, 1.5f, 0.5f) * (((i % 2) == 0) ? 1.0f : -1.0f);
                    glm::vec3 velocity = glm::cross((position - attractors[i]), angular) + glm::vec3(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine) * 0.025f);

                    float mass = (rndDist(rndEngine) * 0.5f + 0.5f) * 75.0f;
                    particle.pos = glm::vec4(position, mass);
                    particle.vel = glm::vec4(velocity, 0.0f);
                }

                // Color gradient offset
                particle.vel.w = (float)i * 1.0f / static_cast<uint32_t>(attractors.size());
            }
        }

        compute.uniformData.particleCount = numParticles;

        VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

        // Create a staging buffer used to upload data to the gpu
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(storageBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, storageBufferSize, 0, &data);
        memcpy(data, particleBuffer.data(), (size_t)storageBufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        
        // Copy initial particle data to all storage buffers
        createBuffer(storageBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffer, storageBufferMemory);
        copyBuffer(stagingBuffer, storageBuffer, storageBufferSize);
        
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        
        storageBufferInfo.buffer = storageBuffer;
        storageBufferInfo.offset = 0;
        storageBufferInfo.range = storageBufferSize;
        
    }
     
    
    void recordCommandBuffer() {
        VkCommandBufferBeginInfo cmdBufInfo = initializers::commandBufferBeginInfo();

        VkClearValue clearValues[2];
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
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

            VkRect2D scissor = initializers::rect2D(swapChainExtent.width, swapChainExtent.height, 0, 0);
            vkCmdSetScissor(graphicsCommandBuffer[i], 0, 1, &scissor);

            vkCmdBindPipeline(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);
            vkCmdBindDescriptorSets(graphicsCommandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSet, 0, nullptr);

            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(graphicsCommandBuffer[i], 0, 1, &storageBuffer, offsets);
            vkCmdDraw(graphicsCommandBuffer[i], numParticles, 1, 0, 0);

            vkCmdEndRenderPass(graphicsCommandBuffer[i]);

            VK_CHECK_RESULT(vkEndCommandBuffer(graphicsCommandBuffer[i]));
        }
    }
    
    void recordComputeCommandBuffer() {
        VkCommandBufferBeginInfo cmdBufInfo = initializers::commandBufferBeginInfo();

        VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));

        // First pass: Calculate particle movement
        // -------------------------------------------------------------------------------------------------------
        vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineCalculate);
        vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);
        vkCmdDispatch(compute.commandBuffer, numParticles / 256, 1, 1);

        // Add memory barrier to ensure that the computer shader has finished writing to the buffer
        VkBufferMemoryBarrier bufferBarrier = initializers::bufferMemoryBarrier();
        bufferBarrier.buffer = storageBuffer;
        bufferBarrier.size = storageBufferInfo.range;
        bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        // Transfer ownership if compute and graphics queue family indices differ
        bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(
                compute.commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &bufferBarrier,
                0, nullptr);

        // Second pass: Integrate particles
        // -------------------------------------------------------------------------------------------------------
        vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineIntegrate);
        vkCmdDispatch(compute.commandBuffer, numParticles / 256, 1, 1);


        vkEndCommandBuffer(compute.commandBuffer);
        
    }
    
    
    void updateUniformBuffer() {
        //static auto startTime = std::chrono::high_resolution_clock::now();
        
        //auto currentTime = std::chrono::high_resolution_clock::now();
        //float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        
        compute.uniformData.deltaT = lastFrameTime * 0.05;
        memcpy(compute.uniformBuffer.bufferMapped, &compute.uniformData, sizeof(Compute::UniformData));
        
        graphics.uniformData.projection = glm::perspective(glm::radians(60.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 512.0f);
        graphics.uniformData.projection[1][1] *= -1;
        graphics.uniformData.view = glm::lookAt(glm::vec3(4.0f, 4.0f, 4.0f), glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        graphics.uniformData.screenDim = glm::vec2((float)swapChainExtent.width, (float)swapChainExtent.height);
        
        memcpy(graphics.uniformBuffer.bufferMapped, &graphics.uniformData, sizeof(Graphics::UniformData));
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
//    vks::NParticles app;
//
//    try {
//        app.run();
//    } catch (const std::exception& e) {
//        std::cerr << e.what() << std::endl;
//        return EXIT_FAILURE;
//    }
//
//    return EXIT_SUCCESS;
//}

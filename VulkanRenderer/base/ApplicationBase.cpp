//
//  ApplicationBase.cpp
//  VulkanRenderer
//
//  Created by Yang Liu on 2024/6/6.
//

#include <stdio.h>
#include "ApplicationBase.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace vks{
    Texture ApplicationBase::createTextureImageFromFile(std::string path, VkFormat format) {
        Texture result;
        
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;
        
        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);
        
        stbi_image_free(pixels);
        
        createImage(texWidth, texHeight, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, result.image, result.imageMemory);
        
        transitionImageLayout(result.image, format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, result.image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(result.image, format, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        
        result.imageView = createImageView(result.image, format, VK_IMAGE_ASPECT_COLOR_BIT);
        
        result.sampler = createTextureSampler();
        
        result.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        result.imageInfo.imageView = result.imageView;
        result.imageInfo.sampler = result.sampler;
        
        return result;
    }
    
    
}

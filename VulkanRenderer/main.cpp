//
//  main.cpp
//  VulkanRenderer
//
//  Created by yang on 2024/6/21.
//

#include <stdio.h>

#include "VolumetricLight/VolumetricLight.h"

template<typename T>
int EntryFunction()
{
    vks::ApplicationBase* app = new T();

    try {
        app->run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    delete app;
    
    return EXIT_SUCCESS;
}


int main() {
    EntryFunction<vks::VolumetricLight>();
}

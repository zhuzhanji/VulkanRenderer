1.Generalized Kuwahara Filter 

2.Cloth Shading

3.Particle System

4.Particles from models and particles morphing


---
# 4.Particles from models and particle morphing
  * technique: compute shader, instancing, shape morphing, depth testing, model loading etc.
    
https://github.com/zhuzhanji/VulkanRenderer/assets/37281560/d1f1cbfd-cfeb-466e-be80-01404251bb1f



https://github.com/zhuzhanji/VulkanRenderer/assets/37281560/82123c9b-9060-4962-bad2-05e1facd5f17





In the above two demos, particles are drawn with GL_POINT, we can replace them with 3D models, like this rock. Instancing is very efficient in this cenario. (57275 rocks overall)

<img width="378" alt="image" src="https://github.com/zhuzhanji/VulkanRenderer/assets/37281560/a29cf219-0cce-45a3-b5fa-e3c7142f003a">

<img width="880" alt="image" src="https://github.com/zhuzhanji/VulkanRenderer/assets/37281560/37d78151-a888-4d1b-9e33-bc46bd306eb9">



# 1.Generalized Kuwahara Filter rendered in two viewports

  - left (radius = 6 pixels), right (radius = 12 pixels)
<img width="1251" alt="image" src="https://github.com/zhuzhanji/VulkanRenderer/assets/37281560/5384369d-74fb-4b2c-b40f-dbad06c9e599">

## Original

https://en.wikipedia.org/wiki/Lion#/media/File:Lion_waiting_in_Namibia.jpg

# 2.Implementation of Cloth Shading
  - implementation of Uncharted 2: Character Lighting and Shading
  - Silk, Cotton and Denim
    
<img width="400" alt="Screenshot 2024-05-20 at 11 06 27" src="https://github.com/zhuzhanji/VulkanRenderer/assets/37281560/b54a1313-4506-4d3d-aebb-f34a26c23974">
<img width="400" alt="Screenshot 2024-05-20 at 11 04 42" src="https://github.com/zhuzhanji/VulkanRenderer/assets/37281560/3958e465-2c8f-4b0c-8319-dce475eefcba">
<img width="400" alt="Screenshot 2024-05-20 at 11 18 43" src="https://github.com/zhuzhanji/VulkanRenderer/assets/37281560/5e28df07-51d2-4030-9b00-c93f7f2183db">

# 3. Particle system using computer shader

https://github.com/zhuzhanji/VulkanRenderer/assets/37281560/7fb729e3-f85b-45ee-81a9-65d5d892708e


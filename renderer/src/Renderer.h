#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Scene.h"
#include "Camera.h"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    bool init();
    void render(Scene& scene, const Camera& camera, const std::string& outputFilename, int frameIndex);
    
private:
    unsigned int compileShader(const char* source, unsigned int type);
    unsigned int createShaderProgram(const char* vertPath, const char* fragPath);
    void saveImage(const std::string& filename);

    int m_width;
    int m_height;
    
    // OpenGL objects
    unsigned int m_fbo;
    unsigned int m_rbo; // Depth/Stencil
    unsigned int m_textureColorBuffer;
    
    unsigned int m_shaderProgram;
    
    // GLFW window (needed for context)
    struct GLFWwindow* m_window;
};

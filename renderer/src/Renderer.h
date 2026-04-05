#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Scene.h"
#include "MultiScene.h"
#include "Camera.h"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    bool init();
    void render(Scene& scene, const Camera& camera, const std::string& outputFilename, int frameIndex);
    void render(MultiScene& scene, const Camera& camera, const std::string& outputFilename, int frameIndex);
    
private:
    unsigned int compileShader(const char* source, unsigned int type);
    unsigned int createShaderProgram(const char* vertPath, const char* fragPath);
    void saveImage(const std::string& filename);
    void initBackgroundQuad();
    void drawBackground(const glm::mat4& view, const glm::mat4& projection);
    void drawSkyGradient(int frameIndex);

    int m_width;
    int m_height;
    
    // OpenGL objects — MSAA offscreen + single-sample resolve for PNG readback
    unsigned int m_msaaFbo;
    unsigned int m_msaaColorRbo;
    unsigned int m_msaaDepthRbo;
    int m_msaaSamples;

    unsigned int m_fbo;           // resolve FBO (single-sample color)
    unsigned int m_textureColorBuffer;

    void bindMsaaTarget();
    void resolveMsaaToTexture();
    
    unsigned int m_shaderProgram;
    unsigned int m_bgShaderProgram;
    unsigned int m_skyGradientProgram;
    unsigned int m_bgVAO;
    unsigned int m_bgVBO;
    unsigned int m_skyVAO;
    
    // GLFW window (needed for context)
    struct GLFWwindow* m_window;
};

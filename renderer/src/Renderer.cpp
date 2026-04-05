#include "Renderer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <algorithm>

#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stb_image_write.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Renderer::Renderer(int width, int height) 
    : m_width(width), m_height(height), m_window(nullptr),
      m_msaaFbo(0), m_msaaColorRbo(0), m_msaaDepthRbo(0), m_msaaSamples(4),
      m_fbo(0), m_textureColorBuffer(0), m_shaderProgram(0), m_bgShaderProgram(0), m_skyGradientProgram(0),
      m_bgVAO(0), m_bgVBO(0), m_skyVAO(0) {}

Renderer::~Renderer() {
    if (m_msaaFbo) glDeleteFramebuffers(1, &m_msaaFbo);
    if (m_msaaColorRbo) glDeleteRenderbuffers(1, &m_msaaColorRbo);
    if (m_msaaDepthRbo) glDeleteRenderbuffers(1, &m_msaaDepthRbo);
    if (m_fbo) glDeleteFramebuffers(1, &m_fbo);
    if (m_textureColorBuffer) glDeleteTextures(1, &m_textureColorBuffer);
    if (m_bgVAO) glDeleteVertexArrays(1, &m_bgVAO);
    if (m_bgVBO) glDeleteBuffers(1, &m_bgVBO);
    if (m_skyVAO) glDeleteVertexArrays(1, &m_skyVAO);
    if (m_skyGradientProgram) glDeleteProgram(m_skyGradientProgram);
    if (m_window) glfwDestroyWindow(m_window);
    glfwTerminate();
}

bool Renderer::init() {
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // Offscreen

    m_window = glfwCreateWindow(m_width, m_height, "SynthSplat Renderer", nullptr, nullptr);
    if (!m_window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    glViewport(0, 0, m_width, m_height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    // --- Multisampled offscreen FBO (renders here, then resolve) ---
    GLint maxSamples = 4;
    glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
    m_msaaSamples = std::max(1, std::min(4, maxSamples));

    glGenFramebuffers(1, &m_msaaFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_msaaFbo);

    glGenRenderbuffers(1, &m_msaaColorRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, m_msaaColorRbo);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, m_msaaSamples, GL_RGB8, m_width, m_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_msaaColorRbo);

    glGenRenderbuffers(1, &m_msaaDepthRbo);
    glBindRenderbuffer(GL_RENDERBUFFER, m_msaaDepthRbo);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, m_msaaSamples, GL_DEPTH24_STENCIL8, m_width, m_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_msaaDepthRbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "MSAA framebuffer is not complete!" << std::endl;
        return false;
    }

    // --- Resolve FBO: single-sample texture for glReadPixels / PNG ---
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    glGenTextures(1, &m_textureColorBuffer);
    glBindTexture(GL_TEXTURE_2D, m_textureColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_textureColorBuffer, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "Resolve framebuffer is not complete!" << std::endl;
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    std::cout << "Offscreen MSAA samples: " << m_msaaSamples << std::endl;

    // Load Shaders
    m_shaderProgram = createShaderProgram("shaders/pbr.vert", "shaders/pbr.frag");
    if (m_shaderProgram == 0) return false;

    // Load background shader (optional checkerboard box — legacy)
    m_bgShaderProgram = createShaderProgram("shaders/background.vert", "shaders/background.frag");
    if (m_bgShaderProgram == 0) return false;

    m_skyGradientProgram = createShaderProgram("shaders/sky.vert", "shaders/sky.frag");
    if (m_skyGradientProgram == 0) return false;

    glGenVertexArrays(1, &m_skyVAO);

    // Initialize background geometry (large box)
    initBackgroundQuad();

    return true;
}

void Renderer::drawSkyGradient(int frameIndex) {
    glDepthMask(GL_FALSE);
    glUseProgram(m_skyGradientProgram);
    glUniform1ui(glGetUniformLocation(m_skyGradientProgram, "uFrame"),
                   static_cast<unsigned int>(frameIndex & 0xFFFF));
    glBindVertexArray(m_skyVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glDepthMask(GL_TRUE);
}

void Renderer::bindMsaaTarget() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_msaaFbo);
    glViewport(0, 0, m_width, m_height);
}

void Renderer::resolveMsaaToTexture() {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_msaaFbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_fbo);
    glBlitFramebuffer(0, 0, m_width, m_height, 0, 0, m_width, m_height,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::initBackgroundQuad() {
    // Create a large box (skybox-like) that surrounds the scene
    // Smaller size so it doesn't overwhelm smaller scenes
    float size = 15.0f;
    float vertices[] = {
        // Back face
        -size, -size, -size,
         size, -size, -size,
         size,  size, -size,
         size,  size, -size,
        -size,  size, -size,
        -size, -size, -size,

        // Front face
        -size, -size,  size,
         size,  size,  size,
         size, -size,  size,
         size,  size,  size,
        -size, -size,  size,
        -size,  size,  size,

        // Left face
        -size,  size,  size,
        -size, -size, -size,
        -size,  size, -size,
        -size, -size, -size,
        -size,  size,  size,
        -size, -size,  size,

        // Right face
         size,  size,  size,
         size,  size, -size,
         size, -size, -size,
         size, -size, -size,
         size, -size,  size,
         size,  size,  size,

        // Bottom face
        -size, -size, -size,
         size, -size,  size,
         size, -size, -size,
         size, -size,  size,
        -size, -size, -size,
        -size, -size,  size,

        // Top face
        -size,  size, -size,
         size,  size, -size,
         size,  size,  size,
         size,  size,  size,
        -size,  size,  size,
        -size,  size, -size,
    };

    glGenVertexArrays(1, &m_bgVAO);
    glGenBuffers(1, &m_bgVBO);
    
    glBindVertexArray(m_bgVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_bgVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    glBindVertexArray(0);
}

void Renderer::drawBackground(const glm::mat4& view, const glm::mat4& projection) {
    glUseProgram(m_bgShaderProgram);
    
    glUniformMatrix4fv(glGetUniformLocation(m_bgShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_bgShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    
    // Disable face culling to see inside of box
    glDisable(GL_CULL_FACE);
    
    glBindVertexArray(m_bgVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
    
    // Re-enable culling for the main model
    glEnable(GL_CULL_FACE);
}

void Renderer::render(Scene& scene, const Camera& camera, const std::string& outputFilename, int frameIndex) {
    bindMsaaTarget();
    
    glClearColor(0.15f, 0.35f, 0.72f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set View/Projection
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = camera.getProjectionMatrix();

    drawSkyGradient(frameIndex);

    // Enable culling for the main model
    glEnable(GL_CULL_FACE);

    glUseProgram(m_shaderProgram);
    
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "camPos"), 1, glm::value_ptr(camera.getPosition()));

    // 2. Lighting with per-frame variation for view-dependent learning
    // Base light direction and intensity
    glm::vec3 baseLightDir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.8f)); // 3/4 angle
    float baseIntensity = 5.0f;
    
    glm::vec3 lightDir = baseLightDir;
    glm::vec3 lightColor = glm::vec3(baseIntensity);

    // Always apply per-frame lighting variation (no longer compile-time flag)
    // This forces 3DGS Spherical Harmonics to learn true view-dependent reflections
    {
        std::mt19937 gen(frameIndex); // Seed with frame index for reproducibility
        std::uniform_real_distribution<float> angleDis(-1.0f, 1.0f);
        std::uniform_real_distribution<float> intensityDis(0.8f, 1.2f);
        
        // Randomize light direction within a 15-degree cone
        float maxAngleRad = glm::radians(15.0f);
        
        // Generate random rotation angles
        float rotX = angleDis(gen) * maxAngleRad;
        float rotZ = angleDis(gen) * maxAngleRad;
        
        // Apply small rotations to the base light direction
        // Rotate around X axis
        glm::vec3 rotatedDir = baseLightDir;
        float cosX = std::cos(rotX);
        float sinX = std::sin(rotX);
        float newY = rotatedDir.y * cosX - rotatedDir.z * sinX;
        float newZ = rotatedDir.y * sinX + rotatedDir.z * cosX;
        rotatedDir.y = newY;
        rotatedDir.z = newZ;
        
        // Rotate around Z axis
        float cosZ = std::cos(rotZ);
        float sinZ = std::sin(rotZ);
        float newX = rotatedDir.x * cosZ - rotatedDir.y * sinZ;
        newY = rotatedDir.x * sinZ + rotatedDir.y * cosZ;
        rotatedDir.x = newX;
        rotatedDir.y = newY;
        
        lightDir = glm::normalize(rotatedDir);
        
        // Randomize intensity between 0.8x and 1.2x of base
        float intensityMultiplier = intensityDis(gen);
        lightColor = glm::vec3(baseIntensity * intensityMultiplier);
    }

    glUniform3fv(glGetUniformLocation(m_shaderProgram, "lightDir"), 1, glm::value_ptr(lightDir));
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));

    // Draw Scene
    scene.draw(m_shaderProgram, glm::mat4(1.0f));

    resolveMsaaToTexture();
    saveImage(outputFilename);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::render(MultiScene& scene, const Camera& camera, const std::string& outputFilename, int frameIndex) {
    bindMsaaTarget();
    
    glClearColor(0.15f, 0.35f, 0.72f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set View/Projection
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = camera.getProjectionMatrix();

    drawSkyGradient(frameIndex);

    // Disable culling so all faces (including back walls) are visible
    glDisable(GL_CULL_FACE);

    glUseProgram(m_shaderProgram);
    
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "camPos"), 1, glm::value_ptr(camera.getPosition()));

    // Initialize isGroundPlane to false
    glUniform1i(glGetUniformLocation(m_shaderProgram, "isGroundPlane"), 0);

    // 2. Lighting with per-frame variation for view-dependent learning
    glm::vec3 baseLightDir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.8f));
    float baseIntensity = 8.0f;  // Brighter lighting for better visibility
    
    glm::vec3 lightDir = baseLightDir;
    glm::vec3 lightColor = glm::vec3(baseIntensity);

    {
        std::mt19937 gen(frameIndex);
        std::uniform_real_distribution<float> angleDis(-1.0f, 1.0f);
        std::uniform_real_distribution<float> intensityDis(0.8f, 1.2f);
        
        float maxAngleRad = glm::radians(15.0f);
        float rotX = angleDis(gen) * maxAngleRad;
        float rotZ = angleDis(gen) * maxAngleRad;
        
        glm::vec3 rotatedDir = baseLightDir;
        float cosX = std::cos(rotX);
        float sinX = std::sin(rotX);
        float newY = rotatedDir.y * cosX - rotatedDir.z * sinX;
        float newZ = rotatedDir.y * sinX + rotatedDir.z * cosX;
        rotatedDir.y = newY;
        rotatedDir.z = newZ;
        
        float cosZ = std::cos(rotZ);
        float sinZ = std::sin(rotZ);
        float newX = rotatedDir.x * cosZ - rotatedDir.y * sinZ;
        newY = rotatedDir.x * sinZ + rotatedDir.y * cosZ;
        rotatedDir.x = newX;
        rotatedDir.y = newY;
        
        lightDir = glm::normalize(rotatedDir);
        
        float intensityMultiplier = intensityDis(gen);
        lightColor = glm::vec3(baseIntensity * intensityMultiplier);
    }

    glUniform3fv(glGetUniformLocation(m_shaderProgram, "lightDir"), 1, glm::value_ptr(lightDir));
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));

    // Draw MultiScene
    scene.draw(m_shaderProgram, m_bgShaderProgram);

    resolveMsaaToTexture();
    saveImage(outputFilename);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::saveImage(const std::string& filename) {
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    std::vector<unsigned char> pixels(m_width * m_height * 3);
    glReadPixels(0, 0, m_width, m_height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
    // Flip vertically because OpenGL origin is bottom-left
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filename.c_str(), m_width, m_height, 3, pixels.data(), m_width * 3);
}

unsigned int Renderer::compileShader(const char* source, unsigned int type) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "Shader Compilation Failed: " << infoLog << std::endl;
    }
    return shader;
}

unsigned int Renderer::createShaderProgram(const char* vertPath, const char* fragPath) {
    std::string vertCode, fragCode;
    std::ifstream vShaderFile, fShaderFile;
    
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
    try {
        vShaderFile.open(vertPath);
        fShaderFile.open(fragPath);
        std::stringstream vShaderStream, fShaderStream;
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();
        vShaderFile.close();
        fShaderFile.close();
        vertCode = vShaderStream.str();
        fragCode = fShaderStream.str();
    } catch (std::ifstream::failure& e) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << e.what() << std::endl;
        return 0;
    }
    
    const char* vShaderCode = vertCode.c_str();
    const char* fShaderCode = fragCode.c_str();
    
    unsigned int vertex = compileShader(vShaderCode, GL_VERTEX_SHADER);
    unsigned int fragment = compileShader(fShaderCode, GL_FRAGMENT_SHADER);
    
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertex);
    glAttachShader(shaderProgram, fragment);
    glLinkProgram(shaderProgram);
    
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "Shader Linking Failed: " << infoLog << std::endl;
    }
    
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    
    return shaderProgram;
}

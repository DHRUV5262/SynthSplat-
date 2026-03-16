#include "Renderer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

#include <glm/gtc/type_ptr.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stb_image_write.h>

Renderer::Renderer(int width, int height) 
    : m_width(width), m_height(height), m_window(nullptr), m_fbo(0), m_rbo(0), m_textureColorBuffer(0), m_shaderProgram(0) {}

Renderer::~Renderer() {
    if (m_fbo) glDeleteFramebuffers(1, &m_fbo);
    if (m_textureColorBuffer) glDeleteTextures(1, &m_textureColorBuffer);
    if (m_rbo) glDeleteRenderbuffers(1, &m_rbo);
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

    // Create FBO
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);

    // Color attachment
    glGenTextures(1, &m_textureColorBuffer);
    glBindTexture(GL_TEXTURE_2D, m_textureColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_textureColorBuffer, 0);

    // Depth/Stencil attachment
    glGenRenderbuffers(1, &m_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, m_rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_width, m_height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "Framebuffer is not complete!" << std::endl;
        return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Load Shaders
    m_shaderProgram = createShaderProgram("shaders/pbr.vert", "shaders/pbr.frag");
    if (m_shaderProgram == 0) return false;

    return true;
}

void Renderer::render(Scene& scene, const Camera& camera, const std::string& outputFilename, int frameIndex) {
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_width, m_height);
    
    // Background color (Sky)
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Enable culling for performance (ensure model winding is correct)
    glEnable(GL_CULL_FACE);

    glUseProgram(m_shaderProgram);

    // Set View/Projection
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = camera.getProjectionMatrix();
    
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "camPos"), 1, glm::value_ptr(camera.getPosition()));

    // Lighting
    glm::vec3 lightDir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.8f)); // 3/4 angle
    glm::vec3 lightColor = glm::vec3(5.0f); // Intensity 5.0

#ifdef VARY_LIGHTING
    // Randomize lighting based on frameIndex (seeded)
    std::mt19937 gen(frameIndex); // Seed with frame index for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_real_distribution<float> colorDis(0.5f, 1.0f); // Bright colors

    lightDir = glm::normalize(glm::vec3(dis(gen), -1.0f, dis(gen))); // Always pointing somewhat down
    lightColor = glm::vec3(colorDis(gen), colorDis(gen), colorDis(gen));
#endif

    glUniform3fv(glGetUniformLocation(m_shaderProgram, "lightDir"), 1, glm::value_ptr(lightDir));
    glUniform3fv(glGetUniformLocation(m_shaderProgram, "lightColor"), 1, glm::value_ptr(lightColor));

    // Draw Scene
    scene.draw(m_shaderProgram, glm::mat4(1.0f));

    // Save Image
    saveImage(outputFilename);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::saveImage(const std::string& filename) {
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

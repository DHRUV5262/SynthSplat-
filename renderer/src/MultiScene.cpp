#include "MultiScene.h"
#include <iostream>
#include <limits>
#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

MultiScene::MultiScene() 
    : m_minBounds(std::numeric_limits<float>::max()),
      m_maxBounds(std::numeric_limits<float>::lowest()) {
}

MultiScene::~MultiScene() {
    if (m_groundVAO) glDeleteVertexArrays(1, &m_groundVAO);
    if (m_groundVBO) glDeleteBuffers(1, &m_groundVBO);
}

void MultiScene::addGroundPlane(float size, float y) {
    m_hasGround = true;
    m_groundSize = size;
    m_groundY = y;
    initGroundPlane();
    updateBounds();
    std::cout << "Ground plane added: size=" << size << ", y=" << y << std::endl;
}

void MultiScene::initGroundPlane() {
    float s = m_groundSize;
    float y = m_groundY;
    float h = 0.3f; // Height/thickness of the ground block
    
    // Ground block vertices - a rectangular box (6 faces, 36 vertices)
    // Top is at y, bottom is at y-h
    float vertices[] = {
        // TOP FACE (green grass) - Normal pointing up
        -s, y, -s,    0.0f, 1.0f, 0.0f,   0.0f, 0.0f,
         s, y, -s,    0.0f, 1.0f, 0.0f,   1.0f, 0.0f,
         s, y,  s,    0.0f, 1.0f, 0.0f,   1.0f, 1.0f,
         s, y,  s,    0.0f, 1.0f, 0.0f,   1.0f, 1.0f,
        -s, y,  s,    0.0f, 1.0f, 0.0f,   0.0f, 1.0f,
        -s, y, -s,    0.0f, 1.0f, 0.0f,   0.0f, 0.0f,
        
        // FRONT FACE - Normal pointing forward (+Z)
        -s, y-h,  s,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,
         s, y-h,  s,   0.0f, 0.0f, 1.0f,   1.0f, 0.0f,
         s, y,    s,   0.0f, 0.0f, 1.0f,   1.0f, 1.0f,
         s, y,    s,   0.0f, 0.0f, 1.0f,   1.0f, 1.0f,
        -s, y,    s,   0.0f, 0.0f, 1.0f,   0.0f, 1.0f,
        -s, y-h,  s,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,
        
        // BACK FACE - Normal pointing backward (-Z)
         s, y-h, -s,   0.0f, 0.0f, -1.0f,  0.0f, 0.0f,
        -s, y-h, -s,   0.0f, 0.0f, -1.0f,  1.0f, 0.0f,
        -s, y,   -s,   0.0f, 0.0f, -1.0f,  1.0f, 1.0f,
        -s, y,   -s,   0.0f, 0.0f, -1.0f,  1.0f, 1.0f,
         s, y,   -s,   0.0f, 0.0f, -1.0f,  0.0f, 1.0f,
         s, y-h, -s,   0.0f, 0.0f, -1.0f,  0.0f, 0.0f,
        
        // LEFT FACE - Normal pointing left (-X)
        -s, y-h, -s,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f,
        -s, y-h,  s,   -1.0f, 0.0f, 0.0f,  1.0f, 0.0f,
        -s, y,    s,   -1.0f, 0.0f, 0.0f,  1.0f, 1.0f,
        -s, y,    s,   -1.0f, 0.0f, 0.0f,  1.0f, 1.0f,
        -s, y,   -s,   -1.0f, 0.0f, 0.0f,  0.0f, 1.0f,
        -s, y-h, -s,   -1.0f, 0.0f, 0.0f,  0.0f, 0.0f,
        
        // RIGHT FACE - Normal pointing right (+X)
         s, y-h,  s,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f,
         s, y-h, -s,   1.0f, 0.0f, 0.0f,   1.0f, 0.0f,
         s, y,   -s,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,
         s, y,   -s,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,
         s, y,    s,   1.0f, 0.0f, 0.0f,   0.0f, 1.0f,
         s, y-h,  s,   1.0f, 0.0f, 0.0f,   0.0f, 0.0f,
        
        // BOTTOM FACE - Normal pointing down (usually not visible)
        -s, y-h,  s,   0.0f, -1.0f, 0.0f,  0.0f, 0.0f,
        -s, y-h, -s,   0.0f, -1.0f, 0.0f,  0.0f, 1.0f,
         s, y-h, -s,   0.0f, -1.0f, 0.0f,  1.0f, 1.0f,
         s, y-h, -s,   0.0f, -1.0f, 0.0f,  1.0f, 1.0f,
         s, y-h,  s,   0.0f, -1.0f, 0.0f,  1.0f, 0.0f,
        -s, y-h,  s,   0.0f, -1.0f, 0.0f,  0.0f, 0.0f,
    };

    glGenVertexArrays(1, &m_groundVAO);
    glGenBuffers(1, &m_groundVBO);

    glBindVertexArray(m_groundVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_groundVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    // Normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    // TexCoord
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    // Tangent (dummy for ground)
    glDisableVertexAttribArray(3);
    glVertexAttrib4f(3, 1.0f, 0.0f, 0.0f, 1.0f);

    glBindVertexArray(0);
}

bool MultiScene::addObject(const std::string& modelPath, glm::vec3 position, glm::vec3 rotation, float scale) {
    LoadedObject obj;
    obj.scene = std::make_unique<Scene>();
    
    if (!obj.scene->load(modelPath)) {
        std::cerr << "Failed to load model: " << modelPath << std::endl;
        return false;
    }
    
    obj.position = position;
    obj.rotation = rotation;
    obj.scale = scale;
    obj.localCenter = obj.scene->getCenter();
    obj.localRadius = obj.scene->getRadius();
    
    std::cout << "Loaded: " << modelPath << std::endl;
    std::cout << "  Local Center: " << obj.localCenter.x << ", " << obj.localCenter.y << ", " << obj.localCenter.z << std::endl;
    std::cout << "  Local Radius: " << obj.localRadius << std::endl;
    std::cout << "  Placed at: " << position.x << ", " << position.y << ", " << position.z << std::endl;
    
    m_objects.push_back(std::move(obj));
    updateBounds();
    
    return true;
}

void MultiScene::updateBounds() {
    m_minBounds = glm::vec3(std::numeric_limits<float>::max());
    m_maxBounds = glm::vec3(std::numeric_limits<float>::lowest());
    
    // Include ground plane
    if (m_hasGround) {
        m_minBounds = glm::min(m_minBounds, glm::vec3(-m_groundSize, m_groundY, -m_groundSize));
        m_maxBounds = glm::max(m_maxBounds, glm::vec3(m_groundSize, m_groundY, m_groundSize));
    }
    
    // Include all objects
    for (const auto& obj : m_objects) {
        // Approximate world-space bounds
        float worldRadius = obj.localRadius * obj.scale;
        glm::vec3 worldCenter = obj.position;
        
        m_minBounds = glm::min(m_minBounds, worldCenter - glm::vec3(worldRadius));
        m_maxBounds = glm::max(m_maxBounds, worldCenter + glm::vec3(worldRadius));
    }
}

glm::vec3 MultiScene::getCenter() const {
    return (m_minBounds + m_maxBounds) * 0.5f;
}

float MultiScene::getRadius() const {
    return glm::length(m_maxBounds - m_minBounds) * 0.5f;
}

void MultiScene::draw(unsigned int shaderProgram, unsigned int bgShaderProgram) {
    glUseProgram(shaderProgram);
    
    // Draw ground plane
    if (m_hasGround && m_groundVAO) {
        // Disable culling for ground plane (visible from both sides)
        glDisable(GL_CULL_FACE);
        
        glm::mat4 groundModel = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(groundModel));
        
        // Set ground material (green, non-metallic)
        glUniform1i(glGetUniformLocation(shaderProgram, "hasAlbedoMap"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "hasNormalMap"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "hasMetallicRoughnessMap"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "hasEmissiveMap"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "hasAoMap"), 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "isGroundPlane"), 1);
        
        glUniform3f(glGetUniformLocation(shaderProgram, "uAlbedo"), 0.2f, 0.6f, 0.2f);
        glUniform1f(glGetUniformLocation(shaderProgram, "uMetallic"), 0.0f);
        glUniform1f(glGetUniformLocation(shaderProgram, "uRoughness"), 0.9f);
        
        glBindVertexArray(m_groundVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);  // 6 faces × 6 vertices
        glBindVertexArray(0);
        
        glUniform1i(glGetUniformLocation(shaderProgram, "isGroundPlane"), 0);
        
        // Re-enable culling for other objects
        glEnable(GL_CULL_FACE);
    }
    
    // Draw all objects
    for (auto& obj : m_objects) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, obj.position);
        model = glm::rotate(model, glm::radians(obj.rotation.x), glm::vec3(1, 0, 0));
        model = glm::rotate(model, glm::radians(obj.rotation.y), glm::vec3(0, 1, 0));
        model = glm::rotate(model, glm::radians(obj.rotation.z), glm::vec3(0, 0, 1));
        model = glm::scale(model, glm::vec3(obj.scale));
        
        // Offset by local center so object sits at the specified position
        model = glm::translate(model, -obj.localCenter);
        
        obj.scene->draw(shaderProgram, model);
    }
}

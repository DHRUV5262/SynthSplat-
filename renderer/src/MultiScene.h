#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "Scene.h"

struct SceneObject {
    std::string modelPath;
    glm::vec3 position;
    glm::vec3 rotation; // Euler angles in degrees
    float scale;
};

class MultiScene {
public:
    MultiScene();
    ~MultiScene();

    void addGroundPlane(float size, float y = 0.0f);
    bool addObject(const std::string& modelPath, glm::vec3 position, glm::vec3 rotation = glm::vec3(0), float scale = 1.0f);
    void draw(unsigned int shaderProgram, unsigned int bgShaderProgram);
    
    glm::vec3 getCenter() const;
    float getRadius() const;

private:
    struct LoadedObject {
        std::unique_ptr<Scene> scene;
        glm::vec3 position;
        glm::vec3 rotation;
        float scale;
        glm::vec3 localCenter;
        float localRadius;
    };

    std::vector<LoadedObject> m_objects;
    
    // Ground plane
    bool m_hasGround = false;
    unsigned int m_groundVAO = 0;
    unsigned int m_groundVBO = 0;
    float m_groundSize = 10.0f;
    float m_groundY = 0.0f;
    
    void initGroundPlane();

    // Scene bounds
    glm::vec3 m_minBounds;
    glm::vec3 m_maxBounds;
    void updateBounds();
};

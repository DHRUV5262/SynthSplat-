#pragma once

#include <string>
#include <vector>
#include <map>
#include <glm/glm.hpp>
#include <tiny_gltf.h>

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
    glm::vec4 Tangent;
};

struct Texture {
    unsigned int id;
    std::string type;
};

struct Material {
    // PBR textures
    int albedoIndex = -1;
    int normalIndex = -1;
    int metallicRoughnessIndex = -1;
    int emissiveIndex = -1;
    int aoIndex = -1;

    // Factors
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    glm::vec3 emissiveFactor = glm::vec3(0.0f);
};

struct Primitive {
    unsigned int VAO, VBO, EBO;
    int indexCount;
    int indexComponentType; // GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, or GL_UNSIGNED_INT
    int materialIndex;
};

struct Mesh {
    std::vector<Primitive> primitives;
};

class Scene {
public:
    Scene();
    ~Scene();

    bool load(const std::string& path);
    void draw(unsigned int shaderProgram, const glm::mat4& modelMatrix);

private:
    void processNode(const tinygltf::Node& node, const tinygltf::Model& model, const glm::mat4& parentTransform, unsigned int shaderProgram);
    void calculateBounds(const tinygltf::Node& node, const tinygltf::Model& model, const glm::mat4& parentTransform);
    void processMesh(const tinygltf::Mesh& mesh, const tinygltf::Model& model);
    void loadTextures(const tinygltf::Model& model);
    void loadMaterials(const tinygltf::Model& model);

    std::vector<Mesh> m_meshes;
    std::vector<unsigned int> m_textures; // OpenGL texture IDs
    std::vector<Material> m_materials;
    
    tinygltf::Model m_model;
    
    // Map from gltf texture index to OpenGL texture ID
    std::map<int, unsigned int> m_textureMap;

    glm::vec3 m_minBounds = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3 m_maxBounds = glm::vec3(std::numeric_limits<float>::lowest());

public:
    glm::vec3 getCenter() const { return (m_minBounds + m_maxBounds) * 0.5f; }
    float getRadius() const { return glm::length(m_maxBounds - m_minBounds) * 0.5f; }
};

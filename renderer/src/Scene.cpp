#include "Scene.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

// Define implementations in main.cpp or here. 
// Since Scene.cpp is a good place for tinygltf implementation:
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// We need to disable STB_IMAGE_WRITE in tinygltf if we want to use it separately, 
// but tinygltf only uses stb_image (read). 
// We will include tiny_gltf.h which includes stb_image.h.
#include <tiny_gltf.h>

Scene::Scene() {}

Scene::~Scene() {
    // Cleanup buffers and textures
}

bool Scene::load(const std::string& path) {
    // Clear any previous data
    m_meshes.clear();
    m_textures.clear();
    m_materials.clear();
    m_textureMap.clear();
    
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    // Detect file type by extension
    bool ret = false;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb") {
        ret = loader.LoadBinaryFromFile(&m_model, &err, &warn, path);
    } else {
        ret = loader.LoadASCIIFromFile(&m_model, &err, &warn, path);
    }

    if (!warn.empty()) {
        std::cout << "TinyGLTF Warning: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "TinyGLTF Error: " << err << std::endl;
    }

    if (!ret) {
        std::cout << "Failed to parse glTF" << std::endl;
        return false;
    }

    loadTextures(m_model);
    loadMaterials(m_model);

    // Process all meshes in the model
    for (const auto& mesh : m_model.meshes) {
        processMesh(mesh, m_model);
    }

    // Traverse nodes to calculate world bounds
    m_minBounds = glm::vec3(std::numeric_limits<float>::max());
    m_maxBounds = glm::vec3(std::numeric_limits<float>::lowest());
    
    const tinygltf::Scene& scene = m_model.scenes[m_model.defaultScene > -1 ? m_model.defaultScene : 0];
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        calculateBounds(m_model.nodes[scene.nodes[i]], m_model, glm::mat4(1.0f));
    }
    
    std::cout << "Scene Bounds: Min(" << m_minBounds.x << ", " << m_minBounds.y << ", " << m_minBounds.z << ")" << std::endl;
    std::cout << "Scene Bounds: Max(" << m_maxBounds.x << ", " << m_maxBounds.y << ", " << m_maxBounds.z << ")" << std::endl;

    return true;
}

void Scene::calculateBounds(const tinygltf::Node& node, const tinygltf::Model& model, const glm::mat4& parentTransform) {
    glm::mat4 nodeTransform = parentTransform;
    
    // Apply node transform (same logic as processNode)
    if (node.matrix.size() == 16) {
        std::vector<float> matrixData(node.matrix.begin(), node.matrix.end());
        nodeTransform = nodeTransform * glm::make_mat4(matrixData.data());
    } else {
        if (node.translation.size() == 3) {
            std::vector<float> translationData(node.translation.begin(), node.translation.end());
            nodeTransform = glm::translate(nodeTransform, glm::make_vec3(translationData.data()));
        }
        if (node.rotation.size() == 4) {
             std::vector<float> rotationData(node.rotation.begin(), node.rotation.end());
            glm::quat q = glm::make_quat(rotationData.data());
            nodeTransform = nodeTransform * glm::mat4_cast(q);
        }
        if (node.scale.size() == 3) {
            std::vector<float> scaleData(node.scale.begin(), node.scale.end());
            nodeTransform = glm::scale(nodeTransform, glm::make_vec3(scaleData.data()));
        }
    }

    if (node.mesh > -1 && static_cast<size_t>(node.mesh) < m_meshes.size()) {
        const Mesh& mesh = m_meshes[node.mesh];
        const tinygltf::Mesh& gltfMesh = model.meshes[node.mesh];
        
        for (size_t primIdx = 0; primIdx < mesh.primitives.size() && primIdx < gltfMesh.primitives.size(); ++primIdx) {
             const tinygltf::Primitive& gltfPrim = gltfMesh.primitives[primIdx];
             
             if (gltfPrim.attributes.find("POSITION") != gltfPrim.attributes.end()) {
                const tinygltf::Accessor& accessor = model.accessors[gltfPrim.attributes.at("POSITION")];
                const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
                const float* positionBuffer = reinterpret_cast<const float*>(&(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                
                for (size_t v = 0; v < accessor.count; ++v) {
                    glm::vec4 pos = nodeTransform * glm::vec4(positionBuffer[v*3], positionBuffer[v*3+1], positionBuffer[v*3+2], 1.0f);
                    m_minBounds = glm::min(m_minBounds, glm::vec3(pos));
                    m_maxBounds = glm::max(m_maxBounds, glm::vec3(pos));
                }
             }
        }
    }

    for (int child : node.children) {
        calculateBounds(model.nodes[child], model, nodeTransform);
    }
}

void Scene::loadTextures(const tinygltf::Model& model) {
    for (size_t i = 0; i < model.textures.size(); ++i) {
        const auto& tex = model.textures[i];
        if (tex.source > -1) {
            const auto& image = model.images[tex.source];
            
            unsigned int textureID;
            glGenTextures(1, &textureID);
            glBindTexture(GL_TEXTURE_2D, textureID);
            
            GLenum format;
            if (image.component == 1)
                format = GL_RED;
            else if (image.component == 3)
                format = GL_RGB;
            else if (image.component == 4)
                format = GL_RGBA;
            else {
                std::cout << "Unknown texture format" << std::endl;
                continue;
            }

            glTexImage2D(GL_TEXTURE_2D, 0, format, image.width, image.height, 0, format, image.pixel_type, &image.image[0]);
            glGenerateMipmap(GL_TEXTURE_2D);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            GLfloat maxAniso = 1.0f;
            glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAniso);
            if (maxAniso > 1.0f) {
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY,
                                (std::min)(16.0f, maxAniso));
            }

            m_textureMap[i] = textureID;
        }
    }
}

void Scene::loadMaterials(const tinygltf::Model& model) {
    m_materials.resize(model.materials.size());
    for (size_t i = 0; i < model.materials.size(); ++i) {
        const auto& mat = model.materials[i];
        Material& myMat = m_materials[i];

        // PBR Metallic Roughness
        if (mat.values.find("baseColorTexture") != mat.values.end()) {
            myMat.albedoIndex = mat.values.at("baseColorTexture").TextureIndex();
        }
        if (mat.values.find("metallicRoughnessTexture") != mat.values.end()) {
            myMat.metallicRoughnessIndex = mat.values.at("metallicRoughnessTexture").TextureIndex();
        }
        if (mat.values.find("baseColorFactor") != mat.values.end()) {
            myMat.baseColorFactor = glm::make_vec4(mat.values.at("baseColorFactor").ColorFactor().data());
        }
        if (mat.values.find("metallicFactor") != mat.values.end()) {
            myMat.metallicFactor = (float)mat.values.at("metallicFactor").Factor();
        }
        if (mat.values.find("roughnessFactor") != mat.values.end()) {
            myMat.roughnessFactor = (float)mat.values.at("roughnessFactor").Factor();
        }

        // Additional maps
        if (mat.additionalValues.find("normalTexture") != mat.additionalValues.end()) {
            myMat.normalIndex = mat.additionalValues.at("normalTexture").TextureIndex();
        }
        if (mat.additionalValues.find("emissiveTexture") != mat.additionalValues.end()) {
            myMat.emissiveIndex = mat.additionalValues.at("emissiveTexture").TextureIndex();
        }
        if (mat.additionalValues.find("occlusionTexture") != mat.additionalValues.end()) {
            myMat.aoIndex = mat.additionalValues.at("occlusionTexture").TextureIndex();
        }
        if (mat.additionalValues.find("emissiveFactor") != mat.additionalValues.end()) {
            myMat.emissiveFactor = glm::make_vec3(mat.additionalValues.at("emissiveFactor").ColorFactor().data());
        }
    }
}

void Scene::processMesh(const tinygltf::Mesh& mesh, const tinygltf::Model& model) {
    Mesh myMesh;
    
    for (const auto& primitive : mesh.primitives) {
        Primitive myPrim;
        myPrim.materialIndex = primitive.material;

        glGenVertexArrays(1, &myPrim.VAO);
        glGenBuffers(1, &myPrim.VBO);
        glGenBuffers(1, &myPrim.EBO);

        glBindVertexArray(myPrim.VAO);

        // Indices
        const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
        const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
        const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, myPrim.EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBufferView.byteLength, &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset], GL_STATIC_DRAW);
        myPrim.indexCount = indexAccessor.count;
        myPrim.indexComponentType = indexAccessor.componentType; // Store component type (5121, 5123, 5125)

        // Attributes
        std::vector<Vertex> vertices;
        // We need to iterate over attributes and interleave or use separate VBOs. 
        // For simplicity, let's construct a single interleaved buffer.
        
        const float* positionBuffer = nullptr;
        const float* normalsBuffer = nullptr;
        const float* texCoordsBuffer = nullptr;
        const float* tangentsBuffer = nullptr;
        
        size_t vertexCount = 0;

        if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
            positionBuffer = reinterpret_cast<const float*>(&(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
            vertexCount = accessor.count;
        }
        if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("NORMAL")];
            const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
            normalsBuffer = reinterpret_cast<const float*>(&(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
        }
        if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
            const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
            texCoordsBuffer = reinterpret_cast<const float*>(&(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
        }
        if (primitive.attributes.find("TANGENT") != primitive.attributes.end()) {
            const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("TANGENT")];
            const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
            tangentsBuffer = reinterpret_cast<const float*>(&(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
        }

        for (size_t v = 0; v < vertexCount; ++v) {
            Vertex vert;
            if (positionBuffer) {
                vert.Position = glm::vec3(positionBuffer[v*3], positionBuffer[v*3+1], positionBuffer[v*3+2]);
                
                // Update bounds (simple approximation, ideally should be transformed by node matrix)
                // Note: This bounds calculation is local to mesh, not world space. 
                // For a proper center, we should traverse nodes and apply transforms.
                // But for single object like DamagedHelmet, this is usually fine if no transform on node.
            }
            if (normalsBuffer) {
                vert.Normal = glm::vec3(normalsBuffer[v*3], normalsBuffer[v*3+1], normalsBuffer[v*3+2]);
            }
            if (texCoordsBuffer) {
                vert.TexCoords = glm::vec2(texCoordsBuffer[v*2], texCoordsBuffer[v*2+1]);
            }
            if (tangentsBuffer) {
                vert.Tangent = glm::vec4(tangentsBuffer[v*4], tangentsBuffer[v*4+1], tangentsBuffer[v*4+2], tangentsBuffer[v*4+3]);
            } else {
                vert.Tangent = glm::vec4(0.0f);
            }
            vertices.push_back(vert);
        }

        glBindBuffer(GL_ARRAY_BUFFER, myPrim.VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

        // Vertex Attributes
        // 0: Pos, 1: Normal, 2: TexCoord, 3: Tangent
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));

        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));

        glBindVertexArray(0);
        
        myMesh.primitives.push_back(myPrim);
        
        std::cout << "Loaded Primitive: " << vertices.size() << " vertices, " << indexAccessor.count << " indices." << std::endl;
    }
    
    m_meshes.push_back(myMesh);
}

void Scene::draw(unsigned int shaderProgram, const glm::mat4& modelMatrix) {
    // Traverse nodes to handle transforms
    const tinygltf::Scene& scene = m_model.scenes[m_model.defaultScene > -1 ? m_model.defaultScene : 0];
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        processNode(m_model.nodes[scene.nodes[i]], m_model, modelMatrix, shaderProgram);
    }
}

void Scene::processNode(const tinygltf::Node& node, const tinygltf::Model& model, const glm::mat4& parentTransform, unsigned int shaderProgram) {
    glm::mat4 nodeTransform = parentTransform;
    
    // Apply node transform
    if (node.matrix.size() == 16) {
        // Explicitly cast to float to avoid double/float mismatch with glm::mat4 (which is float by default)
        std::vector<float> matrixData(node.matrix.begin(), node.matrix.end());
        nodeTransform = nodeTransform * glm::make_mat4(matrixData.data());
    } else {
        if (node.translation.size() == 3) {
            std::vector<float> translationData(node.translation.begin(), node.translation.end());
            nodeTransform = glm::translate(nodeTransform, glm::make_vec3(translationData.data()));
        }
        if (node.rotation.size() == 4) {
             std::vector<float> rotationData(node.rotation.begin(), node.rotation.end());
            glm::quat q = glm::make_quat(rotationData.data());
            nodeTransform = nodeTransform * glm::mat4_cast(q);
        }
        if (node.scale.size() == 3) {
            std::vector<float> scaleData(node.scale.begin(), node.scale.end());
            nodeTransform = glm::scale(nodeTransform, glm::make_vec3(scaleData.data()));
        }
    }

    if (node.mesh > -1 && static_cast<size_t>(node.mesh) < m_meshes.size()) {
        // Draw mesh
        // Note: m_meshes index corresponds to model.meshes index because we pushed them in order
        const Mesh& mesh = m_meshes[node.mesh];
        
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(nodeTransform));

        for (const auto& prim : mesh.primitives) {
            // Bind Material
            if (prim.materialIndex > -1 && static_cast<size_t>(prim.materialIndex) < m_materials.size()) {
                const Material& mat = m_materials[prim.materialIndex];
                
                // Helper lambda to bind texture
                auto bindTex = [&](const char* name, int index, int unit) {
                    glActiveTexture(GL_TEXTURE0 + unit);
                    if (index > -1 && m_textureMap.count(index)) {
                        glBindTexture(GL_TEXTURE_2D, m_textureMap[index]);
                        glUniform1i(glGetUniformLocation(shaderProgram, name), unit);
                        return true;
                    }
                    glBindTexture(GL_TEXTURE_2D, 0);
                    return false;
                };

                bool hasAlbedo = bindTex("albedoMap", mat.albedoIndex, 0);
                bool hasNormal = bindTex("normalMap", mat.normalIndex, 1);
                bool hasMetallic = bindTex("metallicRoughnessMap", mat.metallicRoughnessIndex, 2);
                bool hasEmissive = bindTex("emissiveMap", mat.emissiveIndex, 3);
                bool hasAo = bindTex("aoMap", mat.aoIndex, 4);

                glUniform1i(glGetUniformLocation(shaderProgram, "hasAlbedoMap"), hasAlbedo);
                glUniform1i(glGetUniformLocation(shaderProgram, "hasNormalMap"), hasNormal);
                glUniform1i(glGetUniformLocation(shaderProgram, "hasMetallicRoughnessMap"), hasMetallic);
                glUniform1i(glGetUniformLocation(shaderProgram, "hasEmissiveMap"), hasEmissive);
                glUniform1i(glGetUniformLocation(shaderProgram, "hasAoMap"), hasAo);
                
                glUniform3fv(glGetUniformLocation(shaderProgram, "uAlbedo"), 1, glm::value_ptr(mat.baseColorFactor));
                glUniform1f(glGetUniformLocation(shaderProgram, "uMetallic"), mat.metallicFactor);
                glUniform1f(glGetUniformLocation(shaderProgram, "uRoughness"), mat.roughnessFactor);
            } else {
                // Default material
                glUniform1i(glGetUniformLocation(shaderProgram, "hasAlbedoMap"), 0);
                glUniform1i(glGetUniformLocation(shaderProgram, "hasNormalMap"), 0);
                glUniform1i(glGetUniformLocation(shaderProgram, "hasMetallicRoughnessMap"), 0);
                glUniform1i(glGetUniformLocation(shaderProgram, "hasEmissiveMap"), 0);
                glUniform1i(glGetUniformLocation(shaderProgram, "hasAoMap"), 0);
                
                glUniform3f(glGetUniformLocation(shaderProgram, "uAlbedo"), 1.0f, 1.0f, 1.0f);
                glUniform1f(glGetUniformLocation(shaderProgram, "uMetallic"), 0.0f);
                glUniform1f(glGetUniformLocation(shaderProgram, "uRoughness"), 0.5f);
            }

            glBindVertexArray(prim.VAO);
            glDrawElements(GL_TRIANGLES, prim.indexCount, prim.indexComponentType, 0);
            glBindVertexArray(0);
        }
    }

    for (int child : node.children) {
        processNode(model.nodes[child], model, nodeTransform, shaderProgram);
    }
}

#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <nlohmann/json.hpp>

struct CameraData {
    int frame_id;
    std::string filename;
    glm::vec3 position;
    glm::mat4 view_matrix;
    glm::mat4 projection_matrix;
    int width;
    int height;
    float fov_degrees;

    nlohmann::json toJson() const;
};

class Camera {
public:
    Camera(glm::vec3 position, glm::vec3 target, glm::vec3 up, float fov, float aspect, float nearPlane, float farPlane);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::vec3 getPosition() const;

    static std::vector<Camera> generateOrbitPath(int count, float radius, glm::vec3 target);

private:
    glm::vec3 m_position;
    glm::vec3 m_target;
    glm::vec3 m_up;
    float m_fov;
    float m_aspect;
    float m_near;
    float m_far;
};

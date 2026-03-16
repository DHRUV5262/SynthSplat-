#include "Camera.h"
#include <cmath>
#include <vector>
#include <iostream>

nlohmann::json CameraData::toJson() const {
    nlohmann::json j;
    j["frame_id"] = frame_id;
    j["filename"] = filename;
    j["position"] = {position.x, position.y, position.z};
    
    // Convert matrices to row-major arrays (GLM is column-major by default)
    // But we want to store them in a way that is easy to read. 
    // Usually, JSON arrays are just linear. Let's store as row-major 4x4 array of arrays.
    auto matToVec = [](const glm::mat4& m) {
        std::vector<std::vector<float>> rows(4, std::vector<float>(4));
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 4; ++k) {
                rows[i][k] = m[k][i]; // Transpose because GLM is column-major
            }
        }
        return rows;
    };

    j["view_matrix"] = matToVec(view_matrix);
    j["projection_matrix"] = matToVec(projection_matrix);
    j["width"] = width;
    j["height"] = height;
    j["fov_degrees"] = fov_degrees;
    return j;
}

Camera::Camera(glm::vec3 position, glm::vec3 target, glm::vec3 up, float fov, float aspect, float nearPlane, float farPlane)
    : m_position(position), m_target(target), m_up(up), m_fov(fov), m_aspect(aspect), m_near(nearPlane), m_far(farPlane) {}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(m_position, m_target, m_up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(m_fov), m_aspect, m_near, m_far);
}

glm::vec3 Camera::getPosition() const {
    return m_position;
}

std::vector<Camera> Camera::generateOrbitPath(int count, float radius, glm::vec3 target) {
    std::vector<Camera> cameras;
    
    // 4 rings: 10, 30, 50, 70 degrees elevation
    std::vector<float> elevations = {10.0f, 30.0f, 50.0f, 70.0f};
    
    // Distribute cameras per ring based on circumference (approx cos(elevation))
    // Total 100 cameras.
    // Weights: cos(10)=0.98, cos(30)=0.86, cos(50)=0.64, cos(70)=0.34. Sum = 2.82.
    // Counts: 
    // Ring 1: 100 * 0.98 / 2.82 = 34.7 -> 35
    // Ring 2: 100 * 0.86 / 2.82 = 30.5 -> 30
    // Ring 3: 100 * 0.64 / 2.82 = 22.7 -> 23
    // Ring 4: 100 * 0.34 / 2.82 = 12.0 -> 12
    // Sum = 100. Perfect.
    
    std::vector<int> counts = {35, 30, 23, 12};

    for (size_t i = 0; i < elevations.size(); ++i) {
        float elevation = glm::radians(elevations[i]);
        int ringCount = counts[i];
        
        for (int j = 0; j < ringCount; ++j) {
            float azimuth = glm::radians(360.0f * (float)j / (float)ringCount);
            
            // Spherical to Cartesian
            // y is up. 
            // x = r * cos(el) * cos(az)
            // y = r * sin(el)
            // z = r * cos(el) * sin(az)
            
            float x = radius * std::cos(elevation) * std::cos(azimuth);
            float y = radius * std::sin(elevation);
            float z = radius * std::cos(elevation) * std::sin(azimuth);
            
            glm::vec3 pos(x, y, z);
            
            // Create camera
            // FOV 60, aspect 1.0, near 0.1, far 100.0
            cameras.emplace_back(pos, target, glm::vec3(0, 1, 0), 60.0f, 1.0f, 0.1f, 100.0f);
        }
    }
    
    return cameras;
}

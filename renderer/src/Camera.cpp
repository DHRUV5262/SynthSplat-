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
    j["scene_center"] = {scene_center.x, scene_center.y, scene_center.z};
    j["scene_radius"] = scene_radius;
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

std::vector<Camera> Camera::generateOrbitPath(int count, float radius, glm::vec3 target, float aspect) {
    std::vector<Camera> cameras;
    
    // For 300 cameras: use more elevation rings packed tighter
    // 8 rings: 5, 15, 25, 35, 45, 55, 65, 75 degrees elevation
    std::vector<float> elevations = {5.0f, 15.0f, 25.0f, 35.0f, 45.0f, 55.0f, 65.0f, 75.0f};
    
    // Calculate weights based on cos(elevation) - cameras at lower elevations cover more circumference
    std::vector<float> weights;
    float totalWeight = 0.0f;
    for (float elev : elevations) {
        float w = std::cos(glm::radians(elev));
        weights.push_back(w);
        totalWeight += w;
    }
    
    // Distribute 300 cameras proportionally
    std::vector<int> counts;
    int assignedTotal = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        int ringCount = static_cast<int>(std::round(count * weights[i] / totalWeight));
        counts.push_back(ringCount);
        assignedTotal += ringCount;
    }
    
    // Adjust for rounding errors to hit exactly 'count' cameras
    int diff = count - assignedTotal;
    if (diff != 0) {
        // Add/subtract from the ring with the most cameras
        size_t maxIdx = 0;
        for (size_t i = 1; i < counts.size(); ++i) {
            if (counts[i] > counts[maxIdx]) maxIdx = i;
        }
        counts[maxIdx] += diff;
    }
    
    std::cout << "Camera distribution per ring: ";
    for (size_t i = 0; i < elevations.size(); ++i) {
        std::cout << elevations[i] << "deg=" << counts[i] << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < elevations.size(); ++i) {
        float elevation = glm::radians(elevations[i]);
        int ringCount = counts[i];
        
        for (int j = 0; j < ringCount; ++j) {
            float azimuth = glm::radians(360.0f * static_cast<float>(j) / static_cast<float>(ringCount));
            
            // Spherical to Cartesian (y is up)
            float x = radius * std::cos(elevation) * std::cos(azimuth);
            float y = radius * std::sin(elevation);
            float z = radius * std::cos(elevation) * std::sin(azimuth);
            
            glm::vec3 pos(x + target.x, y + target.y, z + target.z);
            
            cameras.emplace_back(pos, target, glm::vec3(0, 1, 0), 60.0f, aspect, 0.1f, 1000.0f);
        }
    }
    
    return cameras;
}

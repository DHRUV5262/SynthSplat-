#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <filesystem>

#include "Renderer.h"
#include "Scene.h"
#include "Camera.h"

namespace fs = std::filesystem;

int main() {
    const int WIDTH = 800;
    const int HEIGHT = 800;
    const std::string ASSET_PATH = "assets/DamagedHelmet.glb";
    const std::string OUTPUT_DIR = "output";

    // Ensure output directory exists
    if (!fs::exists(OUTPUT_DIR)) {
        fs::create_directory(OUTPUT_DIR);
    }

    // Initialize Renderer
    Renderer renderer(WIDTH, HEIGHT);
    if (!renderer.init()) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return -1;
    }

    // Load Scene
    Scene scene;
    if (!scene.load(ASSET_PATH)) {
        std::cerr << "Failed to load scene: " << ASSET_PATH << std::endl;
        // Check if file exists
        if (!fs::exists(ASSET_PATH)) {
            std::cerr << "File not found: " << fs::absolute(ASSET_PATH) << std::endl;
        }
        return -1;
    }

    // Generate Cameras
    // Calculate scene center and radius
    glm::vec3 center = scene.getCenter();
    float radius = scene.getRadius();
    
    std::cout << "Scene Center: " << center.x << ", " << center.y << ", " << center.z << std::endl;
    std::cout << "Scene Radius: " << radius << std::endl;

    // Adjust orbit radius to fit object (e.g. 2.5x radius)
    float orbitRadius = std::max(3.0f, radius * 2.5f);
    std::cout << "Orbit Radius: " << orbitRadius << std::endl;

    std::vector<Camera> cameras = Camera::generateOrbitPath(100, orbitRadius, center);

    std::vector<CameraData> cameraDataList;

    std::cout << "Starting render of " << cameras.size() << " frames..." << std::endl;

    for (int i = 0; i < cameras.size(); ++i) {
        std::stringstream ss;
        ss << "frame_" << std::setw(4) << std::setfill('0') << i << ".png";
        std::string filename = ss.str();
        std::string fullPath = OUTPUT_DIR + "/" + filename;

        renderer.render(scene, cameras[i], fullPath, i);

        // Store camera data
        CameraData data;
        data.frame_id = i;
        data.filename = filename;
        data.position = cameras[i].getPosition();
        data.view_matrix = cameras[i].getViewMatrix();
        data.projection_matrix = cameras[i].getProjectionMatrix();
        data.width = WIDTH;
        data.height = HEIGHT;
        data.fov_degrees = 60.0f; // As defined in Camera.cpp

        cameraDataList.push_back(data);

        if (i % 10 == 0) {
            std::cout << "Rendered frame " << i << "/" << cameras.size() << std::endl;
        }
    }

    // Save cameras.json
    nlohmann::json j;
    for (const auto& data : cameraDataList) {
        j.push_back(data.toJson());
    }

    std::ofstream jsonFile(OUTPUT_DIR + "/cameras.json");
    jsonFile << j.dump(4);
    jsonFile.close();

    std::cout << "Done! Output saved to " << OUTPUT_DIR << std::endl;

    return 0;
}

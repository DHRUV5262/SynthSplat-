#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#include "Renderer.h"
#include "Scene.h"
#include "MultiScene.h"
#include "Camera.h"

namespace fs = std::filesystem;

// Set to true to use multi-object tabletop scene, false for single model
#define USE_MULTI_SCENE true

// Visual Studio runs the exe from build/bin/Debug — relative "assets/..." points at the wrong folder.
// Walk up from the .exe directory until we find relativePath (e.g. build/assets/foo.glb).
static std::string resolveDataPath(const std::string& relativePath) {
    std::vector<fs::path> candidates;
#ifdef _WIN32
    wchar_t wbuf[MAX_PATH];
    if (GetModuleFileNameW(nullptr, wbuf, MAX_PATH) != 0u) {
        fs::path dir = fs::path(wbuf).parent_path();
        for (int i = 0; i < 8; ++i) {
            candidates.push_back(dir / relativePath);
            fs::path parent = dir.parent_path();
            if (parent == dir) break;
            dir = parent;
        }
    }
#endif
    candidates.push_back(fs::current_path() / relativePath);

    for (const auto& p : candidates) {
        std::error_code ec;
        if (fs::exists(p, ec)) {
            std::error_code ec2;
            fs::path canon = fs::weakly_canonical(p, ec2);
            return ec2 ? p.string() : canon.string();
        }
    }
    return relativePath;
}

int main() {
    // Full HD — more detail for 3DGS; larger PNGs, slower render & training vs 1024²
    const int WIDTH = 1920;
    const int HEIGHT = 1080;
    const float ASPECT = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
    const std::string OUTPUT_DIR = "../output";

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

    glm::vec3 center;
    float radius;
    float orbitRadius;
    std::vector<Camera> cameras;
    std::vector<CameraData> cameraDataList;

#if USE_MULTI_SCENE
    // Low-poly cafe — put the file in renderer/assets/ and copy to build/assets/
    const std::string CAFE_ASSET = "assets/lowpoly_cafe.glb";
    const std::string cafePath = resolveDataPath(CAFE_ASSET);

    MultiScene scene;

    if (!scene.addObject(cafePath,
                         glm::vec3(0.0f, 0.0f, 0.0f),
                         glm::vec3(0.0f, 0.0f, 0.0f),
                         1.0f)) {
        std::cerr << "Failed to load cafe model. Tried: " << cafePath << std::endl;
        std::cerr << "Put lowpoly_cafe.glb in renderer/assets/, rebuild (copies to build/assets/), or edit CAFE_ASSET in main.cpp" << std::endl;
        return -1;
    }

    // Goku on the patio — scale ~1.8 is person-sized vs stools/tables for this GLB; raise more if still tiny.
    // Tweak gokuPos / gokuScale / gokuRotY if he floats, clips the ground, or faces the wrong way.
    const std::string GOKU_ASSET = "assets/gokucharacter3dmodel.glb";
    const std::string gokuPath = resolveDataPath(GOKU_ASSET);
    if (fs::exists(gokuPath)) {
        // Inside the fenced patio, near front-left tables (not behind the side fence).
        const glm::vec3 gokuPos(-0.85f, -0.2f, 1.15f);
        const float gokuRotY = -90.0f;
        const float gokuScale = 2.4f;
        if (!scene.addObject(gokuPath, gokuPos, glm::vec3(0.0f, gokuRotY, 0.0f), gokuScale)) {
            std::cerr << "Warning: could not load Goku model: " << gokuPath << std::endl;
        }
    } else {
        std::cout << "Optional Goku GLB not found (skipped): " << gokuPath << std::endl;
        std::cout << "  Add gokucharacter3dmodel.glb under renderer/assets/ and rebuild to place the character in the scene." << std::endl;
    }

    center = scene.getCenter();
    radius = scene.getRadius();
    
    std::cout << "\n=== Cafe Scene ===" << std::endl;
    std::cout << "Scene Center: " << center.x << ", " << center.y << ", " << center.z << std::endl;
    std::cout << "Scene Radius: " << radius << std::endl;

    // Camera orbits around the city
    glm::vec3 lookAt = center;
    orbitRadius = std::max(8.0f, radius * 0.8f);
    std::cout << "Orbit Radius: " << orbitRadius << std::endl;
    
    cameras = Camera::generateOrbitPath(300, orbitRadius, lookAt, ASPECT);

#else
    // ===== SINGLE MODEL SCENE =====
    const std::string ASSET_PATH = resolveDataPath("assets/DamagedHelmet.glb");
    
    Scene scene;
    if (!scene.load(ASSET_PATH)) {
        std::cerr << "Failed to load scene: " << ASSET_PATH << std::endl;
        if (!fs::exists(ASSET_PATH)) {
            std::cerr << "File not found: " << fs::absolute(ASSET_PATH) << std::endl;
        }
        return -1;
    }

    center = scene.getCenter();
    radius = scene.getRadius();
    
    std::cout << "Scene Center: " << center.x << ", " << center.y << ", " << center.z << std::endl;
    std::cout << "Scene Radius: " << radius << std::endl;

    orbitRadius = std::max(3.0f, radius * 2.5f);
    std::cout << "Orbit Radius: " << orbitRadius << std::endl;

    cameras = Camera::generateOrbitPath(300, orbitRadius, center, ASPECT);
#endif

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
        data.scene_center = center;
        data.scene_radius = radius;

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

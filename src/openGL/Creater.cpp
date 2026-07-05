#include "openGL/Creater.h"

#include <filesystem>
#include <iostream>

#include "openGL/VertexBufferLayout.h"

namespace opengl {

Creater::Creater(const mhd::Configs& configs,
                 const std::filesystem::path& resPath)
    : _showGraphics(configs._showGraphics),
      _currentTextureIndex(0),
      _lastTime(0) {
    std::filesystem::path filePath = resPath / "shaders/basic.shader";

    if (_showGraphics && exists(filePath)) {
        Create(configs._windowWidth, configs._windowHeight);

        /* Make the window's context current */
        glfwMakeContextCurrent(_window);
        glfwSwapInterval(1);

        GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        GLCall(glEnable(GL_BLEND));

        if (glewInit() != GLEW_OK)
            std::cout << "Error!" << std::endl;

        _vb = new VertexBuffer(_positions.data(),
                               static_cast<unsigned int>(_positions.size()));
        _va = new VertexArray();
        _ib = new IndexBuffer(_indices.data(),
                              static_cast<unsigned int>(_indices.size()));

        CreateWindowData();

        _shader = new Shader(filePath.string());
        _textures = new Textures(configs._texturesCount);
        _renderer = new Renderer();
    } else {
        if (!exists(filePath)) {
            std::cout << "Shader file not exists" << std::endl;
        }

        _window = nullptr;
        _shader = nullptr;
        _textures = nullptr;
        _renderer = nullptr;

        _vb = nullptr;
        _va = nullptr;
        _ib = nullptr;
    }
}

Creater::~Creater() {
    if (_showGraphics) {
        delete _renderer;
        delete _textures;
        delete _shader;

        delete _ib;
        delete _va;
        delete _vb;

        glfwTerminate();
    }
}

int Creater::Create(unsigned int width, unsigned int height) {
    /* Initialize the library */
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    _window = glfwCreateWindow(width, height, "2D plasma", NULL, NULL);
    if (!_window) {
        glfwTerminate();
        return -1;
    }

    return 0;
}

void Creater::CreateWindowData() {
    VertexBufferLayout layout;
    layout.Push<float>(2);
    layout.Push<float>(2);
    _va->AddBuffer(*_vb, layout);
    _vb->Unbing();
    _va->Unbind();
    _ib->Unbing();
}

void Creater::PrepareToRun() {
    if (_showGraphics) {
        _shader->Bind();
        _textures->Bind(0);
        _shader->SetUniform1i("u_Texture", 0);
        _lastTime = glfwGetTime();
    }
}

void Creater::UpdateFPSCounter() {
    if (_showGraphics) {
        double currentTime = glfwGetTime();
        double delta = currentTime - _lastTime;
        _frames++;

        if (delta >= 1.0) {
            double fps = double(_frames) / delta;

            std::string title = "FPS: " + std::to_string(fps);
            glfwSetWindowTitle(_window, title.c_str());

            _frames = 0;
            _lastTime = currentTime;
        }
    }
}

void Creater::AddTexture(const unsigned char* buffer, unsigned int width,
                         unsigned int height) {
    if (_showGraphics) {
        _textures->LoadTexture(buffer, width, height,
                               _currentTextureIndex % 32);
    }
}

bool Creater::ShouldOpen() {
    if (_showGraphics)
        return !glfwWindowShouldClose(_window);
    return true;
}

void Creater::Render(bool shouldUpdateTexture) {
    if (_showGraphics) {
        _renderer->Clear();
        _shader->Bind();

        if (shouldUpdateTexture) {
            _currentTextureIndex = _currentTextureIndex % 32;

            _textures->Bind(_currentTextureIndex);
            _shader->SetUniform1i("u_Texture", _currentTextureIndex);
            _currentTextureIndex++;
        }

        _renderer->Draw(*_va, *_ib, *_shader);

        UpdateFPSCounter();
    }
}

void Creater::WindowUpdate() {
    if (_showGraphics) {
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
}
}  // namespace opengl

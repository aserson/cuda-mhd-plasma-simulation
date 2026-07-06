#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>

#include <QApplication>
#include <QMessageBox>
#include <QThread>

#include <yaml-cpp/yaml.h>

#include "Configs.h"
#include "MainWindow.h"
#include "Writer.h"
#include "cuda/Solver.cuh"

// The configs folder is optional in the Qt build: it only serves as the
// starting directory of the Load config dialog
static std::filesystem::path FindConfPath(const std::filesystem::path& exePath) {
    for (auto path = exePath; ; path = path.parent_path()) {
        if (exists(path / "configs"))
            return path / "configs";
        if (path == path.root_path())
            return std::filesystem::path("");
    }
}

static std::filesystem::path FindResPath(std::filesystem::path& exePath) {
    if (exists(exePath / "res")) {
        return exePath / "res";
    }

    if (exists(exePath.parent_path() / "res")) {
        exePath = exePath.parent_path();
        return exePath / "res";
    }

    if (exists(exePath.parent_path().parent_path() / "res")) {
        exePath = exePath.parent_path().parent_path();
        return exePath / "res";
    }

    return std::filesystem::path("");
}

static std::filesystem::path CreateOutputDir(
    const mhd::Configs& configs, const std::filesystem::path& parantPath) {
    if (exists(parantPath) == false)
        create_directories(parantPath);

    tm timeInfo;
    time_t rawTime;
    time(&rawTime);

    char currenOutputDir[80] = "000000_000000";

#if defined(_MSC_VER)
    if (localtime_s(&timeInfo, &rawTime) == 0) {
        strftime(currenOutputDir, sizeof(currenOutputDir), "%Y%m%d_%H%M%S",
                 &timeInfo);
    }
#else
    if (localtime_r(&rawTime, &timeInfo) != nullptr) {
        strftime(currenOutputDir, sizeof(currenOutputDir), "%Y%m%d_%H%M%S",
                 &timeInfo);
    }
#endif

    auto outputPath = parantPath / std::string(currenOutputDir);

    create_directory(outputPath);

    if (configs._saveData) {
        if (configs._saveVorticity)
            create_directory(outputPath / "vorticity");
        if (configs._saveCurrent)
            create_directory(outputPath / "current");
        if (configs._saveStream)
            create_directory(outputPath / "stream");
        if (configs._savePotential)
            create_directory(outputPath / "potential");
    }

    return outputPath;
}

// One simulation session in the real type T (double or float); returns
// when the run finishes or the user stops, restarts or closes the window
template <typename T>
static void runSession(const mhd::Configs& configs, qtui::MainWindow& window,
                       QApplication& app,
                       const std::filesystem::path& outputPath,
                       const std::filesystem::path& resPath) {
    mhd::Writer writer(outputPath, configs, resPath);
    mhd::Solver<T> solver(configs);

    solver.fillNormally(static_cast<unsigned long>(std::time(nullptr)));
    solver.saveOldFields();
    solver.updateTimeStep();

    auto publish = [&]() {
        window.showFrame(writer.getPixels().data(), writer.getPixelsLength());
        window.showStatus(solver._currents);
    };

    if (writer.saveData(solver))
        publish();

    while (solver.shouldContinue() && !window.isClosed() &&
           !window.stopRequested() && !window.startPending()) {
        app.processEvents();

        if (window.isPaused()) {
            QThread::msleep(30);
            continue;
        }

        solver.step();

        solver.updateTimeStep();
        solver.timeStep();

        if (writer.saveData(solver))
            publish();
    }
}

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    std::cout << "This is two-dimensional magnetohydrodynamic simulation"
              << std::endl
              << std::endl;

    std::filesystem::path projectPath =
        std::filesystem::canonical(argv[0]).parent_path();

    std::filesystem::path resPath = FindResPath(projectPath);
    if (resPath.string() == "") {
        std::cout << "Resources folder not exists" << std::endl;
        return -1;
    }

    qtui::MainWindow window(resPath, FindConfPath(projectPath));
    window.show();

    while (!window.isClosed()) {
        app.processEvents();

        if (!window.takeStartRequest()) {
            QThread::msleep(15);
            continue;
        }

        try {
            mhd::Configs configs(YAML::Load(window.configYamlText()));

            std::filesystem::path outputPath;
            if (configs._saveData) {
                outputPath =
                    CreateOutputDir(configs, projectPath / "outputs");
                configs.ParametersSave(outputPath);
                std::cout << "Output directory: "
                          << outputPath.filename().string() << std::endl;
            }

            std::cout << configs.ParametersPrint();

            window.setRunning(true);

            if (configs._singlePrecision) {
                runSession<float>(configs, window, app, outputPath, resPath);
            } else {
                runSession<double>(configs, window, app, outputPath, resPath);
            }
        } catch (const std::exception& error) {
            QMessageBox::warning(&window, "Configuration error",
                                 error.what());
        }

        window.setRunning(false);
    }

    return 0;
}

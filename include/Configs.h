#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>

#if defined(_MSC_VER) && !defined(M_PI)
#define M_PI 3.141592653589793238462643
#endif

namespace mhd {

struct Currents {
    double time = 0;
    double timeStep = 0;
    unsigned int stepNumber = 0;

    double kineticEnergy = 0;
    double magneticEnergy = 0;

    double maxVelocityField = 0;
    double maxMagneticField = 0;
};

class Configs {
private:
    struct DefaultConfigs {
        // Simulation Parameters
        static const unsigned int defaultGridLength = 1024;
        static constexpr char defaultPrecision[] = "double";
        static constexpr double defaultTime = 5.;
        static constexpr double defaultDealCoef = 2. / 3.;
        static constexpr double defaultMaxTimeStep = 0.01;
        static constexpr double defaultCFL = 0.2;
        static const unsigned int defaultTimeStepUpdateInterval = 10;

        // Equation Coefficients
        static constexpr double defaultNu = 1.e-4;
        static constexpr double defaultEta = 1.e-4;
        static constexpr double defaultBeta = 0.;

        // Forcing Coefficients
        static constexpr double defaultKineticForcing = 0.;
        static constexpr double defaultMagneticForcing = 0.;
        static const unsigned int defaultForcingWN = 10;
        static constexpr double defaultForcingBand = 1.5;

        // Initial Condition Coefficients
        static constexpr double defaultKineticEnergy = 0.5;
        static constexpr double defaultMagneticEnergy = 0.5;
        static const unsigned int defaultAverageWN = 10;

        // Output Parameters
        static constexpr double defaultOutputStep = 0.01;
        static constexpr double defaultOutputStart = 0.0;
        static const unsigned int defaultMaxOutputs = 1000;

        // Kernel Run Parameters
        static const unsigned int defaultDimBlockX = 32;
        static const unsigned int defaultDimBlockY = 16;
        static const unsigned int defaultSharedLength = 128;

        // Writer Settings
        static const bool defaultSaveData = false;
        static const bool defaultSavePNG = false;
        static const bool defaultSaveVorticity = true;
        static const bool defaultSaveCurrent = false;
        static const bool defaultSaveStream = false;
        static const bool defaultSavePotential = false;

        // Graphics Settings
        static const bool defaultShowGraphics = true;
        static const unsigned int defaultTexturesCount = 32;
        static const unsigned int defaultWindowWidth = 1024;
        static const unsigned int defaultWindowHeight = 1024;
        static constexpr char defaultColorMap[] = "winter";
    };

    std::filesystem::path _filePath;
    YAML::Node _config;

    // Simulation Parameters
    unsigned int getGridLength() const;
    bool getSinglePrecision() const;
    double getDealCoef() const;
    double getTime() const;
    double getMaxTimeStep() const;
    double getCFL() const;
    unsigned int getTimeStepUpdateInterval() const;

    // Equation Coefficients
    double getNu();
    double getEta();
    double getBeta();

    // Forcing Coefficients
    double getKineticForcing();
    double getMagneticForcing();
    unsigned int getForcingWN();
    double getForcingBand();

    // Initial Condition Coefficients
    double getKineticEnergy();
    double getMagneticEnergy();
    unsigned int getAverageWN();

    // OutputParameters
    double getOutputStep();
    double getOutputStart();
    double getOutputStop();

    // KernelRunParameters
    unsigned int getDimBlockX();
    unsigned int getDimBlockY();
    unsigned int getSharedLength();

    // Writer Settings
    bool getSaveData();
    bool getSavePNG();
    bool getSaveVorticity();
    bool getSaveCurrent();
    bool getSaveStream();
    bool getSavePotential();

    // Graphics Settings
    bool getShowGraphics();
    unsigned int getTexturesCount();
    unsigned int getWindowWidth();
    unsigned int getWindowHeight();
    std::string getColorMap();

    // Shared part of the constructors: reads every parameter from _config
    // (missing keys fall back to the defaults) and computes derived values
    void load();

public:
    Configs(const std::filesystem::path& filePath);
    // Builds the configuration directly from a YAML node; an empty node
    // yields the full default configuration (no file needed)
    explicit Configs(const YAML::Node& config);

    std::string ParametersPrint() const;
    void ParametersSave(const std::filesystem::path& outputDir) const;

    // Simulation Parameters
    unsigned int _gridLength;
    bool _singlePrecision;
    double _gridStep;
    double _lambda;
    unsigned int _dealWN;
    double _time;
    double _cfl;
    double _maxTimeStep;
    unsigned int _timeStepUpdateInterval;

    // Equation Coefficients: beta is the planetary vorticity gradient of
    // the beta-plane approximation (rotation), acting on the kinetic part
    double _nu;
    double _eta;
    double _beta;

    // Forcing Coefficients: energy pumping at the wavenumber ring
    // |k| in [ForcingWN - ForcingBand, ForcingWN + ForcingBand]
    bool _forcingEnabled;
    double _kineticForcing;
    double _magneticForcing;
    unsigned int _forcingWN;
    double _forcingBand;
    double _forcingKSqMin;
    double _forcingKSqMax;

    // Initial Condition Coefficients
    double _kineticEnergy;
    double _magneticEnergy;
    unsigned int _averageWN;

    // Output Parameters
    double _outputStep;
    double _outputStart;
    double _outputStop;

    // Kernel Run Parameters
    unsigned int _dimBlockX;
    unsigned int _dimBlockY;
    unsigned int _sharedLength;
    unsigned int _linearLength;

    // Writer Settings
    bool _saveData;
    bool _savePNG;
    bool _saveVorticity;
    bool _saveCurrent;
    bool _saveStream;
    bool _savePotential;

    // Graphics Settings
    bool _showGraphics;
    unsigned int _texturesCount;
    unsigned int _windowWidth;
    unsigned int _windowHeight;
    std::string _colorMap;
};
}  // namespace mhd

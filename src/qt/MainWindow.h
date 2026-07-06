#pragma once

#include <filesystem>
#include <string>

#include <QImage>
#include <QWidget>

#include "Configs.h"

class QCheckBox;
class QComboBox;
class QFormLayout;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;

namespace qtui {

// Renders the latest simulation frame scaled into the widget
class FieldView : public QWidget {
public:
    explicit FieldView(QWidget* parent = nullptr);

    void setImage(const QImage& image);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QImage _image;
};

// Main window: the field view on the left, all simulation settings on
// the right. The simulation runner polls the control flags between
// event-loop iterations, so no worker thread is needed
class MainWindow : public QWidget {
public:
    explicit MainWindow(const std::filesystem::path& resPath);

    // Consumes the pending start request (polled by the outer loop)
    bool takeStartRequest();
    // True while a start request waits (polled by the running session)
    bool startPending() const { return _startRequested; }
    bool stopRequested() const { return _stopRequested; }
    bool isPaused() const;
    bool isClosed() const { return _closed; }

    // Switches the buttons between the idle and the running states and
    // resets the stop/pause flags
    void setRunning(bool running);

    // The settings panel serialized as YAML: the same format and the
    // same defaults as the configuration files
    std::string configYamlText() const;

    void showFrame(const unsigned char* rgb, unsigned int length);
    void showStatus(const mhd::Currents& currents);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    QWidget* buildSettingsPanel(const std::filesystem::path& resPath);
    QLineEdit* addNumber(QFormLayout* form, const QString& label,
                         double value);

    bool _startRequested = false;
    bool _stopRequested = false;
    bool _closed = false;

    FieldView* _view = nullptr;

    // Status
    QLabel* _statusLabel = nullptr;
    QLabel* _energyLabel = nullptr;

    // Simulation
    QComboBox* _gridLength = nullptr;
    QComboBox* _precision = nullptr;
    QLineEdit* _time = nullptr;
    QLineEdit* _cfl = nullptr;
    QLineEdit* _maxTimeStep = nullptr;

    // Equations
    QLineEdit* _nu = nullptr;
    QLineEdit* _eta = nullptr;
    QLineEdit* _beta = nullptr;

    // Initial conditions
    QLineEdit* _kineticEnergy = nullptr;
    QLineEdit* _magneticEnergy = nullptr;
    QSpinBox* _averageWN = nullptr;

    // Forcing
    QLineEdit* _kineticForcing = nullptr;
    QLineEdit* _magneticForcing = nullptr;
    QSpinBox* _forcingWN = nullptr;
    QLineEdit* _forcingBand = nullptr;

    // Output
    QLineEdit* _outputStep = nullptr;
    QComboBox* _colorMap = nullptr;
    QCheckBox* _saveData = nullptr;
    QCheckBox* _saveVorticity = nullptr;
    QCheckBox* _saveCurrent = nullptr;
    QCheckBox* _saveStream = nullptr;
    QCheckBox* _savePotential = nullptr;

    // Controls
    QPushButton* _startButton = nullptr;
    QPushButton* _pauseButton = nullptr;
    QPushButton* _stopButton = nullptr;
};
}  // namespace qtui

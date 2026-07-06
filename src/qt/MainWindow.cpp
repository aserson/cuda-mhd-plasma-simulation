#include "MainWindow.h"

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDoubleValidator>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QVBoxLayout>

#include <sstream>

namespace qtui {
namespace {
// A scroll area that asks for exactly the height of its content: in a
// tall window it fits the settings and leaves the rest of the column
// empty, in a small one it shrinks and shows the scrollbar
class FitScrollArea : public QScrollArea {
public:
    using QScrollArea::QScrollArea;

    QSize sizeHint() const override {
        if (widget() == nullptr)
            return QScrollArea::sizeHint();
        return widget()->sizeHint() + QSize(2 * frameWidth(), 2 * frameWidth());
    }
};
}  // namespace

FieldView::FieldView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(512, 512);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void FieldView::setImage(const QImage& image) {
    _image = image;
    update();
}

void FieldView::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    if (_image.isNull()) {
        painter.setPen(Qt::gray);
        painter.drawText(rect(), Qt::AlignCenter,
                         "Press Start to run the simulation");
        return;
    }

    // The field is square: fill the largest centered square of the widget
    int side = qMin(width(), height());
    QRect target((width() - side) / 2, (height() - side) / 2, side, side);

    painter.setRenderHint(QPainter::SmoothPixmapTransform);
    painter.drawImage(target, _image);
}

MainWindow::MainWindow(const std::filesystem::path& resPath) {
    setWindowTitle("MHD Simulation");

    _view = new FieldView(this);
    QWidget* side = buildSettingsPanel(resPath);

    // The field view touches the window edges; the settings column keeps
    // its own inner padding
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(_view, 1);
    layout->addWidget(side, 0);

    // Default geometry: the window is exactly as tall as the settings
    // column (no scrollbar) and the field view is exactly square
    int height = side->sizeHint().height();
    if (const QScreen* screen = QGuiApplication::primaryScreen())
        height = qMin(height, screen->availableGeometry().height() - 48);
    resize(height + side->minimumWidth(), height);
}

QLineEdit* MainWindow::addNumber(QFormLayout* form, const QString& label,
                                 double value) {
    auto* edit = new QLineEdit(QString::number(value), this);
    auto* validator = new QDoubleValidator(edit);
    validator->setLocale(QLocale::c());
    edit->setValidator(validator);
    form->addRow(label, edit);
    return edit;
}

QWidget* MainWindow::buildSettingsPanel(const std::filesystem::path& resPath) {
    // The widgets start from the same defaults as a configuration file
    // with no keys set
    const mhd::Configs defaults{YAML::Node()};

    auto* panel = new QWidget(this);
    auto* column = new QVBoxLayout(panel);

    // Status
    auto* statusBox = new QGroupBox("Status", panel);
    auto* statusColumn = new QVBoxLayout(statusBox);
    _statusLabel = new QLabel("Idle", statusBox);
    _energyLabel = new QLabel(QString(), statusBox);
    statusColumn->addWidget(_statusLabel);
    statusColumn->addWidget(_energyLabel);
    column->addWidget(statusBox);

    // Simulation
    auto* simulationBox = new QGroupBox("Simulation", panel);
    auto* simulationForm = new QFormLayout(simulationBox);
    _gridLength = new QComboBox(simulationBox);
    for (unsigned int length = 256; length <= 8192; length *= 2)
        _gridLength->addItem(QString::number(length));
    _gridLength->setCurrentText(QString::number(defaults._gridLength));
    simulationForm->addRow("GridLength", _gridLength);
    _precision = new QComboBox(simulationBox);
    _precision->addItems({"double", "float"});
    _precision->setCurrentIndex(defaults._singlePrecision ? 1 : 0);
    simulationForm->addRow("Precision", _precision);
    _time = addNumber(simulationForm, "Time", defaults._time);
    _cfl = addNumber(simulationForm, "CFL", defaults._cfl);
    _maxTimeStep = addNumber(simulationForm, "MaxTimeStep",
                             defaults._maxTimeStep);
    column->addWidget(simulationBox);

    // Equations
    auto* equationsBox = new QGroupBox("Equation coefficients", panel);
    auto* equationsForm = new QFormLayout(equationsBox);
    _nu = addNumber(equationsForm, "nu", defaults._nu);
    _eta = addNumber(equationsForm, "eta", defaults._eta);
    _beta = addNumber(equationsForm, "beta", defaults._beta);
    column->addWidget(equationsBox);

    // Initial conditions
    auto* initialBox = new QGroupBox("Initial condition", panel);
    auto* initialForm = new QFormLayout(initialBox);
    _kineticEnergy = addNumber(initialForm, "KineticEnergy",
                               defaults._kineticEnergy);
    _magneticEnergy = addNumber(initialForm, "MagneticEnergy",
                                defaults._magneticEnergy);
    _averageWN = new QSpinBox(initialBox);
    _averageWN->setRange(1, 1024);
    _averageWN->setValue((int)defaults._averageWN);
    initialForm->addRow("AverageWN", _averageWN);
    column->addWidget(initialBox);

    // Forcing
    auto* forcingBox = new QGroupBox("Forcing", panel);
    auto* forcingForm = new QFormLayout(forcingBox);
    _kineticForcing = addNumber(forcingForm, "KineticForcing",
                                defaults._kineticForcing);
    _magneticForcing = addNumber(forcingForm, "MagneticForcing",
                                 defaults._magneticForcing);
    _forcingWN = new QSpinBox(forcingBox);
    _forcingWN->setRange(1, 1024);
    _forcingWN->setValue((int)defaults._forcingWN);
    forcingForm->addRow("ForcingWN", _forcingWN);
    _forcingBand = addNumber(forcingForm, "ForcingBand",
                             defaults._forcingBand);
    column->addWidget(forcingBox);

    // Output
    auto* outputBox = new QGroupBox("Output", panel);
    auto* outputForm = new QFormLayout(outputBox);
    _outputStep = addNumber(outputForm, "OutputStep", defaults._outputStep);
    _colorMap = new QComboBox(outputBox);
    std::filesystem::path colorMapsPath = resPath / "colormaps";
    if (exists(colorMapsPath)) {
        for (const auto& entry :
             std::filesystem::directory_iterator(colorMapsPath))
            _colorMap->addItem(
                QString::fromStdString(entry.path().filename().string()));
    }
    _colorMap->setCurrentText(QString::fromStdString(defaults._colorMap));
    outputForm->addRow("ColorMap", _colorMap);
    _saveData = new QCheckBox("SaveData (write fields to disk)", outputBox);
    _saveData->setChecked(defaults._saveData);
    outputForm->addRow(_saveData);
    _saveVorticity = new QCheckBox("Vorticity", outputBox);
    _saveVorticity->setChecked(defaults._saveVorticity);
    _saveCurrent = new QCheckBox("Current", outputBox);
    _saveCurrent->setChecked(defaults._saveCurrent);
    _saveStream = new QCheckBox("Stream", outputBox);
    _saveStream->setChecked(defaults._saveStream);
    _savePotential = new QCheckBox("Potential", outputBox);
    _savePotential->setChecked(defaults._savePotential);
    auto* fieldsRow = new QHBoxLayout();
    fieldsRow->addWidget(_saveVorticity);
    fieldsRow->addWidget(_saveCurrent);
    fieldsRow->addWidget(_saveStream);
    fieldsRow->addWidget(_savePotential);
    outputForm->addRow(fieldsRow);
    auto* hint = new QLabel("The first checked field is displayed", outputBox);
    hint->setStyleSheet("color: gray;");
    outputForm->addRow(hint);
    column->addWidget(outputBox);

    // Compact vertical spacing so the whole column fits a common screen
    // height without scrolling
    column->setSpacing(5);
    for (auto* form : panel->findChildren<QFormLayout*>())
        form->setVerticalSpacing(5);
    statusColumn->setSpacing(2);

    // The settings column scrolls when the window is small; in a tall
    // window it stops at its content height, so the control buttons sit
    // right below the settings instead of the window bottom
    auto* scroll = new FitScrollArea(this);
    scroll->setWidget(panel);
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);

    auto* buttonsRow = new QHBoxLayout();
    _startButton = new QPushButton("Start", this);
    _pauseButton = new QPushButton("Pause", this);
    _pauseButton->setCheckable(true);
    _pauseButton->setEnabled(false);
    _stopButton = new QPushButton("Stop", this);
    _stopButton->setEnabled(false);
    buttonsRow->addWidget(_startButton);
    buttonsRow->addWidget(_pauseButton);
    buttonsRow->addWidget(_stopButton);

    connect(_startButton, &QPushButton::clicked,
            [this] { _startRequested = true; });
    connect(_stopButton, &QPushButton::clicked,
            [this] { _stopRequested = true; });
    connect(_pauseButton, &QPushButton::toggled, [this](bool paused) {
        _pauseButton->setText(paused ? "Resume" : "Pause");
    });

    auto* side = new QWidget(this);
    auto* sideColumn = new QVBoxLayout(side);
    // The field view is flush with the window edges, while the settings
    // column keeps a padding on its right and bottom sides
    sideColumn->setContentsMargins(8, 0, 8, 8);
    // Stretch 0 on the scroll area: it takes exactly its size hint (the
    // settings content) and the spacer absorbs the rest of a tall window;
    // in a small window the Maximum policy lets the scroll area shrink
    sideColumn->addWidget(scroll, 0);
    sideColumn->addLayout(buttonsRow);
    sideColumn->addStretch(1);
    side->setFixedWidth(panel->sizeHint().width() + 48);
    return side;
}

bool MainWindow::takeStartRequest() {
    if (!_startRequested)
        return false;
    _startRequested = false;
    return true;
}

bool MainWindow::isPaused() const {
    return _pauseButton->isChecked();
}

void MainWindow::setRunning(bool running) {
    _stopRequested = false;
    _pauseButton->setChecked(false);
    _pauseButton->setEnabled(running);
    _stopButton->setEnabled(running);
    _startButton->setText(running ? "Restart" : "Start");
    if (!running)
        _statusLabel->setText("Idle");
}

std::string MainWindow::configYamlText() const {
    std::ostringstream yaml;

    yaml << "GridLength : " << _gridLength->currentText().toStdString()
         << "\n";
    yaml << "Precision : " << _precision->currentText().toStdString() << "\n";
    yaml << "Time : " << _time->text().toStdString() << "\n";
    yaml << "CFL : " << _cfl->text().toStdString() << "\n";
    yaml << "MaxTimeStep : " << _maxTimeStep->text().toStdString() << "\n";

    yaml << "nu : " << _nu->text().toStdString() << "\n";
    yaml << "eta : " << _eta->text().toStdString() << "\n";
    yaml << "beta : " << _beta->text().toStdString() << "\n";

    yaml << "KineticEnergy : " << _kineticEnergy->text().toStdString() << "\n";
    yaml << "MagneticEnergy : " << _magneticEnergy->text().toStdString()
         << "\n";
    yaml << "AverageWN : " << _averageWN->value() << "\n";

    yaml << "KineticForcing : " << _kineticForcing->text().toStdString()
         << "\n";
    yaml << "MagneticForcing : " << _magneticForcing->text().toStdString()
         << "\n";
    yaml << "ForcingWN : " << _forcingWN->value() << "\n";
    yaml << "ForcingBand : " << _forcingBand->text().toStdString() << "\n";

    yaml << "OutputStep : " << _outputStep->text().toStdString() << "\n";
    yaml << "ColorMap : " << _colorMap->currentText().toStdString() << "\n";
    yaml << "SaveData : " << (_saveData->isChecked() ? "true" : "false")
         << "\n";
    yaml << "SaveVorticity : "
         << (_saveVorticity->isChecked() ? "true" : "false") << "\n";
    yaml << "SaveCurrent : " << (_saveCurrent->isChecked() ? "true" : "false")
         << "\n";
    yaml << "SaveStream : " << (_saveStream->isChecked() ? "true" : "false")
         << "\n";
    yaml << "SavePotential : "
         << (_savePotential->isChecked() ? "true" : "false") << "\n";

    // The Qt window replaces the OpenGL one, but the painter is still
    // driven by the ShowGraphics setting
    yaml << "ShowGraphics : true\n";

    return yaml.str();
}

void MainWindow::showFrame(const unsigned char* rgb, unsigned int length) {
    QImage image(rgb, (int)length, (int)length, (int)(3 * length),
                 QImage::Format_RGB888);
    // The OpenGL window drew the buffer with the first row at the bottom;
    // flipping keeps the picture orientation the same as before
    _view->setImage(image.flipped(Qt::Vertical));
}

void MainWindow::showStatus(const mhd::Currents& currents) {
    _statusLabel->setText(QString("Time: %1    Step: %2    dt: %3")
                              .arg(currents.time, 0, 'f', 3)
                              .arg(currents.stepNumber)
                              .arg(currents.timeStep, 0, 'g', 4));
    _energyLabel->setText(
        QString("Ekin: %1    Emag: %2    Esum: %3")
            .arg(currents.kineticEnergy, 0, 'f', 4)
            .arg(currents.magneticEnergy, 0, 'f', 4)
            .arg(currents.kineticEnergy + currents.magneticEnergy, 0, 'f', 4));
}

void MainWindow::closeEvent(QCloseEvent* event) {
    _closed = true;
    event->accept();
}
}  // namespace qtui

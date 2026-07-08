#include "MainWindow.h"

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDoubleValidator>
#include <QFileDialog>
#include <QFormLayout>
#include <QMessageBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTabWidget>
#include <QVBoxLayout>

#include <cmath>
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

SpectrumView::SpectrumView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(512, 512);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void SpectrumView::setSpectra(const std::vector<double>& kinetic,
                              const std::vector<double>& magnetic) {
    _kinetic = kinetic;
    _magnetic = magnetic;
    update();
}

void SpectrumView::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    double maxValue = 0.;
    for (double value : _kinetic)
        maxValue = qMax(maxValue, value);
    for (double value : _magnetic)
        maxValue = qMax(maxValue, value);

    if (_kinetic.size() < 3 || maxValue <= 0.) {
        painter.setPen(Qt::gray);
        painter.drawText(rect(), Qt::AlignCenter,
                         "The spectra appear on the next output of a "
                         "running simulation");
        return;
    }

    const QRectF plot(64., 20., width() - 84., height() - 60.);
    const double maxWN = (double)(_kinetic.size() - 1);
    const double logMaxWN = std::log10(maxWN);

    // Vertical range: a fixed number of decades down from the maximum
    const int decades = 10;
    const int maxExp = (int)std::ceil(std::log10(maxValue));

    auto pointX = [&](double k) {
        return plot.left() + plot.width() * std::log10(k) / logMaxWN;
    };
    auto pointY = [&](double value) {
        return plot.top() + plot.height() *
                                ((double)maxExp - std::log10(value)) / decades;
    };

    // Grid: wavenumber decades and every second energy decade
    painter.setPen(QColor(60, 60, 60));
    painter.setFont(QFont(painter.font().family(), 8));
    for (double k = 1.; k <= maxWN; k *= 10.) {
        double x = pointX(k);
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
        painter.setPen(Qt::gray);
        painter.drawText(QRectF(x - 30., plot.bottom() + 6., 60., 16.),
                         Qt::AlignCenter, QString::number(k));
        painter.setPen(QColor(60, 60, 60));
    }
    for (int exponent = maxExp; exponent >= maxExp - decades; exponent -= 2) {
        double y = pointY(std::pow(10., exponent));
        painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
        painter.setPen(Qt::gray);
        painter.drawText(QRectF(0., y - 8., plot.left() - 6., 16.),
                         Qt::AlignRight | Qt::AlignVCenter,
                         QString("1e%1").arg(exponent));
        painter.setPen(QColor(60, 60, 60));
    }
    painter.setPen(Qt::gray);
    painter.drawText(QRectF(plot.center().x() - 20., height() - 22., 40., 16.),
                     Qt::AlignCenter, "k");

    // Curves
    painter.setRenderHint(QPainter::Antialiasing);
    const double minValue = std::pow(10., maxExp - decades);
    auto drawCurve = [&](const std::vector<double>& spectrum,
                         const QColor& color) {
        QPainterPath path;
        bool started = false;
        for (unsigned int k = 1; k < spectrum.size(); k++) {
            if (spectrum[k] < minValue) {
                started = false;
                continue;
            }
            QPointF point(pointX((double)k), pointY(spectrum[k]));
            if (started) {
                path.lineTo(point);
            } else {
                path.moveTo(point);
                started = true;
            }
        }
        painter.setPen(QPen(color, 1.5));
        painter.drawPath(path);
    };

    const QColor kineticColor(80, 180, 255);
    const QColor magneticColor(255, 150, 60);
    drawCurve(_kinetic, kineticColor);
    drawCurve(_magnetic, magneticColor);

    // Legend
    painter.setPen(kineticColor);
    painter.drawText(QRectF(plot.right() - 150., plot.top() + 4., 150., 16.),
                     Qt::AlignLeft, "kinetic");
    painter.setPen(magneticColor);
    painter.drawText(QRectF(plot.right() - 150., plot.top() + 22., 150., 16.),
                     Qt::AlignLeft, "magnetic");
}

MainWindow::MainWindow(const std::filesystem::path& resPath,
                       const std::filesystem::path& configsPath)
    : _configsPath(configsPath) {
    setWindowTitle("MHD Simulation");

    _view = new FieldView(this);
    _spectrum = new SpectrumView(this);
    _tabs = new QTabWidget(this);
    _tabs->setDocumentMode(true);
    _tabs->addTab(_view, "Field");
    _tabs->addTab(_spectrum, "Spectra");
    QWidget* side = buildSettingsPanel(resPath);

    // The widgets start from the same defaults as a configuration file
    // with no keys set
    applyConfigs(mhd::Configs{YAML::Node()});

    // The field view touches the window edges; the settings column keeps
    // its own inner padding
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(_tabs, 1);
    layout->addWidget(side, 0);

    // Default geometry: the window is exactly as tall as the settings
    // column (no scrollbar) and the field view is exactly square
    int height = side->sizeHint().height();
    if (const QScreen* screen = QGuiApplication::primaryScreen())
        height = qMin(height, screen->availableGeometry().height() - 48);
    resize(height + side->minimumWidth(), height);
}

QLineEdit* MainWindow::addNumber(QFormLayout* form, const QString& label) {
    auto* edit = new QLineEdit(this);
    auto* validator = new QDoubleValidator(edit);
    validator->setLocale(QLocale::c());
    edit->setValidator(validator);
    form->addRow(label, edit);
    return edit;
}

QWidget* MainWindow::buildSettingsPanel(const std::filesystem::path& resPath) {
    auto* panel = new QWidget(this);
    auto* column = new QVBoxLayout(panel);

    // Loading a configuration file fills the panel; the values apply to
    // the next started session
    auto* loadRow = new QHBoxLayout();
    auto* loadButton = new QPushButton("Load config...", panel);
    connect(loadButton, &QPushButton::clicked, [this] { loadConfigFile(); });
    loadRow->addWidget(loadButton);
    loadRow->addStretch(1);
    column->addLayout(loadRow);

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
    simulationForm->addRow("GridLength", _gridLength);
    _precision = new QComboBox(simulationBox);
    _precision->addItems({"double", "float"});
    simulationForm->addRow("Precision", _precision);
    _time = addNumber(simulationForm, "Time");
    _cfl = addNumber(simulationForm, "CFL");
    _maxTimeStep = addNumber(simulationForm, "MaxTimeStep");
    column->addWidget(simulationBox);

    // Equations
    auto* equationsBox = new QGroupBox("Equation coefficients", panel);
    auto* equationsForm = new QFormLayout(equationsBox);
    _nu = addNumber(equationsForm, "nu");
    _eta = addNumber(equationsForm, "eta");
    _beta = addNumber(equationsForm, "beta");
    column->addWidget(equationsBox);

    // Initial conditions
    auto* initialBox = new QGroupBox("Initial condition", panel);
    auto* initialForm = new QFormLayout(initialBox);
    _kineticEnergy = addNumber(initialForm, "KineticEnergy");
    _magneticEnergy = addNumber(initialForm, "MagneticEnergy");
    _averageWN = new QSpinBox(initialBox);
    _averageWN->setRange(1, 1024);
    initialForm->addRow("AverageWN", _averageWN);
    column->addWidget(initialBox);

    // Forcing
    auto* forcingBox = new QGroupBox("Forcing", panel);
    auto* forcingForm = new QFormLayout(forcingBox);
    _kineticForcing = addNumber(forcingForm, "KineticForcing");
    _magneticForcing = addNumber(forcingForm, "MagneticForcing");
    _forcingWN = new QSpinBox(forcingBox);
    _forcingWN->setRange(1, 1024);
    forcingForm->addRow("ForcingWN", _forcingWN);
    _forcingBand = addNumber(forcingForm, "ForcingBand");
    column->addWidget(forcingBox);

    // Output
    auto* outputBox = new QGroupBox("Output", panel);
    auto* outputForm = new QFormLayout(outputBox);
    _outputStep = addNumber(outputForm, "OutputStep");
    _colorMap = new QComboBox(outputBox);
    std::filesystem::path colorMapsPath = resPath / "colormaps";
    if (exists(colorMapsPath)) {
        for (const auto& entry :
             std::filesystem::directory_iterator(colorMapsPath))
            _colorMap->addItem(
                QString::fromStdString(entry.path().filename().string()));
    }
    outputForm->addRow("ColorMap", _colorMap);
    _saveData = new QCheckBox("SaveData (write fields to disk)", outputBox);
    outputForm->addRow(_saveData);
    _saveVorticity = new QCheckBox("Vorticity", outputBox);
    _saveCurrent = new QCheckBox("Current", outputBox);
    _saveStream = new QCheckBox("Stream", outputBox);
    _savePotential = new QCheckBox("Potential", outputBox);
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

void MainWindow::applyConfigs(const mhd::Configs& configs) {
    // Combo boxes get the value inserted when it is not in the list yet
    // (a grid length or a color map outside the predefined options)
    QString gridLength = QString::number(configs._gridLength);
    if (_gridLength->findText(gridLength) < 0)
        _gridLength->addItem(gridLength);
    _gridLength->setCurrentText(gridLength);

    _precision->setCurrentIndex(configs._singlePrecision ? 1 : 0);
    _time->setText(QString::number(configs._time));
    _cfl->setText(QString::number(configs._cfl));
    _maxTimeStep->setText(QString::number(configs._maxTimeStep));

    _nu->setText(QString::number(configs._nu));
    _eta->setText(QString::number(configs._eta));
    _beta->setText(QString::number(configs._beta));

    _kineticEnergy->setText(QString::number(configs._kineticEnergy));
    _magneticEnergy->setText(QString::number(configs._magneticEnergy));
    _averageWN->setValue((int)configs._averageWN);

    _kineticForcing->setText(QString::number(configs._kineticForcing));
    _magneticForcing->setText(QString::number(configs._magneticForcing));
    _forcingWN->setValue((int)configs._forcingWN);
    _forcingBand->setText(QString::number(configs._forcingBand));

    _outputStep->setText(QString::number(configs._outputStep));
    QString colorMap = QString::fromStdString(configs._colorMap);
    if (_colorMap->findText(colorMap) < 0)
        _colorMap->addItem(colorMap);
    _colorMap->setCurrentText(colorMap);

    _saveData->setChecked(configs._saveData);
    _saveVorticity->setChecked(configs._saveVorticity);
    _saveCurrent->setChecked(configs._saveCurrent);
    _saveStream->setChecked(configs._saveStream);
    _savePotential->setChecked(configs._savePotential);
}

void MainWindow::loadConfigFile() {
    QString filePath = QFileDialog::getOpenFileName(
        this, "Load configuration",
        QString::fromStdString(_configsPath.string()),
        "YAML files (*.yaml *.yml);;All files (*)");
    if (filePath.isEmpty())
        return;

    try {
        applyConfigs(mhd::Configs{YAML::LoadFile(filePath.toStdString())});
    } catch (const std::exception& error) {
        QMessageBox::warning(this, "Configuration error", error.what());
    }
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

bool MainWindow::spectraVisible() const {
    return _tabs->currentWidget() == _spectrum;
}

void MainWindow::showSpectra(const std::vector<double>& kinetic,
                             const std::vector<double>& magnetic) {
    _spectrum->setSpectra(kinetic, magnetic);
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

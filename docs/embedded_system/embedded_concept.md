<!--

author:   WRO FE SIM Team
email:    fabian.zeiler@tu-freiberg.de
version:  1.4.0
language: de
narrator: Deutsch Female
comment:  Embedded Concept & Architektur-Spezifikation für die Sim-to-Real-Übertragung eines autonomen Rennroboters (PPO-Policy, FreeRTOS, Embedded Linux, Safety Filter).
icon:     https://upload.wikimedia.org/wikipedia/commons/d/de/Logo_TU_Bergakademie_Freiberg.svg

import:   https://raw.githubusercontent.com/LiaTemplates/mermaid_template/0.1.4/README.md

-->

[![LiaScript](https://raw.githubusercontent.com/LiaScript/LiaScript/master/badges/course.svg)](https://liascript.github.io/course/?https://raw.githubusercontent.com/Bigfire3/wro-fe-rl-simulation/main/docs/embedded_system/embedded_concept.md#1)

# Embedded Concept -- Sim-to-Real Autonomous Racing Robot

| Parameter | Kurs- / Projektinformationen |
| --- | --- |
| **Projekt:** | `WRO Future Engineers -- Sim-to-Real Autonomous Racing Robot` |
| **Modul:** | `Softwareentwicklung für Eingebettete Systeme` |
| **Hochschule:** | `Technische Universität Bergakademie Freiberg` |
| **GitHub:** | [Bigfire3/wro-fe-rl-simulation](https://github.com/Bigfire3/wro-fe-rl-simulation/blob/main/docs/embedded_system/embedded_concept.md) |
| **Autoren:** | Fabian Zänker |

-----

![Webots 3D Rennsimulation](../media/Track_Model_5.gif)

![ICP-Lokalisierung & Pfadvisualisierung](../media/Loc_Model_5.gif)

- **Methode:** Deep Reinforcement Learning (PPO) in Webots-Simulation
- **Pipeline:** Sequenzielle 4-Stufen-Architektur (`Perception` $\rightarrow$ `Estimation` $\rightarrow$ `Planning` $\rightarrow$ `Control`)
- **Beobachtung ($\vec{o}$):** $15$-dimensionaler normalisierter Vektor ($\vec{o} \in [-1.0, 1.0]^{15}$)
- **Aktion ($\vec{a}$):** Continuous Steering & Speed Adjustments ($\Delta \delta$, $\Delta v$)
- **Modell-Export:** PyTorch $\rightarrow$ ONNX Actor Model ($18.818$ Parameter, $\approx 74\,\text{KiB}$ Float32 Gewichte)

-----

## 1. Sim-to-Real-Gap

| System-Aspekt | Webots Simulation (SiL) | Reales Embedded System |
| --- | --- | --- |
| **Sensorik** | Rauschfrei, exakt, synchrone Zeitstempel | Rauschen, Bias, Latenzen, Dropouts, asynchrone Zeitstempel |
| **Aktuatorik** | Ideale Lenkwinkel- & Drehmoment-Umsetzung | Totzonen, Spiel (Backlash), Reifenschlupf, PWM-Treiber |
| **Ressourcen** | Unbegrenzter PC (Python, NumPy, PyTorch) | Begrenzter Flash, SRAM, Rechenleistung, Energieressourcen |
| **Geschwindigkeit!** | Direkt aus Physik-Engine | Schätzung via Encodern + IMU (Aktive Redundanz) |

Was ich in der Simulation für die Robustheit noch verbessern könnte:
- **Domain Randomization:** Variation von Reibwerten, Massen, Trägheitsmomenten
- **Stör- & Verzögerungsmodelle:** Künstliches Sensorrauschen & Kommunikations-Latenzen
- **Sensorausfall-Szenarien:** Simulation von Sensor-Dropouts

-----

## 2. Echtzeit-Anforderungen & Timing-Budgets

- **Regelfrequenz:** $10\,\text{Hz}$
- **Sensorabtastung $\rightarrow$ PWM-Ausgabe** $\le 100\,\text{ms}$

| Phase | Zeitbudget | Hauptaufgabe |
| --- | ---: | --- |
| **1. Sensor-Sync** | $15\,\text{ms}$ | Zeitstempel-Zuordnung & Gültigkeits-Check (DMA/ISR) |
| **2. Lokalisierung (ICP)** | $40\,\text{ms}$ | Bounded Processing Time |
| **3. Inferenz (ONNX)** | $15\,\text{ms}$ |  deterministische PPO-Policy-Berechnung |
| **4. Motor- & Lenkansteuerung** | $10\,\text{ms}$ | PWM-Erzeugung & Signalübertragung (CAN/UART) |
| **5. Jitter-Reserve** | $20\,\text{ms}$ | Puffer für Interrupts, Kontextwechsel & Laufzeitschwankungen |

### 2.2 Echtzeit-Klassifikation
- **RL-Policy:** Weiche / Feste Echtzeit (Latenz verschlechtert Fahrqualität, verursacht nicht direkt Crash)
- **Sicherheitsfilter / Not-Aus:** Harte Echtzeit (Deterministische Einhaltung von Fahrzeug- & Kollisionsgrenzen)

-----

## 3. Architekturoptionen & Zielarchitektur

### 3.1 Vergleich der 3 Architektur-Varianten

| Variante | Plattform-Idee | Vorteile | Nachteile / Ausschlussgrund |
| --- | --- | --- | --- |
| **Embedded Linux** | ARM-basierter Single-Board Computer (Jetson / RPi) | Python/ONNX direkt nutzbar, viel RAM für LiDAR/ICP | Kein harter Determinismus (ohne PREEMPT_RT), hoher Energiebedarf, lange Bootzeit |
| **Mikrocontroller (MCU)** | STM32 / ESP32 / ATmega | Geringer Energiebedarf, kurze Bootzeit, direkte Hardware-Kontrolle | **ATmega328 (2 KiB SRAM):** Ungeeignet! Policy-Gewichte ($\approx 74\,\text{KiB}$) sprengen Speicher vollkommen.<br>**STM32F401 (96 KiB SRAM):** Int8-Gewichte ($\approx 19\,\text{KiB}$) passen theoretisch, aber ICP/Kamera sprengen RAM.<br>**ESP32-S3 (512 KiB SRAM):** Inferenz machbar, aber komplette Wahrnehmung (LiDAR/Pointcloud) überfordert MCU. |
| **Heterogene Architektur** | Leistungsrechner + MCU | Perfekte Aufgabentrennung (AMP), Autonomie & Sicherheit entkoppelt | Höhere Systemkomplexität & Bus-Kopplung erforderlich |

### 3.2 Heterogene Zielarchitektur & Aufgabenverteilung

```mermaid @mermaid
flowchart LR
    subgraph Linux [Leistungsrechner - Embedded Linux]
        S[LiDAR / Kamera] --> P[Perzeption & ICP-Lokalisierung]
        P --> O[15D Vector & ONNX Inferenz]
    end

    subgraph MCU [Sicherheits-Controller - FreeRTOS MCU]
        I[IMU / Encoder] --> M[Sensorerfassung & Fusionsfilter]
        M --> SF[Deterministischer Safety Filter]
        O <-->|CAN / UART\nCRC+Timeouts| SF
        SF --> A[Motor & Servo PWM]
        E[Not-Aus Button] --> SF
        SF --> Dis[Hardware-Abschaltpfad]
    end
```

| Komponente | Plattform | Hauptaufgaben | Hardware-Features |
| --- | --- | --- | --- |
| **Leistungsrechner** | Embedded Linux | Kamera/LiDAR, ICP-Lokalisierung, ONNX-Inferenz | Multicore, FPU, viel SRAM/Flash |
| **Sicherheits-Controller** | FreeRTOS | Encoderauswertung, Servo/Motor-PWM, Safety Filter, Watchdog | DMA, Timer, ADC, GPIO, NVIC |
| **Bus-Kopplung** | CAN / UART | Sichere Signalübertragung mit CRC, Sequenznummer & Timeouts | Hardware-CRC, Transceiver |

-----

## 4. RTOS Task-Struktur & Hardware-Abstraktion

### 4.1 FreeRTOS Task-Zerlegung

| Task | Periode / Auslösung | Priorität | Kommunikation & Synchronisation |
| --- | --- | --- | --- |
| `Safety_Task` | Event / $1\text{--}5\,\text{ms}$ | `Highest` | Direct Interrupt / Semaphore |
| `Actuator_Task` | $5\text{--}10\,\text{ms}$ | `High` | Queue (Letzter gültiger Sollwert) |
| `Sensor_Task` | $5\text{--}10\,\text{ms}$ | `High` | DMA / Ringpuffer |
| `Autonomie_Comm_Task` | $100\,\text{ms}$ | `Medium` | Queue |
| `Logging_Task` | $500\text{--}1000\,\text{ms}$ | `Low` | Stream Buffer / UART |

### 4.2 Software-Schichtenarchitektur

```
[ Autonomie- & RL-Anwendung ]
          │
[ Plattformunabhängiges Sensor-/Aktuator-Interface ]
          │
[ Hardware Abstraction Layer (HAL) ]
          │
[ Board Support Package (BSP) & Treiber ]  ──> (GPIO, Timer, ADC, CAN, PWM)
```

RTOS-Mechanismen
- **ISR & DMA** (hardwarenah, latenzarm)
- **Queues/Semaphores** (entkoppeln Producer/Consumer)
- **Mutex & PIP** (Ressourcenschutz ohne Prioritätsinversion)

-----

## 5. Edge-AI Deployment

$$\text{PyTorch (PC)} \xrightarrow{\text{Export}} \text{ONNX (Float32)} \xrightarrow{\text{Quantisierung}} \text{Int8 C-Code}$$

| Format | Speicherbedarf Gewichte | Ziel-Laufzeitumgebung |
| --- | ---: | --- |
| **Float32 (Baseline)** | $\approx 74\,\text{KiB}$ | ONNX Runtime (Embedded Linux) |
| **Int8 (Quantisiert)** | $\approx 19\,\text{KiB}$ | CMSIS-NN (MCU) |

### 5.1 Safety Cage um die Policy
RL-Policy liefert **nur Wunsch-Sollwerte**
**Sicherheitsfilter:**
- Lenkwinkel- & Lenkraten-Begrenzung
- Geschwindigkeits- & Beschleunigungs-Limits
- Plausibilitäts-Check ($NaN$- & Ausreißer-Rejektion)
- Freshness
- Timeout: Not-Aus bei Kommunikationsabriss
- Emergency Stop: Sofortige Abschaltung im Fehlerfall

-----

## 6. Zustandsautomat

```mermaid @mermaid
stateDiagram-v2
    [*] --> Boot: Power On
    Boot --> SelfTest: Hardware OK
    SelfTest --> Calibration: Sensors OK
    Calibration --> Ready: Pose
    Ready --> Driving: Start signal
    Driving --> Ready: Stop
    Driving --> Degraded: Partial sensor failure
    Driving --> SafeStop: E-Stop / Timeout / Deadline Miss
    Degraded --> SafeStop: Failure escalated
    SafeStop --> Ready: Reset
```

- **Boot:** Speicher- und Treiberinitialisierung
- **SelfTest:** Hardware-, Sensor- und Spannungsprüfungen
- **Calibration:** Nullpunktkalibrierung (IMU/Lenkung) & initiale Lokalisierung per OpenCV-Template-Matching (CW/CCW + Position)
- **Ready:** System bereit
- **Driving:** Autonomer Betrieb (**RL-Inferenz nur hier aktiv!**)
- **Degraded:** Niedriggeschwindigkeits-Fallback bei Teilausfall
- **SafeStop:** Not-Aus, sicheres Abbremsen & Abschalten der Aktuatoren

-----

## 7. Verifikation & Test-Pyramide (am Projekt)

| Stufe | Projektbeispiel | Was wird getestet? |
| --- | --- | --- |
| **Host-Tests** | Ackermann-Kinematik, Koordinatentransformationen, $15$D-Normalisierung, Clipping-Logik, ONNX-Referenzausgaben | Reine Mathematik & Logik, kein µC nötig, tausende Tests in CI |
| **On-Target** | Safety-Filter-Grenzwerte auf STM32, PWM-Timing, ADC-Abtastung, Watchdog-Auslösung, CAN-Frame-Integrität | Echter µC, echte Register via JTAG/SWD, verifiziert Firmware-Verhalten |
| **HIL** | Aufgebocktes Fahrzeug $\rightarrow$ Langsame Geradeausfahrt $\rightarrow$ Kurven $\rightarrow$ Hindernisse $\rightarrow$ Fehlerinjektion auf Teststrecke | Gesamtsystem mit Leistungsrechner + MCU + Sensorik, findet Integrationsfehler |

-----

## 8. Dependability & Fehlertoleranz

Fehler-Kaskade (Beispiel: IMU-Ausfall)

$$\underbrace{\text{IMU-Kabelbruch}}_{\text{Fault}} \longrightarrow \underbrace{\text{Fusionsfilter divergiert}}_{\text{Error}} \longrightarrow \underbrace{\text{Fahrzeug verlässt Strecke}}_{\text{Failure}}$$


Konkrete Schutzmaßnahmen

| Maßnahme | Mechanismus im Projekt | Schutzziel |
| --- | --- | --- |
| **Hardware-Watchdog** | MCU-Watchdog-Timer, Reset bei ausbleibender Bedienung | Software-Hänger $\rightarrow$ automatischer Neustart |
| **Heartbeat** | Leistungsrechner sendet periodisches Lebenszeichen an MCU | Erkennung eines Linux-Absturzes oder Kommunikationsabrisses |
| **Informationelle Redundanz** | CRC32 auf allen CAN/UART-Nachrichten, Sequenznummern, ONNX-Modell-Hash bei Boot | Erkennung von Bitfehlern, verlorenen Paketen, falschem Modell |
| **Funktionale Redundanz** | Rad-Encoder + IMU liefern unabhängig Geschwindigkeit (Aktive Redundanz) | Plausibilisierung & Schlupferkennung |
| **Hardware-Abschaltpfad** | MCU-kontrollierter MOSFET trennt Motorbrücke physikalisch | Fail-Safe: Aktuatoren stromlos bei kritischem Fehler |

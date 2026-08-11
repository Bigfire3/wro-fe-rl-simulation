<!--

author:   WRO FE SIM Team
email:    fabian.zeiler@tu-freiberg.de
version:  1.2.0
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
| **Modul:** | `Softwareentwicklung für Eingebettete Systeme / RL Simulation` |
| **Hochschule:** | `Technische Universität Bergakademie Freiberg` |
| **Inhalte:** | `Embedded Architecture, Timing-Budgets, Edge-AI Deployment, Safety Supervision & Dependability` |
| **GitHub:** | [Bigfire3/wro-fe-rl-simulation](https://github.com/Bigfire3/wro-fe-rl-simulation/blob/main/docs/embedded_system/embedded_concept.md) |
| **Autoren:** | Fabian Zänker |

-----

## 1. Projekt-Übersicht & Simulations-Baseline

![Webots 3D Rennsimulation](../media/Track_Model_5.gif)

![ICP-Lokalisierung & Pfadvisualisierung](../media/Loc_Model_5.gif)

### 1.1 Kernfakten
- **Ziel:** Autonomes WRO Future Engineers Rennfahrzeug
- **Methode:** Deep Reinforcement Learning (PPO) in Webots-Simulation
- **Pipeline:** Sequenzielle 4-Stufen-Architektur (`Perception` $\rightarrow$ `Estimation` $\rightarrow$ `Planning` $\rightarrow$ `Control`)
- **Beobachtung ($\vec{o}$):** $15$-dimensionaler normalisierter Vektor ($\vec{o} \in [-1.0, 1.0]^{15}$)
- **Aktion ($\vec{a}$):** Continuous Steering & Speed Adjustments ($\Delta \delta$, $\Delta v$)
- **Modell-Export:** PyTorch $\rightarrow$ ONNX Actor Model ($18.818$ Parameter, $\approx 74\,\text{KiB}$ Float32 Gewichte)

-----

## 2. Sim-to-Real-Lücke (Simulation vs. Realität)

| System-Aspekt | Webots Simulation (SiL) | Reales Embedded System |
| --- | --- | --- |
| **Geschwindigkeit** | Direkt aus Physik-Engine / Supervisor | Schätzung via Encodern + IMU (ADC, Interrupts, Fusionsfilter) |
| **Sensorik** | Rauschfrei, exakt, synchrone Zeitstempel | Rauschen, Bias, Latenzen, Dropouts, asynchrone Zeitstempel |
| **Aktuatorik** | Ideale Lenkwinkel- & Drehmoment-Umsetzung | Totzonen, Sättigung, Spiel (Backlash), Reifenschlupf, PWM-Treiber |
| **Ressourcen** | Unbegrenzter PC (Python, NumPy, PyTorch) | Begrenzter Flash, SRAM, Rechenleistung, Energie- & Thermobudget |
| **Hauptengpass** | Keine zeitlichen Restriktionen | Perzeption & ICP-Lokalisierung (nicht die Policy selbst!) |

### 2.1 Training-Side Mitigation (Robustheitsmaßnahmen)
- **Domain Randomization:** Variation von Reibwerten, Massen, Trägheitsmomenten
- **Stör- & Verzögerungsmodelle:** Künstliches Sensorrauschen & Kommunikations-Latenzen
- **Sensorausfall-Szenarien:** Simulation von Sensor-Dropouts
- **Aktuator-Variabilität:** Parameterstreuung bei Motoren & Servos

-----

## 3. Echtzeit-Anforderungen & Timing-Budgets

- **Regelfrequenz:** $10\,\text{Hz}$ ($100\,\text{ms}$ Kontrollzyklus)
- **Ende-zu-Ende Deadline:** $\le 100\,\text{ms}$ (Sensorabtastung $\rightarrow$ PWM-Ausgabe)

### 3.1 Timing-Budget Zerlegung ($100\,\text{ms}$ Gesamtbudget)

| Phase | Zeitbudget | Hauptaufgabe |
| --- | ---: | --- |
| **1. Sensor-Sync** | $15\,\text{ms}$ | Zeitstempel-Zuordnung & Gültigkeits-Check (DMA/ISR) |
| **2. Lokalisierung (ICP)** | $40\,\text{ms}$ | Bounded Processing Time & Kartenabgleich |
| **3. Inferenz (ONNX)** | $15\,\text{ms}$ | Deterministische PPO-Policy Inferenz |
| **4. Aktuierung & Comm** | $10\,\text{ms}$ | PWM-Erzeugung & CAN/UART-Übertragung |
| **5. Jitter-Reserve** | $20\,\text{ms}$ | Puffer für Interrupts, Kontextwechsel & Varianz |

### 3.2 Echtzeit-Klassifikation
- **RL-Policy (Pfadplanung):** Weiche / Feste Echtzeit (Latenz verschlechtert Fahrqualität, verursacht nicht direkt Crash)
- **Sicherheitsfilter / Not-Stopp:** Harte Echtzeit (Deterministische Einhaltung von Fahrzeug- & Kollisionsgrenzen)

-----

## 4. Zielarchitektur (Heterogener AMP-Ansatz)

```mermaid @mermaid
flowchart LR
    subgraph Linux [Embedded Linux - Jetson / Raspberry Pi]
        S[LiDAR / Kamera] --> P[Perzeption & ICP-Lokalisierung]
        P --> O[15D Vector & ONNX Inference]
    end

    subgraph MCU [FreeRTOS MCU - STM32 / Cortex-M]
        I[IMU / Encoder] --> M[Sensorerfassung & Fusionsfilter]
        M --> SF[Deterministischer Safety Filter]
        O <-->|CAN / UART\nCRC · Timeouts| SF
        SF --> A[Motor & Servo PWM]
        E[E-Stop Button] --> SF
        SF --> Dis[Physical Actuator Disable]
    end
```

### 4.1 Aufgaben- & Hardwareverteilung

| Komponente | Plattform | Hauptaufgaben | Hardware-Features |
| --- | --- | --- | --- |
| **Autonomie-Rechner** | Embedded Linux | Kamera/LiDAR, ICP, ONNX-Runtime, Logging | Multicore, FPU, High SRAM/Flash |
| **Safety Controller** | FreeRTOS MCU | Encoderauswertung, Servo/Motor-PWM, Safety Filter, Watchdog | DMA, Timers, ADC, GPIO, NVIC |
| **Bus-Kopplung** | CAN / UART | Sichere Protokollübertragung mit CRC, Sequence-ID & Timeouts | Hardware-CRC, Transceiver |

-----

## 5. RTOS Task-Struktur & Hardware-Abstraktion

### 5.1 FreeRTOS Task-Zerlegung

| Task | Periode / Auslösung | Priorität | Kommunikation / Sync |
| --- | --- | --- | --- |
| `Safety_Task` | Event / $1\text{--}5\,\text{ms}$ | `Highest` | Direct Interrupt / Semaphore |
| `Actuator_Task` | $5\text{--}10\,\text{ms}$ | `High` | Queue (Letzter gültiger Sollwert) |
| `Sensor_Task` | $5\text{--}10\,\text{ms}$ | `High` | DMA / Ringpuffer |
| `Autonomie_Comm_Task` | $100\,\text{ms}$ | `Medium` | Queue / Mailbox |
| `Telemetry_Task` | $500\text{--}1000\,\text{ms}$ | `Low` | Stream Buffer / UART |

### 5.2 Software-Schichtenarchitektur

```
[ Autonomie- & RL-Anwendung ]
          │
[ Plattformunabhängiges Sensor-/Aktuator-Interface ]
          │
[ Hardware Abstraction Layer (HAL) ]
          │
[ Board Support Package (BSP) & Treiber ]  ──> (GPIO, Timer, ADC, CAN, PWM)
```

### 5.3 Echtzeit-Primitiven
- **ISR & DMA:** Schnelle Datenerfassung ohne CPU-Overhead
- **Queues:** Entkoppelte Variablenübergabe zwischen Tasks
- **Mutex & PIP:** Schutz gemeinsamer Ressourcen ohne Prioritätsinversion

-----

## 6. Edge-AI Deployment & Safety Cage

### 6.1 Deployment-Pipeline
$$\text{PyTorch (PC)} \xrightarrow{\text{Export}} \text{ONNX (Float32)} \xrightarrow{\text{Quantisierung}} \text{Int8 C-Code / TFLite Micro}$$

### 6.2 Ressourcenvergleich (Modellgewichte)

| Format | Speicherbedarf Gewichte | Ziel-Laufzeitumgebung |
| --- | ---: | --- |
| **Float32 (Baseline)** | $\approx 74\,\text{KiB}$ | ONNX Runtime (Embedded Linux) |
| **Int8 (Quantisiert)** | $\approx 18\text{--}19\,\text{KiB}$ | CMSIS-NN / TFLite Micro (MCU) |

### 6.3 Safety Cage um die Policy
- RL-Policy liefert **nur Wunsch-Sollwerte** (*Requested Setpoints*)
- **Prüfkriterien des Sicherheitsfilters:**
  - Lenkwinkel- & Lenkraten-Begrenzung
  - Geschwindigkeits- & Beschleunigungs-Limits
  - Plausibilitäts-Check ($NaN$- & Ausreißer-Rejektion)
  - Sensor-Freshness & Timeout-Monitoring

-----

## 7. Zustandsautomat für den realen Betrieb (State Machine)

```mermaid @mermaid
stateDiagram-v2
    [*] --> Boot: Systemstart
    Boot --> SelfTest: Hardware Init OK
    SelfTest --> Calibration: Sensoren & Aktuatoren OK
    Calibration --> Ready: Zero-Pos & Pose gültig
    Ready --> Driving: Startfreigabe / Armed
    Driving --> Ready: Deaktivierung
    Driving --> Degraded: Teilweiser Sensorausfall
    Driving --> SafeStop: E-Stop / Deadline Miss / Timeout
    Degraded --> SafeStop: Fehler-Eskalation
    SafeStop --> Ready: Quittierung & Reset
```

### 7.1 Betriebsmodi & Rollen
- **Boot:** Speicher- & Treiberinitialisierung
- **SelfTest:** Hardware-, Sensor- & Spannungstests
- **Calibration:** Nullpunktabgleich für Lenksystem & IMU
- **Ready:** System betriebsbereit & scharfgeschaltet
- **Driving:** Autonomer Fahrbetrieb (**RL-Inferenz NUR hier aktiv!**)
- **Degraded:** Notbetrieb mit reduzierter Geschwindigkeit
- **SafeStop:** Sicherer Anhalteweg & Aktuator-Abschaltung

-----

## 8. Verifikation & Testleiter (V-Modell)

```mermaid @mermaid
flowchart LR
    UT[1. Unit Tests\nMathe & Kinematik] --> MIL[2. MiL / SiL\nSimulation & Fault Injection]
    MIL --> HIL[3. HiL Teststand\nFirmware & Watchdog Timing]
    HIL --> VEH[4. Fahrzeugtests\nAufgebockt ➔ Langsam ➔ Vollgas]
```

### 8.1 Teststufen im Überblick
- **Unit Tests:** Transformationen, Ackermann-Kinematik, Normalisierung, Clipping
- **MiL / SiL:** Stochastische Hindernisse, Reibwertvariationen, Sensor-Dropouts
- **HiL (Hardware-in-the-Loop):** Messung von WCET, Jitter, SRAM-Bedarf & CAN-Latenzen
- **Fahrzeugtests:** Aufgebockte Räder $\rightarrow$ Langsame Fahrt $\rightarrow$ Hindernisse $\rightarrow$ Fehlerinjektion auf Teststrecke

-----

## 9. Dependability & Fehlertoleranz

### 9.1 Fehler-Kaskade
$$\text{Fehlerursache (Fault)} \longrightarrow \text{Systemstörung (Error)} \longrightarrow \text{Dienstausfall (Failure)}$$

### 9.2 Schutzmaßnahmen
- **Hardware-Watchdog:** Automatischer Reset bei Software-Hängern
- **Heartbeat:** Linux-MCU Kontrollsignal-Überwachung
- **Informationelle Redundancy:** CRC32, Sequenznummern, Modell-Hashes
- **Funktionale Redundanz:** Rad-Encoder + IMU Fusionsfilter
- **Fail-Safe:** Hardwareseitiger Abschaltpfad zur Deaktivierung der Motorbrücke

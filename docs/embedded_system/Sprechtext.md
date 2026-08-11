# Sprechtext -- Embedded Concept Präsentation

> Orientierungshilfe für den mündlichen Vortrag zu jeder Folie der LiaScript-Präsentation.

---

## Startseite (Titel & Kernfakten)

Ich stelle heute mein Embedded-Konzept vor: den Transfer eines autonomen Rennroboters von der Simulation auf ein reales eingebettetes System.

Das Fahrzeug wird für den WRO Future Engineers Wettbewerb entwickelt. Die Steuerung basiert auf Deep Reinforcement Learning — konkret PPO — und wird komplett in einer Webots-Simulation trainiert.

Die beiden GIFs zeigen die Simulation in Aktion: Einmal die 3D-Ansicht der Rennstrecke mit Hindernissen, und einmal die ICP-basierte Lokalisierung mit dem Pfadvisualisierer.

Die Software-Pipeline besteht aus vier sequenziellen Stufen: Perzeption, Zustandsschätzung, Planung per RL-Policy, und Regelung. Der Beobachtungsvektor hat 15 Dimensionen und ist auf den Bereich minus eins bis plus eins normalisiert. Die Policy gibt zwei kontinuierliche Stellgrößen aus: Lenkwinkeländerung und Geschwindigkeitsänderung.

Das trainierte Modell hat knapp 19.000 Parameter und wird als ONNX-Datei exportiert — das sind etwa 74 Kilobyte im Float32-Format.

---

## Folie 1: Sim-to-Real-Gap

Jetzt zur zentralen Herausforderung: dem Sim-to-Real-Gap.

In der Simulation haben wir perfekte Bedingungen — rauschfreie Sensoren, exakte Zeitstempel, ideale Aktuatoren. Auf dem realen System sieht das ganz anders aus: Sensorrauschen, Bias, Latenzen und Dropouts. Die Aktuatoren zeigen Totzonen, Spiel und Reifenschlupf.

Die Ressourcen sind ebenfalls ein Problem: Statt einem unbegrenzten PC haben wir begrenzten Flash, SRAM und Rechenleistung.

Besonders wichtig — und deshalb mit Ausrufezeichen markiert — ist die Geschwindigkeit. In der Simulation kommt sie direkt aus der Physik-Engine. Im realen System muss sie über Rad-Encoder und IMU geschätzt werden. Das ist übrigens ein Beispiel für aktive Redundanz: Zwei unabhängige Messprinzipien liefern die gleiche Größe.

Um die Robustheit zu verbessern, könnte man im Training Domain Randomization einsetzen, Sensorrauschen und Latenzen injizieren, und Sensor-Dropouts simulieren. Das ist aktuell noch nicht umgesetzt.

---

## Folie 2: Echtzeit-Anforderungen & Timing-Budgets

Die Regelfrequenz beträgt 10 Hertz — das ergibt ein Budget von 100 Millisekunden pro Zyklus.

Die 100 Millisekunden teilen sich auf: 15 ms für die Sensor-Synchronisation, 40 ms für die ICP-Lokalisierung — der mit Abstand größte Posten —, 15 ms für die ONNX-Inferenz, 10 ms für Motor- und Lenkansteuerung, und 20 ms Jitter-Reserve.

Zur Echtzeit-Klassifikation: Die RL-Policy selbst ist weiche bis feste Echtzeit. Wenn sie mal ein paar Millisekunden zu spät kommt, verschlechtert sich die Fahrqualität, aber es gibt nicht sofort einen Crash. Der Sicherheitsfilter und der Not-Aus hingegen sind harte Echtzeit — die müssen deterministisch ihre Deadlines einhalten.

---

## Folie 3: Architekturoptionen & Zielarchitektur

Wir haben drei Architekturvarianten betrachtet.

Variante A, ein reiner Embedded-Linux-Rechner: Python und ONNX laufen direkt, und es gibt genügend RAM für LiDAR und ICP. Das Problem ist der fehlende harte Determinismus ohne PREEMPT_RT-Kernel.

Variante B, ein reiner Mikrocontroller: Ein ATmega328 scheidet sofort aus — er hat nur 2 Kilobyte SRAM, das Modell braucht aber 74 Kilobyte. Der STM32F401 mit 96 Kilobyte SRAM könnte ein quantisiertes Int8-Netz mit 19 Kilobyte theoretisch fassen, aber die ICP-Lokalisierung und die Kameraverarbeitung sprengen den RAM. Auch der ESP32-S3 mit 512 Kilobyte SRAM kann die Inferenz allein bewältigen, aber die komplette Wahrnehmungspipeline mit LiDAR-Pointclouds überfordert ihn.

Deshalb Variante C — die heterogene Architektur: Ein Leistungsrechner unter Embedded Linux übernimmt die rechenintensiven Aufgaben wie Kamera, LiDAR, ICP und ONNX-Inferenz. Ein FreeRTOS-Mikrocontroller übernimmt die sicherheitskritischen und zeitdeterministischen Aufgaben wie Encoder-Auswertung, PWM-Erzeugung, Safety Filter und Watchdog. Die beiden kommunizieren über CAN oder UART mit CRC-Prüfsummen und Timeouts.

---

## Folie 4: RTOS Task-Struktur & Hardware-Abstraktion

Die FreeRTOS-Tasks auf dem Sicherheits-Controller sind nach Priorität gestaffelt.

Der Safety-Task hat die höchste Priorität und reagiert auf Events innerhalb von 1 bis 5 Millisekunden. Danach kommen die Aktuator- und Sensor-Tasks mit hoher Priorität und 5 bis 10 Millisekunden Periode. Die Kommunikation mit dem Leistungsrechner läuft alle 100 Millisekunden auf mittlerer Priorität. Telemetrie und Logging haben die niedrigste Priorität.

Die Software-Architektur ist in Schichten aufgebaut: ganz oben die Autonomie-Anwendung, darunter ein plattformunabhängiges Sensor- und Aktuator-Interface, dann der Hardware Abstraction Layer, und ganz unten das Board Support Package mit den Treibern.

Als Echtzeit-Primitiven werden DMA für die schnelle Datenerfassung, Queues für die entkoppelte Kommunikation zwischen Tasks, und Mutexe mit Priority Inheritance zur Vermeidung von Prioritätsinversion eingesetzt.

---

## Folie 5: Edge-AI Deployment

Die Deployment-Pipeline geht von PyTorch über ONNX im Float32-Format bis zur optionalen Int8-Quantisierung für den Einsatz auf einem Mikrocontroller.

Im Float32-Format belegen die Gewichte etwa 74 Kilobyte und laufen auf dem Leistungsrechner mit ONNX Runtime. Quantisiert auf Int8 reduziert sich das auf 18 bis 19 Kilobyte — dann wäre ein Einsatz auf einer MCU mit CMSIS-NN denkbar.

Wichtig ist der Safety Cage: Die Policy liefert nur Wunsch-Sollwerte. Diese werden vom Sicherheitsfilter geprüft — auf Lenkwinkel- und Geschwindigkeitsgrenzen, NaN-Rejektion, Datenaktualität und Timeouts. Bei Kommunikationsabriss oder kritischen Fehlern greift sofort der Emergency Stop.

---

## Folie 6: Zustandsautomat

Der Zustandsautomat beschreibt die Betriebsphasen des realen Fahrzeugs.

Nach dem Einschalten durchläuft das System Boot und SelfTest. In der Calibration-Phase wird der Nullpunkt der IMU und der Lenkung abgeglichen — und hier kommt auch die Initiallokalisierung per OpenCV zum Einsatz: Über Farberkennung wird der Startbalken erkannt und die Fahrtrichtung — im oder gegen den Uhrzeigersinn — bestimmt.

Danach ist das System im Ready-Zustand. Erst mit der Startfreigabe wechselt es in den Driving-Zustand — und nur dort ist die RL-Inferenz aktiv.

Bei einem teilweisen Sensorausfall geht es in den Degraded-Modus mit reduzierter Geschwindigkeit. Bei einem Not-Aus, Timeout oder einer verpassten Deadline wird sofort in den SafeStop übergegangen — das bedeutet sicheres Anhalten und Abschaltung der Aktuatoren.

---

## Folie 7: Verifikation & Test-Pyramide

Die Test-Pyramide zeigt, wie wir das Projekt auf drei Ebenen testen würden.

Die Basis sind Host-Tests — rein auf dem PC, ohne Hardware. Hier testen wir die Ackermann-Kinematik, die Koordinatentransformationen, die 15D-Normalisierung und die ONNX-Referenzausgaben. Das geht blitzschnell und kann tausendfach in einer CI-Pipeline laufen.

In der Mitte stehen die On-Target-Tests auf einem echten STM32. Hier überprüfen wir den Safety Filter, PWM-Timing, ADC-Abtastung, Watchdog-Auslösung und CAN-Frame-Integrität — alles über JTAG oder SWD.

An der Spitze stehen die HIL-Tests mit dem gesamten Fahrzeug: aufgebockt anfangen, dann langsame Geradeausfahrt, dann Kurven, Hindernisse und schließlich gezielte Fehlerinjektion auf der Teststrecke.

---

## Folie 8: Dependability & Fehlertoleranz

Zum Abschluss die Zuverlässigkeit — am konkreten Beispiel.

Die Fehler-Kaskade zeigt sich zum Beispiel bei einem IMU-Kabelbruch: Der Fehler führt dazu, dass der Fusionsfilter divergiert — daraus wird eine Systemstörung — und das Fahrzeug verlässt die Strecke. Um solche Kaskaden zu unterbrechen, setzen wir fünf konkrete Maßnahmen ein:

Erstens, ein Hardware-Watchdog auf der MCU — wenn die Software hängt, gibt es einen automatischen Reset. Zweitens, ein Heartbeat zwischen Leistungsrechner und MCU — fällt Linux aus, erkennt die MCU das und löst den Safe Stop aus. Drittens, informationelle Redundanz über CRC32 auf allen Nachrichten, Sequenznummern und einen Modell-Hash bei Systemstart. Viertens, funktionale Redundanz durch die parallele Geschwindigkeitsmessung über Encoder und IMU — das ist die aktive Redundanz, die wir auch in der Sim-to-Real-Tabelle gesehen haben. Und fünftens, ein physikalischer Hardware-Abschaltpfad: Ein MOSFET auf der MCU trennt bei kritischem Fehler die Motorbrücke komplett stromlos.

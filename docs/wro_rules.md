# WRO 2026 Future Engineers - Technical Specifications

This document contains only the strict technical, physical, and algorithmic rules required for robot software architecture and simulation environment setup.

---

## 1. Environment & Track Layout
- **Total Dimensions:** 3.0 m x 3.0 m (inner boundary dimensions).
- **Wall Height:** 100 mm. Wall surfaces facing the track are black.
- **Track Geometry:** Divided into 8 sections: 4 straight sections and 4 corner sections.
- **Inner Boundary Variance (Opening Run):** The distance between inner and outer boundaries varies between 600 mm (min) and 1000 mm (max) depending on the randomized layout.
- **Inner Boundary Variance (Obstacle Run):** The inner boundary is a fixed 1.0 m x 1.0 m square. The track width is constantly 1000 mm.
- **Start-Finish Line:** Located in the straight section under the WRO logo (Süd/South).

---

## 2. Obstacle Specifications (Obstacle Run Only)
- **Geometry:** Rectangular cuboid pillars measuring 45 mm x 45 mm x 100 mm (W x D x H).
- **Red Pillars:** Traffic Red (RAL 3020).
- **Green Pillars:** Pure Green (RAL 6037).
- **Distribution:** Exactly 1 pillar per straight section (total of 4 pillars on the track).
- **Pillar Tracking/Shifting:** A pillar is considered hit/invalidated if it is completely pushed out of its circular floor marking.

### Kinematic Steering Logic (Crucial for Path Planning)
- **Red Pillars:** Must be passed completely on the **RIGHT** side.
- **Green Pillars:** Must be passed completely on the **LEFT** side.

---

## 3. Vehicle Physical Constraints
- **Maximum Dimensions:** 300 mm x 200 mm x 300 mm (L x W x H). Dimensions must remain unchanged during the run (no active mechanical transformations).
- **Maximum Weight:** 1.5 kg.
- **Drivetrain Layout:** Must use exactly 4 normal wheels (2 axles). Omnidirectional or ball wheels are strictly illegal.
- **Propulsion:** Max 2 motors. Must drive the axle directly via a mechanical connection. Single-wheel differential driving (skid-steer) is strictly illegal. 
- **Steering:** 1 or 2 steering axles. Controlled by exactly 1 motor (Ackermann geometry).
- **Allowed Drive Types:** Front-Wheel Drive (FWD), Rear-Wheel Drive (RWD), or All-Wheel Drive (AWD with non-independent axle speeds).

---

## 4. Control, Software & Boot Sequences
- **Autonomy:** The vehicle must drive completely autonomously. Wireless communication (Wi-Fi, Bluetooth, RF) must be deactivated.
- **Boot/Init Constraint:** Upon power-on, the robot must enter a passive standby state. No sensor measurements, odometry calculations, or code execution are allowed until the start button is physically pressed.
- **Mission Execution Goal:** Complete exactly 3 consecutive laps error-free within a maximum time limit of 3 minutes (180 seconds).

---

## 5. Optional Advanced Task: Parking Challenge (Obstacle Run Only)
- **Park Location:** Located exclusively in the Start-Finish section, bounded by two Telemagenta (RAL 4010) walls.
- **Magenta Wall Dimensions:** 200 mm x 100 mm x 19 mm (W x H x Thickness).
- **Parking Slot Length:** Dynamically calculated based on the vehicle size:
  $$\text{Slot Length} = 1.5 \times \text{Robot Length}$$
  *(Note: Robot length is measured only for parts below 100 mm in height).*
- **Collision Rules:** Magenta walls must **NEVER** be touched during the lap run, unparking, or parking phase. Touching causes an immediate run termination.
- **Sequence:** After completing 3 laps, the vehicle must enter the parking slot.
- **Definition "Fully Parked":** The entire 2D top-down footprint of the vehicle must stop completely inside the designated parking box.
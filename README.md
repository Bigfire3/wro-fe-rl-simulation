# WRO Future Engineers - Webots Simulation

This repository contains the Webots simulation for the WRO Future Engineers category.

## Project Structure

- `controllers/`: Robot controllers (including `wro_driver`)
- `worlds/`: Webots world files (`track.wbt`)
- `protos/`: Custom robot and object prototypes
- `libraries/`: Shared libraries
- `plugins/`: Simulation plugins

## Features

- **Ackermann Steering**: Implementation of realistic steering geometry.
- **SLAM & Localization**: Lidar-based SLAM and visualization.
- **Autonomous Navigation**: Wall-following and corridor navigation algorithms.

## Getting Started

1. Install [Webots](https://cyberbotics.com/).
2. Clone this repository.
3. Open `worlds/track.wbt` in Webots.
4. The controller `wro_driver` should start automatically.

---
*Created for WRO Future Engineers.*

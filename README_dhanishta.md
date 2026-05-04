# Ultra-HD Solar Panel NDT Pro Simulator

This is a professional-grade Non-Destructive Testing (NDT) simulator designed for microcrack detection in solar panels. It uses high-performance physical simulation and industrial signal processing techniques.

## Core Features
- **Taichi-Accelerated Physics:** Real-time 2D wave propagation using a parallelized FDTD solver.
- **Industrial DSP:** Matched filtering (pulse compression) and Hilbert transforms for high-fidelity A-scan analysis.
- **Hardware-Accelerated UI:** A responsive desktop interface built with DearPyGui, featuring live wavefield textures and interactive plots.
- **Bayesian Localization:** Real-time defect mapping and probability analysis.

## Project Structure
- `main_ndt.py`: The main entry point and UI controller.
- `ndt_engine.py`: The Taichi-based physical simulation engine.
- `ndt_dsp.py`: Signal processing modules (Chirp, Correlation, Envelopes).
- `requirements.txt`: Project dependencies.

## How to Run
1.  **Install Dependencies:**
    ```bash
    pip install taichi scipy dearpygui numpy
    ```
2.  **Execute the Application:**
    ```bash
    python main_ndt.py
    ```

## Engineering Concepts
- **Surface Acoustic Waves (SAW):** The simulator models the propagation of ultrasonic waves across the silicon substrate.
- **Pulse Compression:** Long chirp pulses are compressed into sharp peaks using cross-correlation, allowing for high-resolution defect detection despite noise.
- **A-Scan Analysis:** The analytic signal (via Hilbert transform) provides a clear envelope for detecting echoes from microcracks.

# ESP32/Wokwi Controller Logic Demonstrator

This project demonstrates the gate-control validity checks, safe-mode behavior,
and one-hour AWD interlock on a virtual ESP32. The green, red, and yellow LEDs
represent the requested gate state and safe-mode status.

This is a controller-logic demonstrator, not an ESP32 implementation of the
Random Forest. `requested_open` is supplied as a test input over the serial
monitor. Sensor readings are also supplied as test data; no physical sensor,
gate, farm, or field-calibrated actuator is modeled.

## Open and run in VS Code

1. Install the PlatformIO IDE and Wokwi for VS Code extensions.
2. Open this `wokwi` directory as the VS Code workspace.
3. Build the `esp32dev` PlatformIO environment.
4. Start the Wokwi simulation from the Wokwi extension.
5. Use the serial monitor at 115200 baud and send one command per line.

The project uses Wokwi's documented `wokwi.toml` layout and expects PlatformIO
to build `.pio/build/esp32dev/firmware.bin` and `firmware.elf`.

## Serial protocol

```text
SAMPLE <stage> <depth_cm> <request_open:0|1> <age_ms> <fault:0|1>
```

Stages are `TRANSPLANT`, `VEG`, `FLOWER`, and `RIPEN`. `age_ms` is an explicit
test value: readings older than 3,600,000 ms enter safe mode. `fault` simulates
an externally detected sensor fault. `RESET` clears the freeze detector and
returns the simulated gate to closed. `HELP` prints the command format.

Examples:

```text
RESET
SAMPLE VEG -15 1 0 0
SAMPLE VEG 4.5 1 0 0
SAMPLE FLOWER 0.2 0 0 0
SAMPLE VEG -4 1 3600001 0
SAMPLE VEG -4 1 0 1
```

Expected controller outcomes include:

- `SAMPLE VEG -15 1 0 0`: opens because closing would project below -15 cm.
- `SAMPLE VEG 4.5 1 0 0`: closes because opening would project above +5 cm.
- `SAMPLE FLOWER 0.2 0 0 0`: opens because closing would project below 0 cm.
- A stale or fault-flagged sample enters safe mode and holds the prior gate state.
- Three consecutive identical depth samples enter safe mode on the third sample.

The controller applies the study's configured projection: opening raises depth
by 1.0 cm/hour and closing lowers it by 0.5 cm/hour. It rejects invalid stages,
depths outside -20 to +10 cm, stale readings, and fault-flagged samples. Three
consecutive identical depth samples trigger the simulated frozen-sensor check.
This is a single-input demonstration of the project logic, not a complete
multi-sensor fault detector.

## Interpretation

The LEDs and serial messages validate state transitions in the virtual firmware
when a Wokwi run is performed. They do not provide measured hardware latency,
physical actuator response, calibration evidence, or public-health outcomes.
For thesis reporting, distinguish project implementation/build evidence from a
completed Wokwi run and from physical hardware-in-the-loop results.

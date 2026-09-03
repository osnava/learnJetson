# Q: Button-header pins 3/4/7 — and what voltage domain?

The button header is J14. Per `devkit-carrier-spec.md` §3.4 Table 3-4, p. 28:
pin 3 is the debug-UART receive line "UART2_RXD (DEBUG)", module ball 238,
Input, 3.3V; pin 4 is "UART2_TXD (DEBUG)", ball 236, Output, 3.3V; pin 7 is
Ground. The header's domains are mixed — `SYS_RESET*` (pin 8) and
`FORCE_RECOVERY*` (pin 10) are 1.8V inputs, while the sleep/wake LED pair
runs on 5V: `devkit-carrier-spec.md` §3.4 (p. 28) — "PC_LED-: Connects to LED
Cathode to indicate System Sleep/Wake (Off when system in sleepmode)".

For a USB-TTL console cable that means: TX→pin 3, RX→pin 4, GND→pin 7 at
115200 8N1. Note pin 3 is the board's UART2_RXD — mind adapter-vs-board
TX/RX naming.

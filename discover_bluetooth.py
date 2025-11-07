#!/usr/bin/env python3
"""Simple script to discover Bluetooth devices and print their names."""

import asyncio
from bleak import BleakScanner


async def discover_devices(timeout: float = 10.0) -> None:
    """Discover nearby Bluetooth devices and print their names."""
    print(f"Scanning for Bluetooth devices (timeout: {timeout}s)...")
    print("-" * 50)

    devices = await BleakScanner.discover(timeout=timeout)

    if not devices:
        print("No Bluetooth devices found.")
        print("Make sure Bluetooth is enabled and devices are in range.")
        return

    print(f"\nFound {len(devices)} device(s):\n")
    for i, device in enumerate(devices, 1):
        name = device.name or "Unknown"
        address = device.address
        print(f"{i}. {name} ({address})")


if __name__ == "__main__":
    try:
        asyncio.run(discover_devices())
    except KeyboardInterrupt:
        print("\nScan interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Bluetooth is enabled on your system.")


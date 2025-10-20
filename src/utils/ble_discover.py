import asyncio
import argparse
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterable, Tuple

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData


async def _active_scan_collect(timeout: float = 10.0) -> list[Tuple[BLEDevice, AdvertisementData | None]]:
    """Fallback active scan using a detection callback to collect advertisements.

    This helps on platforms where a one-shot discover() can miss devices.
    """
    collected: dict[str, Tuple[BLEDevice, AdvertisementData | None]] = {}

    def _on_detect(device: BLEDevice, adv: AdvertisementData) -> None:  # type: ignore[override]
        collected[device.address] = (device, adv)

    scanner = BleakScanner(detection_callback=_on_detect)
    try:
        await scanner.start()
        try:
            await asyncio.sleep(timeout)
        finally:
            await scanner.stop()
    except Exception:
        # If scanning fails entirely, return what we have (likely empty)
        pass
    return list(collected.values())


def _format_service_uuids(uuids: Iterable[str] | None) -> str:
    if not uuids:
        return "-"
    return ", ".join(sorted(set(uuids)))


def _format_mfr(manufacturer_data: dict[int, bytes] | None) -> str:
    if not manufacturer_data:
        return "-"
    # Show company IDs in hex and first few bytes
    parts: list[str] = []
    for company_id, data in manufacturer_data.items():
        preview = data[:8].hex()
        parts.append(f"0x{company_id:04X}:{preview}")
    return ", ".join(parts)


@asynccontextmanager
async def _connect_with_timeout(device: BLEDevice, timeout: float) -> AsyncIterator[BleakClient | None]:
    client = BleakClient(device, timeout=timeout)
    try:
        await asyncio.wait_for(client.connect(), timeout=timeout)
        yield client
    except Exception:
        # Swallow connection errors; discovery should continue.
        yield None
    finally:
        if client.is_connected:
            try:
                await client.disconnect()
            except Exception:
                pass


async def _discover(timeout: float = 6.0) -> list[Tuple[BLEDevice, AdvertisementData | None]]:
    """Discover nearby devices and normalize return to list[(BLEDevice, AdvertisementData|None)].

    Bleak versions/platforms vary in discover() return shape when return_adv=True.
    This function handles:
    - dict[BLEDevice, AdvertisementData]
    - list[(BLEDevice, AdvertisementData)]
    - list[(BLEDevice, AdvertisementData, ...)]
    - list[ScanResult-like objects with .device and .advertisement_data]
    - list[BLEDevice] (when return_adv unsupported)
    """

    try:
        raw = await BleakScanner.discover(timeout=timeout, return_adv=True)
    except TypeError:
        # Fallback for older Bleak: discover() without adv
        devices = await BleakScanner.discover(timeout=timeout)
        return [(dev, None) for dev in devices]

    results: list[Tuple[BLEDevice, AdvertisementData | None]] = []

    # Case: mapping
    if hasattr(raw, "items"):
        try:
            for dev, adv in raw.items():
                if isinstance(dev, BLEDevice):
                    results.append((dev, adv))
            return results
        except Exception:
            pass

    # Case: iterable (list/tuple)
    try:
        for entry in raw:  # type: ignore[assignment]
            # ScanResult-like with attributes
            dev = getattr(entry, "device", None)
            adv = getattr(entry, "advertisement_data", None)
            if isinstance(dev, BLEDevice):
                results.append((dev, adv))
                continue

            # Tuple-like
            if isinstance(entry, (list, tuple)):
                if len(entry) >= 2 and isinstance(entry[0], BLEDevice):
                    dev0 = entry[0]
                    adv0 = entry[1] if isinstance(entry[1], AdvertisementData) else None
                    results.append((dev0, adv0))
                elif len(entry) == 1 and isinstance(entry[0], BLEDevice):
                    results.append((entry[0], None))
                else:
                    # Unknown tuple shape; skip
                    continue
                continue

            # Bare BLEDevice
            if isinstance(entry, BLEDevice):
                results.append((entry, None))
                continue
        return results
    except Exception:
        # As a last resort, try treating it as a simple list of devices
        pass

    try:
        return [(dev, None) for dev in raw]  # type: ignore[arg-type]
    except Exception:
        return []


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLE discovery and table output")
    parser.add_argument(
        "--timeout",
        type=float,
        default=6.0,
        help="Quick scan timeout seconds (before active fallback)",
    )
    parser.add_argument(
        "--active",
        action="store_true",
        default=True,
        help="Use active scan (callback collection) instead of quick scan",
    )
    parser.add_argument(
        "--active-timeout",
        type=float,
        default=12.0,
        help="Active scan timeout seconds if quick scan returns nothing",
    )
    parser.add_argument(
        "--min-rssi",
        type=int,
        default=None,
        help="Minimum RSSI (dBm) to include device",
    )
    parser.add_argument(
        "--filter-manufacturer",
        type=str,
        default="0x0059,0x239A",
        help=(
            "Comma-separated Bluetooth company IDs to include (e.g. 0x0059 for Nordic, "
            "0x239A for Adafruit). Use 'any' to disable."
        ),
    )
    parser.add_argument(
        "--filter-uuid-contains",
        type=str,
        default="19B100,1214",
        help=(
            "Comma-separated substrings that any UUID should contain to match. "
            "Defaults aim to match ArduinoBLE example base UUIDs (19B100...1214)."
        ),
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Also show non-matching devices in the table",
    )
    return parser.parse_args()


def _parse_manufacturer_filter(arg: str) -> set[int] | None:
    if not arg or arg.lower() == "any":
        return None
    ids: set[int] = set()
    for part in arg.split(","):
        part = part.strip()
        try:
            if part.lower().startswith("0x"):
                ids.add(int(part, 16))
            else:
                ids.add(int(part))
        except ValueError:
            continue
    return ids or None


def _parse_uuid_filters(arg: str) -> list[str] | None:
    if not arg:
        return None
    parts = [p.strip().lower() for p in arg.split(",") if p.strip()]
    return parts or None


async def main() -> None:
    args = _parse_args()

    if args.active:
        print("Scanning for Bluetooth LE devices... (single active scan)")
        results = await _active_scan_collect(timeout=args.active_timeout)
    else:
        print("Scanning for Bluetooth LE devices... (single quick scan)")
        results = await _discover(timeout=args.timeout)

    if not results:
        print("No devices found. Ensure Bluetooth is ON and Terminal has Bluetooth permission.")
        print("- On macOS: System Settings → Privacy & Security → Bluetooth → enable for your shell/IDE")
        print("- Try moving devices closer and make sure they're advertising (powered on/pairing mode)")
        return

    # Build concise rows directly from advertisements only (no GATT connect)
    # Row: (name, connect_id, rssi, advertised_service_uuids)
    rows: list[tuple[str, str, int, str]] = []
    for device, adv in results:
        name = device.name or "Unknown"
        connect_id = device.address  # What you pass to BleakClient(...)
        rssi_val = getattr(adv, "rssi", getattr(device, "rssi", -999)) or -999
        adv_uuids = getattr(adv, "service_uuids", []) or []
        uuids_str = ", ".join(sorted(set(adv_uuids))) if adv_uuids else "-"
        rows.append((name, connect_id, int(rssi_val), uuids_str))

    # Sort by strongest signal first
    rows.sort(key=lambda r: -r[2])

    # Print one consolidated table: Name | Connect ID | RSSI | UUIDs
    print("\nDevices (Name | Connect ID | RSSI | UUIDs):")
    name_width = max(max((len(r[0]) for r in rows), default=4), len("Name"))
    id_width = max(max((len(r[1]) for r in rows), default=10), len("Connect ID"))
    rssi_width = max(len("RSSI"), 4)
    header = f"{'Name'.ljust(name_width)}  {'Connect ID'.ljust(id_width)}  {'RSSI'.ljust(rssi_width)}  UUIDs"
    print(header)
    print(f"{'-' * name_width}  {'-' * id_width}  {'-' * rssi_width}  {'-' * 5}")
    for name, connect_id, rssi_val, uuids_str in rows:
        print(f"{name.ljust(name_width)}  {connect_id.ljust(id_width)}  {str(rssi_val).ljust(rssi_width)}  {uuids_str}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass



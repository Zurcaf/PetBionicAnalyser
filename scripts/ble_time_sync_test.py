#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
from typing import Optional

from bleak import BleakClient, BleakScanner

SERVICE_UUID = "14f16000-9d9c-470f-9f6a-6e6fe401a001"
CONTROL_UUID = "14f16001-9d9c-470f-9f6a-6e6fe401a001"
STATUS_UUID = "14f16002-9d9c-470f-9f6a-6e6fe401a001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BLE time sync smoke test for petBionics")
    parser.add_argument("--name", default="petBionics", help="BLE device name to scan")
    parser.add_argument("--address", default=None, help="BLE MAC/address (skip scan if provided)")
    parser.add_argument("--duration", type=float, default=8.0, help="Seconds to listen for status notifications")
    return parser.parse_args()


async def resolve_device(name: str, address: Optional[str]):
    if address:
        return address

    print(f"[scan] searching for BLE name '{name}'...")
    device = await BleakScanner.find_device_by_filter(
        lambda d, _: (d.name or "") == name,
        timeout=12.0,
    )
    if not device:
        raise RuntimeError(f"Device '{name}' not found")

    print(f"[scan] found {device.name} @ {device.address}")
    return device.address


async def run() -> None:
    args = parse_args()
    target = await resolve_device(args.name, args.address)

    first_status = asyncio.Event()

    def on_status(_: str, data: bytearray) -> None:
        now_host_ms = int(time.time() * 1000)
        payload = data.decode("utf-8", errors="replace")
        print(f"[notify] {payload}")

        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return

        device_time_ms = int(obj.get("time_ms", 0) or 0)
        synced = bool(obj.get("time_synced", False))
        if synced and device_time_ms > 0:
            err_ms = now_host_ms - device_time_ms
            print(
                "[sync] host_ms=%d device_ms=%d error_ms=%+d"
                % (now_host_ms, device_time_ms, err_ms)
            )

        first_status.set()

    async with BleakClient(target) as client:
        print(f"[conn] connected={client.is_connected}")
        services = client.services
        has_service = any(s.uuid.lower() == SERVICE_UUID for s in services)
        print(f"[gatt] service_present={has_service}")

        host_epoch_s = int(time.time())
        cmd = f"TIME={host_epoch_s}"
        t0 = time.perf_counter()
        await client.write_gatt_char(CONTROL_UUID, cmd.encode("ascii"), response=False)
        t1 = time.perf_counter()
        print(f"[write] sent '{cmd}' in {(t1 - t0) * 1000.0:.1f} ms")

        await client.start_notify(STATUS_UUID, on_status)
        print(f"[notify] listening for {args.duration:.1f} s")

        try:
            await asyncio.wait_for(first_status.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            print("[warn] no notify received in first 3 seconds")

        await asyncio.sleep(max(0.0, args.duration))
        await client.stop_notify(STATUS_UUID)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[stop] interrupted by user")

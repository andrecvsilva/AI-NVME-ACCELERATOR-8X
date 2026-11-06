"""Low-memory summarizer for tracerpt XML exports of kernel FileIO/DiskIO ETW."""

from __future__ import annotations

import ctypes
import math
import ntpath
import random
import xml.etree.ElementTree as ET
from pathlib import Path


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _fields(event: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in event.iter():
        if _local(node.tag) != "data":
            continue
        name = next((v for k, v in node.attrib.items() if _local(k) == "name"), "")
        if name and node.text:
            result[name.casefold().replace("_", "")] = node.text.strip()
    return result


def _metadata(event: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in event.iter():
        name = _local(node.tag)
        if name in {"provider", "eventname", "eventid", "opcode", "task", "timecreated"}:
            if name in result:
                # Keep the numeric System metadata; RenderingInfo may contain
                # a localized name such as "DiskReadInit" for the same tag.
                continue
            value = next((v for k, v in node.attrib.items() if _local(k) in {"name", "systemtime"}), "")
            value = value or (node.text or "").strip()
            if value:
                result[name] = value.casefold()
    return result


def _number(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    try:
        return int(value, 0)
    except ValueError:
        try:
            return int(value, 16)
        except ValueError:
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else None


def _iter_events(xml_path: Path):
    # Clearing alone leaves each Event attached to its parent, so a large
    # tracerpt export still grows to the size of the entire document. Remove
    # each completed event from its parent after the caller has extracted it.
    stack: list[ET.Element] = []
    for action, element in ET.iterparse(xml_path, events=("start", "end")):
        if action == "start":
            stack.append(element)
            continue
        if _local(element.tag) == "event":
            yield element
            element.clear()
            if len(stack) > 1:
                try:
                    stack[-2].remove(element)
                except ValueError:
                    pass
        if stack:
            stack.pop()


def _qpc_frequency() -> int | None:
    if not hasattr(ctypes, "windll"):
        return None
    value = ctypes.c_longlong()
    if ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(value)):
        return int(value.value)
    return None


def analyze_tracerpt_xml(xml_path: str | Path, gguf_path: str | Path,
                         *, qpc_frequency: int | None = None,
                         latency_sample_limit: int = 4096) -> dict:
    """Attribute completed physical disk reads to GGUF FileObject handles.

    Uses the FileObject->FileName mapping emitted by Kernel FileIO and the
    FileObject/TransferSize/HighResResponseTime fields from DiskIO completion
    events. The optional latency sample is bounded to keep memory use low.
    """
    xml_path = Path(xml_path)
    target = str(Path(gguf_path).resolve()).replace("/", "\\").casefold()
    target_drive, target_tail = ntpath.splitdrive(target)
    target_tail = target_tail.rstrip("\\")
    frequency = qpc_frequency if qpc_frequency is not None else _qpc_frequency()
    # FileObject values are kernel pointers and may be reused after close.
    # Keep the mapping in event order so a later file cannot overwrite the
    # earlier mapping before its DiskIO completions are processed.
    names: dict[int, str] = {}
    physical_count = physical_bytes = 0
    physical_latencies: list[float] = []
    logical_count = logical_bytes = 0
    logical_file_objects: set[int] = set()
    rng = random.Random(0)

    for event in _iter_events(xml_path):
        fields = _fields(event)
        meta = _metadata(event)
        file_object = _number(fields.get("fileobject"))
        file_key = _number(fields.get("filekey"))
        file_name = fields.get("filename") or fields.get("name") or fields.get("openpath")
        if file_object is not None and file_name:
            names[file_object] = file_name.replace("/", "\\").casefold()

        event_name = " ".join(meta.get(key, "") for key in ("provider", "eventname", "task"))
        opcode = _number(meta.get("opcode"))
        event_id = _number(meta.get("eventid"))
        name = names.get(file_object) if file_object is not None else None
        if file_key is not None:
            name = names.get(file_key, name)
        matched = name == target if name is not None else False
        if name and not matched and target_drive and target_tail:
            # Kernel ETW may report \Device\HarddiskVolumeN while the requested
            # path uses a drive letter. Compare the full path below the volume.
            matched = name.endswith(target_tail)
        if not matched:
            continue

        # FileIo_ReadWrite: opcode 67 means a file read and IoSize is requested
        # byte count. Keep this distinct from confirmed physical disk reads.
        iosize = _number(fields.get("iosize"))
        if iosize is not None and (opcode == 67 or event_id == 67 or "read" in event_name):
            logical_count += 1
            logical_bytes += max(0, iosize)
            if file_object is not None:
                logical_file_objects.add(file_object)

        # DiskIo_TypeGroup1 read completion: opcode 10, TransferSize and a
        # matching FileObject. This is the physical-I/O attribution used here.
        transfer = _number(fields.get("transfersize"))
        disk_number = _number(fields.get("disknumber"))
        is_read = opcode == 10 or event_id == 10 or "diskio_read" in event_name
        if transfer is None or disk_number is None or not is_read:
            continue
        physical_count += 1
        physical_bytes += max(0, transfer)
        ticks = _number(fields.get("highresresponsetime"))
        if ticks is not None and frequency and frequency > 0:
            latency_ms = ticks * 1000.0 / frequency
            if len(physical_latencies) < latency_sample_limit:
                physical_latencies.append(latency_ms)
            else:
                index = rng.randrange(physical_count)
                if index < latency_sample_limit:
                    physical_latencies[index] = latency_ms

    ordered = sorted(physical_latencies)

    def percentile(p: float):
        if not ordered:
            return None
        return round(ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)], 4)

    return {
        "available": True,
        "source": "tracerpt XML; Kernel FileIO FileObject mapping joined to DiskIO completion FileObject",
        "target_file_path": str(Path(gguf_path).resolve()),
        "physical_file_read_operations": physical_count,
        "physical_file_read_bytes": physical_bytes,
        "physical_file_read_mib": round(physical_bytes / (1024 * 1024), 4),
        # Backward-compatible aliases for model-inference reports.
        "physical_gguf_read_operations": physical_count,
        "physical_gguf_read_bytes": physical_bytes,
        "physical_gguf_read_mib": round(physical_bytes / (1024 * 1024), 4),
        "physical_read_latency_mean_ms": round(sum(physical_latencies) / len(physical_latencies), 4) if physical_latencies else None,
        "physical_read_latency_p50_ms": percentile(0.50),
        "physical_read_latency_p95_ms": percentile(0.95),
        "physical_read_latency_p99_ms": percentile(0.99),
        "latency_sample_count": len(physical_latencies),
        "latency_sample_limit": latency_sample_limit,
        "latency_clock_frequency_hz": frequency,
        "logical_gguf_file_read_operations": logical_count,
        "logical_gguf_file_read_bytes_requested": logical_bytes,
        "logical_read_scope_note": "FileIo_ReadWrite counts file-system read requests, not necessarily NVMe traffic; OS cache can satisfy these requests.",
        "physical_read_scope_note": "DiskIo completion reads are matched by FileObject filename to the GGUF; unrelated reads and cached file reads are excluded.",
        "limitation": "This measures physical disk service response time, not the complete time an inference thread waited on a page fault or storage.",
    }

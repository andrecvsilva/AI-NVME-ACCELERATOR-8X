"""Describe OS-managed paging and summarize bounded model memory observations."""

from __future__ import annotations


def paging_report(observation: dict | None) -> dict:
    observation = observation or {}
    return {
        "policy": "windows_os_managed_file_backed_pages",
        "minimum_system_available_bytes": observation.get("minimum_available_bytes"),
        "peak_process_working_set_bytes": observation.get("peak_working_set_bytes"),
        "peak_process_private_bytes": observation.get("peak_private_bytes"),
        "process_page_fault_count_delta": observation.get("delta_page_fault_count"),
        "process_read_transfer_bytes_delta": observation.get("delta_read_transfer_bytes"),
        "process_read_operation_count_delta": observation.get("delta_read_operation_count"),
        "sample_count": observation.get("sample_count", 0),
        "sample_errors": observation.get("sample_errors", 0),
        "process_probe_errors": observation.get("process_probe_errors", 0),
        "custom_page_eviction": False,
        "attribution_note": "Process totals include runtime, KV cache, Python, and mapped model pages; fault and read counters are process-wide, not GGUF/SSD-only measurements.",
    }

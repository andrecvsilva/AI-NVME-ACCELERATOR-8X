from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from engine.etw_io_analyzer import analyze_tracerpt_xml


class EtwIoAnalyzerTests(unittest.TestCase):
    def test_attributes_physical_reads_to_gguf_and_separates_file_requests(self):
        with TemporaryDirectory() as temp:
            model = Path(temp) / "weights.gguf"
            xml = Path(temp) / "events.xml"
            xml.write_text(f'''<?xml version="1.0"?>
<Events>
 <Event><System><Provider Name="Kernel-FileIO"/><EventID>0</EventID><Opcode>0</Opcode></System>
  <EventData><Data Name="FileObject">0x123</Data><Data Name="FileName">{model}</Data></EventData></Event>
 <Event><System><Provider Name="Kernel-FileIO"/><EventID>67</EventID><Opcode>67</Opcode></System>
  <EventData><Data Name="FileObject">0x777</Data><Data Name="FileKey">0x123</Data><Data Name="IoSize">65536</Data></EventData></Event>
 <Event><System><Provider Name="Kernel-Disk"/><EventID>10</EventID><Opcode>10</Opcode></System>
  <EventData><Data Name="DiskNumber">0</Data><Data Name="FileObject">0x123</Data><Data Name="TransferSize">4096</Data><Data Name="HighResResponseTime">2000</Data></EventData>
  <RenderingInfo><Opcode>DiskReadInit</Opcode></RenderingInfo></Event>
 <Event><System><Provider Name="Kernel-Disk"/><EventID>10</EventID><Opcode>10</Opcode></System>
  <EventData><Data Name="DiskNumber">0</Data><Data Name="FileObject">0x123</Data><Data Name="TransferSize">8192</Data><Data Name="HighResResponseTime">4000</Data></EventData>
  <RenderingInfo><Opcode>DiskReadInit</Opcode></RenderingInfo></Event>
 <Event><System><Provider Name="Kernel-FileIO"/><EventID>0</EventID><Opcode>0</Opcode></System>
  <EventData><Data Name="FileObject">0x123</Data><Data Name="FileName">C:\\unrelated.bin</Data></EventData></Event>
 <Event><System><Provider Name="Kernel-Disk"/><EventID>10</EventID><Opcode>10</Opcode></System>
  <EventData><Data Name="DiskNumber">0</Data><Data Name="FileObject">0x123</Data><Data Name="TransferSize">999999</Data><Data Name="HighResResponseTime">1000</Data></EventData>
  <RenderingInfo><Opcode>DiskReadInit</Opcode></RenderingInfo></Event>
 <Event><System><Provider Name="Kernel-Disk"/><EventID>10</EventID><Opcode>10</Opcode></System>
  <EventData><Data Name="DiskNumber">0</Data><Data Name="FileObject">0x999</Data><Data Name="TransferSize">999999</Data><Data Name="HighResResponseTime">1000</Data></EventData></Event>
</Events>''', encoding="utf-8")
            result = analyze_tracerpt_xml(xml, model, qpc_frequency=1_000_000)
            self.assertEqual(result["physical_gguf_read_operations"], 2)
            self.assertEqual(result["physical_gguf_read_bytes"], 12_288)
            self.assertEqual(result["logical_gguf_file_read_operations"], 1)
            self.assertEqual(result["logical_gguf_file_read_bytes_requested"], 65_536)
            self.assertEqual(result["physical_read_latency_mean_ms"], 3.0)
            self.assertEqual(result["physical_read_latency_p50_ms"], 2.0)

    def test_no_matching_file_produces_zeroes(self):
        with TemporaryDirectory() as temp:
            xml = Path(temp) / "events.xml"
            xml.write_text("<Events />", encoding="utf-8")
            result = analyze_tracerpt_xml(xml, Path(temp) / "weights.gguf", qpc_frequency=1)
            self.assertEqual(result["physical_gguf_read_bytes"], 0)
            self.assertIsNone(result["physical_read_latency_mean_ms"])


if __name__ == "__main__":
    unittest.main()

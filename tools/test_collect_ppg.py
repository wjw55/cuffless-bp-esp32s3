import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_ppg import parse_ppg_row


class ParsePpgRowTests(unittest.TestCase):
    def test_parses_sample_sequence_timestamp_red_and_ir(self):
        self.assertEqual(parse_ppg_row("42,12345,50001,60002"), (42, 12345, 50001, 60002))

    def test_ignores_new_csv_header(self):
        self.assertIsNone(parse_ppg_row("sample_seq,timestamp_ms,red,ir"))


if __name__ == "__main__":
    unittest.main()

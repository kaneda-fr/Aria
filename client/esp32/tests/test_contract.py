"""Host checks: compile the actual firmware core and validate its wire contract.

No microphone, model dependencies, network, or home actions are used.
Run: python3 -m unittest discover -s client/esp32/tests -v
"""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
CLIENT = ROOT / "client" / "esp32"


class FirmwareContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = str(Path(cls.directory.name) / "stream_tests")
        subprocess.run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=address,undefined", "-g",
            "-I", str(CLIENT / "main"), str(CLIENT / "main" / "stream_core.c"),
            str(CLIENT / "tests" / "test_stream_core.c"), "-o", cls.binary,
        ], check=True)

    def test_bounds_gaps_reconnect_and_fragmented_status(self):
        subprocess.run([self.binary], check=True)

    def test_start_is_accepted_by_current_server(self):
        # Isolate the actual server validator without initializing ASR/providers.
        payload = json.loads(subprocess.check_output([self.binary, "--start"]))
        # Check both the current handler and the committed legacy validator.
        sources = [(ROOT / "app" / "main.py").read_text(),
                   subprocess.check_output(['git', 'show', 'HEAD:app/main.py'], cwd=ROOT, text=True)]
        for source in sources:
            tree = ast.parse(source)
            validator = next(node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name == "_valid_start")
            namespace = {"Any": object}
            exec(compile(ast.Module(body=[validator], type_ignores=[]), "app/main.py", "exec"), namespace)
            self.assertTrue(namespace["_valid_start"](payload))
        self.assertEqual(payload["type"], "start")
        self.assertIs(payload["client_vad"], True)
        self.assertEqual(set(payload), {'type', 'sample_rate', 'channels', 'format', 'chunk_ms', 'client_vad', 'client_id'})
        self.assertEqual(payload['client_id'], 'aria-wisp-001122334455')
        self.assertEqual(payload["sample_rate"] * payload["chunk_ms"] * 2 // 1000, 640)


class AudioFrontendTests(unittest.TestCase):
    def test_processing_with_actual_detector(self):
        vendor = CLIENT / 'components' / 'webrtc_vad'
        with tempfile.TemporaryDirectory() as directory:
            objects = []
            for index, source in enumerate(sorted((vendor / 'webrtc' / 'common_audio').glob('*/*.c'))):
                output = str(Path(directory) / f'vendor-{index}.o')
                subprocess.run(['cc', '-std=c11', '-DWEBRTC_POSIX', '-fwrapv',
                                '-fsanitize=address', '-I', str(vendor), '-c', str(source),
                                '-o', output], check=True)
                objects.append(output)
            binary = str(Path(directory) / 'audio-tests')
            subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=address,undefined', '-g', '-I', str(vendor),
                            '-I', str(CLIENT / 'main'), str(CLIENT / 'main' / 'audio_frontend.c'),
                            str(CLIENT / 'tests' / 'test_audio_frontend.c'), *objects,
                            '-lpthread', '-lm', '-o', binary], check=True)
            subprocess.run([binary, str(CLIENT / 'tests' / 'fixtures' / 'webrtc-test-audio.raw')], check=True)


if __name__ == "__main__":
    unittest.main()

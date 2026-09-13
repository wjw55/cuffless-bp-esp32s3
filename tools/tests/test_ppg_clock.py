"""Compile and execute the firmware's actual clock implementation on the host."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PPGClockCTests(unittest.TestCase):
    def test_firmware_clock_simulations(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            gcc = shutil.which('gcc')
            if gcc:
                binary = output / ('clock.exe' if os.name == 'nt' else 'clock')
                subprocess.run([gcc, '-std=c11', '-O2', '-I', str(ROOT/'main'),
                    str(ROOT/'main/ppg_clock.c'), str(ROOT/'tools/tests/test_ppg_clock.c'), '-o', str(binary)], check=True, capture_output=True)
            elif os.name == 'nt':
                base = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Microsoft Visual Studio/2022/Community/VC/Tools/MSVC'
                versions = sorted(base.glob('*/bin/Hostx64/x64/cl.exe'))
                kits = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Windows Kits/10/Include'
                ucrt = sorted(kits.glob('*/ucrt'))
                if not versions or not ucrt:
                    self.skipTest('Host C compiler unavailable; run test_ppg_clock.c before flashing')
                compiler = versions[-1]
                include = compiler.parents[3] / 'include'
                env = dict(os.environ, INCLUDE=str(include)+';'+str(ucrt[-1]))
                objects = []
                for source in (ROOT/'main/ppg_clock.c', ROOT/'tools/tests/test_ppg_clock.c'):
                    obj = output/(source.stem+'.obj')
                    subprocess.run([str(compiler), '/nologo', '/O2', '/GS-', '/I'+str(ROOT/'main'),
                        '/c', str(source), '/Fo'+str(obj)], check=True, capture_output=True, env=env)
                    objects.append(str(obj))
                binary = output/'clock.exe'
                subprocess.run([str(compiler.parent/'link.exe'), '/nologo', '/entry:main',
                    '/subsystem:console', '/nodefaultlib', '/out:'+str(binary), *objects], check=True, capture_output=True)
            else:
                self.skipTest('Host C compiler unavailable')
            result = subprocess.run([str(binary)], capture_output=True)
            self.assertEqual(result.returncode, 0, 'C clock test failure code '+str(result.returncode))


if __name__ == '__main__':
    unittest.main()

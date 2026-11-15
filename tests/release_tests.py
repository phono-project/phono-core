"""Regression: Conda RPATH injection must not survive a release link/install."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from configure import release_environment
from check_elf import inspect


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux ELF toolchain regression')
class ReleaseLink(unittest.TestCase):
    def test_injected_rpaths_are_rejected_and_removed(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            (source / 'probe.cpp').write_text('extern "C" int phono_engine_create() { return 0; }\n')
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.24)\nproject(probe LANGUAGES CXX)\n'
                'add_library(probe SHARED probe.cpp)\n'
                'install(TARGETS probe LIBRARY DESTINATION lib)\n')
            contaminated = root / 'contaminated.so'
            subprocess.run([os.environ['CXX'], '-shared', '-fPIC', source / 'probe.cpp',
                            '-Wl,-rpath,/tmp/forbidden-release-path', '-o', contaminated], check=True)
            with self.assertRaisesRegex(ValueError, 'Loader paths are forbidden'):
                inspect(contaminated, no_rpath=True, c_api_only=True)
            # Both an environment RPATH and the compiler's specs are active.
            environment = {**os.environ, 'LDFLAGS':
                           f'-Wl,-rpath,/tmp/forbidden-release-path -Wl,-rpath-link,{root}'}
            environment = release_environment(environment, root / 'build', 'linux-probe')
            self.assertIn('-Wl,-rpath-link,', environment['LDFLAGS'])
            subprocess.run(['cmake', '-S', source, '-B', root / 'build', '-G', 'Ninja',
                            f'-DCMAKE_CXX_COMPILER={environment["CXX"]}',
                            '-DCMAKE_SKIP_RPATH=ON', '-DCMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH=ON',
                            f'-DCMAKE_INSTALL_PREFIX={root / "installed"}'], env=environment, check=True)
            subprocess.run(['cmake', '--build', root / 'build'], env=environment, check=True)
            subprocess.run(['cmake', '--install', root / 'build'], env=environment, check=True)
            inspect(root / 'installed/lib/libprobe.so', max_glibc='2.28', max_glibcxx='3.4.31',
                    max_cxxabi='1.3.14', no_rpath=True, c_api_only=True)


if __name__ == '__main__':
    unittest.main()

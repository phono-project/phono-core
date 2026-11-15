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
from package_sdk import LINUX_ABI_LIMITS


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux ELF toolchain regression')
class ReleaseLink(unittest.TestCase):
    def test_android_ignores_host_flags_and_uses_the_selected_ndk(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            ndk = root / 'ndk'
            toolchain = ndk / 'build/cmake/android.toolchain.cmake'
            toolchain.parent.mkdir(parents=True)
            toolchain.touch()
            compiler_dir = ndk / 'toolchains/llvm/prebuilt/linux-x86_64/bin'
            compiler_dir.mkdir(parents=True)
            binary = compiler_dir / 'clang-real'
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o755)
            (compiler_dir / 'clang').symlink_to('clang-real')
            (compiler_dir / 'clang++').symlink_to('clang')
            for arch in ('nocona', 'armv8-a'):
                host = {**os.environ, 'ANDROID_NDK_ROOT': str(ndk),
                        'ANDROID_NDK_HOME': '/wrong/preinstalled/ndk',
                        'CFLAGS': f'-march={arch} -isystem /host/include',
                        'CXXFLAGS': f'-march={arch} -isystem /host/include',
                        'CPPFLAGS': '-isystem /host/include',
                        'LDFLAGS': '-L/host/lib -Wl,-rpath,/host/lib'}
                target = release_environment(host, root / 'build', 'android-armv8')
                self.assertEqual(target['ANDROID_NDK_HOME'], str(ndk))
                self.assertEqual(target['CXX'], host['CXX'])
                self.assertFalse(any(target[key] for key in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS')))
                self.assertEqual(host['LDFLAGS'], '-L/host/lib -Wl,-rpath,/host/lib')
            # setup-ndk's optional fs.cp local cache can replace relative links
            # with absolute paths into a previous runner's tool cache. A valid
            # toolchain file alone must not make that restored NDK acceptable.
            (compiler_dir / 'clang++').unlink()
            (compiler_dir / 'clang++').symlink_to(root / 'old-tool-cache/bin/clang')
            with self.assertRaisesRegex(ValueError, 'compiler is missing or not executable'):
                release_environment(host, root / 'build', 'android-armv8')

    def test_injected_rpaths_are_rejected_and_removed(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            # This GCC 13.2 symbol reproduces the ARM64 runtime requirement on
            # any Linux host, while also exercising the existing RPATH guard.
            (source / 'probe.cpp').write_text(
                'namespace std { void ios_base_library_init(); }\n'
                'extern "C" int phono_engine_create() { std::ios_base_library_init(); return 0; }\n')
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
            library = root / 'installed/lib/libprobe.so'
            report = inspect(library, **LINUX_ABI_LIMITS, no_rpath=True, c_api_only=True)
            self.assertEqual(report['required_abi']['GLIBCXX'], '3.4.32')
            with self.assertRaisesRegex(ValueError, 'above allowed GLIBCXX_3.4.31'):
                inspect(library, max_glibcxx='3.4.31')


if __name__ == '__main__':
    unittest.main()

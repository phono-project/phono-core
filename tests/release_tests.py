"""Regression: Conda RPATH injection must not survive a release link/install."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from configure import release_environment, cmake_arguments
from check_macho import inspect as inspect_macho
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


class MacReleasePolicy(unittest.TestCase):
    def test_macos_selects_xcode_and_discards_conda_compiler_flags(self):
        host = {**os.environ, 'CC': '/conda/clang', 'CXX': '/conda/clang++',
                'CFLAGS': '-isystem /conda/include', 'CXXFLAGS': '-stdlib=libc++',
                'CPPFLAGS': '-isystem /conda/include', 'LDFLAGS': '-L/conda/lib',
                'CPATH': '/conda/include', 'LIBRARY_PATH': '/conda/lib',
                'CMAKE_TOOLCHAIN_FILE': '/conda/toolchain.cmake'}
        def xcrun(arguments, **kwargs):
            self.assertEqual(arguments[:3], ['/usr/bin/xcrun', '--sdk', 'macosx'])
            return '/Xcode/SDK\n' if arguments[-1] == '--show-sdk-path' else f'/Xcode/bin/{arguments[-1]}\n'
        with patch('configure.subprocess.check_output', side_effect=xcrun):
            target = release_environment(host, Path('/unused'), 'macos-arm64')
        self.assertEqual(target['CXX'], '/Xcode/bin/clang++')
        self.assertEqual(target['SDKROOT'], '/Xcode/SDK')
        self.assertEqual(target['MACOSX_DEPLOYMENT_TARGET'], '14.0')
        self.assertFalse(any(target[key] for key in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS')))
        self.assertFalse({'CPATH', 'LIBRARY_PATH', 'CMAKE_TOOLCHAIN_FILE'} & target.keys())
        arguments = cmake_arguments('macos-arm64', target)
        self.assertIn('-DCMAKE_CXX_COMPILER=/Xcode/bin/clang++', arguments)
        self.assertIn('-DCMAKE_CXX_FLAGS=', arguments)
        self.assertIn('-DCMAKE_OSX_SYSROOT=/Xcode/SDK', arguments)
        self.assertEqual(host['CXX'], '/conda/clang++')

    def test_macho_audit_rejects_development_paths_and_non_system_runtimes(self):
        def commands(runtime='/usr/lib/libc++.1.dylib', rpath=''):
            return (f'Load command 0\n cmd LC_ID_DYLIB\n name @rpath/libphono_core.dylib (offset 24)\n'
                    f'Load command 1\n cmd LC_LOAD_DYLIB\n name {runtime} (offset 24)\n'
                    f'Load command 2\n cmd LC_LOAD_DYLIB\n name /usr/lib/libSystem.B.dylib (offset 24)\n'
                    f'Load command 3\n cmd LC_BUILD_VERSION\n minos 14.0\n{rpath}')
        for label, load_commands in (
            ('clean', commands()),
            ('conda libc++', commands(runtime='@rpath/libc++.1.dylib')),
            ('development RPATH', commands(rpath='Load command 4\n cmd LC_RPATH\n path /Users/runner/.pixi/lib (offset 12)\n')),
        ):
            with self.subTest(label=label), patch('check_macho.subprocess.check_output',
                    side_effect=['arm64\n', load_commands, '_phono_engine_create\n']):
                if label == 'clean':
                    self.assertEqual(inspect_macho(Path('probe.dylib'))['loader_paths'], [])
                else:
                    with self.assertRaises(ValueError):
                        inspect_macho(Path('probe.dylib'))


@unittest.skipUnless(sys.platform == 'darwin', 'Xcode Mach-O toolchain regression')
class MacReleaseLink(unittest.TestCase):
    def test_native_link_uses_system_libcxx_without_rpath(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            (source / 'probe.cpp').write_text(
                '#include <string>\nextern "C" __attribute__((visibility("default"))) '
                'int phono_engine_create(const char *text) { return std::string(text).append("probe").size(); }\n')
            (source / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.24)\nproject(probe LANGUAGES CXX)\n'
                'add_library(phono_core SHARED probe.cpp)\n'
                'set_target_properties(phono_core PROPERTIES CXX_VISIBILITY_PRESET hidden '
                'VISIBILITY_INLINES_HIDDEN YES MACOSX_RPATH YES INSTALL_NAME_DIR "@rpath" '
                'BUILD_WITH_INSTALL_NAME_DIR YES)\n'
                'install(TARGETS phono_core LIBRARY DESTINATION lib)\n')
            environment = release_environment({**os.environ, 'CC': '/invalid/conda/clang',
                'CXX': '/invalid/conda/clang++', 'CXXFLAGS': '-isystem /invalid/conda/include',
                'LDFLAGS': '-L/invalid/conda/lib -Wl,-rpath,/invalid/conda/lib'},
                root / 'build', 'macos-arm64')
            flags = cmake_arguments('macos-arm64', environment)[3:]
            subprocess.run(['cmake', '-S', source, '-B', root / 'build', '-G', 'Ninja',
                            '-DCMAKE_SKIP_RPATH=ON', f'-DCMAKE_INSTALL_PREFIX={root / "installed"}',
                            *flags], env=environment, check=True)
            subprocess.run(['cmake', '--build', root / 'build'], env=environment, check=True)
            subprocess.run(['cmake', '--install', root / 'build'], env=environment, check=True)
            inspect_macho(root / 'installed/lib/libphono_core.dylib')


if __name__ == '__main__':
    unittest.main()

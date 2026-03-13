from setuptools import setup, find_packages, Command
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info
import subprocess
import shutil
import sys
from pathlib import Path


# 获取项目根目录
project_root = Path(__file__).parent.absolute()
build_dir = project_root / "build"
package_dir = project_root / "rocm_halcyon"


def build_extension():
    """从零开始编译 C++ 扩展模块"""
    print("=" * 60)
    print("Building halcyon_core C++ extension...")
    print("=" * 60)
    
    # 创建 build 目录
    build_dir.mkdir(exist_ok=True)
    
    # 检查是否需要运行 cmake（没有配置文件或 CMakeLists.txt 更新了）
    cmake_cache = build_dir / "CMakeCache.txt"
    cmakelists = project_root / "CMakeLists.txt"
    
    need_cmake = not cmake_cache.exists()
    if cmake_cache.exists() and cmakelists.exists():
        if cmakelists.stat().st_mtime > cmake_cache.stat().st_mtime:
            need_cmake = True
    
    if need_cmake:
        print("Running cmake configuration...")
        result = subprocess.run(
            ["cmake", "-G", "Ninja", ".."],
            cwd=build_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"CMake configuration failed:\n{result.stderr}")
            sys.exit(1)
        print(result.stdout)
    
    # 运行 ninja 编译
    print("Running ninja build...")
    result = subprocess.run(
        ["ninja"],
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Ninja build failed:\n{result.stderr}")
        sys.exit(1)
    print(result.stdout)
    
    # 查找编译好的 .so 文件并复制到包目录
    so_files = list(build_dir.glob("halcyon_core*.so"))
    if not so_files:
        print("Error: No .so file found after build!")
        sys.exit(1)
    
    for so_file in so_files:
        dest = package_dir / so_file.name
        shutil.copy2(so_file, dest)
        print(f"Copied {so_file.name} -> {dest}")
    
    print("=" * 60)
    print("C++ extension build completed successfully!")
    print("=" * 60)


class BuildExtensionFirst:
    """Mixin class to build C++ extension before running the command"""
    def run(self):
        build_extension()
        super().run()


class CustomBuildPy(BuildExtensionFirst, build_py):
    pass


class CustomDevelop(BuildExtensionFirst, develop):
    pass


class CustomInstall(BuildExtensionFirst, install):
    pass


class CustomEggInfo(BuildExtensionFirst, egg_info):
    pass


setup(
    name="rocm_halcyon",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "rocm_halcyon": ["*.so", "*.pyi", "py.typed"],
    },
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "parse",
        "perfetto"
    ],
    python_requires=">=3.7",
    cmdclass={
        "build_py": CustomBuildPy,
        "develop": CustomDevelop,
        "install": CustomInstall,
        "egg_info": CustomEggInfo,
    },
)

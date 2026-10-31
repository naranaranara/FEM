"""컴파일러가 있을 때 선택적인 RIMLS 이차 미분 확장을 빌드해요."""
import sys

from setuptools import Extension, setup


compile_args = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"]
setup(ext_modules=[Extension(
    "src.geometry._rimls_native",
    ["src/geometry/_rimls_native.cpp"],
    language="c++",
    extra_compile_args=compile_args,
    optional=True,
)])

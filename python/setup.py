import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
	def __init__(self, name, sourcedir=''):
		Extension.__init__(self, name, sources=[])
		self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
	def build_extension(self, ext):
		extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
		
		# Required for auto-detection of auxiliary "native" libs
		if not extdir.endswith(os.path.sep):
			extdir += os.path.sep
		
		cfg = 'Debug' if self.debug else 'Release'
		
		cmake_args = [
			f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
			f'-DPYTHON_EXECUTABLE={sys.executable}',
			f'-DCMAKE_BUILD_TYPE={cfg}',
			'-DBUILD_PYTHON=ON',
			'-DBUILD_TESTS=OFF',
		]
		
		build_cuda = os.environ.get('BUILD_CUDA', 'ON')
		cmake_args.append(f'-DBUILD_CUDA={build_cuda}')
		
		build_args = ['--config', cfg]
		
		if hasattr(self, 'parallel') and self.parallel:
			build_args += [f'-j{self.parallel}']
		
		build_temp = Path(self.build_temp)
		build_temp.mkdir(parents=True, exist_ok=True)
		
		print(f"Building in: {build_temp}")
		print(f"CMake args: {cmake_args}")
		
		subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
		subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

version_file = Path(__file__).parent.parent / "VERSION"
version = version_file.read_text().strip()

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
	name='octproengine',
	version=version,
	author='Miroslav Zabic',
	description='High-performance OCT (Optical Coherence Tomography) processing library',
	long_description=long_description,
	long_description_content_type='text/markdown',
	ext_modules=[CMakeExtension('octproengine')],
	cmdclass={'build_ext': CMakeBuild},
	zip_safe=False,
	python_requires='>=3.8',
	install_requires=[
		'numpy>=1.19.0',
	],
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Medical Science Apps.',
		'Topic :: Scientific/Engineering :: Image Processing',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.8',
		'Programming Language :: Python :: 3.9',
		'Programming Language :: Python :: 3.10',
		'Programming Language :: Python :: 3.11',
		'Programming Language :: Python :: 3.12',
	],
)

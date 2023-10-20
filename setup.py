#!/usr/bin/env python
import glob
import os
import shutil
from distutils.cmd import Command
from distutils.core import setup


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    CLEAN_FILES = './build ./dist ./*.pyc ./*.tgz ./*.egg-info'.split(' ')

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        global here

        here = os.getcwd()
        for path_spec in self.CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(here, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(here):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, here))
                print(f"Cleaning {os.path.relpath(path)}")
                shutil.rmtree(path)

setup(
    name='EMIA__utils',
    version='1.0',
    packages=['emia_utils'],
    url='https://github.com/Interactive-Coventry/EMIA__utils',
    license='MIT',
    author='foxelas',
    author_email='foxelas@outlook.com',
    description='Utils for EMIA projects ',
    cmdclass={
        'clean': CleanCommand,
    },
)

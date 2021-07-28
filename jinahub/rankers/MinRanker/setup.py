__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import setuptools

setuptools.setup(
    name='jinahub-MinRanker',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='This is my dummy executor',
    url='https://github.com/jina-ai/executors',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.rankers.minranker'],
    package_dir={'jinahub.rankers': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)

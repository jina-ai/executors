__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import setuptools

setuptools.setup(
    name='jinahub-executor-sentence-encoder',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='Executor encodes sentences into a d-dimensional latent space',
    url='https://github.com/jina-ai/executor-sentence-transformer',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.text.encoders.sentence_encoder'],
    package_dir={'jinahub.text.encoders': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)

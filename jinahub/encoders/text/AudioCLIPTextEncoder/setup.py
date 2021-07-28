__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import setuptools

setuptools.setup(
    name='jinahub-audioclip-text',
    version='1.0',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='Executor that encodes text with the AudioCLIP model',
    url='https://github.com/jina-ai/executors/main/jinahub/encoders/text/AudioCLIPTextEncoder',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.encoder.audioclip_text'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)

__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import setuptools

setuptools.setup(
    name='jinahub-image-tf-encoder',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='This encoder encodes images to vectors.',
    url='https://github.com/jina-ai/EXECUTOR_REPO_NAME',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.encoder.image_tf_encoder'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)

import setuptools


setuptools.setup(
    name='executor-text-encoder',
    version="1",
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description="Executor that encodes text",
    url='https://github.com/jina-ai/executor-text-laser-encoder',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    py_modules=['jinahub.encoder.laser_encoder'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)

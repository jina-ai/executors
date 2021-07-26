import setuptools


setuptools.setup(
    name="executor-transformer-torch-encoder",
    version="2.0",
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description="Executor that encodes text sentences",
    url="https://github.com/jina-ai/executor-transformer-torch-encoder",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    py_modules=['jinahub.encoder.transform_encoder'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open("requirements.txt").readlines(),
    python_requires=">=3.7",
)

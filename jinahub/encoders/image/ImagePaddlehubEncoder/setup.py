import setuptools


setuptools.setup(
    name="executor-image-encoder",
    version="2.0",
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description="Executor that encodes images",
    url="https://github.com/jina-ai/executor-image-paddle-encoder",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    py_modules=['jinahub.encoder.paddle_image'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open("requirements.txt").readlines(),
    python_requires=">=3.7",
)

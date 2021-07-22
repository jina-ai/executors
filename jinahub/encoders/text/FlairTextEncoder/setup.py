import setuptools


setuptools.setup(
    name="executor-text-encoder",
    version="2.0",
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description="Executor that encodes text",
    url="https://github.com/jina-ai/executor-text-flair-encoder",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    py_modules=['jinahub.encoder.flair_text'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open("requirements.txt").readlines(),
    python_requires=">=3.7",
)

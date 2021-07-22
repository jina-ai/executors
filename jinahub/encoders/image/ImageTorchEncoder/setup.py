import setuptools


setuptools.setup(
    name="executor-image-torch-encoder",
    version="2.0.3",
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description="Executor that encodes images into latent space using PyTorch hosted neural networks",
    url="https://github.com/jina-ai/executor-image-torch-encoder",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    py_modules=['jinahub.image.encoder.torch_encoder', 'jinahub.image.encoder.models'],
    package_dir={'jinahub.image.encoder': '.'},
    install_requires=open("requirements.txt").readlines(),
    python_requires=">=3.7",
)

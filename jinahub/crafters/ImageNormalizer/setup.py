import setuptools

setuptools.setup(
    name="executor-image-normalizer",
    py_modules=['jinahub.image.normalizer'],
    package_dir={'jinahub.image': '.'},
    install_requires=open("requirements.txt").readlines(),
    python_requires=">=3.7",
)

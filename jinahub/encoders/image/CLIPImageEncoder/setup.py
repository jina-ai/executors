import setuptools

setuptools.setup(
    name='jinahub-clip-image',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='Executors that encode images',
    url='https://github.com/jina-ai/executor-clip-image',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.encoder.clip_image'],
    package_dir={'jinahub.encoder': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)
